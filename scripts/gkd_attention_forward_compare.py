# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
from pathlib import Path

import torch


LAYER_NODE_ORDER = (
    'A_layer_input',
    'B_linear_qkv_output',
    'C0_core_attention_input_qkv',
    'C_core_attention_output',
    'D_linear_proj_output',
    'E_pre_mlp_input',
    'F_mlp_output',
    'G_layer_output',
)


def tensor_metrics(gpu_tensor, npu_tensor):
    gpu = gpu_tensor.detach().double().cpu().contiguous().reshape(-1)
    npu = npu_tensor.detach().double().cpu().contiguous().reshape(-1)
    if gpu.numel() != npu.numel():
        raise ValueError(f'Tensor sizes differ: GPU={gpu.numel()}, NPU={npu.numel()}')
    difference = gpu - npu
    gpu_norm = torch.linalg.vector_norm(gpu)
    npu_norm = torch.linalg.vector_norm(npu)
    difference_norm = torch.linalg.vector_norm(difference)
    epsilon = torch.finfo(torch.float64).eps
    if gpu_norm.item() == 0.0 and npu_norm.item() == 0.0:
        cosine = 1.0
    else:
        cosine = torch.dot(gpu, npu).item() / max(
            (gpu_norm * npu_norm).item(), epsilon)
    return {
        'gpu_shape': list(gpu_tensor.shape),
        'npu_shape': list(npu_tensor.shape),
        'numel': gpu.numel(),
        'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
        'mean_abs': difference.abs().mean().item() if difference.numel() else 0.0,
        'absolute_l2': difference_norm.item(),
        'relative_l2': difference_norm.item() / max(npu_norm.item(), epsilon),
        'cosine': cosine,
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
    }


def compare_maps(gpu_values, npu_values):
    result = {}
    for path in sorted(set(gpu_values) | set(npu_values)):
        gpu = gpu_values.get(path)
        npu = npu_values.get(path)
        if gpu is None or npu is None:
            result[path] = {
                'gpu_present': gpu is not None,
                'npu_present': npu is not None,
            }
        else:
            result[path] = tensor_metrics(gpu, npu)
    return result


def only_tensor(node_values, node):
    if len(node_values) != 1:
        raise ValueError(f'Expected one tensor at {node}, found paths={sorted(node_values)}')
    return next(iter(node_values.values()))


def primary_tensor(node_values, reference_numel, node):
    candidates = [
        (path, tensor) for path, tensor in node_values.items()
        if tensor.numel() == reference_numel
    ]
    if len(candidates) != 1:
        raise ValueError(
            f'Expected one full-size primary tensor at {node}, found '
            f'{[(path, list(tensor.shape)) for path, tensor in candidates]}')
    return candidates[0]


def residual_closure(payload, input_node, output_node, branch_node):
    nodes = payload['nodes']
    residual_input = only_tensor(nodes[input_node], input_node).float()
    _, residual_output = primary_tensor(
        nodes[output_node], residual_input.numel(), output_node)
    residual_output = residual_output.float().reshape_as(residual_input)
    inferred_branch = residual_output - residual_input
    branch_values = nodes[branch_node]
    primary_path, primary = primary_tensor(
        branch_values, inferred_branch.numel(), branch_node)
    primary = primary.float().reshape_as(inferred_branch)
    candidate = primary
    bias_path = None
    bias_candidates = [
        (path, tensor) for path, tensor in branch_values.items()
        if path != primary_path and tensor.numel() == inferred_branch.shape[-1]
    ]
    if len(bias_candidates) == 1:
        bias_path, bias = bias_candidates[0]
        candidate = candidate + bias.float().reshape(
            *((1,) * (inferred_branch.ndim - 1)), inferred_branch.shape[-1])
    return {
        'inferred_branch': inferred_branch,
        'branch_candidate': candidate,
        'primary_path': primary_path,
        'bias_path': bias_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare GPU/NPU Layer-N attention forward boundary captures.')
    parser.add_argument('--gpu', required=True, help='GPU attention_forward_*.pt path.')
    parser.add_argument('--npu', required=True, help='NPU attention_forward_*.pt path.')
    parser.add_argument('--output', help='Optional JSON result path.')
    args = parser.parse_args()

    gpu = torch.load(args.gpu, map_location='cpu', weights_only=True)
    npu = torch.load(args.npu, map_location='cpu', weights_only=True)
    gpu_nodes = gpu.get('nodes', {})
    npu_nodes = npu.get('nodes', {})
    scope = gpu.get('scope', 'layer')
    invariants = {
        'scope_match': scope == npu.get('scope', 'layer'),
        'layer_target_match': gpu.get('layer_target') == npu.get('layer_target'),
        'node_targets_match': gpu.get('node_targets') == npu.get('node_targets'),
        'step_match': gpu.get('step') == npu.get('step'),
        'micro_batch_match': gpu.get('micro_batch') == npu.get('micro_batch'),
        'all_nodes_present': set(gpu_nodes) == set(npu_nodes),
    }
    failed = [name for name, value in invariants.items() if not value]
    if failed:
        raise ValueError(f'Attention forward trace invariants failed: {failed}')

    if scope == 'layer' and set(gpu_nodes) != set(LAYER_NODE_ORDER):
        raise ValueError(
            f'Layer trace nodes differ: expected={list(LAYER_NODE_ORDER)}, '
            f'actual={sorted(gpu_nodes)}')
    if scope == 'layers' and not gpu_nodes:
        raise ValueError('Layer scan contains no nodes.')

    node_order = LAYER_NODE_ORDER if scope == 'layer' else sorted(gpu_nodes)
    nodes = {node: compare_maps(gpu_nodes[node], npu_nodes[node]) for node in node_order}
    result = {
        'invariants': invariants,
        'scope': scope,
        'gpu_tag': gpu.get('tag'),
        'npu_tag': npu.get('tag'),
        'module_types': {
            'gpu': gpu.get('module_types', {}),
            'npu': npu.get('module_types', {}),
        },
        'nodes': nodes,
    }
    if scope == 'layer':
        qkv_node = nodes['C0_core_attention_input_qkv']
        result['core_attention_input_qkv_summary'] = {
            path.removeprefix('tensor.'): metrics
            for path, metrics in qkv_node.items()
        }
        closures = {}
        for name, input_node, output_node, branch_node in (
                ('attention', 'A_layer_input', 'E_pre_mlp_input', 'D_linear_proj_output'),
                ('mlp', 'E_pre_mlp_input', 'G_layer_output', 'F_mlp_output')):
            gpu_closure = residual_closure(gpu, input_node, output_node, branch_node)
            npu_closure = residual_closure(npu, input_node, output_node, branch_node)
            closures[name] = {
                'inferred_branch_gpu_vs_npu': tensor_metrics(
                    gpu_closure['inferred_branch'], npu_closure['inferred_branch']),
                'gpu_branch_candidate_vs_inferred': tensor_metrics(
                    gpu_closure['branch_candidate'], gpu_closure['inferred_branch']),
                'npu_branch_candidate_vs_inferred': tensor_metrics(
                    npu_closure['branch_candidate'], npu_closure['inferred_branch']),
                'gpu_primary_path': gpu_closure['primary_path'],
                'npu_primary_path': npu_closure['primary_path'],
                'gpu_bias_path': gpu_closure['bias_path'],
                'npu_bias_path': npu_closure['bias_path'],
            }
        result['residual_closures'] = closures
    else:
        layer_summary = {}
        for node in node_order:
            candidates = [
                (path, metrics) for path, metrics in nodes[node].items()
                if 'numel' in metrics
            ]
            if not candidates:
                continue
            path, metrics = max(candidates, key=lambda item: item[1]['numel'])
            layer_summary[node] = {
                'primary_path': path,
                'relative_l2': metrics['relative_l2'],
                'cosine': metrics['cosine'],
                'gpu_norm': metrics['gpu_norm'],
                'npu_norm': metrics['npu_norm'],
            }
        result['layer_summary'] = layer_summary
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
