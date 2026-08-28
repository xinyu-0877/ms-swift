# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import re
from pathlib import Path

import torch


LAYER_NODE_ORDER = (
    'A_layer_input',
    'B0_linear_qkv_output',
    'B1_query_before_qk_norm',
    'B1_key_before_qk_norm',
    'B2_query_after_qk_norm',
    'B2_key_after_qk_norm',
    'C0_core_attention_input_qkv',
    'C_core_attention_output',
    'D_linear_proj_output',
    'E_pre_mlp_input',
    'F_mlp_output',
    'G_layer_output',
)

DECODER_TAIL_NODE_ORDER = (
    'Z00_final_layernorm_output',
    'Z01_output_layer_logits',
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


def _probe_identity(payload):
    probes = payload.get('provenance', {}).get('student_parameter_probes', {}).get('probes', [])
    return {p['name']: (p['full_shape'], p['full_dtype'], p['full_numel'],
                        p['sample_indices'], p['sample']['sha256_fp32']) for p in probes}


def provenance_invariants(gpu, npu):
    gp = gpu.get('provenance')
    np = npu.get('provenance')
    if gp is None or np is None:
        return {'provenance_present': False}
    return {
        'provenance_present': True,
        'input_ids_match': gp.get('input_ids') == np.get('input_ids'),
        'position_ids_match': gp.get('position_ids') == np.get('position_ids'),
        'labels_match': gp.get('labels') == np.get('labels'),
        'num_valid_match': gp.get('num_valid') == np.get('num_valid'),
        'teacher_logits_match': gp.get('teacher_logits') == np.get('teacher_logits'),
        'teacher_topk_logprobs_match': gp.get('teacher_topk_logprobs') == np.get('teacher_topk_logprobs'),
        'teacher_topk_indices_match': gp.get('teacher_topk_indices') == np.get('teacher_topk_indices'),
        'teacher_labels_match': gp.get('teacher_labels') == np.get('teacher_labels'),
        'student_parameter_probes_match': bool(_probe_identity(gpu)) and _probe_identity(gpu) == _probe_identity(npu),
    }


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


def format_percent(value):
    return f'{value * 100.0:.6f}%'


def print_metric(node, path, metrics):
    print(
        f'{node:<32} path={path:<16} '
        f'rel_l2={format_percent(metrics["relative_l2"]):>11} '
        f'cos={metrics["cosine"]:.9f} '
        f'max_abs={metrics["max_abs"]:.8g}')


def print_summary(result, output_path, material_increase, top_k):
    saved_path = output_path if output_path is not None else 'not saved (--output omitted)'
    print(f'Full report: {saved_path}')
    print(f'Scope: {result["scope"]}')
    print(f'Invariants: all passed ({len(result["invariants"])})')
    runtime = result.get('runtime', {}).get('gpu', {})
    print(
        'Runtime: '
        f'fp32_residual_connection={runtime.get("fp32_residual_connection")}, '
        f'torch_dtype={runtime.get("torch_dtype")}, '
        f'padding_free={runtime.get("padding_free")}')

    if result['scope'] == 'layer':
        print('\nA-G boundaries:')
        for node in LAYER_NODE_ORDER:
            for path, metrics in result['nodes'][node].items():
                if 'relative_l2' in metrics:
                    print_metric(node, path, metrics)
        print('\nResidual closures:')
        for name, closure in result['residual_closures'].items():
            native = closure['inferred_branch_gpu_vs_npu']
            gpu = closure['gpu_branch_candidate_vs_inferred']
            npu = closure['npu_branch_candidate_vs_inferred']
            print(
                f'{name:<10} branch_rel_l2={format_percent(native["relative_l2"]):>11} '
                f'gpu_closure={format_percent(gpu["relative_l2"]):>11} '
                f'npu_closure={format_percent(npu["relative_l2"]):>11}')
        return

    print('\nLayer boundaries:')
    print('layer       input_rel_l2 output_rel_l2     delta_pp  output_cos')
    for row in result['layer_increases']:
        output_metrics = result['layer_summary'][f'L{row["layer"]:02d}_output']
        print(
            f'L{row["layer"]:02d} '
            f'{format_percent(row["input_relative_l2"]):>18} '
            f'{format_percent(row["output_relative_l2"]):>14} '
            f'{row["increase_percentage_points"]:+12.6f} '
            f'{output_metrics["cosine"]:.9f}')

    print('\nDecoder tail boundaries:')
    print('boundary                    rel_l2       delta_pp          cos')
    for row in result.get('decoder_tail_increases', []):
        print(
            f'{row["label"]:<27} '
            f'{format_percent(row["relative_l2"]):>11} '
            f'{row["increase_percentage_points"]:+14.6f} '
            f'{row["cosine"]:.9f}')

    print(f'\nTop {min(top_k, len(result["layer_increase_ranking"]))} output-input increases:')
    for rank, row in enumerate(result['layer_increase_ranking'][:top_k], start=1):
        print(
            f'{rank:>2}. L{row["layer"]:02d}: '
            f'{format_percent(row["input_relative_l2"])} -> '
            f'{format_percent(row["output_relative_l2"])} '
            f'({row["increase_percentage_points"]:+.6f} pp)')

    first_material = next(
        (row for row in result['layer_increases'] if row['increase'] >= material_increase),
        None)
    if first_material is None:
        print(f'First material increase: none (threshold={format_percent(material_increase)})')
    else:
        print(
            f'First material increase: L{first_material["layer"]:02d} '
            f'({format_percent(first_material["input_relative_l2"])} -> '
            f'{format_percent(first_material["output_relative_l2"])})')
    recommended = result['recommended_layer']
    if recommended is not None:
        print(
            f'Recommended A-G target: decoder.layers.{recommended["layer"]} '
            f'(basis={result["recommendation_basis"]})')


def main():
    parser = argparse.ArgumentParser(
        description='Compare GPU/NPU Layer-N attention forward boundary captures.')
    parser.add_argument('--gpu', required=True, help='GPU attention_forward_*.pt path.')
    parser.add_argument('--npu', required=True, help='NPU attention_forward_*.pt path.')
    parser.add_argument('--output', help='Optional JSON result path.')
    parser.add_argument('--material-increase', type=float, default=0.001,
                        help='Minimum output-input relative_l2 increase; default 0.001 (0.1%%).')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of largest layer increases printed; default 5.')
    args = parser.parse_args()
    if args.material_increase < 0:
        raise ValueError('--material-increase must be non-negative.')
    if args.top_k <= 0:
        raise ValueError('--top-k must be positive.')

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
        'runtime_match': gpu.get('runtime') == npu.get('runtime'),
    }
    invariants.update(provenance_invariants(gpu, npu))
    failed = [name for name, value in invariants.items() if not value]
    if failed:
        raise ValueError(f'Attention forward trace invariants failed: {failed}')

    if scope == 'layer' and set(gpu_nodes) != set(LAYER_NODE_ORDER):
        raise ValueError(
            f'Layer trace nodes differ: expected={list(LAYER_NODE_ORDER)}, '
            f'actual={sorted(gpu_nodes)}')
    if scope == 'layers' and not gpu_nodes:
        raise ValueError('Layer scan contains no nodes.')
    if scope == 'layers':
        missing_tail = set(DECODER_TAIL_NODE_ORDER) - set(gpu_nodes)
        if missing_tail:
            raise ValueError(
                f'Layer scan is missing decoder-tail nodes: {sorted(missing_tail)}. '
                'Regenerate both captures with the updated trace implementation.')

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
        'checkpoint_provenance': {
            'gpu': gpu.get('checkpoint_provenance'),
            'npu': npu.get('checkpoint_provenance'),
        },
        'runtime': {
            'gpu': gpu.get('runtime'),
            'npu': npu.get('runtime'),
        },
        'nodes': nodes,
    }
    if scope == 'layer':
        qkv_node = nodes['C0_core_attention_input_qkv']
        result['core_attention_input_qkv_summary'] = {
            path.removeprefix('tensor.'): metrics
            for path, metrics in qkv_node.items()
        }
        result['qkv_pipeline_summary'] = {
            'combined_linear_qkv': nodes['B0_linear_qkv_output'],
            'query_before_qk_norm': nodes['B1_query_before_qk_norm'],
            'key_before_qk_norm': nodes['B1_key_before_qk_norm'],
            'query_after_qk_norm': nodes['B2_query_after_qk_norm'],
            'key_after_qk_norm': nodes['B2_key_after_qk_norm'],
            'core_attention_input_qkv': result['core_attention_input_qkv_summary'],
            'value_note': (
                'core_attention input value is the runtime value after QKV split/reshape; '
                'Q/K norm and RoPE do not transform value.'),
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
        layer_increases = []
        for index in sorted({int(m.group(1)) for node in layer_summary
                             if (m := re.fullmatch(r'L(\d+)_input', node))}):
            input_metrics = layer_summary.get(f'L{index:02d}_input')
            output_metrics = layer_summary.get(f'L{index:02d}_output')
            if input_metrics is None or output_metrics is None:
                continue
            increase = output_metrics['relative_l2'] - input_metrics['relative_l2']
            layer_increases.append({
                'layer': index,
                'input_relative_l2': input_metrics['relative_l2'],
                'output_relative_l2': output_metrics['relative_l2'],
                'increase': increase,
                'input_percent': input_metrics['relative_l2'] * 100.0,
                'output_percent': output_metrics['relative_l2'] * 100.0,
                'increase_percentage_points': increase * 100.0,
                'material': increase >= args.material_increase,
            })
        result['layer_increases'] = layer_increases
        result['layer_increase_ranking'] = sorted(layer_increases, key=lambda x: x['increase'], reverse=True)
        result['recommended_layer'] = next(
            (x for x in layer_increases if x['material']),
            result['layer_increase_ranking'][0] if layer_increases else None)
        result['recommendation_basis'] = (
            'first_material_increase'
            if any(x['material'] for x in layer_increases)
            else 'largest_increase_below_threshold')
        last_layer_output = max(
            (node for node in layer_summary if re.fullmatch(r'L\d+_output', node)),
            key=lambda node: int(re.fullmatch(r'L(\d+)_output', node).group(1)),
        )
        tail_chain = (
            ('last_layer_output', last_layer_output),
            ('final_layernorm_output', 'Z00_final_layernorm_output'),
            ('output_layer_logits', 'Z01_output_layer_logits'),
        )
        decoder_tail_increases = []
        previous_metrics = None
        for label, node in tail_chain:
            metrics = layer_summary.get(node)
            if metrics is None:
                raise ValueError(f'Decoder-tail primary tensor is missing: {node}.')
            increase = (
                0.0 if previous_metrics is None
                else metrics['relative_l2'] - previous_metrics['relative_l2'])
            decoder_tail_increases.append({
                'label': label,
                'node': node,
                'relative_l2': metrics['relative_l2'],
                'relative_percent': metrics['relative_l2'] * 100.0,
                'increase': increase,
                'increase_percentage_points': increase * 100.0,
                'cosine': metrics['cosine'],
                'gpu_norm': metrics['gpu_norm'],
                'npu_norm': metrics['npu_norm'],
            })
            previous_metrics = metrics
        result['decoder_tail_increases'] = decoder_tail_increases
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    output_path = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + '\n', encoding='utf-8')
    print_summary(result, output_path, args.material_increase, args.top_k)


if __name__ == '__main__':
    main()
