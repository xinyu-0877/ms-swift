#!/usr/bin/env python3
"""Compare ordinary-SFT GPU/NPU Layer-0 A-G forward captures."""

import argparse
import json
from pathlib import Path

import torch


A_G_NODES = (
    'A_layer_input',
    'B0_linear_qkv_output',
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
    diff = gpu - npu
    gpu_norm = torch.linalg.vector_norm(gpu)
    npu_norm = torch.linalg.vector_norm(npu)
    diff_norm = torch.linalg.vector_norm(diff)
    eps = torch.finfo(torch.float64).eps
    cosine = 1.0 if gpu_norm.item() == 0.0 and npu_norm.item() == 0.0 else (
        torch.dot(gpu, npu).item() / max((gpu_norm * npu_norm).item(), eps))
    return {
        'gpu_shape': list(gpu_tensor.shape),
        'npu_shape': list(npu_tensor.shape),
        'numel': gpu.numel(),
        'max_abs': diff.abs().max().item() if diff.numel() else 0.0,
        'mean_abs': diff.abs().mean().item() if diff.numel() else 0.0,
        'absolute_l2': diff_norm.item(),
        'relative_l2': diff_norm.item() / max(npu_norm.item(), eps),
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
            result[path] = {'gpu_present': gpu is not None, 'npu_present': npu is not None}
        else:
            result[path] = tensor_metrics(gpu, npu)
    return result


def _provenance_invariants(gpu, npu):
    gp = gpu.get('provenance', {})
    np = npu.get('provenance', {})
    invariants = {
        'provenance_present': bool(gp) and bool(np),
        'input_ids_match': gp.get('input_ids') == np.get('input_ids'),
        'position_ids_match': gp.get('position_ids') == np.get('position_ids'),
        'labels_match': gp.get('labels') == np.get('labels'),
        'num_valid_match': gp.get('num_valid') == np.get('num_valid'),
    }
    # The probe list is a compact checkpoint sanity check.  The full model
    # checkpoint and optimizer state are still recorded separately in payload.
    invariants['parameter_probes_match'] = (
        gp.get('student_parameter_probes', []) == np.get('student_parameter_probes', []))
    return invariants


def format_percent(value):
    return f'{value * 100.0:.6f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='GPU sft_attention_forward_*.pt path')
    parser.add_argument('--npu', required=True, help='NPU sft_attention_forward_*.pt path')
    parser.add_argument('--output', help='Optional JSON report path')
    parser.add_argument('--allow-provenance-mismatch', action='store_true',
                        help='Do not fail when input/checkpoint provenance differs')
    args = parser.parse_args()

    gpu = torch.load(args.gpu, map_location='cpu', weights_only=True)
    npu = torch.load(args.npu, map_location='cpu', weights_only=True)
    invariants = {
        'record_type_match': gpu.get('record_type') == npu.get('record_type') == 'sft_attention_forward',
        'scope_match': gpu.get('scope') == npu.get('scope') == 'layer',
        'layer_target_match': gpu.get('layer_target') == npu.get('layer_target'),
        'node_targets_match': gpu.get('node_targets') == npu.get('node_targets'),
        'step_match': gpu.get('step') == npu.get('step') == 0,
        'micro_batch_match': gpu.get('micro_batch') == npu.get('micro_batch') == 0,
        'nodes_match': set(gpu.get('nodes', {})) == set(npu.get('nodes', {})) == set(A_G_NODES),
    }
    gpu_runtime = gpu.get('runtime', {})
    npu_runtime = npu.get('runtime', {})
    runtime_keys = (
        'tensor_model_parallel_size', 'pipeline_model_parallel_size',
        'context_parallel_size', 'micro_batch_size', 'global_batch_size',
        'padding_free', 'sequence_parallel', 'torch_dtype',
    )
    invariants['runtime_compatible'] = all(
        gpu_runtime.get(key) == npu_runtime.get(key) for key in runtime_keys)
    invariants.update(_provenance_invariants(gpu, npu))
    provenance_failures = {
        'input_ids_match', 'position_ids_match', 'labels_match', 'num_valid_match',
        'parameter_probes_match', 'provenance_present',
    }
    failed = [key for key, value in invariants.items()
              if not value and (not args.allow_provenance_mismatch or key not in provenance_failures)]
    if failed:
        raise ValueError(f'SFT A-G invariants failed: {failed}')

    nodes = {
        node: compare_maps(gpu['nodes'][node], npu['nodes'][node])
        for node in A_G_NODES
    }
    result = {
        'invariants': invariants,
        'gpu_tag': gpu.get('tag'),
        'npu_tag': npu.get('tag'),
        'layer_target': gpu.get('layer_target'),
        'runtime': {'gpu': gpu.get('runtime'), 'npu': npu.get('runtime')},
        'checkpoint_provenance': {
            'gpu': gpu.get('checkpoint_provenance'),
            'npu': npu.get('checkpoint_provenance'),
        },
        'module_types': {'gpu': gpu.get('module_types'), 'npu': npu.get('module_types')},
        'nodes': nodes,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n', encoding='utf-8')
    print(f'Layer target: {result["layer_target"]}')
    print('GPU/NPU runtime backend values are reported, but backend names may differ by design.')
    print('boundary                    rel_l2       cosine          max_abs')
    for node in A_G_NODES:
        values = [metric for metric in nodes[node].values() if 'relative_l2' in metric]
        if not values:
            print(f'{node:<28} missing')
            continue
        metric = max(values, key=lambda item: item['numel'])
        print(f'{node:<28} {format_percent(metric["relative_l2"]):>11} '
              f'{metric["cosine"]:.9f} {metric["max_abs"]:.8g}')
    if args.output:
        print(f'Full report: {args.output}')


if __name__ == '__main__':
    main()
