# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
from pathlib import Path

import torch


def metrics(gpu_tensor, npu_tensor):
    gpu = gpu_tensor.detach().double().cpu().contiguous().reshape(-1)
    npu = npu_tensor.detach().double().cpu().contiguous().reshape(-1)
    if gpu.numel() != npu.numel():
        raise ValueError(f'Tensor numel mismatch: GPU={gpu.numel()}, NPU={npu.numel()}.')
    difference = gpu - npu
    gpu_norm = torch.linalg.vector_norm(gpu)
    npu_norm = torch.linalg.vector_norm(npu)
    difference_norm = torch.linalg.vector_norm(difference)
    denominator = max(npu_norm.item(), torch.finfo(torch.float64).eps)
    cosine_denominator = max((gpu_norm * npu_norm).item(), torch.finfo(torch.float64).eps)
    return {
        'gpu_shape': list(gpu_tensor.shape),
        'npu_shape': list(npu_tensor.shape),
        'numel': gpu.numel(),
        'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
        'mean_abs': difference.abs().mean().item() if difference.numel() else 0.0,
        'absolute_l2': difference_norm.item(),
        'relative_l2': difference_norm.item() / denominator,
        'cosine': torch.dot(gpu, npu).item() / cosine_denominator,
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare complete Layer-N self-attention common-input forward replay.')
    parser.add_argument('--gpu-replay', required=True)
    parser.add_argument('--npu-capture', required=True)
    parser.add_argument('--expected-layer', default='decoder.layers.0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    gpu = torch.load(args.gpu_replay, map_location='cpu', weights_only=True)
    npu = torch.load(args.npu_capture, map_location='cpu', weights_only=True)
    invariants = {
        'gpu_mode_replay': gpu.get('mode') == 'replay',
        'npu_mode_capture': npu.get('mode') == 'capture',
        'layer_target_match': gpu.get('layer_target') == npu.get('layer_target'),
        'expected_layer_match': gpu.get('layer_target') == args.expected_layer,
        'attention_target_match': gpu.get('attention_target') == npu.get('attention_target'),
        'mlp_target_match': gpu.get('mlp_target') == npu.get('mlp_target'),
        'step_match': gpu.get('step') == npu.get('step'),
        'micro_batch_match': gpu.get('micro_batch') == npu.get('micro_batch'),
        'runtime_config_match': gpu.get('runtime_config') == npu.get('runtime_config'),
        'common_inputs_match': gpu.get('input_identities') == npu.get('input_identities'),
        'common_parameters_match': (
            gpu.get('parameter_identities') == npu.get('parameter_identities')),
        'attention_output_paths_match': (
            set(gpu.get('attention_outputs', {})) == set(npu.get('attention_outputs', {}))),
        'post_attention_present': (
            gpu.get('post_attention_input') is not None
            and npu.get('post_attention_input') is not None),
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise ValueError(f'Self-attention forward invariants failed: {failed}.')

    attention_outputs = {
        path: metrics(gpu['attention_outputs'][path], npu['attention_outputs'][path])
        for path in sorted(npu['attention_outputs'])
    }
    post_attention = metrics(gpu['post_attention_input'], npu['post_attention_input'])
    result = {
        'invariants': invariants,
        'layer_target': gpu.get('layer_target'),
        'attention_target': gpu.get('attention_target'),
        'runtime_config': gpu.get('runtime_config'),
        'common_input_count': len(gpu.get('input_identities', {})),
        'common_parameter_count': len(gpu.get('parameter_identities', {})),
        'attention_outputs': attention_outputs,
        'post_attention_input': post_attention,
        'interpretation': (
            'attention_outputs are the complete self_attention module outputs; '
            'post_attention_input is the actual same-layer MLP input after the attention '
            'residual/bias/dropout path.'),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f'Full report: {output_path}')
    print(f'Layer: {result["layer_target"]}')
    print(f'Invariants: all passed ({len(invariants)})')
    print(
        f'Common boundary: {result["common_input_count"]} tensor inputs, '
        f'{result["common_parameter_count"]} parameters')
    print('Self-attention outputs:')
    for path, value in attention_outputs.items():
        print(
            f'  {path}: rel_l2={value["relative_l2"]:.9%} '
            f'cos={value["cosine"]:.12f} max_abs={value["max_abs"]:.12g}')
    print(
        f'Post-attention / pre-MLP: rel_l2={post_attention["relative_l2"]:.9%} '
        f'cos={post_attention["cosine"]:.12f} max_abs={post_attention["max_abs"]:.12g}')


if __name__ == '__main__':
    main()
