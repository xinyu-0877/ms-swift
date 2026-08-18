# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import hashlib
import json
from pathlib import Path

import torch


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


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
        cosine = torch.dot(gpu, npu).item() / max((gpu_norm * npu_norm).item(), epsilon)
    return {
        'gpu_shape': list(gpu_tensor.shape),
        'npu_shape': list(npu_tensor.shape),
        'numel': gpu.numel(),
        'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
        'mean_abs': difference.abs().mean().item() if difference.numel() else 0.0,
        'absolute_l2': difference_norm.item(),
        'relative_l2': difference_norm.item() / max(npu_norm.item(), epsilon),
        'cosine': cosine,
        'different_ratio': (gpu != npu).double().mean().item() if difference.numel() else 0.0,
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
    }


def compare_tensor_maps(gpu_values, npu_values):
    result = {}
    for name in sorted(set(gpu_values) | set(npu_values)):
        gpu_value = gpu_values.get(name)
        npu_value = npu_values.get(name)
        if gpu_value is None or npu_value is None:
            result[name] = {
                'gpu_present': gpu_value is not None,
                'npu_present': npu_value is not None,
            }
        else:
            result[name] = tensor_metrics(gpu_value, npu_value)
    return result


def identities(values):
    return {name: tensor_identity(value) for name, value in values.items()}


def normalize_module_config(config):
    aliases = {
        'epsilon': (
            'epsilon', 'module.eps', 'module.epsilon', 'module.layernorm_epsilon',
            'module.layer_norm_epsilon', 'config.eps', 'config.epsilon',
            'config.layernorm_epsilon', 'config.layer_norm_epsilon'),
        'normalization': ('normalization', 'module.normalization', 'config.normalization'),
        'zero_centered_gamma': (
            'zero_centered_gamma', 'module.zero_centered_gamma',
            'config.zero_centered_gamma', 'config.layernorm_zero_centered_gamma'),
    }
    result = {}
    for canonical_name, candidate_names in aliases.items():
        for candidate_name in candidate_names:
            value = config.get(candidate_name)
            if isinstance(value, (bool, int, float, str)):
                result[canonical_name] = value
                break
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Compare GPU/NPU LayerNorm+linear_fc1 isolation replay results.')
    parser.add_argument('--common', required=True, help='NPU-captured fc1_module_common.pt.')
    parser.add_argument('--gpu', required=True, help='GPU fc1_module_* result.')
    parser.add_argument('--npu', required=True, help='NPU fc1_module_* result.')
    parser.add_argument('--output', help='Optional JSON result path.')
    args = parser.parse_args()

    common = torch.load(args.common, map_location='cpu', weights_only=True)
    gpu = torch.load(args.gpu, map_location='cpu', weights_only=True)
    npu = torch.load(args.npu, map_location='cpu', weights_only=True)
    common_input_identity = tensor_identity(common['input'])
    common_parameter_identities = identities(common.get('parameters', {}))
    common_dout_identities = identities(common.get('output_gradients', {}))
    gpu_dout_identities = identities(gpu.get('used_output_gradients', {}))
    npu_dout_identities = identities(npu.get('used_output_gradients', {}))
    common_module_config = normalize_module_config(common.get('module_config', {}))
    gpu_module_config = normalize_module_config(gpu.get('module_config', {}))
    npu_module_config = normalize_module_config(npu.get('module_config', {}))
    common_module_config_match = all(
        gpu_module_config.get(name) == npu_module_config.get(name) == value
        for name, value in common_module_config.items())

    result = {
        'target_match': gpu.get('target') == npu.get('target') == common.get('target'),
        'module_type_match': (
            gpu.get('module_type') == npu.get('module_type') == common.get('module_type')),
        'module_signature_match': gpu.get('module_signature') == npu.get('module_signature'),
        'module_config_match': common_module_config_match,
        'common_module_config': common_module_config,
        'gpu_module_config': gpu_module_config,
        'npu_module_config': npu_module_config,
        'gpu_mode': gpu.get('mode'),
        'npu_mode': npu.get('mode'),
        'execution_modes_valid': (
            gpu.get('mode') == 'replay' and npu.get('mode') in {'capture', 'replay'}),
        'common_input_match': (
            gpu.get('input_identity') == npu.get('input_identity') == common_input_identity),
        'common_parameters_match': (
            gpu.get('parameter_identities')
            == npu.get('parameter_identities')
            == common_parameter_identities),
        'common_dout_match': (
            gpu_dout_identities == npu_dout_identities == common_dout_identities),
        'forward_outputs': compare_tensor_maps(
            gpu.get('forward_outputs', {}), npu.get('forward_outputs', {})),
        'input_gradient': tensor_metrics(gpu['input_gradient'], npu['input_gradient']),
        'parameter_gradients': compare_tensor_maps(
            gpu.get('parameter_gradients', {}), npu.get('parameter_gradients', {})),
    }
    required_matches = (
        'target_match',
        'module_type_match',
        'module_config_match',
        'execution_modes_valid',
        'common_input_match',
        'common_parameters_match',
        'common_dout_match',
    )
    mismatches = [name for name in required_matches if not result[name]]
    if mismatches:
        raise ValueError(f'FC1 isolation comparison invariants failed: {mismatches}')

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
