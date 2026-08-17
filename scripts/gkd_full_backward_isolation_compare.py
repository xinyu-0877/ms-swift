# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
from pathlib import Path

import torch


def _compare_tensor(gpu_tensor, npu_tensor):
    gpu = gpu_tensor.detach().double().cpu().contiguous().reshape(-1)
    npu = npu_tensor.detach().double().cpu().contiguous().reshape(-1)
    if gpu.numel() != npu.numel():
        raise ValueError(f'Tensor sizes differ: GPU={gpu.numel()}, NPU={npu.numel()}')
    difference = gpu - npu
    gpu_norm = torch.linalg.vector_norm(gpu)
    npu_norm = torch.linalg.vector_norm(npu)
    difference_norm = torch.linalg.vector_norm(difference)
    epsilon = torch.finfo(torch.float64).eps
    return {
        'gpu_shape': list(gpu_tensor.shape),
        'npu_shape': list(npu_tensor.shape),
        'numel': gpu.numel(),
        'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
        'mean_abs': difference.abs().mean().item() if difference.numel() else 0.0,
        'relative_l2': difference_norm.item() / max(npu_norm.item(), epsilon),
        'cosine': torch.dot(gpu, npu).item() / max((gpu_norm * npu_norm).item(), epsilon),
        'different_ratio': (gpu != npu).double().mean().item() if difference.numel() else 0.0,
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
    }


def _compare_optional(gpu_tensor, npu_tensor):
    if gpu_tensor is None or npu_tensor is None:
        return {
            'gpu_present': gpu_tensor is not None,
            'npu_present': npu_tensor is not None,
        }
    return _compare_tensor(gpu_tensor, npu_tensor)


def main():
    parser = argparse.ArgumentParser(description='Compare GPU/NPU full backward results with common dLogits.')
    parser.add_argument('--gpu', required=True, help='GPU full_backward_*.pt path.')
    parser.add_argument('--npu', required=True, help='NPU full_backward_*.pt path.')
    parser.add_argument('--output', help='Optional JSON output path.')
    args = parser.parse_args()

    gpu_payload = torch.load(args.gpu, map_location='cpu', weights_only=True)
    npu_payload = torch.load(args.npu, map_location='cpu', weights_only=True)
    gpu_modules = gpu_payload.get('module_gradients', {})
    npu_modules = npu_payload.get('module_gradients', {})
    module_names = sorted(set(gpu_modules) | set(npu_modules))
    modules = {}
    for name in module_names:
        gpu_module = gpu_modules.get(name)
        npu_module = npu_modules.get(name)
        if gpu_module is None or npu_module is None:
            modules[name] = {
                'gpu_present': gpu_module is not None,
                'npu_present': npu_module is not None,
            }
            continue
        modules[name] = {
            'module_type_match': gpu_module.get('module_type') == npu_module.get('module_type'),
            'input_gradient': _compare_optional(
                gpu_module.get('input_gradient'), npu_module.get('input_gradient')),
            'output_gradient': _compare_optional(
                gpu_module.get('output_gradient'), npu_module.get('output_gradient')),
        }

    gpu_parameters = gpu_payload.get('parameter_gradient_samples', {})
    npu_parameters = npu_payload.get('parameter_gradient_samples', {})
    parameter_names = sorted(set(gpu_parameters) | set(npu_parameters))
    parameters = {}
    for name in parameter_names:
        gpu_parameter = gpu_parameters.get(name)
        npu_parameter = npu_parameters.get(name)
        if gpu_parameter is None or npu_parameter is None:
            parameters[name] = {
                'gpu_present': gpu_parameter is not None,
                'npu_present': npu_parameter is not None,
            }
            continue
        indices_match = gpu_parameter.get('sample_indices') == npu_parameter.get('sample_indices')
        parameters[name] = {
            'sample_indices_match': indices_match,
            'sample': _compare_tensor(gpu_parameter['sample'], npu_parameter['sample']),
        }

    result = {
        'common_dlogits_match': gpu_payload.get('common_dlogits') == npu_payload.get('common_dlogits'),
        'gpu_mode': gpu_payload.get('mode'),
        'npu_mode': npu_payload.get('mode'),
        'modules': modules,
        'parameter_gradient_samples': parameters,
    }
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
