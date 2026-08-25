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
    cosine = 1.0 if gpu_norm.item() == npu_norm.item() == 0.0 else (
        torch.dot(gpu, npu).item() / max((gpu_norm * npu_norm).item(), epsilon))
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


def load_payload(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def main():
    parser = argparse.ArgumentParser(
        description='Compare complete MLP forward outputs with common NPU input and parameters.')
    parser.add_argument('--gpu-replay', required=True)
    parser.add_argument('--npu-capture', required=True)
    parser.add_argument('--expected-layer', default='decoder.layers.0')
    parser.add_argument('--output')
    args = parser.parse_args()

    gpu = load_payload(args.gpu_replay)
    npu = load_payload(args.npu_capture)
    gpu_dout = {
        name: tensor_identity(value) for name, value in gpu.get('output_gradients', {}).items()
    }
    npu_dout = {
        name: tensor_identity(value) for name, value in npu.get('output_gradients', {}).items()
    }
    invariants = {
        'gpu_mode_replay': gpu.get('mode') == 'replay',
        'npu_mode_capture': npu.get('mode') == 'capture',
        'layer_target_match': gpu.get('layer_target') == npu.get('layer_target'),
        'expected_layer_match': gpu.get('layer_target') == args.expected_layer,
        'mlp_target_match': gpu.get('mlp_target') == npu.get('mlp_target'),
        'step_match': gpu.get('step') == npu.get('step'),
        'micro_batch_match': gpu.get('micro_batch') == npu.get('micro_batch'),
        'module_type_match': gpu.get('module_type') == npu.get('module_type'),
        'common_input_match': gpu.get('input_identity') == npu.get('input_identity'),
        'common_parameters_match': gpu.get('parameter_identities') == npu.get('parameter_identities'),
        'common_dout_match': gpu_dout == npu_dout,
        'output_paths_match': set(gpu.get('forward_outputs', {})) == set(npu.get('forward_outputs', {})),
    }
    failed = [name for name, value in invariants.items() if not value]
    if failed:
        raise ValueError(f'MLP common-input forward invariants failed: {failed}')

    outputs = {
        name: tensor_metrics(gpu['forward_outputs'][name], npu['forward_outputs'][name])
        for name in sorted(npu['forward_outputs'])
    }
    result = {
        'invariants': invariants,
        'layer_target': gpu.get('layer_target'),
        'mlp_target': gpu.get('mlp_target'),
        'common_input_identity': gpu.get('input_identity'),
        'common_parameter_count': len(gpu.get('parameter_identities', {})),
        'forward_outputs': outputs,
        'interpretation': (
            'GPU replay and NPU capture used the same captured MLP input and parameters. '
            'The common dout is verified only because the training-path replay requires backward completion.'),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
