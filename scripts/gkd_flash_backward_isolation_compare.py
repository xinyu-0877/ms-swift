# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256_fp32(tensor):
    value = tensor.detach().float().cpu().contiguous().reshape(-1)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _compare_tensor(gpu_tensor, npu_tensor):
    gpu = gpu_tensor.detach().double().cpu().contiguous().reshape(-1)
    npu = npu_tensor.detach().double().cpu().contiguous().reshape(-1)
    if gpu.numel() != npu.numel():
        raise ValueError(f'Tensor sizes differ: GPU={gpu.numel()}, NPU={npu.numel()}')

    difference = gpu - npu
    gpu_norm = torch.linalg.vector_norm(gpu)
    npu_norm = torch.linalg.vector_norm(npu)
    difference_norm = torch.linalg.vector_norm(difference)
    denominator = max(npu_norm.item(), torch.finfo(torch.float64).eps)
    cosine_denominator = max(
        (gpu_norm * npu_norm).item(), torch.finfo(torch.float64).eps)
    return {
        'gpu_shape': list(gpu_tensor.shape),
        'npu_shape': list(npu_tensor.shape),
        'numel': gpu.numel(),
        'gpu_dtype': str(gpu_tensor.dtype),
        'npu_dtype': str(npu_tensor.dtype),
        'gpu_sha256_fp32': _sha256_fp32(gpu_tensor),
        'npu_sha256_fp32': _sha256_fp32(npu_tensor),
        'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
        'mean_abs': difference.abs().mean().item() if difference.numel() else 0.0,
        'relative_l2': difference_norm.item() / denominator,
        'cosine': torch.dot(gpu, npu).item() / cosine_denominator,
        'different_ratio': (gpu != npu).double().mean().item() if difference.numel() else 0.0,
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare isolated GPU/NPU Flash Attention backward gradients.')
    parser.add_argument('--gpu', required=True, help='GPU flash_backward_*.pt path.')
    parser.add_argument('--npu', required=True, help='NPU flash_backward_*.pt path.')
    parser.add_argument('--output', help='Optional JSON output path.')
    args = parser.parse_args()

    gpu_payload = torch.load(args.gpu, map_location='cpu', weights_only=True)
    npu_payload = torch.load(args.npu, map_location='cpu', weights_only=True)
    result = {
        'common_inputs_match': gpu_payload.get('qkv') == npu_payload.get('qkv'),
        'common_dout_match': gpu_payload.get('dout', {}).get('sha256')
        == npu_payload.get('dout', {}).get('sha256'),
        'gpu_output_shape': gpu_payload.get('output_shape'),
        'npu_output_shape': npu_payload.get('output_shape'),
        'dq': _compare_tensor(gpu_payload['dq'], npu_payload['dq']),
        'dk': _compare_tensor(gpu_payload['dk'], npu_payload['dk']),
        'dv': _compare_tensor(gpu_payload['dv'], npu_payload['dv']),
    }
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
