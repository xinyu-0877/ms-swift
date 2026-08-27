import argparse
import hashlib
import json

import torch


def tensor_sha256(tensor):
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def load_output(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    output = payload['output'] if isinstance(payload, dict) else payload
    return payload, output.detach().contiguous()


def compare(args):
    gpu_payload, gpu = load_output(args.gpu)
    npu_payload, npu = load_output(args.npu)
    gpu_mode = gpu_payload.get('mode') if isinstance(gpu_payload, dict) else None
    npu_mode = npu_payload.get('mode') if isinstance(npu_payload, dict) else None
    gpu_target = gpu_payload.get('target_prefix') if isinstance(gpu_payload, dict) else None
    npu_target = npu_payload.get('target_prefix') if isinstance(npu_payload, dict) else None
    gpu_module_type = gpu_payload.get('module_type') if isinstance(gpu_payload, dict) else None
    npu_module_type = npu_payload.get('module_type') if isinstance(npu_payload, dict) else None
    gpu_qkv = gpu_payload.get('qkv') if isinstance(gpu_payload, dict) else None
    npu_qkv = npu_payload.get('qkv') if isinstance(npu_payload, dict) else None
    invariants = {
        'gpu_is_replay': gpu_mode == 'replay',
        'npu_is_capture_or_replay': npu_mode in {'capture', 'replay'},
        'target_match': gpu_target == npu_target,
        'expected_target_match': args.expected_target is None or gpu_target == args.expected_target,
        'gpu_module_type_supported': gpu_module_type in {'TEDotProductAttention', 'DotProductAttention'},
        'npu_module_type_supported': npu_module_type in {'TEDotProductAttention', 'DotProductAttention'},
        'qkv_present': isinstance(gpu_qkv, list) and isinstance(npu_qkv, list)
        and len(gpu_qkv) == 3 and len(npu_qkv) == 3,
        'common_qkv_match': gpu_qkv == npu_qkv,
        'numel_match': gpu.numel() == npu.numel(),
    }
    failed = [name for name, value in invariants.items() if not value]
    if failed:
        raise ValueError(f'Core-attention replay invariants failed: {failed}; invariants={invariants}')
    if gpu.numel() != npu.numel():
        raise ValueError(f'Output numel mismatch: GPU {gpu.numel()}, NPU {npu.numel()}.')

    gpu_flat = gpu.double().reshape(-1)
    npu_flat = npu.double().reshape(-1)
    diff = gpu_flat - npu_flat
    gpu_norm = torch.linalg.vector_norm(gpu_flat)
    npu_norm = torch.linalg.vector_norm(npu_flat)
    denominator = torch.maximum(gpu_norm, npu_norm).clamp_min(1e-30)
    cosine_denominator = (gpu_norm * npu_norm).clamp_min(1e-30)
    cosine = (torch.dot(gpu_flat, npu_flat) / cosine_denominator).clamp(-1, 1)
    result = {
        'invariants': invariants,
        'target_prefix': gpu_target,
        'gpu_module_type': gpu_module_type,
        'npu_module_type': npu_module_type,
        'common_qkv_identity': {
            name: identity for name, identity in zip(('query', 'key', 'value'), gpu_qkv)
        },
        'gpu_shape': list(gpu.shape),
        'npu_shape': list(npu.shape),
        'numel': gpu.numel(),
        'gpu_dtype': str(gpu.dtype),
        'npu_dtype': str(npu.dtype),
        'gpu_sha256': tensor_sha256(gpu),
        'npu_sha256': tensor_sha256(npu),
        'max_abs': diff.abs().max().item(),
        'mean_abs': diff.abs().mean().item(),
        'relative_l2': (torch.linalg.vector_norm(diff) / denominator).item(),
        'cosine': cosine.item(),
        'different_ratio': (gpu_flat != npu_flat).double().mean().item(),
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
        'gpu_mode': gpu_mode,
        'npu_mode': npu_mode,
    }
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f'Full report: {args.output}')
    print(f'Target: {gpu_target} (GPU={gpu_module_type}, NPU={npu_module_type})')
    print(f'Invariants: all passed ({len(invariants)})')
    print('Common Q/K/V: matched')
    for name, identity in result['common_qkv_identity'].items():
        print(
            f'  {name}: shape={identity["shape"]} dtype={identity["dtype"]} '
            f'sha256={identity["sha256"]}')
    print(f'Output shapes: GPU={list(gpu.shape)}, NPU={list(npu.shape)}, numel={gpu.numel()}')
    print(f'relative_l2: {result["relative_l2"]:.9%}')
    print(f'cosine: {result["cosine"]:.12f}')
    print(f'max_abs: {result["max_abs"]:.12g}')
    print(f'norms: GPU={result["gpu_norm"]:.12g}, NPU={result["npu_norm"]:.12g}')


def parse_args():
    parser = argparse.ArgumentParser(description='Compare common-QKV GPU/NPU core-attention outputs.')
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--npu', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-target', help='Expected core-attention module path.')
    return parser.parse_args()


if __name__ == '__main__':
    compare(parse_args())
