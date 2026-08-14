import argparse
import hashlib
import json

import torch
import torch.nn.functional as F


def tensor_sha256(tensor):
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'npu':
        torch.npu.synchronize(device)


def load_fc1_output(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    tensor = data['fc1_output'] if isinstance(data, dict) else data
    if tensor.shape[-1] % 2 != 0:
        raise ValueError(f'FC1 output last dimension must be even, got {tensor.shape[-1]}.')
    return tensor.detach().contiguous()


def run(args):
    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    fc1_output = load_fc1_output(args.input)
    common_input_sha256 = tensor_sha256(fc1_output)
    fc1_output = fc1_output.to(device)

    gate, up = torch.chunk(fc1_output, 2, dim=-1)
    silu_gate = F.silu(gate)
    swiglu_output = silu_gate * up
    synchronize(device)

    result = {
        'device': str(device),
        'input_shape': list(fc1_output.shape),
        'input_dtype': str(fc1_output.dtype),
        'input_sha256': common_input_sha256,
        'silu_gate': silu_gate.detach().contiguous().cpu(),
        'swiglu_output': swiglu_output.detach().contiguous().cpu(),
    }
    torch.save(result, args.output)
    print(json.dumps({
        'output': args.output,
        'device': str(device),
        'input_shape': result['input_shape'],
        'input_dtype': result['input_dtype'],
        'input_sha256': common_input_sha256,
        'silu_gate_sha256': tensor_sha256(result['silu_gate']),
        'swiglu_output_sha256': tensor_sha256(result['swiglu_output']),
    }, indent=2))


def tensor_metrics(gpu_tensor, npu_tensor):
    gpu_tensor = gpu_tensor.float().reshape(-1)
    npu_tensor = npu_tensor.float().reshape(-1)
    if gpu_tensor.shape != npu_tensor.shape:
        raise ValueError(f'Tensor shape mismatch: GPU {gpu_tensor.shape}, NPU {npu_tensor.shape}.')
    diff = gpu_tensor - npu_tensor
    gpu_norm = torch.linalg.vector_norm(gpu_tensor)
    npu_norm = torch.linalg.vector_norm(npu_tensor)
    denominator = torch.maximum(gpu_norm, npu_norm).clamp_min(1e-30)
    cosine_denominator = (gpu_norm * npu_norm).clamp_min(1e-30)
    return {
        'shape': list(gpu_tensor.shape),
        'max_abs': diff.abs().max().item(),
        'mean_abs': diff.abs().mean().item(),
        'relative_l2': (torch.linalg.vector_norm(diff) / denominator).item(),
        'cosine': (torch.dot(gpu_tensor, npu_tensor) / cosine_denominator).item(),
        'different_ratio': (gpu_tensor != npu_tensor).float().mean().item(),
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
    }


def compare(args):
    gpu = torch.load(args.gpu, map_location='cpu', weights_only=False)
    npu = torch.load(args.npu, map_location='cpu', weights_only=False)
    result = {
        'common_input_match': gpu['input_sha256'] == npu['input_sha256'],
        'gpu_input_sha256': gpu['input_sha256'],
        'npu_input_sha256': npu['input_sha256'],
        'silu_gate': tensor_metrics(gpu['silu_gate'], npu['silu_gate']),
        'swiglu_output': tensor_metrics(gpu['swiglu_output'], npu['swiglu_output']),
    }
    if not result['common_input_match']:
        raise ValueError('GPU and NPU did not use the same FC1 output.')
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description='Isolate SwiGLU forward precision on GPU and NPU.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--device', required=True, help='For example: cuda:0 or npu:0.')
    run_parser.add_argument('--input', required=True, help='Captured swiglu_fc1_output.pt.')
    run_parser.add_argument('--output', required=True)
    run_parser.set_defaults(func=run)

    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--gpu', required=True)
    compare_parser.add_argument('--npu', required=True)
    compare_parser.add_argument('--output', required=True)
    compare_parser.set_defaults(func=compare)
    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    parsed_args.func(parsed_args)
