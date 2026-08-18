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
    data = torch.load(path, map_location='cpu', weights_only=True)
    tensor = data['fc1_output'] if isinstance(data, dict) else data
    if tensor.shape[-1] % 2 != 0:
        raise ValueError(f'FC1 output last dimension must be even, got {tensor.shape[-1]}.')
    return tensor.detach().contiguous()


def load_backward_inputs(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, dict):
        raise ValueError('SwiGLU backward input must be a dictionary payload.')
    if 'fc1_output' not in data or 'swiglu_dout' not in data:
        raise ValueError('SwiGLU backward input requires fc1_output and swiglu_dout tensors.')
    fc1_output = data['fc1_output'].detach().contiguous()
    swiglu_dout = data['swiglu_dout'].detach().contiguous()
    bias = data.get('fc1_bias')
    bias = bias.detach().contiguous() if bias is not None else None
    if fc1_output.shape[-1] % 2 != 0:
        raise ValueError(f'FC1 output last dimension must be even, got {fc1_output.shape[-1]}.')
    if tuple(swiglu_dout.shape) != tuple(fc1_output.shape[:-1]) + (fc1_output.shape[-1] // 2,):
        raise ValueError(
            f'SwiGLU dout shape {tuple(swiglu_dout.shape)} is incompatible with '
            f'FC1 output shape {tuple(fc1_output.shape)}.')
    if swiglu_dout.dtype != fc1_output.dtype:
        raise ValueError(
            f'SwiGLU dout dtype {swiglu_dout.dtype} does not match FC1 output dtype {fc1_output.dtype}.')
    if bias is not None and bias.dtype != fc1_output.dtype:
        raise ValueError(f'FC1 bias dtype {bias.dtype} does not match FC1 output dtype {fc1_output.dtype}.')
    return data, fc1_output, swiglu_dout, bias


def apply_swiglu(fc1_output, bias, implementation):
    if implementation == 'megatron-fused':
        from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
        fused_bias = bias if bias is not None else torch.zeros_like(fc1_output)
        return bias_swiglu(fc1_output, fused_bias)
    if implementation != 'torch':
        raise ValueError(f'Unsupported SwiGLU implementation: {implementation}')
    if bias is not None:
        fc1_output = fc1_output + bias
    gate, up = torch.chunk(fc1_output, 2, dim=-1)
    return F.silu(gate) * up


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


def run_backward(args):
    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    payload, fc1_output, swiglu_dout, bias = load_backward_inputs(args.input)
    input_sha256 = tensor_sha256(fc1_output)
    dout_sha256 = tensor_sha256(swiglu_dout)
    bias_sha256 = tensor_sha256(bias) if bias is not None else None

    fc1_output = fc1_output.to(device).requires_grad_(True)
    swiglu_dout = swiglu_dout.to(device=device)
    bias = bias.to(device=device, dtype=fc1_output.dtype) if bias is not None else None
    swiglu_output = apply_swiglu(fc1_output, bias, args.implementation)
    if tuple(swiglu_output.shape) != tuple(swiglu_dout.shape):
        raise ValueError(
            f'SwiGLU output shape {tuple(swiglu_output.shape)} does not match '
            f'common dout shape {tuple(swiglu_dout.shape)}.')
    (fc1_output_gradient,) = torch.autograd.grad(
        outputs=swiglu_output,
        inputs=fc1_output,
        grad_outputs=swiglu_dout,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    synchronize(device)

    result = {
        'device': str(device),
        'implementation': args.implementation,
        'target': payload.get('target'),
        'dout_target': payload.get('dout_target'),
        'step': payload.get('step'),
        'micro_batch': payload.get('micro_batch'),
        'input_shape': list(fc1_output.shape),
        'input_dtype': str(fc1_output.dtype),
        'input_sha256': input_sha256,
        'dout_shape': list(swiglu_dout.shape),
        'dout_dtype': str(swiglu_dout.dtype),
        'dout_sha256': dout_sha256,
        'bias_sha256': bias_sha256,
        'fc1_output_gradient': fc1_output_gradient.detach().contiguous().cpu(),
    }
    torch.save(result, args.output)
    print(json.dumps({
        'output': args.output,
        'device': result['device'],
        'implementation': result['implementation'],
        'target': result['target'],
        'dout_target': result['dout_target'],
        'input_shape': result['input_shape'],
        'input_dtype': result['input_dtype'],
        'input_sha256': result['input_sha256'],
        'dout_shape': result['dout_shape'],
        'dout_dtype': result['dout_dtype'],
        'dout_sha256': result['dout_sha256'],
        'bias_sha256': result['bias_sha256'],
        'gradient_sha256': tensor_sha256(result['fc1_output_gradient']),
    }, indent=2))


def tensor_metrics(gpu_tensor, npu_tensor, denominator_mode='max'):
    gpu_tensor = gpu_tensor.double().reshape(-1)
    npu_tensor = npu_tensor.double().reshape(-1)
    if gpu_tensor.shape != npu_tensor.shape:
        raise ValueError(f'Tensor shape mismatch: GPU {gpu_tensor.shape}, NPU {npu_tensor.shape}.')
    diff = gpu_tensor - npu_tensor
    gpu_norm = torch.linalg.vector_norm(gpu_tensor)
    npu_norm = torch.linalg.vector_norm(npu_tensor)
    if denominator_mode == 'npu':
        denominator = npu_norm.clamp_min(torch.finfo(torch.float64).eps)
    elif denominator_mode == 'max':
        denominator = torch.maximum(gpu_norm, npu_norm).clamp_min(1e-30)
    else:
        raise ValueError(f'Unsupported denominator mode: {denominator_mode}')
    cosine_denominator = (gpu_norm * npu_norm).clamp_min(1e-30)
    cosine = (torch.dot(gpu_tensor, npu_tensor) / cosine_denominator).clamp(-1, 1)
    return {
        'shape': list(gpu_tensor.shape),
        'max_abs': diff.abs().max().item(),
        'mean_abs': diff.abs().mean().item(),
        'relative_l2': (torch.linalg.vector_norm(diff) / denominator).item(),
        'cosine': cosine.item(),
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


def compare_backward(args):
    gpu = torch.load(args.gpu, map_location='cpu', weights_only=True)
    npu = torch.load(args.npu, map_location='cpu', weights_only=True)
    result = {
        'common_input_match': gpu.get('input_sha256') == npu.get('input_sha256'),
        'common_dout_match': gpu.get('dout_sha256') == npu.get('dout_sha256'),
        'common_bias_match': gpu.get('bias_sha256') == npu.get('bias_sha256'),
        'implementation_match': gpu.get('implementation') == npu.get('implementation'),
        'gpu_device': gpu.get('device'),
        'npu_device': npu.get('device'),
        'target_match': gpu.get('target') == npu.get('target'),
        'dout_target_match': gpu.get('dout_target') == npu.get('dout_target'),
        'fc1_output_gradient': tensor_metrics(
            gpu['fc1_output_gradient'], npu['fc1_output_gradient'], denominator_mode='npu'),
    }
    required_matches = (
        'common_input_match',
        'common_dout_match',
        'common_bias_match',
        'implementation_match',
        'target_match',
        'dout_target_match',
    )
    mismatches = [key for key in required_matches if not result[key]]
    if mismatches:
        raise ValueError(f'SwiGLU backward comparison invariants failed: {mismatches}')
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description='Isolate SwiGLU forward/backward precision on GPU and NPU.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--device', required=True, help='For example: cuda:0 or npu:0.')
    run_parser.add_argument('--input', required=True, help='Captured swiglu_fc1_output.pt.')
    run_parser.add_argument('--output', required=True)
    run_parser.set_defaults(func=run)

    backward_parser = subparsers.add_parser('run-backward')
    backward_parser.add_argument('--device', required=True, help='For example: cuda:0 or npu:0.')
    backward_parser.add_argument(
        '--input', required=True, help='Captured payload containing common fc1_output and swiglu_dout.')
    backward_parser.add_argument(
        '--implementation', choices=('torch', 'megatron-fused'), default='torch',
        help='Use the mathematical PyTorch path or Megatron fused bias-SwiGLU implementation.')
    backward_parser.add_argument('--output', required=True)
    backward_parser.set_defaults(func=run_backward)

    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--gpu', required=True)
    compare_parser.add_argument('--npu', required=True)
    compare_parser.add_argument('--output', required=True)
    compare_parser.set_defaults(func=compare)

    compare_backward_parser = subparsers.add_parser('compare-backward')
    compare_backward_parser.add_argument('--gpu', required=True)
    compare_backward_parser.add_argument('--npu', required=True)
    compare_backward_parser.add_argument('--output', required=True)
    compare_backward_parser.set_defaults(func=compare_backward)
    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    parsed_args.func(parsed_args)
