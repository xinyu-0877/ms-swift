"""Run a CPU reference for a captured Megatron FC1/RMSNorm boundary.

The capture format is intentionally inspected at runtime because mcore bridge
versions expose slightly different parameter names.  This utility never
imports torch_npu and therefore remains a CPU-only baseline.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _identity(value):
    if not torch.is_tensor(value):
        return value
    return {'shape': list(value.shape), 'dtype': str(value.dtype)}


def inspect_payload(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    print(json.dumps({key: _identity(value) for key, value in payload.items()}, indent=2, default=str))


def _metrics(reference, actual):
    x = reference.detach().double().reshape(-1)
    y = actual.detach().double().reshape(-1)
    diff = x - y
    xn = torch.linalg.vector_norm(x)
    yn = torch.linalg.vector_norm(y)
    return {
        'shape': list(reference.shape),
        'max_abs': float(diff.abs().max()) if diff.numel() else 0.0,
        'mean_abs': float(diff.abs().mean()) if diff.numel() else 0.0,
        'absolute_l2': float(torch.linalg.vector_norm(diff)),
        'relative_l2': float(torch.linalg.vector_norm(diff) / yn.clamp_min(1e-30)),
        'cosine': float((torch.dot(x, y) / (xn * yn).clamp_min(1e-30)).clamp(-1, 1)),
        'reference_norm': float(xn),
        'actual_norm': float(yn),
    }


def _cpu_fc1(payload):
    x = payload['input'].cpu()
    weight = payload['parameters']['weight'].cpu()
    norm_weight = payload['parameters']['layer_norm_weight'].cpu()
    eps = float(payload.get('module_config', {}).get('epsilon', 1e-6))
    x_var = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(x_var + eps)
    normalized = (normalized * norm_weight.float()).to(x.dtype)
    return F.linear(normalized, weight)


def run(args):
    payload = torch.load(args.input, map_location='cpu', weights_only=True)
    x = payload['input'].detach().cpu().requires_grad_(True)
    weight = payload['parameters']['weight'].detach().cpu().requires_grad_(True)
    norm_weight = payload['parameters']['layer_norm_weight'].detach().cpu().requires_grad_(True)
    working = dict(payload, input=x, parameters=dict(payload['parameters'], weight=weight,
                                                      layer_norm_weight=norm_weight))
    output = _cpu_fc1(working)
    result = {
        'target': payload.get('target'),
        'step': payload.get('step'),
        'micro_batch': payload.get('micro_batch'),
        'implementation': 'cpu_fp32_rms_stats_bf16_linear',
        'output': output,
        'output_identity': _identity(output),
    }
    output_gradients = payload.get('output_gradients', {})
    if output_gradients:
        grad = output_gradients.get('output.0')
        if grad is None:
            grad = next(iter(output_gradients.values()))
        grad = grad.detach().cpu()
        input_gradient, weight_gradient, norm_gradient = torch.autograd.grad(
            output, (x, weight, norm_weight), grad_outputs=grad, retain_graph=False)
        result.update({
            'input_gradient': input_gradient.detach(),
            'weight_gradient': weight_gradient.detach(),
            'layer_norm_weight_gradient': norm_gradient.detach(),
        })
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'output'}, indent=2, default=str))


def compare(args):
    cpu = torch.load(args.cpu, map_location='cpu', weights_only=True)
    npu = torch.load(args.npu, map_location='cpu', weights_only=True)
    npu_output = npu['forward_outputs']['output.0']
    result = {
        'target_match': cpu.get('target') == npu.get('target'),
        'step_match': cpu.get('step') == npu.get('step'),
        'micro_batch_match': cpu.get('micro_batch') == npu.get('micro_batch'),
        'forward': _metrics(cpu['output'], npu_output),
    }
    if not all(result[key] for key in ('target_match', 'step_match', 'micro_batch_match')):
        raise ValueError(f'FC1 CPU/NPU provenance mismatch: {result}')
    for key, npu_key in (('input_gradient', 'input_gradient'),
                         ('weight_gradient', 'parameter_gradients')):
        if key in cpu and npu_key in npu:
            npu_value = npu[npu_key]
            if isinstance(npu_value, dict):
                npu_value = npu_value.get('weight' if key == 'weight_gradient' else 'input_gradient')
            if npu_value is None:
                raise ValueError(f'Missing NPU value for {key}')
            result[key] = _metrics(cpu[key], npu_value)
    if 'layer_norm_weight_gradient' in cpu and isinstance(npu.get('parameter_gradients'), dict):
        result['layer_norm_weight_gradient'] = _metrics(
            cpu['layer_norm_weight_gradient'], npu['parameter_gradients']['layer_norm_weight'])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('inspect', 'run', 'compare'))
    parser.add_argument('--input')
    parser.add_argument('--cpu')
    parser.add_argument('--npu')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'inspect':
        inspect_payload(args.input)
    elif args.command == 'run':
        run(args)
    else:
        if not args.cpu or not args.npu:
            raise SystemExit('compare requires --cpu and --npu')
        compare(args)


if __name__ == '__main__':
    main()
