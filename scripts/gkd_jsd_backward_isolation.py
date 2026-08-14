# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def _raw_sha256(tensor):
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _summary(tensor):
    value = tensor.detach().float().cpu().contiguous()
    flat = value.reshape(-1)
    result = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'numel': tensor.numel(),
        'sha256_raw': _raw_sha256(tensor),
    }
    if flat.numel():
        result.update({
            'min': flat.min().item(),
            'max': flat.max().item(),
            'mean': flat.mean().item(),
            'std': flat.std(unbiased=False).item(),
            'norm': flat.norm().item(),
        })
    return result


def _gradient_topk(gradient, k):
    flat = gradient.detach().float().cpu().reshape(-1)
    k = min(k, flat.numel())
    _, indices = torch.topk(flat.abs(), k=k)
    vocab_size = gradient.shape[-1]
    seq_len = gradient.shape[-2]
    result = []
    for flat_index in indices.tolist():
        token_id = flat_index % vocab_size
        row = flat_index // vocab_size
        seq_idx = row % seq_len
        batch_idx = row // seq_len
        result.append({
            'flat_index': flat_index,
            'batch_idx': batch_idx,
            'seq_idx': seq_idx,
            'token_id': token_id,
            'value': flat[flat_index].item(),
        })
    return result


def _device(name):
    if name.startswith('npu'):
        import torch_npu  # noqa: F401
    return torch.device(name)


def run(args):
    payload = _load(args.input)
    device = _device(args.device)
    if args.fp32:
        os.environ['SWIFT_GKD_JSD_FP32'] = '1'
    else:
        os.environ.pop('SWIFT_GKD_JSD_FP32', None)

    student = payload['student_logits'].to(device).detach().requires_grad_(True)
    labels = payload['labels'].to(device)
    teacher = TeacherOutput(
        full_logits=(
            payload['teacher_logits'].to(device)
            if payload.get('teacher_logits') is not None else None),
        topk_logprobs=(
            payload['teacher_topk_logprobs'].to(device)
            if payload.get('teacher_topk_logprobs') is not None else None),
        topk_indices=(
            payload['teacher_topk_indices'].to(device)
            if payload.get('teacher_topk_indices') is not None else None),
        opsd_teacher_labels=(
            payload['opsd_teacher_labels'].to(device)
            if payload.get('opsd_teacher_labels') is not None else None),
    )
    total, num_valid = gkd_loss(
        student,
        teacher,
        labels,
        beta=float(payload['beta']),
        temperature=float(payload['temperature']),
    )
    loss = total / num_valid.float().clamp_min(1)
    loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'npu':
        torch.npu.synchronize(device)

    gradient = student.grad.detach().cpu().contiguous()
    result = {
        'device': str(device),
        'input_path': str(Path(args.input).resolve()),
        'step': payload.get('step'),
        'micro_batch': payload.get('micro_batch'),
        'beta': float(payload['beta']),
        'temperature': float(payload['temperature']),
        'fp32_jsd': bool(args.fp32),
        'loss': float(loss.detach().float().cpu()),
        'total': float(total.detach().float().cpu()),
        'num_valid': int(num_valid.detach().cpu()),
        'student_logits_sha256': _raw_sha256(payload['student_logits']),
        'teacher_logits_sha256': (
            _raw_sha256(payload['teacher_logits'])
            if payload.get('teacher_logits') is not None else None),
        'labels_sha256': _raw_sha256(payload['labels']),
        'gradient': _summary(gradient),
        'gradient_abs_topk': _gradient_topk(gradient, args.topk),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'result': result, 'gradient': gradient}, output)
    json_path = output.with_suffix('.json')
    with json_path.open('w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Saved full gradient: {output}')
    print(f'Saved summary: {json_path}')


def compare(args):
    gpu = _load(args.gpu)
    npu = _load(args.npu)
    gpu_result = gpu['result']
    npu_result = npu['result']
    gpu_gradient = gpu['gradient'].float().contiguous()
    npu_gradient = npu['gradient'].float().contiguous()
    if gpu_gradient.shape != npu_gradient.shape:
        raise ValueError(
            f'Gradient shape mismatch: GPU={tuple(gpu_gradient.shape)}, NPU={tuple(npu_gradient.shape)}')
    for key in ('student_logits_sha256', 'teacher_logits_sha256', 'labels_sha256'):
        if gpu_result.get(key) != npu_result.get(key):
            raise ValueError(f'Common input mismatch for {key}: GPU={gpu_result.get(key)}, NPU={npu_result.get(key)}')

    difference = gpu_gradient - npu_gradient
    gpu_flat = gpu_gradient.reshape(-1)
    npu_flat = npu_gradient.reshape(-1)
    diff_flat = difference.reshape(-1)
    cosine = F.cosine_similarity(gpu_flat.unsqueeze(0), npu_flat.unsqueeze(0), dim=1)[0]
    gpu_loss = float(gpu_result['loss'])
    npu_loss = float(npu_result['loss'])
    result = {
        'shape': list(gpu_gradient.shape),
        'common_inputs_match': True,
        'gpu_loss': gpu_loss,
        'npu_loss': npu_loss,
        'loss_abs_diff': abs(gpu_loss - npu_loss),
        'loss_relative_diff': abs(gpu_loss - npu_loss) / max(abs(gpu_loss), abs(npu_loss), 1e-30),
        'gradient_max_abs': diff_flat.abs().max().item(),
        'gradient_mean_abs': diff_flat.abs().mean().item(),
        'gradient_relative_l2': diff_flat.norm().item() / max(npu_flat.norm().item(), 1e-30),
        'gradient_cosine': cosine.item(),
        'gradient_different_ratio': (diff_flat != 0).float().mean().item(),
        'gpu_gradient_norm': gpu_flat.norm().item(),
        'npu_gradient_norm': npu_flat.norm().item(),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n', encoding='utf-8')
        print(f'Saved comparison: {output}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run and compare isolated GKD JSD backward computations.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--device', required=True, help='cuda:0 or npu:0')
    run_parser.add_argument('--input', required=True, help='Captured common JSD input .pt')
    run_parser.add_argument('--output', required=True, help='Output .pt containing the full student-logits gradient')
    run_parser.add_argument('--topk', type=int, default=32)
    run_parser.add_argument('--fp32', action='store_true', help='Compute JSD internals in FP32')
    run_parser.set_defaults(func=run)

    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--gpu', required=True)
    compare_parser.add_argument('--npu', required=True)
    compare_parser.add_argument('--output')
    compare_parser.set_defaults(func=compare)

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    arguments.func(arguments)
