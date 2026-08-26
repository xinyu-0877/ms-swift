# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import math
import re
from pathlib import Path

import torch


EPSILON = torch.finfo(torch.float64).eps
CHUNK_NUMEL = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare native dLogits and decoder backward boundaries per micro-batch.')
    parser.add_argument('--gpu-dir', required=True)
    parser.add_argument('--npu-dir', required=True)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def discover(directory, step):
    payloads = {}
    pattern = f'microbatch_backward_*_step_{step:06d}_micro_*.pt'
    for path in Path(directory).glob(pattern):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') != 'swift_gkd_microbatch_backward_v1':
            continue
        if int(payload.get('step', -1)) != step:
            continue
        micro_batch = int(payload['micro_batch'])
        if micro_batch in payloads:
            raise ValueError(
                f'Duplicate micro-batch {micro_batch} captures under {directory}.')
        payloads[micro_batch] = (path, payload)
    if not payloads:
        raise ValueError(f'No step {step} micro-batch backward captures under {directory}.')
    return payloads


def compare_tensor(gpu_tensor, npu_tensor):
    if tuple(gpu_tensor.shape) != tuple(npu_tensor.shape):
        raise ValueError(
            f'Tensor shapes differ: GPU={tuple(gpu_tensor.shape)}, NPU={tuple(npu_tensor.shape)}')
    gpu = gpu_tensor.detach().cpu().contiguous().reshape(-1)
    npu = npu_tensor.detach().cpu().contiguous().reshape(-1)
    gpu_sq = npu_sq = difference_sq = dot = absolute_sum = 0.0
    max_abs = 0.0
    finite = True
    for start in range(0, gpu.numel(), CHUNK_NUMEL):
        end = min(start + CHUNK_NUMEL, gpu.numel())
        gpu_chunk = gpu[start:end].double()
        npu_chunk = npu[start:end].double()
        finite &= bool(torch.isfinite(gpu_chunk).all() and torch.isfinite(npu_chunk).all())
        difference = gpu_chunk - npu_chunk
        gpu_sq += torch.dot(gpu_chunk, gpu_chunk).item()
        npu_sq += torch.dot(npu_chunk, npu_chunk).item()
        difference_sq += torch.dot(difference, difference).item()
        dot += torch.dot(gpu_chunk, npu_chunk).item()
        absolute_sum += difference.abs().sum().item()
        if difference.numel():
            max_abs = max(max_abs, difference.abs().max().item())
    gpu_norm = math.sqrt(max(gpu_sq, 0.0))
    npu_norm = math.sqrt(max(npu_sq, 0.0))
    difference_norm = math.sqrt(max(difference_sq, 0.0))
    if gpu_norm == 0.0 and npu_norm == 0.0:
        cosine = 1.0
    elif gpu_norm == 0.0 or npu_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / max(gpu_norm * npu_norm, EPSILON)
    return {
        'shape': list(gpu_tensor.shape),
        'gpu_dtype': str(gpu_tensor.dtype),
        'npu_dtype': str(npu_tensor.dtype),
        'numel': gpu.numel(),
        'max_abs': max_abs,
        'mean_abs': absolute_sum / gpu.numel() if gpu.numel() else 0.0,
        'absolute_l2': difference_norm,
        'relative_l2': difference_norm / max(npu_norm, EPSILON),
        'cosine': cosine,
        'gpu_norm': gpu_norm,
        'npu_norm': npu_norm,
        'finite': finite,
    }


def compare_optional(gpu_tensor, npu_tensor):
    if gpu_tensor is None or npu_tensor is None:
        return {
            'gpu_present': gpu_tensor is not None,
            'npu_present': npu_tensor is not None,
        }
    return compare_tensor(gpu_tensor, npu_tensor)


def layer_index(name):
    match = re.fullmatch(r'model\d+\.decoder\.layers\.(\d+)', name)
    return int(match.group(1)) if match else None


def compare_micro_batch(gpu, npu):
    gpu_boundaries = gpu.get('boundaries', {})
    npu_boundaries = npu.get('boundaries', {})
    invariants = {
        'format_match': gpu.get('format') == npu.get('format'),
        'step_match': gpu.get('step') == npu.get('step'),
        'micro_batch_match': gpu.get('micro_batch') == npu.get('micro_batch'),
        'runtime_match': gpu.get('runtime') == npu.get('runtime'),
        'input_ids_match': (
            gpu.get('provenance', {}).get('input_ids')
            == npu.get('provenance', {}).get('input_ids')),
        'position_ids_match': (
            gpu.get('provenance', {}).get('position_ids')
            == npu.get('provenance', {}).get('position_ids')),
        'labels_match': (
            gpu.get('provenance', {}).get('labels')
            == npu.get('provenance', {}).get('labels')),
        'num_valid_match': (
            gpu.get('provenance', {}).get('num_valid')
            == npu.get('provenance', {}).get('num_valid')),
        'teacher_logits_match': (
            gpu.get('provenance', {}).get('teacher_logits')
            == npu.get('provenance', {}).get('teacher_logits')),
        'teacher_topk_logprobs_match': (
            gpu.get('provenance', {}).get('teacher_topk_logprobs')
            == npu.get('provenance', {}).get('teacher_topk_logprobs')),
        'teacher_topk_indices_match': (
            gpu.get('provenance', {}).get('teacher_topk_indices')
            == npu.get('provenance', {}).get('teacher_topk_indices')),
        'teacher_labels_match': (
            gpu.get('provenance', {}).get('teacher_labels')
            == npu.get('provenance', {}).get('teacher_labels')),
        'boundary_sets_match': set(gpu_boundaries) == set(npu_boundaries),
    }
    failed = [name for name, value in invariants.items() if not value]
    if failed:
        raise ValueError(
            f'Micro-batch {gpu.get("micro_batch")} invariants failed: {failed}')

    boundaries = {}
    for name in sorted(gpu_boundaries):
        gpu_boundary = gpu_boundaries[name]
        npu_boundary = npu_boundaries[name]
        boundaries[name] = {
            'module_type_match': (
                gpu_boundary.get('module_type') == npu_boundary.get('module_type')),
            'input_gradient': compare_optional(
                gpu_boundary.get('input_gradient'), npu_boundary.get('input_gradient')),
            'output_gradient': compare_optional(
                gpu_boundary.get('output_gradient'), npu_boundary.get('output_gradient')),
        }

    dlogits = compare_tensor(gpu['dlogits'], npu['dlogits'])
    layer_rows = []
    for name, values in boundaries.items():
        index = layer_index(name)
        metrics = values['output_gradient']
        if index is None or 'relative_l2' not in metrics:
            continue
        layer_rows.append({
            'layer': index,
            'name': name,
            'relative_l2': metrics['relative_l2'],
            'cosine': metrics['cosine'],
            'gpu_norm': metrics['gpu_norm'],
            'npu_norm': metrics['npu_norm'],
        })
    layer_rows.sort(key=lambda row: row['layer'], reverse=True)
    for position, row in enumerate(layer_rows):
        row['increase_from_previous_boundary'] = (
            None if position == 0
            else row['relative_l2'] - layer_rows[position - 1]['relative_l2'])
    largest_increases = sorted(
        (row for row in layer_rows if row['increase_from_previous_boundary'] is not None),
        key=lambda row: row['increase_from_previous_boundary'],
        reverse=True,
    )
    return {
        'invariants': invariants,
        'micro_batch': int(gpu['micro_batch']),
        'dlogits': dlogits,
        'boundaries': boundaries,
        'layers_backward_order': layer_rows,
        'largest_adjacent_increases': largest_increases[:10],
    }


def main():
    args = parse_args()
    if args.step < 0:
        raise ValueError('--step must be non-negative.')
    gpu_payloads = discover(args.gpu_dir, args.step)
    npu_payloads = discover(args.npu_dir, args.step)
    if set(gpu_payloads) != set(npu_payloads):
        raise ValueError(
            f'Micro-batch sets differ: GPU={sorted(gpu_payloads)}, NPU={sorted(npu_payloads)}')
    micro_batches = []
    for micro_batch in sorted(gpu_payloads):
        _, gpu = gpu_payloads[micro_batch]
        _, npu = npu_payloads[micro_batch]
        micro_batches.append(compare_micro_batch(gpu, npu))
    report = {
        'step': args.step,
        'gpu_dir': str(Path(args.gpu_dir)),
        'npu_dir': str(Path(args.npu_dir)),
        'micro_batch_sets_match': True,
        'micro_batches': micro_batches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'step': args.step,
        'micro_batches': [{
            'micro_batch': item['micro_batch'],
            'dlogits_relative_l2': item['dlogits']['relative_l2'],
            'dlogits_cosine': item['dlogits']['cosine'],
            'largest_adjacent_increases': item['largest_adjacent_increases'][:5],
        } for item in micro_batches],
    }, indent=2))


if __name__ == '__main__':
    main()
