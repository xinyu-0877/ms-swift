# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Compare GPU/NPU Megatron gradient clipping decisions.')
    parser.add_argument('--gpu', required=True, help='GPU rank0_grad_clip.jsonl')
    parser.add_argument('--npu', required=True, help='NPU rank0_grad_clip.jsonl')
    parser.add_argument('--output', help='Optional JSON report path')
    parser.add_argument(
        '--near-threshold-ratio', type=float, default=0.02,
        help='Mark a norm as near the threshold within this relative distance (default: 0.02).')
    return parser.parse_args()


def load_records(path):
    records = {}
    with Path(path).open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('record_type') != 'grad_clip_step':
                continue
            step = int(record['step'])
            if step in records:
                raise ValueError(f'{path}:{line_number}: duplicate grad_clip_step {step}')
            records[step] = record
    if not records:
        raise ValueError(f'No grad_clip_step records found in {path}')
    return records


def relative_difference(left, right):
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def classify(gpu, npu):
    gpu_clipped = bool(gpu['clipped'])
    npu_clipped = bool(npu['clipped'])
    if gpu_clipped and not npu_clipped:
        return 'gpu_only'
    if npu_clipped and not gpu_clipped:
        return 'npu_only'
    if gpu_clipped:
        return 'both'
    return 'neither'


def main():
    args = parse_args()
    if args.near_threshold_ratio < 0:
        raise ValueError('--near-threshold-ratio must be non-negative')
    gpu_records = load_records(args.gpu)
    npu_records = load_records(args.npu)
    gpu_steps = set(gpu_records)
    npu_steps = set(npu_records)
    common_steps = sorted(gpu_steps & npu_steps)
    if not common_steps:
        raise ValueError('GPU and NPU files have no common runtime steps')

    rows = []
    threshold_match = True
    all_finite = True
    for step in common_steps:
        gpu = gpu_records[step]
        npu = npu_records[step]
        gpu_norm = gpu['optimizer_reported_grad_norm']
        npu_norm = npu['optimizer_reported_grad_norm']
        gpu_threshold = float(gpu['clip_grad'])
        npu_threshold = float(npu['clip_grad'])
        finite = (
            gpu_norm is not None and npu_norm is not None
            and math.isfinite(float(gpu_norm)) and math.isfinite(float(npu_norm)))
        all_finite &= finite
        same_threshold = math.isclose(gpu_threshold, npu_threshold, rel_tol=0.0, abs_tol=0.0)
        threshold_match &= same_threshold
        threshold = gpu_threshold if same_threshold else None
        near_threshold = False
        if finite and threshold is not None and threshold > 0:
            near_threshold = min(
                abs(float(gpu_norm) - threshold), abs(float(npu_norm) - threshold),
            ) / threshold <= args.near_threshold_ratio
        row = {
            'step': step,
            'gpu_norm': gpu_norm,
            'npu_norm': npu_norm,
            'norm_relative_difference': (
                relative_difference(float(gpu_norm), float(npu_norm)) if finite else None),
            'gpu_clip_grad': gpu_threshold,
            'npu_clip_grad': npu_threshold,
            'gpu_clipped': bool(gpu['clipped']),
            'npu_clipped': bool(npu['clipped']),
            'gpu_clip_coefficient': gpu.get('clip_coefficient'),
            'npu_clip_coefficient': npu.get('clip_coefficient'),
            'classification': classify(gpu, npu),
            'asymmetric_trigger': bool(gpu['clipped']) != bool(npu['clipped']),
            'near_threshold': near_threshold,
            'update_successful_match': gpu['update_successful'] == npu['update_successful'],
        }
        rows.append(row)

    asymmetric_steps = [row['step'] for row in rows if row['asymmetric_trigger']]
    report = {
        'invariants': {
            'common_steps_present': bool(common_steps),
            'step_sets_match': gpu_steps == npu_steps,
            'clip_thresholds_match': threshold_match,
            'all_norms_finite': all_finite,
        },
        'gpu_only_steps': sorted(gpu_steps - npu_steps),
        'npu_only_steps': sorted(npu_steps - gpu_steps),
        'asymmetric_trigger_steps': asymmetric_steps,
        'asymmetric_trigger_found': bool(asymmetric_steps),
        'steps': rows,
        'interpretation': (
            'clipped is derived from optimizer.step returned grad_norm. Confirm the installed Megatron version '
            'returns the pre-clip total norm before treating this as definitive.'),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + '\n', encoding='utf-8')

    print(json.dumps(report['invariants'], ensure_ascii=False, indent=2))
    print(f'common steps: {common_steps[0]}..{common_steps[-1]} ({len(common_steps)})')
    print(f'asymmetric trigger steps: {asymmetric_steps or "none"}')
    for row in rows:
        if row['asymmetric_trigger'] or row['near_threshold']:
            print(
                f"step={row['step']} gpu_norm={row['gpu_norm']} npu_norm={row['npu_norm']} "
                f"classification={row['classification']} gpu_coeff={row['gpu_clip_coefficient']} "
                f"npu_coeff={row['npu_clip_coefficient']}")


if __name__ == '__main__':
    main()
