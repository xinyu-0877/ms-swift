#!/usr/bin/env python3
"""Compare realtime GPU/NPU teacher statistics in GKD alignment JSONL logs."""

import argparse
import json
from pathlib import Path


def load_jsonl(path):
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc
        if value.get('record_type') == 'forward':
            records.append(value)
    return records


def rel_diff(a, b):
    if a is None or b is None:
        return None
    denominator = max(abs(float(a)), abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / denominator


def identity(record, name):
    value = record.get(name) or {}
    return value.get('sha256') or value.get('sha256_fp32')


def compare(args):
    gpu = {(r.get('step'), r.get('micro_batch')): r for r in load_jsonl(args.gpu)}
    npu = {(r.get('step'), r.get('micro_batch')): r for r in load_jsonl(args.npu)}
    common = sorted(set(gpu) & set(npu), key=lambda x: (x[0] is None, x[0], x[1]))
    only_gpu, only_npu = sorted(set(gpu) - set(npu)), sorted(set(npu) - set(gpu))
    rows = []
    metrics = ('norm', 'mean', 'std', 'rms', 'min', 'max',
               'target_logprob_mean', 'target_logprob_std', 'top1_target_agreement')
    for key in common:
        g, n = gpu[key], npu[key]
        gt, nt = g.get('teacher_logits') or {}, n.get('teacher_logits') or {}
        input_match = all(identity(g, field) == identity(n, field)
                          for field in ('input_ids', 'position_ids', 'labels'))
        row = {
            'step': key[0], 'micro_batch': key[1],
            'input_match': input_match,
            'num_valid_gpu': g.get('num_valid'), 'num_valid_npu': n.get('num_valid'),
            'teacher_metrics': {name: rel_diff(gt.get(name), nt.get(name)) for name in metrics},
        }
        row['max_metric_relative_diff'] = max(
            (v for v in row['teacher_metrics'].values() if v is not None), default=None)
        rows.append(row)

    def mean_metric(name):
        values = [r['teacher_metrics'][name] for r in rows if r['teacher_metrics'][name] is not None]
        return sum(values) / len(values) if values else None

    summary = {
        'gpu': str(args.gpu), 'npu': str(args.npu),
        'gpu_forward_records': len(gpu), 'npu_forward_records': len(npu),
        'common_records': len(common), 'only_gpu': only_gpu, 'only_npu': only_npu,
        'input_mismatch_count': sum(not r['input_match'] for r in rows),
        'num_valid_mismatch_count': sum(r['num_valid_gpu'] != r['num_valid_npu'] for r in rows),
        'mean_metric_relative_diff': {name: mean_metric(name) for name in metrics},
        'max_metric_relative_diff': max((r['max_metric_relative_diff'] or 0 for r in rows), default=0),
        'rows': rows,
    }
    print(f'GPU/NPU forward records: {len(gpu)} / {len(npu)}; common: {len(common)}')
    print(f'Input mismatches: {summary["input_mismatch_count"]}; num_valid mismatches: '
          f'{summary["num_valid_mismatch_count"]}')
    print('\nTeacher metric relative differences (|GPU-NPU|/max(|GPU|,|NPU|))')
    for name, value in summary['mean_metric_relative_diff'].items():
        print(f'  {name:26s}: {value:.6%}' if value is not None else f'  {name:26s}: n/a')
    print(f'  {"MAX across metrics":26s}: {summary["max_metric_relative_diff"]:.6%}')
    print('\nLargest steps:')
    for row in sorted(rows, key=lambda r: r['max_metric_relative_diff'] or -1, reverse=True)[:args.top]:
        max_diff = row['max_metric_relative_diff']
        print(f'  step={row["step"]} micro={row["micro_batch"]} '
              f'max={max_diff:.6%} input_match={row["input_match"]}'
              if max_diff is not None else
              f'  step={row["step"]} micro={row["micro_batch"]} max=n/a '
              f'input_match={row["input_match"]}')
    if only_gpu or only_npu:
        print(f'\nUnpaired keys: GPU-only={only_gpu[:10]} NPU-only={only_npu[:10]}')
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\nReport written to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, type=Path, help='GPU alignment JSONL')
    parser.add_argument('--npu', required=True, type=Path, help='NPU alignment JSONL')
    parser.add_argument('--output', type=Path, help='Optional JSON report path')
    parser.add_argument('--top', type=int, default=10, help='Number of largest steps to print')
    compare(parser.parse_args())


if __name__ == '__main__':
    main()
