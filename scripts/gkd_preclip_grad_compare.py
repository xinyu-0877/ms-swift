# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import math
from pathlib import Path

import torch


EPSILON = torch.finfo(torch.float64).eps


def parse_steps(value):
    try:
        steps = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError('steps must be comma-separated integers') from error
    if not steps or any(step < 0 for step in steps) or len(steps) != len(set(steps)):
        raise argparse.ArgumentTypeError('steps must be unique non-negative integers')
    return steps


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare complete GPU/NPU pre-clip gradients by named parameter.')
    parser.add_argument('--gpu-dir', required=True, help='GPU capture root directory.')
    parser.add_argument('--npu-dir', required=True, help='NPU capture root directory.')
    parser.add_argument('--steps', type=parse_steps, default=parse_steps('0,13'))
    parser.add_argument('--output', required=True, help='Output JSON report path.')
    return parser.parse_args()


def discover(root, step):
    root = Path(root)
    manifests = {}
    for path in root.glob(f'step_{step:06d}_rank_*/manifest.json'):
        payload = json.loads(path.read_text(encoding='utf-8'))
        rank = int(payload['rank'])
        if rank in manifests:
            raise ValueError(f'Duplicate rank {rank} captures under {root} for step {step}')
        manifests[rank] = (path.parent, payload)
    if not manifests:
        raise ValueError(f'No step {step} capture manifests found under {root}')
    return manifests


def parameter_map(manifest, label):
    result = {}
    for entry in manifest.get('parameters', []):
        name = entry['name']
        if name in result:
            raise ValueError(f'{label}: duplicate parameter name {name}')
        result[name] = entry
    return result


def load_part(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not torch.is_tensor(value):
        raise ValueError(f'Gradient part is not a tensor: {path}')
    return value.detach().reshape(-1)


def compare_parameter(gpu_dir, npu_dir, gpu, npu):
    if gpu['shape'] != npu['shape'] or int(gpu['numel']) != int(npu['numel']):
        raise ValueError(f"Shape/numel mismatch for {gpu['name']}")
    if gpu.get('gradient_shape') != npu.get('gradient_shape'):
        raise ValueError(f"Gradient shape mismatch for {gpu['name']}")
    if bool(gpu['gradient_present']) != bool(npu['gradient_present']):
        raise ValueError(f"Gradient presence mismatch for {gpu['name']}")
    row = {
        'name': gpu['name'],
        'shape': gpu['shape'],
        'numel': int(gpu['numel']),
        'gpu_dtype': gpu.get('gradient_dtype'),
        'npu_dtype': npu.get('gradient_dtype'),
        'gpu_source': gpu.get('gradient_source'),
        'npu_source': npu.get('gradient_source'),
        'gradient_present': bool(gpu['gradient_present']),
    }
    if not row['gradient_present']:
        row.update({'gpu_norm_sq': 0.0, 'npu_norm_sq': 0.0, 'difference_norm_sq': 0.0, 'dot': 0.0})
        return row

    gpu_parts = gpu.get('parts', [])
    npu_parts = npu.get('parts', [])
    gpu_ranges = [(part['start'], part['end']) for part in gpu_parts]
    npu_ranges = [(part['start'], part['end']) for part in npu_parts]
    if gpu_ranges != npu_ranges:
        raise ValueError(f"Gradient chunk ranges differ for {gpu['name']}")
    start = 0
    for part in gpu_parts:
        end = int(part['end'])
        if int(part['start']) != start or end <= start:
            raise ValueError(f"Invalid gradient chunk coverage for {gpu['name']}")
        start = end
    if start != int(gpu['numel']):
        raise ValueError(f"Incomplete gradient chunk coverage for {gpu['name']}")
    gpu_norm_sq = npu_norm_sq = difference_norm_sq = dot = 0.0
    finite = True
    for gpu_part, npu_part in zip(gpu_parts, npu_parts):
        gpu_value = load_part(gpu_dir / gpu_part['file']).double()
        npu_value = load_part(npu_dir / npu_part['file']).double()
        if gpu_value.shape != npu_value.shape:
            raise ValueError(f"Gradient part shape mismatch for {gpu['name']}")
        finite &= bool(torch.isfinite(gpu_value).all() and torch.isfinite(npu_value).all())
        difference = gpu_value - npu_value
        gpu_norm_sq += torch.dot(gpu_value, gpu_value).item()
        npu_norm_sq += torch.dot(npu_value, npu_value).item()
        difference_norm_sq += torch.dot(difference, difference).item()
        dot += torch.dot(gpu_value, npu_value).item()
    row.update({
        'gpu_norm_sq': gpu_norm_sq,
        'npu_norm_sq': npu_norm_sq,
        'difference_norm_sq': difference_norm_sq,
        'dot': dot,
        'finite': finite,
    })
    return row


def finish_metrics(rows):
    total_gpu_sq = sum(row['gpu_norm_sq'] for row in rows)
    total_npu_sq = sum(row['npu_norm_sq'] for row in rows)
    total_difference_sq = sum(row['difference_norm_sq'] for row in rows)
    total_dot = sum(row['dot'] for row in rows)
    for row in rows:
        gpu_norm = math.sqrt(max(row.pop('gpu_norm_sq'), 0.0))
        npu_norm = math.sqrt(max(row.pop('npu_norm_sq'), 0.0))
        difference_norm = math.sqrt(max(row.pop('difference_norm_sq'), 0.0))
        dot = row.pop('dot')
        if gpu_norm == 0.0 and npu_norm == 0.0:
            cosine = 1.0
        elif gpu_norm == 0.0 or npu_norm == 0.0:
            cosine = 0.0
        else:
            cosine = dot / max(gpu_norm * npu_norm, EPSILON)
        row.update({
            'gpu_norm': gpu_norm,
            'npu_norm': npu_norm,
            'absolute_l2': difference_norm,
            'relative_l2': difference_norm / max(npu_norm, EPSILON),
            'cosine': cosine,
            'norm_relative_difference': abs(gpu_norm - npu_norm) / max(gpu_norm, npu_norm, EPSILON),
            'gpu_norm_share': (gpu_norm * gpu_norm / total_gpu_sq) if total_gpu_sq else 0.0,
            'npu_norm_share': (npu_norm * npu_norm / total_npu_sq) if total_npu_sq else 0.0,
            'difference_share': (
                difference_norm * difference_norm / total_difference_sq if total_difference_sq else 0.0),
        })
    global_gpu_norm = math.sqrt(max(total_gpu_sq, 0.0))
    global_npu_norm = math.sqrt(max(total_npu_sq, 0.0))
    global_difference_norm = math.sqrt(max(total_difference_sq, 0.0))
    if global_gpu_norm == 0.0 and global_npu_norm == 0.0:
        global_cosine = 1.0
    elif global_gpu_norm == 0.0 or global_npu_norm == 0.0:
        global_cosine = 0.0
    else:
        global_cosine = total_dot / max(global_gpu_norm * global_npu_norm, EPSILON)
    return {
        'gpu_norm': global_gpu_norm,
        'npu_norm': global_npu_norm,
        'absolute_l2': global_difference_norm,
        'relative_l2': global_difference_norm / max(global_npu_norm, EPSILON),
        'cosine': global_cosine,
        'norm_relative_difference': abs(global_gpu_norm - global_npu_norm) / max(
            global_gpu_norm, global_npu_norm, EPSILON),
    }


def compare_step(gpu_root, npu_root, step):
    gpu_manifests = discover(gpu_root, step)
    npu_manifests = discover(npu_root, step)
    if set(gpu_manifests) != set(npu_manifests):
        raise ValueError(
            f'Step {step} rank sets differ: GPU={sorted(gpu_manifests)}, NPU={sorted(npu_manifests)}')
    rows = []
    dtype_match = source_match = all_finite = True
    metadata = []
    for rank in sorted(gpu_manifests):
        gpu_dir, gpu_manifest = gpu_manifests[rank]
        npu_dir, npu_manifest = npu_manifests[rank]
        if gpu_manifest.get('format') != 'swift_gkd_preclip_grad_v1':
            raise ValueError(f'Unsupported GPU capture format at {gpu_dir}')
        if npu_manifest.get('format') != 'swift_gkd_preclip_grad_v1':
            raise ValueError(f'Unsupported NPU capture format at {npu_dir}')
        if int(gpu_manifest['step']) != step or int(npu_manifest['step']) != step:
            raise ValueError(f'Manifest step mismatch for requested step {step}, rank {rank}')
        gpu_parameters = parameter_map(gpu_manifest, f'GPU step {step} rank {rank}')
        npu_parameters = parameter_map(npu_manifest, f'NPU step {step} rank {rank}')
        if set(gpu_parameters) != set(npu_parameters):
            missing_gpu = sorted(set(npu_parameters) - set(gpu_parameters))
            missing_npu = sorted(set(gpu_parameters) - set(npu_parameters))
            raise ValueError(
                f'Step {step} rank {rank} parameter sets differ: '
                f'GPU-missing={missing_gpu}, NPU-missing={missing_npu}')
        for name in sorted(gpu_parameters):
            row = compare_parameter(
                gpu_dir, npu_dir, gpu_parameters[name], npu_parameters[name])
            row['rank'] = rank
            dtype_match &= row['gpu_dtype'] == row['npu_dtype']
            source_match &= row['gpu_source'] == row['npu_source']
            all_finite &= row.get('finite', True)
            rows.append(row)
        metadata.append({
            'rank': rank,
            'gpu_tag': gpu_manifest.get('tag'),
            'npu_tag': npu_manifest.get('tag'),
            'gpu_clip_grad': gpu_manifest.get('clip_grad'),
            'npu_clip_grad': npu_manifest.get('clip_grad'),
            'gpu_main_grads_dtype': gpu_manifest.get('main_grads_dtype'),
            'npu_main_grads_dtype': npu_manifest.get('main_grads_dtype'),
        })
    global_metrics = finish_metrics(rows)
    rows.sort(key=lambda row: (row['rank'], row['name']))
    qualified_names = lambda ordered: [f"rank{row['rank']}.{row['name']}" for row in ordered]
    return {
        'invariants': {
            'rank_sets_match': True,
            'parameter_sets_match': True,
            'shapes_and_numel_match': True,
            'gradient_presence_match': True,
            'gradient_dtypes_match': dtype_match,
            'gradient_sources_match': source_match,
            'all_values_finite': all_finite,
        },
        'step': step,
        'metadata': metadata,
        'global': global_metrics,
        'parameters': rows,
        'rankings': {
            'by_difference_share': qualified_names(sorted(rows, key=lambda row: row['difference_share'], reverse=True)),
            'by_relative_l2': qualified_names(sorted(rows, key=lambda row: row['relative_l2'], reverse=True)),
            'by_npu_norm_share': qualified_names(sorted(rows, key=lambda row: row['npu_norm_share'], reverse=True)),
        },
    }


def main():
    args = parse_args()
    report = {
        'gpu_dir': str(Path(args.gpu_dir)),
        'npu_dir': str(Path(args.npu_dir)),
        'steps': [compare_step(args.gpu_dir, args.npu_dir, step) for step in args.steps],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'steps': [{
            'step': item['step'],
            'global': item['global'],
            'top_difference_share': item['rankings']['by_difference_share'][:10],
        } for item in report['steps']],
    }, indent=2))


if __name__ == '__main__':
    main()
