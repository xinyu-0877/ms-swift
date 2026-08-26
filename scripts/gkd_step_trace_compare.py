# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import math
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare sampled GPU/NPU GKD forward/backward traces for one runtime step.')
    parser.add_argument('--gpu', required=True, help='GPU rank0_alignment.jsonl path.')
    parser.add_argument('--npu', required=True, help='NPU rank0_alignment.jsonl path.')
    parser.add_argument('--step', type=int, required=True, help='Zero-based runtime step to compare.')
    parser.add_argument('--expected-microbatches', type=int, default=4)
    parser.add_argument('--gpu-grad-clip', help='Optional GPU rank0_grad_clip.jsonl path.')
    parser.add_argument('--npu-grad-clip', help='Optional NPU rank0_grad_clip.jsonl path.')
    parser.add_argument('--output', required=True, help='JSON report path.')
    return parser.parse_args()


def load_jsonl(path, step):
    records = []
    with Path(path).open(encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: invalid JSON') from error
            if record.get('step') == step:
                records.append(record)
    if not records:
        raise ValueError(f'{path}: no records found for runtime step {step}')
    return records


def index_records(records, record_type, key_fields):
    result = {}
    for record in records:
        if record.get('record_type') != record_type:
            continue
        key = tuple(record.get(field) for field in key_fields)
        if key in result:
            raise ValueError(f'Duplicate {record_type} record for key={key}')
        result[key] = record
    return result


def relative_difference(gpu, npu):
    if gpu is None or npu is None:
        return None
    return abs(float(gpu) - float(npu)) / max(abs(float(npu)), 1e-30)


def sample_payload(summary):
    if summary is None:
        return None
    nested = summary.get('sample')
    return nested if isinstance(nested, dict) else summary


def sample_metrics(gpu_summary, npu_summary):
    gpu_sample = sample_payload(gpu_summary)
    npu_sample = sample_payload(npu_summary)
    if gpu_sample is None or npu_sample is None:
        return {
            'gpu_present': gpu_sample is not None,
            'npu_present': npu_sample is not None,
        }
    gpu_values = gpu_sample.get('values')
    npu_values = npu_sample.get('values')
    if gpu_values is None or npu_values is None:
        return {
            'sample_values_present': False,
            'gpu_sha256_fp32': gpu_sample.get('sha256_fp32'),
            'npu_sha256_fp32': npu_sample.get('sha256_fp32'),
        }
    if len(gpu_values) != len(npu_values):
        return {
            'sample_values_present': True,
            'gpu_sample_size': len(gpu_values),
            'npu_sample_size': len(npu_values),
        }
    gpu_values = [float(value) for value in gpu_values]
    npu_values = [float(value) for value in npu_values]
    difference = [gpu - npu for gpu, npu in zip(gpu_values, npu_values)]
    gpu_norm = math.sqrt(sum(value * value for value in gpu_values))
    npu_norm = math.sqrt(sum(value * value for value in npu_values))
    difference_norm = math.sqrt(sum(value * value for value in difference))
    denominator = max(gpu_norm * npu_norm, 1e-30)
    cosine = sum(gpu * npu for gpu, npu in zip(gpu_values, npu_values)) / denominator
    result = {
        'sample_values_present': True,
        'sample_size': len(gpu_values),
        'sample_relative_l2': difference_norm / max(npu_norm, 1e-30),
        'sample_cosine': cosine,
        'sample_max_abs': max((abs(value) for value in difference), default=0.0),
        'gpu_sample_norm': gpu_norm,
        'npu_sample_norm': npu_norm,
        'sample_sha256_match': gpu_sample.get('sha256_fp32') == npu_sample.get('sha256_fp32'),
    }
    if 'full_shape' in gpu_summary or 'full_shape' in npu_summary:
        result['tensor_metadata_match'] = all(
            gpu_summary.get(field) == npu_summary.get(field)
            for field in ('full_shape', 'full_dtype', 'full_numel', 'sample_indices'))
    return result


def logits_metrics(gpu, npu):
    if gpu is None or npu is None:
        return {'gpu_present': gpu is not None, 'npu_present': npu is not None}
    result = sample_metrics(gpu, npu)
    result.update({
        'location_match': all(gpu.get(field) == npu.get(field) for field in ('batch_idx', 'seq_idx', 'token_ids')),
        'full_selected_vector_norm_relative_difference': relative_difference(gpu.get('norm'), npu.get('norm')),
        'gpu_full_selected_vector_norm': gpu.get('norm'),
        'npu_full_selected_vector_norm': npu.get('norm'),
    })
    return result


def compare_forward(gpu_records, npu_records, expected_microbatches):
    gpu = index_records(gpu_records, 'forward', ('micro_batch',))
    npu = index_records(npu_records, 'forward', ('micro_batch',))
    gpu_micros = {key[0] for key in gpu}
    npu_micros = {key[0] for key in npu}
    expected = set(range(expected_microbatches))
    rows = []
    invariant_rows = []
    for micro_batch in sorted(gpu_micros | npu_micros):
        gpu_record = gpu.get((micro_batch,))
        npu_record = npu.get((micro_batch,))
        if gpu_record is None or npu_record is None:
            rows.append({
                'micro_batch': micro_batch,
                'gpu_present': gpu_record is not None,
                'npu_present': npu_record is not None,
            })
            continue
        invariants = {
            'input_ids_match': gpu_record.get('input_ids') == npu_record.get('input_ids'),
            'position_ids_match': gpu_record.get('position_ids') == npu_record.get('position_ids'),
            'labels_match': gpu_record.get('labels') == npu_record.get('labels'),
            'num_valid_match': gpu_record.get('num_valid') == npu_record.get('num_valid'),
        }
        teacher = logits_metrics(gpu_record.get('teacher_logits'), npu_record.get('teacher_logits'))
        if gpu_record.get('teacher_logits') is not None or npu_record.get('teacher_logits') is not None:
            gpu_teacher = gpu_record.get('teacher_logits') or {}
            npu_teacher = npu_record.get('teacher_logits') or {}
            invariants['teacher_logits_sample_match'] = all((
                gpu_teacher.get('batch_idx') == npu_teacher.get('batch_idx'),
                gpu_teacher.get('seq_idx') == npu_teacher.get('seq_idx'),
                gpu_teacher.get('token_ids') == npu_teacher.get('token_ids'),
                gpu_teacher.get('values') == npu_teacher.get('values'),
                gpu_teacher.get('norm') == npu_teacher.get('norm'),
            ))
        if gpu_record.get('teacher_logits') is None and npu_record.get('teacher_logits') is None:
            invariants.update({
                'teacher_topk_indices_match': (
                    gpu_record.get('teacher_topk_indices') == npu_record.get('teacher_topk_indices')),
                'teacher_topk_logprobs_sha_match': (
                    (gpu_record.get('teacher_topk_logprobs') or {}).get('sha256_fp32')
                    == (npu_record.get('teacher_topk_logprobs') or {}).get('sha256_fp32')),
            })
        invariant_rows.append(invariants)
        gpu_loss = gpu_record.get('loss') or {}
        npu_loss = npu_record.get('loss') or {}
        gpu_jsd = gpu_loss.get('jsd_loss')
        npu_jsd = npu_loss.get('jsd_loss')
        rows.append({
            'micro_batch': micro_batch,
            'invariants': invariants,
            'gpu_num_valid': gpu_record.get('num_valid'),
            'npu_num_valid': npu_record.get('num_valid'),
            'gpu_jsd_loss': gpu_jsd,
            'npu_jsd_loss': npu_jsd,
            'jsd_loss_absolute_difference': (
                abs(float(gpu_jsd) - float(npu_jsd)) if gpu_jsd is not None and npu_jsd is not None else None),
            'jsd_loss_relative_difference': relative_difference(gpu_jsd, npu_jsd),
            'student_logits': logits_metrics(
                gpu_record.get('student_logits'), npu_record.get('student_logits')),
            'teacher_logits': teacher,
        })
    return {
        'invariants': {
            'gpu_microbatches': sorted(gpu_micros),
            'npu_microbatches': sorted(npu_micros),
            'expected_microbatches': sorted(expected),
            'microbatch_sets_match': gpu_micros == npu_micros,
            'expected_microbatches_present': gpu_micros == npu_micros == expected,
            'all_input_invariants_match': bool(invariant_rows) and all(
                all(values.values()) for values in invariant_rows),
        },
        'microbatches': rows,
    }


def compare_tensor_records(gpu_records, npu_records, record_type, key_fields, tensor_fields):
    gpu = index_records(gpu_records, record_type, key_fields)
    npu = index_records(npu_records, record_type, key_fields)
    rows = []
    for key in sorted(set(gpu) | set(npu), key=str):
        gpu_record = gpu.get(key)
        npu_record = npu.get(key)
        row = {field: value for field, value in zip(key_fields, key)}
        if gpu_record is None or npu_record is None:
            row.update({'gpu_present': gpu_record is not None, 'npu_present': npu_record is not None})
        else:
            row['module_type_match'] = gpu_record.get('module_type') == npu_record.get('module_type')
            for field in tensor_fields:
                row[field] = sample_metrics(gpu_record.get(field), npu_record.get(field))
        rows.append(row)
    return {
        'record_sets_match': set(gpu) == set(npu),
        'gpu_record_count': len(gpu),
        'npu_record_count': len(npu),
        'records': rows,
    }


def compare_logits_backward(gpu_records, npu_records):
    gpu = index_records(gpu_records, 'student_logits_backward', ('micro_batch',))
    npu = index_records(npu_records, 'student_logits_backward', ('micro_batch',))
    rows = []
    for key in sorted(set(gpu) | set(npu)):
        gpu_record = gpu.get(key)
        npu_record = npu.get(key)
        rows.append({
            'micro_batch': key[0],
            'gradient': sample_metrics(
                gpu_record.get('gradient') if gpu_record else None,
                npu_record.get('gradient') if npu_record else None),
        })
    return {'record_sets_match': set(gpu) == set(npu), 'microbatches': rows}


def parameter_map(record):
    state = (record or {}).get('trainable_state') or {}
    return {parameter['name']: parameter for parameter in state.get('parameters', [])}


def compare_optimizer(gpu_records, npu_records):
    gpu_map = index_records(gpu_records, 'optimizer_step', ('step',))
    npu_map = index_records(npu_records, 'optimizer_step', ('step',))
    gpu = next(iter(gpu_map.values()), None)
    npu = next(iter(npu_map.values()), None)
    if gpu is None or npu is None:
        return {'gpu_present': gpu is not None, 'npu_present': npu is not None}
    gpu_parameters = parameter_map(gpu)
    npu_parameters = parameter_map(npu)
    parameters = []
    for name in sorted(set(gpu_parameters) | set(npu_parameters)):
        gpu_parameter = gpu_parameters.get(name)
        npu_parameter = npu_parameters.get(name)
        row = {'name': name}
        if gpu_parameter is None or npu_parameter is None:
            row.update({
                'gpu_present': gpu_parameter is not None,
                'npu_present': npu_parameter is not None,
            })
        else:
            row.update({
                'sample_indices_match': gpu_parameter.get('indices') == npu_parameter.get('indices'),
                'gradient': sample_metrics(
                    gpu_parameter.get('gradient'), npu_parameter.get('gradient')),
                'parameter_delta': sample_metrics(
                    gpu_parameter.get('delta'), npu_parameter.get('delta')),
            })
        parameters.append(row)
    return {
        'learning_rate_match': gpu.get('learning_rate') == npu.get('learning_rate'),
        'gpu_learning_rate': gpu.get('learning_rate'),
        'npu_learning_rate': npu.get('learning_rate'),
        'update_successful_match': gpu.get('update_successful') == npu.get('update_successful'),
        'gpu_grad_norm': gpu.get('grad_norm'),
        'npu_grad_norm': npu.get('grad_norm'),
        'grad_norm_relative_difference': relative_difference(gpu.get('grad_norm'), npu.get('grad_norm')),
        'parameter_sets_match': set(gpu_parameters) == set(npu_parameters),
        'parameters': parameters,
    }


def load_grad_clip(path, step):
    records = [
        record for record in load_jsonl(path, step)
        if record.get('record_type') == 'grad_clip_step'
    ]
    if len(records) != 1:
        raise ValueError(f'{path}: expected one grad_clip_step for step {step}, found {len(records)}')
    return records[0]


def compare_grad_clip(gpu_path, npu_path, step):
    if gpu_path is None and npu_path is None:
        return None
    if gpu_path is None or npu_path is None:
        raise ValueError('--gpu-grad-clip and --npu-grad-clip must be provided together')
    gpu = load_grad_clip(gpu_path, step)
    npu = load_grad_clip(npu_path, step)
    return {
        'clip_grad_match': gpu.get('clip_grad') == npu.get('clip_grad'),
        'gpu_grad_norm': gpu.get('optimizer_reported_grad_norm'),
        'npu_grad_norm': npu.get('optimizer_reported_grad_norm'),
        'grad_norm_relative_difference': relative_difference(
            gpu.get('optimizer_reported_grad_norm'), npu.get('optimizer_reported_grad_norm')),
        'gpu_clipped': gpu.get('clipped'),
        'npu_clipped': npu.get('clipped'),
        'clipping_decision_match': gpu.get('clipped') == npu.get('clipped'),
        'gpu_clip_coefficient': gpu.get('clip_coefficient'),
        'npu_clip_coefficient': npu.get('clip_coefficient'),
        'update_successful_match': gpu.get('update_successful') == npu.get('update_successful'),
    }


def layer_index(name):
    match = re.search(r'decoder\.layers\.(\d+)(?:\.|$)', name or '')
    return int(match.group(1)) if match else None


def ranked_records(section, tensor_field):
    ranked = []
    for row in section['records']:
        metrics = row.get(tensor_field) or {}
        value = metrics.get('sample_relative_l2')
        if value is None:
            continue
        ranked.append({
            'name': row.get('name'),
            'micro_batch': row.get('micro_batch'),
            'layer': layer_index(row.get('name')),
            'sample_relative_l2': value,
            'sample_percent': value * 100.0,
            'sample_cosine': metrics.get('sample_cosine'),
        })
    return sorted(ranked, key=lambda row: row['sample_relative_l2'], reverse=True)


def main():
    args = parse_args()
    if args.step < 0:
        raise ValueError('--step must be non-negative')
    if args.expected_microbatches <= 0:
        raise ValueError('--expected-microbatches must be positive')

    gpu_records = load_jsonl(args.gpu, args.step)
    npu_records = load_jsonl(args.npu, args.step)
    forward = compare_forward(gpu_records, npu_records, args.expected_microbatches)
    operator_forward = compare_tensor_records(
        gpu_records,
        npu_records,
        'operator_forward',
        ('micro_batch', 'name', 'call_index'),
        ('input', 'output'),
    )
    module_backward = compare_tensor_records(
        gpu_records,
        npu_records,
        'module_backward',
        ('micro_batch', 'name'),
        ('input_gradient', 'output_gradient'),
    )
    report = {
        'scope': {
            'runtime_step': args.step,
            'expected_microbatches': args.expected_microbatches,
            'metrics_are_sampled': True,
            'relative_l2_denominator': 'NPU sample L2 norm',
            'module_backward_microbatch_attribution_reliable': args.expected_microbatches == 1,
        },
        'forward': forward,
        'student_logits_backward': compare_logits_backward(gpu_records, npu_records),
        'operator_forward': operator_forward,
        'module_backward': module_backward,
        'optimizer_step': compare_optimizer(gpu_records, npu_records),
        'grad_clip': compare_grad_clip(args.gpu_grad_clip, args.npu_grad_clip, args.step),
    }
    report['rankings'] = {
        'forward_output_sample_relative_l2': ranked_records(operator_forward, 'output'),
        'backward_output_gradient_sample_relative_l2': ranked_records(
            module_backward, 'output_gradient'),
        'backward_input_gradient_sample_relative_l2': ranked_records(
            module_backward, 'input_gradient'),
    }
    invariants = forward['invariants']
    report['accepted_for_sampled_comparison'] = all((
        invariants['microbatch_sets_match'],
        invariants['expected_microbatches_present'],
        invariants['all_input_invariants_match'],
        operator_forward['record_sets_match'],
        module_backward['record_sets_match'],
    ))
    report['interpretation'] = (
        'All tensor comparisons in this report use deterministic sampled values, not full tensors. '
        'Use the result to select a layer or micro-batch for a full-tensor isolation experiment; '
        'do not diagnose an operator from sampled relative_l2 alone. With multiple micro-batches, '
        'module_backward micro_batch labels come from shared trainer context and are not reliable; '
        'student_logits_backward labels are bound to each forward closure.')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({
        'runtime_step': args.step,
        'accepted_for_sampled_comparison': report['accepted_for_sampled_comparison'],
        'forward_invariants': invariants,
        'optimizer_grad_norm_relative_difference': report['optimizer_step'].get(
            'grad_norm_relative_difference'),
        'grad_clip': report['grad_clip'],
    }, ensure_ascii=False, indent=2))
    for title, ranking in report['rankings'].items():
        print(f'\n{title}:')
        for row in ranking[:10]:
            print(
                f"  micro={row['micro_batch']} name={row['name']} "
                f"relative_l2={row['sample_relative_l2']:.6g} "
                f"percent={row['sample_percent']:.6g}% cosine={row['sample_cosine']}")
    print(f'\nFull report: {output}')


if __name__ == '__main__':
    main()
