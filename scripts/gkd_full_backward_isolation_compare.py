# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
from pathlib import Path

import torch


def _compare_tensor(gpu_tensor, npu_tensor):
    gpu = gpu_tensor.detach().double().cpu().contiguous().reshape(-1)
    npu = npu_tensor.detach().double().cpu().contiguous().reshape(-1)
    if gpu.numel() != npu.numel():
        raise ValueError(f'Tensor sizes differ: GPU={gpu.numel()}, NPU={npu.numel()}')
    difference = gpu - npu
    gpu_norm = torch.linalg.vector_norm(gpu)
    npu_norm = torch.linalg.vector_norm(npu)
    difference_norm = torch.linalg.vector_norm(difference)
    epsilon = torch.finfo(torch.float64).eps
    return {
        'gpu_shape': list(gpu_tensor.shape),
        'npu_shape': list(npu_tensor.shape),
        'numel': gpu.numel(),
        'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
        'mean_abs': difference.abs().mean().item() if difference.numel() else 0.0,
        'relative_l2': difference_norm.item() / max(npu_norm.item(), epsilon),
        'cosine': torch.dot(gpu, npu).item() / max((gpu_norm * npu_norm).item(), epsilon),
        'different_ratio': (gpu != npu).double().mean().item() if difference.numel() else 0.0,
        'gpu_norm': gpu_norm.item(),
        'npu_norm': npu_norm.item(),
    }


def _compare_optional(gpu_tensor, npu_tensor):
    if gpu_tensor is None or npu_tensor is None:
        return {
            'gpu_present': gpu_tensor is not None,
            'npu_present': npu_tensor is not None,
        }
    return _compare_tensor(gpu_tensor, npu_tensor)


def _module(payload, name):
    key = f'model0.{name}'
    if key not in payload:
        raise ValueError(f'Required backward boundary is missing: {key}.')
    return payload[key]


def _required_gradient(module, field, name):
    gradient = module.get(field)
    if gradient is None:
        raise ValueError(f'Required {field} is missing at {name}.')
    return gradient


def _format_percent(value):
    return f'{value * 100.0:.6f}%'


def _probe_identity(provenance):
    probes = provenance.get('student_parameter_probes', {}).get('probes', [])
    return {
        probe['name']: (
            probe['full_shape'], probe['full_dtype'], probe['full_numel'],
            probe['sample_indices'], probe['sample']['sha256_fp32'],
        )
        for probe in probes
    }


def _provenance_invariants(gpu, npu):
    if gpu is None or npu is None:
        return {'provenance_present': False}
    gpu_checkpoint = gpu.get('checkpoint', {})
    npu_checkpoint = npu.get('checkpoint', {})
    checkpoint_semantic_keys = ('finetune', 'no_load_optim', 'no_load_rng', 'seed', 'data_seed')
    return {
        'provenance_present': True,
        'input_ids_match': gpu.get('input_ids') == npu.get('input_ids'),
        'position_ids_match': gpu.get('position_ids') == npu.get('position_ids'),
        'labels_match': gpu.get('labels') == npu.get('labels'),
        'num_valid_match': gpu.get('num_valid') == npu.get('num_valid'),
        'teacher_logits_match': gpu.get('teacher_logits') == npu.get('teacher_logits'),
        'teacher_topk_logprobs_match': (
            gpu.get('teacher_topk_logprobs') == npu.get('teacher_topk_logprobs')),
        'teacher_topk_indices_match': (
            gpu.get('teacher_topk_indices') == npu.get('teacher_topk_indices')),
        'teacher_labels_match': gpu.get('teacher_labels') == npu.get('teacher_labels'),
        'runtime_match': gpu.get('runtime') == npu.get('runtime'),
        'checkpoint_semantics_match': all(
            gpu_checkpoint.get(key) == npu_checkpoint.get(key)
            for key in checkpoint_semantic_keys),
        'student_parameter_probes_match': (
            bool(_probe_identity(gpu)) and _probe_identity(gpu) == _probe_identity(npu)),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare GPU/NPU full backward results with common dLogits.')
    parser.add_argument('--gpu', required=True, help='GPU full_backward_*.pt path.')
    parser.add_argument('--npu', required=True, help='NPU full_backward_*.pt path.')
    parser.add_argument('--output', help='Optional JSON output path.')
    parser.add_argument('--all-layers', action='store_true',
                        help='Require output layer, final norm, and decoder layers 27..0; print localization summary.')
    args = parser.parse_args()

    gpu_payload = torch.load(args.gpu, map_location='cpu', weights_only=True)
    npu_payload = torch.load(args.npu, map_location='cpu', weights_only=True)
    gpu_modules = gpu_payload.get('module_gradients', {})
    npu_modules = npu_payload.get('module_gradients', {})
    module_names = sorted(set(gpu_modules) | set(npu_modules))
    modules = {}
    for name in module_names:
        gpu_module = gpu_modules.get(name)
        npu_module = npu_modules.get(name)
        if gpu_module is None or npu_module is None:
            modules[name] = {
                'gpu_present': gpu_module is not None,
                'npu_present': npu_module is not None,
            }
            continue
        modules[name] = {
            'module_type_match': gpu_module.get('module_type') == npu_module.get('module_type'),
            'input_gradient': _compare_optional(
                gpu_module.get('input_gradient'), npu_module.get('input_gradient')),
            'output_gradient': _compare_optional(
                gpu_module.get('output_gradient'), npu_module.get('output_gradient')),
        }

    gpu_parameters = gpu_payload.get('parameter_gradient_samples', {})
    npu_parameters = npu_payload.get('parameter_gradient_samples', {})
    parameter_names = sorted(set(gpu_parameters) | set(npu_parameters))
    parameters = {}
    for name in parameter_names:
        gpu_parameter = gpu_parameters.get(name)
        npu_parameter = npu_parameters.get(name)
        if gpu_parameter is None or npu_parameter is None:
            parameters[name] = {
                'gpu_present': gpu_parameter is not None,
                'npu_present': npu_parameter is not None,
            }
            continue
        indices_match = gpu_parameter.get('sample_indices') == npu_parameter.get('sample_indices')
        parameters[name] = {
            'sample_indices_match': indices_match,
            'sample': _compare_tensor(gpu_parameter['sample'], npu_parameter['sample']),
        }

    invariants = {
        'gpu_mode_replay': gpu_payload.get('mode') == 'replay',
        'npu_mode_capture': npu_payload.get('mode') == 'capture',
        'step_match': gpu_payload.get('step') == npu_payload.get('step'),
        'micro_batch_match': gpu_payload.get('micro_batch') == npu_payload.get('micro_batch'),
        'common_dlogits_match': gpu_payload.get('common_dlogits') == npu_payload.get('common_dlogits'),
        'backward_patterns_match': gpu_payload.get('backward_patterns') == npu_payload.get('backward_patterns'),
        'module_sets_match': set(gpu_modules) == set(npu_modules),
    }
    invariants.update(_provenance_invariants(
        gpu_payload.get('provenance'), npu_payload.get('provenance')))
    if args.all_layers:
        expected_names = {
            'model0.output_layer', 'model0.decoder.final_layernorm',
            *(f'model0.decoder.layers.{index}' for index in range(28)),
        }
        gpu_module_names = set(gpu_modules)
        npu_module_names = set(npu_modules)
        missing_gpu = sorted(expected_names - gpu_module_names)
        missing_npu = sorted(expected_names - npu_module_names)
        extra_gpu = sorted(gpu_module_names - expected_names)
        extra_npu = sorted(npu_module_names - expected_names)
        # Other enabled backward-debug patterns may legitimately add captures.
        # Only the 30 boundaries used by --all-layers are mandatory here.
        invariants['all_layer_boundaries_present'] = not missing_gpu and not missing_npu
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        detail = ''
        if args.all_layers and not invariants.get('all_layer_boundaries_present', True):
            detail = f' missing_gpu={missing_gpu}, missing_npu={missing_npu}.'
        raise ValueError(f'Common-dLogits backward invariants failed: {failed}.{detail}')

    result = {
        'invariants': invariants,
        'common_dlogits_match': invariants['common_dlogits_match'],
        'gpu_mode': gpu_payload.get('mode'),
        'npu_mode': npu_payload.get('mode'),
        'modules': modules,
        'parameter_gradient_samples': parameters,
    }
    if args.all_layers:
        result['all_layer_boundary_inventory'] = {
            'expected_count': len(expected_names),
            'gpu_count': len(gpu_modules),
            'npu_count': len(npu_modules),
            'missing_gpu': missing_gpu,
            'missing_npu': missing_npu,
            'extra_gpu': extra_gpu,
            'extra_npu': extra_npu,
        }
        chain = [{
            'label': 'common_dlogits',
            'module': None,
            'gradient_field': 'common_dlogits',
            'crossed_module': None,
            'relative_l2': 0.0,
            'cosine': 1.0,
        }]
        tail_specs = (
            ('output_layer_input', 'output_layer', 'input_gradient', 'output_layer'),
            ('final_layernorm_input', 'decoder.final_layernorm', 'input_gradient',
             'decoder.final_layernorm'),
        )
        for label, module_name, field, crossed_module in tail_specs:
            gpu_gradient = _required_gradient(
                _module(gpu_modules, module_name), field, module_name)
            npu_gradient = _required_gradient(
                _module(npu_modules, module_name), field, module_name)
            chain.append({
                'label': label,
                'module': module_name,
                'gradient_field': field,
                'crossed_module': crossed_module,
                **_compare_tensor(gpu_gradient, npu_gradient),
            })
        for index in range(27, -1, -1):
            module_name = f'decoder.layers.{index}'
            gpu_gradient = _required_gradient(
                _module(gpu_modules, module_name), 'output_gradient', module_name)
            npu_gradient = _required_gradient(
                _module(npu_modules, module_name), 'output_gradient', module_name)
            chain.append({
                'label': f'layer_{index:02d}_output_gradient',
                'module': module_name,
                'gradient_field': 'output_gradient',
                # Layer 27 output and final-layernorm input are the same logical
                # boundary. Each subsequent row is reached by crossing layer N+1.
                'crossed_module': None if index == 27 else f'decoder.layers.{index + 1}',
                **_compare_tensor(gpu_gradient, npu_gradient),
            })
        for index, item in enumerate(chain):
            previous = chain[index - 1] if index else None
            item['increase'] = 0.0 if previous is None else item['relative_l2'] - previous['relative_l2']
            item['increase_percentage_points'] = item['increase'] * 100.0
        transitions = [
            {
                'from': chain[index - 1]['label'],
                'to': chain[index]['label'],
                'crossed_module': chain[index]['crossed_module'],
                'increase': chain[index]['increase'],
                'increase_percentage_points': chain[index]['increase_percentage_points'],
            }
            for index in range(1, len(chain))
            if chain[index]['crossed_module'] is not None
        ]
        result['backward_chain'] = chain
        result['largest_adjacent_increases'] = sorted(
            transitions, key=lambda item: item['increase'], reverse=True)[:10]
        result['interpretation'] = (
            'This common-dLogits chain localizes real-path gradient growth while each backend '
            'retains its native forward activations. It does not independently diagnose a '
            'backward operator; common activation, parameters, and dout are required for that. '
            'Layer 0 input gradient is unavailable from the TransformerLayer full-backward hook '
            'because hidden states are passed by keyword, so the chain localizes decoder layers '
            '27 through 1 but does not measure traversal through Layer 0.')
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.all_layers:
        saved_path = args.output or 'not saved (--output omitted)'
        print(f'Full report: {saved_path}')
        print(f'Invariants: all passed ({len(invariants)})')
        inventory = result['all_layer_boundary_inventory']
        print(
            f'Module inventory: expected={inventory["expected_count"]}, '
            f'gpu={inventory["gpu_count"]}, npu={inventory["npu_count"]}')
        if inventory['extra_gpu']:
            print(f'Ignored extra matched modules: {inventory["extra_gpu"]}')
        print('\nCommon-dLogits backward chain:')
        print('boundary                         rel_l2     delta_pp          cos  crossed_module')
        for item in result['backward_chain']:
            print(
                f'{item["label"]:<32} {_format_percent(item["relative_l2"]):>11} '
                f'{item["increase_percentage_points"]:+12.6f} {item["cosine"]:.9f}  '
                f'{item["crossed_module"] or "boundary alias"}')
        print('\nTop adjacent increases (localization only):')
        for rank, item in enumerate(result['largest_adjacent_increases'][:5], start=1):
            print(
                f'{rank}. {item["from"]} -> {item["to"]}: '
                f'{item["increase_percentage_points"]:+.6f} pp '
                f'(crossed {item["crossed_module"]})')
        print('\nNote: native forward activations differ; these rows do not prove an operator fault.')
    else:
        print(output_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
