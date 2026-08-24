# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import hashlib
import json
from pathlib import Path

import torch


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def tensor_metrics(first_tensor, reference_tensor):
    first = first_tensor.detach().double().cpu().contiguous().reshape(-1)
    reference = reference_tensor.detach().double().cpu().contiguous().reshape(-1)
    if first.numel() != reference.numel():
        raise ValueError(
            f'Tensor sizes differ: first={first.numel()}, reference={reference.numel()}')
    difference = first - reference
    first_norm = torch.linalg.vector_norm(first)
    reference_norm = torch.linalg.vector_norm(reference)
    difference_norm = torch.linalg.vector_norm(difference)
    if difference.numel():
        max_abs_flat_index = int(difference.abs().argmax().item())
        remaining = max_abs_flat_index
        max_abs_coordinate = []
        for size in reversed(first_tensor.shape):
            max_abs_coordinate.append(remaining % size)
            remaining //= size
        max_abs_coordinate.reverse()
        first_at_max_abs = first[max_abs_flat_index].item()
        reference_at_max_abs = reference[max_abs_flat_index].item()
    else:
        max_abs_flat_index = None
        max_abs_coordinate = None
        first_at_max_abs = None
        reference_at_max_abs = None
    epsilon = torch.finfo(torch.float64).eps
    if first_norm.item() == 0.0 and reference_norm.item() == 0.0:
        cosine = 1.0
    else:
        cosine = torch.dot(first, reference).item() / max(
            (first_norm * reference_norm).item(), epsilon)
    return {
        'shape': list(first_tensor.shape),
        'numel': first.numel(),
        'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
        'max_abs_flat_index': max_abs_flat_index,
        'max_abs_coordinate': max_abs_coordinate,
        'first_at_max_abs': first_at_max_abs,
        'reference_at_max_abs': reference_at_max_abs,
        'mean_abs': difference.abs().mean().item() if difference.numel() else 0.0,
        'absolute_l2': difference_norm.item(),
        'relative_l2': difference_norm.item() / max(reference_norm.item(), epsilon),
        'cosine': cosine,
        'first_norm': first_norm.item(),
        'reference_norm': reference_norm.item(),
    }


def optional_metrics(first, reference):
    if first is None or reference is None:
        return {
            'first_present': first is not None,
            'reference_present': reference is not None,
        }
    return tensor_metrics(first, reference)


def tensor_map_metrics(first, reference):
    result = {}
    for name in sorted(set(first) | set(reference)):
        result[name] = optional_metrics(first.get(name), reference.get(name))
    return result


def validate_checks(checks, description):
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise ValueError(f'{description} invariants failed: {failed}')
    return checks


def tensor_summary(tensor):
    value = tensor.detach().double().cpu().contiguous().reshape(-1)
    return {
        'shape': list(tensor.shape),
        'numel': value.numel(),
        'max_abs': value.abs().max().item() if value.numel() else 0.0,
        'mean_abs': value.abs().mean().item() if value.numel() else 0.0,
        'absolute_l2': torch.linalg.vector_norm(value).item(),
    }


def merge_closure(payload):
    residual = payload['layer_output_gradient']
    local_mlp = payload['mlp_local_input_gradient']
    observed = payload['shared_input_gradient']
    reconstructed = residual.float() + local_mlp.float()
    residual_flat = residual.detach().double().reshape(-1)
    local_flat = local_mlp.detach().double().reshape(-1)
    residual_norm = torch.linalg.vector_norm(residual_flat)
    local_norm = torch.linalg.vector_norm(local_flat)
    reconstructed_norm = torch.linalg.vector_norm(reconstructed.detach().double().reshape(-1))
    epsilon = torch.finfo(torch.float64).eps
    component_cosine = torch.dot(residual_flat, local_flat).item() / max(
        (residual_norm * local_norm).item(), epsilon)
    output_gradients = payload.get('output_gradients', {})
    primary_output_path = sorted(output_gradients)[0] if output_gradients else None
    return {
        'observed_vs_residual_plus_mlp': tensor_metrics(observed, reconstructed),
        'mlp_output_vs_layer_output_gradient': (
            tensor_metrics(output_gradients[primary_output_path], residual)
            if primary_output_path is not None else None),
        'primary_mlp_output_gradient_path': primary_output_path,
        'residual_norm': residual_norm.item(),
        'mlp_local_norm': local_norm.item(),
        'merged_norm': reconstructed_norm.item(),
        'residual_mlp_cosine': component_cosine,
        'cancellation_ratio': reconstructed_norm.item() / max(
            residual_norm.item() + local_norm.item(), epsilon),
    }


def load_payload(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def validate_capture(gpu, npu):
    checks = {
        'capture_modes': gpu.get('mode') == npu.get('mode') == 'capture',
        'layer_target': gpu.get('layer_target') == npu.get('layer_target'),
        'mlp_target': gpu.get('mlp_target') == npu.get('mlp_target'),
        'step': gpu.get('step') == npu.get('step'),
        'micro_batch': gpu.get('micro_batch') == npu.get('micro_batch'),
        'local_gradient_source': (
            gpu.get('mlp_local_input_gradient_source')
            == npu.get('mlp_local_input_gradient_source')
            == 'fc1_module_full_backward_hook'),
    }
    return validate_checks(checks, 'MLP merge capture')


def validate_replay_sources(payload, x_capture, dout_capture, parameter_capture):
    return {
        'mode_replay': payload.get('mode') == 'replay',
        'input_match': payload.get('input_identity') == x_capture.get('input_identity'),
        'dout_match': {
            name: tensor_identity(value)
            for name, value in payload.get('output_gradients', {}).items()
        } == {
            name: tensor_identity(value)
            for name, value in dout_capture.get('output_gradients', {}).items()
        },
        'parameters_match': payload.get('parameter_identities')
        == parameter_capture.get('parameter_identities'),
        'local_gradient_source_match': (
            payload.get('mlp_local_input_gradient_source')
            == 'fc1_module_full_backward_hook'),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Layer-N MLP residual merge captures and optional four-way MLP replays.')
    parser.add_argument('--gpu-capture', required=True)
    parser.add_argument('--npu-capture', required=True)
    parser.add_argument(
        '--common-npu-replay',
        help='GPU replay using NPU input, output gradient, and parameters.')
    parser.add_argument('--x-npu-dy-npu')
    parser.add_argument('--x-gpu-dy-npu')
    parser.add_argument('--x-npu-dy-gpu')
    parser.add_argument('--x-gpu-dy-gpu')
    parser.add_argument('--output')
    args = parser.parse_args()

    gpu = load_payload(args.gpu_capture)
    npu = load_payload(args.npu_capture)
    result = {
        'capture_invariants': validate_capture(gpu, npu),
        'native_gpu_vs_npu': {
            'mlp_input_x': tensor_metrics(gpu['input'], npu['input']),
            'mlp_forward_outputs': tensor_map_metrics(
                gpu['forward_outputs'], npu['forward_outputs']),
            'layer_output_gradient_residual_branch': tensor_metrics(
                gpu['layer_output_gradient'], npu['layer_output_gradient']),
            'mlp_output_gradients': tensor_map_metrics(
                gpu['output_gradients'], npu['output_gradients']),
            'mlp_local_input_gradient': tensor_metrics(
                gpu['mlp_local_input_gradient'], npu['mlp_local_input_gradient']),
            'shared_input_gradient_after_merge': tensor_metrics(
                gpu['shared_input_gradient'], npu['shared_input_gradient']),
        },
        'merge_closure': {
            'gpu': merge_closure(gpu),
            'npu': merge_closure(npu),
        },
    }

    if args.common_npu_replay:
        replay = load_payload(args.common_npu_replay)
        checks = validate_replay_sources(replay, npu, npu, npu)
        validate_checks(checks, 'Common-NPU replay source')
        result['common_npu_replay'] = {
            'invariants': checks,
            'gpu_vs_npu_forward_outputs': tensor_map_metrics(
                replay['forward_outputs'], npu['forward_outputs']),
            'gpu_vs_npu_local_input_gradient': tensor_metrics(
                replay['mlp_local_input_gradient'], npu['mlp_local_input_gradient']),
        }

    replay_paths = {
        'x_npu_dy_npu': args.x_npu_dy_npu,
        'x_gpu_dy_npu': args.x_gpu_dy_npu,
        'x_npu_dy_gpu': args.x_npu_dy_gpu,
        'x_gpu_dy_gpu': args.x_gpu_dy_gpu,
    }
    provided = [name for name, path in replay_paths.items() if path]
    if provided and len(provided) != 4:
        missing = [name for name, path in replay_paths.items() if not path]
        raise ValueError(f'Provide all four replay files or none; missing={missing}')
    if provided:
        replays = {name: load_payload(path) for name, path in replay_paths.items()}
        sources = {
            'x_npu_dy_npu': (npu, npu),
            'x_gpu_dy_npu': (gpu, npu),
            'x_npu_dy_gpu': (npu, gpu),
            'x_gpu_dy_gpu': (gpu, gpu),
        }
        invariants = {}
        for name, replay in replays.items():
            x_source, dout_source = sources[name]
            invariants[name] = validate_replay_sources(replay, x_source, dout_source, npu)
        failed = {
            name: [key for key, value in checks.items() if not value]
            for name, checks in invariants.items()
            if not all(checks.values())
        }
        if failed:
            raise ValueError(f'MLP replay source invariants failed: {failed}')

        gradients = {
            name: replay['mlp_local_input_gradient'] for name, replay in replays.items()
        }
        baseline = gradients['x_npu_dy_npu']
        interaction = (
            gradients['x_gpu_dy_gpu'].float()
            - gradients['x_gpu_dy_npu'].float()
            - gradients['x_npu_dy_gpu'].float()
            + gradients['x_npu_dy_npu'].float())
        result['four_way_replay'] = {
            'invariants': invariants,
            'common_npu_x_dy_gpu_vs_npu': tensor_metrics(
                gradients['x_npu_dy_npu'], npu['mlp_local_input_gradient']),
            'gpu_native_replay_fidelity': tensor_metrics(
                gradients['x_gpu_dy_gpu'], gpu['mlp_local_input_gradient']),
            'x_effect_with_npu_dy': tensor_metrics(
                gradients['x_gpu_dy_npu'], baseline),
            'dy_effect_with_npu_x': tensor_metrics(
                gradients['x_npu_dy_gpu'], baseline),
            'combined_native_x_dy_effect': tensor_metrics(
                gradients['x_gpu_dy_gpu'], baseline),
            'interaction_term': tensor_summary(interaction),
        }

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
