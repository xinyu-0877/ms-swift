#!/usr/bin/env python3
"""Print a compact Layer 6 backward localization report."""
import argparse
import json
from pathlib import Path
import torch


def metric(a, b):
    if a is None or b is None:
        return {'present': a is not None and b is not None}
    x = a.detach().double().cpu().reshape(-1)
    y = b.detach().double().cpu().reshape(-1)
    if x.numel() != y.numel():
        return {'present': True, 'shape_match': False, 'gpu_numel': x.numel(), 'npu_numel': y.numel()}
    d = x - y
    xn, yn = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    eps = torch.finfo(torch.float64).eps
    return {
        'present': True, 'shape_match': True,
        'relative_l2': (torch.linalg.vector_norm(d) / max(yn, eps)).item(),
        'cosine': (torch.dot(x, y) / max(xn * yn, eps)).item(),
        'max_abs': d.abs().max().item() if d.numel() else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--npu', required=True)
    p.add_argument('--output')
    a = p.parse_args()
    gpu = torch.load(a.gpu, map_location='cpu', weights_only=True)
    npu = torch.load(a.npu, map_location='cpu', weights_only=True)
    gm, nm = gpu.get('module_gradients', {}), npu.get('module_gradients', {})
    names = [
        'output_layer', 'decoder.final_layernorm', 'decoder.layers.6',
        'decoder.layers.6.mlp', 'decoder.layers.6.mlp.linear_fc2',
        'decoder.layers.6.mlp.linear_fc1', 'decoder.layers.6.self_attention',
        'decoder.layers.6.self_attention.linear_proj',
        'decoder.layers.6.self_attention.core_attention',
        'decoder.layers.6.self_attention.linear_qkv',
    ]
    rows = []
    for name in names:
        key = f'model0.{name}'
        g, n = gm.get(key, {}), nm.get(key, {})
        for field in ('input_gradient', 'output_gradient'):
            rows.append({'boundary': f'{name}.{field}', **metric(g.get(field), n.get(field))})
    row_by_boundary = {row['boundary']: row for row in rows}
    display = [
        ('layer_output_dout', 'decoder.layers.6.output_gradient'),
        ('mlp_output_dout', 'decoder.layers.6.mlp.output_gradient'),
        ('fc2_output_dout', 'decoder.layers.6.mlp.linear_fc2.output_gradient'),
        ('fc2_input_dx', 'decoder.layers.6.mlp.linear_fc2.input_gradient'),
        ('fc1_output_dout', 'decoder.layers.6.mlp.linear_fc1.output_gradient'),
        ('mlp_local_input_dx', 'decoder.layers.6.mlp.linear_fc1.input_gradient'),
        ('self_attention_output_dout', 'decoder.layers.6.self_attention.output_gradient'),
        ('linear_proj_output_dout', 'decoder.layers.6.self_attention.linear_proj.output_gradient'),
        ('linear_proj_input_dx', 'decoder.layers.6.self_attention.linear_proj.input_gradient'),
        ('core_attention_output_dout', 'decoder.layers.6.self_attention.core_attention.output_gradient'),
        ('core_attention_first_input_grad', 'decoder.layers.6.self_attention.core_attention.input_gradient'),
        ('linear_qkv_output_dout', 'decoder.layers.6.self_attention.linear_qkv.output_gradient'),
        ('linear_qkv_local_input_dx', 'decoder.layers.6.self_attention.linear_qkv.input_gradient'),
    ]
    result = {
        'invariants': {
            'gpu_mode_replay': gpu.get('mode') == 'replay',
            'npu_mode_capture': npu.get('mode') == 'capture',
            'step_match': gpu.get('step') == npu.get('step') == 0,
            'micro_batch_match': gpu.get('micro_batch') == npu.get('micro_batch') == 0,
            'common_dlogits_match': gpu.get('common_dlogits') == npu.get('common_dlogits'),
        },
        'boundaries': rows,
        'display_order': [{'label': label, **row_by_boundary[boundary]} for label, boundary in display],
        'interpretation': 'Native common-dLogits localization only; it does not isolate an operator. Use common activation, parameters, and dout for isolation.',
    }
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Layer 6 backward localization (common dLogits)')
    print('semantic boundary                                      rel_l2       cosine')
    for row in result['display_order']:
        if not row.get('shape_match', row.get('present', False)):
            print(f"{row['label']:<52} unavailable")
            continue
        print(f"{row['label']:<52} {100*row.get('relative_l2', 0.0):9.6f}%  {row['cosine']:.9f}")
    print('\nNotes:')
    print('- TransformerLayer input_gradient may be unavailable because hidden_states is passed by keyword.')
    print('- core_attention_first_input_grad is normally dQ only, not combined dQ/dK/dV.')
    print('- Compare branch boundaries above; do not subtract unrelated tensor errors as an operator metric.')
    failed = [k for k, v in result['invariants'].items() if not v]
    if failed:
        raise ValueError(f'Layer 6 invariants failed: {failed}')


if __name__ == '__main__':
    main()
