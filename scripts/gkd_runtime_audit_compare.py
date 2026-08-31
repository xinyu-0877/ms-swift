import argparse
import json
from pathlib import Path


def _load_json(path):
    if not path:
        return None
    if not Path(path).exists():
        print(f'WARNING: optional file not found, skipped: {path}')
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _step_loss(path, step=0):
    if not path:
        return None
    if not Path(path).exists():
        print(f'WARNING: optional alignment file not found, skipped: {path}')
        return None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get('record_type') == 'forward' and item.get('step') == step:
            loss = item.get('loss', {})
            for key in ('jsd_loss', 'loss'):
                value = loss.get(key) if isinstance(loss, dict) else None
                if isinstance(value, (int, float)):
                    return float(value)
            value = item.get('loss')
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _relative(gpu, npu):
    if gpu is None or npu is None:
        return None
    return abs(gpu - npu) / max(abs(gpu), 1e-30)


def _extract_forward(path):
    if not path:
        return {}
    data = _load_json(path)
    if data is None:
        return {}
    result = {}
    for row in data.get('layers', data.get('layer_boundaries', [])):
        label = row.get('layer', row.get('label'))
        if label is not None and 'output_rel_l2' in row:
            result[str(label)] = row['output_rel_l2']
    tail = data.get('decoder_tail', data.get('tail_boundaries', {}))
    if isinstance(tail, dict):
        for key, value in tail.items():
            if isinstance(value, dict) and 'rel_l2' in value:
                result[key] = value['rel_l2']
    return result


def _extract_backward(path):
    if not path:
        return {}
    data = _load_json(path)
    if data is None:
        return {}
    return {str(row.get('boundary')): row.get('rel_l2')
            for row in data.get('boundaries', data.get('chain', []))
            if row.get('boundary') is not None and row.get('rel_l2') is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--npu', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-on-alignment')
    parser.add_argument('--npu-on-alignment')
    parser.add_argument('--gpu-off-alignment')
    parser.add_argument('--npu-off-alignment')
    parser.add_argument('--gpu-on-forward')
    parser.add_argument('--npu-on-forward')
    parser.add_argument('--gpu-off-forward')
    parser.add_argument('--npu-off-forward')
    parser.add_argument('--gpu-on-backward')
    parser.add_argument('--npu-on-backward')
    parser.add_argument('--gpu-off-backward')
    parser.add_argument('--npu-off-backward')
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()
    gpu = _load_json(args.gpu)
    npu = _load_json(args.npu)
    differences = []

    def compare(path, left, right):
        if left != right:
            differences.append({'path': path, 'gpu': left, 'npu': right})

    for key in ('step', 'micro_batch', 'model_training', 'args', 'effective_ddp', 'accumulation',
                'inputs', 'labels', 'num_valid', 'attention_mask', 'packed_seq_params',
                'teacher_logits', 'parameters'):
        compare(key, gpu.get(key), npu.get(key))
    compare('module_routes', gpu.get('module_routes'), npu.get('module_routes'))
    losses = {}
    for mode in ('on', 'off'):
        gl = _step_loss(getattr(args, f'gpu_{mode}_alignment'), args.step)
        nl = _step_loss(getattr(args, f'npu_{mode}_alignment'), args.step)
        losses[mode] = {'gpu': gl, 'npu': nl, 'relative_error': _relative(gl, nl)}
    forward = {}
    backward = {}
    for mode in ('on', 'off'):
        gf, nf = _extract_forward(getattr(args, f'gpu_{mode}_forward')), _extract_forward(getattr(args, f'npu_{mode}_forward'))
        gb, nb = _extract_backward(getattr(args, f'gpu_{mode}_backward')), _extract_backward(getattr(args, f'npu_{mode}_backward'))
        forward[mode] = {key: {'gpu': gf.get(key), 'npu': nf.get(key), 'relative_error': _relative(gf.get(key), nf.get(key))}
                         for key in sorted(set(gf) | set(nf))}
        backward[mode] = {key: {'gpu': gb.get(key), 'npu': nb.get(key), 'relative_error': _relative(gb.get(key), nb.get(key))}
                          for key in sorted(set(gb) | set(nb))}
    report = {'gpu': args.gpu, 'npu': args.npu, 'difference_count': len(differences),
              'differences': differences,
              'loss': losses, 'forward': forward, 'backward': backward,
              'summary': 'PASS: audit fields match' if not differences else 'DIFF: inspect listed fields'}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(report['summary'])
    for item in differences:
        print(f"DIFF {item['path']}")
    for mode in ('on', 'off'):
        value = losses[mode]['relative_error']
        print(f"recompute_{mode}: loss relative error = {value * 100:.6f}%" if value is not None else f"recompute_{mode}: loss unavailable")
    on_error, off_error = losses['on']['relative_error'], losses['off']['relative_error']
    if on_error is not None and off_error is not None:
        print(f"loss change (off - on) = {(off_error - on_error) * 100:+.6f} pp")
        print('Interpretation: recompute affects drift' if off_error < on_error else 'Interpretation: recompute does not reduce drift')
    print(f'Full report: {args.output}')


if __name__ == '__main__':
    main()
