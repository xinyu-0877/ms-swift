import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--npu', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    gpu = json.loads(Path(args.gpu).read_text(encoding='utf-8'))
    npu = json.loads(Path(args.npu).read_text(encoding='utf-8'))
    differences = []

    def compare(path, left, right):
        if left != right:
            differences.append({'path': path, 'gpu': left, 'npu': right})

    for key in ('step', 'micro_batch', 'model_training', 'args', 'effective_ddp', 'accumulation',
                'inputs', 'labels', 'num_valid', 'attention_mask', 'packed_seq_params',
                'teacher_logits', 'parameters'):
        compare(key, gpu.get(key), npu.get(key))
    compare('module_routes', gpu.get('module_routes'), npu.get('module_routes'))
    report = {'gpu': args.gpu, 'npu': args.npu, 'difference_count': len(differences),
              'differences': differences,
              'summary': 'PASS: audit fields match' if not differences else 'DIFF: inspect listed fields'}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(report['summary'])
    for item in differences:
        print(f"DIFF {item['path']}")
    print(f'Full report: {args.output}')


if __name__ == '__main__':
    main()
