"""Check top-k consistency between GPU and NPU outputs."""
import torch
import json
import sys
from pathlib import Path

def check_topk_consistency(gpu_file, npu_file):
    """Compare top-k indices from GPU and NPU teacher outputs."""
    gpu_data = json.loads(Path(gpu_file).read_text())
    npu_data = json.loads(Path(npu_file).read_text())

    # Extract teacher output indices if available
    gpu_teacher = gpu_data.get('teacher_logits', {})
    npu_teacher = npu_data.get('teacher_logits', {})

    if not gpu_teacher or not npu_teacher:
        print("WARNING: No teacher logits found in input files")
        return

    # Compare topk_indices if present
    gpu_indices = gpu_teacher.get('topk_indices')
    npu_indices = npu_teacher.get('topk_indices')

    if gpu_indices and npu_indices:
        mismatches = 0
        total = 0
        for i, (g, n) in enumerate(zip(gpu_indices, npu_indices)):
            if g != n:
                mismatches += 1
                if mismatches <= 10:  # Show first 10 mismatches
                    print(f"Position {i}: GPU top-k = {g}, NPU top-k = {n}")
            total += 1

        mismatch_rate = mismatches / total if total > 0 else 0
        print(f"\nTop-K Consistency: {mismatches}/{total} mismatches ({mismatch_rate*100:.2f}%)")

        if mismatch_rate > 0.01:
            print("⚠️  High top-k mismatch rate detected!")
            print("This will cause cascading differences in all downstream computations.")
    else:
        print("No top-k indices found - full vocab mode")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python check_topk_consistency.py <gpu_audit.json> <npu_audit.json>")
        sys.exit(1)
    check_topk_consistency(sys.argv[1], sys.argv[2])
