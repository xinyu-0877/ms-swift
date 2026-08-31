#!/usr/bin/env python3
"""
GKD GPU-NPU 精度漂移诊断工具

用法:
    python diagnose_precision_drift.py \
        --gpu-audit gpu_step0_audit.json \
        --npu-audit npu_step0_audit.json \
        --gpu-alignment gpu_alignment.jsonl \
        --npu-alignment npu_alignment.jsonl

输出:
    - 精度漂移的具体位置
    - 建议的修复方案
    - 可疑算子列表
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import math


def load_json(path):
    """Load JSON file."""
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_jsonl(path):
    """Load JSONL file."""
    if not Path(path).exists():
        return []
    records = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def relative_error(a, b):
    """Calculate relative error."""
    if a is None or b is None:
        return None
    if abs(a) < 1e-10 and abs(b) < 1e-10:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-10)


def compare_tensors(gpu_tensor, npu_tensor):
    """Compare tensor summaries."""
    if gpu_tensor is None or npu_tensor is None:
        return {'status': 'missing'}

    # Compare shapes
    if gpu_tensor.get('shape') != npu_tensor.get('shape'):
        return {
            'status': 'shape_mismatch',
            'gpu_shape': gpu_tensor.get('shape'),
            'npu_shape': npu_tensor.get('shape'),
        }

    # Compare statistics
    errors = {}
    for key in ['min', 'max', 'mean', 'std', 'norm']:
        gpu_val = gpu_tensor.get(key)
        npu_val = npu_tensor.get(key)
        if gpu_val is not None and npu_val is not None:
            errors[key] = relative_error(gpu_val, npu_val)

    max_error = max(errors.values()) if errors else 0.0

    return {
        'status': 'compared',
        'max_relative_error': max_error,
        'errors': errors,
        'gpu_sha256': gpu_tensor.get('sha256_fp32'),
        'npu_sha256': npu_tensor.get('sha256_fp32'),
        'exact_match': gpu_tensor.get('sha256_fp32') == npu_tensor.get('sha256_fp32'),
    }


def analyze_operator_drift(gpu_records, npu_records):
    """Analyze operator-level drift."""
    gpu_ops = {
        (r['step'], r['micro_batch'], r['name'], r['call_index']): r
        for r in gpu_records if r.get('record_type') == 'operator_forward'
    }
    npu_ops = {
        (r['step'], r['micro_batch'], r['name'], r['call_index']): r
        for r in npu_records if r.get('record_type') == 'operator_forward'
    }

    common_keys = set(gpu_ops.keys()) & set(npu_ops.keys())

    operator_errors = []
    for key in sorted(common_keys):
        step, micro_batch, name, call_index = key
        gpu_op = gpu_ops[key]
        npu_op = npu_ops[key]

        output_cmp = compare_tensors(
            gpu_op.get('output'),
            npu_op.get('output')
        )

        if output_cmp['status'] == 'compared':
            operator_errors.append({
                'step': step,
                'micro_batch': micro_batch,
                'name': name,
                'call_index': call_index,
                'module_type': gpu_op.get('module_type'),
                'max_error': output_cmp['max_relative_error'],
                'exact_match': output_cmp['exact_match'],
            })

    return operator_errors


def analyze_backward_drift(gpu_records, npu_records):
    """Analyze backward gradient drift."""
    gpu_bwd = {
        (r['step'], r['micro_batch'], r['name']): r
        for r in gpu_records if r.get('record_type') == 'module_backward'
    }
    npu_bwd = {
        (r['step'], r['micro_batch'], r['name']): r
        for r in npu_records if r.get('record_type') == 'module_backward'
    }

    common_keys = set(gpu_bwd.keys()) & set(npu_bwd.keys())

    gradient_errors = []
    for key in sorted(common_keys):
        step, micro_batch, name = key
        gpu_grad = gpu_bwd[key]
        npu_grad = npu_bwd[key]

        input_grad_cmp = compare_tensors(
            gpu_grad.get('input_gradient'),
            npu_grad.get('input_gradient')
        )

        if input_grad_cmp['status'] == 'compared':
            gradient_errors.append({
                'step': step,
                'micro_batch': micro_batch,
                'name': name,
                'max_error': input_grad_cmp['max_relative_error'],
            })

    return gradient_errors


def extract_loss_values(records):
    """Extract loss values from alignment records."""
    losses = []
    for rec in records:
        if rec.get('record_type') == 'forward':
            step = rec.get('step')
            loss_data = rec.get('loss', {})

            # Try different loss field names
            loss_val = None
            if isinstance(loss_data, dict):
                loss_val = loss_data.get('jsd_loss') or loss_data.get('loss')
            elif isinstance(loss_data, (int, float)):
                loss_val = loss_data

            if loss_val is not None:
                losses.append({'step': step, 'loss': float(loss_val)})

    return losses


def diagnose(args):
    """Main diagnostic function."""
    print("=" * 80)
    print("GKD GPU-NPU 精度漂移诊断工具")
    print("=" * 80)

    # Load audit files
    print("\n[1/5] 加载审计文件...")
    gpu_audit = load_json(args.gpu_audit)
    npu_audit = load_json(args.npu_audit)

    if not gpu_audit or not npu_audit:
        print("❌ 错误: 无法加载审计文件")
        return 1

    # Load alignment files
    print("[2/5] 加载对齐日志...")
    gpu_alignment = load_jsonl(args.gpu_alignment)
    npu_alignment = load_jsonl(args.npu_alignment)

    # Compare basic configuration
    print("\n[3/5] 对比基础配置...")
    config_diffs = []
    for key in ['args', 'effective_ddp', 'accumulation']:
        if gpu_audit.get(key) != npu_audit.get(key):
            config_diffs.append(key)

    if config_diffs:
        print(f"⚠️  配置差异: {', '.join(config_diffs)}")
    else:
        print("✅ 基础配置一致")

    # Compare inputs
    print("\n[4/5] 对比输入数据...")
    input_cmp = compare_tensors(
        gpu_audit.get('inputs', {}).get('input_ids'),
        npu_audit.get('inputs', {}).get('input_ids')
    )

    if input_cmp.get('exact_match'):
        print("✅ 输入数据完全一致")
    else:
        print(f"⚠️  输入数据存在差异: {input_cmp}")

    # Compare teacher logits
    teacher_cmp = compare_tensors(
        gpu_audit.get('teacher_logits'),
        npu_audit.get('teacher_logits')
    )

    if teacher_cmp.get('status') == 'compared':
        if teacher_cmp.get('exact_match'):
            print("✅ Teacher logits 完全一致")
        else:
            print(f"⚠️  Teacher logits 相对误差: {teacher_cmp['max_relative_error']:.6e}")

    # Analyze operator drift
    print("\n[5/5] 分析算子级别漂移...")
    operator_errors = analyze_operator_drift(gpu_alignment, npu_alignment)

    if operator_errors:
        # Find first divergence
        operator_errors.sort(key=lambda x: (x['step'], x['call_index'], x['max_error']))

        print(f"\n找到 {len(operator_errors)} 个算子输出差异")
        print("\n前 10 个最大误差:")
        for i, err in enumerate(sorted(operator_errors, key=lambda x: -x['max_error'])[:10]):
            status = "✅" if err['max_error'] < 1e-5 else "⚠️" if err['max_error'] < 1e-3 else "❌"
            print(f"  {status} {err['name']:50s} | 相对误差: {err['max_error']:.6e} | {err['module_type']}")

        # Identify first significant drift
        significant_drift = [e for e in operator_errors if e['max_error'] > 1e-4]
        if significant_drift:
            first_drift = significant_drift[0]
            print(f"\n🔍 首次显著漂移 (相对误差 > 1e-4):")
            print(f"   算子: {first_drift['name']}")
            print(f"   类型: {first_drift['module_type']}")
            print(f"   步数: {first_drift['step']}, 微批次: {first_drift['micro_batch']}")
            print(f"   误差: {first_drift['max_error']:.6e}")
    else:
        print("✅ 未发现算子级别差异（可能未启用 OPERATOR_DEBUG）")

    # Analyze gradient drift
    gradient_errors = analyze_backward_drift(gpu_alignment, npu_alignment)

    if gradient_errors:
        print(f"\n找到 {len(gradient_errors)} 个梯度差异")
        gradient_errors.sort(key=lambda x: -x['max_error'])

        print("\n前 5 个最大梯度误差:")
        for err in gradient_errors[:5]:
            status = "✅" if err['max_error'] < 1e-4 else "⚠️" if err['max_error'] < 1e-2 else "❌"
            print(f"  {status} {err['name']:50s} | 相对误差: {err['max_error']:.6e}")

    # Compare loss progression
    print("\n" + "=" * 80)
    print("Loss 对比")
    print("=" * 80)

    gpu_losses = extract_loss_values(gpu_alignment)
    npu_losses = extract_loss_values(npu_alignment)

    if gpu_losses and npu_losses:
        print(f"\nGPU 训练步数: {len(gpu_losses)}, NPU 训练步数: {len(npu_losses)}")

        common_steps = min(len(gpu_losses), len(npu_losses))
        loss_errors = []

        for i in range(common_steps):
            gpu_loss = gpu_losses[i]['loss']
            npu_loss = npu_losses[i]['loss']
            err = relative_error(gpu_loss, npu_loss)
            loss_errors.append({
                'step': gpu_losses[i]['step'],
                'gpu_loss': gpu_loss,
                'npu_loss': npu_loss,
                'relative_error': err,
            })

        # Show loss at key steps
        key_steps = [0, common_steps // 4, common_steps // 2, 3 * common_steps // 4, common_steps - 1]
        print("\n关键步数的 Loss 对比:")
        print(f"{'步数':>6s} | {'GPU Loss':>12s} | {'NPU Loss':>12s} | {'相对误差':>12s} | {'状态':>6s}")
        print("-" * 70)

        for step_idx in key_steps:
            if step_idx < len(loss_errors):
                err_data = loss_errors[step_idx]
                status = "✅" if err_data['relative_error'] < 1e-4 else "⚠️" if err_data['relative_error'] < 1e-2 else "❌"
                print(f"{err_data['step']:6d} | {err_data['gpu_loss']:12.6e} | {err_data['npu_loss']:12.6e} | "
                      f"{err_data['relative_error']:12.6e} | {status}")

        # Check if drift is accumulating
        early_avg = sum(e['relative_error'] for e in loss_errors[:min(10, len(loss_errors))]) / min(10, len(loss_errors))
        late_avg = sum(e['relative_error'] for e in loss_errors[-min(10, len(loss_errors)):]) / min(10, len(loss_errors))

        print(f"\n早期平均相对误差 (前10步): {early_avg:.6e}")
        print(f"后期平均相对误差 (后10步): {late_avg:.6e}")

        if late_avg > 2 * early_avg:
            print("❌ 检测到累积性漂移: 后期误差显著大于早期误差")
        else:
            print("✅ 误差相对稳定，非累积性漂移")

    # Generate recommendations
    print("\n" + "=" * 80)
    print("诊断结论与建议")
    print("=" * 80)

    issues = []

    # Check teacher logit consistency
    if teacher_cmp.get('status') == 'compared' and not teacher_cmp.get('exact_match'):
        issues.append({
            'priority': 'HIGH',
            'issue': 'Teacher logits 不一致',
            'description': f"相对误差: {teacher_cmp['max_relative_error']:.6e}",
            'solution': '检查 teacher 模型加载是否一致，是否使用了相同的随机种子',
        })

    # Check for early drift
    if operator_errors and operator_errors[0]['max_error'] > 1e-4:
        issues.append({
            'priority': 'HIGH',
            'issue': f"首个算子即出现显著漂移: {operator_errors[0]['name']}",
            'description': f"相对误差: {operator_errors[0]['max_error']:.6e}",
            'solution': '检查该算子的 NPU 实现，可能需要提升精度或使用不同算法',
        })

    # Check for accumulating drift
    if gpu_losses and npu_losses and late_avg > 2 * early_avg:
        issues.append({
            'priority': 'HIGH',
            'issue': '检测到累积性误差',
            'description': f"后期误差 ({late_avg:.6e}) 是早期 ({early_avg:.6e}) 的 {late_avg/early_avg:.1f} 倍",
            'solution': '考虑在关键路径使用 FP32: export SWIFT_GKD_JSD_FP32=1',
        })

    # Check for high gradient errors
    if gradient_errors and gradient_errors[0]['max_error'] > 1e-2:
        issues.append({
            'priority': 'MEDIUM',
            'issue': f"梯度误差较大: {gradient_errors[0]['name']}",
            'description': f"相对误差: {gradient_errors[0]['max_error']:.6e}",
            'solution': '检查混合精度设置，考虑使用 FP32 主梯度',
        })

    if not issues:
        print("\n✅ 未发现明显问题，精度对齐良好")
    else:
        print(f"\n发现 {len(issues)} 个问题:\n")
        for i, issue in enumerate(sorted(issues, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]), 1):
            priority_icon = "🔴" if issue['priority'] == 'HIGH' else "🟡" if issue['priority'] == 'MEDIUM' else "🟢"
            print(f"{i}. {priority_icon} [{issue['priority']}] {issue['issue']}")
            print(f"   描述: {issue['description']}")
            print(f"   建议: {issue['solution']}\n")

    print("\n" + "=" * 80)
    print("下一步行动")
    print("=" * 80)
    print("""
1. 如果 teacher logits 不一致:
   - 确保两边使用相同的 checkpoint
   - 检查随机种子设置: --seed

2. 如果检测到早期漂移:
   - 启用 FP32 模式: export SWIFT_GKD_JSD_FP32=1
   - 隔离问题算子: export SWIFT_GKD_OPERATOR_DEBUG=1

3. 如果检测到累积性漂移:
   - 提升关键路径精度 (参考 docs/GKD_PRECISION_ALIGNMENT_GUIDE.md)
   - 检查 vocab_parallel_log_softmax 实现
   - 验证 top-k 一致性: python scripts/check_topk_consistency.py

4. 收集更多数据:
   - 启用详细日志: export SWIFT_GKD_OPERATOR_DEBUG=1
   - 捕获中间结果: export SWIFT_GKD_JSD_ISOLATION_MODE=capture
""")

    return 0 if not issues else 1


def main():
    parser = argparse.ArgumentParser(description='Diagnose GKD precision drift between GPU and NPU')
    parser.add_argument('--gpu-audit', required=True, help='GPU audit JSON file')
    parser.add_argument('--npu-audit', required=True, help='NPU audit JSON file')
    parser.add_argument('--gpu-alignment', help='GPU alignment JSONL file')
    parser.add_argument('--npu-alignment', help='NPU alignment JSONL file')

    args = parser.parse_args()

    return diagnose(args)


if __name__ == '__main__':
    sys.exit(main())
