# GPU/NPU 精度对齐问题修复方案

基于代码分析，以下是按优先级排序的修复建议。

## 🔴 Priority 1: 立即修复（高影响）

### 1.1 修复 DDP grad_reduce_in_fp32 映射

**问题：** `accumulate_allreduce_grads_in_fp32` 参数未正确映射到 DDP 配置

**修复文件：** `swift/megatron/utils/megatron_lm_utils.py`

**当前代码（第 522-526 行）：**
```python
for f in dataclasses.fields(DistributedDataParallelConfig):
    key = f.name
    if hasattr(args, key):
        kwargs[key] = getattr(args, key)
ddp_config = DistributedDataParallelConfig(**kwargs)
```

**修复后代码：**
```python
for f in dataclasses.fields(DistributedDataParallelConfig):
    key = f.name
    if hasattr(args, key):
        kwargs[key] = getattr(args, key)

# 显式映射 accumulate_allreduce_grads_in_fp32 到 grad_reduce_in_fp32
if hasattr(args, 'accumulate_allreduce_grads_in_fp32'):
    kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
    logger.info(f'DDP grad_reduce_in_fp32 set to {args.accumulate_allreduce_grads_in_fp32} '
                f'(from accumulate_allreduce_grads_in_fp32)')

ddp_config = DistributedDataParallelConfig(**kwargs)
```

**验证方法：**
```bash
# 在训练日志中应该看到：
# DDP grad_reduce_in_fp32 set to True (from accumulate_allreduce_grads_in_fp32)
```

### 1.2 增强 fp32_residual_connection 验证

**问题：** FP32 residual 配置可能静默失败

**修复文件：** `swift/megatron/model/utils.py`

**当前代码（第 73-75 行）：**
```python
if args.megatron_extra_kwargs:
    kwargs.update(args.megatron_extra_kwargs)
config = ModelConfig(**kwargs)
```

**修复后代码：**
```python
if args.megatron_extra_kwargs:
    kwargs.update(args.megatron_extra_kwargs)
    
    # 验证 fp32_residual_connection 是否被请求
    requested_fp32_residual = args.megatron_extra_kwargs.get('fp32_residual_connection', False)
    if requested_fp32_residual:
        logger.info('FP32 residual connection requested via megatron_extra_kwargs')

config = ModelConfig(**kwargs)

# 验证配置是否生效
if args.megatron_extra_kwargs and args.megatron_extra_kwargs.get('fp32_residual_connection'):
    actual_value = getattr(config, 'fp32_residual_connection', False)
    if not actual_value:
        raise RuntimeError(
            'fp32_residual_connection was requested but not enabled in ModelConfig. '
            'This may indicate:\n'
            '  1. Your Megatron-Core version does not support this feature\n'
            '  2. mcore_bridge version is incompatible\n'
            '  3. NPU environment requires different initialization order\n'
            f'Config type: {type(config).__name__}\n'
            f'Config module: {type(config).__module__}')
    else:
        logger.info(f'✓ fp32_residual_connection successfully enabled: {actual_value}')
```

## 🟠 Priority 2: 重要增强（中影响）

### 2.1 添加精度配置诊断工具

**新建文件：** `swift/megatron/utils/precision_validator.py`

```python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Precision configuration validator for GPU/NPU alignment."""
import torch
from swift.utils import get_logger

logger = get_logger()


def validate_precision_config(args, config, ddp_config, models):
    """验证精度配置的一致性，返回诊断报告。"""
    
    report = {
        'args': {},
        'model_config': {},
        'ddp_config': {},
        'runtime': {},
        'warnings': [],
        'errors': [],
    }
    
    # 1. 检查参数配置
    report['args']['torch_dtype'] = str(args.torch_dtype)
    report['args']['bf16'] = args.bf16
    report['args']['fp16'] = args.fp16
    report['args']['main_grads_dtype'] = str(args.main_grads_dtype)
    report['args']['accumulate_allreduce_grads_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
    
    # 2. 检查 ModelConfig
    report['model_config']['params_dtype'] = str(getattr(config, 'params_dtype', None))
    report['model_config']['fp32_residual_connection'] = getattr(
        config, 'fp32_residual_connection', False)
    
    # 3. 检查 DDP 配置
    if ddp_config:
        report['ddp_config']['grad_reduce_in_fp32'] = getattr(
            ddp_config, 'grad_reduce_in_fp32', None)
        report['ddp_config']['overlap_grad_reduce'] = getattr(
            ddp_config, 'overlap_grad_reduce', None)
    
    # 4. 检查运行时状态
    if models:
        sample_param = next(models[0].parameters())
        report['runtime']['model_param_dtype'] = str(sample_param.dtype)
        report['runtime']['model_device'] = str(sample_param.device)
        
        # 检查梯度 dtype
        sample_param_with_grad = None
        for param in models[0].parameters():
            if param.requires_grad:
                sample_param_with_grad = param
                break
        
        if sample_param_with_grad is not None:
            main_grad = getattr(sample_param_with_grad, 'main_grad', None)
            report['runtime']['main_grad_exists'] = main_grad is not None
            report['runtime']['main_grad_dtype'] = str(main_grad.dtype) if main_grad else None
    
    # 5. 验证一致性
    # 检查 BF16 + FP32 main_grad 应该启用 grad_reduce_in_fp32
    if (args.bf16 and 
        args.main_grads_dtype == torch.float32 and
        not args.accumulate_allreduce_grads_in_fp32):
        report['warnings'].append(
            'BF16 training with FP32 main_grads but accumulate_allreduce_grads_in_fp32=False. '
            'This may cause gradient precision issues.')
    
    # 检查 DDP 配置是否匹配
    if ddp_config and args.accumulate_allreduce_grads_in_fp32:
        ddp_fp32 = getattr(ddp_config, 'grad_reduce_in_fp32', None)
        if ddp_fp32 is False:
            report['errors'].append(
                'accumulate_allreduce_grads_in_fp32=True but DDP grad_reduce_in_fp32=False. '
                'Gradient reduction will NOT use FP32!')
        elif ddp_fp32 is None:
            report['warnings'].append(
                'Cannot verify DDP grad_reduce_in_fp32 setting.')
    
    # 检查 FP32 residual 请求是否生效
    requested_fp32_residual = False
    if hasattr(args, 'megatron_extra_kwargs') and args.megatron_extra_kwargs:
        requested_fp32_residual = args.megatron_extra_kwargs.get(
            'fp32_residual_connection', False)
    
    if requested_fp32_residual:
        actual_fp32_residual = getattr(config, 'fp32_residual_connection', False)
        if not actual_fp32_residual:
            report['errors'].append(
                'fp32_residual_connection was requested but is not enabled in runtime config!')
    
    # 输出报告
    logger.info('='*70)
    logger.info('Precision Configuration Validation Report')
    logger.info('='*70)
    
    for key, value in report['args'].items():
        logger.info(f'  args.{key}: {value}')
    
    logger.info('')
    for key, value in report['model_config'].items():
        logger.info(f'  config.{key}: {value}')
    
    if report['ddp_config']:
        logger.info('')
        for key, value in report['ddp_config'].items():
            logger.info(f'  ddp_config.{key}: {value}')
    
    if report['runtime']:
        logger.info('')
        for key, value in report['runtime'].items():
            logger.info(f'  runtime.{key}: {value}')
    
    if report['warnings']:
        logger.info('')
        logger.warning('Warnings:')
        for warning in report['warnings']:
            logger.warning(f'  ⚠ {warning}')
    
    if report['errors']:
        logger.info('')
        logger.error('Errors:')
        for error in report['errors']:
            logger.error(f'  ✗ {error}')
    
    logger.info('='*70)
    
    return report
```

### 2.2 在训练器中集成验证

**修改文件：** `swift/megatron/trainers/base.py`

在 `__init__` 方法的模型初始化后添加：

```python
# 在模型和 DDP 配置完成后
if os.getenv('SWIFT_VALIDATE_PRECISION_CONFIG', '1') == '1':
    from swift.megatron.utils.precision_validator import validate_precision_config
    precision_report = validate_precision_config(
        self.args, self.config, 
        getattr(self.wrapped_models[0], 'ddp_config', None),
        self.unwrapped_models)
    
    if precision_report['errors']:
        raise RuntimeError(
            f'Precision configuration validation failed with {len(precision_report["errors"])} error(s). '
            'See log above for details.')
```

## 🟡 Priority 3: 调试和监控（低影响，高价值）

### 3.1 添加精度漂移监控

**新建文件：** `swift/megatron/callbacks/precision_monitor.py`

```python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Monitor precision drift during training."""
import os
import torch
from swift.utils import get_logger

logger = get_logger()


class PrecisionDriftMonitor:
    """监控训练过程中的精度漂移。"""
    
    def __init__(self, enabled=None):
        if enabled is None:
            enabled = os.getenv('SWIFT_MONITOR_PRECISION_DRIFT', '0') == '1'
        self.enabled = enabled
        self.step = 0
        
    def on_step_begin(self, trainer, step):
        """在每个训练步骤开始时检查。"""
        if not self.enabled:
            return
        self.step = step
    
    def on_forward_end(self, trainer, outputs, labels):
        """前向传播结束后检查 logits。"""
        if not self.enabled:
            return
        
        logits = outputs if torch.is_tensor(outputs) else outputs.get('logits')
        if logits is None:
            return
        
        # 检查异常值
        if torch.isnan(logits).any():
            logger.error(f'Step {self.step}: NaN detected in student logits!')
        if torch.isinf(logits).any():
            logger.error(f'Step {self.step}: Inf detected in student logits!')
        
        # 记录 norm（每 10 步）
        if self.step % 10 == 0:
            logits_norm = logits.float().norm().item()
            logits_mean = logits.float().mean().item()
            logits_std = logits.float().std().item()
            logger.debug(
                f'Step {self.step} logits: norm={logits_norm:.4f}, '
                f'mean={logits_mean:.4f}, std={logits_std:.4f}')
    
    def on_backward_end(self, trainer):
        """反向传播结束后检查梯度。"""
        if not self.enabled:
            return
        
        # 每 10 步检查一次梯度
        if self.step % 10 != 0:
            return
        
        total_norm = 0.0
        nan_count = 0
        inf_count = 0
        
        for model in trainer.unwrapped_models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                grad = getattr(param, 'main_grad', param.grad)
                if grad is None:
                    continue
                
                if torch.isnan(grad).any():
                    nan_count += 1
                    logger.error(f'Step {self.step}: NaN in gradient {name}')
                
                if torch.isinf(grad).any():
                    inf_count += 1
                    logger.error(f'Step {self.step}: Inf in gradient {name}')
                
                total_norm += grad.float().norm().item() ** 2
        
        total_norm = total_norm ** 0.5
        logger.debug(f'Step {self.step} gradient norm: {total_norm:.4f}')
        
        if nan_count > 0 or inf_count > 0:
            logger.error(
                f'Step {self.step}: Detected {nan_count} NaN gradients, '
                f'{inf_count} Inf gradients')
```

## 🔵 Priority 4: 长期改进

### 4.1 统一算子融合策略

**目标：** 确保 GPU 和 NPU 使用相同的融合策略

**方案：**
1. 创建统一的算子融合配置
2. 在 runtime audit 中记录实际使用的融合策略
3. 对比 GPU/NPU 的融合差异

### 4.2 引入参考实现验证

**方案：**
1. 添加纯 FP32 参考模式
2. 定期与参考模式对比
3. 量化误差累积速率

## 📊 验证清单

修复完成后，使用以下清单验证：

- [ ] GPU 和 NPU 都输出精度配置验证报告
- [ ] 验证报告显示 0 个错误
- [ ] DDP `grad_reduce_in_fp32` 在两端都为 True（当 BF16 + FP32 main_grad）
- [ ] 如果使用 `fp32_residual_connection`，验证其确实生效
- [ ] Base step 0 的 loss 相对误差 < 1%
- [ ] Layer 0-27 输出的 relative_l2 趋势相似
- [ ] 梯度 norm 相对误差 < 2%

## 🚀 快速测试脚本

```bash
# 1. 在 GPU 上测试
SWIFT_VALIDATE_PRECISION_CONFIG=1 \
python your_train_script.py \
    --bf16 true \
    --main_grads_dtype fp32 \
    --accumulate_allreduce_grads_in_fp32 true \
    --train_iters 1

# 2. 在 NPU 上测试（相同参数）
SWIFT_VALIDATE_PRECISION_CONFIG=1 \
python your_train_script.py \
    --bf16 true \
    --main_grads_dtype fp32 \
    --accumulate_allreduce_grads_in_fp32 true \
    --train_iters 1

# 3. 对比两端的验证报告
```

## 💡 额外建议

1. **短期缓解方案：** 如果修复后仍有 1-2% 误差，可以接受并调整 GKD 训练策略：
   - 增大 temperature 到 3.0
   - 降低 KD loss 权重到 0.3
   - 增加 student 自身 CE loss 权重

2. **监控策略：** 即使误差在可接受范围，也建议：
   - 记录每个 checkpoint 的 loss 差异
   - 监控是否有突然的误差跳变
   - 定期对比 GPU/NPU 训练的最终模型性能

3. **文档更新：** 在项目文档中说明：
   - GPU/NPU 精度对齐的已知限制
   - 推荐的精度配置
   - 如何验证配置是否正确
