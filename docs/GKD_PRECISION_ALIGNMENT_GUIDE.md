# GKD GPU-NPU 精度对齐排查指南

## 问题描述
Swift Megatron GKD 在 GPU 和 NPU 上训练时出现精度无法对齐的问题。单算子测试正常，表明问题来源于**累积误差**。

## 🔍 根因分析

### 1. 数值计算累积误差的主要来源

#### 1.1 Vocab-Parallel Log-Softmax (高优先级)
**位置**: `swift/megatron/trainers/vocab_parallel_utils.py:19-53`

```python
def vocab_parallel_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    
    exp_logits = torch.exp(logits - logits_max)  # ⚠️ exp 精度差异
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    
    log_softmax = logits - logits_max - torch.log(sum_exp)  # ⚠️ log 精度差异
```

**问题点**:
- `torch.exp()` 和 `torch.log()` 在 GPU (CUDA) 和 NPU (CANN) 上的底层实现不同
- 两次 `all_reduce` 操作的浮点累加顺序可能不同
- 当 vocab_size 很大时（如 100K+），累积误差显著

**影响范围**: 每个 token 的 log_softmax 都受影响，误差逐层累积

#### 1.2 JSD Loss 中的 LogSumExp (高优先级)
**位置**: `swift/rlhf_trainers/gkd_loss.py:134`

```python
m_log = torch.logsumexp(
    torch.stack([s_log + log_1_minus_beta, t_log + log_beta]), dim=0)
```

**问题点**:
- `torch.logsumexp` 内部使用 `log(sum(exp(x)))` 模式
- GPU 和 NPU 对 exp/log 的优化策略不同
- beta 参数的 log 变换引入额外误差

#### 1.3 Temperature Scaling (中优先级)
**位置**: `swift/rlhf_trainers/gkd_loss.py:257-258`

```python
s_logits = s_logits / temperature
t_logits = t_logits / temperature
```

**问题点**:
- 除法操作在 FP16/BF16 下精度损失
- temperature 通常是 < 1.0 的小数，除法会放大 logits 值
- 放大后的值进入 exp() 操作，误差进一步放大

#### 1.4 Top-K 选择不一致 (致命问题)
**位置**: `swift/megatron/trainers/gkd_utils.py:7-29`

```python
local_topk_vals, local_topk_ids = torch.topk(logits, k=k, dim=-1)
```

**问题点**:
- 当多个 logit 值非常接近时，GPU 和 NPU 可能选择不同的索引
- 例如: logits = [2.001, 2.000, 1.999]，k=2 时可能选择不同的两个
- 一旦 top-k 索引不同，后续所有计算都会分叉

### 2. 梯度计算中的精度问题

#### 2.1 混合精度训练
**位置**: `swift/megatron/trainers/gkd_trainer.py:445-449`

```python
gradient = getattr(parameter, 'main_grad', None)
if gradient is None:
    gradient = parameter.grad
```

**问题点**:
- `main_grad` (FP32) vs `grad` (FP16/BF16) 的选择
- BF16 的尾数位数少，累积误差更大
- NPU 的 BF16 实现可能与 GPU 不同

#### 2.2 梯度裁剪
**位置**: `swift/megatron/trainers/gkd_trainer.py:677-705`

梯度裁剪前的 norm 计算涉及:
- 所有参数梯度的平方和
- 开方操作
- 如果 norm 计算有微小差异，裁剪系数就不同

## 🎯 诊断步骤

### Step 1: 确认 FP32 模式下是否对齐
```bash
# 强制 JSD 计算使用 FP32
export SWIFT_GKD_JSD_FP32=1

# 如果对齐，则确认是 exp/log 的精度问题
# 如果仍不对齐，检查 top-k 或数据输入
```

### Step 2: 检查 Top-K 一致性
```bash
# 使用提供的脚本检查
python scripts/check_topk_consistency.py \
    gpu_audit.json \
    npu_audit.json
```

如果 top-k mismatch rate > 1%，这是主要问题源。

### Step 3: 逐层对比前向传播
```bash
# 启用逐层输出对比
export SWIFT_GKD_OPERATOR_DEBUG=1
export SWIFT_GKD_OPERATOR_DEBUG_LAYER_IO=1
export SWIFT_GKD_OPERATOR_DEBUG_LAYER_IDS=0,1,2  # 前三层

# 运行后对比 alignment.jsonl 中的 operator_forward 记录
```

### Step 4: 隔离 JSD 计算
```bash
# 捕获某一步的 JSD 输入
export SWIFT_GKD_JSD_ISOLATION_MODE=capture
export SWIFT_GKD_JSD_ISOLATION_DIR=./jsd_debug
export SWIFT_GKD_JSD_ISOLATION_STEP=13
export SWIFT_GKD_JSD_ISOLATION_MICRO_BATCH=1
```

然后在 CPU 上用 Python 验证 JSD 计算结果是否一致。

### Step 5: 对比梯度
```bash
# 捕获某一步的全部梯度
export SWIFT_GKD_PRECLIP_GRAD_DIR=./grad_debug
export SWIFT_GKD_PRECLIP_GRAD_TAG=gpu  # 或 npu
export SWIFT_GKD_PRECLIP_GRAD_STEPS=100

# 然后用脚本对比两者的梯度差异
```

## 🔧 修复方案

### 方案 1: 提升关键算子精度 (推荐)

#### 1.1 修改 vocab_parallel_log_softmax
```python
def vocab_parallel_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return torch.nn.functional.log_softmax(logits, dim=-1)
    
    tp_group = mpu.get_tensor_model_parallel_group()
    
    # ✅ 强制使用 FP32 计算以提高精度
    original_dtype = logits.dtype
    logits = logits.float()
    
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    
    # ✅ 使用更稳定的计算方式
    shifted_logits = logits - logits_max
    exp_logits = torch.exp(shifted_logits)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    
    # ✅ 避免 log(0) 的数值问题
    log_sum_exp = torch.log(sum_exp + 1e-8)
    log_softmax = shifted_logits - log_sum_exp
    
    # 转回原始精度
    return log_softmax.to(original_dtype)
```

#### 1.2 修改 JSD Loss 计算
```python
def jsd_loss(s_logits, t_logits, beta, log_softmax_fn, kl_div_fn, chunk_size=512):
    N = s_logits.size(0)
    if N == 0:
        return s_logits.new_zeros(())
    
    # ✅ 强制 FP32 计算
    original_dtype = s_logits.dtype
    s_logits = s_logits.float()
    t_logits = t_logits.float()
    
    total = torch.tensor(0.0, dtype=torch.float32, device=s_logits.device)
    
    # ✅ 预计算 log(beta) 和 log(1-beta) 使用更高精度
    if beta != 0 and beta != 1:
        beta_t = torch.tensor(beta, dtype=torch.float64).float()
        log_beta = torch.log(beta_t)
        log_1_minus_beta = torch.log(1.0 - beta_t)
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        s_log = log_softmax_fn(s_logits[start:end])
        t_log = log_softmax_fn(t_logits[start:end])
        
        if beta == 0:
            jsd = kl_div_fn(s_log, t_log)
        elif beta == 1:
            jsd = kl_div_fn(t_log, s_log)
        else:
            # ✅ 使用更稳定的 LogSumExp
            stack = torch.stack([
                s_log + log_1_minus_beta,
                t_log + log_beta
            ])
            m_log = torch.logsumexp(stack, dim=0)
            jsd = beta_t * kl_div_fn(m_log, t_log) + (1 - beta_t) * kl_div_fn(m_log, s_log)
        
        total = total + jsd.sum()
    
    return total.to(original_dtype)
```

### 方案 2: 确保 Top-K 一致性

#### 2.1 添加确定性 Top-K
```python
def vocab_parallel_topk_deterministic(logits: torch.Tensor, k: int) -> tuple:
    """Deterministic top-k that breaks ties by index."""
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size == 1:
        # ✅ 添加小的 tie-breaker 确保确定性
        tie_breaker = torch.arange(
            logits.shape[-1], 
            dtype=logits.dtype, 
            device=logits.device
        ) * 1e-10
        logits_with_tb = logits + tie_breaker
        return torch.topk(logits_with_tb, k=k, dim=-1)
    
    # ... 原有的 TP 逻辑，但使用 logits_with_tb
```

### 方案 3: 使用环境变量控制精度

在 `gkd_loss.py` 中添加更多控制开关:

```python
# 在 gkd_loss 函数开始处
USE_FP32_LOG_SOFTMAX = os.getenv('SWIFT_GKD_FP32_LOG_SOFTMAX', '0') == '1'
USE_FP32_JSD = os.getenv('SWIFT_GKD_JSD_FP32', '0') == '1'
USE_FP32_KL_DIV = os.getenv('SWIFT_GKD_FP32_KL_DIV', '0') == '1'

if USE_FP32_LOG_SOFTMAX:
    original_log_softmax_fn = log_softmax_fn
    log_softmax_fn = lambda x: original_log_softmax_fn(x.float()).to(x.dtype)
```

## 📊 验证方法

### 对比关键指标
```python
# 使用 scripts/gkd_runtime_audit_compare.py
python scripts/gkd_runtime_audit_compare.py \
    --gpu gpu_audit.json \
    --npu npu_audit.json \
    --gpu-on-alignment gpu_on_alignment.jsonl \
    --npu-on-alignment npu_on_alignment.jsonl \
    --output comparison.json
```

查看输出中的:
- `loss.relative_error`: 应该 < 1e-4
- `forward.*.relative_error`: 逐层误差，找出首次偏离的层
- `backward.*.relative_error`: 梯度误差

### 期望结果
- ✅ FP32 模式: relative_error < 1e-5
- ✅ BF16 模式: relative_error < 1e-3
- ✅ 200 步后累积: relative_error < 1e-2

## 🚨 已知问题

### NPU 特有问题
1. **CANN 算子精度**: NPU 的某些算子（特别是 exp/log）可能使用查表法，精度略低于 GPU
2. **AllReduce 顺序**: NPU 的集合通信顺序可能与 NCCL 不同
3. **BF16 支持**: 检查 NPU 是否真正支持 BF16 还是模拟的

### 检查方法
```bash
# 检查 NPU 是否支持 FP32 残差连接
python scripts/check_fp32_residual_support.py
```

## 📝 总结

**最可能的根因（按概率排序）**:
1. ⭐⭐⭐ Vocab-parallel log_softmax 中的 exp/log 精度差异
2. ⭐⭐⭐ Top-K 选择不一致导致的分叉
3. ⭐⭐ JSD loss 中的 logsumexp 累积误差
4. ⭐ 梯度计算的混合精度问题
5. ⭐ AllReduce 的浮点累加顺序差异

**快速验证路径**:
```bash
# 1. 先试最简单的 - 全 FP32
export SWIFT_GKD_JSD_FP32=1

# 2. 如果还不行，检查 top-k
python scripts/check_topk_consistency.py gpu.json npu.json

# 3. 如果 top-k 一致但仍有问题，深入算子级别
export SWIFT_GKD_OPERATOR_DEBUG=1
```

**终极方案**:
如果上述方案都无效，考虑使用数值等价但算法不同的实现（如用 Gumbel-Softmax 替代标准 softmax）。
