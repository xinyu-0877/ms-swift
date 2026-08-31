# GPU/NPU GKD 精度问题排查记录

更新时间：2026-08-26

## 1. 排查目标

目标是在相同模型、输入和训练配置下，定位 NVIDIA GPU 与昇腾 NPU 的 GKD 训练误差来源，并将单步 loss 相对误差控制在 1% 以内。

当前模型和主要配置：

- Qwen3 系列 Megatron 模型；
- BF16；
- Flash Attention；
- TP=1、CP=1；
- 单算子隔离通常使用一个 micro-batch，避免梯度累积污染结果。

本文严格区分三类结论：

1. **原生路径观察**：GPU/NPU 分别执行自己的前向和反向，只在节点上记录张量；
2. **共同 dLogits 整网反向**：只固定反向入口梯度，前向激活仍来自各自设备；
3. **单算子隔离**：固定共同输入、共同参数和共同上游梯度，再比较局部输出或梯度。

只有第三类实验可以直接判断某个算子的局部 GPU/NPU 实现是否异常。

## 2. 指标和判断标准

主要指标：

```text
relative_l2 = ||GPU张量 - NPU张量||₂ / ||参考张量||₂
```

记录中通常以小数保存：

```text
0.01  = 1%
0.001 = 0.1%
```

隔离实验的参考阈值：

| relative_l2 | 解释 |
|---:|---|
| `< 0.1%` | 强对齐 |
| `0.1%～1%` | 局部可接受，但可能随深度累积 |
| `>= 1%` | 需要继续定位当前边界或更早生产者 |

同时检查 cosine、GPU/NPU norm 和张量形状。不能只使用 SHA256 或 `different_ratio` 判断数值问题。

## 3. 已控制的实验条件

已确认或在有效实验中校验：

- 相同 HF-sharded 学生 checkpoint；
- 相同 `input_ids`、labels、position IDs、`num_valid`；
- 相同 teacher logits 或公共 teacher cache；
- 相同数据顺序和 micro-batch；
- 相同 BF16 精度；
- 相同 `padding_free` 配置；
- 相同 attention backend 边界；
- 相同 TP/CP；
- 相同目标层和模块边界。

NPU 导出的公共 HF-sharded checkpoint 可以同时被 GPU/NPU 加载。NPU MCore `.distcp` 不能直接作为公共 checkpoint，因为存在元数据和 Transformer Engine `_extra_state` 兼容问题。

## 4. 关于 step 13 的重要边界

曾观察到原训练第 13 步 GPU/NPU 梯度误差约为 48%。后续 Layer 27 排查脚本使用：

```bash
--model /data/gkd_megatron/common_checkpoint/checkpoint-13-hf-sharded
--finetune true
--no_load_optim true
--no_load_rng true
--train_iters 1
```

因此后续实验实际代表：

```text
checkpoint-13 的学生权重
+ 新训练进程的 step 0 / micro-batch 0
+ teacher cache 的 step 0 logits
```

它不是原训练 step 13 现场的完整恢复，因为没有恢复：

- 原 Adam 状态；
- 原 RNG 状态；
- 原数据迭代位置；
- 原 step 13 输入；
- 原 step 13 对应 teacher logits；
- 原 global batch 的梯度累积过程。

这不影响共同输入、共同参数、共同 dout 的单算子结论，但不能用这些实验直接解释原训练 step 13 为什么发生 48% 梯度突变。

要复现原 step 13，应从产生第 13 次更新前的 checkpoint 恢复参数、优化器、RNG 和数据位置，并使用对应输入与 teacher logits。

## 5. 前向排查结果

### 5.1 早期逐层抽样

公共 checkpoint 和输入下：

| 位置 | 前向 relative_l2 |
|---|---:|
| Embedding | 一致 |
| Layer 0 输出 | 约 0.046%～0.123% |
| Layer 1 输出 | 约 0.171%～0.636% |
| Layer 2 输出 | 约 1.64%～2.47% |
| Student logits | 约 0.58%～0.74% |
| JSD loss | 约 0.094% |

Layer 0～2 的结果部分来自抽样日志，不能代替完整张量结果。现有证据提示 Layer 1→2 可能是前向首次超过 1% 的区间，但仍需全层 full-tensor 扫描确认。

### 5.2 Layer 27 Attention A～E 完整张量

| 节点 | relative_l2 |
|---|---:|
| A：Layer 27 输入 | 1.295% |
| B：linear_qkv 输出 | 1.331% |
| C：core_attention 输出 | 2.509% |
| D：linear_proj 输出 | 2.653% |
| E：Attention residual 后/MLP 输入 | 1.342% |
| Attention 分支，按 E-A 推算 | 2.911% |

结论：

- Layer 27 输入前已经存在 1.295% 误差，Layer 27 不是前向误差起点；
- Attention 分支对已有输入差异敏感；
- residual merge 后误差为 1.342%，没有继续放大；
- C/D 输入已经不同，因此 C/D 的百分比级差异不能证明 Attention kernel 错误；
- 两端 residual closure 都约为 1.18%，主要来自 BF16 大张量相减反推分支，不支持 residual add 单侧异常。

## 6. 共同 dLogits 整网反向结果

使用 NPU 捕获的完整学生 `dLogits` 在 GPU replay，反向入口梯度严格一致。

### 6.1 跨层梯度误差

| 反向边界 | relative_l2 |
|---|---:|
| output_layer | 0.280% |
| Layer 27 | 0.701% |
| Layer 21 | 2.499% |
| Layer 14 | 3.939% |
| Layer 7 | 4.517% |
| Layer 2 | 5.321% |
| Layer 0 | 5.148% |

说明：

- 同一 `dLogits` 向浅层传播时，梯度差异增长到约 5%；
- Layer 27→21 和 Layer 21→14 都有明显增长；
- Layer 2→0 没有继续恶化；
- 差异在优化器之前已经存在；
- 共同 `dLogits` 只固定反向入口，不固定每层前向激活，因此不能直接归因于某个反向算子。

### 6.2 Layer 27→21 展开

| 边界 | relative_l2 |
|---|---:|
| Layer 27 | 0.701% |
| Layer 26 | 2.154% |
| Layer 25 | 2.249% |
| Layer 24 | 2.370% |
| Layer 23 | 2.362% |
| Layer 22 | 2.495% |
| Layer 21 | 2.499% |

首次最大增长位于完整 Layer 27 backward，即 Layer 27 输出梯度到 Layer 26 输出梯度之间。

### 6.3 Layer 27 内部原生反向

```text
MLP / FC2 output grad：       0.701%
FC2 input grad：              0.731%
FC1 output grad：             1.606%
MLP local dInput：            3.014%
merge 后共享梯度：            2.046%

self_attention output grad：  2.046%
linear_proj input grad：      1.497%
core_attention first dInput： 3.323%
linear_qkv output grad：      2.236%
linear_qkv input grad：       3.024%
Layer 26 output grad：        2.154%
```

这些节点使用各自设备的前向激活，只能定位增长区间，不能作为单算子错误证据。

## 7. 已完成的算子隔离

### 7.1 汇总表

| 算子/模块 | 固定条件 | 主要结果 | 结论 |
|---|---|---:|---|
| JSD forward/backward | student logits、teacher logits、labels | loss 约 0.000314%；dLogits 约 0.0391% | 不是主要来源 |
| `linear_proj` forward | 输入、权重 | 约 0.0122% | 强对齐 |
| `linear_fc2` forward | 输入、权重 | 约 0.0104% | 强对齐 |
| SwiGLU forward | FC1 输出 | bitwise identical | 排除前向异常 |
| SwiGLU backward | FC1 输出、bias、dout | 0～0.0000159% | 强对齐 |
| Flash Attention forward | Q/K/V | 约 0.013% | 强对齐 |
| Flash Attention backward | Q/K/V、dout | dQ/dK/dV 约 0.17%～0.33% | 局部 BF16 小误差 |
| FC1 + RMSNorm backward | 输入、参数、dout | dInput 0.2822% | 局部可接受 |
| 完整 MLP backward | MLP 输入、全部参数、dout | dInput 0.2844% | 无百分比级异常 |
| MLP residual merge | 两分支梯度及 closure | merge 后 3.014% 降为 2.046% | add 未放大误差 |

目前没有发现：

```text
共同输入 + 共同参数 + 共同 dout
→ 单个已隔离算子仍产生 >= 1% 的 GPU/NPU 局部误差
```

### 7.2 Flash Attention

共同 Q/K/V 的 forward：

```text
relative_l2 ≈ 0.01299%
cosine ≈ 0.9999999916
```

共同 Q/K/V 和 dout 的 backward：

| Layer | dQ | dK | dV |
|---|---:|---:|---:|
| 0 | 0.174% | 0.262% | 0.229% |
| 2 | 0.329% | 0.270% | 0.281% |
| 26 | 0.206% | 0.279% | 0.265% |

所有 cosine 高于约 0.999994。该误差可能参与长期累积，但不是明显损坏的算子，也不能按层数简单线性相加。

### 7.3 FC1 + RMSNorm

正确使用 `module_full_backward_hook` 后：

```text
forward output：          0.0156%
local dInput：            0.2822%
RMSNorm weight gradient： 0.1800%
FC1 weight gradient：     0.000762%
```

早期报告的 dInput 114.8% 无效。原因是 tensor hook 捕获了与 residual 共享的合并梯度，而不是 FC1 模块局部 dInput。

### 7.4 MLP 四路 replay

在同一 GPU 实现上组合 GPU/NPU 捕获的输入激活 `x` 和上游梯度 `dy`：

| 组合 | 解释 | dInput 差异 |
|---|---|---:|
| NPU x + NPU dy，GPU/NPU 实现比较 | 纯实现差异 | 0.2844% |
| GPU x + NPU dy | 只改变前向激活 | 2.5579% |
| NPU x + GPU dy | 只改变上游梯度 | 1.2132% |
| GPU x + GPU dy | 两者同时改变 | 3.0186% |

结论：

- 完整 MLP 本身没有百分比级局部错误；
- 原生 MLP dInput 的约 3.014% 差异主要由已不同的前向激活和上游梯度共同造成；
- 前向激活贡献更大；
- residual 与 MLP 局部梯度 cosine 约为 -0.497，merge 发生部分抵消，没有放大误差。

## 8. 尚未完成共同条件隔离的组件

| 组件 | 当前状态 |
|---|---|
| `linear_fc2` backward | 仅完整 MLP 覆盖，未单独隔离 |
| Attention 前 RMSNorm + `linear_qkv` backward | 未隔离 |
| RoPE backward | 未隔离 |
| `linear_proj` backward | 未隔离 |
| 完整 Self-Attention backward | 未隔离 |
| Attention residual add | 观察过原生路径，未做共同条件隔离 |
| 最终 RMSNorm backward | 未隔离 |
| `output_layer` backward | 未隔离 |
| Embedding weight gradient | 未隔离 |
| dropout/bias-dropout 路径 | 未单独隔离 |
| Adam optimizer step | 未隔离 |

不应无目标地逐个隔离以上组件。应先找到前向首次突增层，再只隔离该层发生突增的分支。

## 9. 当前问题性质判断

现有证据更符合：

```text
GPU/NPU BF16 kernel 产生局部小数值差异
→ 前向激活逐层不同
→ Attention/MLP Jacobian 对输入差异敏感
→ 反向继续传播和累积
→ 浅层梯度差异达到约 5%
```

目前尚未证明：

- 某个共同条件下仍有 >=1% 误差的坏算子；
- residual add 导致误差突增；
- JSD 是主要误差源；
- Flash Attention 存在百分比级局部错误；
- Layer 27 是前向误差起点；
- 原训练 step 13 的 48% 梯度误差由某个已定位算子直接造成。

当前更可能是硬件/框架的局部舍入差异与模型数值敏感性共同作用。换模型可能改变敏感程度，但不能证明底层差异已经解决。

## 10. 无效实验和注意事项

### 10.1 整网 FP32

整网设置 FP32 后 GPU/NPU 都出现 NaN，该实验无效。当前 ms-swift 中 `torch_dtype=float32` 可能落入特殊 mixed-precision 映射，且 Megatron + Transformer Engine + Flash/NPU 路径未证明支持该组合。

恢复配置：

```bash
--torch_dtype bfloat16
--bf16 true
--fp16 false
--attention_softmax_in_fp32 true
```

### 10.2 Teacher cache 与 global batch

若 teacher cache 保存时 `micro_batch_size=1`、`global_batch_size=4`，而诊断时使用 `micro_batch_size=1`、`global_batch_size=1`，只要两边都使用 step 0 / micro-batch 0 且输入一致，局部诊断有效。

teacher logits 不参与学生 Transformer 层的前向计算，因此不会影响纯前向 Layer 输出定位。但如果缓存 logits 与当前输入不对应，loss 和原生反向梯度不再代表正常训练。

### 10.3 activation checkpoint

原始 forward hook 不能直接证明 Flash backward 对齐，因为 backward 期间可能重算 Attention。Flash backward 隔离采用共同 Q/K/V 和共同 dout 直接运行局部 VJP。

### 10.4 脚本续行

Bash 续行符 `\` 后不能带空格，否则可能产生 `--lr: command not found` 或多余参数。

## 11. 当前前向定位工具

已实现两级 full-tensor hook：

1. `layers`：记录 Layer 0～27 每层输入和最终输出；
2. `layer`：记录指定层 A～G。

单层 A～G：

```text
A Layer 输入
B linear_qkv 输出
C core_attention 输出
D linear_proj 输出
E Attention residual 后 / MLP 输入
F MLP 输出
G Layer 最终输出
```

相关文件：

```text
swift/megatron/trainers/gkd_attention_forward_debug.py
scripts/gkd_attention_forward_compare.py
scripts/gkd_attention_forward_env.sh
```

## 12. 历史下一步计划（已部分完成）

### 第一优先级：前向全层扫描

使用完整张量记录 Layer 0～27 的输入和最终输出，计算：

```text
relative_l2
cosine
gpu_norm
npu_norm
```

目标是找到第一个相邻层明显突增的位置。现有抽样提示 Layer 1→2，但必须由 full-tensor 结果确认。

### 第二优先级：展开首次突增层

若确认 Layer 2 首次突增，则记录 Layer 2 A～G：

- A 已超过 1%：继续向 Layer 1 或更早层定位；
- B/C/D 突增：优先隔离 Attention 前 RMSNorm、QKV、RoPE、完整 Attention 或 projection；
- E 突增：检查 Attention residual 闭合；
- F/G 突增：检查 MLP 分支，但不重复已证明无异常的 Layer 27 MLP，除非形状或执行配置不同。

### 第三优先级：原 step 13 复现

如果最终目标是解释原训练 step 13 的 48% 梯度误差，应恢复：

```text
step 13 更新前的共同参数
Adam exp_avg / exp_avg_sq / FP32 master weights
RNG 状态
数据迭代位置
原 step 13 输入
对应 teacher logits
原 global batch 和梯度累积配置
```

然后先比较更新前参数和输入，再使用共同 dLogits 判断差异来自 loss 入口、前向状态、反向传播还是优化器更新。

### 后续反向定位

前向起点解释清楚后，再逐层展开反向 Layer 21～14，找出最大相邻边界增长，然后只隔离对应层的具体分支。

暂时不要重复 JSD、Flash forward、SwiGLU、FC1+RMSNorm、完整 MLP 和 MLP residual merge 隔离，除非模型、层形状、精度或 backend 已改变。

## 13. 2026-08-25/26 新增排查结果

本章是当前权威补充；如果与前文的“尚未获得 full-tensor 结果”或“下一步计划”冲突，以本章为准。

### 13.1 两条实验线必须分开

后续排查实际包含两条不同实验线：

```text
实验线 A：checkpoint-13-hf-sharded + fresh runtime step 0
用途：固定 checkpoint 的前向定位和共同边界算子隔离。

实验线 B：当前 Base/初始化模型 + 实际 runtime step 0
用途：检查训练开始时已经存在的 loss、student logits 和 grad_norm 差异。
```

实验线 A 不是原始训练 step 13 的精确重放，因为使用了 `--no_load_optim true`、
`--no_load_rng true`，且没有恢复原始 batch、teacher logits、数据位置和梯度累积状态。
实验线 B 也不能替代原始 step 13 现场。两条实验线的数值不能直接混合归因。

### 13.2 checkpoint-13 fresh step 0 全层前向结果

相同 checkpoint 和输入下，full-tensor `relative_l2` 为：

| 边界 | relative_l2 |
|---|---:|
| Layer 0 input | 0 |
| Layer 0 output | 0.201% |
| Layer 1 output | 0.374% |
| Layer 2 output | 1.454% |
| Layer 3～26 output | 约 1.45% 逐渐降到 1.23% |
| Layer 27 output | 2.531% |

因此，固定 checkpoint 的误差从 Layer 0 开始产生；Layer 2 是第一次超过 1% 的层，
Layer 27 又出现一次明显增长。不能再把“Layer 1→2 首次增长”仅视为抽样推测。

### 13.3 Layer 0 A～G full-tensor 前向

| 节点 | relative_l2 |
|---|---:|
| A：Layer input | 0 |
| B0：fused RMSNorm + linear_qkv output | 0.00982% |
| raw Q | 0.01103% |
| raw K | 0.00767% |
| Q after QK norm | 0.01409% |
| K after QK norm | 0.00103% |
| Q after RoPE | 0.01592% |
| K after RoPE | 0.00108% |
| V | 0.00778% |
| C：core_attention output | 0.03111% |
| D：linear_proj output | 0.06652% |
| E：pre-MLP input | 0.07104% |
| F：MLP output | 0.27356% |
| G：Layer output | 0.20101% |

结论：Layer 0 的第一处差异出现在 fused RMSNorm + linear_qkv 后，但只有约
`0.00982%`；之后经 Attention、projection 和 MLP 连续传播。没有出现单个边界突然产生
百分比级误差的证据。

Layer 0 完整 MLP 的共同输入、共同参数 replay：

```text
原生 E 输入误差：          0.07104%
原生 F 输出误差：          0.27356%
共同输入 MLP 输出误差：    0.04682%
cosine：                   0.9999998904
```

共同输入后误差下降约 82.9%。这说明 Layer 0 MLP 主要放大已经存在的输入差异，
不是明显损坏的 MLP 算子。82.9% 只用于描述误差下降，不能作为严格的加法归因比例。

### 13.4 Layer 2 A～G full-tensor 前向

| 节点 | relative_l2 |
|---|---:|
| A：Layer input | 0.374% |
| B：linear_qkv output | 0.393% |
| raw Q | 0.4001% |
| Q after QK norm | 0.4889% |
| Q after RoPE | 0.5137% |
| raw K | 0.3398% |
| K after QK norm | 0.2646% |
| K after RoPE | 0.2669% |
| V | 0.5357% |
| C：core_attention output | 0.748% |
| D：linear_proj output | 0.757% |
| E：pre-MLP input | 0.452% |
| F：MLP output | 1.456% |
| G：Layer output | 1.454% |

Layer 2 common-QKV core-attention replay：

```text
relative_l2：0.03469%
cosine：     0.99999994
```

因此原生 C 的 `0.748%` 主要来自已经不同的原生 Q/K/V 经 Attention 传播，
而不是 core-attention 实现自身产生 `0.748%`。Q/K norm 和 RoPE 也没有异常跳变。

Layer 2 的主要前向增长发生在 `E 0.452% → F 1.456%`。结合 Layer 0 和 Layer 27
完整 MLP 的共同边界结果，当前证据更支持 MLP 对输入误差敏感，而不是 MLP 算子损坏。

### 13.5 当前已完成隔离实验的统一结论

| 模块 | 共同边界结果 | 状态 |
|---|---:|---|
| JSD loss | 约 0.000314% | 已完成 forward/backward |
| JSD dStudentLogits | 约 0.0391% | 已完成 |
| linear_proj forward | 约 0.0122% | 已完成 |
| linear_fc2 forward | 约 0.0104% | 已完成 |
| Layer 0 Flash/Core Attention forward | 约 0.01299% | 已完成 |
| Layer 2 Core Attention forward | 约 0.03469% | 已完成 |
| Flash backward | dQ/dK/dV 约 0.17%～0.33% | 已完成 Layer 0/2/26 |
| SwiGLU backward | 0～0.0000159% | 已完成 |
| Layer 27 fused RMSNorm + FC1 dInput | 0.2822% | 已完成 |
| Layer 27 complete MLP dInput | 0.2844% | 已完成 |
| Layer 0 complete MLP forward | 0.04682% | 已完成 |

目前所有在共同输入、共同参数、共同 dout 条件下完成的算子隔离结果都小于 1%。
因此当前没有证据支持“某个单算子实现明显损坏”。

### 13.6 梯度裁剪排查结论

已比较 step 1～49 的裁剪记录：

```text
非对称触发 step：36、41、42、43、47

step 36：
GPU grad_norm = 0.8952，不裁剪
NPU grad_norm = 1.1942，仅 NPU 裁剪
```

因此，梯度裁剪不能解释最初 step 13 的误差突变，因为第一次非对称触发发生在 step 36。
它可能在后续训练中继续放大已经分叉的参数轨迹。

需要修正的机制理解：

- norm clipping 在阈值处是连续的，不是数值不连续开关；
- 两边同时裁剪会分别归一化梯度长度，但不会消除梯度方向差异；
- 后续两边都不裁剪，也不会让已经分叉的参数自动重新同步；
- “训练初期梯度大，所以裁剪后两边误差必然更小”不成立，仍需比较完整梯度向量的
  `relative_l2` 和 cosine。

### 13.7 当前实际 runtime step 0 的正确基线

此前一组 loss 约为 `0.624` 的日志不是本次严格 step 0 实验，已废弃。正确日志满足：

```text
step = 0，micro_batch = 0
input_ids / position_ids / labels SHA256 完全一致
num_valid = 157
三组 256 点参数采样 SHA256 一致
teacher logits norm 和抽样值一致（尚非 full-tensor hash）
learning_rate = 0
parameter delta = 0
```

正确测量结果：

| 项目 | GPU | NPU | 差异 |
|---|---:|---:|---:|
| JSD loss | 3.067898989 | 3.057455540 | 绝对差 0.010443449 |
| JSD loss relative | - | - | 0.0034157，即 0.3416% |
| student logits norm | 2091.186768 | 2101.522461 | 0.4918% |
| global grad norm | 128.091354 | 127.306122 | 0.6168% |

必须明确：无量纲相对误差 `0.0034` 等于 `0.34%`，不是 `0.0034%`。

这次 step 0 forward 发生在 Adam、梯度裁剪和参数更新之前，且 `lr=0`、参数增量为 0。
因此 step 0 loss 差异的优先来源是 student forward，而不是优化器或梯度裁剪。
共同 logits 下的 JSD 已高度对齐，所以不优先重复 JSD 隔离。

三个参数的梯度采样 norm 差异分别约为 embedding `4.73%`、Layer 0 QKV `0.588%`、
Layer 2 FC2 `1.398%`。这些只是 256 点采样 norm 差异，不是 full-tensor
`relative_l2`，不能用来认定对应 backward 算子有问题。

### 13.8 GPU/NPU Attention 软件栈

GPU 安装 `flash-attn` 而 NPU 没有安装同名包是正常现象，不代表 NPU 没有融合 Attention：

```text
GPU：ms-swift → Megatron Core / Transformer Engine → CUDA Attention backend
NPU：ms-swift → MindSpeed repatch → torch_npu → CANN fused attention
```

NVIDIA `flash-attn` 是 CUDA 扩展，不应直接安装到 Ascend 环境。判断算子是否一致，应比较
共同 Q/K/V、参数和 dout 下的数值，而不是只比较 `pip list` 包名。

版本 A/B 测试必须使用兼容的软件栈组合，一次只改变一个兼容 bundle，并复用同一共同 payload。
不要随意单独升级 PyTorch、torch_npu、CANN 或 MindSpeed 中的某一个组件。

### 13.9 诊断脚本的单 micro-batch 要求

Megatron 中：

```text
num_microbatches = global_batch_size / (micro_batch_size × data_parallel_size)
```

MLP merge、全层/单层前向 capture、FC1 isolation 和 common-dLogits isolation 均要求
`num_microbatches == 1`。单卡时必须使用：

```bash
--train_iters 1 \
--micro_batch_size 1 \
--global_batch_size 1
```

仅设置 shell 变量 `GRAD_ACC=4` 不会改变 Megatron 的累积配置，除非它被实际传给训练参数。
另外，`source scripts/gkd_attention_forward_env.sh ...` 只负责导出 hook 环境变量，
不会自行启动训练或立即生成 `.pt` 文件；必须在同一个 shell 里继续运行原始 GKD 命令，
文件才会在目标 train step 的 `finalize_train_step` 阶段保存。

### 13.10 当前下一步

针对实验线 B 的正确 step 0 batch 和模型配置：

1. 使用 `layers` scope 对 GPU/NPU 做全层 full-tensor forward scan。
2. 找到第一个 output `relative_l2` 相对 input 明显增长的层。
3. 仅对该层运行 A～G `layer` trace。
4. 前向来源明确后，再由 NPU capture common dLogits、GPU replay，定位反向首次明显增长边界。
5. 只对选中的分支做共同 activation + parameters + dout 隔离。

对当前 step 0 不应先查 Adam 或梯度裁剪，因为此时 `lr=0` 且参数增量为 0。
除非新的模型、shape、配置或软件栈改变了共同边界条件，也不应重复已经完成的 JSD、Flash、
SwiGLU、FC1 或完整 MLP 隔离。

原始训练 step 13 的 48% 梯度突变仍是独立未解问题。最终要解释它，必须恢复第 13 次更新前的：

```text
完整模型参数
Adam exp_avg / exp_avg_sq
FP32 master parameters
RNG 状态
数据迭代位置和原始 batch
对应 teacher logits
原始 global batch 和梯度累积状态
```

## 14. 2026-08-27/28 权威更新

本章补充 13.10 之后完成的实验。如果本章与 13.10 的“当前下一步”冲突，以本章为准。
以下结果属于实验线 B，即当前 Base/初始化模型的实际 runtime step 0；不能与
checkpoint-13 fresh runtime step 0 的数值混合归因。

### 14.1 Base step 0 全层 full-tensor 前向

所有 17 项 invariant 通过。关键边界如下：

| 边界 | relative_l2 | 相对输入增量 |
|---|---:|---:|
| Layer 0 input | 0 | - |
| Layer 0 output | 0.205443% | +0.205443 pp |
| Layer 1 output | 0.366277% | +0.160834 pp |
| Layer 2 output | 0.023887% | -0.342391 pp |
| Layer 16 output | 0.289642% | +0.065952 pp |
| Layer 20 output | 0.574357% | +0.082263 pp |
| Layer 26 output | 0.901512% | +0.037974 pp |
| Layer 27 output | 1.357126% | +0.455614 pp |
| final RMSNorm output | 1.417083% | +0.059958 pp |
| output logits | 1.455194% | +0.038111 pp |

结论：

- 最早误差注入点是 Layer 0，而不是 Layer 2；
- Layer 2 在当前 Base step 0 中显著抵消了前两层误差，不能把相邻层误差理解为单调累加；
- Layer 3～26 重新缓慢累计，Layer 27 是最后一个显著增长点；
- final RMSNorm 与 output layer 合计只增加约 0.098 个百分点，输出端不是主要前向突增位置；
- 完整 28 层结果符合“多处分散的小误差传播和抵消”，不符合单个坏算子突然注入百分比级误差。

### 14.2 Base step 0 Layer 0 A～G

| 节点 | relative_l2 |
|---|---:|
| A：Layer input | 0 |
| B0：linear_qkv output | 0.009009% |
| Q before QK norm | 0.007209% |
| K before QK norm | 0.011000% |
| Q after QK norm | 0.013019% |
| K after QK norm | 0.001318% |
| core_attention input Q | 0.013920% |
| core_attention input K | 0.001535% |
| core_attention input V | 0.012493% |
| C：core_attention output | 0.039218% |
| D：linear_proj output | 0.076154% |
| E：pre-MLP input | 0.080616% |
| F：MLP output | 0.276297% |
| G：Layer output | 0.205443% |

Layer 0 完整 Self-Attention common-input/common-parameter replay：

```text
self_attention output relative_l2 = 0.076154%
post-attention / pre-MLP input    = 0.080616%
```

所有 common input、parameter、runtime invariant 均通过。因此 Layer 0 的完整 Self-Attention
在共同边界下没有百分比级实现误差；它产生约 0.08% 的局部差异，随后 MLP 对该差异敏感，
最终 Layer 输出为 0.205443%。

### 14.3 Base step 0 Layer 27 A～G

| 节点 | relative_l2 |
|---|---:|
| A：Layer input | 0.901512% |
| B0：linear_qkv output | 0.938085% |
| Q after QK norm | 0.707546% |
| K after QK norm | 0.695134% |
| core_attention input V | 1.597243% |
| C：core_attention output | 1.951504% |
| D：linear_proj output | 2.035174% |
| E：pre-MLP input | 0.942054% |
| F：MLP output | 0.795913% |
| G：Layer output | 1.357126% |

Layer 27 common-QKV core-attention replay：

```text
common Q/K/V SHA256：完全一致
relative_l2 = 0.119736%
cosine      = 0.9999992832
```

原生路径的 C=1.951504% 主要来自已经不同的 Q/K/V，尤其 V，而不是 core-attention
实现本身产生约 1.95% 误差。共同 QKV 下的局部误差仍低于 1%。

### 14.4 Base step 0 common-dLogits 全层反向

NPU capture 完整 dLogits，GPU replay；step、micro-batch、输入、teacher、参数探针、
runtime 和共同 dLogits 等 20 项 invariant 全部通过。30 个必需边界均存在；另外捕获到的
Layer 0 QKV 和 Layer 2 FC2 只是旧 debug pattern，不影响全层链路。

关键反向边界：

| 反向边界 | relative_l2 | 穿过模块后的增量 |
|---|---:|---:|
| common dLogits | 0 | - |
| output_layer input gradient | 0.264749% | output_layer：+0.264749 pp |
| final RMSNorm input gradient | 0.462020% | final RMSNorm：+0.197272 pp |
| Layer 27 output gradient | 0.462020% | 同一逻辑边界 |
| Layer 26 output gradient | 1.272441% | Layer 27：+0.810421 pp |
| Layer 25 output gradient | 1.754167% | Layer 26：+0.481726 pp |
| Layer 20 output gradient | 2.510742% | Layer 21：+0.267430 pp |
| Layer 17 output gradient | 2.666641% | Layer 18：+0.221668 pp |
| Layer 6 output gradient | 3.362880% | Layer 7：+0.216802 pp |
| Layer 5 output gradient | 4.173953% | Layer 6：+0.811073 pp |
| Layer 2 output gradient | 4.027138% | Layer 3：-0.097418 pp |
| Layer 0 output gradient | 4.009176% | Layer 1：-0.004955 pp |

最大的相邻增长区间：

```text
穿过 Layer 6：  +0.811073 pp
穿过 Layer 27： +0.810421 pp
穿过 Layer 26： +0.481726 pp
穿过 Layer 21： +0.267430 pp
穿过 output_layer：+0.264749 pp
```

这些 `delta_pp` 只是相邻边界 relative_l2 的差，不是对应算子的隔离误差。
common-dLogits 只固定整网反向入口，两端仍保留各自的前向激活：

```text
dx_gpu = common_dy × J_gpu(x_gpu)
dx_npu = common_dy × J_npu(x_npu)
```

因此该实验能定位真实训练路径中的梯度增长，但不能证明 Layer 6 或 Layer 27 的 backward
算子损坏。Layer 5～0 没有继续放大，约 4% 的差异主要在更高层形成。

由于 TransformerLayer 的 hidden states 以关键字参数传入，full-backward hook 没有可靠的
Layer 0 input gradient；当前链路能测量穿过 Layer 27～1，不能声称测量了穿过 Layer 0。

### 14.5 已完成反向隔离与未覆盖范围

真正满足共同 activation/input、共同参数和共同 dout 的反向隔离如下：

| 算子/路径 | 已测结果 | 结论 |
|---|---:|---|
| JSD backward | dStudentLogits 0.0391% | 强对齐 |
| core-attention backward | Layer 0/2/26 dQ/dK/dV 0.174%～0.329% | 局部可接受 |
| SwiGLU backward | 0～0.0000159% | 强对齐 |
| Layer 27 fused RMSNorm + FC1 | dInput 0.2822% | 局部可接受 |
| Layer 27 complete MLP | dInput 0.2844% | 局部可接受 |
| Layer 27 MLP residual merge | closure 约 0.17% | 未发现 add 异常 |

这里此前称为 “Flash Attention backward” 的实验，实际 hook 边界是
`decoder.layers.N.self_attention.core_attention`（`TEDotProductAttention`），固定共同
Q/K/V 和共同 Attention dout，比较 dQ/dK/dV。它就是模型功能边界上的 core-attention
backward 隔离，不是完整 Self-Attention backward，也不是直接 hook GPU 内部
`FlashAttention` 子 kernel。

尚未独立完成的主要反向隔离：

```text
linear_fc2 backward
pre-attention RMSNorm + linear_qkv backward
QK Norm backward
RoPE backward
linear_proj backward
完整 Self-Attention backward
Attention residual merge（共同条件）
final RMSNorm backward
output_layer backward
embedding weight gradient
dropout / bias-dropout backward
Adam optimizer update
```

### 14.6 梯度精度配置实验

两端此前都记录到：

```text
args.main_grads_dtype = torch.float32
局部 gradient dtype  = torch.bfloat16
gradient source       = main_grad
```

配置中的 FP32 与局部 backward gradient 为 BF16 并不矛盾；理想链路是：

```text
BF16 local grad → FP32 main_grad 累积/归约 → FP32 Adam
```

曾发现 ms-swift 参数名 `accumulate_allreduce_grads_in_fp32` 与 Megatron DDP 字段
`grad_reduce_in_fp32` 之间需要显式映射。两端使用修改后代码重跑，loss 误差曲线趋势没有
明显改善，部分 step 仍达到约 5%。因此 FP32 main_grad 累积不是当前 forward 差异的根因，
也不足以单独消除训练轨迹分叉；它最多影响多 micro-batch 累积与归约阶段。

当前主要精度链路：

```text
模型参数、hidden states、Q/K/V、logits：BF16
局部 activation gradient：BF16
main_grad：应以运行时 buffer 为准；映射生效时为 FP32
Adam FP32 master parameter：FP32
Adam exp_avg / exp_avg_sq：默认 FP32
```

### 14.7 数值溢出判断

当前没有发现 Inf/NaN 数值溢出的证据：

- step 1～49 梯度 norm 的有限性检查通过；
- loss、激活、dLogits 和全层 backward 张量均可正常计算 relative_l2/cosine；
- 未观察到 `update_successful=False` 或因 overflow 跳过更新；
- DDP 配置启用了 `check_for_nan_in_grad=True`。

此前观察到的 `3232`、`3280` 以及差值 `48` 远低于 BF16 最大有限值约 `3.39e38`，
不是溢出。它说明 BF16 在大数值区域的 ULP 变粗，属于量化/舍入误差。

当前应区分：

```text
Inf/NaN overflow：未发现
BF16 舍入与大数值区域量化：明确存在
GPU/NPU kernel 分块和累加顺序差异：可能存在
误差沿深层 forward/backward 传播：已经测到
```

### 14.8 FP32 residual A/B 实验状态

下一项低成本精度实验是两端同时保持 BF16 参数和分支计算，只启用 FP32 residual
connection，然后重跑 Base/step 0 全层 forward。项目通过以下配置传给
`mcore_bridge.ModelConfig`：

```bash
--megatron_extra_kwargs '{"fp32_residual_connection": true}'
```

新增诊断脚本会记录最终运行时 `fp32_residual_connection`，并在请求 FP32 residual 但实际
配置为 False 时立即失败，防止生成无效结果。

NPU 上直接裸执行 `from mcore_bridge import ModelConfig` 曾因缺少 NVIDIA
`transformer_engine` 失败。这不能说明字段不支持；原因是未先加载 MindSpeed 适配器。
NPU 检查必须先执行 `import swift.megatron`，复现真实训练入口的补丁顺序。不要因此在
Ascend 环境安装 NVIDIA Transformer Engine。

截至本次文档更新，FP32 residual 的 GPU/NPU 数值结果尚未产生，因此不能提前判断它是否
改善对齐。验收时需与 14.1 基线比较 Layer 0/1/27、final RMSNorm 和 logits：

- 多数边界误差下降超过约 30%：BF16 residual 逐层量化是重要来源；
- 变化小于约 10%：residual 精度不是主因；
- 前层改善但后层重新增长：residual 有贡献，但分支 kernel/reduction 仍是主要来源。

### 14.9 当前最高优先级

在 FP32 residual A/B 结果产生前，算子排查的最高优先级仍是 Base/step 0 Layer 6：

1. 对 Layer 6 做原生 common-dLogits 内部反向边界拆分；
2. 找到 MLP、Self-Attention 或 residual merge 中最大的增长分支；
3. 只对该分支做共同 activation + parameters + dout 隔离；
4. 然后处理 Layer 27，但不重复已经完成的 MLP、FC1、SwiGLU 和 core-attention backward；
5. Layer 27 如需新增隔离，应优先完整 Self-Attention、linear_proj、QKV/RMSNorm 和
   Attention residual merge。

原始训练 step 13 的 48% 事件仍需恢复完整训练状态后单独复现，不能用当前 Base/step 0
或 checkpoint-13 fresh runtime step 0 的结果直接替代。

## 15. 2026-08-31 CPU 基线与 NPU 算子复核

本轮验证在 `swift-ascend` 容器内完成，操作范围限定为 `/workspace/ms-swift` 和
`/data/gkd_megatron`。容器内未找到精确路径 `/data/megatron_test`，因此使用
`/data/gkd_megatron` 中已有的公共输入和 NPU capture。CPU 运行通过
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 禁止加载 `torch_npu`。

### 15.1 JSD 公共输入

输入：

```text
/data/gkd_megatron/jsd_isolation/jsd_inputs_step_000013_micro_001.pt
```

两端使用相同 student logits、teacher logits 和 labels，且 `step=13`、
`micro_batch=1`。

| 模式 | CPU loss | NPU loss | 绝对差 | 相对差 | dStudentLogits relative_l2 |
|---|---:|---:|---:|---:|---:|
| BF16 | 1.0611157417 | 1.0624607801 | 0.0013450384 | 0.126596% | 0.234666% |
| FP32 JSD | 1.0617146492 | 1.0616934299 | 0.0000212193 | 0.001999% | 0.016836% |

结果文件：

```text
/data/gkd_megatron/jsd_isolation/jsd_cpu_npu_compare.json
/data/gkd_megatron/jsd_isolation/jsd_cpu_npu_fp32_compare.json
```

结论：在公共输入下，JSD 的 BF16 `exp/log/logsumexp` 路径存在可测的 CPU/NPU
局部差异。启用 `SWIFT_GKD_JSD_FP32=1` 后，loss 相对差下降约 63 倍，
dStudentLogits relative L2 下降约 14 倍。该结果支持 FP32 JSD 作为缓解选项，
但尚未证明它能够消除整网训练轨迹分叉，因此本轮没有修改训练脚本默认配置。

### 15.2 Layer 6 FC1/RMSNorm 公共边界

NPU capture 和公共输入：

```text
/data/gkd_megatron/layer6_fc1_backward/fc1_module_common.pt
/data/gkd_megatron/layer6_fc1_backward/fc1_module_npu_capture.pt
```

CPU 参考实现为 FP32 RMSNorm 统计、BF16 归一化结果和 BF16 linear，由
`scripts/gkd_cpu_fc1_baseline.py` 执行。CPU 与 NPU 使用相同输入、参数和
`output.0` 上游梯度。

| 边界 | relative_l2 | max_abs | cosine |
|---|---:|---:|---:|
| FC1 forward output | 0.006657% | 0.015625 | 0.9999999978 |
| local input gradient | 0.283396% | 6.1035e-05 | 0.9999959844 |
| FC1 weight gradient | 0.001778% | 6.1035e-05 | 0.9999999998 |
| RMSNorm weight gradient | 0% | 0 | 1 |

报告：

```text
/data/gkd_megatron/layer6_fc1_backward/cpu_npu_fc1_compare.json
```

结论：Layer 6 FC1/RMSNorm 在共同输入、共同参数、共同 dout 下没有达到 1% 的
局部误差阈值，不支持将当前整网反向误差增长归因于该算子。该结果与已有的
SwiGLU 公共输入测试一致，SwiGLU forward 的 CPU/NPU `relative_l2=0`。

### 15.3 当前判断与下一步

1. JSD BF16 数值路径是目前已确认的局部 NPU 精度差异源，FP32 JSD 能显著降低该差异。
2. Layer 6 FC1/RMSNorm 与 SwiGLU 没有发现百分比级局部异常。
3. 仍不能据此断言整网 NPU 训练误差已解决；BF16 前向激活传播、Attention 以及多层
   反向累积仍需按共同 activation、parameter、dout 边界分析。
4. 下一次 A/B 应在相同 GKD 单步脚本中同时设置 `SWIFT_GKD_JSD_FP32=1`，比较
   student logits、loss、pre-clip gradient 和后续参数增量；不要混用 BF16 与 FP32
   的 capture。
