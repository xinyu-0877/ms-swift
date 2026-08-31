# GKD CPU/NPU 算子基线（2026-08-31）

本次验证只使用 `swift-ascend` 容器内的 `/workspace/ms-swift` 和
`/data/gkd_megatron`。容器内未找到精确路径 `/data/megatron_test`；实际使用的
公共输入和 NPU capture 均来自 `/data/gkd_megatron`。

## JSD 公共输入

输入文件：

```text
/data/gkd_megatron/jsd_isolation/jsd_inputs_step_000013_micro_001.pt
```

CPU 与 NPU 使用相同的 student logits、teacher logits 和 labels。

| 模式 | loss（CPU） | loss（NPU） | loss 相对差 | dStudentLogits relative_l2 | cosine |
|---|---:|---:|---:|---:|---:|
| BF16 | 1.0611157417 | 1.0624607801 | 0.126596% | 0.234666% | 0.99999857 |
| FP32 JSD | 1.0617146492 | 1.0616934299 | 0.001999% | 0.016836% | 约 1.0 |

报告文件：

```text
/data/gkd_megatron/jsd_isolation/jsd_cpu_npu_compare.json
/data/gkd_megatron/jsd_isolation/jsd_cpu_npu_fp32_compare.json
```

结论：JSD 的 BF16 `exp/log/logsumexp` 路径存在可测的 NPU/CPU 局部差异；在相同
输入下启用 `SWIFT_GKD_JSD_FP32=1` 后，loss 差下降约 63 倍，梯度 relative_l2
下降约 14 倍。该结果支持将 FP32 JSD 作为缓解选项，但尚未证明它能消除整网训练
轨迹分叉，因此不自动修改训练脚本默认值。

## Layer 6 FC1/RMSNorm 公共边界

NPU capture：

```text
/data/gkd_megatron/layer6_fc1_backward/fc1_module_common.pt
/data/gkd_megatron/layer6_fc1_backward/fc1_module_npu_capture.pt
```

CPU 参考实现为 FP32 RMSNorm 统计、BF16 归一化结果和 BF16 linear；实现脚本为
`scripts/gkd_cpu_fc1_baseline.py`。CPU 与 NPU 使用相同输入、参数和 `output.0`
上游梯度。

| 边界 | relative_l2 | max_abs | cosine |
|---|---:|---:|---:|
| FC1 forward output | 0.006657% | 0.015625 | 0.9999999978 |
| local input gradient | 0.283396% | 6.1035e-05 | 0.9999959844 |
| FC1 weight gradient | 0.001778% | 6.1035e-05 | 0.9999999998 |
| RMSNorm weight gradient | 0% | 0 | 1 |

完整报告：

```text
/data/gkd_megatron/layer6_fc1_backward/cpu_npu_fc1_compare.json
```

结论：Layer 6 FC1/RMSNorm 在共同输入、共同参数、共同 dout 下没有达到 1% 的
局部误差阈值，不支持将当前整网梯度增长归因于该算子。

## 当前判断

1. JSD BF16 数值路径是已确认的局部误差源，FP32 JSD 能显著降低该局部差异。
2. Layer 6 FC1/RMSNorm、SwiGLU（CPU/NPU forward `relative_l2=0`）未发现异常。
3. 尚不能据此断言整网 NPU 训练误差已解决；模型 BF16 前向激活传播、Attention
   与多层反向累积仍需按共同 activation/parameter/dout 边界分析。
4. 所有百分比均为 dimensionless relative L2 转换后的百分数；例如 `0.126596%`
   对应小数 `0.00126596`。

## 复现实验

CPU-only（容器内不加载 `torch_npu`）：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python scripts/gkd_jsd_backward_isolation.py run --device cpu \
  --input /data/gkd_megatron/jsd_isolation/jsd_inputs_step_000013_micro_001.pt \
  --output /data/gkd_megatron/jsd_isolation/jsd_backward_cpu.pt
```

FP32 JSD 对照时在 CPU、NPU 两端都添加 `--fp32`，并使用同一输入文件；不要混用
BF16 与 FP32 结果。
