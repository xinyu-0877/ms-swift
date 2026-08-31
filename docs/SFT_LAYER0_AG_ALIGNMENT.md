# SFT Layer-0 A-G 对齐

这套诊断在普通 `megatron sft` 的第 0 个训练 step 捕获完整 tensor：

```text
A layer input
B linear_qkv output
C core_attention output
D linear_proj output
E post-attention residual / pre-MLP input
F MLP output
G layer output
```

要求 GPU 和 NPU 使用同一个 checkpoint、数据顺序和 seed，并且诊断运行使用
`--train_iters 1 --micro_batch_size 1 --global_batch_size 1`、TP/PP/CP 均为 1。

## 运行

在对应 GPU 或 NPU 环境执行：

```bash
bash scripts/qwen3_0_6b_sft_layer0_ag.sh gpu
bash scripts/qwen3_0_6b_sft_layer0_ag.sh npu
```

脚本会在 `${ALIGN_ROOT}/runs/.../trace/` 生成：

```text
sft_attention_forward_gpu_step0.pt
sft_attention_forward_npu_step0.pt
```

将两个文件放在同一环境后比较：

```bash
bash scripts/sft_layer0_ag_compare.sh \
  /path/to/sft_attention_forward_gpu_step0.pt \
  /path/to/sft_attention_forward_npu_step0.pt \
  /path/to/sft_layer0_ag_compare.json
```

报告使用完整 tensor 的 `relative_l2`、cosine、`max_abs`，并在比较前校验
`input_ids`、`position_ids`、labels、有效 token 数以及参数 probe。attention backend
名称允许不同，因为 GPU 和 NPU 的实际 kernel 栈本来就不同；其余运行配置会记录在
payload 中供人工复核。

参数 probe 的比较按参数名匹配，不依赖两端记录顺序；dtype 差异只作为 warning，
采样值仍统一转换为 FP32 后比较。如果确实确认两端使用的 checkpoint 不同，只为定位
查看 A-G 数值时可显式加入 `--allow-parameter-probe-mismatch`，报告会保留 mismatch；
这种结果不能作为严格的同参数算子对齐结论。

如果只想把现有 SFT 脚本改成诊断运行，可以在 `megatron sft` 前加入：

```bash
source scripts/sft_attention_forward_env.sh /data/sft_megatron_alignment/trace gpu_step0 decoder.layers.0
```

同时把该次运行改为一个 micro-batch。环境脚本只注册 hooks，不会替代原有的
`megatron sft` 命令。
