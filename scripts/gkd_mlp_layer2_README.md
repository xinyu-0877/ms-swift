# Layer 2 complete-MLP common-boundary isolation

This experiment determines whether the Layer 2 MLP itself produces the observed
single-coordinate forward difference. Use the same one-step GKD command for all
runs, with `micro_batch_size=global_batch_size=1`.

The NPU capture must be copied to the GPU environment before replay. Keep the
same HF-sharded student checkpoint, input batch, BF16 configuration, padding
mode, TP/CP sizes, and teacher cache for every run.

## 1. NPU native capture

```bash
source scripts/gkd_mlp_merge_env.sh \
  capture \
  /data/c50063518/gkd_megatron/mlp_layer2 \
  npu_capture \
  decoder.layers.2

# Run the existing one-step GKD command in this shell.
```

Expected file:

```text
/data/c50063518/gkd_megatron/mlp_layer2/mlp_merge_npu_capture.pt
```

## 2. GPU native capture

```bash
source scripts/gkd_mlp_merge_env.sh \
  capture \
  /data/c50063518/gkd_megatron/mlp_layer2 \
  gpu_capture \
  decoder.layers.2

# Run the identical one-step GKD command in this shell.
```

## 3. GPU replay with common NPU boundaries

```bash
NPU_CAPTURE=/data/c50063518/gkd_megatron/mlp_layer2/mlp_merge_npu_capture.pt

source scripts/gkd_mlp_merge_env.sh \
  replay \
  /data/c50063518/gkd_megatron/mlp_layer2 \
  gpu_common_npu \
  "${NPU_CAPTURE}" \
  "${NPU_CAPTURE}" \
  "${NPU_CAPTURE}" \
  decoder.layers.2

# Run the identical one-step GKD command in this shell.
```

The replay fixes the complete MLP input, every named MLP parameter, and every
MLP output gradient to the NPU capture. Backend-local Transformer Engine state
remains local.

## 4. Compare

```bash
python scripts/gkd_mlp_merge_compare.py \
  --gpu-capture /data/c50063518/gkd_megatron/mlp_layer2/mlp_merge_gpu_capture.pt \
  --npu-capture /data/c50063518/gkd_megatron/mlp_layer2/mlp_merge_npu_capture.pt \
  --common-npu-replay /data/c50063518/gkd_megatron/mlp_layer2/mlp_merge_gpu_common_npu.pt \
  --output /data/c50063518/gkd_megatron/mlp_layer2/mlp_layer2_compare.json
```

Read these fields first:

```text
native_gpu_vs_npu.mlp_forward_outputs
common_npu_replay.gpu_vs_npu_forward_outputs
common_npu_replay.gpu_vs_npu_local_input_gradient
```

Each tensor metric also reports `max_abs_coordinate`, `first_at_max_abs`, and
`reference_at_max_abs`. Check whether the native `(0, 0, 35)` difference near
48 remains under the common-NPU replay.

- Common forward `relative_l2 < 0.001` (0.1%): the complete MLP forward is
  strongly aligned; native input sensitivity produced the larger difference.
- Common forward between `0.001` and `0.01`: locally different but below the
  operator-investigation threshold; inspect the top differing coordinate.
- Common forward `relative_l2 >= 0.01` or `max_abs` near 48: split the MLP into
  fused RMSNorm+FC1, SwiGLU, and FC2 using the same common boundaries.
- Common forward aligned but local dInput `relative_l2 >= 0.01`: isolate the
  MLP backward path rather than the forward path.

Do not attribute the native Layer 2 difference to the MLP implementation unless
the common-input/common-parameter replay also diverges.
