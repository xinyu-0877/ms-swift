#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/workspace/Megatron-LM}"
export MEGATRON_LM_PATH
export PYTHONPATH="${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36497}"

# ============================================================
# Base directories
# ============================================================

GKD_ROOT="${GKD_ROOT:-/data/gkd_megatron}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"

RUN_NAME="${RUN_NAME:-megatron_gkd_npu_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GKD_ROOT}/runs/${RUN_NAME}"

export XDG_CACHE_HOME="${GKD_ROOT}/cache"
export MODELSCOPE_CACHE="${GKD_ROOT}/cache/modelscope"
export HF_HOME="${GKD_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Do not reuse stale teacher logits in a normal training run.
unset SWIFT_GKD_TEACHER_CACHE_MODE
unset SWIFT_GKD_TEACHER_CACHE_REUSE_STEP

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=25
export SWIFT_GKD_ALIGNMENT_DEBUG_DIR=/data/gkd_megatron/teacher_realtime_debug

# Determinism settings are independent from the precision switch.
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1
export PYTHONHASHSEED=42
export SWIFT_GKD_DETERMINISTIC=1
export SWIFT_GKD_DETERMINISTIC_SEED=42

# NPU configuration
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# ============================================================
# Model and dataset
# ============================================================

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B}"

DATASET_EN="${DATASET_EN:-/data/gkd/datasets/alpaca-gpt4-data-en}"
DATASET_ZH="${DATASET_ZH:-/data/gkd/datasets/alpaca-gpt4-data-zh}"

# ============================================================
# Training parameters
# ============================================================

TUNER_TYPE="${TUNER_TYPE:-full}"
MAX_STEPS="${MAX_STEPS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
SEED="${SEED:-42}"

# ============================================================
# Precision settings: keep every FP32 setting in this block
# ============================================================

GKD_FP32="${GKD_FP32:-true}"

case "${GKD_FP32,,}" in
  1|true|yes|on)
    GKD_FP32=true
    # Keep GPU/NPU effective precision symmetric: only core attention uses BF16.
    export SWIFT_GKD_FLASH_BF16=1
    export SWIFT_GKD_STRICT_FP32=1
    export SWIFT_GKD_JSD_FP32=1
    export SWIFT_GKD_DTYPE_AUDIT=1
    export SWIFT_GKD_RUNTIME_AUDIT_PATH="${RUN_DIR}/runtime_audit.json"

    PRECISION_ARGS=(
      --torch_dtype float32
      --fp16 false
      --bf16 false
      --apply_query_key_layer_scaling false
      --attention_softmax_in_fp32 true
      --use_precision_aware_optimizer false
      --main_grads_dtype fp32
      --main_params_dtype fp32
      --exp_avg_dtype fp32
      --exp_avg_sq_dtype fp32
      --accumulate_allreduce_grads_in_fp32 true
      --megatron_extra_kwargs '{"fp32_residual_connection": true}'
    )
    ;;
  0|false|no|off)
    GKD_FP32=false
    unset SWIFT_GKD_FLASH_BF16
    unset SWIFT_GKD_STRICT_FP32
    unset SWIFT_GKD_JSD_FP32
    unset SWIFT_GKD_DTYPE_AUDIT
    unset SWIFT_GKD_RUNTIME_AUDIT_PATH

    PRECISION_ARGS=(--torch_dtype bfloat16)
    ;;
  *)
    echo "Error: GKD_FP32 must be true or false, got: ${GKD_FP32}" >&2
    exit 2
    ;;
esac

# ============================================================
# Create output directories
# ============================================================

mkdir -p \
  "${MODELSCOPE_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/checkpoints" \
  "${RUN_DIR}/tensorboard"

LOG_FILE="${RUN_DIR}/logs/train.log"

echo "=================================================="
echo "Run name: ${RUN_NAME}"
echo "Repository: ${REPO_ROOT}"
echo "Cache: ${GKD_ROOT}/cache"
echo "Log: ${LOG_FILE}"
echo "GKD FP32: ${GKD_FP32}"
echo "=================================================="

# ============================================================
# Environment checks
# ============================================================

if ! command -v megatron >/dev/null 2>&1; then
  echo "Error: megatron command was not found" >&2
  exit 1
fi

python - <<'PY'
import torch
import torch_npu

print('PyTorch version:', torch.__version__)
print('NPU available:', torch.npu.is_available())
if not torch.npu.is_available():
    raise RuntimeError('No Ascend NPU is available')
PY

npu-smi info

# ============================================================
# Save environment and configuration
# ============================================================

{
  echo "run_name=${RUN_NAME}"
  echo "student_model=${STUDENT_MODEL}"
  echo "teacher_model=${TEACHER_MODEL}"
  echo "dataset_en=${DATASET_EN}"
  echo "dataset_zh=${DATASET_ZH}"
  echo "tuner_type=${TUNER_TYPE}"
  echo "max_steps=${MAX_STEPS}"
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "eval_batch_size=${EVAL_BATCH_SIZE}"
  echo "gradient_accumulation_steps=${GRAD_ACC}"
  echo "gkd_fp32=${GKD_FP32}"
  echo "flash_bf16=${SWIFT_GKD_FLASH_BF16:-0}"
  echo "attention_backend=flash"
  echo "padding_free=true"
  echo "ascend_devices=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "start_time=$(date -Iseconds)"
} > "${RUN_DIR}/run_config.txt"

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>&1 || true
python --version > "${RUN_DIR}/python_version.txt" 2>&1 || true
pip freeze > "${RUN_DIR}/pip_freeze.txt" 2>&1 || true
npu-smi info > "${RUN_DIR}/device_info.txt" 2>&1 || true
cp "$0" "${RUN_DIR}/run_script.sh" 2>/dev/null || true

# ============================================================
# Training
# ============================================================

echo "Starting GKD training"
echo "Full log: ${LOG_FILE}"

set +e
megatron rlhf \
  --rlhf_type gkd \
  --model "${STUDENT_MODEL}" \
  --save_safetensors true \
  --teacher_model "${TEACHER_MODEL}" \
  --tuner_type "${TUNER_TYPE}" \
  --dataset \
    "${DATASET_EN}" \
    "${DATASET_ZH}" \
  "${PRECISION_ARGS[@]}" \
  --train_iters "${MAX_STEPS}" \
  --seed "${SEED}" \
  --dataset_shuffle false \
  --micro_batch_size "${TRAIN_BATCH_SIZE}" \
  --global_batch_size 4 \
  --lr 1e-5 \
  --lr_warmup_fraction 0.05 \
  --logging_steps 1 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --max_length 1024 \
  --max_completion_length 256 \
  --lmbda 0 \
  --seq_kd false \
  --use_vllm false \
  --beta 0.5 \
  --temperature 1 \
  --attention_backend flash \
  --dataloader_num_workers 0 \
  --train_dataloader_shuffle false \
  --bias_dropout_fusion false \
  --bias_activation_fusion false \
  --cross_entropy_loss_fusion false \
  --padding_free true \
  --gradient_accumulation_fusion false \
  --masked_softmax_fusion false \
  --tensorboard_dir "${RUN_DIR}/tensorboard" \
  --report_to tensorboard \
  --output_dir "${RUN_DIR}/checkpoints" \
  --clip_grad 0 \
  2>&1 | tee "${LOG_FILE}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

date -Iseconds > "${RUN_DIR}/end_time.txt"

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Training failed with exit code ${TRAIN_STATUS}; see ${LOG_FILE}" >&2
  exit "${TRAIN_STATUS}"
fi

echo "=================================================="
echo "Training completed"
echo "Run directory: ${RUN_DIR}"
echo "Training log: ${LOG_FILE}"
echo "Checkpoint: ${RUN_DIR}/checkpoints"
echo "TensorBoard: ${RUN_DIR}/tensorboard"
echo "Runtime audit: ${SWIFT_GKD_RUNTIME_AUDIT_PATH:-disabled}"
echo "=================================================="
