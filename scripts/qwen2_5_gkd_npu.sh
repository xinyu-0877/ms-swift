#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/workspace/Megatron-LM}"
export MEGATRON_LM_PATH
export PYTHONPATH="${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

GKD_ROOT="${GKD_ROOT:-/data/gkd_megatron}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
RUN_NAME="${RUN_NAME:-qwen2_5_0_5b_gkd_npu_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GKD_ROOT}/runs/${RUN_NAME}"

export XDG_CACHE_HOME="${GKD_ROOT}/cache"
export MODELSCOPE_CACHE="${GKD_ROOT}/cache/modelscope"
export HF_HOME="${GKD_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-7B}"
DATASET_EN="${DATASET_EN:-/data/gkd/datasets/alpaca-gpt4-data-en#100}"
DATASET_ZH="${DATASET_ZH:-/data/gkd/datasets/alpaca-gpt4-data-zh#100}"

MAX_STEPS="${MAX_STEPS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
SEED="${SEED:-42}"
USE_HF="${USE_HF:-true}"

# Prevent stale precision-isolation settings from changing a normal training run.
unset SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP
unset SWIFT_GKD_ALIGNMENT_DEBUG_STEPS
unset SWIFT_GKD_ALIGNMENT_DEBUG_DIR
unset SWIFT_GKD_OPERATOR_DEBUG
unset SWIFT_GKD_BACKWARD_DEBUG
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE
unset SWIFT_GKD_ATTENTION_FORWARD_DIR
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE
unset SWIFT_GKD_TEACHER_CACHE_MODE
unset SWIFT_GKD_TEACHER_CACHE_DIR
unset SWIFT_GKD_TEACHER_CACHE_REUSE_STEP

mkdir -p \
  "${MODELSCOPE_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/checkpoints" \
  "${RUN_DIR}/tensorboard"

LOG_FILE="${RUN_DIR}/logs/train.log"

if ! command -v megatron >/dev/null 2>&1; then
  echo "ERROR: megatron command was not found in PATH." >&2
  exit 1
fi

python - <<'PY'
import torch
import torch_npu
from importlib.metadata import version

print('PyTorch version:', torch.__version__)
print('NPU available:', torch.npu.is_available())
if not torch.npu.is_available():
    raise RuntimeError('No Ascend NPU is available.')
print('mcore-bridge version:', version('mcore-bridge'))
PY

npu-smi info

{
  echo "run_name=${RUN_NAME}"
  echo "backend=npu"
  echo "student_model=${STUDENT_MODEL}"
  echo "teacher_model=${TEACHER_MODEL}"
  echo "dataset_en=${DATASET_EN}"
  echo "dataset_zh=${DATASET_ZH}"
  echo "max_steps=${MAX_STEPS}"
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "eval_batch_size=${EVAL_BATCH_SIZE}"
  echo "gradient_accumulation_steps=${GRAD_ACC}"
  echo "seed=${SEED}"
  echo "use_hf=${USE_HF}"
  echo "visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "start_time=$(date -Iseconds)"
} > "${RUN_DIR}/run_config.txt"

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>&1 || true
python --version > "${RUN_DIR}/python_version.txt" 2>&1 || true
pip freeze > "${RUN_DIR}/pip_freeze.txt" 2>&1 || true
npu-smi info > "${RUN_DIR}/device_info.txt" 2>&1 || true
cp "$0" "${RUN_DIR}/run_script.sh" 2>/dev/null || true

echo "Starting Qwen2.5 7B -> 0.5B GKD on NPU"
echo "Run directory: ${RUN_DIR}"

set +e
megatron rlhf \
  --rlhf_type gkd \
  --model "${STUDENT_MODEL}" \
  --teacher_model "${TEACHER_MODEL}" \
  --use_hf "${USE_HF}" \
  --tuner_type full \
  --dataset "${DATASET_EN}" "${DATASET_ZH}" \
  --torch_dtype bfloat16 \
  --train_iters "${MAX_STEPS}" \
  --seed "${SEED}" \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --dataloader_num_workers 0 \
  --micro_batch_size "${TRAIN_BATCH_SIZE}" \
  --global_batch_size 4 \
  --lr 1e-5 \
  --lr_warmup_fraction 0.05 \
  --logging_steps 1 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --save_safetensors true \
  --max_length 1024 \
  --max_completion_length 256 \
  --lmbda 0 \
  --seq_kd false \
  --use_vllm false \
  --beta 1 \
  --temperature 1 \
  --attention_backend flash \
  --bias_dropout_fusion false \
  --bias_activation_fusion false \
  --cross_entropy_loss_fusion false \
  --gradient_accumulation_fusion false \
  --masked_softmax_fusion false \
  --padding_free true \
  --tensorboard_dir "${RUN_DIR}/tensorboard" \
  --report_to tensorboard \
  --output_dir "${RUN_DIR}/checkpoints" \
  2>&1 | tee "${LOG_FILE}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

date -Iseconds > "${RUN_DIR}/end_time.txt"
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Training failed with exit code ${TRAIN_STATUS}. Log: ${LOG_FILE}" >&2
  exit "${TRAIN_STATUS}"
fi

echo "Training completed. Run directory: ${RUN_DIR}"
