#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

# NPU side of the Qwen3-0.6B Megatron SFT alignment run.
# Override paths/devices through environment variables when needed.
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/workspace/Megatron-LM}"
export MEGATRON_LM_PATH
export PYTHONPATH="${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_PORT="${MASTER_PORT:-40825}"

ALIGN_ROOT="${ALIGN_ROOT:-/data/sft_megatron_alignment}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
RUN_NAME="qwen3_0.6b_megatron_sft_npu_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ALIGN_ROOT}/runs/${RUN_NAME}"

export XDG_CACHE_HOME="${ALIGN_ROOT}/cache"
export MODELSCOPE_CACHE="${ALIGN_ROOT}/cache/modelscope"
export HF_HOME="${ALIGN_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export NPROC_PER_NODE=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DATASET_EN="${DATASET_EN:-/data/gkd/datasets/alpaca-gpt4-data-en#100}"
DATASET_ZH="${DATASET_ZH:-/data/gkd/datasets/alpaca-gpt4-data-zh#100}"
TRAIN_ITERS="${TRAIN_ITERS:-500}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
SEED="${SEED:-42}"

mkdir -p \
  "${MODELSCOPE_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/checkpoints" \
  "${RUN_DIR}/tensorboard"

LOG_FILE="${RUN_DIR}/logs/train.log"

if ! command -v megatron >/dev/null 2>&1; then
  echo "Error: megatron command was not found"
  exit 1
fi

python - <<'PY'
import torch
import torch_npu

print('PyTorch version:', torch.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU device count:', torch.npu.device_count())
if not torch.npu.is_available():
    raise RuntimeError('NPU is not available in the current environment')
PY

npu-smi info

{
  echo "platform=npu"
  echo "run_name=${RUN_NAME}"
  echo "model=${MODEL}"
  echo "dataset_en=${DATASET_EN}"
  echo "dataset_zh=${DATASET_ZH}"
  echo "tuner_type=full"
  echo "torch_dtype=bfloat16"
  echo "train_iters=${TRAIN_ITERS}"
  echo "micro_batch_size=${MICRO_BATCH_SIZE}"
  echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
  echo "seed=${SEED}"
  echo "visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "start_time=$(date -Iseconds)"
} > "${RUN_DIR}/run_config.txt"

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>&1 || true
python --version > "${RUN_DIR}/python_version.txt" 2>&1 || true
pip freeze > "${RUN_DIR}/pip_freeze.txt" 2>&1 || true
npu-smi info > "${RUN_DIR}/device_info.txt" 2>&1 || true
cp "$0" "${RUN_DIR}/run_script.sh" 2>/dev/null || true

echo "Starting NPU Megatron SFT alignment run: ${RUN_NAME}"
echo "Full log: ${LOG_FILE}"

set +e
megatron sft \
  --model "${MODEL}" \
  --save_safetensors true \
  --tuner_type full \
  --dataset \
    "${DATASET_EN}" \
    "${DATASET_ZH}" \
  --torch_dtype bfloat16 \
  --train_iters "${TRAIN_ITERS}" \
  --seed "${SEED}" \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --lr 1e-5 \
  --lr_warmup_fraction 0.05 \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 2 \
  --max_length 1024 \
  --attention_backend flash \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --bias_dropout_fusion false \
  --bias_activation_fusion false \
  --cross_entropy_loss_fusion false \
  --padding_free true \
  --gradient_accumulation_fusion false \
  --masked_softmax_fusion false \
  --finetune true \
  --tensorboard_dir "${RUN_DIR}/tensorboard" \
  --report_to tensorboard \
  --output_dir "${RUN_DIR}/checkpoints" \
  2>&1 | tee "${LOG_FILE}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

date -Iseconds > "${RUN_DIR}/end_time.txt"
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Training failed with exit code ${TRAIN_STATUS}; see ${LOG_FILE}"
  exit "${TRAIN_STATUS}"
fi

echo "NPU alignment run completed: ${RUN_DIR}"
