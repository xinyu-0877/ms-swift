#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

# One-step Qwen3-0.6B Megatron SFT run that captures Layer-0 A-G tensors.
# Usage:
#   bash scripts/qwen3_0_6b_sft_layer0_ag.sh gpu
#   bash scripts/qwen3_0_6b_sft_layer0_ag.sh npu

PLATFORM=${1:-${PLATFORM:-gpu}}
if [[ "${PLATFORM}" != gpu && "${PLATFORM}" != npu ]]; then
    echo 'PLATFORM must be gpu or npu' >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/workspace/Megatron-LM}
export MEGATRON_LM_PATH
export PYTHONPATH="${REPO_ROOT}:${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_PORT=${MASTER_PORT:-40825}
export NPROC_PER_NODE=1

ALIGN_ROOT=${ALIGN_ROOT:-/data/sft_megatron_alignment}
RUN_NAME="qwen3_0.6b_megatron_sft_${PLATFORM}_layer0_ag_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ALIGN_ROOT}/runs/${RUN_NAME}"
TRACE_DIR="${RUN_DIR}/trace"

export XDG_CACHE_HOME="${ALIGN_ROOT}/cache"
export MODELSCOPE_CACHE="${ALIGN_ROOT}/cache/modelscope"
export HF_HOME="${ALIGN_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

if [[ "${PLATFORM}" == gpu ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
else
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
fi

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
DATASET_EN=${DATASET_EN:-/data/gkd/datasets/alpaca-gpt4-data-en#100}
DATASET_ZH=${DATASET_ZH:-/data/gkd/datasets/alpaca-gpt4-data-zh#100}
SEED=${SEED:-42}

mkdir -p \
    "${MODELSCOPE_CACHE}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${RUN_DIR}/logs" \
    "${RUN_DIR}/checkpoints" \
    "${RUN_DIR}/tensorboard" \
    "${TRACE_DIR}"

if ! command -v megatron >/dev/null 2>&1; then
    echo 'Error: megatron command was not found' >&2
    exit 1
fi

if [[ "${PLATFORM}" == gpu ]]; then
    python - <<'PY'
import torch

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available in the current environment')
PY
    nvidia-smi
else
    python - <<'PY'
import torch
import torch_npu  # noqa: F401

print('PyTorch version:', torch.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU device count:', torch.npu.device_count())
if not torch.npu.is_available():
    raise RuntimeError('Ascend NPU is not available in the current environment')
PY
    npu-smi info
fi

# The trace helper only configures hooks.  The training command below must
# still use exactly one micro-batch for an unambiguous step-0 comparison.
source "${REPO_ROOT}/scripts/sft_attention_forward_env.sh" \
    "${TRACE_DIR}" "${PLATFORM}_step0" decoder.layers.0

python - <<'PY'
import os
from pathlib import Path

required = {
    'SWIFT_SFT_ATTENTION_FORWARD_TRACE': '1',
    'SWIFT_SFT_ATTENTION_FORWARD_DIR': None,
    'SWIFT_SFT_ATTENTION_FORWARD_TAG': None,
}
for key, expected in required.items():
    value = os.environ.get(key)
    if not value or (expected is not None and value != expected):
        raise RuntimeError(f'{key} is not configured: {value!r}')
print('Verified SFT trace environment:', {
    key: os.environ[key] for key in required
})
PY
python "${REPO_ROOT}/scripts/check_sft_attention_trace_install.py"

LOG_FILE="${RUN_DIR}/logs/train.log"
{
    echo "platform=${PLATFORM}"
    echo "run_name=${RUN_NAME}"
    echo "model=${MODEL}"
    echo "dataset_en=${DATASET_EN}"
    echo "dataset_zh=${DATASET_ZH}"
    echo 'train_iters=1'
    echo 'micro_batch_size=1'
    echo 'global_batch_size=1'
    echo 'tensor_model_parallel_size=1'
    echo 'pipeline_model_parallel_size=1'
    echo 'context_parallel_size=1'
    echo 'torch_dtype=bfloat16'
    echo 'padding_free=true'
    echo 'attention_backend=flash'
    echo "seed=${SEED}"
    echo "start_time=$(date -Iseconds)"
} > "${RUN_DIR}/run_config.txt"

git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>&1 || true
python --version > "${RUN_DIR}/python_version.txt" 2>&1 || true
pip freeze > "${RUN_DIR}/pip_freeze.txt" 2>&1 || true
cp "$0" "${RUN_DIR}/run_script.sh" 2>/dev/null || true

echo "Starting ${PLATFORM} Layer-0 A-G capture: ${RUN_NAME}"
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
    --train_iters 1 \
    --seed "${SEED}" \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --micro_batch_size 1 \
    --global_batch_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
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
    echo "Training failed with exit code ${TRAIN_STATUS}; see ${LOG_FILE}" >&2
    exit "${TRAIN_STATUS}"
fi

TRACE_FILE="${TRACE_DIR}/sft_attention_forward_${PLATFORM}_step0.pt"
if [[ ! -f "${TRACE_FILE}" ]]; then
    echo "Expected trace was not created: ${TRACE_FILE}" >&2
    exit 1
fi
echo "Capture completed: ${TRACE_FILE}"
