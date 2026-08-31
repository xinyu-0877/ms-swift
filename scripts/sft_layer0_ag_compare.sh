#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: bash scripts/sft_layer0_ag_compare.sh GPU_PT NPU_PT [OUTPUT_JSON]
GPU_FILE=${1:-}
NPU_FILE=${2:-}
OUTPUT_FILE=${3:-sft_layer0_ag_compare.json}

if [[ -z "${GPU_FILE}" || -z "${NPU_FILE}" ]]; then
    echo 'Usage: bash scripts/sft_layer0_ag_compare.sh GPU_PT NPU_PT [OUTPUT_JSON]' >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python "${SCRIPT_DIR}/sft_attention_forward_compare.py" \
    --gpu "${GPU_FILE}" \
    --npu "${NPU_FILE}" \
    --output "${OUTPUT_FILE}"

