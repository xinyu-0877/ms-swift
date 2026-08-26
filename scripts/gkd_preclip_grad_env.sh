#!/usr/bin/env bash
# Source this file, then run the unchanged GKD training command in the same shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/gkd_preclip_grad_env.sh <output-dir> <gpu|npu> [steps]" >&2
  exit 2
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: source scripts/gkd_preclip_grad_env.sh <output-dir> <gpu|npu> [steps]" >&2
  return 2
fi

export SWIFT_GKD_PRECLIP_GRAD_DIR="$1"
export SWIFT_GKD_PRECLIP_GRAD_TAG="$2"
export SWIFT_GKD_PRECLIP_GRAD_STEPS="${3:-0,13}"
export SWIFT_GKD_PRECLIP_GRAD_CHUNK_NUMEL="${SWIFT_GKD_PRECLIP_GRAD_CHUNK_NUMEL:-4194304}"

# This run only needs final accumulated pre-clip gradients.
export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=0
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0
unset SWIFT_GKD_JSD_ISOLATION_MODE
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE

echo "GKD full pre-clip gradient capture: tag=${SWIFT_GKD_PRECLIP_GRAD_TAG}, "\
"steps=${SWIFT_GKD_PRECLIP_GRAD_STEPS}, dir=${SWIFT_GKD_PRECLIP_GRAD_DIR}"
echo "Now run the original GKD command in this shell with train_iters >= 14."
