#!/usr/bin/env bash
# Source this file, then run the unchanged GKD training command in the same shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/gkd_microbatch_backward_env.sh <output-dir> <tag> [step]" >&2
  exit 2
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: source scripts/gkd_microbatch_backward_env.sh <output-dir> <tag> [step]" >&2
  return 2
fi

_gkd_mb_backward_step="${3:-0}"
export SWIFT_GKD_MICROBATCH_BACKWARD_TRACE=1
export SWIFT_GKD_MICROBATCH_BACKWARD_DIR="$1"
export SWIFT_GKD_MICROBATCH_BACKWARD_TAG="$2"
export SWIFT_GKD_MICROBATCH_BACKWARD_STEP="${_gkd_mb_backward_step}"

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP="${_gkd_mb_backward_step}"
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS="$((_gkd_mb_backward_step + 1))"
export SWIFT_GKD_ALIGNMENT_DEBUG_DIR="$1/alignment_$2"
export SWIFT_GKD_ALIGNMENT_SAMPLE_COUNT=32
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0

unset SWIFT_GKD_PRECLIP_GRAD_STEPS
unset SWIFT_GKD_JSD_ISOLATION_MODE
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE

echo "GKD native micro-batch backward trace: tag=$2, step=${_gkd_mb_backward_step}, dir=$1"
echo "Run the original GKD command now; keep micro_batch_size=1 and global_batch_size=4."
unset _gkd_mb_backward_step
