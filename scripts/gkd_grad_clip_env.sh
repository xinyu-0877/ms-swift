#!/usr/bin/env bash
# Source this file before the unchanged GKD launch command.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/gkd_grad_clip_env.sh <output-dir> <tag> [start-step] [end-step-exclusive]" >&2
  exit 2
fi

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: source scripts/gkd_grad_clip_env.sh <output-dir> <tag> [start-step] [end-step-exclusive]" >&2
  return 2
fi

export SWIFT_GKD_GRAD_CLIP_DEBUG_DIR="$1"
export SWIFT_GKD_GRAD_CLIP_DEBUG_TAG="$2"
export SWIFT_GKD_GRAD_CLIP_DEBUG_START_STEP="${3:-0}"
export SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS="${4:-30}"

# Keep the heavier alignment and operator probes disabled for this experiment.
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

echo "GKD grad-clip debug: tag=${SWIFT_GKD_GRAD_CLIP_DEBUG_TAG}, "\
"steps=[${SWIFT_GKD_GRAD_CLIP_DEBUG_START_STEP},${SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS}), "\
"dir=${SWIFT_GKD_GRAD_CLIP_DEBUG_DIR}"
