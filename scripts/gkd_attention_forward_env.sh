#!/usr/bin/env bash
# Source before the existing one-step training command:
# source scripts/gkd_attention_forward_env.sh DIR TAG

_gkd_attention_dir=${1:-}
_gkd_attention_tag=${2:-}

if [[ -z "${_gkd_attention_dir}" || -z "${_gkd_attention_tag}" ]]; then
    echo 'gkd_attention_forward_env.sh: DIR and TAG are required' >&2
    return 2 2>/dev/null || exit 2
fi

unset SWIFT_GKD_JSD_ISOLATION_MODE
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0

export SWIFT_GKD_ATTENTION_FORWARD_TRACE=1
export SWIFT_GKD_ATTENTION_FORWARD_DIR="${_gkd_attention_dir}"
export SWIFT_GKD_ATTENTION_FORWARD_TAG="${_gkd_attention_tag}"
export SWIFT_GKD_ATTENTION_FORWARD_LAYER_TARGET=decoder.layers.27
export SWIFT_GKD_ATTENTION_FORWARD_STEP=0
export SWIFT_GKD_ATTENTION_FORWARD_MICRO_BATCH=0

unset _gkd_attention_dir _gkd_attention_tag
