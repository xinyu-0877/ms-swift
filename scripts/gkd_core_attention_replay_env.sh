#!/usr/bin/env bash
# Source before the unchanged one-step GKD command:
# source scripts/gkd_core_attention_replay_env.sh DIR capture|replay TAG [LAYER_TARGET]

_gkd_core_dir=${1:-}
_gkd_core_mode=${2:-}
_gkd_core_tag=${3:-}
_gkd_core_layer=${4:-decoder.layers.2}

if [[ -z "${_gkd_core_dir}" || ! "${_gkd_core_mode}" =~ ^(capture|replay)$ || -z "${_gkd_core_tag}" ]]; then
    echo 'Usage: source scripts/gkd_core_attention_replay_env.sh DIR capture|replay TAG [LAYER_TARGET]' >&2
    return 2 2>/dev/null || exit 2
fi

unset SWIFT_GKD_JSD_ISOLATION_MODE
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0

export SWIFT_GKD_FLASH_ISOLATION_MODE="${_gkd_core_mode}"
export SWIFT_GKD_FLASH_ISOLATION_DIR="${_gkd_core_dir}"
export SWIFT_GKD_FLASH_ISOLATION_TARGET_PREFIX="${_gkd_core_layer}.self_attention.core_attention"
export SWIFT_GKD_FLASH_ISOLATION_TAG="${_gkd_core_tag}"
export SWIFT_GKD_FLASH_ISOLATION_STEP=0
export SWIFT_GKD_FLASH_ISOLATION_MICRO_BATCH=0
export SWIFT_GKD_FLASH_BACKWARD_ISOLATION=0

echo "GKD core-attention common-QKV: mode=${_gkd_core_mode}, "\
"target=${SWIFT_GKD_FLASH_ISOLATION_TARGET_PREFIX}, dir=${_gkd_core_dir}"

unset _gkd_core_dir _gkd_core_mode _gkd_core_tag _gkd_core_layer
