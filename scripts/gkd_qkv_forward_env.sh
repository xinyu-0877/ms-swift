#!/usr/bin/env bash
# Source before the unchanged one-step GKD command:
# source scripts/gkd_qkv_forward_env.sh DIR TAG [LAYER_TARGET]

_gkd_qkv_dir=${1:-}
_gkd_qkv_tag=${2:-}
_gkd_qkv_layer=${3:-decoder.layers.2}

if [[ -z "${_gkd_qkv_dir}" || -z "${_gkd_qkv_tag}" ]]; then
    echo 'Usage: source scripts/gkd_qkv_forward_env.sh DIR TAG [LAYER_TARGET]' >&2
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
export SWIFT_GKD_ATTENTION_FORWARD_DIR="${_gkd_qkv_dir}"
export SWIFT_GKD_ATTENTION_FORWARD_TAG="${_gkd_qkv_tag}"
export SWIFT_GKD_ATTENTION_FORWARD_SCOPE=layer
export SWIFT_GKD_ATTENTION_FORWARD_LAYER_TARGET="${_gkd_qkv_layer}"
export SWIFT_GKD_ATTENTION_FORWARD_STEP=0
export SWIFT_GKD_ATTENTION_FORWARD_MICRO_BATCH=0

echo "GKD QKV forward trace: target=${_gkd_qkv_layer}, tag=${_gkd_qkv_tag}, dir=${_gkd_qkv_dir}"

unset _gkd_qkv_dir _gkd_qkv_tag _gkd_qkv_layer
