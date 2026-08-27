#!/usr/bin/env bash
# Capture: source scripts/gkd_self_attention_forward_env.sh capture DIR TAG [LAYER_TARGET]
# Replay:  source scripts/gkd_self_attention_forward_env.sh replay DIR TAG CAPTURE_FILE [LAYER_TARGET]

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'This helper must be sourced, not executed.' >&2
    exit 2
fi

_gkd_sa_mode=${1:-}
_gkd_sa_dir=${2:-}
_gkd_sa_tag=${3:-}
_gkd_sa_source=${4:-}
if [[ "${_gkd_sa_mode}" == 'capture' ]]; then
    _gkd_sa_layer=${4:-decoder.layers.0}
else
    _gkd_sa_layer=${5:-decoder.layers.0}
fi

if [[ ! "${_gkd_sa_mode}" =~ ^(capture|replay)$ || -z "${_gkd_sa_dir}" || -z "${_gkd_sa_tag}" ]]; then
    echo 'Usage: source scripts/gkd_self_attention_forward_env.sh capture DIR TAG [LAYER_TARGET]' >&2
    echo '   or: source scripts/gkd_self_attention_forward_env.sh replay DIR TAG CAPTURE_FILE [LAYER_TARGET]' >&2
    return 2
fi
if [[ "${_gkd_sa_mode}" == 'replay' && -z "${_gkd_sa_source}" ]]; then
    echo 'Replay requires CAPTURE_FILE.' >&2
    return 2
fi

unset SWIFT_GKD_JSD_ISOLATION_MODE
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE
unset SWIFT_GKD_MICROBATCH_BACKWARD_TRACE
unset SWIFT_GKD_PRECLIP_GRAD_STEPS
unset SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_ALIGNMENT_DEBUG_DIR="${_gkd_sa_dir}/alignment_${_gkd_sa_tag}"
export SWIFT_GKD_ALIGNMENT_SAMPLE_COUNT=32
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0

export SWIFT_GKD_SELF_ATTENTION_FORWARD_MODE="${_gkd_sa_mode}"
export SWIFT_GKD_SELF_ATTENTION_FORWARD_DIR="${_gkd_sa_dir}"
export SWIFT_GKD_SELF_ATTENTION_FORWARD_TAG="${_gkd_sa_tag}"
export SWIFT_GKD_SELF_ATTENTION_FORWARD_LAYER_TARGET="${_gkd_sa_layer}"
export SWIFT_GKD_SELF_ATTENTION_FORWARD_STEP=0
export SWIFT_GKD_SELF_ATTENTION_FORWARD_MICRO_BATCH=0
if [[ "${_gkd_sa_mode}" == 'replay' ]]; then
    export SWIFT_GKD_SELF_ATTENTION_FORWARD_SOURCE="${_gkd_sa_source}"
else
    unset SWIFT_GKD_SELF_ATTENTION_FORWARD_SOURCE
fi

echo "GKD complete self-attention forward: mode=${_gkd_sa_mode}, layer=${_gkd_sa_layer}, dir=${_gkd_sa_dir}"
echo 'Run the original GKD command with train_iters=1, micro_batch_size=1, global_batch_size=1, TP=PP=CP=1.'

unset _gkd_sa_mode _gkd_sa_dir _gkd_sa_tag _gkd_sa_source _gkd_sa_layer
