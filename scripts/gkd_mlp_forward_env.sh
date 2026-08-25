#!/usr/bin/env bash
# Source before the unchanged one-step GKD command:
# Capture: source scripts/gkd_mlp_forward_env.sh capture DIR TAG [LAYER_TARGET]
# Replay:  source scripts/gkd_mlp_forward_env.sh replay DIR TAG CAPTURE_FILE [LAYER_TARGET]

_gkd_mlp_forward_mode=${1:-}
_gkd_mlp_forward_dir=${2:-}
_gkd_mlp_forward_tag=${3:-}
_gkd_mlp_forward_capture=${4:-}
if [[ "${_gkd_mlp_forward_mode}" == 'capture' ]]; then
    _gkd_mlp_forward_layer=${4:-decoder.layers.0}
else
    _gkd_mlp_forward_layer=${5:-decoder.layers.0}
fi

if [[ "${_gkd_mlp_forward_mode}" != 'capture' && "${_gkd_mlp_forward_mode}" != 'replay' ]]; then
    echo 'gkd_mlp_forward_env.sh: mode must be capture or replay' >&2
    return 2 2>/dev/null || exit 2
fi
if [[ -z "${_gkd_mlp_forward_dir}" || -z "${_gkd_mlp_forward_tag}" ]]; then
    echo 'gkd_mlp_forward_env.sh: DIR and TAG are required' >&2
    return 2 2>/dev/null || exit 2
fi
if [[ "${_gkd_mlp_forward_mode}" == 'replay' && -z "${_gkd_mlp_forward_capture}" ]]; then
    echo 'gkd_mlp_forward_env.sh: replay requires CAPTURE_FILE' >&2
    return 2 2>/dev/null || exit 2
fi

unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0
export SWIFT_GKD_MLP_MERGE_MODE="${_gkd_mlp_forward_mode}"
export SWIFT_GKD_MLP_MERGE_DIR="${_gkd_mlp_forward_dir}"
export SWIFT_GKD_MLP_MERGE_TAG="${_gkd_mlp_forward_tag}"
export SWIFT_GKD_MLP_MERGE_LAYER_TARGET="${_gkd_mlp_forward_layer}"
export SWIFT_GKD_MLP_MERGE_STEP=0
export SWIFT_GKD_MLP_MERGE_MICRO_BATCH=0

if [[ "${_gkd_mlp_forward_mode}" == 'replay' ]]; then
    export SWIFT_GKD_MLP_MERGE_X_FILE="${_gkd_mlp_forward_capture}"
    export SWIFT_GKD_MLP_MERGE_DOUT_FILE="${_gkd_mlp_forward_capture}"
    export SWIFT_GKD_MLP_MERGE_PARAMETER_FILE="${_gkd_mlp_forward_capture}"
else
    unset SWIFT_GKD_MLP_MERGE_X_FILE
    unset SWIFT_GKD_MLP_MERGE_DOUT_FILE
    unset SWIFT_GKD_MLP_MERGE_PARAMETER_FILE
fi

echo "GKD common-input MLP forward: mode=${_gkd_mlp_forward_mode}, "\
"target=${_gkd_mlp_forward_layer}.mlp, dir=${_gkd_mlp_forward_dir}"

unset _gkd_mlp_forward_mode _gkd_mlp_forward_dir _gkd_mlp_forward_tag
unset _gkd_mlp_forward_capture _gkd_mlp_forward_layer
