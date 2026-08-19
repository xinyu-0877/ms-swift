#!/usr/bin/env bash
# Source this file before the existing one-step GKD training command.
# Capture: source scripts/gkd_mlp_merge_env.sh capture DIR TAG
# Replay:  source scripts/gkd_mlp_merge_env.sh replay DIR TAG X_FILE DOUT_FILE [PARAM_FILE]

_gkd_merge_fail() {
    echo "gkd_mlp_merge_env.sh: $*" >&2
    return 2 2>/dev/null || exit 2
}

_gkd_merge_mode=${1:-}
_gkd_merge_dir=${2:-}
_gkd_merge_tag=${3:-}

if [[ "${_gkd_merge_mode}" != "capture" && "${_gkd_merge_mode}" != "replay" ]]; then
    _gkd_merge_fail 'mode must be capture or replay' || return 2 2>/dev/null || exit 2
fi
if [[ -z "${_gkd_merge_dir}" || -z "${_gkd_merge_tag}" ]]; then
    _gkd_merge_fail 'DIR and TAG are required' || return 2 2>/dev/null || exit 2
fi

unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0
export SWIFT_GKD_MLP_MERGE_MODE="${_gkd_merge_mode}"
export SWIFT_GKD_MLP_MERGE_DIR="${_gkd_merge_dir}"
export SWIFT_GKD_MLP_MERGE_TAG="${_gkd_merge_tag}"
export SWIFT_GKD_MLP_MERGE_LAYER_TARGET=decoder.layers.27
export SWIFT_GKD_MLP_MERGE_STEP=0
export SWIFT_GKD_MLP_MERGE_MICRO_BATCH=0

if [[ "${_gkd_merge_mode}" == "replay" ]]; then
    _gkd_merge_x_file=${4:-}
    _gkd_merge_dout_file=${5:-}
    _gkd_merge_parameter_file=${6:-${_gkd_merge_x_file}}
    if [[ -z "${_gkd_merge_x_file}" || -z "${_gkd_merge_dout_file}" ]]; then
        _gkd_merge_fail 'replay requires X_FILE and DOUT_FILE' || return 2 2>/dev/null || exit 2
    fi
    export SWIFT_GKD_MLP_MERGE_X_FILE="${_gkd_merge_x_file}"
    export SWIFT_GKD_MLP_MERGE_DOUT_FILE="${_gkd_merge_dout_file}"
    export SWIFT_GKD_MLP_MERGE_PARAMETER_FILE="${_gkd_merge_parameter_file}"
else
    unset SWIFT_GKD_MLP_MERGE_X_FILE
    unset SWIFT_GKD_MLP_MERGE_DOUT_FILE
    unset SWIFT_GKD_MLP_MERGE_PARAMETER_FILE
fi

unset _gkd_merge_mode _gkd_merge_dir _gkd_merge_tag
unset _gkd_merge_x_file _gkd_merge_dout_file _gkd_merge_parameter_file
unset -f _gkd_merge_fail
