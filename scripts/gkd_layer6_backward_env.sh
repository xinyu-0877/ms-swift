#!/usr/bin/env bash
# Configure the existing common-input/common-dout module isolation hook for
# Layer 6. The trainer hook is named FC1 isolation for historical reasons,
# but it accepts any exact named_modules() target.
#
# FC1/RMSNorm:
#   source scripts/gkd_layer6_backward_env.sh fc1 capture DIR TAG
#   source scripts/gkd_layer6_backward_env.sh fc1 replay  DIR TAG
# Complete self-attention:
#   source scripts/gkd_layer6_backward_env.sh attention capture DIR TAG
#   source scripts/gkd_layer6_backward_env.sh attention replay  DIR TAG

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'This helper must be sourced, not executed.' >&2
    exit 2
fi

_gkd_l6_kind=${1:-}
_gkd_l6_mode=${2:-}
_gkd_l6_dir=${3:-}
_gkd_l6_tag=${4:-}

if [[ ! "${_gkd_l6_kind}" =~ ^(fc1|attention)$ || ! "${_gkd_l6_mode}" =~ ^(capture|replay)$ \
      || -z "${_gkd_l6_dir}" || -z "${_gkd_l6_tag}" ]]; then
    echo 'Usage: source scripts/gkd_layer6_backward_env.sh fc1|attention capture|replay DIR TAG' >&2
    return 2
fi

if [[ "${_gkd_l6_kind}" == 'fc1' ]]; then
    _gkd_l6_target='decoder.layers.6.mlp.linear_fc1'
else
    _gkd_l6_target='decoder.layers.6.self_attention'
fi

# Disable unrelated probes so the selected module is run exactly once.
unset SWIFT_GKD_JSD_ISOLATION_MODE
unset SWIFT_GKD_DLOGITS_ISOLATION_MODE
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE
unset SWIFT_GKD_SELF_ATTENTION_FORWARD_MODE
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE
unset SWIFT_GKD_MICROBATCH_BACKWARD_TRACE
unset SWIFT_GKD_PRECLIP_GRAD_STEPS
unset SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_ALIGNMENT_DEBUG_DIR="${_gkd_l6_dir}/alignment_${_gkd_l6_tag}"
export SWIFT_GKD_ALIGNMENT_SAMPLE_COUNT=32
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0

# The current trainer implementation uses these names for the generic VJP
# hook. Its target is exact, so no other layer is intercepted.
export SWIFT_GKD_FC1_ISOLATION_MODE="${_gkd_l6_mode}"
export SWIFT_GKD_FC1_ISOLATION_DIR="${_gkd_l6_dir}"
export SWIFT_GKD_FC1_ISOLATION_TARGET="${_gkd_l6_target}"
export SWIFT_GKD_FC1_ISOLATION_STEP=0
export SWIFT_GKD_FC1_ISOLATION_MICRO_BATCH=0
export SWIFT_GKD_FC1_ISOLATION_TAG="${_gkd_l6_tag}"

mkdir -p "${_gkd_l6_dir}"
echo "GKD Layer 6 backward isolation: kind=${_gkd_l6_kind}, mode=${_gkd_l6_mode}, target=${_gkd_l6_target}"
echo 'Run the original GKD command with train_iters=1, micro_batch_size=1, global_batch_size=1 and TP=PP=CP=1.'

unset _gkd_l6_kind _gkd_l6_mode _gkd_l6_dir _gkd_l6_tag _gkd_l6_target
