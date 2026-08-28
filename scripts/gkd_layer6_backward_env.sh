#!/usr/bin/env bash
# Capture: source scripts/gkd_layer6_backward_env.sh capture DIR TAG
# Replay:  source scripts/gkd_layer6_backward_env.sh replay DIR TAG

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'This helper must be sourced, not executed.' >&2
    exit 2
fi

_gkd_l6_mode=${1:-}
_gkd_l6_dir=${2:-}
_gkd_l6_tag=${3:-}
if [[ ! "${_gkd_l6_mode}" =~ ^(capture|replay)$ || -z "${_gkd_l6_dir}" || -z "${_gkd_l6_tag}" ]]; then
    echo 'Usage: source scripts/gkd_layer6_backward_env.sh capture|replay DIR TAG' >&2
    return 2
fi

# Disable unrelated probes so the payload is small and the module inventory is unambiguous.
unset SWIFT_GKD_JSD_ISOLATION_MODE SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE SWIFT_GKD_ATTENTION_FORWARD_TRACE
unset SWIFT_GKD_SELF_ATTENTION_FORWARD_MODE SWIFT_GKD_MICROBATCH_BACKWARD_TRACE
unset SWIFT_GKD_PRECLIP_GRAD_STEPS SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_ALIGNMENT_DEBUG_DIR="${_gkd_l6_dir}/alignment_${_gkd_l6_tag}"
export SWIFT_GKD_ALIGNMENT_SAMPLE_COUNT=16
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0

export SWIFT_GKD_DLOGITS_ISOLATION_MODE="${_gkd_l6_mode}"
export SWIFT_GKD_DLOGITS_ISOLATION_DIR="${_gkd_l6_dir}"
export SWIFT_GKD_DLOGITS_ISOLATION_TAG="${_gkd_l6_tag}"
export SWIFT_GKD_DLOGITS_ISOLATION_STEP=0
export SWIFT_GKD_DLOGITS_ISOLATION_MICRO_BATCH=0
export SWIFT_GKD_DLOGITS_BACKWARD_PATTERNS='output_layer,decoder.final_layernorm,decoder.layers.6,decoder.layers.6.mlp,decoder.layers.6.mlp.linear_fc2,decoder.layers.6.mlp.linear_fc1,decoder.layers.6.self_attention,decoder.layers.6.self_attention.linear_proj,decoder.layers.6.self_attention.core_attention,decoder.layers.6.self_attention.linear_qkv'
export SWIFT_GKD_DLOGITS_PARAMETER_PATTERNS=''

echo "GKD Layer 6 common-dLogits backward: mode=${_gkd_l6_mode}, dir=${_gkd_l6_dir}"
echo 'Run the original Base GKD command with train_iters=1, micro_batch_size=1, global_batch_size=1, TP=PP=CP=1.'

unset _gkd_l6_mode _gkd_l6_dir _gkd_l6_tag
