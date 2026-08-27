#!/usr/bin/env bash
# Capture: source scripts/gkd_step0_all_layers_backward_env.sh capture DIR TAG
# Replay:  source scripts/gkd_step0_all_layers_backward_env.sh replay DIR TAG

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'This helper must be sourced, not executed.' >&2
    exit 2
fi

_gkd_bw_mode=${1:-}
_gkd_bw_dir=${2:-}
_gkd_bw_tag=${3:-}
if [[ ! "${_gkd_bw_mode}" =~ ^(capture|replay)$ || -z "${_gkd_bw_dir}" || -z "${_gkd_bw_tag}" ]]; then
    echo 'Usage: source scripts/gkd_step0_all_layers_backward_env.sh capture|replay DIR TAG' >&2
    return 2
fi

unset SWIFT_GKD_JSD_ISOLATION_MODE
unset SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE
unset SWIFT_GKD_FLASH_ISOLATION_MODE
unset SWIFT_GKD_FLASH_BACKWARD_ISOLATION
unset SWIFT_GKD_SWIGLU_ISOLATION_MODE
unset SWIFT_GKD_FC1_ISOLATION_MODE
unset SWIFT_GKD_MLP_MERGE_MODE
unset SWIFT_GKD_ATTENTION_FORWARD_TRACE
unset SWIFT_GKD_SELF_ATTENTION_FORWARD_MODE
unset SWIFT_GKD_MICROBATCH_BACKWARD_TRACE
unset SWIFT_GKD_PRECLIP_GRAD_STEPS
unset SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS

_gkd_bw_patterns='output_layer,decoder.final_layernorm'
for ((_gkd_bw_layer=27; _gkd_bw_layer>=0; _gkd_bw_layer--)); do
    _gkd_bw_patterns+=",decoder.layers.${_gkd_bw_layer}"
done

export SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP=0
export SWIFT_GKD_ALIGNMENT_DEBUG_STEPS=1
export SWIFT_GKD_ALIGNMENT_DEBUG_DIR="${_gkd_bw_dir}/alignment_${_gkd_bw_tag}"
export SWIFT_GKD_ALIGNMENT_SAMPLE_COUNT=32
export SWIFT_GKD_OPERATOR_DEBUG=0
export SWIFT_GKD_BACKWARD_DEBUG=0

export SWIFT_GKD_DLOGITS_ISOLATION_MODE="${_gkd_bw_mode}"
export SWIFT_GKD_DLOGITS_ISOLATION_DIR="${_gkd_bw_dir}"
export SWIFT_GKD_DLOGITS_ISOLATION_TAG="${_gkd_bw_tag}"
export SWIFT_GKD_DLOGITS_ISOLATION_STEP=0
export SWIFT_GKD_DLOGITS_ISOLATION_MICRO_BATCH=0
export SWIFT_GKD_DLOGITS_BACKWARD_PATTERNS="${_gkd_bw_patterns}"
export SWIFT_GKD_DLOGITS_PARAMETER_PATTERNS=''

echo "GKD step-0 all-layer common-dLogits backward: mode=${_gkd_bw_mode}, dir=${_gkd_bw_dir}"
echo 'Run the original Base GKD command with train_iters=1, micro_batch_size=1, global_batch_size=1, TP=PP=CP=1.'

unset _gkd_bw_mode _gkd_bw_dir _gkd_bw_tag _gkd_bw_patterns _gkd_bw_layer
