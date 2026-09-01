#!/usr/bin/env bash
# Configure the ordinary-SFT Layer-0 A-G forward capture.
# Usage: source scripts/sft_attention_forward_env.sh DIR TAG [LAYER_TARGET]

_sft_attention_dir=${1:-}
_sft_attention_tag=${2:-}
_sft_attention_layer=${3:-decoder.layers.0}

if [[ -z "${_sft_attention_dir}" || -z "${_sft_attention_tag}" ]]; then
    echo 'Usage: source scripts/sft_attention_forward_env.sh DIR TAG [LAYER_TARGET]' >&2
    return 2 2>/dev/null || exit 2
fi

unset SWIFT_GKD_ATTENTION_FORWARD_TRACE
unset SWIFT_GKD_ATTENTION_FORWARD_DIR
unset SWIFT_GKD_ATTENTION_FORWARD_TAG

export SWIFT_SFT_ATTENTION_FORWARD_TRACE=1
export SWIFT_SFT_ATTENTION_FORWARD_DIR="${_sft_attention_dir}"
export SWIFT_SFT_ATTENTION_FORWARD_TAG="${_sft_attention_tag}"
export SWIFT_SFT_ATTENTION_FORWARD_LAYER_TARGET="${_sft_attention_layer}"
export SWIFT_SFT_ATTENTION_FORWARD_STEP=0
export SWIFT_SFT_ATTENTION_FORWARD_MICRO_BATCH=0

echo "SFT Layer-0 A-G trace enabled: dir=${SWIFT_SFT_ATTENTION_FORWARD_DIR}, "\
     "tag=${SWIFT_SFT_ATTENTION_FORWARD_TAG}, "\
     "layer=${SWIFT_SFT_ATTENTION_FORWARD_LAYER_TARGET}, step=0, micro_batch=0"

unset _sft_attention_dir _sft_attention_tag _sft_attention_layer
