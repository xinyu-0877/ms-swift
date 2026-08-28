#!/usr/bin/env bash
# Source before the one-step Base GKD command:
# source scripts/gkd_fp32_residual_forward_env.sh DIR TAG
# The training command must also pass:
# --megatron_extra_kwargs '{"fp32_residual_connection": true}'

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'This helper must be sourced, not executed.' >&2
    exit 2
fi

_gkd_fp32_residual_dir=${1:-}
_gkd_fp32_residual_tag=${2:-}
if [[ -z "${_gkd_fp32_residual_dir}" || -z "${_gkd_fp32_residual_tag}" ]]; then
    echo 'Usage: source scripts/gkd_fp32_residual_forward_env.sh DIR TAG' >&2
    return 2
fi

_gkd_fp32_residual_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${_gkd_fp32_residual_script_dir}/gkd_attention_forward_env.sh" \
    "${_gkd_fp32_residual_dir}" \
    "${_gkd_fp32_residual_tag}" \
    layers
export SWIFT_GKD_REQUIRE_FP32_RESIDUAL=1

echo 'FP32 residual runtime validation enabled.'
echo 'Add --megatron_extra_kwargs '\''{"fp32_residual_connection": true}'\'' to the Base GKD command.'

unset _gkd_fp32_residual_dir _gkd_fp32_residual_tag _gkd_fp32_residual_script_dir
