#!/usr/bin/env bash
# Joint, resumable DIA-NN + Carafe2 normalization queue with a live terminal panel.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../.." && pwd)
diann_root=${1:-"${DIANN_FINETUNE_OUTPUT_ROOT:-$workspace_root/../diann_finetune_out}/rt_normalization_scaling_v1"}
carafe2_root=${2:-"${CARAFE2_FINETUNE_OUTPUT_ROOT:-$workspace_root/finetune_output}/rt_normalization_scaling_v1"}
[[ $# -eq 0 ]] || shift
[[ $# -eq 0 ]] || shift

exec "${QUEUE_PYTHON:-python3}" "$script_dir/run_rt_normalization_queue.py" \
  --diann-root "$diann_root" \
  --carafe2-root "$carafe2_root" \
  --modes "${RT_TARGET_MODES:-global_60min,source_native,anchor_irt,linear,relative_rt,lowess,isotonic,monotonic_spline,calib_rt,kde,chronologer_hi}" \
  "$@"
