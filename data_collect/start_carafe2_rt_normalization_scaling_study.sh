#!/usr/bin/env bash
# Resumable Carafe2-only normalization queue with live terminal progress.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
default_output_root=./shared_data/carafe2_finetune_out
study_root=${CARAFE2_NORMALIZATION_STUDY_ROOT:-"${CARAFE2_FINETUNE_OUTPUT_ROOT:-$default_output_root}/rt_normalization_scaling_v1"}
if (( $# > 0 )) && [[ $1 != -* ]]; then
  study_root=$1
  shift
fi
export STUDY_GPUS=${STUDY_GPUS:-0,1,2}
export STUDY_JOBS_PER_GPU=${STUDY_JOBS_PER_GPU:-2}

exec "${QUEUE_PYTHON:-python3}" "$script_dir/run_rt_normalization_queue.py" \
  --family Carafe2 \
  --carafe2-root "$study_root" \
  --modes "${RT_TARGET_MODES:-global_60min,source_native,anchor_irt,linear,relative_rt,lowess,isotonic,monotonic_spline,calib_rt,kde,chronologer_hi}" \
  "$@"
