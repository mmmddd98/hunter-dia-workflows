#!/usr/bin/env bash
# Resumable DIA-NN-only normalization queue with live terminal progress.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
default_output_root=./shared_data/diann_fintune_out
study_root=${DIANN_NORMALIZATION_STUDY_ROOT:-"${DIANN_FINETUNE_OUTPUT_ROOT:-$default_output_root}/rt_normalization_scaling_v1"}
if (( $# > 0 )) && [[ $1 != -* ]]; then
  study_root=$1
  shift
fi
export STUDY_GPUS=${STUDY_GPUS:-0,1,2}
export STUDY_JOBS_PER_GPU=${STUDY_JOBS_PER_GPU:-2}

exec "${QUEUE_PYTHON:-python3}" "$script_dir/run_rt_normalization_queue.py" \
  --family DIA-NN \
  --diann-root "$study_root" \
  --modes "${RT_TARGET_MODES:-global_60min,source_native,anchor_irt,linear,relative_rt,lowess,isotonic,monotonic_spline,calib_rt,kde,chronologer_hi}" \
  "$@"
