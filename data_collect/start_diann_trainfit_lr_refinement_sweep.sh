#!/usr/bin/env bash
# Resume-safe launcher: runs the focused DIA-NN sweep, then its full analysis.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
study_root=${DIANN_LR_REFINEMENT_ROOT:-./shared_data/diann_fintune_out/diann_trainfit_lr_refinement_v1}

if [[ ! -f "$study_root/sweep_manifest.json" ]]; then
  python3 "$script_dir/prepare_rt_lr_refinement_sweep.py" \
    --family diann --output-dir "$study_root"
fi

python3 "$script_dir/sweep_template/parallel_gpu_sweep.py" \
  --manifest "$study_root/sweep_manifest.json" \
  --gpus "${STUDY_GPUS:-0,1,2}" \
  --jobs-per-gpu "${STUDY_JOBS_PER_GPU:-1}" \
  --shared-slot-root "${STUDY_SHARED_SLOT_ROOT:-/tmp/diann_trainfit_lr_refinement_gpu_slots}" "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  python3 "$script_dir/analyze_diann_trainfit_sweep.py" \
    --sweep-dir "$study_root" --output-dir "$study_root/analysis"
fi
