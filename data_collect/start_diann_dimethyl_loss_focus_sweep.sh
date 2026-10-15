#!/usr/bin/env bash
# Resume-safe launcher for the 288-run Dimethyl-first, fixed-step DIA-NN sweep.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sweep_root=${DIANN_DIMETHYL_LOSS_FOCUS_ROOT:-./shared_data/diann_fintune_out/diann_dimethyl_loss_focus_fixed_steps_v2_chronologer_anchor}

if [[ ! -f "$sweep_root/sweep_manifest.json" ]]; then
  python3 "$script_dir/prepare_diann_dimethyl_loss_focus_sweep.py" --output-dir "$sweep_root"
fi

python3 "$script_dir/sweep_template/parallel_gpu_sweep.py" \
  --manifest "$sweep_root/sweep_manifest.json" \
  --gpus "${STUDY_GPUS:-0,1,2}" \
  --jobs-per-gpu "${STUDY_JOBS_PER_GPU:-1}" \
  --shared-slot-root "${STUDY_SHARED_SLOT_ROOT:-/tmp/diann_dimethyl_loss_focus_gpu_slots}" \
  "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  python3 "$script_dir/analyze_diann_dimethyl_loss_focus_sweep.py" \
    --sweep-dir "$sweep_root" --output-dir "$sweep_root/analysis"
  python3 "$script_dir/plot_diann_dimethyl_loss_focus_sweep.py" \
    --analysis-dir "$sweep_root/analysis"
fi
