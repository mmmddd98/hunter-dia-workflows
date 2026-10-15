#!/usr/bin/env bash
# Validation-only, resumable DIA-NN source-specific versus combined sweep.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
study_root=${DIANN_SOURCE_SPECIFIC_SWEEP_ROOT:-./shared_data/diann_fintune_out/diann_source_specific_3target_vs_combined_v5}

if [[ ! -f "$study_root/sweep_manifest.json" ]]; then
  python3 "$script_dir/prepare_diann_source_specific_sweep.py" --output-dir "$study_root"
fi

python3 "$script_dir/sweep_template/parallel_gpu_sweep.py" \
  --manifest "$study_root/sweep_manifest.json" \
  --gpus "${STUDY_GPUS:-0,1,2}" \
  --jobs-per-gpu "${STUDY_JOBS_PER_GPU:-2}" \
  --shared-slot-root "${STUDY_SHARED_SLOT_ROOT:-/tmp/diann_source_specific_sweep_gpu_slots}" "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  python3 "$script_dir/analyze_diann_source_specific_sweep.py" --study-root "$study_root"
  python3 "$script_dir/register_diann_lr_refinement_comparators.py" \
    --output "$study_root/analysis/historical_combined_registry.json"
fi
