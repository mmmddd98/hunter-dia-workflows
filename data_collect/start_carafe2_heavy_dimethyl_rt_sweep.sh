#!/usr/bin/env bash
# Resume-safe Carafe2 Heavy-Dimethyl main sweep; test data are never referenced.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${CARAFE2_HEAVY_DIMETHYL_ROOT:-./shared_data/carafe2_finetune_out/carafe2_heavy_dimethyl_rt_v1}
if [[ ! -f "$root/sweep_manifest.json" ]]; then
  python3 "$script_dir/prepare_heavy_dimethyl_rt_sweep.py" --family carafe2 --arm main --output-dir "$root"
fi
python3 "$script_dir/sweep_template/parallel_gpu_sweep.py" --manifest "$root/sweep_manifest.json" --gpus "${STUDY_GPUS:-0,1,2}" --jobs-per-gpu "${STUDY_JOBS_PER_GPU:-1}" --shared-slot-root "${STUDY_SHARED_SLOT_ROOT:-/tmp/carafe2_heavy_dimethyl_gpu_slots}" "$@"
if [[ " $* " != *" --dry-run "* ]]; then python3 "$script_dir/analyze_heavy_dimethyl_rt_sweep.py" --sweep-dir "$root" --output-dir "$root/analysis"; fi
