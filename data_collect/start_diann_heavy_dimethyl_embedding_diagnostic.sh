#!/usr/bin/env bash
# Separate DIA-NN embedding-only diagnostic (96 runs); never mixed into main ranking.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${DIANN_HEAVY_DIMETHYL_EMBEDDING_ROOT:-./shared_data/diann_fintune_out/diann_heavy_dimethyl_embedding_v1}
if [[ ! -f "$root/sweep_manifest.json" ]]; then
  python3 "$script_dir/prepare_heavy_dimethyl_rt_sweep.py" --family diann --arm embedding --output-dir "$root"
fi
python3 "$script_dir/sweep_template/parallel_gpu_sweep.py" --manifest "$root/sweep_manifest.json" --gpus "${STUDY_GPUS:-0,1,2}" --jobs-per-gpu "${STUDY_JOBS_PER_GPU:-1}" --shared-slot-root "${STUDY_SHARED_SLOT_ROOT:-/tmp/diann_heavy_embedding_gpu_slots}" "$@"
if [[ " $* " != *" --dry-run "* ]]; then python3 "$script_dir/analyze_heavy_dimethyl_rt_sweep.py" --sweep-dir "$root" --output-dir "$root/analysis"; fi
