#!/usr/bin/env bash
# Freeze a small validation-selected finalist set, then predict it once on test.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sweep_dir=${CARAFE2_HEAVY_DIMETHYL_ROOT:-./shared_data/carafe2_finetune_out/carafe2_heavy_dimethyl_rt_v1}
test_dir=${CARAFE2_HEAVY_DIMETHYL_TEST_ROOT:-"$sweep_dir/test"}

if [[ ! -f "$test_dir/test_manifest.json" ]]; then
  python3 "$script_dir/prepare_heavy_dimethyl_test_evaluation.py" \
    --sweep-dir "$sweep_dir" \
    --output-dir "$test_dir" \
    --top-per-target "${HEAVY_DIMETHYL_TEST_TOP_PER_TARGET:-1}"
fi

python3 "$script_dir/sweep_template/parallel_gpu_sweep.py" \
  --manifest "$test_dir/test_manifest.json" \
  --gpus "${STUDY_GPUS:-0,1,2}" \
  --jobs-per-gpu "${STUDY_JOBS_PER_GPU:-1}" \
  --shared-slot-root "${STUDY_SHARED_SLOT_ROOT:-/tmp/carafe2_heavy_dimethyl_gpu_slots}" \
  "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  python3 "$script_dir/analyze_heavy_dimethyl_final_test.py" --test-evaluation-dir "$test_dir"
fi
