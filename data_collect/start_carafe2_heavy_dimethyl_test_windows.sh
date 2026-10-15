#!/usr/bin/env bash
# Start each frozen finalist/control test prediction pack in a tmux window.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../.." && pwd)
sweep_dir=${CARAFE2_HEAVY_DIMETHYL_ROOT:-./shared_data/carafe2_finetune_out/carafe2_heavy_dimethyl_rt_v1}
test_dir=${CARAFE2_HEAVY_DIMETHYL_TEST_ROOT:-"$sweep_dir/test"}
session=${HEAVY_DIMETHYL_TEST_TMUX_SESSION:-carafe2_heavy_dimethyl_test}

if ! command -v tmux >/dev/null; then
  echo "tmux is required for separate progress-bar windows." >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session '$session' already exists; attach with: tmux attach -t $session" >&2
  exit 1
fi
if [[ ! -f "$test_dir/test_manifest.json" ]]; then
  python3 "$script_dir/prepare_heavy_dimethyl_test_evaluation.py" \
    --sweep-dir "$sweep_dir" \
    --output-dir "$test_dir" \
    --top-per-target "${HEAVY_DIMETHYL_TEST_TOP_PER_TARGET:-1}"
fi
python3 "$script_dir/split_heavy_dimethyl_test_manifests.py" --test-evaluation-dir "$test_dir"

mapfile -t window_specs < <(python3 - "$test_dir/window_manifests/index.json" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8")):
    print(f"{item['name']}\t{item['manifest']}")
PY
)
if [[ ${#window_specs[@]} -eq 0 ]]; then
  echo "No finalist manifests were generated." >&2
  exit 1
fi

start_window() {
  local name=$1 manifest=$2 command
  command="cd '$workspace_root' && python3 '$script_dir/sweep_template/parallel_gpu_sweep.py' --manifest '$manifest' --gpus '${STUDY_GPUS:-0,1,2}' --jobs-per-gpu '${STUDY_JOBS_PER_GPU:-1}' --shared-slot-root '${STUDY_SHARED_SLOT_ROOT:-/tmp/carafe2_heavy_dimethyl_gpu_slots}'; status=\$?; printf '\\nFinished %s (exit %s). Keep this window open for the log.\\n' '$name' \"\$status\"; exec bash"
  if [[ -z ${window_started:-} ]]; then
    tmux new-session -d -s "$session" -n "$name" "$command"
    window_started=1
  else
    tmux new-window -d -t "$session:" -n "$name" "$command"
  fi
}

for spec in "${window_specs[@]}"; do
  IFS=$'\t' read -r name manifest <<< "$spec"
  start_window "$name" "$manifest"
done

echo "Started ${#window_specs[@]} tmux windows in '$session'."
echo "Attach with: tmux attach -t $session"
echo "Each window has one finalist plus its three-seed matched control and shows its own runner progress bar."
echo "After both windows finish, run: python3 $script_dir/analyze_heavy_dimethyl_final_test.py --test-evaluation-dir $test_dir"
