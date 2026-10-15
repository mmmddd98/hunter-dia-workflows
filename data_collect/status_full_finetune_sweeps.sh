#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../.." && pwd)
experiment_root=${1:-"${FINETUNE_SWEEP_OUTPUT_ROOT:-$workspace_root/finetune_output}/full_multiorganism_finetune_sweep_v1"}
model=${2:-both}
if [[ "$model" != "both" && "$model" != "diann" && "$model" != "carafe2" ]]; then
  echo "Usage: $0 [experiment_root] [both|diann|carafe2]" >&2
  exit 2
fi
count_files() {
  local root=$1 pattern=$2
  if [[ -d "$root" ]]; then
    find "$root" -path "$pattern" -type f 2>/dev/null | wc -l
  else
    echo 0
  fi
}
models=(diann carafe2)
if [[ "$model" != "both" ]]; then
  models=("$model")
fi
for current_model in "${models[@]}"; do
  root="$experiment_root/$current_model"
  controls=$(count_files "$root/controls" '*/validation_metrics.json')
  broad=$(count_files "$root" '*/trial.*/seed.*/validation_metrics.json')
  confirmation=$(count_files "$root/confirmation" '*/validation_metrics.json')
  final_tests=$(count_files "$root" '*/final_test_metrics.json')
  printf '%-8s controls=%s/8 broad=%s/240 confirmation=%s/96 final_test=%s/10\n' \
    "$current_model" "$controls" "$broad" "$confirmation" "$final_tests"
done

if [[ -f "$experiment_root/analysis/sweep_configuration_summary.csv" ]]; then
  echo "Analysis: $experiment_root/analysis/sweep_report.html"
fi

ps -eo pid,etimes,cmd | rg \
  'full_finetune|run_targeted_wandb_sweep|run_carafe2_wandb_sweep|train_carafe_rt.py|src/main/resources/py/v2/ai.py' \
  || true
