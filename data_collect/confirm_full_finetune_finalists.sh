#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../.." && pwd)
diann_root=${DIANN_FINETUNE_ROOT:-"$workspace_root/../diann-fintune"}
carafe_root=${CARAFE2_ROOT:-"$workspace_root/carafe2-semi"}
diann_python=${DIANN_PYTHON:-python3}
carafe_python=${CARAFE2_PYTHON:-python3}
experiment_root=${1:-"${FINETUNE_SWEEP_OUTPUT_ROOT:-$workspace_root/finetune_output}/full_multiorganism_finetune_sweep_v1"}
prepared="$experiment_root/prepared"
diann_out="$experiment_root/diann"
carafe_out="$experiment_root/carafe2"
analysis="$experiment_root/analysis"
finalists="$experiment_root/finalists"
mkdir -p "$experiment_root/logs" "$analysis"

[[ -s "$finalists/diann_finalists.json" ]] || { echo "Missing DIA-NN finalists; finish broad sweep first" >&2; exit 1; }
[[ -s "$finalists/carafe2_finalists.json" ]] || { echo "Missing Carafe2 finalists; finish broad sweep first" >&2; exit 1; }

(
  flock -n 9 || { echo "DIA-NN finalist confirmation is already active"; exit 0; }
  export CUDA_VISIBLE_DEVICES=${DIANN_CUDA_DEVICE:-0}
  "$diann_python" "$script_dir/run_full_finetune_campaign.py" \
    --model diann --prepared-dir "$prepared" --output-dir "$diann_out/confirmation" \
    --configuration-map "$finalists/diann_finalists.json" --seeds 2026,2027,2028,2029,2030,2031,2032,2033 \
    --python "$diann_python" --device cuda \
    --wandb-config "$diann_root/configs/wandb.json" \
    --wandb-group full-multiorganism-confirmation-v1
) 9>"$diann_out/.confirmation.lock" > >(tee "$experiment_root/logs/diann_confirmation.log") 2>&1 &
diann_pid=$!

(
  flock -n 9 || { echo "Carafe2 finalist confirmation is already active"; exit 0; }
  export CUDA_VISIBLE_DEVICES=${CARAFE2_CUDA_DEVICE:-1}
  "$carafe_python" \
    "$script_dir/run_full_finetune_campaign.py" \
    --model carafe2 --prepared-dir "$prepared" --output-dir "$carafe_out/confirmation" \
    --configuration-map "$finalists/carafe2_finalists.json" --seeds 2026,2027,2028,2029,2030,2031,2032,2033 \
    --python "$carafe_python" --device gpu \
    --wandb-config "$carafe_root/configs/wandb.json" \
    --wandb-group full-multiorganism-confirmation-v1
) 9>"$carafe_out/.confirmation.lock" > >(tee "$experiment_root/logs/carafe2_confirmation.log") 2>&1 &
carafe_pid=$!

wait "$diann_pid"
wait "$carafe_pid"

"$carafe_python" \
  "$script_dir/analyze_full_finetune_sweep.py" \
  --diann-dir "$diann_out" --carafe2-dir "$carafe_out" --output-dir "$analysis"
"$carafe_python" \
  "$script_dir/evaluate_full_finetune_winners.py" \
  --analysis-dir "$analysis" --prepared-dir "$prepared"
"$carafe_python" \
  "$script_dir/analyze_full_finetune_sweep.py" \
  --diann-dir "$diann_out" --carafe2-dir "$carafe_out" --output-dir "$analysis"

echo "Final confirmation and held-out test complete: $analysis"
