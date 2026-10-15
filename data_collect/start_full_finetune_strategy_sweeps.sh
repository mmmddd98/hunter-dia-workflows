#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../.." && pwd)
diann_root=${DIANN_FINETUNE_ROOT:-"$workspace_root/../diann-fintune"}
carafe_root=${CARAFE2_ROOT:-"$workspace_root/carafe2-semi"}
diann_python=${DIANN_PYTHON:-python3}
carafe_python=${CARAFE2_PYTHON:-python3}
diann_model_dir=${DIANN_MODEL_DIR:-"$workspace_root/../diann_sc/DIA-NN-2.5.0/diann-2.5.0/models"}
experiment_root=${1:-"${FINETUNE_SWEEP_OUTPUT_ROOT:-$workspace_root/finetune_output}/full_multiorganism_finetune_sweep_v1"}
model=${2:-both}
if [[ "$model" != "both" && "$model" != "diann" && "$model" != "carafe2" ]]; then
  echo "Usage: $0 [experiment_root] [both|diann|carafe2]" >&2
  exit 2
fi
hours=${SWEEP_HOURS:-240}
max_configs=${SWEEP_CONFIGS_PER_MODEL:-120}
prepared="$experiment_root/prepared"
diann_out="$experiment_root/diann"
carafe_out="$experiment_root/carafe2"
analysis="$experiment_root/analysis"
mkdir -p "$experiment_root/logs" "$diann_out" "$carafe_out" "$analysis"

# The sweep needs only scalar metrics and configurations in W&B.  Keep all
# models, prediction tables, reports, figures, source snapshots, and console
# output local to avoid multiplying storage by hundreds of runs.
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
export WANDB_CONSOLE=off

if [[ ! -f "$prepared/sweep_data_manifest.json" ]]; then
  "$carafe_python" \
    "$script_dir/prepare_full_finetune_sweep.py" \
    --output-dir "$prepared"
fi

run_diann() {
  (
    flock -n 9 || { echo "DIA-NN full-data sweep is already active"; exit 0; }
    export CUDA_VISIBLE_DEVICES=${DIANN_CUDA_DEVICE:-0}
    "$diann_python" "$script_dir/run_full_finetune_campaign.py" \
      --model diann --prepared-dir "$prepared" --output-dir "$diann_out/controls" \
      --configuration-map "$diann_root/configs/rt_full_multiorganism_controls.json" \
      --seeds 2024,2025 --python "$diann_python" --device cuda \
      --wandb-config "$diann_root/configs/wandb.json" \
      --wandb-group full-multiorganism-controls-v1
    completed=$(find "$diann_out" -path '*/trial.*/seed.2025/validation_metrics.json' -type f | wc -l)
    remaining=$((max_configs - completed))
    if (( remaining > 0 )); then
      resume=()
      [[ -s "$diann_out/sweep_id.txt" ]] && resume=(--sweep-id "$(tr -d '\n' < "$diann_out/sweep_id.txt")")
      "$diann_python" "$diann_root/run_targeted_wandb_sweep.py" \
        --hours "$hours" --max-runs "$remaining" --seeds 2024,2025 \
        --in-dir "$prepared/train" --validation-dir "$prepared/validation" \
        --model-dir "$diann_model_dir" \
        --output-dir "$diann_out" \
        --sweep-config "$diann_root/configs/rt_full_multiorganism_strategy_sweep.json" \
        --wandb-config "$diann_root/configs/wandb.json" \
        --python "$diann_python" "${resume[@]}"
    fi
  ) 9>"$diann_out/.broad_sweep.lock"
}

run_carafe2() {
  (
    flock -n 9 || { echo "Carafe2 full-data sweep is already active"; exit 0; }
    export CUDA_VISIBLE_DEVICES=${CARAFE2_CUDA_DEVICE:-1}
    "$carafe_python" \
      "$script_dir/run_full_finetune_campaign.py" \
      --model carafe2 --prepared-dir "$prepared" --output-dir "$carafe_out/controls" \
      --configuration-map "$carafe_root/configs/carafe2_rt_full_multiorganism_controls.json" \
      --seeds 2024,2025 --python "$carafe_python" --device gpu \
      --wandb-config "$carafe_root/configs/wandb.json" \
      --wandb-group full-multiorganism-controls-v1
    completed=$(find "$carafe_out" -path '*/trial.*/seed.2025/validation_metrics.json' -type f | wc -l)
    remaining=$((max_configs - completed))
    if (( remaining > 0 )); then
      resume=()
      [[ -s "$carafe_out/sweep_id.txt" ]] && resume=(--sweep-id "$(tr -d '\n' < "$carafe_out/sweep_id.txt")")
      "$carafe_python" \
        "$carafe_root/run_carafe2_wandb_sweep.py" \
        --hours "$hours" --max-runs "$remaining" --seeds 2024,2025 \
        --in-dir "$prepared/train" --validation-dir "$prepared/validation" \
        --output-dir "$carafe_out" --tf-type rt --device gpu \
        --sweep-config "$carafe_root/configs/carafe2_rt_full_multiorganism_strategy_sweep.json" \
        --wandb-config "$carafe_root/configs/wandb.json" \
        --python "$carafe_python" "${resume[@]}"
    fi
  ) 9>"$carafe_out/.broad_sweep.lock"
}

if [[ "$model" == "diann" ]]; then
  run_diann 2>&1 | tee "$experiment_root/logs/diann_broad.log"
  exit "${PIPESTATUS[0]}"
fi
if [[ "$model" == "carafe2" ]]; then
  run_carafe2 2>&1 | tee "$experiment_root/logs/carafe2_broad.log"
  exit "${PIPESTATUS[0]}"
fi

run_diann > >(tee "$experiment_root/logs/diann_broad.log") 2>&1 &
diann_pid=$!
run_carafe2 > >(tee "$experiment_root/logs/carafe2_broad.log") 2>&1 &
carafe_pid=$!
wait "$diann_pid"
wait "$carafe_pid"

"$carafe_python" \
  "$script_dir/analyze_full_finetune_sweep.py" \
  --diann-dir "$diann_out" --carafe2-dir "$carafe_out" --output-dir "$analysis"
"$carafe_python" \
  "$script_dir/select_full_finetune_finalists.py" \
  --analysis-dir "$analysis" --output-dir "$experiment_root/finalists" --top 12

echo "Broad sweeps complete. Plots: $analysis"
echo "Next: $script_dir/confirm_full_finetune_finalists.sh $experiment_root"
