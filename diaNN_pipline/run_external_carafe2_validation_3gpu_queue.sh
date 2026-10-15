#!/usr/bin/env bash
# Create all external-Carafe2 validation libraries strictly sequentially, while
# each individual library prediction uses all three GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$ROOT/run_external_carafe2_rt_search.sh"
MACHINE_CONFIG="$ROOT/config/carafe_machines/gpu_pc.local.cfg"
SHARE_ROOT="./shared_data/carafe_share/2_5_0"
SOURCE_VALIDATION_ROOT="./shared_data/carafe2_finetune_out/carafe2_source_specific_3target_vs_combined_v1/prepared_targets"
GPU_LIST="0,1,2"
RUN_LABEL="carafe2_3gpu_batch256_$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage: bash run_external_carafe2_validation_3gpu_queue.sh [options]

Runs all 15 planned external-Carafe2 predicted-library jobs one after another.
Every individual job uses all selected GPUs for chunk prediction. Fine-tuning and
DIA-NN Step 5 are not run. The failed C. elegans source-overall candidate is
included as a fresh job with a new search ID.

Options:
  --gpus <0,1,2>          Comma-separated GPUs for every job (default: 0,1,2)
  --run-label <safe-name> Unique queue/job prefix (default: timestamped)
EOF
}

while (($#)); do
  case "$1" in
    --gpus) GPU_LIST="${2:-}"; shift 2 ;;
    --run-label) RUN_LABEL="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'ERROR: --gpus must be a comma-separated GPU list' >&2; exit 2; }
[[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'ERROR: unsafe --run-label' >&2; exit 2; }
[[ -x "$LAUNCHER" ]] || { echo "ERROR: launcher missing: $LAUNCHER" >&2; exit 2; }
[[ -f "$MACHINE_CONFIG" ]] || { echo "ERROR: machine config missing: $MACHINE_CONFIG" >&2; exit 2; }
[[ -d "$SOURCE_VALIDATION_ROOT" ]] || { echo "ERROR: source validation root missing: $SOURCE_VALIDATION_ROOT" >&2; exit 2; }

QUEUE_DIR="$SHARE_ROOT/external_carafe2_queues/$RUN_LABEL"
[[ ! -e "$QUEUE_DIR" ]] || { echo "ERROR: Queue already exists: $QUEUE_DIR" >&2; exit 2; }
mkdir -p "$QUEUE_DIR/logs"
MASTER_LOG="$QUEUE_DIR/queue.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

timestamp() { date '+%F %T'; }
status_value() {
  python3 - "$1" "$2" <<'PY'
import json, pathlib, sys
try:
    print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get(sys.argv[2], ""))
except (OSError, ValueError, TypeError):
    print("")
PY
}

# Protect the host and VRAM: this queue must be the only Carafe prediction on
# the selected GPUs. A live foreign job is not killed or modified.
assert_gpus_idle() {
  local status_file status pid visible
  while IFS= read -r status_file; do
    status="$(status_value "$status_file" status)"
    visible="$(status_value "$status_file" cuda_visible_devices)"
    pid="$(status_value "$status_file" pid)"
    [[ "$status" == running ]] || continue
    [[ ",$visible," == *",${GPU_LIST%%,*},"* || ",$GPU_LIST," == *",$visible,"* ]] || continue
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: Active external-Carafe job occupies selected GPU(s): $(dirname "$(dirname "$status_file")")" >&2
      echo 'Wait for it to finish before starting this sequential queue.' >&2
      exit 1
    fi
  done < <(find "$SHARE_ROOT" -path '*/carafe/balanced/external_*/meta/status.json' -type f -print)
}

# Fields: experiment|model-id|role|validation-root (empty = pooled partition).
JOBS=(
  'combined_md48_cluster|carafe2rt_709a44ce0f5c|pooled_celegans_overall|'
  'carafe_trainingdateta_human_source_human|carafe2rt_709a44ce0f5c|pooled_human_overall|'
  'triplemix_source_a_entrap_neu|carafe2rt_709a44ce0f5c|pooled_source_a_overall|'
  'triplemix_bigger_source_b_entrap_neu|carafe2rt_709a44ce0f5c|pooled_source_b_overall|'
  'combined_md48_cluster|carafe2rt_0f9b11a215e5|pooled_celegans_dimethyl|'
  'carafe_trainingdateta_human_source_human|carafe2rt_0f9b11a215e5|pooled_human_dimethyl|'
  'triplemix_source_a_entrap_neu|carafe2rt_0f9b11a215e5|pooled_source_a_dimethyl|'
  'triplemix_bigger_source_b_entrap_neu|carafe2rt_0f9b11a215e5|pooled_source_b_dimethyl|'
  "combined_md48_cluster|carafe2rt_c1fd5c23319d|source_celegans_overall_retry|$SOURCE_VALIDATION_ROOT"
  "combined_md48_cluster|carafe2rt_d18820021476|source_celegans_dimethyl|$SOURCE_VALIDATION_ROOT"
  "carafe_trainingdateta_human_source_human|carafe2rt_79e32b81ed0a|source_human_overall_dimethyl|$SOURCE_VALIDATION_ROOT"
  "triplemix_source_a_entrap_neu|carafe2rt_64ec74bfccc0|source_source_a_overall|$SOURCE_VALIDATION_ROOT"
  "triplemix_source_a_entrap_neu|carafe2rt_3947baffabfd|source_source_a_dimethyl|$SOURCE_VALIDATION_ROOT"
  "triplemix_bigger_source_b_entrap_neu|carafe2rt_983d67273372|source_source_b_overall|$SOURCE_VALIDATION_ROOT"
  "triplemix_bigger_source_b_entrap_neu|carafe2rt_dbff1bac29b6|source_source_b_dimethyl|$SOURCE_VALIDATION_ROOT"
)

assert_gpus_idle
echo "[$(timestamp)] Starting sequential 3-GPU queue: $RUN_LABEL"
echo "GPUs per job: $GPU_LIST | MS2 prediction batch: ${CARAFE_MS2_PRED_BATCH_SIZE:-from-machine-config}"
echo "Queue log: $MASTER_LOG"

for index in "${!JOBS[@]}"; do
  IFS='|' read -r experiment model_id role validation_root <<< "${JOBS[$index]}"
  ordinal=$(printf '%02d' "$((index + 1))")
  search_id="${RUN_LABEL}_${ordinal}_${role}"
  job_dir="$SHARE_ROOT/$experiment/carafe/balanced/external_${model_id}_${search_id}"
  job_log="$QUEUE_DIR/logs/${ordinal}_${role}.launcher.log"
  [[ ! -e "$job_dir" ]] || { echo "ERROR: Refusing to overwrite existing job: $job_dir" >&2; exit 1; }

  cmd=(bash "$LAUNCHER" --experiment "$experiment" --model-id "$model_id" --search-id "$search_id" --machine-config "$MACHINE_CONFIG" --carafe-device gpu --gpu "$GPU_LIST" --library-only)
  [[ -n "$validation_root" ]] && cmd+=(--validation-root "$validation_root")
  echo "[$(timestamp)] Starting $ordinal/${#JOBS[@]}: $role"
  printf 'Command: '; printf '%q ' "${cmd[@]}"; echo
  "${cmd[@]}" 2>&1 | tee "$job_log"

  final_status="$(status_value "$job_dir/meta/status.json" status)"
  [[ "$final_status" == succeeded ]] || {
    echo "ERROR: $role did not complete successfully (status: ${final_status:-missing})." >&2
    echo "Carafe log: $job_dir/carafe_output/carafe.log" >&2
    exit 1
  }
  echo "[$(timestamp)] Completed $ordinal/${#JOBS[@]}: $role"
done

echo "[$(timestamp)] Queue complete: all ${#JOBS[@]} libraries succeeded."
