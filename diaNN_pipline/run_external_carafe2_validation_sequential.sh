#!/usr/bin/env bash
# Generate the 15 external-Carafe2 validation libraries strictly one at a time.
# No fine-tuning and no DIA-NN Step 5 are performed here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$ROOT/run_external_carafe2_rt_search.sh"
MACHINE_CONFIG="$ROOT/config/carafe_machines/gpu_pc.local.cfg"
SHARE_ROOT="./shared_data/carafe_share/2_5_0"
SOURCE_VALIDATION_ROOT="./shared_data/carafe2_finetune_out/carafe2_source_specific_3target_vs_combined_v1/prepared_targets"
GPU=0
POLL_SECONDS=20
RUN_LABEL="manual_sequential_$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage: bash run_external_carafe2_validation_sequential.sh [options]

Runs the 15 planned predicted-library jobs one after another on exactly one GPU.
It waits for an already active external-Carafe2 job on that GPU before starting.
Completed jobs with the same job ID are skipped; an incomplete existing job stops
the runner, so that it is never overwritten accidentally.

Options:
  --gpu <index>           GPU to use (default: 0)
  --run-label <safe-name> Name for the central sequential-runner log directory
  --poll-seconds <n>      Wait interval while another job is active (default: 20)
EOF
}

while (($#)); do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --run-label) RUN_LABEL="${2:-}"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'ERROR: --gpu must be a non-negative integer' >&2; exit 2; }
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: --poll-seconds must be positive' >&2; exit 2; }
[[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'ERROR: unsafe --run-label' >&2; exit 2; }
[[ -x "$LAUNCHER" ]] || { echo "ERROR: launcher missing: $LAUNCHER" >&2; exit 2; }
[[ -f "$MACHINE_CONFIG" ]] || { echo "ERROR: machine config missing: $MACHINE_CONFIG" >&2; exit 2; }
[[ -d "$SOURCE_VALIDATION_ROOT" ]] || { echo "ERROR: source validation root missing: $SOURCE_VALIDATION_ROOT" >&2; exit 2; }

RUNNER_LOG_DIR="$SHARE_ROOT/external_carafe2_sequential_runs/$RUN_LABEL"
mkdir -p "$RUNNER_LOG_DIR"
MASTER_LOG="$RUNNER_LOG_DIR/runner.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

timestamp() { date '+%F %T'; }
status_value() {
  local file="$1" key="$2"
  python3 - "$file" "$key" <<'PY'
import json, pathlib, sys
try:
    print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get(sys.argv[2], ""))
except (OSError, ValueError, TypeError):
    print("")
PY
}

# Do not place a second job on the chosen GPU. This also recognizes a manually
# started job, such as the currently running C. elegans source-specific model.
wait_for_gpu_to_be_free() {
  local status_file status gpu pid job
  while :; do
    status_file=''
    while IFS= read -r candidate; do
      status="$(status_value "$candidate" status)"
      gpu="$(status_value "$candidate" cuda_visible_devices)"
      pid="$(status_value "$candidate" pid)"
      [[ "$status" == running && "$gpu" == "$GPU" ]] || continue
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        status_file="$candidate"
        break
      fi
      echo "ERROR: Stale running status (PID $pid is absent): $candidate" >&2
      echo 'Inspect the job before continuing; it will not be overwritten.' >&2
      exit 1
    done < <(find "$SHARE_ROOT" -path '*/carafe/balanced/external_*/meta/status.json' -type f -print)

    [[ -z "$status_file" ]] && return
    job="$(dirname "$(dirname "$status_file")")"
    echo "[$(timestamp)] GPU $GPU is in use by $job; waiting ${POLL_SECONDS}s ..."
    sleep "$POLL_SECONDS"
  done
}

# Fields: experiment|model ID|stable manual search ID|role|validation root.
# An empty validation-root uses the canonical pooled-model validation partition.
JOBS=(
  'combined_md48_cluster|carafe2rt_709a44ce0f5c|manual_celegans_pooled_overall_v1|pooled_celegans_overall|'
  'carafe_trainingdateta_human_source_human|carafe2rt_709a44ce0f5c|manual_human_pooled_overall_v1|pooled_human_overall|'
  'triplemix_source_a_entrap_neu|carafe2rt_709a44ce0f5c|manual_source_a_pooled_overall_v1|pooled_source_a_overall|'
  'triplemix_bigger_source_b_entrap_neu|carafe2rt_709a44ce0f5c|manual_source_b_pooled_overall_v1|pooled_source_b_overall|'
  'combined_md48_cluster|carafe2rt_0f9b11a215e5|manual_celegans_pooled_dimethyl_v1|pooled_celegans_dimethyl|'
  'carafe_trainingdateta_human_source_human|carafe2rt_0f9b11a215e5|manual_human_pooled_dimethyl_v1|pooled_human_dimethyl|'
  'triplemix_source_a_entrap_neu|carafe2rt_0f9b11a215e5|manual_source_a_pooled_dimethyl_v1|pooled_source_a_dimethyl|'
  'triplemix_bigger_source_b_entrap_neu|carafe2rt_0f9b11a215e5|manual_source_b_pooled_dimethyl_v1|pooled_source_b_dimethyl|'
  "combined_md48_cluster|carafe2rt_c1fd5c23319d|manual_celegans_source_overall_v1|source_celegans_overall|$SOURCE_VALIDATION_ROOT"
  "combined_md48_cluster|carafe2rt_d18820021476|manual_celegans_source_dimethyl_v1|source_celegans_dimethyl|$SOURCE_VALIDATION_ROOT"
  "carafe_trainingdateta_human_source_human|carafe2rt_79e32b81ed0a|manual_human_source_overall_dimethyl_v1|source_human_overall_dimethyl|$SOURCE_VALIDATION_ROOT"
  "triplemix_source_a_entrap_neu|carafe2rt_64ec74bfccc0|manual_source_a_source_overall_v1|source_source_a_overall|$SOURCE_VALIDATION_ROOT"
  "triplemix_source_a_entrap_neu|carafe2rt_3947baffabfd|manual_source_a_source_dimethyl_v1|source_source_a_dimethyl|$SOURCE_VALIDATION_ROOT"
  "triplemix_bigger_source_b_entrap_neu|carafe2rt_983d67273372|manual_source_b_source_overall_v1|source_source_b_overall|$SOURCE_VALIDATION_ROOT"
  "triplemix_bigger_source_b_entrap_neu|carafe2rt_dbff1bac29b6|manual_source_b_source_dimethyl_v1|source_source_b_dimethyl|$SOURCE_VALIDATION_ROOT"
)

echo "[$(timestamp)] Sequential external-Carafe2 library runner"
echo "GPU: $GPU | central log: $MASTER_LOG"
echo "Each Carafe log: <job>/carafe_output/carafe.log"

for index in "${!JOBS[@]}"; do
  IFS='|' read -r experiment model_id search_id role validation_root <<< "${JOBS[$index]}"
  job_dir="$SHARE_ROOT/$experiment/carafe/balanced/external_${model_id}_${search_id}"
  status_file="$job_dir/meta/status.json"

  if [[ -e "$job_dir" ]]; then
    existing_status="$(status_value "$status_file" status)"
    if [[ "$existing_status" == completed ]]; then
      echo "[$(timestamp)] $((index + 1))/$((${#JOBS[@]})): already completed, skip: $role"
      continue
    fi
    if [[ "$existing_status" == running ]]; then
      echo "[$(timestamp)] $((index + 1))/$((${#JOBS[@]})): active existing job, waiting: $role"
      wait_for_gpu_to_be_free
      final_status="$(status_value "$status_file" status)"
      [[ "$final_status" == completed ]] || {
        echo "ERROR: Existing job did not finish successfully ($final_status): $job_dir" >&2
        exit 1
      }
      echo "[$(timestamp)] Existing job completed: $role"
      continue
    fi
    echo "ERROR: Existing incomplete job ($existing_status): $job_dir" >&2
    echo 'Use a new search-id only after inspecting its log; no files were changed.' >&2
    exit 1
  fi

  wait_for_gpu_to_be_free
  cmd=(bash "$LAUNCHER" --experiment "$experiment" --model-id "$model_id" --search-id "$search_id" --machine-config "$MACHINE_CONFIG" --carafe-device gpu --gpu "$GPU" --library-only)
  [[ -n "$validation_root" ]] && cmd+=(--validation-root "$validation_root")
  echo "[$(timestamp)] Starting $((index + 1))/$((${#JOBS[@]})) on GPU $GPU: $role"
  printf 'Command: '; printf '%q ' "${cmd[@]}"; echo
  "${cmd[@]}"

  final_status="$(status_value "$status_file" status)"
  [[ "$final_status" == completed ]] || {
    echo "ERROR: Job ended without successful completion ($final_status): $job_dir" >&2
    exit 1
  }
  echo "[$(timestamp)] Completed: $role"
done

echo "[$(timestamp)] All 15 external-Carafe2 libraries completed successfully."
