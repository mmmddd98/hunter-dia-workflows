#!/usr/bin/env bash
# Create all validation libraries for the pooled and source-specific Carafe2 RT finalists.
# The queue always runs the two pooled models first (on every source), followed by
# each source's own validation-selected finalist(s). It deliberately stops before DIA-NN Step 5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$ROOT/run_external_carafe2_rt_search.sh"
GPU_CONFIG="$ROOT/config/carafe_machines/gpu_pc.local.cfg"
SOURCE_VALIDATION_ROOT="./shared_data/carafe2_finetune_out/carafe2_source_specific_3target_vs_combined_v1/prepared_targets"
SHARE_ROOT="./shared_data/carafe_share/2_5_0"

RUN_LABEL="carafe2_finalists_$(date +%Y%m%d_%H%M%S)"
GPU_LIST="0,1,2"
MAX_PARALLEL=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash run_external_carafe2_validation_queue.sh [options]

Generates spectral libraries only (no DIA-NN Step 5) in this fixed order:
  1. pooled Chronologer-HI and Anchor-iRT finalists × all four datasets (8 jobs)
  2. source-specific overall/Dimethyl finalists (7 jobs; Human has one shared model)

Options:
  --run-label <safe-name>  Stable name used in job/search IDs (default: timestamped)
  --gpus <0,1,2>           GPU indices to schedule round-robin (default: 0,1,2)
  --max-parallel <n>        Concurrent library jobs (default: number of requested GPUs)
  --dry-run                 Print the complete 15-job plan without creating jobs

Run persistently, for example:
  nohup bash run_external_carafe2_validation_queue.sh --run-label carafe2_finalists_v1 \
    > /tmp/carafe2_finalists_v1.queue.log 2>&1 &
EOF
}

while (($#)); do
  case "$1" in
    --run-label) RUN_LABEL="${2:-}"; shift 2 ;;
    --gpus) GPU_LIST="${2:-}"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'ERROR: unsafe --run-label' >&2; exit 2; }
[[ -x "$LAUNCHER" ]] || { echo "ERROR: launcher missing: $LAUNCHER" >&2; exit 2; }
[[ -f "$GPU_CONFIG" ]] || { echo "ERROR: GPU config missing: $GPU_CONFIG" >&2; exit 2; }
[[ -d "$SOURCE_VALIDATION_ROOT" ]] || { echo "ERROR: source-specific validation root missing" >&2; exit 2; }

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
(( ${#GPUS[@]} > 0 )) || { echo 'ERROR: --gpus must contain at least one GPU index' >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid GPU index: $gpu" >&2; exit 2; }
done
MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: --max-parallel must be positive' >&2; exit 2; }
(( MAX_PARALLEL <= ${#GPUS[@]} )) || { echo 'ERROR: --max-parallel may not exceed the requested GPU count' >&2; exit 2; }

QUEUE_DIR="$SHARE_ROOT/external_carafe2_queues/$RUN_LABEL"
PLAN="$QUEUE_DIR/plan.tsv"
LEDGER="$QUEUE_DIR/ledger.tsv"

# Fields: phase, experiment, model-id, role, validation-root (empty = canonical root).
JOBS=(
  'pooled|combined_md48_cluster|carafe2rt_709a44ce0f5c|pooled_overall|'
  'pooled|carafe_trainingdateta_human_source_human|carafe2rt_709a44ce0f5c|pooled_overall|'
  'pooled|triplemix_source_a_entrap_neu|carafe2rt_709a44ce0f5c|pooled_overall|'
  'pooled|triplemix_bigger_source_b_entrap_neu|carafe2rt_709a44ce0f5c|pooled_overall|'
  'pooled|combined_md48_cluster|carafe2rt_0f9b11a215e5|pooled_dimethyl|'
  'pooled|carafe_trainingdateta_human_source_human|carafe2rt_0f9b11a215e5|pooled_dimethyl|'
  'pooled|triplemix_source_a_entrap_neu|carafe2rt_0f9b11a215e5|pooled_dimethyl|'
  'pooled|triplemix_bigger_source_b_entrap_neu|carafe2rt_0f9b11a215e5|pooled_dimethyl|'
  "source|combined_md48_cluster|carafe2rt_c1fd5c23319d|source_celegans_overall|$SOURCE_VALIDATION_ROOT"
  "source|combined_md48_cluster|carafe2rt_d18820021476|source_celegans_dimethyl|$SOURCE_VALIDATION_ROOT"
  "source|carafe_trainingdateta_human_source_human|carafe2rt_79e32b81ed0a|source_human_overall_dimethyl|$SOURCE_VALIDATION_ROOT"
  "source|triplemix_source_a_entrap_neu|carafe2rt_64ec74bfccc0|source_source_a_overall|$SOURCE_VALIDATION_ROOT"
  "source|triplemix_source_a_entrap_neu|carafe2rt_3947baffabfd|source_source_a_dimethyl|$SOURCE_VALIDATION_ROOT"
  "source|triplemix_bigger_source_b_entrap_neu|carafe2rt_983d67273372|source_source_b_overall|$SOURCE_VALIDATION_ROOT"
  "source|triplemix_bigger_source_b_entrap_neu|carafe2rt_dbff1bac29b6|source_source_b_dimethyl|$SOURCE_VALIDATION_ROOT"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Queue %s: %s jobs, GPUs %s, max parallel %s\n' "$RUN_LABEL" "${#JOBS[@]}" "$GPU_LIST" "$MAX_PARALLEL"
  printf '%-3s %-7s %-38s %-24s %s\n' '#' phase experiment role model_id
  for i in "${!JOBS[@]}"; do
    IFS='|' read -r phase experiment model role validation_root <<< "${JOBS[$i]}"
    printf '%-3s %-7s %-38s %-24s %s\n' "$((i + 1))" "$phase" "$experiment" "$role" "$model"
  done
  exit 0
fi

[[ ! -e "$QUEUE_DIR" ]] || { echo "ERROR: Queue directory already exists: $QUEUE_DIR" >&2; exit 2; }
mkdir -p "$QUEUE_DIR/logs"
printf 'ordinal\tphase\texperiment\tmodel_id\trole\tsearch_id\tgpu\tvalidation_root\n' > "$PLAN"
printf 'timestamp\tstate\tordinal\tphase\texperiment\tmodel_id\trole\tgpu\texit_code\tlog\n' > "$LEDGER"

declare -a PIDS=() PID_ORDINAL=() PID_PHASE=() PID_EXPERIMENT=() PID_MODEL=() PID_ROLE=() PID_GPU=() PID_LOG=()
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$@" >> "$LEDGER"; }

reap_one() {
  local index pid code
  while :; do
    for index in "${!PIDS[@]}"; do
      pid="${PIDS[$index]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        if wait "$pid"; then code=0; else code=$?; fi
        record "finished" "${PID_ORDINAL[$index]}" "${PID_PHASE[$index]}" "${PID_EXPERIMENT[$index]}" "${PID_MODEL[$index]}" "${PID_ROLE[$index]}" "${PID_GPU[$index]}" "$code" "${PID_LOG[$index]}"
        printf 'Finished #%s on GPU %s (exit %s): %s\n' "${PID_ORDINAL[$index]}" "${PID_GPU[$index]}" "$code" "${PID_ROLE[$index]}"
        unset 'PIDS[index]' 'PID_ORDINAL[index]' 'PID_PHASE[index]' 'PID_EXPERIMENT[index]' 'PID_MODEL[index]' 'PID_ROLE[index]' 'PID_GPU[index]' 'PID_LOG[index]'
        PIDS=("${PIDS[@]}"); PID_ORDINAL=("${PID_ORDINAL[@]}"); PID_PHASE=("${PID_PHASE[@]}"); PID_EXPERIMENT=("${PID_EXPERIMENT[@]}")
        PID_MODEL=("${PID_MODEL[@]}"); PID_ROLE=("${PID_ROLE[@]}"); PID_GPU=("${PID_GPU[@]}"); PID_LOG=("${PID_LOG[@]}")
        return
      fi
    done
    sleep 3
  done
}

next_free_gpu() {
  local candidate assigned
  for candidate in "${GPUS[@]}"; do
    assigned=0
    for gpu_in_use in "${PID_GPU[@]:-}"; do
      [[ "$gpu_in_use" == "$candidate" ]] && { assigned=1; break; }
    done
    [[ "$assigned" == 0 ]] && { printf '%s\n' "$candidate"; return; }
  done
  echo 'ERROR: no free GPU despite an available queue slot' >&2
  exit 1
}

for i in "${!JOBS[@]}"; do
  IFS='|' read -r phase experiment model role validation_root <<< "${JOBS[$i]}"
  while (( ${#PIDS[@]} >= MAX_PARALLEL )); do reap_one; done
  ordinal=$((i + 1))
  gpu="$(next_free_gpu)"
  search_id="${RUN_LABEL}_${ordinal}_${role}"
  log="$QUEUE_DIR/logs/$(printf '%02d' "$ordinal")_${role}.log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ordinal" "$phase" "$experiment" "$model" "$role" "$search_id" "$gpu" "$validation_root" >> "$PLAN"
  cmd=(bash "$LAUNCHER" --experiment "$experiment" --model-id "$model" --profile balanced --search-id "$search_id" --machine-config "$GPU_CONFIG" --carafe-device gpu --gpu "$gpu" --library-only)
  [[ -n "$validation_root" ]] && cmd+=(--validation-root "$validation_root")
  printf 'Starting #%s/%s on GPU %s: %s (%s)\n' "$ordinal" "${#JOBS[@]}" "$gpu" "$experiment" "$role"
  "${cmd[@]}" >"$log" 2>&1 &
  pid=$!
  PIDS+=("$pid"); PID_ORDINAL+=("$ordinal"); PID_PHASE+=("$phase"); PID_EXPERIMENT+=("$experiment"); PID_MODEL+=("$model"); PID_ROLE+=("$role"); PID_GPU+=("$gpu"); PID_LOG+=("$log")
  record "started" "$ordinal" "$phase" "$experiment" "$model" "$role" "$gpu" '' "$log"
done

while (( ${#PIDS[@]} > 0 )); do reap_one; done
failed="$(awk -F '\t' 'NR > 1 && $2 == "finished" && $9 != "0" {count++} END {print count + 0}' "$LEDGER")"
printf 'Queue complete: %s jobs, %s failed. Ledger: %s\n' "${#JOBS[@]}" "$failed" "$LEDGER"
(( failed == 0 ))
