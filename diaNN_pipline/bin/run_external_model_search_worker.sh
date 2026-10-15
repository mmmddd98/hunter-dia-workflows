#!/usr/bin/env bash
# Runs on the selected executor.  The launcher has already validated provenance.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 <resolved_config.env>" >&2; exit 2; }
source "$1"
RESULT_PROGRESS_FILE="${RESULT_PROGRESS_FILE:-$RESULT_DIR/progress.txt}"
status(){ printf '{"state":"%s","updated_at":"%s"}\n' "$1" "$(date -Is)" > "$RESULT_DIR/status.json"; }
progress_detail(){ printf 'model_id=%s;experiment=%s;search_id=%s;executor=cluster;host=%s;job_id=%s;result_dir=%s' "$MODEL_ID" "$EXPERIMENT" "$SEARCH_ID" "$(hostname)" "${SLURM_JOB_ID:-local}" "$RESULT_DIR"; }
append_external_progress() {
  local step="$1" state="$2" detail="$3" start_ts="${4:-}" ts elapsed=""
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  [[ -n "$start_ts" ]] && elapsed=";elapsed=$(( $(date +%s) - start_ts ))s"
  detail="${detail//$'\t'/ }"; detail="${detail//$'\n'/ }"
  mkdir -p "$(dirname "$CENTRAL_PROGRESS_FILE")"
  (
    flock -x 9
    [[ -s "$CENTRAL_PROGRESS_FILE" ]] || printf '# Central external DIA-NN model-search progress (timestamp<TAB>step<TAB>state<TAB>details)\n' >&9
    printf '%s\t%s\t%s\t%s%s\n' "$ts" "$step" "$state" "$detail" "$elapsed" >&9
  ) 9>>"$CENTRAL_PROGRESS_FILE"
  mkdir -p "$(dirname "$RESULT_PROGRESS_FILE")"
  (
    flock -x 9
    [[ -s "$RESULT_PROGRESS_FILE" ]] || printf '# External DIA-NN model-search progress (timestamp<TAB>step<TAB>state<TAB>details)\n' >&9
    printf '%s\t%s\t%s\t%s%s\n' "$ts" "$step" "$state" "$detail" "$elapsed" >&9
  ) 9>>"$RESULT_PROGRESS_FILE"
}
run_with_ram_logger() {
  local step="$1" log_file="$2"; shift 2
  local logger_pid=0 rc
  if [[ "${RAM_LOG_ENABLE:-1}" == "1" ]]; then
    mkdir -p "$(dirname "$log_file")"
    # The logger follows this worker's process tree.  Pipeline_GB is therefore
    # per external search, while SysUsed/Avail describe the shared compute node
    # and make simultaneous runs directly comparable in the MA-plot server.
    bash "$PIPE_ROOT/ram_logger.sh" "$RAM_LOG_INTERVAL" "$log_file" "$$" >/dev/null 2>&1 &
    logger_pid=$!
  fi
  set +e
  "$@"; rc=$?
  if (( logger_pid > 0 )); then kill "$logger_pid" >/dev/null 2>&1 || true; wait "$logger_pid" >/dev/null 2>&1 || true; fi
  set -e
  return "$rc"
}
CURRENT_STEP="external_model_search_worker.sh"
trap 'rc=$?; status failed; append_external_progress "$CURRENT_STEP" "FAILED" "$(progress_detail);exit_code=$rc" || true; exit $rc' ERR
mkdir -p "$RESULT_DIR" "$RESULT_DIR/04_refined_lib" "$RESULT_DIR/05_quant" "$RESULT_DIR/logs" "$RESULT_DIR/tmp"
rsync -a --checksum --info=progress2 "$PACKAGE/" "$RESULT_DIR/model_package/"
(cd "$RESULT_DIR/model_package" && sha256sum -c checksums.sha256)
append_external_progress "external_model_search_worker.sh" "START" "$(progress_detail);ram_log=${RAM_LOG_ENABLE:-1};ram_interval=${RAM_LOG_INTERVAL:-10}"
status running_step4
CURRENT_STEP="step4_create_tuned_predlib.sh"; step4_start="$(date +%s)"
append_external_progress "$CURRENT_STEP" "START" "$(progress_detail);rt_model=external;fr_model=original"
run_with_ram_logger step4 "$RESULT_DIR/04_refined_lib/ram_step4_$(date +%Y%m%d_%H%M%S).txt" "${STEP4[@]}"
lib="$RESULT_DIR/04_refined_lib/refined_lib.predicted.speclib"
[[ -f "$lib" ]] || { echo "Expected Step-4 library is missing: $lib" >&2; exit 1; }
append_external_progress "$CURRENT_STEP" "OK" "$(progress_detail);library=$lib" "$step4_start"
status step4_complete
status running_step5
CURRENT_STEP="step5_quant.sh"; step5_start="$(date +%s)"
append_external_progress "$CURRENT_STEP" "START" "$(progress_detail);library=$lib"
run_with_ram_logger step5 "$RESULT_DIR/05_quant/ram_step5_$(date +%Y%m%d_%H%M%S).txt" "${STEP5[@]}"
[[ -f "$RESULT_DIR/05_quant/final_report.parquet" ]] || { echo "Expected Step-5 report is missing" >&2; exit 1; }
append_external_progress "$CURRENT_STEP" "OK" "$(progress_detail);report=$RESULT_DIR/05_quant/final_report.parquet" "$step5_start"
step5_log="$RESULT_DIR/logs/step5_quant.log"
if [[ -f "$step5_log" ]] && grep -q 'Running Arabidopsis reference evaluation' "$step5_log"; then
  append_external_progress "arabidopsis_reference_evaluation" "OK" "$(progress_detail);report=$RESULT_DIR/05_quant/final_report.parquet"
fi
if [[ -f "$step5_log" ]] && grep -q 'Running E. coli centered evaluation' "$step5_log"; then
  append_external_progress "ecoli_centered_evaluation" "OK" "$(progress_detail);report=$RESULT_DIR/05_quant/final_report.parquet"
fi
status complete
CURRENT_STEP="maplot_server_ready"; maplot_start="$(date +%s)"
append_external_progress "$CURRENT_STEP" "START" "$(progress_detail);report=$RESULT_DIR/05_quant/final_report.parquet"
append_external_progress "$CURRENT_STEP" "OK" "$(progress_detail);report=$RESULT_DIR/05_quant/final_report.parquet" "$maplot_start"
append_external_progress "external_model_search_worker.sh" "OK" "$(progress_detail)"
