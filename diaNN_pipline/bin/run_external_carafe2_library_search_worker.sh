#!/usr/bin/env bash
# Execute one already prepared DIA-NN Step-5 search with a Carafe2 library.
set -euo pipefail

[[ $# == 1 ]] || { echo "Usage: $0 <resolved_config.env>" >&2; exit 2; }
# shellcheck source=/dev/null
source "$1"

status() {
  printf '{"state":"%s","updated_at":"%s"}\n' "$1" "$(date -Is)" > "$RESULT_DIR/status.json"
}

append_progress() {
  local state="$1" detail="$2" ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  mkdir -p "$(dirname "$CENTRAL_PROGRESS_FILE")"
  (
    flock -x 9
    [[ -s "$CENTRAL_PROGRESS_FILE" ]] \
      || printf '# External Carafe2 library-search progress (timestamp<TAB>state<TAB>details)\n' >&9
    printf '%s\t%s\t%s\n' "$ts" "$state" "$detail" >&9
  ) 9>>"$CENTRAL_PROGRESS_FILE"
}

detail() {
  printf 'model_id=%s;experiment=%s;search_id=%s;host=%s;job_id=%s;result_dir=%s' \
    "$MODEL_ID" "$EXPERIMENT" "$SEARCH_ID" "$(hostname)" "${SLURM_JOB_ID:-local}" "$RESULT_DIR"
}

CURRENT_STATE="starting"
trap 'rc=$?; status failed; append_progress FAILED "$(detail);stage=$CURRENT_STATE;exit_code=$rc" || true; exit $rc' ERR

mkdir -p "$RESULT_DIR/05_quant" "$RESULT_DIR/logs" "$RESULT_DIR/tmp"
status running_step5
CURRENT_STATE="step5_quant"
append_progress START "$(detail);library=$LIBRARY_FILE"
"${STEP5[@]}"

[[ -f "$RESULT_DIR/05_quant/final_report.parquet" ]] \
  || { echo "Expected DIA-NN report is missing: $RESULT_DIR/05_quant/final_report.parquet" >&2; exit 1; }

status complete
CURRENT_STATE="complete"
append_progress OK "$(detail);report=$RESULT_DIR/05_quant/final_report.parquet"
echo "DIA-NN search complete: $RESULT_DIR/05_quant/final_report.parquet"
