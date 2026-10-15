#!/usr/bin/env bash
# Run the four standard per-dataset Carafe2 baselines after a paused external
# model queue has finished its current job. Each run fine-tunes Carafe locally
# and creates one experiment-specific library; DIA-NN final searches stay on
# the cluster and are intentionally not started here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_CONFIG="$ROOT/config/carafe_machines/gpu_pc.local.cfg"
PIPELINE_VERSION="2_5_0"
JOB_NAME="carafe_standard_baseline_v1"
GPU_LIST="0,1,2"

wait_status_file=""
paused_queue_pid=""
log_root="./shared_data/carafe_share/2_5_0/standard_carafe_baseline_queue_v1"

usage() {
  echo "Usage: $0 --wait-status-file FILE --paused-queue-pid PID" >&2
}

while (($#)); do
  case "$1" in
    --wait-status-file) wait_status_file="${2:-}"; shift 2 ;;
    --paused-queue-pid) paused_queue_pid="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -f "$wait_status_file" ]] || { echo "Missing status file: $wait_status_file" >&2; exit 2; }
[[ "$paused_queue_pid" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ -f "$MACHINE_CONFIG" ]] || { echo "Missing machine configuration: $MACHINE_CONFIG" >&2; exit 2; }
mkdir -p "$log_root"
master_log="$log_root/standard_baseline_queue.log"
exec > >(tee -a "$master_log") 2>&1

status_value() {
  python3 - "$wait_status_file" <<'PY'
import json
import pathlib
import sys
try:
    print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get("status", ""))
except (OSError, ValueError, TypeError):
    print("")
PY
}

echo "[$(date '+%F %T')] Waiting for the active external Human library job."
while :; do
  status="$(status_value)"
  case "$status" in
    succeeded) break ;;
    failed|cancelled)
      echo "External Human job ended with '$status'; standard baseline queue not started." >&2
      exit 1
      ;;
  esac
  sleep 60
done

echo "[$(date '+%F %T')] External Human job succeeded; waiting for paused external queue to exit."
while kill -0 "$paused_queue_pid" 2>/dev/null; do sleep 5; done

experiments=(
  combined_md48_cluster
  triplemix_source_a_entrap_neu
  carafe_trainingdateta_human_source_human
  triplemix_bigger_source_b_entrap_neu
)

for index in "${!experiments[@]}"; do
  experiment="${experiments[$index]}"
  ordinal=$((index + 1))
  job_dir="./shared_data/carafe_share/${PIPELINE_VERSION}/${experiment}/carafe/balanced/${JOB_NAME}"
  status_file="$job_dir/meta/status.json"
  echo "[$(date '+%F %T')] Starting standard Carafe baseline $ordinal/${#experiments[@]}: $experiment"
  bash "$ROOT/run_carafe_profile.sh" \
    --experiment "$experiment" \
    --pipeline-version "$PIPELINE_VERSION" \
    --job "balanced/$JOB_NAME" \
    --machine-config "$MACHINE_CONFIG" \
    --gpu "$GPU_LIST"
  result="$(python3 - "$status_file" <<'PY'
import json
import pathlib
import sys
try:
    print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get("status", ""))
except (OSError, ValueError, TypeError):
    print("")
PY
)"
  [[ "$result" == succeeded ]] || {
    echo "Standard Carafe baseline failed: $experiment (status: ${result:-missing})" >&2
    exit 1
  }
  echo "[$(date '+%F %T')] Completed standard Carafe baseline $ordinal/${#experiments[@]}: $experiment"
done

echo "[$(date '+%F %T')] All four Carafe libraries are complete. Run the recorded return scripts on the cluster for the final DIA-NN searches."
