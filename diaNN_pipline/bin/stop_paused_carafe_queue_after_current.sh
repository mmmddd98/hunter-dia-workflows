#!/usr/bin/env bash
# Stop a deliberately paused sequential Carafe queue after its current job has
# completed. The queue parent must already be SIGSTOP-ed; this guard prevents
# a subsequent queued library job from being started.
set -euo pipefail

usage() {
  echo "Usage: $0 --queue-pid PID --status-file FILE [--poll-seconds N]" >&2
}

queue_pid=""
status_file=""
poll_seconds=60

while (($#)); do
  case "$1" in
    --queue-pid) queue_pid="${2:-}"; shift 2 ;;
    --status-file) status_file="${2:-}"; shift 2 ;;
    --poll-seconds) poll_seconds="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "$queue_pid" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ -f "$status_file" ]] || { echo "Missing status file: $status_file" >&2; exit 2; }
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }

queue_command="$(ps -p "$queue_pid" -o args= 2>/dev/null || true)"
[[ "$queue_command" == *run_external_carafe2_validation_3gpu_queue.sh* ]] || {
  echo "PID $queue_pid is not the expected Carafe queue launcher." >&2
  exit 2
}
queue_state="$(ps -p "$queue_pid" -o stat= 2>/dev/null | tr -d '[:space:]')"
[[ "$queue_state" == T* ]] || {
  echo "Refusing: queue launcher $queue_pid is not paused (state: ${queue_state:-missing})." >&2
  exit 2
}

echo "[$(date '+%F %T')] Watching $status_file; queue launcher $queue_pid is paused."
while kill -0 "$queue_pid" 2>/dev/null; do
  status="$(python3 - "$status_file" <<'PY'
import json
import pathlib
import sys
try:
    print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get("status", ""))
except (OSError, ValueError, TypeError):
    print("")
PY
)"
  case "$status" in
    succeeded)
      queue_state="$(ps -p "$queue_pid" -o stat= 2>/dev/null | tr -d '[:space:]')"
      [[ "$queue_state" == T* ]] || {
        echo "[$(date '+%F %T')] Refusing: queue is no longer paused (state: ${queue_state:-missing})." >&2
        exit 1
      }
      echo "[$(date '+%F %T')] Current Carafe job succeeded; terminating paused queue launcher $queue_pid."
      kill -TERM "$queue_pid"
      exit 0
      ;;
    failed|cancelled)
      echo "[$(date '+%F %T')] Current job ended with status '$status'; queue left paused for manual review." >&2
      exit 1
      ;;
  esac
  sleep "$poll_seconds"
done

echo "[$(date '+%F %T')] Queue launcher already exited."
