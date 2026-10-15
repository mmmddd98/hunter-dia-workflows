#!/usr/bin/env bash
set -euo pipefail

# RAM Logger for DIA-NN processes
# Usage: ./ram_logger.sh [interval_seconds] [output_file]

INTERVAL="${1:-5}"
OUTFILE="${2:-ram_log_$(date +%Y%m%d_%H%M%S).txt}"
TARGET_PID="${3:-}"

[[ "$INTERVAL" =~ ^[0-9]+$ ]] || { echo "ERROR: interval must be integer >= 1" >&2; exit 1; }
(( INTERVAL >= 1 )) || { echo "ERROR: interval must be >= 1" >&2; exit 1; }

mkdir -p "$(dirname "$OUTFILE")"

get_all_pids() {
    local parent=$1
    echo $parent
    local children=$(pgrep -P $parent 2>/dev/null)
    for child in $children; do
        get_all_pids $child
    done
}

echo "# RAM Logger started: $(date)" | tee "$OUTFILE"
echo "# Interval: ${INTERVAL}s | Output: $OUTFILE | Target PIDs from: $TARGET_PID" | tee -a "$OUTFILE"
echo "# Timestamp, Pipeline_GB, System_Used_GB, Available_GB, Swap_GB" | tee -a "$OUTFILE"
echo "#--------------------------------------------------------" | tee -a "$OUTFILE"

while true; do
  TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"

  if [ -n "$TARGET_PID" ]; then
      ALL_PIDS=$(get_all_pids $TARGET_PID | tr '\n' ',' | sed 's/,$//')
      if [ -n "$ALL_PIDS" ]; then
          MY_CODE_GB="$(ps -o rss= -p $ALL_PIDS 2>/dev/null | awk '{sum += $1} END {printf "%.2f", sum/1024/1024}')"
      else
          MY_CODE_GB="0.00"
      fi
  else
      MY_CODE_GB="$(ps aux | grep -i diann | grep -v grep | awk '{sum += $6} END {printf "%.2f", sum/1024/1024}')"
  fi
  MY_CODE_GB="${MY_CODE_GB:-0.00}"

  read -r USED_GB AVAIL_GB <<< "$(free -m | awk '/^Mem:/ {printf "%.2f %.2f", $3/1024, $7/1024}')"
  SWAP_GB="$(free -m | awk '/^Swap:/ {printf "%.2f", $3/1024}')"

  BAR="$(python3 -c "print('█' * int(float('${MY_CODE_GB:-0}')/5))" 2>/dev/null || true)"

  LINE="$TIMESTAMP | Pipeline: ${MY_CODE_GB:-0} GB | SysUsed: $USED_GB GB | Avail: $AVAIL_GB GB | Swap: $SWAP_GB GB"
  echo "$LINE" | tee -a "$OUTFILE"
  echo "  $BAR" | tee -a "$OUTFILE"

  sleep "$INTERVAL"
done
