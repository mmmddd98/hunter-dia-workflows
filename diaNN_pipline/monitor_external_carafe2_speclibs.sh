#!/usr/bin/env bash
# Compact live monitor for parallel external-Carafe2 spectral-library jobs.
set -euo pipefail

SHARE_ROOT="./shared_data/carafe_share/2_5_0"
INTERVAL=5
ONCE=0
RUN_LABEL=""

usage() {
  cat <<'EOF'
Usage: bash monitor_external_carafe2_speclibs.sh [--share-root <path>] [--run-label <label>] [--interval <seconds>] [--once]

Shows three honest GPU-utilisation bars plus the current durable Carafe job
stages. When Carafe has reached chunked prediction or assembly, its own latest
log line provides a real chunk percentage and ETA. GPU utilisation is never
treated as a completion percentage.
EOF
}

while (($#)); do
  case "$1" in
    --share-root) SHARE_ROOT="${2:?missing path}"; shift 2 ;;
    --run-label) RUN_LABEL="${2:?missing label}"; shift 2 ;;
    --interval) INTERVAL="${2:?missing seconds}"; shift 2 ;;
    --once) ONCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "--interval must be positive" >&2; exit 2; }
[[ -z "$RUN_LABEL" || "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "--run-label must be a safe name" >&2; exit 2; }

bar() {
  local percent="$1" width=30 filled empty
  filled=$(( percent * width / 100 ))
  empty=$(( width - filled ))
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
}

render() {
  clear
  printf 'External Carafe2 Speclib monitor — %s\n\n' "$(date '+%F %T')"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits |
    while IFS=, read -r index util used total; do
      index="${index// /}"; util="${util// /}"; used="${used// /}"; total="${total// /}"
      printf 'GPU %s [%s] %3s%%  %5s / %5s MiB\n' "$index" "$(bar "$util")" "$util" "$used" "$total"
    done
  printf '\nCarafe job phases (running, queued, or failed):\n'
  local found=0 status_file status stage message job detail
  while IFS= read -r status_file; do
    status="$(jq -r '.status // "unknown"' "$status_file" 2>/dev/null || true)"
    [[ "$status" == "running" || "$status" == "queued" || "$status" == "failed" ]] || continue
    stage="$(jq -r '.stage // "unknown"' "$status_file")"
    message="$(jq -r '.message // ""' "$status_file")"
    job="$(basename "$(dirname "$(dirname "$status_file")")")"
    detail=''
    if [[ -f "$(dirname "$status_file")/carafe.log" ]]; then
      detail="$(LC_ALL=C sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g' "$(dirname "$status_file")/carafe.log" | grep -E 'Prediction progress:|Assembly progress:' | tail -n 1 | sed -E 's/^.*(Prediction progress:|Assembly progress:)/\1/' || true)"
    fi
    [[ -n "$detail" ]] || detail="$message"
    printf '  %-8s %-20s %-42s %s\n' "$status" "$stage" "$job" "$detail"
    found=1
  done < <(find "$SHARE_ROOT" -path '*/carafe_output/run_status.json' -type f -print 2>/dev/null | while IFS= read -r path; do
    [[ -z "$RUN_LABEL" || "$path" == *"$RUN_LABEL"* ]] && printf '%s\n' "$path"
  done | sort)
  [[ "$found" == 1 ]] || echo '  no active external Carafe jobs found'
  local ledger label
  while IFS= read -r ledger; do
    label="$(basename "$(dirname "$ledger")")"
    awk -F '\t' -v label="$label" '
      NR == 1 { next }
      $2 == "started" { started++ }
      $2 == "finished" { finished++; if ($9 == "0") ok++; else failed++ }
      END { printf "  Queue %-28s started %2d | finished %2d | ok %2d | failed %2d\n", label, started + 0, finished + 0, ok + 0, failed + 0 }
    ' "$ledger"
  done < <(find "$SHARE_ROOT/external_carafe2_queues" -mindepth 2 -maxdepth 2 -name ledger.tsv -type f -print 2>/dev/null | while IFS= read -r path; do
    [[ -z "$RUN_LABEL" || "$(basename "$(dirname "$path")")" == "$RUN_LABEL" ]] && printf '%s\n' "$path"
  done | sort)
  printf '\nRefresh: %ss  |  Ctrl-C exits\n' "$INTERVAL"
}

while :; do
  render
  [[ "$ONCE" == 1 ]] && exit 0
  sleep "$INTERVAL"
done
