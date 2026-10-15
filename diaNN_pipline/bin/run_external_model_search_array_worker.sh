#!/usr/bin/env bash
# SLURM array task: select one prepared search and run its isolated worker.
set -euo pipefail
[[ $# == 1 && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Usage: $0 <envfile-list> (inside a SLURM array)" >&2; exit 2; }
envfile="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")"
[[ -n "$envfile" && -f "$envfile" ]] || { echo "No prepared envfile for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$root/bin/run_external_model_search_worker.sh" "$envfile"
