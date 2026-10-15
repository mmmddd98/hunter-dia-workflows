#!/usr/bin/env bash
# Prepare several independent experiment/model searches, then submit one SLURM
# array with a concurrency cap.  This script intentionally delegates all
# provenance checks to the canonical single-search launcher.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.cfg"
die(){ echo "ERROR: $*" >&2; exit 1; }
EXPS=(); MODELS=(); FORWARD=(); PARALLEL=""; TEST=0; USER_THREADS=""
while (($#)); do
  case "$1" in
    --experiment) EXPS+=("${2:-}"); shift 2;;
    --model-id) MODELS+=("${2:-}"); shift 2;;
    --parallel) PARALLEL="${2:-}"; shift 2;;
    --threads) USER_THREADS="${2:-}"; shift 2;;
    --test) TEST=1; shift;;
    *) FORWARD+=("$1"); [[ $# -gt 1 && "$1" =~ ^(--search-id|--executor|--ram-log-interval)$ ]] && { FORWARD+=("$2"); shift 2; } || shift;;
  esac
done
[[ ${#EXPS[@]} -ge 2 ]] || die "Batch mode requires --experiment at least twice"
[[ ${#MODELS[@]} -ge 1 ]] || die "Batch mode requires --model-id"
if [[ ${#MODELS[@]} -ne 1 && ${#MODELS[@]} -ne ${#EXPS[@]} ]]; then
  die "Pass one --model-id for all experiments, or one --model-id per --experiment in the same order"
fi
[[ "$PARALLEL" =~ ^[2-9][0-9]*$ ]] || die "--parallel must be an integer >= 2"
[[ ${#EXPS[@]} -ge "$PARALLEL" ]] || die "--parallel cannot exceed the number of experiments"
executor="${EXTERNAL_MODEL_DEFAULT_EXECUTOR:-cluster}"
for ((i=0;i<${#FORWARD[@]};i++)); do [[ "${FORWARD[$i]}" == --executor ]] && executor="${FORWARD[$((i+1))]}"; done
[[ "$executor" == cluster ]] || die "Bounded batch mode currently requires --executor cluster"
[[ -n "${EXTERNAL_MODEL_CLUSTER_SUBMIT_COMMAND:-}" ]] || die "EXTERNAL_MODEL_CLUSTER_SUBMIT_COMMAND is not configured"
envfiles=(); threads_used=()
for i in "${!EXPS[@]}"; do
  exp="${EXPS[$i]}"
  if [[ ${#MODELS[@]} -eq 1 ]]; then model="${MODELS[0]}"; else model="${MODELS[$i]}"; fi
  search_args=(--experiment "$exp" --model-id "$model" "${FORWARD[@]}")
  preview="$(bash "$ROOT/run_external_model_search.sh" "${search_args[@]}" --test)" || exit $?
  if [[ -n "$USER_THREADS" ]]; then t="$USER_THREADS"; else
    t="$(sed -nE 's/.*THREADS=([0-9]+).*/\1/p' <<< "$preview" | head -1)"
    [[ "$t" =~ ^[1-9][0-9]*$ ]] || die "Could not derive recorded THREADS for $exp"
    t=$(( (t + PARALLEL - 1) / PARALLEL ))
  fi
  threads_used+=("$t")
  if [[ "$TEST" == 1 ]]; then
    echo "===== $exp / $model (THREADS=$t; parallel cap=$PARALLEL) ====="
    bash "$ROOT/run_external_model_search.sh" "${search_args[@]}" --threads "$t" --test
  else
    prepared="$(bash "$ROOT/run_external_model_search.sh" "${search_args[@]}" --threads "$t" --prepare-only | sed -n 's/^PREPARED_ENVFILE=//p')"
    [[ -n "$prepared" && -f "$prepared" ]] || die "Preparation failed for $exp / $model"
    envfiles+=("$prepared")
  fi
done
[[ "$TEST" == 1 ]] && exit 0

# A caller that already owns a SLURM allocation can run the prepared workers
# directly on that node.  This is deliberately plain background Bash: the
# caller controls the node allocation, while DIA-NN receives its per-search
# thread limit from --threads.  Do not submit a nested SLURM array here.
if [[ "${EXTERNAL_MODEL_DIRECT_EXECUTION:-0}" == "1" ]]; then
  worker="$ROOT/bin/run_external_model_search_worker.sh"
  running_pids=() running_names=() rc=0
  for i in "${!envfiles[@]}"; do
    bash "$worker" "${envfiles[$i]}" &
    running_pids+=("$!")
    running_names+=("${EXPS[$i]}")
    if (( ${#running_pids[@]} >= PARALLEL )); then
      if ! wait "${running_pids[0]}"; then
        echo "ERROR: external search failed: ${running_names[0]}" >&2
        rc=1
      fi
      running_pids=("${running_pids[@]:1}")
      running_names=("${running_names[@]:1}")
    fi
  done
  for i in "${!running_pids[@]}"; do
    if ! wait "${running_pids[$i]}"; then
      echo "ERROR: external search failed: ${running_names[$i]}" >&2
      rc=1
    fi
  done
  exit "$rc"
fi

list="$(dirname "${envfiles[0]}")/commands/batch_envfiles_$(date +%Y%m%d_%H%M%S).txt"
printf '%s\n' "${envfiles[@]}" > "$list"
read -r -a submit <<< "$EXTERNAL_MODEL_CLUSTER_SUBMIT_COMMAND"
array_end=$((${#envfiles[@]} - 1))
"${submit[@]}" "--array=0-${array_end}%${PARALLEL}" "$ROOT/bin/run_external_model_search_array_worker.sh" "$list"
for envfile in "${envfiles[@]}"; do result="$(dirname "$envfile")"; printf '{"state":"submitted","mode":"slurm_array","parallel_cap":%s}\n' "$PARALLEL" > "$result/status.json"; done
