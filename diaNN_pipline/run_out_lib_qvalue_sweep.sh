#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${SCRIPT_DIR}"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
source "${PIPELINE_ROOT}/config/entrapment.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

warn(){ _log "WARN" "$*"; }

usage() {
  cat <<'EOF'
Usage:
  bash run_out_lib_qvalue_sweep.sh \
    --expname <name|auto> \
    --ds <dataset> \
    --qvalues <csv> \
    --out-lib-qvalues <csv> \
    [--fasta <filename>|--fasta-abs <path>|--fasta-kind full|swiss|--fasta-entrap [full|swiss]] \
    [--fasta-entrap-shuffle fixed-r|simple] \
    [--mods <preset>] \
    [--set KEY=VALUE ...] \
    [--resume] \
    [--ram-log] \
    [--ram-log-steps all|step1,step5] \
    [--ram-log-interval <seconds>] \
    [--ram-plot] \
    [--parallel 1|2] \
    [--test] \
    [-v|--verbose <0..4>]

Runs Step 1 once, then Step 5 for every combination of:
  --qvalues          DIA-NN Step5 --qvalue / STEP5_REPORT_QVALUE
  --out-lib-qvalues  DIA-NN Step5 --out-lib-qvalue for empirical library generation

The script intentionally skips calibration, tuning and refined-library steps.
EOF
}

resolve_dataset_cfg() {
  local ds_input="$1"
  local datasets_dir="${PIPELINE_ROOT}/datasets"

  if [[ "$ds_input" == /* || "$ds_input" == ./* || "$ds_input" == ../* ]]; then
    if [[ -f "$ds_input" ]]; then
      echo "$(cd "$(dirname "$ds_input")" && pwd)/$(basename "$ds_input")"
      return 0
    fi
    die "Dataset path does not exist: $ds_input"
  fi

  if [[ -f "${datasets_dir}/${ds_input}/config.yaml" ]]; then
    echo "${datasets_dir}/${ds_input}/config.yaml"
    return 0
  fi
  if [[ -f "${datasets_dir}/${ds_input}.yaml" ]]; then
    echo "${datasets_dir}/${ds_input}.yaml"
    return 0
  fi
  if [[ -f "${datasets_dir}/${ds_input}" ]]; then
    echo "${datasets_dir}/${ds_input}"
    return 0
  fi

  die "Dataset config not found for --ds '${ds_input}'."
}

read_env_value() {
  local env_file="$1"
  local wanted_key="$2"
  local line raw key value

  [[ -f "$env_file" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    raw="${line#${line%%[![:space:]]*}}"
    [[ -z "$raw" || "$raw" == \#* || "$raw" != *"="* ]] && continue
    key="${raw%%=*}"
    value="${raw#*=}"
    key="${key%${key##*[![:space:]]}}"
    key="${key#${key%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    value="${value#${value%%[![:space:]]*}}"
    if [[ "$key" == "$wanted_key" ]]; then
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      echo "$value"
      return 0
    fi
  done < "$env_file"

  return 1
}

resolve_base_dir_abs() {
  local ds_base_dir_raw="$1"
  local env_file="${PIPELINE_ROOT}/.env"
  local dataset_root="${DATASET_ROOT:-}"
  local base_trimmed

  base_trimmed="${ds_base_dir_raw#${ds_base_dir_raw%%[![:space:]]*}}"
  base_trimmed="${base_trimmed%${base_trimmed##*[![:space:]]}}"
  [[ -n "$base_trimmed" ]] || die "base_dir is empty"

  if [[ "$base_trimmed" == /* ]]; then
    echo "$base_trimmed"
    return 0
  fi

  if [[ -z "$dataset_root" ]]; then
    dataset_root="$(read_env_value "$env_file" "DATASET_ROOT" || true)"
  fi

  if [[ -z "$dataset_root" ]]; then
    dataset_root="${SHARED_STORAGE_ROOT:-}"
  fi
  if [[ -z "$dataset_root" ]]; then
    dataset_root="$(read_env_value "$env_file" "SHARED_STORAGE_ROOT" || true)"
  fi

  [[ -n "$dataset_root" ]] || die "Relative base_dir requires DATASET_ROOT"
  [[ "$dataset_root" == /* ]] || die "DATASET_ROOT must be absolute, got: ${dataset_root}"

  echo "${dataset_root%/}/${base_trimmed#./}"
}

parse_dataset_yaml() {
  local cfg_file="$1"
  local dataset_type="${2:-prehunter}"
  local py_bin="${PIPELINE_PYTHON_PATH:-python3}"

  require_exec "$py_bin"
  eval "$("$py_bin" - "$cfg_file" "$dataset_type" <<'PY'
import shlex
import sys

cfg_path = sys.argv[1]
dataset_type = (sys.argv[2] if len(sys.argv) > 2 else "prehunter").strip().lower()
if dataset_type not in {"prehunter", "hunter"}:
    raise ValueError("DATASET_TYPE must be 'prehunter' or 'hunter'")

try:
    import yaml
except Exception:
    print("echo 'ERROR: PyYAML is required for dataset parsing' >&2")
    print("exit 1")
    raise SystemExit(0)

with open(cfg_path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}

def get(obj, keys, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def req(obj, keys):
    val = get(obj, keys, None)
    if val in (None, ""):
        raise ValueError(f"Missing required key in dataset yaml: {'.'.join(keys)}")
    return val

def q(value):
    return shlex.quote("" if value is None else str(value))

def emit(key, value):
    print(f"{key}={q(value)}")

def prefixed_dir(source_name, rel_dir):
    if rel_dir in (None, ""):
        return source_name
    rel = str(rel_dir).strip("/")
    if not rel:
        return source_name
    if not source_name or rel == source_name or rel.startswith(f"{source_name}/"):
        return rel
    return f"{source_name}/{rel}"

try:
    dataset_id = req(data, ["meta", "dataset_id"])
    sources = get(data, ["sources"], None)
    if isinstance(sources, dict):
        selected_source_name = dataset_type
        source_block = sources.get(selected_source_name)
        if not isinstance(source_block, dict):
            raise ValueError(f"Dataset yaml does not define sources.{dataset_type}")
        source_data = get(source_block, ["data"], {}) or {}
        base_dir = get(sources, ["base_dir"], "") or get(source_data, ["base_dir"], "")
        if not base_dir:
            raise ValueError("sources.base_dir or sources.<type>.data.base_dir must be set")
        dir_train = prefixed_dir(selected_source_name, get(source_data, ["dirs", "train"], ""))
        dir_test = prefixed_dir(selected_source_name, get(source_data, ["dirs", "test"], ""))
        train_files = get(source_data, ["train_files"], []) or []
        test_files = get(source_data, ["test_files"], []) or []
    else:
        selected_source_name = ""
        source_data = get(data, ["data"], {}) or {}
        base_dir = req(source_data, ["base_dir"])
        dir_train = get(source_data, ["dirs", "train"], "")
        dir_test = get(source_data, ["dirs", "test"], "")
        train_files = get(source_data, ["train_files"], []) or []
        test_files = get(source_data, ["test_files"], []) or []

    if not isinstance(train_files, list) or not isinstance(test_files, list):
        raise ValueError("train_files and test_files must be lists")

    emit("DS_META_DATASET_ID", dataset_id)
    emit("DS_DATASET_TYPE", selected_source_name or dataset_type)
    emit("DS_SOURCE_NAME", selected_source_name)
    emit("DS_DATA_BASE_DIR", base_dir)
    emit("DS_DIR_TRAIN", dir_train)
    emit("DS_DIR_TEST", dir_test)
    emit("DS_FASTA", get(data, ["fasta", "fasta"], ""))
    emit("DS_FASTA_SWISS", get(data, ["fasta", "fasta_swiss"], ""))
    emit("DS_TRAIN_FILES", "|".join(str(x) for x in train_files))
    emit("DS_TEST_FILES", "|".join(str(x) for x in test_files))
except Exception as exc:
    print(f"echo {q('ERROR: ' + str(exc))} >&2")
    print("exit 1")
PY
)"
}

entrapment_filename_for_fasta() {
  local fasta_path="$1"
  local shuffle_mode="${2:-${FASTA_ENTRAP_SHUFFLE_MODE:-fixed-r}}"
  local base stem
  base="$(basename "$fasta_path")"
  case "$base" in
    *.fasta) stem="${base%.fasta}" ;;
    *.fa) stem="${base%.fa}" ;;
    *.faa) stem="${base%.faa}" ;;
    *.fas) stem="${base%.fas}" ;;
    *) stem="$base" ;;
  esac
  case "$shuffle_mode" in
    fixed-r|fixed_r|rfixed|r-fixed|preserve-r|preserve_r)
      echo "${stem}.entrapment_rfixed_combined.fasta"
      ;;
    simple|legacy|legacy-simple)
      echo "${stem}.entrapment_combined.fasta"
      ;;
    *)
      die "Invalid FASTA_ENTRAP_SHUFFLE_MODE '${shuffle_mode}' (allowed: fixed-r|simple)"
      ;;
  esac
}

ensure_entrapment_fasta() {
  local source_fasta="$1"
  local entrap_fasta="$2"
  local py_bin="${ENTRAPMENT_PYTHON_BIN:-${PIPELINE_PYTHON_PATH:-python3}}"
  local entrap_dir entrap_name

  require_file "$source_fasta"
  if [[ -f "$entrap_fasta" ]]; then
    FASTA_ENTRAP_ACTION="existing"
    return 0
  fi

  FASTA_ENTRAP_ACTION="create"
  if [[ "$TEST_MODE" == "1" ]]; then
    return 0
  fi

  require_exec "$py_bin"
  entrap_dir="$(dirname "$entrap_fasta")"
  entrap_name="$(basename "$entrap_fasta")"
  mkdir -p "$entrap_dir"
  PYTHONPATH="${PIPELINE_ROOT}/.." "$py_bin" -m entrapment.cli \
    --fasta "$source_fasta" \
    --out-dir "$entrap_dir" \
    --filename "$entrap_name" \
    --shuffle-mode "${FASTA_ENTRAP_SHUFFLE_MODE:-fixed-r}" \
    --seed "${FASTA_ENTRAP_SEED:-1337}" >/dev/null
  require_file "$entrap_fasta"
}

resolve_data_path() {
  local base_dir="$1"
  local rel_dir="$2"
  local file_name="$3"
  if [[ "$file_name" == /* ]]; then
    echo "$file_name"
  elif [[ -n "$rel_dir" ]]; then
    echo "${base_dir%/}/${rel_dir%/}/${file_name}"
  else
    echo "${base_dir%/}/${file_name}"
  fi
}

resolve_existing_data_file_path() {
  local base_dir="$1"
  local rel_dir="$2"
  local file_name="$3"
  local candidate

  candidate="$(resolve_data_path "$base_dir" "$rel_dir" "$file_name")"
  [[ -f "$candidate" ]] && { echo "$candidate"; return 0; }
  if [[ "$file_name" != *.* ]]; then
    candidate="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.mzML")"
    [[ -f "$candidate" ]] && { echo "$candidate"; return 0; }
    candidate="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.raw")"
    [[ -f "$candidate" ]] && { echo "$candidate"; return 0; }
  fi
  return 1
}

normalize_split_list() {
  local input="$1"
  local arr_name="$2"
  local -n arr_ref="$arr_name"
  IFS='|' read -r -a arr_ref <<< "$input"
  if [[ ${#arr_ref[@]} -eq 1 && -z "${arr_ref[0]}" ]]; then
    arr_ref=()
  fi
}

split_csv() {
  local input="$1"
  local arr_name="$2"
  local -n arr_ref="$arr_name"
  local token trimmed

  arr_ref=()
  IFS=',' read -r -a raw_parts <<< "$input"
  for token in "${raw_parts[@]}"; do
    trimmed="${token//[[:space:]]/}"
    [[ -n "$trimmed" ]] && arr_ref+=("$trimmed")
  done
  (( ${#arr_ref[@]} > 0 )) || die "CSV value resolves to an empty list: ${input}"
}

wait_for_parallel_slot() {
  local max_jobs="$1"
  local -n pids_ref="$2"
  local failed_var="$3"
  local -n failed_ref="$failed_var"
  local pid remaining=()

  while (( ${#pids_ref[@]} >= max_jobs )); do
    if wait -n; then
      :
    else
      failed_ref=1
    fi

    remaining=()
    for pid in "${pids_ref[@]}"; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        remaining+=("$pid")
      fi
    done
    pids_ref=("${remaining[@]}")
  done
}

wait_for_remaining_jobs() {
  local -n pids_ref="$1"
  local failed_var="$2"
  local -n failed_ref="$failed_var"
  local pid

  for pid in "${pids_ref[@]}"; do
    if wait "$pid"; then
      :
    else
      failed_ref=1
    fi
  done
  pids_ref=()
}

qvalue_tag() {
  local value="$1"
  printf '%s' "$value" | sed -e 's/[^A-Za-z0-9]/_/g' -e 's/_$//' -e 's/^_/q_/'
}

resolve_step1_outlib() {
  local outdir_s1="$1"
  local expname="$2"
  local parquet_candidate="${outdir_s1}/step1_${expname}_predicted.parquet"
  local speclib_candidates=()

  if [[ -f "$parquet_candidate" ]]; then
    echo "$parquet_candidate"
    return 0
  fi

  shopt -s nullglob
  speclib_candidates=("${outdir_s1}/step1_${expname}_predicted"*.speclib)
  shopt -u nullglob

  if (( ${#speclib_candidates[@]} > 0 )); then
    echo "${speclib_candidates[0]}"
    return 0
  fi
  return 1
}

run_with_optional_ram_logger() {
  local step_key="$1"
  local log_dir="$2"
  local log_tag="$3"
  local cmd_array_name="$4"
  local -n __cmd_ref="$cmd_array_name"

  if ! ram_step_selected "$step_key"; then
    "${__cmd_ref[@]}"
    return $?
  fi

  mkdir -p "$log_dir"
  local ts safe_tag log_file logger_pid cmd_rc
  ts="$(date +'%Y%m%d_%H%M%S')"
  safe_tag="${log_tag//[^A-Za-z0-9._-]/_}"
  log_file="${log_dir%/}/ram_${safe_tag}_${ts}.txt"
  info "RAM logger enabled for ${step_key}: ${log_file}"

  set +e
  bash "$RAM_LOGGER_SCRIPT" "$RAM_LOG_INTERVAL" "$log_file" "$$" >/dev/null 2>&1 &
  logger_pid=$!
  "${__cmd_ref[@]}"
  cmd_rc=$?
  kill "$logger_pid" >/dev/null 2>&1 || true
  wait "$logger_pid" >/dev/null 2>&1 || true
  set -e

  if [[ "${RAM_PLOT_ENABLE:-0}" == "1" ]]; then
    generate_ram_plot_for_log "$log_file" "$step_key"
  fi

  return "$cmd_rc"
}

ram_step_selected() {
  local step_key="$1"
  [[ "${RAM_LOG_ENABLE:-0}" == "1" ]] || return 1

  local steps_csv="${RAM_LOG_STEPS:-all}"
  [[ -n "$steps_csv" ]] || steps_csv="all"
  [[ "$steps_csv" == "all" ]] && return 0

  case ",${steps_csv}," in
    *,"${step_key}",*) return 0 ;;
    *) return 1 ;;
  esac
}

generate_ram_plot_for_log() {
  local log_file="$1"
  local step_key="${2:-step}"

  [[ -f "$log_file" ]] || return 0
  local out_png="${log_file%.txt}.png"
  [[ "$out_png" == "$log_file" ]] && out_png="${log_file}.png"

  local py_bin="${RAM_PLOT_PYTHON_BIN:-${PIPELINE_PYTHON_PATH:-python3}}"
  set +e
  "$py_bin" "$RAM_PLOT_SCRIPT" --input "$log_file" --output "$out_png" --title "RAM usage (${step_key})"
  local plot_rc=$?
  set -e

  if (( plot_rc == 0 )); then
    info "RAM plot written: ${out_png}"
  else
    warn "RAM plot generation failed for ${log_file} (exit ${plot_rc}); continuing"
  fi
}

cleanup_run_pid_file() {
  if [[ -n "${RUN_PID_META_FILE:-}" && -f "$RUN_PID_META_FILE" ]]; then
    printf 'ended_at=%s\n' "$(date -Is)" >> "$RUN_PID_META_FILE"
  fi
}

register_run_pid_file() {
  RUN_PID_FILE="${EXPDIR}/run_all.pid"
  RUN_PID_META_FILE="${EXPDIR}/run_all.pid.meta"
  local start_ticks
  start_ticks="$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null || true)"
  mkdir -p "$EXPDIR"
  printf '%s\n' "$$" > "$RUN_PID_FILE"
  {
    printf 'pid=%s\n' "$$"
    printf 'start_ticks=%s\n' "$start_ticks"
    printf 'script=%s\n' "run_out_lib_qvalue_sweep.sh"
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'expdir=%s\n' "$EXPDIR"
    printf 'cmd='
    printf '%q ' "$0" "$@"
    printf '\n'
  } > "$RUN_PID_META_FILE"
  trap cleanup_run_pid_file EXIT
}

get_timestamp() {
  date +'%Y-%m-%dT%H:%M:%S%z'
}

append_progress_start() {
  local step_name="$1"
  local detail="${2:-}"
  printf '%s\t%s\tSTART\t%s\n' "$(get_timestamp)" "$step_name" "$detail" >> "$PROGRESS_FILE"
}

append_progress_ok() {
  local step_name="$1"
  local detail="${2:-}"
  local start_ts="${3:-}"
  local ts elapsed_str=""
  ts="$(get_timestamp)"
  if [[ -n "$start_ts" ]]; then
    local start_sec end_sec elapsed_sec
    start_sec=$(date -d "$start_ts" +%s 2>/dev/null || echo 0)
    end_sec=$(date -d "$ts" +%s 2>/dev/null || echo 0)
    elapsed_sec=$((end_sec - start_sec))
    if (( elapsed_sec > 0 )); then
      elapsed_str=";elapsed=${elapsed_sec}s"
    fi
  fi
  printf '%s\t%s\tOK\t%s%s\n' "$ts" "$step_name" "$detail" "$elapsed_str" >> "$PROGRESS_FILE"
}

FASTA_NAME=""
FASTA_ABS=""
FASTA_KIND="${FASTA_KIND:-full}"
FASTA_ENTRAP=0
FASTA_ENTRAP_KIND=""
FASTA_ENTRAP_ACTION="none"
FASTA_ENTRAP_SHUFFLE_MODE="${FASTA_ENTRAP_SHUFFLE_MODE:-fixed-r}"
FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=0
DS_INPUT=""
QVALUES_CSV=""
OUT_LIB_QVALUES_CSV=""
TEST_MODE=0
RESUME_MODE=0
PARALLEL=1
CLI_OVERRIDES=()
VERBOSE="${VERBOSE:-0}"
RAM_LOG_ENABLE="${RAM_LOG_ENABLE:-0}"
RAM_LOG_INTERVAL="${RAM_LOG_INTERVAL:-10}"
RAM_LOG_STEPS="${RAM_LOG_STEPS:-all}"
RAM_LOGGER_SCRIPT="${PIPELINE_ROOT}/ram_logger.sh"
RAM_PLOT_ENABLE="${RAM_PLOT_ENABLE:-0}"
RAM_PLOT_SCRIPT="${PIPELINE_ROOT}/plots/step_04_vary_2026_04_01/ramPlotter.py"
RAM_PLOT_PYTHON_BIN="${RAM_PLOT_PYTHON_BIN:-${PIPELINE_PYTHON_PATH:-python3}}"

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --ds)
      ((i+=1)); DS_INPUT="${ARGS[$i]:-}"
      [[ -n "$DS_INPUT" ]] || die "--ds needs a value"
      ;;
    --qvalues)
      ((i+=1)); QVALUES_CSV="${ARGS[$i]:-}"
      [[ -n "$QVALUES_CSV" ]] || die "--qvalues needs a CSV value"
      ;;
    --out-lib-qvalues)
      ((i+=1)); OUT_LIB_QVALUES_CSV="${ARGS[$i]:-}"
      [[ -n "$OUT_LIB_QVALUES_CSV" ]] || die "--out-lib-qvalues needs a CSV value"
      ;;
    --fasta)
      ((i+=1)); FASTA_NAME="${ARGS[$i]:-}"
      [[ -n "$FASTA_NAME" ]] || die "--fasta needs a value"
      ;;
    --fasta-abs|--fastaabs)
      ((i+=1)); FASTA_ABS="${ARGS[$i]:-}"
      [[ -n "$FASTA_ABS" ]] || die "--fasta-abs needs a value"
      ;;
    --fasta-kind|--fasta-source)
      ((i+=1)); FASTA_KIND="${ARGS[$i]:-}"
      [[ -n "$FASTA_KIND" ]] || die "--fasta-kind needs full|swiss"
      ;;
    --fasta-entrap)
      FASTA_ENTRAP=1
      next="${ARGS[$((i+1))]:-}"
      if [[ -n "$next" && "$next" != --* ]]; then
        ((i+=1))
        FASTA_ENTRAP_KIND="$next"
      fi
      ;;
    --fasta-entrap=*)
      FASTA_ENTRAP=1
      FASTA_ENTRAP_KIND="${ARGS[$i]#*=}"
      [[ -n "$FASTA_ENTRAP_KIND" ]] || die "--fasta-entrap needs full|swiss"
      ;;
    --fasta-entrap-shuffle)
      ((i+=1)); FASTA_ENTRAP_SHUFFLE_MODE="${ARGS[$i]:-}"
      [[ -n "$FASTA_ENTRAP_SHUFFLE_MODE" ]] || die "--fasta-entrap-shuffle needs fixed-r|simple"
      FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=1
      ;;
    --fasta-entrap-shuffle=*)
      FASTA_ENTRAP_SHUFFLE_MODE="${ARGS[$i]#*=}"
      [[ -n "$FASTA_ENTRAP_SHUFFLE_MODE" ]] || die "--fasta-entrap-shuffle needs fixed-r|simple"
      FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=1
      ;;
    --mods)
      ((i+=1)); MOD_CONFIG="${ARGS[$i]:-}"
      [[ -n "$MOD_CONFIG" ]] || die "--mods needs a preset name"
      CLI_OVERRIDES+=("MOD_CONFIG")
      ;;
    --set)
      ((i+=1))
      override="${ARGS[$i]:-}"
      [[ -n "$override" && "$override" == *=* ]] || die "--set expects KEY=VALUE"
      key_raw="${override%%=*}"
      key="${key_raw^^}"
      value="${override#*=}"
      [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "Invalid --set key '$key_raw'"
      printf -v "$key" '%s' "$value"
      CLI_OVERRIDES+=("$key")
      if [[ "$key" == "FASTA_ENTRAP_SHUFFLE_MODE" ]]; then
        FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=1
      fi
      ;;
    --ram-log)
      RAM_LOG_ENABLE=1
      ;;
    --ram-log-steps)
      ((i+=1)); RAM_LOG_STEPS="${ARGS[$i]:-}"
      [[ -n "$RAM_LOG_STEPS" ]] || die "--ram-log-steps needs a value"
      RAM_LOG_ENABLE=1
      ;;
    --ram-log-interval)
      ((i+=1)); RAM_LOG_INTERVAL="${ARGS[$i]:-}"
      [[ "$RAM_LOG_INTERVAL" =~ ^[0-9]+$ ]] || die "--ram-log-interval must be an integer"
      ;;
    --ram-plot)
      RAM_PLOT_ENABLE=1
      RAM_LOG_ENABLE=1
      ;;
    --parallel)
      ((i+=1)); PARALLEL="${ARGS[$i]:-}"
      [[ "$PARALLEL" =~ ^[0-9]+$ ]] || die "--parallel must be 1 or 2"
      ;;
    --resume)
      RESUME_MODE=1
      ;;
    --test)
      TEST_MODE=1
      ;;
    --expname)
      ((i+=1))
      [[ -n "${ARGS[$i]:-}" ]] || die "--expname needs a value"
      ;;
    -v|--v)
      VERBOSE=1
      ;;
    --verbose)
      next="${ARGS[$((i+1))]:-}"
      if [[ "$next" =~ ^[0-4]$ ]]; then
        VERBOSE="$next"
        ((i+=1))
      else
        VERBOSE=2
      fi
      ;;
    --verbose=*)
      VERBOSE="${ARGS[$i]#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: ${ARGS[$i]}"
      ;;
  esac
  ((i+=1))
done

[[ -n "$DS_INPUT" ]] || die "--ds is required"
[[ -n "$QVALUES_CSV" ]] || die "--qvalues is required"
[[ -n "$OUT_LIB_QVALUES_CSV" ]] || die "--out-lib-qvalues is required"
case "$PARALLEL" in
  1|2) ;;
  *) die "--parallel supports only 1 or 2 for this sweep" ;;
esac

RAM_LOG_STEPS="${RAM_LOG_STEPS,,}"
RAM_LOG_STEPS="${RAM_LOG_STEPS// /}"
for token in ${RAM_LOG_STEPS//,/ }; do
  case "$token" in
    all|step1|step5) ;;
    *) die "Invalid --ram-log-steps token '${token}' (allowed: all,step1,step5)" ;;
  esac
done

FASTA_KIND="${FASTA_ENTRAP_KIND:-$FASTA_KIND}"
FASTA_KIND="${FASTA_KIND,,}"
case "$FASTA_KIND" in
  full|swiss) ;;
  *) die "Invalid FASTA kind '${FASTA_KIND}' (allowed: full|swiss)" ;;
esac

FASTA_ENTRAP_SHUFFLE_MODE="${FASTA_ENTRAP_SHUFFLE_MODE,,}"
case "$FASTA_ENTRAP_SHUFFLE_MODE" in
  fixed-r|fixed_r|rfixed|r-fixed|preserve-r|preserve_r)
    FASTA_ENTRAP_SHUFFLE_MODE="fixed-r"
    ;;
  simple|legacy|legacy-simple)
    FASTA_ENTRAP_SHUFFLE_MODE="simple"
    ;;
  *)
    die "Invalid FASTA_ENTRAP_SHUFFLE_MODE '${FASTA_ENTRAP_SHUFFLE_MODE}' (allowed: fixed-r|simple)"
    ;;
esac

DIANN_VERBOSE="${DIANN_VERBOSE:-${VERBOSE:-0}}"
[[ "$DIANN_VERBOSE" =~ ^[0-4]$ ]] || die "Invalid verbose level '$DIANN_VERBOSE' (allowed: 0..4)"
if (( DIANN_VERBOSE > 0 )); then
  VERBOSE=1
else
  VERBOSE=0
fi
export VERBOSE

parse_expname "$@"
if [[ "$TEST_MODE" == "1" ]]; then
  : "${OUTBASE:?OUTBASE not set}"
  EXPDIR="${OUTBASE%/}/${EXPNAME}"
  S1DIR="${EXPDIR}/01_predlib"
  S5DIR="${EXPDIR}/05_quant"
  export EXPDIR S1DIR S5DIR
else
  set_paths_for_exp
  PROGRESS_FILE="${EXPDIR}/progress.txt"
  export PROGRESS_FILE
  register_run_pid_file "$@"
fi

if [[ "$RESUME_MODE" == "1" && "$FASTA_ENTRAP" == "1" && "$FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT" == "0" && -f "${PROGRESS_FILE:-}" ]]; then
  if grep -q 'step1_predlib.sh.*fasta=.*\.entrapment_combined\.fasta' "$PROGRESS_FILE" 2>/dev/null; then
    FASTA_ENTRAP_SHUFFLE_MODE="simple"
    info "Resume mode: detected legacy simple entrapment FASTA in progress.txt; keeping FASTA_ENTRAP_SHUFFLE_MODE=simple for this run"
  elif grep -q 'step1_predlib.sh.*fasta=.*\.entrapment_rfixed_combined\.fasta' "$PROGRESS_FILE" 2>/dev/null; then
    FASTA_ENTRAP_SHUFFLE_MODE="fixed-r"
    info "Resume mode: detected fixed-r entrapment FASTA in progress.txt; keeping FASTA_ENTRAP_SHUFFLE_MODE=fixed-r for this run"
  fi
fi

load_mod_config
split_csv "$QVALUES_CSV" QVALUES
split_csv "$OUT_LIB_QVALUES_CSV" OUT_LIB_QVALUES

STEP5_THREADS_PER_JOB="${THREADS:-1}"
if (( PARALLEL > 1 )); then
  [[ "${THREADS:-}" =~ ^[0-9]+$ ]] || die "THREADS must be numeric when --parallel is used"
  (( THREADS >= PARALLEL )) || die "THREADS (${THREADS}) must be >= --parallel (${PARALLEL})"
  STEP5_THREADS_PER_JOB=$(( THREADS / PARALLEL ))
  (( STEP5_THREADS_PER_JOB >= 1 )) || STEP5_THREADS_PER_JOB=1
  if (( THREADS % PARALLEL != 0 )); then
    warn "THREADS=${THREADS} does not divide evenly by --parallel=${PARALLEL}; using THREADS=${STEP5_THREADS_PER_JOB} per Step5 job."
  fi
fi

DATASET_TYPE="${DATASET_TYPE:-prehunter}"
DATASET_CFG="$(resolve_dataset_cfg "$DS_INPUT")"
require_file "$DATASET_CFG"
parse_dataset_yaml "$DATASET_CFG" "$DATASET_TYPE"
BASE_DIR_ABS="$(resolve_base_dir_abs "${DS_DATA_BASE_DIR}")"
[[ -d "$BASE_DIR_ABS" ]] || die "Missing base_dir directory: ${BASE_DIR_ABS}"

if [[ "$FASTA_ENTRAP" == "1" ]]; then
  case "$FASTA_KIND" in
    full) FASTA_FROM_YAML="${DS_FASTA:-}"; FASTA_YAML_KEY="fasta.fasta" ;;
    swiss) FASTA_FROM_YAML="${DS_FASTA_SWISS:-}"; FASTA_YAML_KEY="fasta.fasta_swiss" ;;
  esac
  [[ -n "$FASTA_FROM_YAML" ]] || die "--fasta-entrap ${FASTA_KIND} requested, but ${FASTA_YAML_KEY} is empty"
  [[ "$FASTA_FROM_YAML" != */* ]] || die "Dataset YAML ${FASTA_YAML_KEY} must be a FASTABASE filename"
  FASTA_ENTRAP_SOURCE="${FASTABASE%/}/${FASTA_FROM_YAML}"
  FASTA_ENTRAP_TARGET="${FASTABASE%/}/$(entrapment_filename_for_fasta "$FASTA_FROM_YAML" "$FASTA_ENTRAP_SHUFFLE_MODE")"
  ensure_entrapment_fasta "$FASTA_ENTRAP_SOURCE" "$FASTA_ENTRAP_TARGET"
  FASTA_FILE="$FASTA_ENTRAP_TARGET"
elif [[ -n "$FASTA_NAME" ]]; then
  [[ "$FASTA_NAME" != */* ]] || die "--fasta expects only a filename from FASTABASE"
  FASTA_FILE="${FASTABASE%/}/${FASTA_NAME}"
elif [[ -n "$FASTA_ABS" ]]; then
  [[ "$FASTA_ABS" == /* ]] || die "--fasta-abs expects an absolute path"
  FASTA_FILE="$FASTA_ABS"
else
  case "$FASTA_KIND" in
    full) FASTA_FROM_YAML="${DS_FASTA:-}"; FASTA_YAML_KEY="fasta.fasta" ;;
    swiss) FASTA_FROM_YAML="${DS_FASTA_SWISS:-}"; FASTA_YAML_KEY="fasta.fasta_swiss" ;;
  esac
  [[ -n "$FASTA_FROM_YAML" ]] || die "No FASTA provided and dataset YAML ${FASTA_YAML_KEY} is empty"
  [[ "$FASTA_FROM_YAML" != */* ]] || die "Dataset YAML ${FASTA_YAML_KEY} must be a FASTABASE filename"
  FASTA_FILE="${FASTABASE%/}/${FASTA_FROM_YAML}"
fi

if [[ "$FASTA_ENTRAP" == "1" && "$TEST_MODE" == "1" && "$FASTA_ENTRAP_ACTION" == "create" ]]; then
  :
else
  require_file "$FASTA_FILE"
fi

COPIED_FASTA="${EXPDIR}/$(basename "$FASTA_FILE")"
if [[ "$TEST_MODE" != "1" ]]; then
  src_real="$(readlink -f "$FASTA_FILE")"
  dst_real="$(readlink -m "$COPIED_FASTA")"
  if [[ "$src_real" != "$dst_real" ]]; then
    cp -f "$FASTA_FILE" "$COPIED_FASTA"
    info "Copied FASTA to experiment base dir: ${COPIED_FASTA}"
  fi
fi

normalize_split_list "${DS_TRAIN_FILES:-}" DS_TRAIN_FILES_ARR
normalize_split_list "${DS_TEST_FILES:-}" DS_TEST_FILES_ARR

STEP5_FILE_SOURCE_LABEL="test_files"
STEP5_FILE_SOURCE_DIR="${DS_DIR_TEST:-}"
STEP5_FILE_NAMES=("${DS_TEST_FILES_ARR[@]}")
if (( ${#STEP5_FILE_NAMES[@]} == 0 )); then
  warn "Dataset test_files is empty; Step5 falls back to train_files"
  STEP5_FILE_SOURCE_LABEL="train_files"
  STEP5_FILE_SOURCE_DIR="${DS_DIR_TRAIN:-}"
  STEP5_FILE_NAMES=("${DS_TRAIN_FILES_ARR[@]}")
fi
(( ${#STEP5_FILE_NAMES[@]} > 0 )) || die "Step5 input file list is empty"

STEP5_RAW_FILES_RESOLVED=()
for step5_file_name in "${STEP5_FILE_NAMES[@]}"; do
  step5_raw_file="$(resolve_existing_data_file_path "$BASE_DIR_ABS" "$STEP5_FILE_SOURCE_DIR" "$step5_file_name" || true)"
  [[ -n "$step5_raw_file" ]] || die "Could not resolve Step5 file '${step5_file_name}'"
  STEP5_RAW_FILES_RESOLVED+=("$step5_raw_file")
done

STEP1_SCRIPT="${PIPELINE_ROOT}/bin/step1_predlib.sh"
STEP5_SCRIPT="${PIPELINE_ROOT}/bin/step5_quant.sh"
require_file "$STEP1_SCRIPT"
require_file "$STEP5_SCRIPT"

STEP1_CMD=(bash "$STEP1_SCRIPT" --expname "$EXPNAME" --fastaabs "$COPIED_FASTA")
if (( DIANN_VERBOSE > 0 )); then
  STEP1_CMD+=(--verbose "$DIANN_VERBOSE")
fi
for key in "${CLI_OVERRIDES[@]}"; do
  STEP1_CMD+=(--set "${key}=${!key}")
done

echo "Dataset config       : ${DATASET_CFG}"
echo "Dataset id           : ${DS_META_DATASET_ID}"
echo "Dataset type         : ${DS_DATASET_TYPE}"
echo "Step5 files source   : ${STEP5_FILE_SOURCE_LABEL}"
echo "Step5 files n        : ${#STEP5_RAW_FILES_RESOLVED[@]}"
echo "FASTA source         : ${FASTA_FILE}"
echo "FASTA run copy       : ${COPIED_FASTA}"
echo "Experiment dir       : ${EXPDIR}"
echo "Step5 qvalues        : ${QVALUES[*]}"
echo "Out-lib qvalues      : ${OUT_LIB_QVALUES[*]}"
echo "Step5 parallel jobs  : ${PARALLEL}"
echo "Step5 threads/job    : ${STEP5_THREADS_PER_JOB}"
echo "Modification preset  : ${MOD_CONFIG:-legacy}"
echo ""

if [[ "$TEST_MODE" == "1" ]]; then
  echo "TEST MODE: no directories/files will be created; DIA-NN will not run."
  if [[ "$FASTA_ENTRAP" == "1" ]]; then
    echo "TEST MODE: FASTA entrapment ${FASTA_ENTRAP_ACTION}: ${FASTA_ENTRAP_SOURCE:-} -> ${FASTA_ENTRAP_TARGET:-}"
    echo "TEST MODE: FASTA entrapment shuffle mode: ${FASTA_ENTRAP_SHUFFLE_MODE}"
  fi
  echo "Step1 command:"
  printf '%q ' "${STEP1_CMD[@]}" --test
  echo
fi

if [[ "$TEST_MODE" != "1" ]]; then
  {
    echo "experiment_type=out_lib_qvalue_sweep"
    echo "script=run_out_lib_qvalue_sweep.sh"
    echo "qvalues=${QVALUES_CSV}"
    echo "out_lib_qvalues=${OUT_LIB_QVALUES_CSV}"
    echo "dataset=${DS_INPUT}"
    echo "dataset_type=${DATASET_TYPE}"
  } > "${EXPDIR}/experiment_type.txt"

  if [[ "$RESUME_MODE" == "1" ]] && STEP1_OUTLIB_EXISTING="$(resolve_step1_outlib "$S1DIR" "$EXPNAME" || true)" && [[ -n "$STEP1_OUTLIB_EXISTING" ]]; then
    info "Resume: reusing existing Step1 library: ${STEP1_OUTLIB_EXISTING}"
  else
    info "Running Step1 once"
    STEP1_START_TS="$(get_timestamp)"
    append_progress_start "step1_predlib.sh" "mode=out-lib-qvalue-sweep;fasta=${COPIED_FASTA}"
    run_with_optional_ram_logger "step1" "$S1DIR" "step1_predlib" STEP1_CMD
    append_progress_ok "step1_predlib.sh" "mode=out-lib-qvalue-sweep;fasta=${COPIED_FASTA}" "$STEP1_START_TS"
  fi
fi

if [[ "$TEST_MODE" == "1" ]]; then
  STEP1_OUTLIB="${S1DIR}/step1_${EXPNAME}_predicted.parquet"
else
  STEP1_OUTLIB="$(resolve_step1_outlib "$S1DIR" "$EXPNAME" || true)"
  [[ -n "$STEP1_OUTLIB" ]] || die "Step1 output library not found in ${S1DIR}"
fi

run_step5_combo() {
  local qvalue="$1"
  local out_lib_qvalue="$2"
  local q_tag out_lib_tag out_sub out_dir sweep_extra_flags raw_file key
  local step5_start_ts step5_detail
  local -a step5_cmd

  q_tag="$(qvalue_tag "$qvalue")"
  out_lib_tag="$(qvalue_tag "$out_lib_qvalue")"
  out_sub="q_${q_tag}__outlibq_${out_lib_tag}"
  out_dir="${S5DIR}/${out_sub}"
  sweep_extra_flags="${EXTRA_FLAGS:+${EXTRA_FLAGS} }--out-lib-qvalue ${out_lib_qvalue}"

  step5_cmd=(
    bash "$STEP5_SCRIPT"
    --expname "$EXPNAME"
    --fastaabs "$COPIED_FASTA"
    --lib "$STEP1_OUTLIB"
    --out-sub "$out_sub"
    --set "STEP5_REPORT_QVALUE=${qvalue}"
    --set "EXTRA_FLAGS=${sweep_extra_flags}"
  )
  for raw_file in "${STEP5_RAW_FILES_RESOLVED[@]}"; do
    step5_cmd+=(--f "$raw_file")
  done
  if (( DIANN_VERBOSE > 0 )); then
    step5_cmd+=(--verbose "$DIANN_VERBOSE")
  fi
  for key in "${CLI_OVERRIDES[@]}"; do
    [[ "$key" == "STEP5_REPORT_QVALUE" || "$key" == "EXTRA_FLAGS" || "$key" == "THREADS" ]] && continue
    step5_cmd+=(--set "${key}=${!key}")
  done
  step5_cmd+=(--set "THREADS=${STEP5_THREADS_PER_JOB}")

  if [[ "$TEST_MODE" == "1" ]]; then
    echo "Step5 command (${out_sub}):"
    printf '%q ' "${step5_cmd[@]}" --test
    echo
    return 0
  fi

  if [[ "$RESUME_MODE" == "1" && -f "${out_dir}/final_report.parquet" ]]; then
    info "Resume: skipping completed Step5 run ${out_sub}"
    return 0
  fi

  info "Running Step5 qvalue=${qvalue}, out-lib-qvalue=${out_lib_qvalue} (${out_sub}, THREADS=${STEP5_THREADS_PER_JOB})"
  step5_start_ts="$(get_timestamp)"
  step5_detail="mode=out-lib-qvalue-sweep;qvalue=${qvalue};out_lib_qvalue=${out_lib_qvalue};out_sub=${out_sub};parallel=${PARALLEL};threads=${STEP5_THREADS_PER_JOB};files=${#STEP5_RAW_FILES_RESOLVED[@]}"
  append_progress_start "step5_quant.sh" "$step5_detail"
  run_with_optional_ram_logger "step5" "$out_dir" "step5_${out_sub}" step5_cmd
  append_progress_ok "step5_quant.sh" "$step5_detail" "$step5_start_ts"
}

STEP5_PIDS=()
STEP5_FAILED=0
for qvalue in "${QVALUES[@]}"; do
  for out_lib_qvalue in "${OUT_LIB_QVALUES[@]}"; do
    if [[ "$TEST_MODE" == "1" || "$PARALLEL" == "1" ]]; then
      run_step5_combo "$qvalue" "$out_lib_qvalue"
    else
      wait_for_parallel_slot "$PARALLEL" STEP5_PIDS STEP5_FAILED
      run_step5_combo "$qvalue" "$out_lib_qvalue" &
      STEP5_PIDS+=("$!")
    fi
  done
done

if [[ "$TEST_MODE" != "1" && "$PARALLEL" != "1" ]]; then
  wait_for_remaining_jobs STEP5_PIDS STEP5_FAILED
  (( STEP5_FAILED == 0 )) || die "At least one parallel Step5 run failed"
fi

info "Out-lib qvalue sweep finished"
