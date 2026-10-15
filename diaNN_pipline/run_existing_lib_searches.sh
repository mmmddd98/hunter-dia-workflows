#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${SCRIPT_DIR}"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

warn(){ _log "WARN" "$*"; }

usage() {
  cat <<'EOF'
Usage:
  run_existing_lib_searches.sh --source-exp <existing_exp> --ds <dataset> [options]

Purpose:
  Re-run only Step5 searches for a new dataset, reusing FASTA, modification
  settings, and spectral libraries from an existing experiment.

Required:
  --source-exp <name>        Existing experiment containing 05_quant and libraries
  --ds <dataset>             New dataset selector (same resolution as run_all.sh)

Options:
  --target-exp <name>        Where to write new searches. Default: --source-exp
  --dataset-type <type>      prehunter|hunter. Default from DATASET_TYPE config
  --search <name>            Search to rerun. Repeatable. Values: all, baseline,
                             tuned, or an existing out-sub such as
                             tuned/balanced_rt_fr_lr_0p00011_d0
  --out-label <label>        Group folder below 05_quant. Default:
                             <dataset_id>_<dataset_type>. Aliases:
  --group-name, --run-name
  --resume                   Skip searches whose target report already exists
  --set KEY=VALUE            Override Step5 variable for the rerun (later wins)
  --ram-log                  Enable RAM logging for each Step5 re-search
                             (default follows config/defaults.cfg)
  --ram-log-interval <sec>   RAM logger interval in seconds
  --ram-plot                 Write one RAM plot per RAM log file
  --test                     Validate and print Step5 commands without running DIA-NN
  --verbose [0..4]           Forward verbosity to Step5
  -h | --help                Show this help

Examples:
  bash run_existing_lib_searches.sh --source-exp exp_a --ds md048 --dataset-type hunter
  bash run_existing_lib_searches.sh --source-exp exp_a --ds md048 --search baseline --test
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

  [[ -n "$dataset_root" ]] || die "Relative base_dir requires DATASET_ROOT (machine config, environment, or ${env_file})"
  [[ "$dataset_root" == /* ]] || die "DATASET_ROOT must be absolute, got: ${dataset_root}"
  echo "${dataset_root%/}/${base_trimmed#./}"
}

parse_dataset_yaml() {
  local cfg_file="$1"
  local dataset_type="$2"
  local py_bin="${PIPELINE_PYTHON_PATH:-python3}"

  require_exec "$py_bin"

  eval "$("$py_bin" - "$cfg_file" "$dataset_type" <<'PY'
import shlex
import sys

cfg_path = sys.argv[1]
dataset_type = sys.argv[2].strip().lower()

if dataset_type not in {"prehunter", "hunter"}:
    raise ValueError("dataset type must be prehunter or hunter")

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
    value = get(obj, keys, None)
    if value in (None, ""):
        raise ValueError(f"Missing required key in dataset yaml: {'.'.join(keys)}")
    return value

def emit(key, value):
    print(f"{key}={shlex.quote('' if value is None else str(value))}")

def prefixed_dir(source_name, rel_dir):
    if rel_dir in (None, ""):
        return source_name
    rel = str(rel_dir).strip("/")
    if not rel:
        return source_name
    if rel == source_name or rel.startswith(f"{source_name}/"):
        return rel
    return f"{source_name}/{rel}" if source_name else rel

try:
    dataset_id = req(data, ["meta", "dataset_id"])
    sources = get(data, ["sources"], None)
    if isinstance(sources, dict):
        source = sources.get(dataset_type)
        if not isinstance(source, dict):
            raise ValueError(f"Dataset yaml does not define sources.{dataset_type}")
        source_data = get(source, ["data"], {}) or {}
        base_dir = get(sources, ["base_dir"], "") or get(source_data, ["base_dir"], "")
        if not base_dir:
            raise ValueError("sources.base_dir or sources.<type>.data.base_dir must be set")
        source_name = dataset_type
        dir_train = prefixed_dir(source_name, get(source_data, ["dirs", "train"], ""))
        dir_test = prefixed_dir(source_name, get(source_data, ["dirs", "test"], ""))
        train_files = get(source_data, ["train_files"], []) or []
        test_files = get(source_data, ["test_files"], []) or []
    else:
        source_data = get(data, ["data"], {}) or {}
        base_dir = req(source_data, ["base_dir"])
        source_name = ""
        dir_train = get(source_data, ["dirs", "train"], "")
        dir_test = get(source_data, ["dirs", "test"], "")
        train_files = get(source_data, ["train_files"], []) or []
        test_files = get(source_data, ["test_files"], []) or []

    if not isinstance(train_files, list) or not isinstance(test_files, list):
        raise ValueError("train_files and test_files must be lists")

    emit("DS_META_DATASET_ID", dataset_id)
    emit("DS_DATASET_TYPE", source_name or dataset_type)
    emit("DS_DATA_BASE_DIR", base_dir)
    emit("DS_DIR_TRAIN", dir_train)
    emit("DS_DIR_TEST", dir_test)
    emit("DS_TRAIN_FILES", "|".join(str(x) for x in train_files))
    emit("DS_TEST_FILES", "|".join(str(x) for x in test_files))
except Exception as exc:
    print(f"echo {shlex.quote('ERROR: ' + str(exc))} >&2")
    print("exit 1")
PY
)"
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
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  if [[ "$file_name" != *.* ]]; then
    candidate="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.mzML")"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi

    candidate="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.raw")"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  return 1
}

sanitize_label() {
  local value="$1"
  value="${value//[^A-Za-z0-9._-]/_}"
  value="${value##_}"
  value="${value%%_}"
  [[ -n "$value" ]] || value="rerun"
  echo "$value"
}

new_out_sub_for_search() {
  local old_out_sub="$1"
  local label="$2"
  if [[ "$old_out_sub" == tuned/* ]]; then
    echo "${label}/${old_out_sub#tuned/}"
  else
    echo "${label}/${old_out_sub}"
  fi
}

report_exists_for_out_sub() {
  local target_expdir="$1"
  local out_sub="$2"
  local out_dir="${target_expdir}/05_quant/${out_sub}"

  [[ -f "${out_dir}/final_report.parquet" ]] && return 0
  [[ -f "${out_dir}/final_report-first-pass.parquet" ]] && return 0
  [[ -f "${out_dir}/final_report.tsv" ]] && return 0
  return 1
}

resolve_existing_lib_file() {
  local input="$1"
  local dir stem
  local -a candidates

  if [[ -f "$input" ]]; then
    echo "$input"
    return 0
  fi

  if [[ -f "${input}.skyline.speclib" ]]; then
    echo "${input}.skyline.speclib"
    return 0
  fi

  dir="$(dirname "$input")"
  stem="$(basename "$input")"
  stem="${stem%.parquet}"

  shopt -s nullglob
  candidates=("${dir}/${stem}"*.speclib)
  shopt -u nullglob

  if (( ${#candidates[@]} > 0 )); then
    if (( ${#candidates[@]} > 1 )); then
      warn "Multiple library fallbacks found for ${input}; using ${candidates[0]}"
    else
      warn "Using library fallback for missing ${input}: ${candidates[0]}"
    fi
    echo "${candidates[0]}"
    return 0
  fi

  return 1
}

ram_logging_enabled() {
  [[ "${RAM_LOG_ENABLE:-0}" == "1" ]]
}

run_with_optional_ram_logger() {
  local out_sub="$1"
  local log_tag="$2"
  local cmd_array_name="$3"
  local -n __cmd_ref="$cmd_array_name"

  if ! ram_logging_enabled; then
    "${__cmd_ref[@]}"
    return $?
  fi

  local log_dir="${TARGET_EXPDIR}/05_quant/${out_sub}"
  local ts safe_tag log_file logger_pid cmd_rc
  mkdir -p "$log_dir"
  ts="$(date +'%Y%m%d_%H%M%S')"
  safe_tag="${log_tag//[^A-Za-z0-9._-]/_}"
  log_file="${log_dir%/}/ram_${safe_tag}_${ts}.txt"

  info "RAM logger enabled for Step5 re-search: ${log_file}"

  set +e
  bash "$RAM_LOGGER_SCRIPT" "$RAM_LOG_INTERVAL" "$log_file" "$$" >/dev/null 2>&1 &
  logger_pid=$!

  "${__cmd_ref[@]}"
  cmd_rc=$?

  kill "$logger_pid" >/dev/null 2>&1 || true
  wait "$logger_pid" >/dev/null 2>&1 || true
  set -e

  if [[ "${RAM_PLOT_ENABLE:-0}" == "1" ]]; then
    generate_ram_plot_for_log "$log_file"
  fi

  return "$cmd_rc"
}

generate_ram_plot_for_log() {
  local log_file="$1"
  local out_png="${log_file%.txt}.png"
  local py_bin="${RAM_PLOT_PYTHON_BIN:-${PIPELINE_PYTHON_PATH:-python3}}"
  local plot_rc

  [[ "${RAM_PLOT_ENABLE:-0}" == "1" ]] || return 0
  [[ -f "$log_file" ]] || return 0

  set +e
  "$py_bin" "$RAM_PLOT_SCRIPT" --input "$log_file" --output "$out_png" --title "RAM usage (Step5 re-search)"
  plot_rc=$?
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
  RUN_PID_FILE="${TARGET_EXPDIR}/run_all.pid"
  RUN_PID_META_FILE="${TARGET_EXPDIR}/run_all.pid.meta"
  local start_ticks
  start_ticks="$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null || true)"
  mkdir -p "$TARGET_EXPDIR"
  printf '%s\n' "$$" > "$RUN_PID_FILE"
  {
    printf 'pid=%s\n' "$$"
    printf 'start_ticks=%s\n' "$start_ticks"
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'expdir=%s\n' "$TARGET_EXPDIR"
    printf 'script=%s\n' "run_existing_lib_searches.sh"
    printf 'cmd='
    printf '%q ' "$0" "$@"
    printf '\n'
  } > "$RUN_PID_META_FILE"
  trap cleanup_run_pid_file EXIT
}

append_progress_event() {
  local step_name="$1"
  local state="$2"
  local detail="${3:-}"
  local ts
  [[ "$TEST_MODE" == "1" ]] && return 0
  ts="$(date +'%Y-%m-%dT%H:%M:%S%z')"
  mkdir -p "$TARGET_EXPDIR"
  printf '%s\t%s\t%s\t%s\n' "$ts" "$step_name" "$state" "$detail" >> "${TARGET_EXPDIR}/progress.txt"
}

search_selected() {
  local out_sub="$1"
  local selector

  if (( ${#SEARCH_FILTERS[@]} == 0 )); then
    return 0
  fi

  for selector in "${SEARCH_FILTERS[@]}"; do
    case "$selector" in
      all)
        return 0
        ;;
      baseline)
        [[ "$out_sub" == "baseline" ]] && return 0
        ;;
      tuned)
        [[ "$out_sub" == tuned/* ]] && return 0
        ;;
      *)
        [[ "$out_sub" == "$selector" ]] && return 0
        ;;
    esac
  done

  return 1
}

discover_step5_searches() {
  local expdir="$1"
  local py_bin="${PIPELINE_PYTHON_PATH:-python3}"

  "$py_bin" - "$expdir" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

expdir = Path(sys.argv[1])
files = [expdir / "all_startcmds.txt"]
files.extend(sorted((expdir / "05_quant").glob("**/startcmd.txt")))

seen = set()
records = []

def command_lines(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if any(token.endswith("step5_quant.sh") for token in tokens):
            yield tokens

for path in files:
    for tokens in command_lines(path):
        record = {"lib": "", "fastaabs": "", "fasta": "", "out_sub": "", "sets": []}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "--lib" and i + 1 < len(tokens):
                i += 1
                record["lib"] = tokens[i]
            elif token in {"--fastaabs", "--fasta-abs"} and i + 1 < len(tokens):
                i += 1
                record["fastaabs"] = tokens[i]
            elif token == "--fasta" and i + 1 < len(tokens):
                i += 1
                record["fasta"] = tokens[i]
            elif token == "--out-sub" and i + 1 < len(tokens):
                i += 1
                record["out_sub"] = tokens[i]
            elif token == "--set" and i + 1 < len(tokens):
                i += 1
                record["sets"].append(tokens[i])
            i += 1

        if not record["lib"]:
            continue
        if not record["out_sub"]:
            record["out_sub"] = "baseline"
        key = record["out_sub"]
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

for record in sorted(records, key=lambda item: (item["out_sub"] != "baseline", item["out_sub"])):
    print("\x1e".join([
        record["out_sub"],
        record["lib"],
        record["fastaabs"],
        record["fasta"],
        "\x1f".join(record["sets"]),
    ]))
PY
}

SOURCE_EXP=""
TARGET_EXP=""
DS_INPUT=""
DATASET_TYPE="${DATASET_TYPE:-prehunter}"
OUT_LABEL=""
TEST_MODE=0
RESUME_MODE=0
VERBOSE="${VERBOSE:-0}"
RAM_LOG_ENABLE="${RAM_LOG_ENABLE:-1}"
RAM_LOG_INTERVAL="${RAM_LOG_INTERVAL:-10}"
RAM_LOGGER_SCRIPT="${PIPELINE_ROOT}/ram_logger.sh"
RAM_PLOT_ENABLE="${RAM_PLOT_ENABLE:-0}"
RAM_PLOT_SCRIPT="${PIPELINE_ROOT}/plots/step_04_vary_2026_04_01/ramPlotter.py"
RAM_PLOT_PYTHON_BIN="${RAM_PLOT_PYTHON_BIN:-${PIPELINE_PYTHON_PATH:-python3}}"
SEARCH_FILTERS=()
RERUN_OVERRIDES=()

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --source-exp|--existing-exp)
      ((i+=1))
      SOURCE_EXP="${ARGS[$i]:-}"
      [[ -n "$SOURCE_EXP" ]] || die "--source-exp needs a value"
      ;;
    --target-exp|--expname)
      ((i+=1))
      TARGET_EXP="${ARGS[$i]:-}"
      [[ -n "$TARGET_EXP" ]] || die "--target-exp needs a value"
      ;;
    --ds)
      ((i+=1))
      DS_INPUT="${ARGS[$i]:-}"
      [[ -n "$DS_INPUT" ]] || die "--ds needs a value"
      ;;
    --dataset-type)
      ((i+=1))
      DATASET_TYPE="${ARGS[$i]:-}"
      [[ -n "$DATASET_TYPE" ]] || die "--dataset-type needs a value"
      ;;
    --search)
      ((i+=1))
      search_value="${ARGS[$i]:-}"
      [[ -n "$search_value" ]] || die "--search needs a value"
      SEARCH_FILTERS+=("$search_value")
      ;;
    --out-label|--group-name|--run-name)
      ((i+=1))
      OUT_LABEL="${ARGS[$i]:-}"
      [[ -n "$OUT_LABEL" ]] || die "${ARGS[$((i-1))]} needs a value"
      ;;
    --resume)
      RESUME_MODE=1
      ;;
    --set)
      ((i+=1))
      override="${ARGS[$i]:-}"
      [[ -n "$override" ]] || die "--set needs KEY=VALUE"
      [[ "$override" == *=* ]] || die "--set expects KEY=VALUE"
      key_raw="${override%%=*}"
      key="${key_raw^^}"
      value="${override#*=}"
      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid --set key '$key'"
      RERUN_OVERRIDES+=("${key}=${value}")
      if [[ "${key^^}" == "DATASET_TYPE" ]]; then
        DATASET_TYPE="$value"
      fi
      ;;
    --ram-log)
      RAM_LOG_ENABLE=1
      ;;
    --ram-log-interval)
      ((i+=1))
      RAM_LOG_INTERVAL="${ARGS[$i]:-}"
      [[ -n "$RAM_LOG_INTERVAL" ]] || die "--ram-log-interval needs a value"
      RAM_LOG_ENABLE=1
      ;;
    --ram-plot)
      RAM_PLOT_ENABLE=1
      RAM_LOG_ENABLE=1
      ;;
    --test)
      TEST_MODE=1
      ;;
    --v|-v)
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

[[ -n "$SOURCE_EXP" ]] || die "--source-exp is required"
[[ -n "$DS_INPUT" ]] || die "--ds is required"
TARGET_EXP="${TARGET_EXP:-$SOURCE_EXP}"
DATASET_TYPE="${DATASET_TYPE,,}"
case "$DATASET_TYPE" in
  prehunter|hunter) ;;
  *) die "--dataset-type must be prehunter or hunter" ;;
esac
[[ "$VERBOSE" =~ ^[0-4]$ ]] || die "Invalid verbose level '$VERBOSE' (allowed: 0..4)"
if [[ "$RAM_LOG_ENABLE" == "1" ]]; then
  [[ "$RAM_LOG_INTERVAL" =~ ^[0-9]+$ ]] || die "--ram-log-interval must be an integer >= 1"
  (( RAM_LOG_INTERVAL >= 1 )) || die "--ram-log-interval must be >= 1"
  require_file "$RAM_LOGGER_SCRIPT"
fi
if [[ "$RAM_PLOT_ENABLE" == "1" ]]; then
  require_file "$RAM_PLOT_SCRIPT"
  require_exec "$RAM_PLOT_PYTHON_BIN"
fi

SOURCE_EXPDIR="${OUTBASE%/}/${SOURCE_EXP}"
TARGET_EXPDIR="${OUTBASE%/}/${TARGET_EXP}"
[[ -d "$SOURCE_EXPDIR" ]] || die "Source experiment does not exist: ${SOURCE_EXPDIR}"
if [[ "$TEST_MODE" != "1" ]]; then
  register_run_pid_file "$@"
fi

DATASET_CFG="$(resolve_dataset_cfg "$DS_INPUT")"
parse_dataset_yaml "$DATASET_CFG" "$DATASET_TYPE"
BASE_DIR_ABS="$(resolve_base_dir_abs "${DS_DATA_BASE_DIR}")"
[[ -d "$BASE_DIR_ABS" ]] || die "Missing base_dir directory: ${BASE_DIR_ABS}"

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
  [[ -n "$step5_raw_file" ]] || die "Could not resolve Step5 file '${step5_file_name}' from ${STEP5_FILE_SOURCE_LABEL}"
  STEP5_RAW_FILES_RESOLVED+=("$step5_raw_file")
done

if [[ -z "$OUT_LABEL" ]]; then
  OUT_LABEL="${DS_META_DATASET_ID}_${DS_DATASET_TYPE}"
fi
OUT_LABEL="$(sanitize_label "$OUT_LABEL")"

STEP5_SCRIPT="${PIPELINE_ROOT}/bin/step5_quant.sh"
require_file "$STEP5_SCRIPT"

mapfile -t SEARCH_RECORDS < <(discover_step5_searches "$SOURCE_EXPDIR")
(( ${#SEARCH_RECORDS[@]} > 0 )) || die "No existing Step5 commands found in ${SOURCE_EXPDIR}"

selected_count=0
ran_count=0

echo "Source experiment : ${SOURCE_EXPDIR}"
echo "Target experiment : ${TARGET_EXPDIR}"
echo "Dataset config    : ${DATASET_CFG}"
echo "Dataset type      : ${DS_DATASET_TYPE}"
echo "Step5 files src   : ${STEP5_FILE_SOURCE_LABEL}"
echo "Step5 files n     : ${#STEP5_RAW_FILES_RESOLVED[@]}"
echo "Output group      : 05_quant/${OUT_LABEL}"
echo "Resume            : $([[ "$RESUME_MODE" == "1" ]] && echo "on" || echo "off")"
echo "Verbose           : ${VERBOSE}"
echo "RAM log           : $([[ "$RAM_LOG_ENABLE" == "1" ]] && echo "on (${RAM_LOG_INTERVAL}s)" || echo "off")"
echo "RAM plot          : $([[ "$RAM_PLOT_ENABLE" == "1" ]] && echo "on" || echo "off")"
echo ""

for record in "${SEARCH_RECORDS[@]}"; do
  IFS=$'\036' read -r old_out_sub lib_file fasta_abs fasta_name set_blob <<< "$record"
  [[ -n "$old_out_sub" ]] || old_out_sub="baseline"

  if [[ "$old_out_sub" == "${OUT_LABEL}/"* ]]; then
    continue
  fi

  if ! search_selected "$old_out_sub"; then
    continue
  fi

  selected_count=$((selected_count + 1))
  lib_file_resolved="$(resolve_existing_lib_file "$lib_file" || true)"
  [[ -n "$lib_file_resolved" ]] || die "Missing library for search '${old_out_sub}': ${lib_file}"

  new_out_sub="$(new_out_sub_for_search "$old_out_sub" "$OUT_LABEL")"

  if [[ "$RESUME_MODE" == "1" ]] && report_exists_for_out_sub "$TARGET_EXPDIR" "$new_out_sub"; then
    info "Resume: skipping '${old_out_sub}' because report exists in 05_quant/${new_out_sub}"
    continue
  fi

  STEP5_CMD=(
    bash "$STEP5_SCRIPT"
    --expname "$TARGET_EXP"
    --lib "$lib_file_resolved"
    --out-sub "$new_out_sub"
  )

  if [[ -n "$fasta_abs" ]]; then
    require_file "$fasta_abs"
    STEP5_CMD+=(--fastaabs "$fasta_abs")
  elif [[ -n "$fasta_name" ]]; then
    STEP5_CMD+=(--fasta "$fasta_name")
  else
    die "Existing search '${old_out_sub}' has no FASTA argument in its Step5 command"
  fi

  for raw_file in "${STEP5_RAW_FILES_RESOLVED[@]}"; do
    STEP5_CMD+=(--f "$raw_file")
  done

  if (( VERBOSE > 0 )); then
    STEP5_CMD+=(--verbose "$VERBOSE")
  fi

  if [[ -n "$set_blob" ]]; then
    IFS=$'\037' read -r -a ORIGINAL_SETS <<< "$set_blob"
    for set_value in "${ORIGINAL_SETS[@]}"; do
      [[ -n "$set_value" ]] || continue
      set_key="${set_value%%=*}"
      if [[ "${set_key^^}" == "DATASET_TYPE" ]]; then
        continue
      fi
      STEP5_CMD+=(--set "$set_value")
    done
  fi

  for override in "${RERUN_OVERRIDES[@]}"; do
    override_key="${override%%=*}"
    if [[ "${override_key^^}" == "DATASET_TYPE" ]]; then
      continue
    fi
    STEP5_CMD+=(--set "$override")
  done

  STEP5_CMD+=(--set "DATASET_TYPE=${DS_DATASET_TYPE}")

  if [[ "$TEST_MODE" == "1" ]]; then
    STEP5_CMD+=(--test)
    printf 'Would rerun %-45s -> %s\n' "$old_out_sub" "$new_out_sub"
    printf '%q ' "${STEP5_CMD[@]}"
    echo
  else
    info "Rerunning search '${old_out_sub}' -> '${new_out_sub}'"
    progress_detail="group=${OUT_LABEL};search=${new_out_sub};source=${old_out_sub};mode=re-search;dataset=${DS_META_DATASET_ID};dataset_type=${DS_DATASET_TYPE}"
    append_progress_event "run_existing_lib_searches.sh" "START" "$progress_detail"
    set +e
    log_tag="step5_research_${OUT_LABEL}_${old_out_sub//\//_}"
    run_with_optional_ram_logger "$new_out_sub" "$log_tag" STEP5_CMD
    step_rc=$?
    set -e
    if (( step_rc == 0 )); then
      append_progress_event "run_existing_lib_searches.sh" "OK" "$progress_detail"
    else
      append_progress_event "run_existing_lib_searches.sh" "FAIL" "${progress_detail};exit=${step_rc}"
      exit "$step_rc"
    fi
  fi
  ran_count=$((ran_count + 1))
done

(( selected_count > 0 )) || die "No searches matched the requested --search filters"

if [[ "$TEST_MODE" == "1" ]]; then
  echo ""
  echo "OK: validation successful; ${ran_count} Step5 command(s) would run."
else
  echo ""
  echo "OK: completed ${ran_count} Step5 re-search command(s)."
fi
