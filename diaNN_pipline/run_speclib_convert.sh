#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${SCRIPT_DIR}"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
source "${PIPELINE_ROOT}/config/entrapment.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

usage() {
  cat <<'EOF'
Usage:
  run_speclib_convert.sh --lib <library.speclib> [options]

Required:
  --lib <path.speclib>       Input DIA-NN .speclib library.

Auto-detection:
  --expname <name|auto>      Optional override. If omitted, the script tries to
                             infer the experiment name from the experiment tree.
  --ds <dataset>             Optional override. If omitted, the dataset is read
                             from the experiment metadata when available.

Output:
  --out-lib <path.parquet>   Output parquet path. Default:
                             <OUTBASE>/<expname>/00_speclib_convert/<input_basename>.parquet

FASTA:
  --fasta <filename>         FASTA filename from FASTABASE.
  --fasta-abs <abs_path>     Absolute FASTA path.
  --fasta-kind <full|swiss>  Dataset YAML FASTA selector. Default: full.
  --no-fasta                 Convert without FASTA annotation.
  --fasta-entrap-shuffle <fixed-r|simple>
                             Entrapment FASTA naming mode when --fasta-entrap
                             is inferred or selected. Default: fixed-r.

Modifications:
  --mods <preset>            Load config/mods/<preset>.cfg. Default from defaults.cfg.
  --no-mods                  Do not pass fixed/variable modification flags.

Other:
  --threads <n>              Set THREADS for DIA-NN directly.
  --set KEY=VALUE            Override simple config variables, e.g. THREADS=8.
  --extra-flags "..."        Extra DIA-NN flags appended at the end.
  --test                     Print command only; do not run DIA-NN.
  -v | --verbose             Debug logging.
  -h | --help                Show this help.

Notes:
  The script prefers to infer experiment name, dataset, FASTA, and modification
  preset from the surrounding experiment folder (run_all.pid.meta).
  Use --fasta, --fasta-abs, --fasta-kind, --mods, or --no-fasta/--no-mods to
  override inferred values.
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

parse_dataset_yaml_fasta() {
  local cfg_file="$1"
  local py_bin="${PIPELINE_PYTHON_PATH:-python3}"

  require_exec "$py_bin"

  eval "$($py_bin - "$cfg_file" <<'PY'
import shlex
import sys

cfg_path = sys.argv[1]

try:
    import yaml
except Exception:
    print("echo 'ERROR: PyYAML is required for dataset parsing (pip install pyyaml)' >&2")
    print("exit 1")
    raise SystemExit(0)

with open(cfg_path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}

def get(obj, keys, default=""):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def q(value):
    return shlex.quote("" if value is None else str(value))

print(f"DS_META_DATASET_ID={q(get(data, ['meta', 'dataset_id']))}")
print(f"DS_FASTA={q(get(data, ['fasta', 'fasta']))}")
print(f"DS_FASTA_SWISS={q(get(data, ['fasta', 'fasta_swiss']))}")
print(f"DS_CONTAMINANTS_FASTA={q(get(data, ['fasta', 'contaminants_fasta']))}")
PY
)"
}

resolve_fasta_from_value() {
  local fasta_value="$1"
  local fasta_base

  [[ -n "$fasta_value" ]] || return 1

  if [[ "$fasta_value" == /* ]]; then
    echo "$fasta_value"
    return 0
  fi

  [[ "$fasta_value" != */* ]] || die "Dataset FASTA '${fasta_value}' contains a path. Use --fasta-abs to override."
  fasta_base="${FASTABASE%/}"
  [[ -n "$fasta_base" ]] || fasta_base="/"
  echo "${fasta_base}/${fasta_value}"
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

resolve_experiment_dir_from_path() {
  local start_path="$1"
  local dir

  if [[ -f "$start_path" ]]; then
    dir="$(cd "$(dirname "$start_path")" && pwd)"
  elif [[ -d "$start_path" ]]; then
    dir="$(cd "$start_path" && pwd)"
  else
    return 1
  fi

  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/run_all.pid.meta" || -f "$dir/progress.txt" || -f "$dir/all_startcmds.txt" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done

  return 1
}

load_experiment_metadata() {
  local expdir="$1"
  local meta_file="${expdir}/run_all.pid.meta"
  local cmd_line line

  META_EXPNAME=""
  META_DS_INPUT=""
  META_MOD_CONFIG=""
  META_FASTA_NAME=""
  META_FASTA_ABS=""
  META_FASTA_KIND=""
  META_FASTA_ENTRAP=0
  META_FASTA_ENTRAP_SHUFFLE_MODE=""
  META_NO_FASTA=0
  META_NO_MODS=0

  [[ -f "$meta_file" ]] || return 0

  cmd_line=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == cmd=* ]]; then
      cmd_line="${line#cmd=}"
      break
    fi
  done < "$meta_file"

  [[ -n "$cmd_line" ]] || return 0

  eval "set -- $cmd_line"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --expname)
        shift
        META_EXPNAME="${1:-}"
        ;;
      --ds)
        shift
        META_DS_INPUT="${1:-}"
        ;;
      --mods)
        shift
        META_MOD_CONFIG="${1:-}"
        ;;
      --set)
        shift
        case "${1:-}" in
          MOD_CONFIG=*)
            META_MOD_CONFIG="${1#*=}"
            ;;
          FASTA_ENTRAP_SHUFFLE_MODE=*)
            META_FASTA_ENTRAP_SHUFFLE_MODE="${1#*=}"
            ;;
        esac
        ;;
      --fasta)
        shift
        META_FASTA_NAME="${1:-}"
        ;;
      --fasta-abs|--fastaabs)
        shift
        META_FASTA_ABS="${1:-}"
        ;;
      --fasta-kind|--fasta-source)
        shift
        META_FASTA_KIND="${1:-}"
        ;;
      --fasta-entrap)
        META_FASTA_ENTRAP=1
        ;;
      --fasta-entrap=*)
        META_FASTA_ENTRAP=1
        META_FASTA_KIND="${1#*=}"
        ;;
      --fasta-entrap-shuffle)
        shift
        META_FASTA_ENTRAP_SHUFFLE_MODE="${1:-}"
        ;;
      --fasta-entrap-shuffle=*)
        META_FASTA_ENTRAP_SHUFFLE_MODE="${1#*=}"
        ;;
      --no-fasta)
        META_NO_FASTA=1
        ;;
      --no-mods)
        META_NO_MODS=1
        ;;
    esac
    shift
  done
}

resolve_speclib_input() {
  local lib_input="$1"
  local expdir="$2"
  local lib_name
  local -a candidates=()

  if [[ -f "$lib_input" ]]; then
    echo "$lib_input"
    return 0
  fi

  if [[ "$lib_input" == /* ]]; then
    die "Speclib path does not exist: ${lib_input}"
  fi

  if [[ -n "$expdir" && -f "${expdir}/${lib_input}" ]]; then
    echo "${expdir}/${lib_input}"
    return 0
  fi

  lib_name="$(basename "$lib_input")"
  if [[ -d "$expdir" ]]; then
    mapfile -t candidates < <(find "$expdir" -type f -name "$lib_name" | sort)
  fi

  if (( ${#candidates[@]} == 0 )); then
    die "Speclib not found: ${lib_input} (tried current directory and ${expdir})"
  fi

  if (( ${#candidates[@]} > 1 )); then
    warn "Multiple speclib candidates found for ${lib_name}, using first: ${candidates[0]}"
  fi

  echo "${candidates[0]}"
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

append_progress() {
  local state="$1"
  local details="$2"
  mkdir -p "$EXPDIR"
  printf '%s\tspeclib_convert\t%s\t%s\n' "$(date -Is)" "$state" "$details" >> "${EXPDIR}/progress.txt"
}

EXPNAME=""
DS_INPUT=""
LIB_IN=""
OUT_LIB=""
FASTA_NAME=""
FASTA_ABS=""
FASTA_KIND="full"
FASTA_ENTRAP=0
FASTA_ENTRAP_SHUFFLE_MODE="${FASTA_ENTRAP_SHUFFLE_MODE:-fixed-r}"
NO_FASTA=0
NO_MODS=0
TEST_MODE=0
EXTRA_FLAGS=""
CLI_OVERRIDES=()
EXPNAME_CLI_SET=0
DS_CLI_SET=0
FASTA_CLI_SET=0
MODS_CLI_SET=0
NO_FASTA_CLI_SET=0
NO_MODS_CLI_SET=0

ARGS=("$@")
i=0
while (( i < ${#ARGS[@]} )); do
  case "${ARGS[$i]}" in
    --expname)
      ((i+=1)); EXPNAME="${ARGS[$i]:-}"; EXPNAME_CLI_SET=1 ;;
    --ds)
      ((i+=1)); DS_INPUT="${ARGS[$i]:-}"; DS_CLI_SET=1 ;;
    --lib|--speclib)
      ((i+=1)); LIB_IN="${ARGS[$i]:-}" ;;
    --out-lib)
      ((i+=1)); OUT_LIB="${ARGS[$i]:-}" ;;
    --fasta)
      ((i+=1)); FASTA_NAME="${ARGS[$i]:-}"; FASTA_CLI_SET=1 ;;
    --fasta-abs|--fastaabs)
      ((i+=1)); FASTA_ABS="${ARGS[$i]:-}"; FASTA_CLI_SET=1 ;;
    --fasta-kind|--fasta-source)
      ((i+=1)); FASTA_KIND="${ARGS[$i]:-}"; FASTA_CLI_SET=1 ;;
    --fasta-entrap)
      FASTA_ENTRAP=1; FASTA_CLI_SET=1 ;;
    --fasta-entrap=*)
      FASTA_ENTRAP=1
      FASTA_KIND="${ARGS[$i]#*=}"
      FASTA_CLI_SET=1 ;;
    --fasta-entrap-shuffle)
      ((i+=1)); FASTA_ENTRAP_SHUFFLE_MODE="${ARGS[$i]:-}" ;;
    --fasta-entrap-shuffle=*)
      FASTA_ENTRAP_SHUFFLE_MODE="${ARGS[$i]#*=}" ;;
    --no-fasta)
      NO_FASTA=1; NO_FASTA_CLI_SET=1 ;;
    --threads)
      ((i+=1))
      THREADS="${ARGS[$i]:-}"
      [[ -n "$THREADS" ]] || die "--threads expects a value"
      [[ "$THREADS" =~ ^[0-9]+$ ]] || die "--threads must be numeric"
      CLI_OVERRIDES+=("THREADS") ;;
    --mods)
      ((i+=1)); MOD_CONFIG="${ARGS[$i]:-}"; MODS_CLI_SET=1; CLI_OVERRIDES+=("MOD_CONFIG") ;;
    --no-mods)
      NO_MODS=1; NO_MODS_CLI_SET=1 ;;
    --set)
      ((i+=1))
      kv="${ARGS[$i]:-}"
      [[ "$kv" == *=* ]] || die "--set expects KEY=VALUE"
      key="${kv%%=*}"
      val="${kv#*=}"
      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid --set key: $key"
      printf -v "$key" '%s' "$val"
      [[ "$key" == "MOD_CONFIG" ]] && MODS_CLI_SET=1
      [[ "$key" == "THREADS" ]] && CLI_OVERRIDES+=("THREADS")
      CLI_OVERRIDES+=("$key")
      ;;
    --extra-flags)
      ((i+=1)); EXTRA_FLAGS="${ARGS[$i]:-}" ;;
    --test|--dry-run)
      TEST_MODE=1 ;;
    -v|--v|--verbose)
      VERBOSE=1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "Unknown argument: ${ARGS[$i]}" ;;
  esac
  ((i+=1))
done

[[ -n "$LIB_IN" ]] || die "--lib is required"

require_exec "$DIANN_BIN"

SOURCE_EXPDIR=""
if SOURCE_EXPDIR="$(resolve_experiment_dir_from_path "$LIB_IN" 2>/dev/null)"; then
  :
fi
if [[ -z "$EXPNAME" && -n "$SOURCE_EXPDIR" ]]; then
  EXPNAME="$(basename "$SOURCE_EXPDIR")"
fi
if [[ -z "$EXPNAME" ]]; then
  EXPNAME="speclib_convert_$(date +'%Y_%m_%d_%H_%M_%S')"
fi
export EXPNAME

if [[ -n "$SOURCE_EXPDIR" ]]; then
  load_experiment_metadata "$SOURCE_EXPDIR"
fi

if [[ -z "$DS_INPUT" ]]; then
  DS_INPUT="${META_DS_INPUT:-}"
fi
if [[ -z "$DS_INPUT" ]]; then
  die "Could not infer --ds from the experiment metadata; pass --ds explicitly"
fi

if (( NO_MODS_CLI_SET == 0 )) && [[ -n "$META_MOD_CONFIG" ]] && [[ "$MODS_CLI_SET" == "0" ]]; then
  MOD_CONFIG="$META_MOD_CONFIG"
fi

if (( NO_FASTA_CLI_SET == 0 )) && [[ "${META_NO_FASTA:-0}" == "1" ]]; then
  NO_FASTA=1
fi
if (( FASTA_CLI_SET == 0 )); then
  if [[ -n "$META_FASTA_NAME" ]]; then
    FASTA_NAME="$META_FASTA_NAME"
  elif [[ -n "$META_FASTA_ABS" ]]; then
    FASTA_ABS="$META_FASTA_ABS"
  elif [[ -n "$META_FASTA_KIND" ]]; then
    FASTA_KIND="$META_FASTA_KIND"
  fi
  if (( META_FASTA_ENTRAP == 1 )); then
    FASTA_ENTRAP=1
  fi
  if [[ -n "$META_FASTA_ENTRAP_SHUFFLE_MODE" ]]; then
    FASTA_ENTRAP_SHUFFLE_MODE="$META_FASTA_ENTRAP_SHUFFLE_MODE"
  fi
fi
if [[ -z "$FASTA_KIND" ]]; then
  FASTA_KIND="full"
fi

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

DATASET_CFG="$(resolve_dataset_cfg "$DS_INPUT")"
parse_dataset_yaml_fasta "$DATASET_CFG"

if [[ "$FASTA_KIND" != "full" && "$FASTA_KIND" != "swiss" ]]; then
  die "Invalid --fasta-kind '${FASTA_KIND}' (allowed: full|swiss)"
fi
if [[ -n "$FASTA_NAME" && -n "$FASTA_ABS" ]]; then
  die "Use either --fasta or --fasta-abs, not both"
fi
if (( NO_FASTA == 1 )) && { [[ -n "$FASTA_NAME" ]] || [[ -n "$FASTA_ABS" ]] || (( FASTA_ENTRAP == 1 )); }; then
  die "Use either --no-fasta or --fasta/--fasta-abs, not both"
fi

FASTA_FILE=""
if (( NO_FASTA == 0 )); then
  if (( FASTA_ENTRAP == 1 )); then
    if [[ "$FASTA_KIND" == "full" ]]; then
      FASTA_FROM_YAML="${DS_FASTA:-}"
    else
      FASTA_FROM_YAML="${DS_FASTA_SWISS:-}"
    fi
    [[ -n "$FASTA_FROM_YAML" ]] || die "Dataset YAML has no FASTA for --fasta-kind ${FASTA_KIND}; use --fasta/--fasta-abs or --no-fasta"
    [[ "$FASTA_FROM_YAML" != */* ]] || die "Dataset FASTA '${FASTA_FROM_YAML}' must be a FASTABASE filename"
    FASTA_FILE="$(resolve_fasta_from_value "$(entrapment_filename_for_fasta "$FASTA_FROM_YAML" "$FASTA_ENTRAP_SHUFFLE_MODE")")"
    if [[ ! -f "$FASTA_FILE" && "$FASTA_ENTRAP_SHUFFLE_MODE" == "fixed-r" ]]; then
      legacy_fasta_file="$(resolve_fasta_from_value "$(entrapment_filename_for_fasta "$FASTA_FROM_YAML" simple)")"
      if [[ -f "$legacy_fasta_file" ]]; then
        warn "Fixed-R entrapment FASTA not found; using legacy simple entrapment FASTA: ${legacy_fasta_file}"
        FASTA_FILE="$legacy_fasta_file"
      fi
    fi
  elif [[ -n "$FASTA_NAME" ]]; then
    [[ "$FASTA_NAME" != */* ]] || die "--fasta expects a FASTABASE filename, not a path"
    FASTA_FILE="$(resolve_fasta_from_value "$FASTA_NAME")"
  elif [[ -n "$FASTA_ABS" ]]; then
    [[ "$FASTA_ABS" == /* ]] || die "--fasta-abs expects an absolute path"
    FASTA_FILE="$FASTA_ABS"
  else
    if [[ "$FASTA_KIND" == "full" ]]; then
      FASTA_FROM_YAML="${DS_FASTA:-}"
    else
      FASTA_FROM_YAML="${DS_FASTA_SWISS:-}"
    fi
    [[ -n "$FASTA_FROM_YAML" ]] || die "Dataset YAML has no FASTA for --fasta-kind ${FASTA_KIND}; use --fasta/--fasta-abs or --no-fasta"
    FASTA_FILE="$(resolve_fasta_from_value "$FASTA_FROM_YAML")"
  fi
  require_file "$FASTA_FILE"
fi

load_mod_config

outbase="${OUTBASE%/}"
[[ -n "$outbase" ]] || outbase="/"
if [[ -z "$SOURCE_EXPDIR" ]]; then
  EXPDIR="${outbase}/${EXPNAME}"
else
  EXPDIR="$SOURCE_EXPDIR"
fi
CONVERT_DIR="${EXPDIR}/00_speclib_convert"

LIB_IN="$(resolve_speclib_input "$LIB_IN" "$SOURCE_EXPDIR")"
require_file "$LIB_IN"

if (( TEST_MODE == 0 )); then
  mkdir -p "$CONVERT_DIR"
fi

if [[ -z "$OUT_LIB" ]]; then
  lib_base="$(basename "$LIB_IN")"
  lib_stem="${lib_base%.*}"
  OUT_LIB="${CONVERT_DIR}/${lib_stem}.parquet"
elif [[ "$OUT_LIB" != /* ]]; then
  OUT_LIB="${CONVERT_DIR}/${OUT_LIB}"
fi
if (( TEST_MODE == 0 )); then
  mkdir -p "$(dirname "$OUT_LIB")"
fi

DIANN_CMD=(
  "$DIANN_BIN"
  --threads "$THREADS"
  --lib "$LIB_IN"
  --out-lib "$OUT_LIB"
)

if [[ -n "$FASTA_FILE" ]]; then
  DIANN_CMD+=(--fasta "$FASTA_FILE")
fi

if (( NO_MODS == 0 )); then
  if [[ "${enable_UNIMOD4:-0}" == "1" ]]; then
    DIANN_CMD+=(--unimod4)
  fi
  append_mod_args DIANN_CMD
fi

if [[ -n "${EXTRA_FLAGS:-}" ]]; then
  read -r -a EXTRA_FLAGS_ARR <<< "$EXTRA_FLAGS"
  DIANN_CMD+=("${EXTRA_FLAGS_ARR[@]}")
fi

STARTCMD_FILE="${CONVERT_DIR}/startcmd.txt"
LOG_FILE="${CONVERT_DIR}/speclib_convert.log"

if (( TEST_MODE == 0 )); then
  {
    echo "# $(date -Is)"
    echo "# dataset: ${DATASET_CFG}"
    echo "# experiment: ${EXPNAME}"
    echo "# input: ${LIB_IN}"
    echo "# output: ${OUT_LIB}"
    echo "# fasta: ${FASTA_FILE:-<none>}"
    echo "# mods: $([[ "$NO_MODS" == "1" ]] && echo '<none>' || mod_config_summary)"
    quote_cmd "${DIANN_CMD[@]}"
  } > "$STARTCMD_FILE"
fi

echo "Experiment : ${EXPNAME}"
echo "Dataset    : ${DATASET_CFG}"
echo "Input lib  : ${LIB_IN}"
echo "Output lib : ${OUT_LIB}"
echo "FASTA      : ${FASTA_FILE:-<none>}"
if (( NO_MODS == 0 )); then
  print_mod_config_details
else
  echo "Modification config: <disabled>"
fi
echo "Startcmd   : ${STARTCMD_FILE}$([[ "$TEST_MODE" == "1" ]] && echo ' (not written in test mode)')"
echo "Log        : ${LOG_FILE}$([[ "$TEST_MODE" == "1" ]] && echo ' (not written in test mode)')"
echo
echo "DIA-NN command:"
quote_cmd "${DIANN_CMD[@]}"

if (( TEST_MODE == 1 )); then
  echo
  echo "TEST MODE: DIA-NN was not run."
  exit 0
fi

append_progress "START" "lib=${LIB_IN};out_lib=${OUT_LIB};fasta=${FASTA_FILE:-none}"
set +e
"${DIANN_CMD[@]}" > >(tee "$LOG_FILE") 2>&1
rc=$?
set -e

if (( rc != 0 )); then
  append_progress "FAIL" "exit=${rc};lib=${LIB_IN};out_lib=${OUT_LIB}"
  die "DIA-NN speclib conversion failed with exit code ${rc}; see ${LOG_FILE}"
fi

require_file "$OUT_LIB"
append_progress "OK" "lib=${LIB_IN};out_lib=${OUT_LIB};fasta=${FASTA_FILE:-none}"
echo "OK: converted library written to ${OUT_LIB}"
