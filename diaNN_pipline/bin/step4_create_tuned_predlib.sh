#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

warn(){ _log "WARN" "$*"; }

normalize_instrument_name() {
  local raw="$1"
  local compact
  compact="${raw,,}"
  compact="${compact//[[:space:]_-]/}"
  case "$compact" in
    timstof|timstofpro|timstofpro2)
      echo "timstof"
      ;;
    elite)
      echo "elite"
      ;;
    exploris)
      echo "exploris"
      ;;
    qe|qexactive|qexactivehf|qexactivehfx)
      echo "qe"
      ;;
    *)
      echo "$compact"
      ;;
  esac
}

load_cfg_relaxed() {
  local cfg_file="$1"
  local line raw key value safe_key assignment
  while IFS= read -r line || [[ -n "$line" ]]; do
    raw="${line#${line%%[![:space:]]*}}"
    [[ -z "$raw" ]] && continue
    [[ "$raw" == \#* ]] && continue
    [[ "$raw" == *"="* ]] || continue

    key="${raw%%=*}"
    value="${raw#*=}"
    key="${key%${key##*[![:space:]]}}"
    key="${key#${key%%[![:space:]]*}}"
    safe_key="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_' '_')"
    [[ "$safe_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    assignment="${safe_key}=${value}"
    eval "$assignment"
  done < "$cfg_file"
}

load_cfg_relaxed "${PIPELINE_ROOT}/config/step4.cfg"

warn_unmapped_enable_flags() {
  local key value
  for key in $(compgen -A variable | grep '^enable_'); do
    value="${!key:-0}"
    [[ "$value" == "1" ]] || continue

    case "$key" in
      enable_UNIMOD4|enable_REANNOTATE|enable_SEMI|enable_PEPTIDOFORMS|enable_FASTA_SEARCH|enable_MATRICES|enable_GEN_SPEC_LIB|enable_CONT_QUANT_EXQCLUDE_CRAP|enable_REANALYSE|enable_RT_PROFILING)
        ;;
      *)
        warn "enable-flag '$key=1' ist gesetzt, aber in step4_create_tuned_predlib.sh nicht auf ein DIA-NN Argument gemappt."
        ;;
    esac
  done
}

warn_unmapped_enable_flags

auto_detect_predlib() {
  local search_dir="$1"
  local expname="$2"
  local default_candidates=()
  local strict_candidates=()

  default_candidates=(
    "${search_dir}/step1_${expname}_predicted.parquet"
    "${search_dir}/step1_${expname}_predicted.predicted.speclib"
    "${search_dir}/step1_${expname}_predicted.speclib"
  )

  local path
  for path in "${default_candidates[@]}"; do
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done

  shopt -s nullglob
  strict_candidates=(
    "${search_dir}"/*.predicted.parquet
    "${search_dir}"/*.predicted.speclib
  )
  shopt -u nullglob

  local uniq=()
  local seen existing
  for path in "${strict_candidates[@]}"; do
    seen=0
    for existing in "${uniq[@]:-}"; do
      if [[ "$existing" == "$path" ]]; then
        seen=1
        break
      fi
    done
    if [[ "$seen" == "0" ]]; then
      uniq+=("$path")
    fi
  done

  if [[ ${#uniq[@]} -eq 1 ]]; then
    printf '%s\n' "${uniq[0]}"
    return 0
  fi

  if [[ ${#uniq[@]} -eq 0 ]]; then
    die "No predicted library found in ${search_dir} (tried expname defaults and *.predicted.parquet/*.predicted.speclib)."
  fi

  die "Multiple predicted libraries found in ${search_dir}; keep one or specify exactly one predicted library file in that folder."
}

require_abs_path() {
  local path="$1"
  local label="$2"
  [[ "$path" == /* ]] || die "${label} must be an absolute path, got: ${path}"
}

find_single_required_in_dir() {
  local dir="$1"
  local pattern="$2"
  local label="$3"
  local matches=()

  shopt -s nullglob
  matches=("${dir}"/${pattern})
  shopt -u nullglob

  if [[ ${#matches[@]} -eq 1 ]]; then
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  if [[ ${#matches[@]} -eq 0 ]]; then
    die "Missing ${label} in ${dir} (expected pattern: ${pattern})"
  fi
  die "Ambiguous ${label} in ${dir}: found ${#matches[@]} files for pattern ${pattern}."
}

find_single_optional_in_dir() {
  local dir="$1"
  local pattern="$2"
  local label="$3"
  local matches=()

  shopt -s nullglob
  matches=("${dir}"/${pattern})
  shopt -u nullglob

  if [[ ${#matches[@]} -eq 1 ]]; then
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  if [[ ${#matches[@]} -eq 0 ]]; then
    printf '%s\n' ""
    return 0
  fi
  die "Ambiguous ${label} in ${dir}: found ${#matches[@]} files for pattern ${pattern}."
}

CLI_OVERRIDES=()
FASTA_NAME=""
FASTA_ABS=""
MODELS_DIR_INPUT=""
TOKENS_INPUT=""
RT_MODEL_INPUT=""
IM_MODEL_INPUT=""
FR_MODEL_INPUT=""
VERBOSE="${VERBOSE:-0}"
TEST_MODE=0
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --fasta)
      ((i+=1))
      FASTA_NAME="${ARGS[$i]:-}"
      [[ -n "$FASTA_NAME" ]] || die "--fasta needs a value"
      ;;
    --fastaabs)
      ((i+=1))
      FASTA_ABS="${ARGS[$i]:-}"
      [[ -n "$FASTA_ABS" ]] || die "--fastaabs needs a value"
      ;;
    --models-dir)
      ((i+=1))
      MODELS_DIR_INPUT="${ARGS[$i]:-}"
      [[ -n "$MODELS_DIR_INPUT" ]] || die "--models-dir needs a value"
      ;;
    --tokens)
      ((i+=1))
      TOKENS_INPUT="${ARGS[$i]:-}"
      [[ -n "$TOKENS_INPUT" ]] || die "--tokens needs a value"
      ;;
    --rt-model)
      ((i+=1))
      RT_MODEL_INPUT="${ARGS[$i]:-}"
      [[ -n "$RT_MODEL_INPUT" ]] || die "--rt-model needs a value"
      ;;
    --im-model)
      ((i+=1))
      IM_MODEL_INPUT="${ARGS[$i]:-}"
      [[ -n "$IM_MODEL_INPUT" ]] || die "--im-model needs a value"
      ;;
    --fr-model)
      ((i+=1))
      FR_MODEL_INPUT="${ARGS[$i]:-}"
      [[ -n "$FR_MODEL_INPUT" ]] || die "--fr-model needs a value"
      ;;
    --set)
      ((i+=1))
      override="${ARGS[$i]:-}"
      [[ -n "$override" ]] || die "--set needs KEY=VALUE"
      [[ "$override" == *=* ]] || die "--set expects KEY=VALUE"
      key="${override%%=*}"
      value="${override#*=}"
      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid --set key '$key' (use letters, 0-9, _)"
      printf -v "$key" '%s' "$value"
      CLI_OVERRIDES+=("$key")
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
    --test)
      TEST_MODE=1
      ;;
    --expname)
      ((i+=1))
      [[ -n "${ARGS[$i]:-}" ]] || die "--expname needs a value"
      ;;
    -h|--help)
      sed -n '1,220p' "$0"
      exit 0
      ;;
    --lib|--ref)
      die "Do not pass --lib/--ref in Step 4. --lib is always taken from config (LIB, default empty) like in Step 1."
      ;;
    *)
      die "Unknown argument: ${ARGS[$i]}"
      ;;
  esac
  ((i+=1))
done

DIANN_VERBOSE="${DIANN_VERBOSE:-${VERBOSE:-0}}"
[[ "$DIANN_VERBOSE" =~ ^[0-4]$ ]] || die "Invalid verbose level '$DIANN_VERBOSE' (allowed: 0..4)"

if (( DIANN_VERBOSE > 0 )); then
  VERBOSE=1
else
  VERBOSE=0
fi
export VERBOSE

for key in "${CLI_OVERRIDES[@]}"; do
  debug "CLI override applied: ${key}=${!key}"
done
refresh_diann_paths_from_bin "${CLI_OVERRIDES[@]}"
load_mod_config

parse_expname "$@"

if [[ "$TEST_MODE" == "1" ]]; then
  : "${OUTBASE:?OUTBASE not set}"
  outbase="${OUTBASE%/}"
  [[ -n "$outbase" ]] || outbase="/"
  EXPDIR="${outbase}/${EXPNAME}"
  S3DIR="${EXPDIR}/03_tune_models"
  STEP4_BASEDIR_NAME="${STEP4_BASEDIR_NAME:-04_refined_lib}"
  S4ROOT="${EXPDIR}/${STEP4_BASEDIR_NAME}"
  STEP4_SUBDIR="${STEP4_SUBDIR:-}"
  if [[ -n "$STEP4_SUBDIR" ]]; then
    S4DIR="${S4ROOT}/${STEP4_SUBDIR}"
  else
    S4DIR="$S4ROOT"
  fi
  export EXPDIR S3DIR S4ROOT S4DIR
else
  set_paths_for_exp
  STEP4_BASEDIR_NAME="${STEP4_BASEDIR_NAME:-04_refined_lib}"
  S4ROOT="${EXPDIR}/${STEP4_BASEDIR_NAME}"
  STEP4_SUBDIR="${STEP4_SUBDIR:-}"
  if [[ -n "$STEP4_SUBDIR" ]]; then
    S4DIR="${S4ROOT}/${STEP4_SUBDIR}"
  else
    S4DIR="$S4ROOT"
  fi
  export S4ROOT S4DIR
fi

require_exec "$DIANN_BIN"

if [[ -n "$FASTA_NAME" && -n "$FASTA_ABS" ]]; then
  die "Use either --fasta <filename in FASTABASE> or --fastaabs <absolute path>, not both"
fi

if [[ -n "$FASTA_NAME" ]]; then
  [[ "$FASTA_NAME" != */* ]] || die "--fasta expects only a filename (from FASTABASE), no path separators"
  fasta_base="${FASTABASE%/}"
  [[ -n "$fasta_base" ]] || fasta_base="/"
  FASTA_FILE="${fasta_base}/${FASTA_NAME}"
elif [[ -n "$FASTA_ABS" ]]; then
  [[ "$FASTA_ABS" == /* ]] || die "--fastaabs expects an absolute path"
  FASTA_FILE="$FASTA_ABS"
else
  die "Missing FASTA for Step 4. Pass --fasta <filename> or --fastaabs <absolute path>."
fi
require_file "$FASTA_FILE"

TOKENS_RAW="${TOKENS_INPUT:-${TOKENS:-}}"
RT_MODEL_RAW="${RT_MODEL_INPUT:-${RT_MODEL:-}}"
IM_MODEL_RAW="${IM_MODEL_INPUT:-${IM_MODEL:-}}"
FR_MODEL_RAW="${FR_MODEL_INPUT:-${FR_MODEL:-}}"
MODELS_DIR_RAW="${MODELS_DIR_INPUT:-${MODELS_DIR:-}}"

if [[ -z "$MODELS_DIR_RAW" && -z "$TOKENS_RAW" && -z "$RT_MODEL_RAW" && -z "$IM_MODEL_RAW" && -z "$FR_MODEL_RAW" ]]; then
  die "Missing model input. Use --models-dir <abs_path> OR provide --tokens plus model path(s)."
fi

if [[ -n "$MODELS_DIR_RAW" && ( -n "$TOKENS_RAW" || -n "$RT_MODEL_RAW" || -n "$IM_MODEL_RAW" || -n "$FR_MODEL_RAW" ) ]]; then
  die "Use either --models-dir (or MODELS_DIR) OR explicit --tokens/--*-model inputs, not both."
fi

TOKENS_FILE=""
RT_MODEL_FILE=""
IM_MODEL_FILE=""
FR_MODEL_FILE=""

if [[ -n "$MODELS_DIR_RAW" ]]; then
  require_abs_path "$MODELS_DIR_RAW" "MODELS_DIR"
  [[ -d "$MODELS_DIR_RAW" ]] || die "MODELS_DIR does not exist or is not a directory: $MODELS_DIR_RAW"

  TOKENS_FILE="$(find_single_required_in_dir "$MODELS_DIR_RAW" "*.dict.txt" "tokens file")"
  RT_MODEL_FILE="$(find_single_optional_in_dir "$MODELS_DIR_RAW" "*.tuned_rt.pt" "RT model")"
  IM_MODEL_FILE="$(find_single_optional_in_dir "$MODELS_DIR_RAW" "*.tuned_im.pt" "IM model")"
  FR_MODEL_FILE="$(find_single_optional_in_dir "$MODELS_DIR_RAW" "*.tuned_fr.pt" "FR model")"

  [[ -n "$RT_MODEL_FILE" || -n "$IM_MODEL_FILE" || -n "$FR_MODEL_FILE" ]] || die "No model files found in ${MODELS_DIR_RAW}. Need at least one of: *.tuned_rt.pt, *.tuned_im.pt, *.tuned_fr.pt"
else
  [[ -n "$TOKENS_RAW" ]] || die "Missing tokens input. Provide --tokens <abs_path> or set TOKENS in config/step4.cfg"
  require_abs_path "$TOKENS_RAW" "TOKENS"
  TOKENS_FILE="$TOKENS_RAW"
  require_file "$TOKENS_FILE"

  if [[ -n "$RT_MODEL_RAW" ]]; then
    require_abs_path "$RT_MODEL_RAW" "RT_MODEL"
    RT_MODEL_FILE="$RT_MODEL_RAW"
    require_file "$RT_MODEL_FILE"
  fi
  if [[ -n "$IM_MODEL_RAW" ]]; then
    require_abs_path "$IM_MODEL_RAW" "IM_MODEL"
    IM_MODEL_FILE="$IM_MODEL_RAW"
    require_file "$IM_MODEL_FILE"
  fi
  if [[ -n "$FR_MODEL_RAW" ]]; then
    require_abs_path "$FR_MODEL_RAW" "FR_MODEL"
    FR_MODEL_FILE="$FR_MODEL_RAW"
    require_file "$FR_MODEL_FILE"
  fi

  [[ -n "$RT_MODEL_FILE" || -n "$IM_MODEL_FILE" || -n "$FR_MODEL_FILE" ]] || die "No model provided. Set at least one of: RT_MODEL, IM_MODEL, FR_MODEL"
fi

INSTRUMENT_VALUE="${INSTRUMENT:-${ACQUISITION_INSTRUMENT:-}}"
INSTRUMENT_NORMALIZED=""
if [[ -n "$INSTRUMENT_VALUE" ]]; then
  INSTRUMENT_NORMALIZED="$(normalize_instrument_name "$INSTRUMENT_VALUE")"
fi

if [[ -n "$IM_MODEL_FILE" && "$INSTRUMENT_NORMALIZED" != "timstof" ]]; then
  warn "IM model supplied but instrument is '${INSTRUMENT_NORMALIZED:-unset}' (not timstof); ignoring IM model for Step4"
  IM_MODEL_FILE=""
fi

ENABLE_PRE_SEARCH="${ENABLE_PRE_SEARCH:-${enable_PRE_SEARCH:-${enable_pre_search:-0}}}"
[[ "$ENABLE_PRE_SEARCH" =~ ^[01]$ ]] || die "Invalid ENABLE_PRE_SEARCH '$ENABLE_PRE_SEARCH' (allowed 0 or 1)"
ENABLE_GEN_SPEC_LIB="${ENABLE_GEN_SPEC_LIB:-${enable_GEN_SPEC_LIB:-0}}"
[[ "$ENABLE_GEN_SPEC_LIB" =~ ^[01]$ ]] || die "Invalid ENABLE_GEN_SPEC_LIB '$ENABLE_GEN_SPEC_LIB' (allowed 0 or 1)"

MASS_ACC_VALUE="${MASS_ACC_MS2:-${MASS_ACC:-20}}"
MASS_ACC_MS1_VALUE="${MASS_ACC_MS1:-${MASS_ACC:-10}}"

STEP4_NAME_SUFFIX="${STEP4_NAME_SUFFIX:-}"
if [[ -n "$STEP4_NAME_SUFFIX" && "${STEP4_NAME_SUFFIX:0:1}" != "_" ]]; then
  STEP4_NAME_SUFFIX="_${STEP4_NAME_SUFFIX}"
fi

OUTLIB="${OUTLIB:-${S4DIR}/refined_lib${STEP4_NAME_SUFFIX}.predicted.speclib}"
REPORT="${REPORT:-${S4DIR}/refined_lib_report${STEP4_NAME_SUFFIX}.parquet}"
STEP4_TEMP_DIR="${S4DIR}/Temp"
STARTCMD_FILE="${S4DIR}/startcmd${STEP4_NAME_SUFFIX}.txt"
STEP4_LOG_FILE="${S4DIR}/step4_create_tuned_predlib${STEP4_NAME_SUFFIX}.log"

if [[ "$TEST_MODE" != "1" ]]; then
  mkdir -p "$S4DIR" "$STEP4_TEMP_DIR"
else
  outbase_resolved="${OUTBASE%/}"
  [[ -n "$outbase_resolved" ]] || outbase_resolved="/"
  [[ -d "$outbase_resolved" ]] || die "OUTBASE does not exist (test mode): $outbase_resolved"
  [[ -w "$outbase_resolved" ]] || die "OUTBASE is not writable (test mode): $outbase_resolved"
  if [[ -e "$S4DIR" && ! -d "$S4DIR" ]]; then
    die "Step4 output path exists but is not a directory (test mode): $S4DIR"
  fi
fi

DIANN_CMD=(
  "$DIANN_BIN"
  --threads "$THREADS"
  --verbose "$DIANN_VERBOSE"
  --predictor
  --lib "${LIB:-}"
  --out-lib "$OUTLIB"
  --out "$REPORT"
  --temp "$STEP4_TEMP_DIR"
  --fasta "$FASTA_FILE"
  --tokens "$TOKENS_FILE"
  --qvalue "$QVALUE"
  --cut "$CUT"
  --missed-cleavages "$MISSED_CLEAVAGES"
  --min-pep-len "$MIN_PEP_LEN" --max-pep-len "$MAX_PEP_LEN"
  --min-pr-mz "$MIN_PR_MZ" --max-pr-mz "$MAX_PR_MZ"
  --min-pr-charge "$MIN_PR_CHARGE" --max-pr-charge "$MAX_PR_CHARGE"
  --min-fr-mz "$MIN_FR_MZ" --max-fr-mz "$MAX_FR_MZ"
  --mass-acc-ms1 "$MASS_ACC_MS1_VALUE" --mass-acc "$MASS_ACC_VALUE"
)

if [[ -n "$RT_MODEL_FILE" ]]; then
  DIANN_CMD+=(--rt-model "$RT_MODEL_FILE")
fi
if [[ -n "$IM_MODEL_FILE" ]]; then
  DIANN_CMD+=(--im-model "$IM_MODEL_FILE")
fi
if [[ -n "$FR_MODEL_FILE" ]]; then
  DIANN_CMD+=(--fr-model "$FR_MODEL_FILE")
fi
if [[ "$ENABLE_PRE_SEARCH" == "1" ]]; then
  DIANN_CMD+=(--pre-search)
fi
if [[ "${enable_MATRICES:-0}" == "1" ]]; then
  DIANN_CMD+=(--matrices)
fi
if [[ "$ENABLE_GEN_SPEC_LIB" == "1" ]]; then
  DIANN_CMD+=(--gen-spec-lib)
fi
if [[ "${enable_REANNOTATE:-0}" == "1" ]]; then
  DIANN_CMD+=(--reannotate)
fi
if [[ "${enable_FASTA_SEARCH:-0}" == "1" ]]; then
  DIANN_CMD+=(--fasta-search)
fi
if [[ "${enable_CONT_QUANT_EXQCLUDE_CRAP:-0}" == "1" ]]; then
  DIANN_CMD+=(--cont-quant-exclude cRAP-)
fi
if [[ "${enable_SEMI:-0}" == "1" ]]; then
  DIANN_CMD+=(--semi)
fi
if [[ "${enable_PEPTIDOFORMS:-0}" == "1" ]]; then
  DIANN_CMD+=(--peptidoforms)
fi
if [[ "${enable_REANALYSE:-0}" == "1" ]]; then
  DIANN_CMD+=(--reanalyse)
fi
if [[ "${enable_RT_PROFILING:-0}" == "1" ]]; then
  DIANN_CMD+=(--rt-profiling)
fi
if [[ "${enable_UNIMOD4:-0}" == "1" ]]; then
  DIANN_CMD+=(--unimod4)
fi

append_mod_args DIANN_CMD

if [[ -n "${EXTRA_FLAGS:-}" ]]; then
  read -r -a EXTRA_FLAGS_ARR <<< "$EXTRA_FLAGS"
  DIANN_CMD+=("${EXTRA_FLAGS_ARR[@]}")
fi

if [[ "$TEST_MODE" == "1" ]]; then
  emit_test_output() {
    echo "TEST MODE: no directories/files will be created; DIA-NN will not run."
    echo "DIANN_BIN   : $DIANN_BIN"
    echo "EXPNAME     : ${EXPNAME:-}"
    echo "LIB         : ${LIB:-}"
    echo "FASTA       : $FASTA_FILE"
    echo "TOKENS      : $TOKENS_FILE"
    if [[ -n "$MODELS_DIR_RAW" ]]; then
      echo "MODELS_DIR  : $MODELS_DIR_RAW"
    else
      echo "MODELS_DIR  : (not used)"
    fi
    if [[ -n "$RT_MODEL_FILE" ]]; then echo "RT_MODEL    : $RT_MODEL_FILE"; else echo "RT_MODEL    : (not set/found)"; fi
    if [[ -n "$IM_MODEL_FILE" ]]; then echo "IM_MODEL    : $IM_MODEL_FILE"; else echo "IM_MODEL    : (not set/found)"; fi
    if [[ -n "$FR_MODEL_FILE" ]]; then echo "FR_MODEL    : $FR_MODEL_FILE"; else echo "FR_MODEL    : (not set/found)"; fi
    echo "PRE_SEARCH  : $ENABLE_PRE_SEARCH"
    echo "GEN_SPEC_LIB: $ENABLE_GEN_SPEC_LIB"
    echo "OUTDIR      : $S4DIR"
    echo "OUTLIB      : $OUTLIB"
    echo "REPORT      : $REPORT"
    echo "TMPDIR      : $STEP4_TEMP_DIR"
    echo "DIA-NN command (without --test):"
    printf '%q ' "${DIANN_CMD[@]}"
    echo
    echo "OK: validation successful (test mode)."
  }

  if [[ -t 1 ]] && command -v less >/dev/null 2>&1; then
    emit_test_output | less
  else
    emit_test_output
  fi
  exit 0
fi

exec > >(tee -a "${STEP4_LOG_FILE}") 2>&1

{
  echo "# script invocation command"
  printf '%q ' "$0"
  printf '%q ' "$@"
  echo
  echo "# step4_create_tuned_predlib start command"
  printf '%q ' "${DIANN_CMD[@]}"
  echo
} > "$STARTCMD_FILE"

info "Step4 (create tuned predlib) started"
info "LIB=${LIB:-}"
info "FASTA=${FASTA_FILE}"
info "TOKENS=${TOKENS_FILE}"
if [[ -n "$MODELS_DIR_RAW" ]]; then
  info "MODELS_DIR=${MODELS_DIR_RAW}"
fi
if [[ -n "$RT_MODEL_FILE" ]]; then info "RT_MODEL=${RT_MODEL_FILE}"; else info "RT_MODEL not provided"; fi
if [[ -n "$IM_MODEL_FILE" ]]; then info "IM_MODEL=${IM_MODEL_FILE}"; else info "IM_MODEL not provided"; fi
if [[ -n "$FR_MODEL_FILE" ]]; then info "FR_MODEL=${FR_MODEL_FILE}"; else info "FR_MODEL not provided"; fi
info "INSTRUMENT=${INSTRUMENT_VALUE:-<unset>} (normalized=${INSTRUMENT_NORMALIZED:-<unset>})"
info "Saved start command: ${STARTCMD_FILE}"

"${DIANN_CMD[@]}"

info "Step4 (create tuned predlib) finished"
