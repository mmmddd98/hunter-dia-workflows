#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
source "${PIPELINE_ROOT}/config/cal.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

warn(){ _log "WARN" "$*"; }

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

load_cfg_relaxed "${PIPELINE_ROOT}/config/step5.cfg"

warn_unmapped_enable_flags() {
  local key value
  for key in $(compgen -A variable | grep '^enable_'); do
    value="${!key:-0}"
    [[ "$value" == "1" ]] || continue

    case "$key" in
      enable_UNIMOD4|enable_REANNOTATE|enable_SEMI|enable_PEPTIDOFORMS|enable_FASTA_SEARCH|enable_MATRICES|enable_GEN_SPEC_LIB|enable_CONT_QUANT_EXQCLUDE_CRAP|enable_REANALYSE|enable_RT_PROFILING|enable_PRE_SEARCH|enable_pre_search|enable_REPORT_DECOYS|enable_eval_run_arabidopsis_reference|enable_eval_run_ecoli_centered)
        ;;
      *)
        warn "enable-flag '$key=1' is set but not mapped in step5_quant.sh."
        ;;
    esac
  done
}

warn_unmapped_enable_flags

resolve_raw_input() {
  local input="$1"
  if [[ "$input" == /* ]]; then
    echo "$input"
  else
    local raw_base="${RAWBASE%/}"
    [[ -n "$raw_base" ]] || raw_base="/"
    echo "${raw_base}/${input}"
  fi
}

resolve_lib_input() {
  local input="$1"
  if [[ "$input" == /* ]]; then
    echo "$input"
  elif [[ "$input" == */* ]]; then
    echo "$input"
  else
    echo "${S1DIR}/${input}"
  fi
}

CLI_OVERRIDES=()
RAW_INPUTS=()
RUN_LIST_INPUT=""
LIB_INPUT=""
FASTA_NAME=""
FASTA_ABS=""
OUT_SUB_INPUT=""
OUT_DIR_INPUT=""
TEMP_DIR_INPUT=""
LOG_DIR_INPUT=""
VERBOSE="${VERBOSE:-0}"
TEST_MODE=0
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --f)
      ((i+=1))
      raw_input="${ARGS[$i]:-}"
      [[ -n "$raw_input" ]] || die "--f needs a value"
      RAW_INPUTS+=("$raw_input")
      ;;
    --run-list)
      ((i+=1))
      RUN_LIST_INPUT="${ARGS[$i]:-}"
      [[ -n "$RUN_LIST_INPUT" ]] || die "--run-list needs a value"
      ;;
    --lib)
      ((i+=1))
      LIB_INPUT="${ARGS[$i]:-}"
      [[ -n "$LIB_INPUT" ]] || die "--lib needs a value"
      ;;
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
    --out-sub)
      ((i+=1))
      OUT_SUB_INPUT="${ARGS[$i]:-}"
      [[ -n "$OUT_SUB_INPUT" ]] || die "--out-sub needs a value"
      ;;
    --out-dir)
      ((i+=1))
      OUT_DIR_INPUT="${ARGS[$i]:-}"
      [[ -n "$OUT_DIR_INPUT" ]] || die "--out-dir needs a value"
      ;;
    --temp-dir)
      ((i+=1))
      TEMP_DIR_INPUT="${ARGS[$i]:-}"
      [[ -n "$TEMP_DIR_INPUT" ]] || die "--temp-dir needs a value"
      ;;
    --log-dir)
      ((i+=1))
      LOG_DIR_INPUT="${ARGS[$i]:-}"
      [[ -n "$LOG_DIR_INPUT" ]] || die "--log-dir needs a value"
      ;;
    --set)
      ((i+=1))
      override="${ARGS[$i]:-}"
      [[ -n "$override" ]] || die "--set needs KEY=VALUE"
      [[ "$override" == *=* ]] || die "--set expects KEY=VALUE"
      key="${override%%=*}"
      value="${override#*=}"
      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid --set key '$key' (use A-Z, a-z, 0-9, _)"
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
        VERBOSE=1
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
    *)
      die "Unknown argument: ${ARGS[$i]}"
      ;;
  esac
  ((i+=1))
done

DIANN_VERBOSE="${DIANN_VERBOSE:-${VERBOSE:-0}}"
[[ "$DIANN_VERBOSE" =~ ^[0-4]$ ]] || die "Invalid verbose level '$DIANN_VERBOSE' (allowed: 0..4)"
STEP5_REPORT_QVALUE="${STEP5_REPORT_QVALUE:-1}"
[[ "$STEP5_REPORT_QVALUE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "Invalid STEP5_REPORT_QVALUE '$STEP5_REPORT_QVALUE' (expected numeric 0..1)"

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
if [[ -n "$OUT_DIR_INPUT" ]]; then
  [[ "$OUT_DIR_INPUT" == /* ]] || die "--out-dir expects an absolute path"
  [[ -z "${OUT_SUB_INPUT}" || "$OUT_SUB_INPUT" != /* ]] || die "--out-sub must be relative when used with --out-dir"
  EXPDIR="$(dirname "$OUT_DIR_INPUT")"
  S1DIR="${EXPDIR}/01_predlib"
  S2DIR="${EXPDIR}/02_tunelib"
  S5DIR="${OUT_DIR_INPUT%/}"
  [[ -n "$S5DIR" ]] || S5DIR="/"
  export EXPDIR S1DIR S2DIR S5DIR
elif [[ "$TEST_MODE" == "1" ]]; then
  : "${OUTBASE:?OUTBASE not set}"
  outbase="${OUTBASE%/}"
  [[ -n "$outbase" ]] || outbase="/"
  EXPDIR="${outbase}/${EXPNAME}"
  S1DIR="${EXPDIR}/01_predlib"
  S2DIR="${EXPDIR}/02_tunelib"
  S5DIR="${EXPDIR}/05_quant"
  export EXPDIR S1DIR S2DIR S5DIR
else
  set_paths_for_exp
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
  die "Step 5 requires FASTA via --fasta <filename> or --fastaabs <absolute path>."
fi
require_file "$FASTA_FILE"

if [[ -n "$LIB_INPUT" ]]; then
  LIB="$(resolve_lib_input "$LIB_INPUT")"
elif [[ -n "${LIB:-}" ]]; then
  LIB="$(resolve_lib_input "$LIB")"
else
  die "Step 5 requires --lib <path_or_name> (or --set LIB=...)."
fi
require_file "$LIB"

if [[ ${#RAW_INPUTS[@]} -eq 0 ]]; then
  if [[ -n "$RUN_LIST_INPUT" ]]; then
    RUN_LIST_FILE="$RUN_LIST_INPUT"
  else
    die "No raw files provided. Use --f <rawfile> (repeatable) or --run-list <file>."
  fi

  [[ -f "$RUN_LIST_FILE" ]] || die "Run-list file not found: ${RUN_LIST_FILE}"
  while IFS= read -r line || [[ -n "$line" ]]; do
    trimmed="${line#${line%%[![:space:]]*}}"
    [[ -z "$trimmed" ]] && continue
    [[ "$trimmed" == \#* ]] && continue
    RAW_INPUTS+=("$trimmed")
  done < "$RUN_LIST_FILE"
fi

[[ ${#RAW_INPUTS[@]} -gt 0 ]] || die "No raw files collected for Step 5"

RAW_FILES=()
for raw_input in "${RAW_INPUTS[@]}"; do
  raw_resolved="$(resolve_raw_input "$raw_input")"
  require_file "$raw_resolved"
  RAW_FILES+=("$raw_resolved")
done

OUT_SUB="${OUT_SUB_INPUT:-${OUT_SUB:-}}"
if [[ -n "$OUT_SUB" ]]; then
  [[ "$OUT_SUB" != /* ]] || die "--out-sub must be a relative path under ${S5DIR}"
  [[ "$OUT_SUB" != *".."* ]] || die "--out-sub must not contain '..'"
  OUT_SUB="${OUT_SUB#./}"
  OUT_SUB="${OUT_SUB%/}"
  [[ -n "$OUT_SUB" ]] || die "--out-sub resolves to empty path"
  S5OUTDIR="${S5DIR}/${OUT_SUB}"
else
  S5OUTDIR="$S5DIR"
fi

if [[ -n "$TEMP_DIR_INPUT" ]]; then
  [[ "$TEMP_DIR_INPUT" == /* ]] || die "--temp-dir expects an absolute path"
  STEP5_TEMP_DIR="${TEMP_DIR_INPUT%/}"
else
  STEP5_TEMP_DIR="${TMPBASE%/}/${EXPNAME}/step5_quant/${OUT_SUB:-default}"
fi
[[ -n "$STEP5_TEMP_DIR" ]] || die "Step5 temp directory resolves to an empty path"

if [[ -n "$LOG_DIR_INPUT" ]]; then
  [[ "$LOG_DIR_INPUT" == /* ]] || die "--log-dir expects an absolute path"
  STEP5_LOG_DIR="${LOG_DIR_INPUT%/}"
else
  STEP5_LOG_DIR="$S5OUTDIR"
fi
[[ -n "$STEP5_LOG_DIR" ]] || die "Step5 log directory resolves to an empty path"

if [[ "$TEST_MODE" != "1" ]]; then
  mkdir -p "$S5OUTDIR"
  mkdir -p "$STEP5_TEMP_DIR" "$STEP5_LOG_DIR"
else
  outbase_resolved="${OUTBASE%/}"
  [[ -n "$outbase_resolved" ]] || outbase_resolved="/"
  [[ -d "$outbase_resolved" ]] || die "OUTBASE does not exist (test mode): $outbase_resolved"
  [[ -w "$outbase_resolved" ]] || die "OUTBASE is not writable (test mode): $outbase_resolved"
  if [[ -e "$S5OUTDIR" && ! -d "$S5OUTDIR" ]]; then
    die "Step5 output path exists but is not a directory (test mode): $S5OUTDIR"
  fi
fi

REPORT="${REPORT:-${S5OUTDIR}/final_report.parquet}"

ENABLE_PRE_SEARCH="${ENABLE_PRE_SEARCH:-${enable_PRE_SEARCH:-${enable_pre_search:-0}}}"
[[ "$ENABLE_PRE_SEARCH" =~ ^[01]$ ]] || die "Invalid ENABLE_PRE_SEARCH '$ENABLE_PRE_SEARCH' (allowed 0 or 1)"
ENABLE_GEN_SPEC_LIB="${ENABLE_GEN_SPEC_LIB:-${enable_GEN_SPEC_LIB:-0}}"
[[ "$ENABLE_GEN_SPEC_LIB" =~ ^[01]$ ]] || die "Invalid ENABLE_GEN_SPEC_LIB '$ENABLE_GEN_SPEC_LIB' (allowed 0 or 1)"

ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE="${ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE:-${enable_eval_run_arabidopsis_reference:-0}}"
[[ "$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE" =~ ^[01]$ ]] || die "Invalid ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE '$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE' (allowed 0 or 1)"
ENABLE_EVAL_RUN_ECOLI_CENTERED="${ENABLE_EVAL_RUN_ECOLI_CENTERED:-${enable_eval_run_ecoli_centered:-0}}"
[[ "$ENABLE_EVAL_RUN_ECOLI_CENTERED" =~ ^[01]$ ]] || die "Invalid ENABLE_EVAL_RUN_ECOLI_CENTERED '$ENABLE_EVAL_RUN_ECOLI_CENTERED' (allowed 0 or 1)"

EVAL_ARABIDOPSIS_REFERENCE_PYTHON_FILE="${EVAL_ARABIDOPSIS_REFERENCE_PYTHON_FILE:-${eval_arabidopsis_reference_python_file:-}}"
EVAL_ECOLI_CENTERED_PYTHON_FILE="${EVAL_ECOLI_CENTERED_PYTHON_FILE:-${eval_ecoli_centered_python_file:-}}"
EVAL_PARQUET_FILE="${EVAL_PARQUET_FILE:-${eval_parquet_file:-final_report.parquet}}"
EVAL_TSV_FILE="${EVAL_TSV_FILE:-${eval_tsv_file:-final_report.stats.tsv}}"
EVAL_PYTHON_PATH="${EVAL_PYTHON_PATH:-${eval_python_path:-python3}}"

EVAL_PARQUET_PATH="${S5OUTDIR}/${EVAL_PARQUET_FILE}"
EVAL_TSV_PATH="${S5OUTDIR}/${EVAL_TSV_FILE}"

if [[ "$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE" == "1" || "$ENABLE_EVAL_RUN_ECOLI_CENTERED" == "1" ]]; then
  require_exec "$EVAL_PYTHON_PATH"
fi
if [[ "$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE" == "1" ]]; then
  [[ -n "$EVAL_ARABIDOPSIS_REFERENCE_PYTHON_FILE" ]] || die "ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE=1 but eval_arabidopsis_reference_python_file is empty"
  require_file "$EVAL_ARABIDOPSIS_REFERENCE_PYTHON_FILE"
fi
if [[ "$ENABLE_EVAL_RUN_ECOLI_CENTERED" == "1" ]]; then
  [[ -n "$EVAL_ECOLI_CENTERED_PYTHON_FILE" ]] || die "ENABLE_EVAL_RUN_ECOLI_CENTERED=1 but eval_ecoli_centered_python_file is empty"
  require_file "$EVAL_ECOLI_CENTERED_PYTHON_FILE"
fi

ARABIDOPSIS_REFERENCE_CMD=()
if [[ "$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE" == "1" ]]; then
  ARABIDOPSIS_REFERENCE_CMD=("$EVAL_PYTHON_PATH" "$EVAL_ARABIDOPSIS_REFERENCE_PYTHON_FILE" "$EVAL_PARQUET_PATH")
fi
ECOLI_CENTERED_CMD=()
if [[ "$ENABLE_EVAL_RUN_ECOLI_CENTERED" == "1" ]]; then
  ECOLI_CENTERED_CMD=("$EVAL_PYTHON_PATH" "$EVAL_ECOLI_CENTERED_PYTHON_FILE" "$EVAL_TSV_PATH")
fi

DIANN_CMD=(
  "$DIANN_BIN"
  --threads "$THREADS"
  --verbose "$DIANN_VERBOSE"
  --fasta "$FASTA_FILE"
  --lib "$LIB"
  --out "$REPORT"
  --temp "$STEP5_TEMP_DIR"
  --qvalue "$STEP5_REPORT_QVALUE"
  --cut "$CUT"
  --missed-cleavages "$MISSED_CLEAVAGES"
  --mass-acc-ms1 "$MASS_ACC_MS1" --mass-acc "${MASS_ACC_MS2:-${MASS_ACC}}"
)

for raw_file in "${RAW_FILES[@]}"; do
  DIANN_CMD+=(--f "$raw_file")
done

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
if [[ "${enable_REPORT_DECOYS:-0}" == "1" ]]; then
  DIANN_CMD+=(--report-decoys)
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

STARTCMD_FILE="${STEP5_LOG_DIR}/startcmd.txt"
STEP_LOG="${STEP5_LOG_DIR}/step5_quant.log"

if [[ "$TEST_MODE" == "1" ]]; then
  emit_test_output() {
    echo "TEST MODE: no directories/files will be created; DIA-NN will not run."
    echo "DIANN_BIN   : $DIANN_BIN"
    echo "EXPNAME     : ${EXPNAME:-}"
    echo "FASTA       : $FASTA_FILE"
    echo "LIB         : $LIB"
    echo "Runs        : ${#RAW_FILES[@]}"
    if [[ -n "$OUT_SUB" ]]; then
      echo "OUT_SUB     : $OUT_SUB"
    else
      echo "OUT_SUB     : (none)"
    fi
    echo "OUTDIR      : $S5OUTDIR"
    echo "TEMPDIR     : $STEP5_TEMP_DIR"
    echo "REPORT      : $REPORT"
    echo "LOGDIR      : $STEP5_LOG_DIR"
    echo "LOG         : $STEP_LOG"
    echo "STARTCMD    : $STARTCMD_FILE"
    echo "REPORT_QVAL : $STEP5_REPORT_QVALUE"
    echo "ANALYSIS_QV : $QVALUE"
    echo "REPORT_DECOY: ${enable_REPORT_DECOYS:-0}"
    echo "PRE_SEARCH  : $ENABLE_PRE_SEARCH"
    echo "GEN_SPEC_LIB: $ENABLE_GEN_SPEC_LIB"
    echo "EVAL_ARABIDOPSIS_REFERENCE: $ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE"
    echo "EVAL_ECOLI_CENTERED : $ENABLE_EVAL_RUN_ECOLI_CENTERED"
    echo "DIA-NN command (without --test):"
    printf '%q ' "${DIANN_CMD[@]}"
    echo
    if [[ "$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE" == "1" ]]; then
      echo "Arabidopsis reference eval command (result file existence not checked in test mode):"
      printf '%q ' "${ARABIDOPSIS_REFERENCE_CMD[@]}"
      echo
    fi
    if [[ "$ENABLE_EVAL_RUN_ECOLI_CENTERED" == "1" ]]; then
      echo "E. coli centered eval command (result file existence not checked in test mode):"
      printf '%q ' "${ECOLI_CENTERED_CMD[@]}"
      echo
    fi
    echo "OK: validation successful (test mode)."
  }

  if [[ -t 1 ]] && command -v less >/dev/null 2>&1; then
    emit_test_output | less
  else
    emit_test_output
  fi
  exit 0
fi

exec > >(tee -a "$STEP_LOG") 2>&1
info "Step5 log file: ${STEP_LOG}"

{
  echo "# script invocation command"
  printf '%q ' "$0"
  printf '%q ' "$@"
  echo
  echo "# step5_quant start command"
  printf '%q ' "${DIANN_CMD[@]}"
  echo
  if [[ "$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE" == "1" ]]; then
    echo "# arabidopsis_reference eval command"
    printf '%q ' "${ARABIDOPSIS_REFERENCE_CMD[@]}"
    echo
  fi
  if [[ "$ENABLE_EVAL_RUN_ECOLI_CENTERED" == "1" ]]; then
    echo "# ecoli_centered eval command"
    printf '%q ' "${ECOLI_CENTERED_CMD[@]}"
    echo
  fi
} > "$STARTCMD_FILE"
info "Saved start command: ${STARTCMD_FILE}"

info "Step5 (quant) started"
debug "LIB=${LIB}"
debug "REPORT=${REPORT}"
debug "S5OUTDIR=${S5OUTDIR}"
debug "RAW files count=${#RAW_FILES[@]}"
debug "DIA-NN command: ${DIANN_CMD[*]}"

"${DIANN_CMD[@]}"

if [[ "$ENABLE_EVAL_RUN_ARABIDOPSIS_REFERENCE" == "1" ]]; then
  [[ -f "$EVAL_PARQUET_PATH" ]] || die "Arabidopsis reference eval input file not found: $EVAL_PARQUET_PATH"
  info "Running Arabidopsis reference evaluation"
  debug "Arabidopsis reference command: ${ARABIDOPSIS_REFERENCE_CMD[*]}"
  "${ARABIDOPSIS_REFERENCE_CMD[@]}"
fi

if [[ "$ENABLE_EVAL_RUN_ECOLI_CENTERED" == "1" ]]; then
  [[ -f "$EVAL_TSV_PATH" ]] || die "E. coli centered eval input file not found: $EVAL_TSV_PATH"
  info "Running E. coli centered evaluation"
  debug "E. coli centered command: ${ECOLI_CENTERED_CMD[*]}"
  "${ECOLI_CENTERED_CMD[@]}"
fi

info "Step5 (quant) finished"
