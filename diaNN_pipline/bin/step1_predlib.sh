#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
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

load_cfg_relaxed "${PIPELINE_ROOT}/config/step1.cfg"

warn_unmapped_enable_flags() {
  local key value
  for key in $(compgen -A variable | grep '^enable_'); do
    value="${!key:-0}"
    [[ "$value" == "1" ]] || continue

    case "$key" in
      enable_UNIMOD4|enable_REANNOTATE|enable_SEMI|enable_PEPTIDOFORMS|enable_FASTA_SEARCH|enable_MATRICES|enable_GEN_SPEC_LIB|enable_CONT_QUANT_EXQCLUDE_CRAP|enable_REANALYSE|enable_RT_PROFILING)
        ;;
      *)
        warn "enable-flag '$key=1' ist gesetzt, aber in step1_predlib.sh nicht auf ein DIA-NN Argument gemappt."
        ;;
    esac
  done
}

warn_unmapped_enable_flags

# required CLI input: either filename in FASTABASE or absolute path
FASTA_NAME=""
FASTA_ABS=""
CLI_OVERRIDES=()
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
    --set)
      ((i+=1))
      override="${ARGS[$i]:-}"
      [[ -n "$override" ]] || die "--set needs KEY=VALUE"
      [[ "$override" == *=* ]] || die "--set expects KEY=VALUE"
      key="${override%%=*}"
      value="${override#*=}"
      [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "Invalid --set key '$key' (use A-Z, 0-9, _)"
      printf -v "$key" '%s' "$value"
      CLI_OVERRIDES+=("$key")
      ;;
  esac
  ((i+=1))
done

DIANN_VERBOSE="${DIANN_VERBOSE:-${VERBOSE:-0}}"
[[ "$DIANN_VERBOSE" =~ ^[0-4]$ ]] || die "Invalid verbose level '$DIANN_VERBOSE' (allowed: 0..4)"

# Internal pipeline debug is on when DIA-NN verbose > 0
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
  S1DIR="${EXPDIR}/01_predlib"
  export EXPDIR S1DIR
else
  set_paths_for_exp
fi

STEP_LOG="${S1DIR}/step1_predlib.log"
if [[ "$TEST_MODE" != "1" ]]; then
  exec > >(tee -a "$STEP_LOG") 2>&1
  info "Step1 log file: ${STEP_LOG}"
fi

require_exec "$DIANN_BIN"

if [[ -n "$FASTA_NAME" && -n "$FASTA_ABS" ]]; then
  die "Use either --fasta <filename> or --fastaabs <absolute path>, not both"
fi

if [[ -n "$FASTA_NAME" ]]; then
  [[ "$FASTA_NAME" != */* ]] || die "--fasta expects only a filename, no path separators"
  fasta_base="${FASTABASE%/}"
  [[ -n "$fasta_base" ]] || fasta_base="/"
  FASTA="${fasta_base}/${FASTA_NAME}"
elif [[ -n "$FASTA_ABS" ]]; then
  [[ "$FASTA_ABS" == /* ]] || die "--fastaabs expects an absolute path"
  FASTA="$FASTA_ABS"
else
  die "One FASTA input is required: --fasta <filename> or --fastaabs <absolute path>"
fi
require_file "$FASTA"

# OUTLIB="${S1DIR}/step1_${EXPNAME}_predicted.speclib"
OUTLIB="${S1DIR}/step1_${EXPNAME}_predicted.parquet"
REPORT="${S1DIR}/predlib_report.parquet"
STEP1_TEMP_DIR="${S1DIR}/Temp"
if [[ "$TEST_MODE" != "1" ]]; then
  mkdir -p "$STEP1_TEMP_DIR"
else
  outbase_resolved="${OUTBASE%/}"
  [[ -n "$outbase_resolved" ]] || outbase_resolved="/"
  [[ -d "$outbase_resolved" ]] || die "OUTBASE does not exist (test mode): $outbase_resolved"
  [[ -w "$outbase_resolved" ]] || die "OUTBASE is not writable (test mode): $outbase_resolved"
  if [[ -e "$S1DIR" && ! -d "$S1DIR" ]]; then
    die "Step1 output path exists but is not a directory (test mode): $S1DIR"
  fi
fi

info "Step1 (predlib) started"
debug "SCRIPT_DIR=${SCRIPT_DIR}"
debug "PIPELINE_ROOT=${PIPELINE_ROOT}"
debug "DIANN_BIN=${DIANN_BIN}"
debug "FASTA=${FASTA}"
debug "OUTLIB=${OUTLIB}"
debug "REPORT=${REPORT}"
debug "STEP1_TEMP_DIR=${STEP1_TEMP_DIR}"

DIANN_CMD=(
  "$DIANN_BIN"
  --lib "${LIB:-}"
  --threads "$THREADS"
  --verbose "$DIANN_VERBOSE"
  --fasta "$FASTA"
  --predictor
  --out-lib "$OUTLIB"
  --out "$REPORT"
  --temp "$STEP1_TEMP_DIR"
  --qvalue "$QVALUE"
  --cut "$CUT"
  --missed-cleavages "$MISSED_CLEAVAGES"
  --min-pep-len "$MIN_PEP_LEN" --max-pep-len "$MAX_PEP_LEN"
  --min-pr-mz "$MIN_PR_MZ" --max-pr-mz "$MAX_PR_MZ"
  --min-pr-charge "$MIN_PR_CHARGE" --max-pr-charge "$MAX_PR_CHARGE"
  --min-fr-mz "$MIN_FR_MZ" --max-fr-mz "$MAX_FR_MZ"
)

if [[ "${enable_MATRICES:-0}" == "1" ]]; then
  DIANN_CMD+=(--matrices)
fi

if [[ "${enable_GEN_SPEC_LIB:-0}" == "1" ]]; then
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

STARTCMD_FILE="${S1DIR}/startcmd.txt"
if [[ "$TEST_MODE" == "1" ]]; then
  emit_test_output() {
    echo "TEST MODE: no directories/files will be created; DIA-NN will not run."
    echo "DIANN_BIN  : $DIANN_BIN"
    echo "EXPNAME    : ${EXPNAME:-}"
    echo "FASTA      : $FASTA"
    echo "OUTDIR     : $S1DIR"
    echo "REPORT     : $REPORT"
    echo "OUTLIB     : $OUTLIB"
    echo "TMPDIR     : $STEP1_TEMP_DIR"
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

{
  echo "# script invocation command"
  printf '%q ' "$0"
  printf '%q ' "$@"
  echo
  echo "# step1_predlib start command"
  printf '%q ' "${DIANN_CMD[@]}"
  echo
} > "$STARTCMD_FILE"
info "Saved start command: ${STARTCMD_FILE}"

debug "DIA-NN command: ${DIANN_CMD[*]}"

run_cmd_with_diagnostics "Step1 DIA-NN run" DIANN_CMD "$STEP_LOG"

info "Step1 (predlib) finished"
