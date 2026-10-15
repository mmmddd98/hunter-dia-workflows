#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# make_calibration_lib.sh
# Build a small calibration library for DIA-NN --ref (Calibration lib)
#
# This generates a SMALL empirical library from 1 (or a few) good runs,
# using your Step-1 predicted library as the search library.
#
# Usage:
#   bash bin/make_calibration_lib.sh \
#     [--diann /path/to/diann-linux] \
#     [--set KEY=VALUE] \
#     [--verbose 0..4 | --v] \
#     [--profile balanced|fast|slow|all] \
#     [--test] \
#     --fasta  db.fasta \
#     [--fastaabs /abs/path/to/db.fasta] \
#     --lib    /path/to/step1_predicted.predicted.speclib \
#     --expname my_exp \
#     [--outdir /path/to/out/calib_ref] \
#     --f /path/to/good_run_01.mzML \
#     [--f /path/to/good_run_02.mzML ...]
#
# Output:
#   <outdir>/calib_ref.parquet     (use this with: --ref <...>/calib_ref.parquet)
#   <outdir>/calib_report.parquet  (QC/report)
#   <outdir>/startcmd.txt          (exact command)
#   <outdir>/make_calibration_lib.log
#
# Test mode (--test):
#   validates inputs and prints the exact DIA-NN command that would run,
#   but does not create directories/files and does not execute DIA-NN.
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
source "${PIPELINE_ROOT}/config/cal.cfg"
source "${PIPELINE_ROOT}/config/profile.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

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

MAKE_CAL_CFG="${PIPELINE_ROOT}/config/make_cal.cfg"
if [[ -f "$MAKE_CAL_CFG" ]]; then
  load_cfg_relaxed "$MAKE_CAL_CFG"
fi

warn(){ _log "WARN" "$*"; }

warn_unmapped_enable_flags() {
  local key value
  for key in $(compgen -A variable | grep '^enable_'); do
    value="${!key:-0}"
    [[ "$value" == "1" ]] || continue

    case "$key" in
      enable_UNIMOD4|enable_REANNOTATE|enable_SEMI|enable_PEPTIDOFORMS|enable_FASTA_SEARCH|enable_MATRICES|enable_GEN_SPEC_LIB|enable_CONT_QUANT_EXQCLUDE_CRAP|enable_REANALYSE|enable_RT_PROFILING|enable_PRE_SEARCH|enable_pre_search)
        ;;
      *)
        warn "enable-flag '$key=1' ist gesetzt, aber in make_cal_lib.sh nicht auf ein DIA-NN Argument gemappt."
        ;;
    esac
  done
}

warn_unmapped_enable_flags

# ensure sensible fallbacks if values are not set in cfg
DIANN_BIN="${DIANN_BIN:-diann-2.3.0}"
VAR_MODS="${VAR_MODS:-0}"
DIANN_VERBOSE="${DIANN_VERBOSE:-2}"
enable_GEN_SPEC_LIB="${enable_GEN_SPEC_LIB:-1}"
TEST_MODE=0
PROFILE="balanced"

# ---- Parse args ----
FASTA_NAME=""
FASTA_ABS=""
FASTA=""
PREDLIB=""
OUTDIR=""
RAW_FILES=()
CLI_OVERRIDES=()

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --diann)
      ((i+=1)); DIANN_BIN="${ARGS[$i]:-}";;
    --set)
      ((i+=1))
      override="${ARGS[$i]:-}"
      [[ -n "$override" ]] || die "--set needs KEY=VALUE"
      [[ "$override" == *=* ]] || die "--set expects KEY=VALUE"
      key_raw="${override%%=*}"
      key="${key_raw^^}"
      value="${override#*=}"
      [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "Invalid --set key '$key' (use A-Z, 0-9, _)"
      printf -v "$key" '%s' "$value"
      CLI_OVERRIDES+=("$key")
      ;;
    --fasta)
      ((i+=1)); FASTA_NAME="${ARGS[$i]:-}"
      [[ -n "$FASTA_NAME" ]] || die "--fasta needs a value";;
    --fastaabs)
      ((i+=1)); FASTA_ABS="${ARGS[$i]:-}"
      [[ -n "$FASTA_ABS" ]] || die "--fastaabs needs a value";;
    --lib|--predlib)
      ((i+=1)); PREDLIB="${ARGS[$i]:-}";;
    --outdir)
      ((i+=1)); OUTDIR="${ARGS[$i]:-}";;
    --expname)
      ((i+=1)); expname_arg="${ARGS[$i]:-}"
      [[ -n "$expname_arg" ]] || die "--expname needs a value";;
    --f)
      ((i+=1)); f="${ARGS[$i]:-}"
      [[ -n "$f" ]] || die "--f needs a value"
      RAW_FILES+=("$f");;
    --profile)
      ((i+=1)); PROFILE="${ARGS[$i]:-}"
      [[ -n "$PROFILE" ]] || die "--profile needs a value (balanced|fast|slow|all)"
      ;;
    --v|-v)
      DIANN_VERBOSE=1;;
    --verbose)
      next="${ARGS[$((i+1))]:-}"
      if [[ "$next" =~ ^[0-4]$ ]]; then
        DIANN_VERBOSE="$next"
        ((i+=1))
      else
        DIANN_VERBOSE=2
      fi
      ;;
    --verbose=*)
      DIANN_VERBOSE="${ARGS[$i]#*=}";;
    --test)
      TEST_MODE=1;;
    -h|--help)
      sed -n '1,60p' "$0"
      exit 0
      ;;
    *)
      die "Unknown argument: ${ARGS[$i]}"
      ;;
  esac
  ((i+=1))
done

PROFILE="${PROFILE,,}"
case "$PROFILE" in
  balanced|fast|slow|all)
    ;;
  *)
    die "Invalid --profile '$PROFILE' (allowed: balanced|fast|slow|all)"
    ;;
esac

for profile_key in balanced fast slow; do
  if [[ ! -v PROFILE_FLAGS[$profile_key] ]]; then
    die "Missing PROFILE_FLAGS[$profile_key] in config/profile.cfg"
  fi
done

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

[[ -n "$DIANN_BIN" ]] || die "Missing --diann /path/to/diann-linux"
require_exec "$DIANN_BIN"

if [[ -n "$FASTA_NAME" && -n "$FASTA_ABS" ]]; then
  die "Use either --fasta <filename in FASTABASE> or --fastaabs <absolute path>, not both"
fi

if [[ -n "$FASTA_NAME" ]]; then
  [[ "$FASTA_NAME" != */* ]] || die "--fasta expects only a filename (from FASTABASE), no path separators"
  fasta_base="${FASTABASE%/}"
  [[ -n "$fasta_base" ]] || fasta_base="/"
  FASTA="${fasta_base}/${FASTA_NAME}"
elif [[ -n "$FASTA_ABS" ]]; then
  [[ "$FASTA_ABS" == /* ]] || die "--fastaabs expects an absolute path"
  FASTA="$FASTA_ABS"
else
  die "Missing FASTA: use --fasta <filename in FASTABASE> or --fastaabs <absolute path>"
fi
require_file "$FASTA"

[[ -n "$PREDLIB" ]] || die "Missing --lib /path/to/step1_predicted*.speclib"
require_file "$PREDLIB"

parse_expname "$@"

if [[ "$TEST_MODE" == "1" ]]; then
  : "${OUTBASE:?OUTBASE not set}"
  outbase="${OUTBASE%/}"
  [[ -n "$outbase" ]] || outbase="/"
  EXPDIR="${outbase}/${EXPNAME}"
  export EXPDIR
else
  set_paths_for_exp
fi

if [[ -z "$OUTDIR" ]]; then
  OUTDIR="${EXPDIR}/cal_lib"
fi

[[ ${#RAW_FILES[@]} -ge 1 ]] || die "Provide at least one run via --f <file>"

for rf in "${RAW_FILES[@]}"; do
  [[ -f "$rf" ]] || die "Run file not found: $rf"
done

PROFILE_LIST=()
if [[ "$PROFILE" == "all" ]]; then
  PROFILE_LIST=(balanced fast slow)
else
  PROFILE_LIST=("$PROFILE")
fi

if [[ "$TEST_MODE" == "1" ]]; then
  echo "TEST MODE: no directories/files will be created; DIA-NN will not run."
  echo "DIANN_BIN  : $DIANN_BIN"
  echo "EXPNAME    : ${EXPNAME:-}"
  echo "FASTA      : $FASTA"
  echo "PREDLIB    : $PREDLIB"
  echo "PROFILE    : $PROFILE"
  echo "Runs       : ${#RAW_FILES[@]}"
fi

for profile_name in "${PROFILE_LIST[@]}"; do
  profile_outdir="${OUTDIR%/}/${profile_name}"

  REPORT="${profile_outdir}/calib_report_${profile_name}.parquet"
  OUTLIB="${profile_outdir}/calib_ref_${profile_name}.parquet"
  TMPDIR="${profile_outdir}/Temp"
  LOG="${profile_outdir}/make_calibration_lib_${profile_name}.log"
  STARTCMD="${profile_outdir}/startcmd_${profile_name}.txt"

  if [[ "$TEST_MODE" != "1" ]]; then
    mkdir -p "$profile_outdir"
    mkdir -p "$TMPDIR"
  else
    if [[ -e "$profile_outdir" && ! -d "$profile_outdir" ]]; then
      die "OUTDIR exists but is not a directory (test mode): $profile_outdir"
    fi

    check_dir="$profile_outdir"
    while [[ ! -e "$check_dir" && "$check_dir" != "/" ]]; do
      check_dir="$(dirname "$check_dir")"
    done
    [[ -d "$check_dir" ]] || die "No existing parent directory for OUTDIR (test mode): $profile_outdir"
    [[ -w "$check_dir" ]] || die "Nearest existing parent is not writable (test mode): $check_dir"
  fi

  DIANN_CMD=(
    "$DIANN_BIN"
    --threads "$THREADS"
    --verbose "$DIANN_VERBOSE"
    --fasta "$FASTA"
    --lib "$PREDLIB"
    --out "$REPORT"
    --out-lib "$OUTLIB"
    --temp "$TMPDIR"
    --qvalue "$QVALUE"
    --cut "$CUT"
    --missed-cleavages "$MISSED_CLEAVAGES"
    --min-pep-len "$MIN_PEP_LEN" --max-pep-len "$MAX_PEP_LEN"
    --min-pr-charge "$MIN_PR_CHARGE" --max-pr-charge "$MAX_PR_CHARGE"
    --min-pr-mz "$MIN_PR_MZ" --max-pr-mz "$MAX_PR_MZ"
    --min-fr-mz "$MIN_FR_MZ" --max-fr-mz "$MAX_FR_MZ"
    --mass-acc-ms1 "$MASS_ACC_MS1"
    --mass-acc "$MASS_ACC"
    --id-profiling
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

  ENABLE_PRE_SEARCH="${enable_PRE_SEARCH:-${enable_pre_search:-0}}"
  if [[ "$ENABLE_PRE_SEARCH" == "1" ]]; then
    DIANN_CMD+=(--pre-search)
  fi

  for rf in "${RAW_FILES[@]}"; do
    DIANN_CMD+=(--f "$rf")
  done

  if [[ "$enable_UNIMOD4" == "1" ]]; then
    DIANN_CMD+=(--unimod4)
  fi

  append_mod_args DIANN_CMD

  if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    read -r -a EXTRA_FLAGS_ARR <<< "$EXTRA_FLAGS"
    DIANN_CMD+=("${EXTRA_FLAGS_ARR[@]}")
  fi

  profile_flags="${PROFILE_FLAGS[$profile_name]}"
  if [[ -n "$profile_flags" ]]; then
    read -r -a PROFILE_FLAGS_ARR <<< "$profile_flags"
    DIANN_CMD+=("${PROFILE_FLAGS_ARR[@]}")
  fi

  if [[ "$TEST_MODE" == "1" ]]; then
    echo ""
    echo "Profile    : ${profile_name}"
    echo "OUTDIR     : ${profile_outdir}"
    echo "REPORT     : ${REPORT}"
    echo "OUTLIB     : ${OUTLIB}"
    echo "TMPDIR     : ${TMPDIR}"
    echo "DIA-NN command (without --test):"
    printf '%q ' "${DIANN_CMD[@]}"
    echo
    continue
  fi

  {
    echo "# profile: ${profile_name}"
    echo "# script invocation command"
    printf '%q ' "$0"
    printf '%q ' "$@"
    echo
    echo "# make_calibration_lib start command"
    printf '%q ' "${DIANN_CMD[@]}"
    echo
  } > "$STARTCMD"

  {
    echo "DIANN_BIN  : $DIANN_BIN"
    echo "EXPNAME    : ${EXPNAME:-}"
    echo "PROFILE    : ${profile_name}"
    echo "FASTA      : $FASTA"
    echo "PREDLIB    : $PREDLIB"
    echo "OUTDIR     : ${profile_outdir}"
    echo "REPORT     : $REPORT"
    echo "OUTLIB     : $OUTLIB"
    echo "TMPDIR     : $TMPDIR"
    echo "STARTCMD   : $STARTCMD"
    echo "Runs       : ${#RAW_FILES[@]}"
    echo "Running DIA-NN..."
    run_cmd_with_diagnostics "make_cal_lib DIA-NN run (profile=${profile_name})" DIANN_CMD "$LOG"
  } 2>&1 | tee -a "$LOG"

  [[ -s "$OUTLIB" ]] || die "Calibration library not created (empty/missing): $OUTLIB"
  echo "OK: created calibration library (${profile_name}): $OUTLIB"
done

if [[ "$TEST_MODE" == "1" ]]; then
  echo ""
  echo "OK: validation successful (test mode)."
fi
