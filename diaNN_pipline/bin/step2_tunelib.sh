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

load_cfg_relaxed "${PIPELINE_ROOT}/config/step2.cfg"

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

auto_detect_predlib() {
  local search_dir="$1"
  local candidates=()
  shopt -s nullglob
  candidates=("${search_dir}"/*.predicted.speclib)
  shopt -u nullglob

  if [[ ${#candidates[@]} -eq 1 ]]; then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  if [[ ${#candidates[@]} -eq 0 ]]; then
    die "No --lib provided and no '*.predicted.speclib' found in ${search_dir}."
  fi

  die "No --lib provided and multiple '*.predicted.speclib' files found in ${search_dir}; specify --lib explicitly."
}

resolve_ref_input() {
  local input="$1"
  if [[ "$input" == /* ]]; then
    echo "$input"
  elif [[ "$input" == */* ]]; then
    echo "$input"
  else
    echo "${EXPDIR}/cal_lib/${input}"
  fi
}

resolve_default_ref_for_profile() {
  local profile_name="$1"
  local profile_ref="${EXPDIR}/cal_lib/${profile_name}/calib_ref_${profile_name}.parquet"
  local profile_ref_legacy_in_profile="${EXPDIR}/cal_lib/${profile_name}/calib_ref.parquet"
  local legacy_ref="${EXPDIR}/cal_lib/calib_ref.parquet"

  if [[ -f "$profile_ref" ]]; then
    echo "$profile_ref"
    return 0
  fi

  if [[ -f "$profile_ref_legacy_in_profile" ]]; then
    warn "Profile ref with suffix not found for '${profile_name}', falling back to legacy-in-profile ref: ${profile_ref_legacy_in_profile}"
    echo "$profile_ref_legacy_in_profile"
    return 0
  fi

  if [[ "$profile_name" != "all" && -f "$legacy_ref" ]]; then
    warn "Profile ref not found for '${profile_name}', falling back to legacy ref: ${legacy_ref}"
    echo "$legacy_ref"
    return 0
  fi

  return 1
}

warn_unmapped_enable_flags() {
  local key value
  for key in $(compgen -A variable | grep '^enable_'); do
    value="${!key:-0}"
    [[ "$value" == "1" ]] || continue

    case "$key" in
      enable_UNIMOD4|enable_REANNOTATE|enable_SEMI|enable_PEPTIDOFORMS|enable_PROTEOFORMS|enable_GEN_SPEC_LIB|enable_EXPORT_QUANT|enable_PRE_SEARCH|enable_pre_search)
        ;;
      *)
        warn "enable-flag '$key=1' ist gesetzt, aber in step2_tunelib.sh nicht auf ein DIA-NN Argument gemappt."
        ;;
    esac
  done
}

warn_unmapped_enable_flags

CLI_OVERRIDES=()
RAW_INPUTS=()
LIB_INPUT=""
REF_INPUT=""
NO_CAL_LIB="${NO_CAL_LIB:-0}"
FASTA_NAME=""
FASTA_ABS=""
CALTYPE_INPUT=""
MASS_ACC_CAL_INPUT=""
MIN_CAL_INPUT=""
RUN_LIST_INPUT=""
VERBOSE="${VERBOSE:-0}"
TEST_MODE=0
PROFILE="balanced"
PROFILE_PARALLEL=0
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
    --ref)
      ((i+=1))
      REF_INPUT="${ARGS[$i]:-}"
      [[ -n "$REF_INPUT" ]] || die "--ref needs a value"
      ;;
    --no-cal-lib)
      NO_CAL_LIB=1
      ;;
    --profile)
      ((i+=1))
      PROFILE="${ARGS[$i]:-}"
      [[ -n "$PROFILE" ]] || die "--profile needs a value (balanced|fast|slow|all)"
      ;;
    --profile-parallel)
      PROFILE_PARALLEL=1
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
    --caltype)
      ((i+=1))
      CALTYPE_INPUT="${ARGS[$i]:-}"
      [[ -n "$CALTYPE_INPUT" ]] || die "--caltype needs a value"
      ;;
    --mass-acc-cal)
      ((i+=1))
      MASS_ACC_CAL_INPUT="${ARGS[$i]:-}"
      [[ -n "$MASS_ACC_CAL_INPUT" ]] || die "--mass-acc-cal needs a numeric value"
      ;;
    --min-cal)
      ((i+=1))
      MIN_CAL_INPUT="${ARGS[$i]:-}"
      [[ -n "$MIN_CAL_INPUT" ]] || die "--min-cal needs an integer value"
      ;;
    --expname)
      ((i+=1))
      [[ -n "${ARGS[$i]:-}" ]] || die "--expname needs a value"
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
    -h|--help)
      sed -n '1,120p' "$0"
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

if [[ "$PROFILE_PARALLEL" == "1" && "$PROFILE" != "all" ]]; then
  warn "--profile-parallel is only used with --profile all; running normal mode for profile=${PROFILE}."
  PROFILE_PARALLEL=0
fi

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

if [[ -n "$MASS_ACC_CAL_INPUT" ]]; then
  MASS_ACC_CAL="$MASS_ACC_CAL_INPUT"
fi
if [[ -n "$MIN_CAL_INPUT" ]]; then
  MIN_CAL="$MIN_CAL_INPUT"
fi

if [[ -n "$CALTYPE_INPUT" && "$CALTYPE_INPUT" != "fixed" ]]; then
  die "Invalid --caltype '${CALTYPE_INPUT}'. Allowed: fixed"
fi

if [[ -n "${MASS_ACC_CAL:-}" ]]; then
  [[ "${MASS_ACC_CAL}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "Invalid --mass-acc-cal value '${MASS_ACC_CAL}' (expected number)"
fi
if [[ -n "${MIN_CAL:-}" ]]; then
  [[ "${MIN_CAL}" =~ ^[0-9]+$ ]] || die "Invalid --min-cal value '${MIN_CAL}' (expected integer)"
fi

parse_expname "$@"

if [[ "$TEST_MODE" == "1" ]]; then
  : "${OUTBASE:?OUTBASE not set}"
  outbase="${OUTBASE%/}"
  [[ -n "$outbase" ]] || outbase="/"
  EXPDIR="${outbase}/${EXPNAME}"
  S1DIR="${EXPDIR}/01_predlib"
  S2DIR="${EXPDIR}/02_tunelib"
  export EXPDIR S1DIR S2DIR
else
  set_paths_for_exp
fi

STEP_LOG="${S2DIR}/step2_tunelib.log"
if [[ "$TEST_MODE" != "1" ]]; then
  exec > >(tee -a "$STEP_LOG") 2>&1
  info "Step2 log file: ${STEP_LOG}"
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
elif [[ -n "${FASTA:-}" ]]; then
  if [[ "$FASTA" == /* ]]; then
    FASTA_FILE="$FASTA"
  elif [[ "$FASTA" == */* ]]; then
    FASTA_FILE="$FASTA"
  else
    fasta_base="${FASTABASE%/}"
    [[ -n "$fasta_base" ]] || fasta_base="/"
    FASTA_FILE="${fasta_base}/${FASTA}"
  fi
else
  die "Missing FASTA for Step 2. Set FASTA in config/step2.cfg or pass --fasta <filename> / --fastaabs <absolute path>."
fi
require_file "$FASTA_FILE"

if [[ -n "$LIB_INPUT" ]]; then
  PREDLIB="$(resolve_lib_input "$LIB_INPUT")"
else
  info "--lib not provided; searching in ${S1DIR} for '*.predicted.speclib'"
  PREDLIB="$(auto_detect_predlib "$S1DIR")"
fi
require_file "$PREDLIB"

ENABLE_PRE_SEARCH="${enable_PRE_SEARCH:-${enable_pre_search:-0}}"
CAL_MODE="ref"
[[ "$NO_CAL_LIB" =~ ^[01]$ ]] || die "Invalid NO_CAL_LIB '$NO_CAL_LIB' (allowed 0 or 1)"

if [[ "$NO_CAL_LIB" == "1" ]]; then
  [[ -z "$REF_INPUT" ]] || die "--no-cal-lib cannot be combined with --ref"
  [[ -z "$CALTYPE_INPUT" ]] || die "--no-cal-lib cannot be combined with --caltype"
  [[ -z "$MASS_ACC_CAL_INPUT" && -z "$MIN_CAL_INPUT" ]] || die "--no-cal-lib cannot be combined with --mass-acc-cal/--min-cal"
  CAL_MODE="none"
elif [[ "$CALTYPE_INPUT" == "fixed" ]]; then
  CAL_MODE="fixed"
fi

if [[ "$CAL_MODE" == "none" ]]; then
  info "--no-cal-lib enabled; Step2 will not use --ref and will not apply fixed calibration parameters."
elif [[ "$CAL_MODE" == "fixed" ]]; then
  if [[ -n "$REF_INPUT" ]]; then
    die "--ref cannot be used with --caltype fixed"
  fi

  [[ -n "${MASS_ACC_CAL:-}" ]] || die "--caltype fixed requires MASS_ACC_CAL (set in config/step2.cfg or via --mass-acc-cal)."
  [[ -n "${MIN_CAL:-}" ]] || die "--caltype fixed requires MIN_CAL (set in config/step2.cfg or via --min-cal)."
else
  if [[ -n "$MASS_ACC_CAL_INPUT" || -n "$MIN_CAL_INPUT" ]]; then
    die "--mass-acc-cal/--min-cal are only allowed with --caltype fixed"
  fi

  if [[ -n "$REF_INPUT" ]]; then
    REF_INPUT="$(resolve_ref_input "$REF_INPUT")"
    require_file "$REF_INPUT"
    info "--caltype not set; using explicit --ref: ${REF_INPUT}"
  else
    if [[ "$PROFILE" == "all" ]]; then
      info "--caltype not set; using profile refs from ${EXPDIR}/cal_lib/<profile>/calib_ref.parquet"
    else
      info "--caltype not set; using default ref mode for profile '${PROFILE}'"
    fi
  fi
fi

if [[ "$CAL_MODE" != "none" && "$ENABLE_PRE_SEARCH" != "1" ]]; then
  die "Calibration requires --pre-search. Set enable_PRE_SEARCH=1 in config/step2.cfg or pass --set enable_PRE_SEARCH=1."
fi

PROFILE_LIST=()
if [[ "$PROFILE" == "all" ]]; then
  PROFILE_LIST=(balanced fast slow)
else
  PROFILE_LIST=("$PROFILE")
fi

PROFILE_PARALLEL_THREADS=""
if [[ "$PROFILE_PARALLEL" == "1" && "$PROFILE" == "all" ]]; then
  [[ "${THREADS:-}" =~ ^[0-9]+$ ]] || die "THREADS must be numeric for --profile-parallel (current: '${THREADS:-}')"
  (( THREADS >= 1 )) || die "THREADS must be >= 1 for --profile-parallel"
  PROFILE_PARALLEL_THREADS=$(( THREADS / 3 ))
  if (( PROFILE_PARALLEL_THREADS < 1 )); then
    PROFILE_PARALLEL_THREADS=1
  fi
fi

if [[ "$TEST_MODE" == "1" ]]; then
  outbase_resolved="${OUTBASE%/}"
  [[ -n "$outbase_resolved" ]] || outbase_resolved="/"
  [[ -d "$outbase_resolved" ]] || die "OUTBASE does not exist (test mode): $outbase_resolved"
  [[ -w "$outbase_resolved" ]] || die "OUTBASE is not writable (test mode): $outbase_resolved"
  if [[ -e "$S2DIR" && ! -d "$S2DIR" ]]; then
    die "Step2 output path exists but is not a directory (test mode): $S2DIR"
  fi
fi

if [[ ${#RAW_INPUTS[@]} -eq 0 ]]; then
  if [[ -n "$RUN_LIST_INPUT" ]]; then
    RUN_LIST_FILE="$RUN_LIST_INPUT"
  else
    RUN_LIST_FILE="${LISTBASE}/good_runs.txt"
  fi

  if [[ -f "$RUN_LIST_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      trimmed="${line#${line%%[![:space:]]*}}"
      [[ -z "$trimmed" ]] && continue
      [[ "$trimmed" == \#* ]] && continue
      RAW_INPUTS+=("$trimmed")
    done < "$RUN_LIST_FILE"
  else
    die "No raw files provided. Use --f <rawfile> (repeatable) or create ${RUN_LIST_FILE}."
  fi
fi

[[ ${#RAW_INPUTS[@]} -gt 0 ]] || die "No raw files collected for Step 2"

RAW_FILES=()
for raw_input in "${RAW_INPUTS[@]}"; do
  raw_resolved="$(resolve_raw_input "$raw_input")"
  require_file "$raw_resolved"
  RAW_FILES+=("$raw_resolved")
done

enable_GEN_SPEC_LIB="${enable_GEN_SPEC_LIB:-1}"

run_step2_for_profile() {
  local profile_name="$1"
  local profile_threads="$2"
  local profile_outdir report outlib step2_temp_dir startcmd_file profile_log
  local ref_file_for_profile=""
  local -a diann_cmd

  profile_outdir="${S2DIR}/${profile_name}"

  report="${profile_outdir}/tune_search_report_${profile_name}.parquet"
  outlib="${profile_outdir}/tune_lib_${profile_name}.parquet"
  step2_temp_dir="${profile_outdir}/Temp"
  startcmd_file="${profile_outdir}/startcmd_${profile_name}.txt"
  profile_log="${profile_outdir}/step2_tunelib_${profile_name}.log"

  if [[ "$TEST_MODE" != "1" ]]; then
    mkdir -p "$step2_temp_dir"
  fi

  if [[ "$CAL_MODE" == "ref" ]]; then
    if [[ -n "$REF_INPUT" ]]; then
      ref_file_for_profile="$(resolve_ref_input "$REF_INPUT")"
    else
      ref_file_for_profile="$(resolve_default_ref_for_profile "$profile_name" || true)"
      [[ -n "$ref_file_for_profile" ]] || die "No default calibration ref found for profile '${profile_name}' (expected ${EXPDIR}/cal_lib/${profile_name}/calib_ref_${profile_name}.parquet)."
      info "Using default profile ref (${profile_name}): ${ref_file_for_profile}"
    fi
    require_file "$ref_file_for_profile"
  fi

  diann_cmd=(
    "$DIANN_BIN"
    --threads "$profile_threads"
    --verbose "$DIANN_VERBOSE"
    --lib "$PREDLIB"
    --out "$report"
    --out-lib "$outlib"
    --temp "$step2_temp_dir"
    --qvalue "$QVALUE"
    --cut "$CUT"
    --missed-cleavages "$MISSED_CLEAVAGES"
    --min-pep-len "$MIN_PEP_LEN" --max-pep-len "$MAX_PEP_LEN"
    --min-pr-mz "$MIN_PR_MZ" --max-pr-mz "$MAX_PR_MZ"
    --min-pr-charge "$MIN_PR_CHARGE" --max-pr-charge "$MAX_PR_CHARGE"
    --min-fr-mz "$MIN_FR_MZ" --max-fr-mz "$MAX_FR_MZ"
    --mass-acc-ms1 "$MASS_ACC_MS1" --mass-acc "${MASS_ACC_MS2:-${MASS_ACC}}"
    --fasta "$FASTA_FILE"
  )

  for raw_file in "${RAW_FILES[@]}"; do
    diann_cmd+=(--f "$raw_file")
  done

  if [[ "$ENABLE_PRE_SEARCH" == "1" ]]; then
    diann_cmd+=(--pre-search)
  fi

  if [[ "$CAL_MODE" == "ref" ]]; then
    diann_cmd+=(--ref "$ref_file_for_profile")
  fi

  if [[ "$CAL_MODE" == "fixed" ]]; then
    diann_cmd+=(--mass-acc-cal "$MASS_ACC_CAL")
    diann_cmd+=(--min-cal "$MIN_CAL")
  fi

  if [[ "${enable_GEN_SPEC_LIB}" == "1" ]]; then
    diann_cmd+=(--gen-spec-lib)
  fi

  # Carafe requires the MS2.Scan report column to map DIA-NN identifications
  # back to the scans in its training mzML files.
  if [[ "${enable_EXPORT_QUANT:-0}" == "1" ]]; then
    diann_cmd+=(--export-quant)
  fi

  if [[ "${enable_REANNOTATE:-0}" == "1" ]]; then
    diann_cmd+=(--reannotate)
  fi

  if [[ "${enable_SEMI:-0}" == "1" ]]; then
    diann_cmd+=(--semi)
  fi

  if [[ "${enable_PEPTIDOFORMS:-0}" == "1" ]]; then
    diann_cmd+=(--peptidoforms)
  fi

  if [[ "${enable_PROTEOFORMS:-0}" == "1" ]]; then
    diann_cmd+=(--proteoforms)
  fi

  if [[ "${enable_UNIMOD4:-0}" == "1" ]]; then
    diann_cmd+=(--unimod4)
  fi

  append_mod_args diann_cmd

  if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    read -r -a EXTRA_FLAGS_ARR <<< "$EXTRA_FLAGS"
    diann_cmd+=("${EXTRA_FLAGS_ARR[@]}")
  fi

  if [[ "$TEST_MODE" == "1" ]]; then
    echo ""
    echo "Profile    : ${profile_name}"
    echo "DIANN_BIN  : $DIANN_BIN"
    echo "EXPNAME    : ${EXPNAME:-}"
    echo "FASTA      : $FASTA_FILE"
    echo "PREDLIB    : $PREDLIB"
    echo "OUTDIR     : $profile_outdir"
    echo "REPORT     : $report"
    echo "OUTLIB     : $outlib"
    echo "TMPDIR     : $step2_temp_dir"
    echo "THREADS    : $profile_threads"
    echo "PRE_SEARCH : $ENABLE_PRE_SEARCH"
    if [[ "$CAL_MODE" == "ref" ]]; then
      echo "CAL_MODE   : ref"
      echo "REF        : ${ref_file_for_profile}"
    elif [[ "$CAL_MODE" == "fixed" ]]; then
      echo "CAL_MODE   : fixed"
      echo "MASS_ACC_CAL: ${MASS_ACC_CAL:-<none>}"
      echo "MIN_CAL   : ${MIN_CAL:-<none>}"
      echo "REF        : <none>"
    else
      echo "CAL_MODE   : none"
      echo "REF        : <none>"
    fi
    echo "Runs       : ${#RAW_FILES[@]}"
    echo "DIA-NN command (without --test):"
    printf '%q ' "${diann_cmd[@]}"
    echo
    return 0
  fi

  {
    echo "# profile: ${profile_name}"
    echo "# script invocation command"
    printf '%q ' "$0"
    printf '%q ' "$@"
    echo
    echo "# step2_tunelib start command"
    printf '%q ' "${diann_cmd[@]}"
    echo
  } > "$startcmd_file"
  info "Saved start command (${profile_name}): ${startcmd_file}"

  {
    echo "Profile    : ${profile_name}"
    echo "DIANN_BIN  : $DIANN_BIN"
    echo "EXPNAME    : ${EXPNAME:-}"
    echo "FASTA      : $FASTA_FILE"
    echo "PREDLIB    : $PREDLIB"
    echo "OUTDIR     : $profile_outdir"
    echo "REPORT     : $report"
    echo "OUTLIB     : $outlib"
    echo "TMPDIR     : $step2_temp_dir"
    echo "THREADS    : $profile_threads"
    echo "CAL_MODE   : $CAL_MODE"
    if [[ "$CAL_MODE" == "ref" ]]; then
      echo "REF        : ${ref_file_for_profile}"
    else
      echo "REF        : <none>"
    fi
    echo "Runs       : ${#RAW_FILES[@]}"
    echo "Running DIA-NN..."
    run_cmd_with_diagnostics "Step2 DIA-NN run (profile=${profile_name})" diann_cmd "$profile_log"
  } 2>&1 | tee -a "$profile_log"
}

if [[ "$TEST_MODE" == "1" ]]; then
  echo "TEST MODE: no directories/files will be created; DIA-NN will not run."
  echo "PROFILE    : $PROFILE"
  echo "PARALLEL   : $PROFILE_PARALLEL"
  if [[ "$PROFILE_PARALLEL" == "1" && "$PROFILE" == "all" ]]; then
    echo "THREADS/profile: $PROFILE_PARALLEL_THREADS"
  fi
  for profile_name in "${PROFILE_LIST[@]}"; do
    if [[ "$PROFILE_PARALLEL" == "1" && "$PROFILE" == "all" ]]; then
      run_step2_for_profile "$profile_name" "$PROFILE_PARALLEL_THREADS"
    else
      run_step2_for_profile "$profile_name" "$THREADS"
    fi
  done
  echo ""
  echo "OK: validation successful (test mode)."
  exit 0
fi

info "Step2 (tunelib) started"
debug "PREDLIB=${PREDLIB}"
debug "RAW files count=${#RAW_FILES[@]}"

if [[ "$PROFILE_PARALLEL" == "1" && "$PROFILE" == "all" ]]; then
  info "Running Step2 for profiles in parallel (threads per profile=${PROFILE_PARALLEL_THREADS})"
  STEP2_PROFILE_PIDS=()
  STEP2_PROFILE_NAMES=()
  for profile_name in "${PROFILE_LIST[@]}"; do
    run_step2_for_profile "$profile_name" "$PROFILE_PARALLEL_THREADS" &
    STEP2_PROFILE_PIDS+=("$!")
    STEP2_PROFILE_NAMES+=("$profile_name")
  done

  step2_failed=0
  for idx in "${!STEP2_PROFILE_PIDS[@]}"; do
    step2_pid="${STEP2_PROFILE_PIDS[$idx]}"
    profile_name="${STEP2_PROFILE_NAMES[$idx]}"
    if ! wait "$step2_pid"; then
      step2_failed=1
      warn "Step2 profile failed: ${profile_name}"
    fi
  done

  (( step2_failed == 0 )) || die "At least one Step2 profile run failed"
else
  for profile_name in "${PROFILE_LIST[@]}"; do
    run_step2_for_profile "$profile_name" "$THREADS"
  done
fi

info "Step2 (tunelib) finished"
