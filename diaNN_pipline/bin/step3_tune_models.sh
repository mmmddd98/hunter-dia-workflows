#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
# step3.cfg should be KEY=VALUE lines
source "${PIPELINE_ROOT}/config/step3.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

warn(){ _log "WARN" "$*"; }

expand_tune_lr_values() {
  local spec="$1"
  python - "$spec" <<'PY'
import sys

spec = sys.argv[1].strip()
if not spec:
  raise SystemExit("Empty TUNE_LR value")

def fmt(v: float) -> str:
  s = f"{v:.12g}"
  return s

values = []

if "," in spec:
  parts = [p.strip() for p in spec.split(",") if p.strip()]
  if not parts:
    raise SystemExit("Invalid TUNE_LR list syntax")
  for p in parts:
    try:
      v = float(p)
    except ValueError:
      raise SystemExit(f"Invalid TUNE_LR value '{p}'")
    values.append(v)
elif ":" in spec:
  parts = [p.strip() for p in spec.split(":")]
  if len(parts) != 3:
    raise SystemExit("Range syntax must be start:stop:step")
  try:
    start, stop, step = map(float, parts)
  except ValueError:
    raise SystemExit("Range syntax expects numeric start:stop:step")
  if step == 0:
    raise SystemExit("Range step must not be 0")
  if (stop - start) * step < 0:
    raise SystemExit("Range step direction does not reach stop")

  v = start
  tol = abs(step) * 1e-9
  if step > 0:
    while v <= stop + tol:
      values.append(v)
      v += step
  else:
    while v >= stop - tol:
      values.append(v)
      v += step
else:
  try:
    values = [float(spec)]
  except ValueError:
    raise SystemExit(f"Invalid TUNE_LR value '{spec}'")

if not values:
  raise SystemExit("No LR values produced from TUNE_LR")

filtered = []
skipped_zero = 0
for v in values:
  if v < 0:
    raise SystemExit(f"TUNE_LR values must be >= 0, got {v}")
  if v == 0:
    skipped_zero += 1
    continue
  filtered.append(v)

if not filtered:
  raise SystemExit("All TUNE_LR values resolve to 0. At least one value must be > 0.")

if skipped_zero:
  print(f"[step3] Skipping {skipped_zero} zero LR value(s); only LR > 0 will run.", file=sys.stderr)

for v in filtered:
  print(fmt(v))
PY
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

resolve_default_tunelib_for_profile() {
  local profile_name="$1"
  local profile_tunelib="${S2DIR}/${profile_name}/tune_lib_${profile_name}.parquet"
  local profile_tunelib_legacy_in_profile="${S2DIR}/${profile_name}/tune_lib.parquet"
  local legacy_tunelib="${S2DIR}/tune_lib.parquet"

  if [[ -f "$profile_tunelib" ]]; then
    echo "$profile_tunelib"
    return 0
  fi

  if [[ -f "$profile_tunelib_legacy_in_profile" ]]; then
    warn "Profile tune_lib with suffix not found for '${profile_name}', falling back to legacy-in-profile tune_lib: ${profile_tunelib_legacy_in_profile}"
    echo "$profile_tunelib_legacy_in_profile"
    return 0
  fi

  if [[ "$profile_name" != "all" && -f "$legacy_tunelib" ]]; then
    warn "Profile tune_lib not found for '${profile_name}', falling back to legacy tune_lib: ${legacy_tunelib}"
    echo "$legacy_tunelib"
    return 0
  fi

  return 1
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

# -------------------------
# CLI parsing
# -------------------------
CLI_OVERRIDES=()
FASTA_NAME=""
FASTA_ABS=""
REF_INPUT=""
NO_CAL_LIB="${NO_CAL_LIB:-0}"
VERBOSE="${VERBOSE:-0}"
TEST_MODE=0
PROFILE="balanced"
PROFILE_PARALLEL=0
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
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
      sed -n '1,140p' "$0"
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
[[ "$DIANN_VERBOSE" =~ ^[0-4]$ ]] || die "Invalid DIANN_VERBOSE '$DIANN_VERBOSE' (allowed 0..4)"
[[ "$NO_CAL_LIB" =~ ^[01]$ ]] || die "Invalid NO_CAL_LIB '$NO_CAL_LIB' (allowed 0 or 1)"
if [[ "$NO_CAL_LIB" == "1" && -n "$REF_INPUT" ]]; then
  die "--no-cal-lib cannot be combined with --ref"
fi

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

has_cli_override() {
  local lookup_key="$1"
  local key
  for key in "${CLI_OVERRIDES[@]}"; do
    if [[ "$key" == "$lookup_key" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_instrument() {
  local raw_value="$1"
  local compact

  compact="${raw_value,,}"
  compact="${compact//[[:space:]_-]/}"

  case "$compact" in
    "")
      echo ""
      ;;
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

apply_tune_targets() {
  local target_spec="$1"
  local target_spec_norm token
  local -a target_tokens

  target_spec_norm="${target_spec,,}"
  target_spec_norm="${target_spec_norm//[[:space:]]/}"

  TUNE_RT=0
  TUNE_IM=0
  TUNE_FR=0

  if [[ -z "$target_spec_norm" ]]; then
    return 0
  fi

  if [[ "$target_spec_norm" == "all" ]]; then
    TUNE_RT=1
    TUNE_IM=1
    TUNE_FR=1
    return 0
  fi

  IFS=',' read -r -a target_tokens <<< "$target_spec_norm"
  for token in "${target_tokens[@]}"; do
    case "$token" in
      rt)
        TUNE_RT=1
        ;;
      im)
        TUNE_IM=1
        ;;
      fr)
        TUNE_FR=1
        ;;
      *)
        die "Invalid TUNE_TARGETS token '${token}' (allowed: rt,im,fr or all)"
        ;;
    esac
  done
}

# -------------------------
# Paths
# -------------------------
parse_expname "$@"
if [[ "$TEST_MODE" == "1" ]]; then
  : "${OUTBASE:?OUTBASE not set}"
  outbase="${OUTBASE%/}"
  [[ -n "$outbase" ]] || outbase="/"
  EXPDIR="${outbase}/${EXPNAME}"
  S2DIR="${EXPDIR}/02_tunelib"
  STEP3_BASEDIR_NAME="${STEP3_BASEDIR_NAME:-03_tune_models}"
  [[ "$STEP3_BASEDIR_NAME" != /* && "$STEP3_BASEDIR_NAME" != *".."* ]] || die "STEP3_BASEDIR_NAME must be a relative path without '..'"
  S3DIR="${EXPDIR}/${STEP3_BASEDIR_NAME}"
  export EXPDIR S2DIR S3DIR
else
  set_paths_for_exp
fi
require_exec "$DIANN_BIN"

PROFILE_LIST=()
if [[ "$PROFILE" == "all" ]]; then
  PROFILE_LIST=(balanced fast slow)
else
  PROFILE_LIST=("$PROFILE")
fi

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
  die "Missing FASTA for Step 3. Set FASTA in config/step3.cfg or pass --fasta <filename> / --fastaabs <absolute path>."
fi
require_file "$FASTA_FILE"

RESOLVED_REF_INPUT=""
if [[ "$NO_CAL_LIB" == "1" ]]; then
  info "--no-cal-lib enabled; Step3 will not use --ref."
elif [[ -n "$REF_INPUT" ]]; then
  RESOLVED_REF_INPUT="$(resolve_ref_input "$REF_INPUT")"
  require_file "$RESOLVED_REF_INPUT"
  if [[ "$PROFILE" == "all" ]]; then
    warn "Using explicit --ref for all profiles: ${RESOLVED_REF_INPUT}"
  fi
fi

if [[ "$TEST_MODE" != "1" ]]; then
  mkdir -p "$S3DIR"
else
  outbase_resolved="${OUTBASE%/}"
  [[ -n "$outbase_resolved" ]] || outbase_resolved="/"
  [[ -d "$outbase_resolved" ]] || die "OUTBASE does not exist (test mode): $outbase_resolved"
  [[ -w "$outbase_resolved" ]] || die "OUTBASE is not writable (test mode): $outbase_resolved"
  if [[ -e "$S3DIR" && ! -d "$S3DIR" ]]; then
    die "Step3 output path exists but is not a directory (test mode): $S3DIR"
  fi
fi

LOSS_EXTRACT_SCRIPT="${PIPELINE_ROOT}/bin/extract_loss_curves.sh"
COMBINED_PLOT_SCRIPT="${PIPELINE_ROOT}/bin/plot_combined_lr_losses.sh"

require_file "$LOSS_EXTRACT_SCRIPT"
require_file "$COMBINED_PLOT_SCRIPT"

# -------------------------
# Defaults (safe)
# -------------------------
TUNE_RT="${TUNE_RT:-}"
TUNE_IM="${TUNE_IM:-}"
TUNE_FR="${TUNE_FR:-}"
TUNE_TARGETS="${TUNE_TARGETS:-rt,fr}"
INSTRUMENT="${INSTRUMENT:-}"

if ! has_cli_override "TUNE_RT" && ! has_cli_override "TUNE_IM" && ! has_cli_override "TUNE_FR"; then
  tune_targets_norm="${TUNE_TARGETS,,}"
  tune_targets_norm="${tune_targets_norm//[[:space:]]/}"
  if [[ -z "$tune_targets_norm" ]]; then
    TUNE_RT=1
    TUNE_IM=0
    TUNE_FR=1
  else
    apply_tune_targets "$TUNE_TARGETS"
  fi
fi

TUNE_RT="${TUNE_RT:-1}"
TUNE_IM="${TUNE_IM:-0}"
TUNE_FR="${TUNE_FR:-1}"

INSTRUMENT_NORMALIZED="$(normalize_instrument "$INSTRUMENT")"
if [[ "$TUNE_IM" == "1" && "$INSTRUMENT_NORMALIZED" != "timstof" ]]; then
  if [[ -n "$INSTRUMENT" ]]; then
    warn "Disabling TUNE_IM: instrument '${INSTRUMENT}' is not timstof."
  else
    warn "Disabling TUNE_IM: instrument is unknown. Set --set INSTRUMENT=timstof to enable IM tuning."
  fi
  TUNE_IM=0
fi

# README default lr = 0.0005 :contentReference[oaicite:7]{index=7}
TUNE_LR="${TUNE_LR:-0.0005}"
TUNE_RESTRICT_LAYERS="${TUNE_RESTRICT_LAYERS:-0}"
ENABLE_PRE_SEARCH="${ENABLE_PRE_SEARCH:-${enable_PRE_SEARCH:-${enable_pre_search:-0}}}"

[[ "$TUNE_RT" =~ ^[01]$ ]] || die "Invalid TUNE_RT '$TUNE_RT' (allowed 0 or 1)"
[[ "$TUNE_IM" =~ ^[01]$ ]] || die "Invalid TUNE_IM '$TUNE_IM' (allowed 0 or 1)"
[[ "$TUNE_FR" =~ ^[01]$ ]] || die "Invalid TUNE_FR '$TUNE_FR' (allowed 0 or 1)"
if [[ -n "$TUNE_TARGETS" ]]; then
  tune_targets_validate="${TUNE_TARGETS,,}"
  tune_targets_validate="${tune_targets_validate//[[:space:]]/}"
  [[ "$tune_targets_validate" =~ ^(all|rt|im|fr|rt,im|rt,fr|im,rt|im,fr|fr,rt|fr,im|rt,im,fr|rt,fr,im|im,rt,fr|im,fr,rt|fr,rt,im|fr,im,rt)$ ]] || die "Invalid TUNE_TARGETS '$TUNE_TARGETS'"
fi
[[ "$TUNE_RESTRICT_LAYERS" =~ ^[01]$ ]] || die "Invalid TUNE_RESTRICT_LAYERS '$TUNE_RESTRICT_LAYERS' (allowed 0 or 1)"
[[ "$ENABLE_PRE_SEARCH" =~ ^[01]$ ]] || die "Invalid enable_PRE_SEARCH '$ENABLE_PRE_SEARCH' (allowed 0 or 1)"

if [[ "$TUNE_RT" != "1" && "$TUNE_IM" != "1" && "$TUNE_FR" != "1" ]]; then
  die "No active tuning dimensions. Set TUNE_TARGETS to include at least one of rt, im, fr (or use --set TUNE_RT/TUNE_IM/TUNE_FR)."
fi

mapfile -t TUNE_LR_VALUES < <(expand_tune_lr_values "$TUNE_LR")
[[ ${#TUNE_LR_VALUES[@]} -gt 0 ]] || die "No TUNE_LR values resolved from '$TUNE_LR'"

MASS_ACC_VALUE="${MASS_ACC_MS2:-${MASS_ACC:-20}}"
MASS_ACC_MS1_VALUE="${MASS_ACC_MS1:-${MASS_ACC:-10}}"

build_diann_cmd_for_lr() {
  local lr="$1"
  local s3_tunelib="$2"
  local run_out="$3"
  local ref_file="$4"
  local threads_value="$5"

  local -a cmd=(
    "$DIANN_BIN"
    --threads "$threads_value"
    --verbose "$DIANN_VERBOSE"
    --tune-lib "$s3_tunelib"
    --out "$run_out"
    --fasta "$FASTA_FILE"
    --qvalue "$QVALUE"
    --cut "$CUT"
    --missed-cleavages "$MISSED_CLEAVAGES"
    --min-pep-len "$MIN_PEP_LEN" --max-pep-len "$MAX_PEP_LEN"
    --min-pr-mz "$MIN_PR_MZ" --max-pr-mz "$MAX_PR_MZ"
    --min-pr-charge "$MIN_PR_CHARGE" --max-pr-charge "$MAX_PR_CHARGE"
    --min-fr-mz "$MIN_FR_MZ" --max-fr-mz "$MAX_FR_MZ"
    --mass-acc-ms1 "$MASS_ACC_MS1_VALUE" --mass-acc "$MASS_ACC_VALUE"
  )

  if [[ -n "$ref_file" ]]; then
    cmd+=(--ref "$ref_file")
  fi

  if [[ "$TUNE_RT" == "1" ]]; then
    cmd+=(--tune-rt)
  fi
  if [[ "$TUNE_IM" == "1" ]]; then
    cmd+=(--tune-im)
  fi
  if [[ "$TUNE_FR" == "1" ]]; then
    cmd+=(--tune-fr)
  fi

  cmd+=(--tune-lr "$lr")

  if [[ "$TUNE_RESTRICT_LAYERS" == "1" ]]; then
    cmd+=(--tune-restrict-layers)
  fi
  if [[ "$ENABLE_PRE_SEARCH" == "1" ]]; then
    cmd+=(--pre-search)
  fi
  if [[ "${enable_REANNOTATE:-0}" == "1" ]]; then
    cmd+=(--reannotate)
  fi
  if [[ "${enable_SEMI:-0}" == "1" ]]; then
    cmd+=(--semi)
  fi
  if [[ "${enable_PEPTIDOFORMS:-0}" == "1" ]]; then
    cmd+=(--peptidoforms)
  fi
  if [[ "${enable_PROTEOFORMS:-0}" == "1" ]]; then
    cmd+=(--proteoforms)
  fi
  if [[ "${enable_UNIMOD4:-0}" == "1" ]]; then
    cmd+=(--unimod4)
  fi

  append_mod_args cmd

  if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    read -r -a EXTRA_FLAGS_ARR <<< "$EXTRA_FLAGS"
    cmd+=("${EXTRA_FLAGS_ARR[@]}")
  fi

  printf '%q ' "${cmd[@]}"
  echo
}

PROFILE_PARALLEL_THREADS=""
if [[ "$PROFILE_PARALLEL" == "1" && "$PROFILE" == "all" ]]; then
  [[ "${THREADS:-}" =~ ^[0-9]+$ ]] || die "THREADS must be numeric for --profile-parallel (current: '${THREADS:-}')"
  (( THREADS >= 1 )) || die "THREADS must be >= 1 for --profile-parallel"
  PROFILE_PARALLEL_THREADS=$(( THREADS / 3 ))
  if (( PROFILE_PARALLEL_THREADS < 1 )); then
    PROFILE_PARALLEL_THREADS=1
  fi
fi

run_step3_for_profile() {
  local profile_name="$1"
  local threads_value="$2"
  local profile_s3dir tunelib_src ref_file log_file startcmd_file name_suffix profile_dir_suffix
  local current_diann_bin_name current_diann_version

  profile_dir_suffix="${STEP3_PROFILE_DIR_SUFFIX:-}"
  if [[ -n "$profile_dir_suffix" ]]; then
    [[ "$profile_dir_suffix" =~ ^[A-Za-z0-9._-]+$ ]] || die "STEP3_PROFILE_DIR_SUFFIX may contain only letters, digits, ., _, -"
    profile_dir_suffix="_${profile_dir_suffix}"
  fi
  profile_s3dir="${S3DIR}/${profile_name}${profile_dir_suffix}"
  name_suffix="${STEP3_NAME_SUFFIX:-}"
  if [[ -n "$name_suffix" ]]; then
    [[ "$name_suffix" =~ ^[A-Za-z0-9._-]+$ ]] || die "STEP3_NAME_SUFFIX may contain only letters, digits, ., _, -"
    name_suffix="_${name_suffix}"
  fi
  current_diann_bin_name="${DIANN_BIN##*/}"
  current_diann_version="${current_diann_bin_name#diann-}"
  tunelib_src="$(resolve_default_tunelib_for_profile "$profile_name" || true)"
  [[ -n "$tunelib_src" ]] || die "Missing profile tune_lib for '${profile_name}' (expected ${S2DIR}/${profile_name}/tune_lib_${profile_name}.parquet)"
  require_file "$tunelib_src"

  if [[ "$NO_CAL_LIB" == "1" ]]; then
    ref_file=""
  elif [[ -n "$RESOLVED_REF_INPUT" ]]; then
    ref_file="$RESOLVED_REF_INPUT"
  else
    ref_file="$(resolve_default_ref_for_profile "$profile_name" || true)"
    [[ -n "$ref_file" ]] || die "Missing profile ref for '${profile_name}' (expected ${EXPDIR}/cal_lib/${profile_name}/calib_ref_${profile_name}.parquet)"
    require_file "$ref_file"
  fi

  if [[ "$TEST_MODE" != "1" ]]; then
    mkdir -p "$profile_s3dir"
  fi

  log_file="${profile_s3dir}/tune_models_${profile_name}${name_suffix}.log"
  startcmd_file="${profile_s3dir}/startcmd_${profile_name}${name_suffix}.txt"

  if [[ "$TEST_MODE" == "1" ]]; then
    echo "PROFILE    : ${profile_name}"
    echo "DIANN_BIN  : $DIANN_BIN"
    echo "EXPNAME    : ${EXPNAME:-}"
    echo "TUNELIB_SRC: ${tunelib_src}"
    echo "REF        : ${ref_file:-<none>}"
    echo "FASTA      : $FASTA_FILE"
    echo "OUTDIR     : $profile_s3dir"
    echo "THREADS    : $threads_value"
    echo "INSTRUMENT : ${INSTRUMENT:-<unset>} (normalized=${INSTRUMENT_NORMALIZED:-<unset>})"
    echo "TUNE_TARGETS: ${TUNE_TARGETS}"
    echo "TUNE_RT    : $TUNE_RT"
    echo "TUNE_IM    : $TUNE_IM"
    echo "TUNE_FR    : $TUNE_FR"
    echo "TUNE_LR spec: $TUNE_LR"
    echo "TUNE_LR values: ${TUNE_LR_VALUES[*]}"
    echo "PRE_SEARCH : $ENABLE_PRE_SEARCH"
    echo "TUNE_RESTRICT_LAYERS: $TUNE_RESTRICT_LAYERS"
    for lr in "${TUNE_LR_VALUES[@]}"; do
      lr_tag="$(printf '%s' "$lr" | tr '.' 'p')"
      run_dir="${profile_s3dir}/lr_${lr_tag}"
      s3_tunelib="${run_dir}/tune_lib_${profile_name}${name_suffix}_lr_${lr_tag}.parquet"
      run_out="${run_dir}/tune_report_${profile_name}${name_suffix}_lr_${lr_tag}.parquet"
      run_log="${run_dir}/run.log"
      echo "RUN_DIR (profile=${profile_name}, lr=${lr}) : ${run_dir}"
      echo "TUNELIB (lr=${lr}) : ${s3_tunelib}"
      echo "OUT (lr=${lr}) : ${run_out}"
      echo "RUN_LOG (lr=${lr}) : ${run_log}"
      echo "LOSS CSV/PNG (if model present in log):"
      echo "  ${run_dir}/LR_${lr_tag}_IM_LOSS.csv"
      echo "  ${run_dir}/LR_${lr_tag}_IM_LOSS.png"
      echo "  ${run_dir}/LR_${lr_tag}_RT_LOSS.csv"
      echo "  ${run_dir}/LR_${lr_tag}_RT_LOSS.png"
      echo "  ${run_dir}/LR_${lr_tag}_FR_LOSS.csv"
      echo "  ${run_dir}/LR_${lr_tag}_FR_LOSS.png"
      echo "DIA-NN command (lr=${lr}, without --test):"
      build_diann_cmd_for_lr "$lr" "$s3_tunelib" "$run_out" "$ref_file" "$threads_value"
    done
    echo "COMBINED PLOTS (created after all LR runs):"
    echo "  ${profile_s3dir}/ALL_LR_LOSS.csv"
    echo "  ${profile_s3dir}/ALL_LR_IM_LOSS.png"
    echo "  ${profile_s3dir}/ALL_LR_RT_LOSS.png"
    echo "  ${profile_s3dir}/ALL_LR_FR_LOSS.png"
    return 0
  fi

  {
    echo "# profile: ${profile_name}"
    echo "# script invocation command"
    printf '%q ' "$0"
    printf '%q ' "$@"
    echo
    echo "# step3_tune_models start commands"
    for lr in "${TUNE_LR_VALUES[@]}"; do
      lr_tag="$(printf '%s' "$lr" | tr '.' 'p')"
      run_dir="${profile_s3dir}/lr_${lr_tag}"
      s3_tunelib="${run_dir}/tune_lib_${profile_name}${name_suffix}_lr_${lr_tag}.parquet"
      run_out="${run_dir}/tune_report_${profile_name}${name_suffix}_lr_${lr_tag}.parquet"
      build_diann_cmd_for_lr "$lr" "$s3_tunelib" "$run_out" "$ref_file" "$threads_value"
    done
  } > "$startcmd_file"

  {
    echo "PROFILE=${profile_name}"
    echo "TUNELIB source=${tunelib_src}"
    echo "REF=${ref_file:-<none>}"
    echo "FASTA=${FASTA_FILE}"
    echo "THREADS=${threads_value}"
    echo "INSTRUMENT=${INSTRUMENT:-<unset>} (normalized=${INSTRUMENT_NORMALIZED:-<unset>})"
    echo "TUNE_TARGETS=${TUNE_TARGETS}"
    echo "TUNE_RT=${TUNE_RT}  TUNE_IM=${TUNE_IM}  TUNE_FR=${TUNE_FR}"
    echo "TUNE_LR spec=${TUNE_LR}"
    echo "TUNE_LR values=${TUNE_LR_VALUES[*]}"
    echo "PRE_SEARCH=${ENABLE_PRE_SEARCH}  TUNE_RESTRICT_LAYERS=${TUNE_RESTRICT_LAYERS}"
    echo "Saved start command: ${startcmd_file}"

    for lr in "${TUNE_LR_VALUES[@]}"; do
      lr_tag="$(printf '%s' "$lr" | tr '.' 'p')"
      run_dir="${profile_s3dir}/lr_${lr_tag}"
      s3_tunelib="${run_dir}/tune_lib_${profile_name}${name_suffix}_lr_${lr_tag}.parquet"
      run_out="${run_dir}/tune_report_${profile_name}${name_suffix}_lr_${lr_tag}.parquet"
      run_log="${run_dir}/run.log"
      run_startcmd="${run_dir}/startcmd_${profile_name}${name_suffix}.txt"

      mkdir -p "$run_dir"
      cp -f "$tunelib_src" "$s3_tunelib"
      echo "Running profile=${profile_name}, LR=${lr} in ${run_dir}"

      {
        echo "# profile: ${profile_name}"
        echo "# script invocation command"
        printf '%q ' "$0"
        printf '%q ' "$@"
        echo
        echo "# step3_tune_models start command (lr=${lr})"
      } > "$run_startcmd"

      cmd_string="$(build_diann_cmd_for_lr "$lr" "$s3_tunelib" "$run_out" "$ref_file" "$threads_value")"
      echo "$cmd_string" >> "$run_startcmd"
      run_cmd_string_with_diagnostics "Step3 DIA-NN run (profile=${profile_name}, lr=${lr})" "$cmd_string" "$run_log" 2>&1 | tee -a "$run_log"

      echo "Extracting loss curves for profile=${profile_name}, LR=${lr} from ${run_log}"
      LOSS_EXTRACT_CMD=(
        env LOSS_PLOTS_DIR="${run_dir}/loss_plots" bash "$LOSS_EXTRACT_SCRIPT"
        --log "$run_log"
        --outdir "$run_dir"
        --lr-tag "$lr_tag"
        --diann-version "$current_diann_version"
      )
      run_cmd_with_diagnostics "Step3 loss extraction (profile=${profile_name}, lr=${lr})" LOSS_EXTRACT_CMD "$log_file"
    done

    echo "Generating combined LR loss plots in ${profile_s3dir}"
    COMBINED_PLOT_CMD=(env LOSS_PLOTS_DIR="${profile_s3dir}/loss_plots" bash "$COMBINED_PLOT_SCRIPT" --s3dir "$profile_s3dir" --diann-version "$current_diann_version")
    run_cmd_with_diagnostics "Step3 combined LR plots (profile=${profile_name})" COMBINED_PLOT_CMD "$log_file"
  } 2>&1 | tee -a "$log_file"
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
      run_step3_for_profile "$profile_name" "$PROFILE_PARALLEL_THREADS"
    else
      run_step3_for_profile "$profile_name" "$THREADS"
    fi
    echo ""
  done
  echo "OK: validation successful (test mode)."
  exit 0
fi

info "Step3 (tune models) started"
if [[ "$PROFILE_PARALLEL" == "1" && "$PROFILE" == "all" ]]; then
  info "Running Step3 for profiles in parallel (threads per profile=${PROFILE_PARALLEL_THREADS})"
  STEP3_PIDS=()
  STEP3_PROFILES=()
  for profile_name in "${PROFILE_LIST[@]}"; do
    run_step3_for_profile "$profile_name" "$PROFILE_PARALLEL_THREADS" &
    STEP3_PIDS+=("$!")
    STEP3_PROFILES+=("$profile_name")
  done

  step3_failed=0
  for idx in "${!STEP3_PIDS[@]}"; do
    step3_pid="${STEP3_PIDS[$idx]}"
    step3_profile="${STEP3_PROFILES[$idx]}"
    if ! wait "$step3_pid"; then
      step3_failed=1
      warn "Step3 profile failed: ${step3_profile}"
    fi
  done

  (( step3_failed == 0 )) || die "At least one Step3 profile run failed"
else
  for profile_name in "${PROFILE_LIST[@]}"; do
    run_step3_for_profile "$profile_name" "$THREADS"
  done
fi

info "Step3 (tune models) finished"

# NOTE: DIA-NN will write outputs like:
#  tune_lib*.dict.txt and tuned model files (older: *.tuned_rt.pt / *.tuned_im.pt / *.tuned_fr.pt,
#  DIA-NN 2.5.0 distillation: *.rt.d0.pt / *.rt.d1.pt / *.rt.d2.pt, analogous for im/fr).
# The exact location/naming is derived from the tune-lib name. Check alongside ${TUNELIB} and in the log.
