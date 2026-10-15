#!/usr/bin/env bash
# Run a restricted-layer variant from an already completed experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash run_restricted_layers_variant.sh \
    --expname <existing_experiment> [--profile balanced] \
    [--list-source-runs | --source-tuned-run <relative_tuned_output>] \
    [--tune-lr <same_lr_or_sweep>] [--variant-name restricted_layers] \
    [--set KEY=VALUE ...] [--resume] [--auto-resume] [--verbose <0-4>] [--test]

FASTA, test files, selected tuning LR, verbosity, and saved --set overrides
are read from the original tuned Step-5 start command. Explicit --fastaabs,
--run-list, --f, --tune-lr, --set, and --verbose values override them.

--resume skips verified completed restricted artifacts. Incomplete restricted
outputs are moved to a timestamped .interrupted directory before rerunning;
nothing is deleted. --auto-resume retries failed runs with --resume.
EOF
}

ORIGINAL_ARGS=("$@")
EXPNAME="" FASTA="" RUN_LIST="" PROFILE="balanced" TUNE_LR=""
VARIANT="restricted_layers" VERBOSE="" TEST_MODE=0
SOURCE_TUNED_RUN="" LIST_SOURCE_RUNS=0
OVERRIDES=()
RAW_FILES=()
NO_CAL_LIB=0
RESUME_MODE=0 AUTO_RESUME=0 AUTO_RESUME_SLEEP="${AUTO_RESUME_SLEEP:-60}"
AUTO_RESUME_MAX_ATTEMPTS="${AUTO_RESUME_MAX_ATTEMPTS:-0}"
AUTO_RESUME_LOG_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expname) EXPNAME="${2:?--expname needs a value}"; shift 2 ;;
    --fastaabs) FASTA="${2:?--fastaabs needs a value}"; shift 2 ;;
    --run-list) RUN_LIST="${2:?--run-list needs a value}"; shift 2 ;;
    --f) RAW_FILES+=("${2:?--f needs a value}"); shift 2 ;;
    --profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
    --source-tuned-run) SOURCE_TUNED_RUN="${2:?--source-tuned-run needs a value}"; shift 2 ;;
    --list-source-runs) LIST_SOURCE_RUNS=1; shift ;;
    --tune-lr) TUNE_LR="${2:?--tune-lr needs a value}"; shift 2 ;;
    --variant-name) VARIANT="${2:?--variant-name needs a value}"; shift 2 ;;
    --set) OVERRIDES+=("${2:?--set needs KEY=VALUE}"); shift 2 ;;
    --no-cal-lib) NO_CAL_LIB=1; shift ;;
    --resume) RESUME_MODE=1; shift ;;
    --auto-resume|--autoresume) AUTO_RESUME=1; shift ;;
    --auto-resume-sleep) AUTO_RESUME_SLEEP="${2:?--auto-resume-sleep needs a value}"; shift 2 ;;
    --auto-resume-sleep=*) AUTO_RESUME_SLEEP="${1#*=}"; shift ;;
    --auto-resume-max-attempts) AUTO_RESUME_MAX_ATTEMPTS="${2:?--auto-resume-max-attempts needs a value}"; shift 2 ;;
    --auto-resume-max-attempts=*) AUTO_RESUME_MAX_ATTEMPTS="${1#*=}"; shift ;;
    --auto-resume-log-dir) AUTO_RESUME_LOG_DIR="${2:?--auto-resume-log-dir needs a value}"; shift 2 ;;
    --auto-resume-log-dir=*) AUTO_RESUME_LOG_DIR="${1#*=}"; shift ;;
    --verbose) VERBOSE="${2:?--verbose needs a value}"; shift 2 ;;
    --test) TEST_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$EXPNAME" ]] || { usage >&2; exit 2; }
[[ "$PROFILE" != "all" && "$PROFILE" != *,* ]] || { echo "Run one profile per invocation (e.g. --profile balanced)." >&2; exit 2; }
[[ "$VARIANT" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--variant-name may contain only letters, digits, ., _, -" >&2; exit 2; }

REQUEST_NO_CAL_LIB="$NO_CAL_LIB"
USER_OVERRIDES=("${OVERRIDES[@]}")
source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
NO_CAL_LIB="$REQUEST_NO_CAL_LIB"
# Do not assume the currently configured DIA-NN version. Existing experiments
# can live below any version directory (for example 2_5_0) or an additional
# output root. Locate the actual finished experiment first.
search_roots=("${OUTROOT%/}")
if [[ -n "${DIANN_ADDITIONAL_OUTROOTS:-}" ]]; then
  IFS=':' read -r -a extra_roots <<< "$DIANN_ADDITIONAL_OUTROOTS"
  search_roots+=("${extra_roots[@]}")
fi
experiment_matches=()
for root in "${search_roots[@]}"; do
  [[ -d "$root" ]] || continue
  while IFS= read -r candidate; do
    [[ -d "$candidate/02_tunelib" && -d "$candidate/05_quant" ]] && experiment_matches+=("$candidate")
  done < <(find "$root" -mindepth 2 -maxdepth 2 -type d -name "$EXPNAME" -print 2>/dev/null)
done
if (( ${#experiment_matches[@]} == 0 )); then
  echo "Original experiment was not found below configured DIA-NN output roots: ${EXPNAME}" >&2
  exit 2
fi
if (( ${#experiment_matches[@]} > 1 )); then
  echo "Original experiment name is ambiguous; found:" >&2
  printf '  %s\n' "${experiment_matches[@]}" >&2
  exit 2
fi
EXPDIR="${experiment_matches[0]}"
ORIGINAL_OUTBASE="$(dirname "$EXPDIR")"
echo "Using original experiment directory: ${EXPDIR}"

# Prefer the original tuned search command over the baseline command: it is the
# durable record of the exact FASTA, test files, selected LR, and overrides.
mapfile -t source_startcmds < <(find "$EXPDIR/05_quant/tuned" -type f -name startcmd.txt -print 2>/dev/null | sort)
(( ${#source_startcmds[@]} > 0 )) || { echo "No original tuned Step-5 start command found below $EXPDIR/05_quant/tuned" >&2; exit 2; }
if [[ "$LIST_SOURCE_RUNS" == 1 ]]; then
  echo "Available original tuned runs for ${EXPNAME}:"
  for candidate in "${source_startcmds[@]}"; do
    rel="${candidate#${EXPDIR}/05_quant/tuned/}"
    echo "  ${rel%/startcmd.txt}"
  done
  exit 0
fi
if [[ -n "$SOURCE_TUNED_RUN" ]]; then
  [[ "$SOURCE_TUNED_RUN" != /* && "$SOURCE_TUNED_RUN" != *".."* ]] || { echo "--source-tuned-run must be a relative tuned-output path" >&2; exit 2; }
  source_startcmd="${EXPDIR}/05_quant/tuned/${SOURCE_TUNED_RUN%/}/startcmd.txt"
  [[ -f "$source_startcmd" ]] || { echo "Unknown tuned run '${SOURCE_TUNED_RUN}'. Use --list-source-runs." >&2; exit 2; }
elif (( ${#source_startcmds[@]} == 1 )); then
  source_startcmd="${source_startcmds[0]}"
else
  echo "Multiple original tuned runs exist. Choose one with --source-tuned-run:" >&2
  for candidate in "${source_startcmds[@]}"; do
    rel="${candidate#${EXPDIR}/05_quant/tuned/}"
    echo "  ${rel%/startcmd.txt}" >&2
  done
  exit 2
fi
echo "Using original tuned run: ${source_startcmd#${EXPDIR}/05_quant/tuned/}"
SOURCE_RUN_LABEL="${source_startcmd#${EXPDIR}/05_quant/tuned/}"
SOURCE_RUN_LABEL="${SOURCE_RUN_LABEL%/startcmd.txt}"
mapfile -t saved_fields < <(python3 - "$source_startcmd" <<'PY'
import pathlib, re, shlex, sys
path = pathlib.Path(sys.argv[1])
command = next((line.strip() for line in path.read_text().splitlines()
                if line.strip() and not line.startswith('#')), None)
if not command:
    raise SystemExit('No invocation found')
args = shlex.split(command)
for i, arg in enumerate(args):
    if arg == '--fastaabs' and i + 1 < len(args): print('FASTA=' + args[i + 1])
    elif arg == '--f' and i + 1 < len(args): print('RAW=' + args[i + 1])
    elif arg == '--run-list' and i + 1 < len(args): print('RUN_LIST=' + args[i + 1])
    elif arg == '--verbose' and i + 1 < len(args): print('VERBOSE=' + args[i + 1])
    elif arg == '--set' and i + 1 < len(args): print('SET=' + args[i + 1])
lib = next((args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '--lib'), '')
if lib: print('SOURCE_LIB=' + lib)
match = re.search(r'/lr_([^/]+)/', lib)
if match: print('SELECTED_LR=' + match.group(1).replace('p', '.'))
for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    try:
        executable = shlex.split(line)[0]
    except ValueError:
        continue
    if pathlib.Path(executable).name.startswith('diann-'):
        print('DIANN_BIN=' + executable)
        break
PY
)
SAVED_RAW_FILES=()
SAVED_OVERRIDES=()
SAVED_FASTA="" SAVED_RUN_LIST="" SAVED_VERBOSE="" SAVED_SELECTED_LR="" SAVED_DIANN_BIN="" SAVED_SOURCE_LIB=""
for field in "${saved_fields[@]}"; do
  case "$field" in
    FASTA=*) SAVED_FASTA="${field#FASTA=}" ;;
    RAW=*) SAVED_RAW_FILES+=("${field#RAW=}") ;;
    RUN_LIST=*) SAVED_RUN_LIST="${field#RUN_LIST=}" ;;
    VERBOSE=*) SAVED_VERBOSE="${field#VERBOSE=}" ;;
    SET=*) SAVED_OVERRIDES+=("${field#SET=}") ;;
    SELECTED_LR=*) SAVED_SELECTED_LR="${field#SELECTED_LR=}" ;;
    DIANN_BIN=*) SAVED_DIANN_BIN="${field#DIANN_BIN=}" ;;
    SOURCE_LIB=*) SAVED_SOURCE_LIB="${field#SOURCE_LIB=}" ;;
  esac
done
# Step 4 can contain run-level context (notably INSTRUMENT) that Step 5 does
# not repeat. Recover its saved overrides from the library consumed by Step 5.
if [[ -n "$SAVED_SOURCE_LIB" ]]; then
  source_step4_startcmd="$(find "$(dirname "$SAVED_SOURCE_LIB")" -maxdepth 1 -type f -name 'startcmd*.txt' -print 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "$source_step4_startcmd" ]]; then
    mapfile -t step4_saved_overrides < <(python3 - "$source_step4_startcmd" <<'PY'
import pathlib, shlex, sys
path = pathlib.Path(sys.argv[1])
command = next((line.strip() for line in path.read_text().splitlines()
                if line.strip() and not line.startswith('#')), None)
if command:
    args = shlex.split(command)
    for i, arg in enumerate(args[:-1]):
        if arg == '--set': print(args[i + 1])
PY
)
    # Step 4 stores its own output naming overrides. They must never be copied
    # into this variant, otherwise they would overwrite the _restricted_layers
    # directories. INSTRUMENT is the only additional execution context Step 4
    # can contribute beyond the Step-5 invocation.
    for item in "${step4_saved_overrides[@]}"; do
      [[ "$item" == INSTRUMENT=* ]] && SAVED_OVERRIDES=("$item" "${SAVED_OVERRIDES[@]}")
    done
  fi
fi
[[ -n "$FASTA" ]] || FASTA="$SAVED_FASTA"
[[ -n "$RUN_LIST" || ${#RAW_FILES[@]} -gt 0 ]] || { RUN_LIST="$SAVED_RUN_LIST"; RAW_FILES=("${SAVED_RAW_FILES[@]}"); }
[[ -n "$VERBOSE" ]] || VERBOSE="$SAVED_VERBOSE"
[[ -n "$TUNE_LR" ]] || TUNE_LR="$SAVED_SELECTED_LR"
for item in "${SAVED_OVERRIDES[@]}"; do OVERRIDES=("$item" "${OVERRIDES[@]}"); done
user_set_diann_bin=0
for item in "${USER_OVERRIDES[@]}"; do [[ "$item" == DIANN_BIN=* ]] && user_set_diann_bin=1; done
if [[ -n "$SAVED_DIANN_BIN" && "$user_set_diann_bin" == 0 ]]; then
  OVERRIDES=("DIANN_BIN=${SAVED_DIANN_BIN}" "${OVERRIDES[@]}")
fi
user_set_outbase=0
for item in "${USER_OVERRIDES[@]}"; do [[ "$item" == OUTBASE=* ]] && user_set_outbase=1; done
if [[ "$user_set_outbase" == 0 ]]; then
  OVERRIDES=("OUTBASE=${ORIGINAL_OUTBASE}" "${OVERRIDES[@]}")
fi
for item in "${SAVED_OVERRIDES[@]}"; do [[ "$item" == "NO_CAL_LIB=1" ]] && NO_CAL_LIB=1; done
[[ -n "$SAVED_DIANN_BIN" ]] && echo "Using DIA-NN binary from original run: ${SAVED_DIANN_BIN}"

# A recovered command can contain an old value and the user can intentionally
# override it (e.g. THREADS=40). Keep only the last value of every key so the
# preview is unambiguous and every downstream step receives the same value.
declare -A seen_override_keys=()
effective_overrides=()
for ((override_index=${#OVERRIDES[@]} - 1; override_index>=0; override_index--)); do
  override_item="${OVERRIDES[$override_index]}"
  override_key="${override_item%%=*}"
  [[ -n "${seen_override_keys[$override_key]:-}" ]] && continue
  seen_override_keys["$override_key"]=1
  effective_overrides=("$override_item" "${effective_overrides[@]}")
done
OVERRIDES=("${effective_overrides[@]}")
EFFECTIVE_THREADS="${THREADS:-}"
for item in "${OVERRIDES[@]}"; do
  [[ "$item" == THREADS=* ]] && EFFECTIVE_THREADS="${item#THREADS=}"
done

[[ "$FASTA" == /* && -f "$FASTA" ]] || { echo "Could not resolve an existing FASTA from the original run; pass --fastaabs." >&2; exit 2; }
(( ${#RAW_FILES[@]} > 0 )) || [[ -n "$RUN_LIST" ]] || { echo "Could not resolve test files; pass --run-list or --f." >&2; exit 2; }
(( ${#RAW_FILES[@]} == 0 )) || [[ -z "$RUN_LIST" ]] || { echo "Use either --run-list or repeated --f, not both." >&2; exit 2; }
[[ -z "$RUN_LIST" || ( "$RUN_LIST" == /* && -f "$RUN_LIST" ) ]] || { echo "Saved --run-list does not exist; pass a new one." >&2; exit 2; }
for raw in "${RAW_FILES[@]}"; do [[ "$raw" == /* && -f "$raw" ]] || { echo "Saved test file does not exist: $raw" >&2; exit 2; }; done
[[ -n "$TUNE_LR" ]] || { echo "Could not infer the original selected LR; pass --tune-lr." >&2; exit 2; }
[[ -d "${EXPDIR}/02_tunelib/${PROFILE}" ]] || { echo "Missing existing Step-2 input: ${EXPDIR}/02_tunelib/${PROFILE}" >&2; exit 2; }
if [[ "$NO_CAL_LIB" != 1 ]]; then
  [[ -d "${EXPDIR}/cal_lib" ]] || { echo "Missing existing calibration input: ${EXPDIR}/cal_lib" >&2; exit 2; }
fi

STEP2_VARIANT_REL="02_tunelib/${PROFILE}_${VARIANT}"
STEP2_VARIANT_DIR="${EXPDIR}/${STEP2_VARIANT_REL}"
STEP3_DIR="03_tune_models"
STEP3_PROFILE_DIR="${PROFILE}_${VARIANT}"
STEP4_DIR="04_refined_lib"
STEP5_SUB="tuned_${VARIANT}/${PROFILE}"
STEP3_ABS="${EXPDIR}/${STEP3_DIR}/${STEP3_PROFILE_DIR}"
PROGRESS_FILE="${EXPDIR}/progress.txt"

# Progress is strictly append-only. Existing run_all entries are never edited
# or removed; the variant/source fields make these entries independently
# identifiable in the shared experiment history.
append_variant_progress() {
  local status="$1"
  local step_name="$2"
  local detail="$3"
  printf '%s\t%s\t%s\t%s\n' "$(date +'%Y-%m-%dT%H:%M:%S%z')" "$step_name" "$status" "$detail" >> "$PROGRESS_FILE"
}

append_variant_manifest() {
  mkdir -p "$STEP2_VARIANT_DIR"
  printf '%s\tvariant=%s\tsource_run=%s\tsource_step5_startcmd=%s\tsource_lib=%s\tsource_tunelib=%s\n' \
    "$(date +'%Y-%m-%dT%H:%M:%S%z')" "$VARIANT" "$SOURCE_RUN_LABEL" \
    "$source_startcmd" "$SAVED_SOURCE_LIB" "${EXPDIR}/02_tunelib/${PROFILE}/tune_lib_${PROFILE}.parquet" \
    >> "${STEP2_VARIANT_DIR}/source_manifest.txt"
}

is_nonempty_file() {
  [[ -s "$1" ]]
}

model_exists_for_target() {
  local lr_dir="$1" target="$2"
  case "$target" in
    rt) compgen -G "${lr_dir}/*.tuned_rt.pt" >/dev/null || compgen -G "${lr_dir}/*.rt.d*.pt" >/dev/null ;;
    im) compgen -G "${lr_dir}/*.tuned_im.pt" >/dev/null || compgen -G "${lr_dir}/*.im.d*.pt" >/dev/null ;;
    fr) compgen -G "${lr_dir}/*.tuned_fr.pt" >/dev/null || compgen -G "${lr_dir}/*.fr.d*.pt" >/dev/null ;;
    *) return 1 ;;
  esac
}

expected_tune_targets() {
  local targets="rt,fr" item value instrument="" tune_rt="" tune_im="" tune_fr=""
  for item in "${OVERRIDES[@]}"; do
    case "$item" in
      TUNE_TARGETS=*) targets="${item#TUNE_TARGETS=}" ;;
      INSTRUMENT=*) instrument="${item#INSTRUMENT=}" ;;
      TUNE_RT=*) tune_rt="${item#TUNE_RT=}" ;;
      TUNE_IM=*) tune_im="${item#TUNE_IM=}" ;;
      TUNE_FR=*) tune_fr="${item#TUNE_FR=}" ;;
    esac
  done
  targets="${targets,,}"
  targets="${targets//[[:space:]]/}"
  [[ "$targets" == "all" ]] && targets="rt,im,fr"
  IFS=',' read -r -a target_values <<< "$targets"
  for value in "${target_values[@]}"; do
    case "$value" in
      rt) [[ -n "$tune_rt" && "$tune_rt" != "1" ]] && continue ;;
      im) [[ -n "$tune_im" && "$tune_im" != "1" ]] && continue ;;
      fr) [[ -n "$tune_fr" && "$tune_fr" != "1" ]] && continue ;;
    esac
    [[ "$value" == "im" && "${instrument,,}" != "timstof" ]] && continue
    printf '%s\n' "$value"
  done
}

step3_lr_complete() {
  local lr_dir="$1" target
  compgen -G "${lr_dir}/*.dict.txt" >/dev/null || return 1
  compgen -G "${lr_dir}/tune_report_*.parquet" >/dev/null || return 1
  while IFS= read -r target; do
    model_exists_for_target "$lr_dir" "$target" || return 1
  done < <(expected_tune_targets)
}

backup_incomplete_dir() {
  local dir="$1" stage="$2" backup_root backup_name
  [[ -e "$dir" ]] || return 0
  backup_root="$(dirname "$dir")/.interrupted"
  backup_name="$(basename "$dir")_${stage}_$(date +'%Y%m%d_%H%M%S')"
  mkdir -p "$backup_root"
  mv "$dir" "${backup_root}/${backup_name}"
  append_variant_progress BACKUP "$stage" "variant=${VARIANT};moved=${dir};backup=${backup_root}/${backup_name};reason=incomplete_resume"
  echo "Resume: preserved incomplete ${stage} output at ${backup_root}/${backup_name}"
}

run_variant_stage() {
  local step_name="$1" detail="$2" started rc
  shift 2
  started="$(date +%s)"
  append_variant_progress START "$step_name" "$detail"
  if "$@"; then
    append_variant_progress OK "$step_name" "${detail};elapsed=$(( $(date +%s) - started ))s"
    return 0
  fi
  rc=$?
  append_variant_progress FAILED "$step_name" "${detail};elapsed=$(( $(date +%s) - started ))s;exit=${rc}"
  return "$rc"
}

auto_resume_has_arg() {
  local needle="$1" arg
  shift
  for arg in "$@"; do [[ "$arg" == "$needle" ]] && return 0; done
  return 1
}

maybe_run_auto_resume_supervisor() {
  local attempt=1 rc=0 ts log_file
  local child_args=() arg skip_next=0
  (( AUTO_RESUME == 1 )) || return 0
  [[ "$AUTO_RESUME_SLEEP" =~ ^[0-9]+$ ]] || { echo "--auto-resume-sleep must be a non-negative integer" >&2; exit 2; }
  [[ "$AUTO_RESUME_MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || { echo "--auto-resume-max-attempts must be a non-negative integer" >&2; exit 2; }
  [[ -n "$AUTO_RESUME_LOG_DIR" ]] || AUTO_RESUME_LOG_DIR="${EXPDIR}/auto_resume_logs/${VARIANT}"
  mkdir -p "$AUTO_RESUME_LOG_DIR"
  for ((i=0; i<${#ORIGINAL_ARGS[@]}; i++)); do
    arg="${ORIGINAL_ARGS[$i]}"
    case "$arg" in
      --auto-resume|--autoresume) ;;
      --auto-resume-sleep|--auto-resume-max-attempts|--auto-resume-log-dir) ((i+=1)) ;;
      --auto-resume-sleep=*|--auto-resume-max-attempts=*|--auto-resume-log-dir=*) ;;
      *) child_args+=("$arg") ;;
    esac
  done
  while true; do
    ts="$(date +'%Y%m%d_%H%M%S')"
    log_file="${AUTO_RESUME_LOG_DIR%/}/restricted_auto_resume_attempt_${attempt}_${ts}.log"
    child_cmd=(bash "${SCRIPT_DIR}/run_restricted_layers_variant.sh" "${child_args[@]}")
    if (( attempt > 1 )) && ! auto_resume_has_arg "--resume" "${child_args[@]}"; then child_cmd+=(--resume); fi
    { printf '[%s] restricted auto-resume attempt=%s\ncommand=' "$(date -Is)" "$attempt"; printf '%q ' "${child_cmd[@]}"; echo; } | tee "$log_file"
    set +e
    "${child_cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    (( rc == 0 )) && { echo "Restricted auto-resume completed successfully on attempt ${attempt}."; exit 0; }
    echo "Restricted auto-resume attempt ${attempt} failed (exit ${rc}). Log: ${log_file}" >&2
    (( AUTO_RESUME_MAX_ATTEMPTS > 0 && attempt >= AUTO_RESUME_MAX_ATTEMPTS )) && exit "$rc"
    attempt=$((attempt + 1))
    echo "Sleeping ${AUTO_RESUME_SLEEP}s before retry with --resume..."
    sleep "$AUTO_RESUME_SLEEP"
  done
}

sets_step3=()
for item in "${OVERRIDES[@]}"; do sets_step3+=(--set "$item"); done
# Keep these identity-defining values last: a recovered source command must
# never redirect this variant into the ordinary Step-3 output or disable the
# restricted-layer DIA-NN flag.
sets_step3+=(--set "STEP3_BASEDIR_NAME=${STEP3_DIR}" --set "STEP3_PROFILE_DIR_SUFFIX=${VARIANT}" --set "STEP3_NAME_SUFFIX=${VARIANT}" --set "TUNE_RESTRICT_LAYERS=1")
[[ -n "$TUNE_LR" ]] && sets_step3+=(--set "TUNE_LR=${TUNE_LR}")
verbose_args=()
[[ -n "$VERBOSE" ]] && verbose_args=(--verbose "$VERBOSE")

step3=(bash "${PIPELINE_ROOT}/bin/step3_tune_models.sh" --expname "$EXPNAME" --fastaabs "$FASTA" --profile "$PROFILE" "${sets_step3[@]}" "${verbose_args[@]}")
[[ "$NO_CAL_LIB" == 1 ]] && step3+=(--no-cal-lib)
if [[ "$TEST_MODE" == 1 ]]; then
  [[ "$RESUME_MODE" == 1 ]] && echo "RESUME TEST: completed restricted artifacts would be kept; incomplete outputs would be moved to .interrupted/."
  [[ "$AUTO_RESUME" == 1 ]] && echo "AUTO-RESUME TEST: retry supervisor is not started in --test mode."
  step3+=(--test)
  printf 'Step 3 preview: '; printf '%q ' "${step3[@]}"; echo
  "${step3[@]}"
  tune_lr_effective="$TUNE_LR"
  if [[ -z "$tune_lr_effective" ]]; then
    tune_lr_effective="$(bash -c 'source "$1"; printf "%s" "$TUNE_LR"' _ "${PIPELINE_ROOT}/config/step3.cfg")"
  fi
  mapfile -t test_lrs < <(python3 - "$tune_lr_effective" <<'PY'
import sys
spec = sys.argv[1].strip()
if ',' in spec:
    values = [float(x.strip()) for x in spec.split(',') if x.strip()]
elif ':' in spec:
    start, stop, step = map(float, spec.split(':'))
    values, value = [], start
    eps = abs(step) * 1e-9
    while (step > 0 and value <= stop + eps) or (step < 0 and value >= stop - eps):
        if value > 0:
            values.append(value)
        value += step
else:
    values = [float(spec)]
for value in values:
    if value > 0:
        print(f"{value:.12g}")
PY
)
  echo "Step 4/5 preview (model files are intentionally not required in --test):"
  for lr in "${test_lrs[@]}"; do
    lr_tag="${lr//./p}"
    lr_dir="${STEP3_ABS}/lr_${lr_tag}"
    step4_sub="${PROFILE}_${VARIANT}/lr_${lr_tag}"
    suffix="${PROFILE}_${VARIANT}_lr_${lr_tag}"
    preview_step4=(bash "${PIPELINE_ROOT}/bin/step4_create_tuned_predlib.sh" --expname "$EXPNAME" --fastaabs "$FASTA" --tokens "${lr_dir}/tune_lib_${PROFILE}_${VARIANT}_lr_${lr_tag}.dict.txt" --rt-model "${lr_dir}/tune_lib_${PROFILE}_${VARIANT}_lr_${lr_tag}.rt.d0.pt" --fr-model "${lr_dir}/tune_lib_${PROFILE}_${VARIANT}_lr_${lr_tag}.fr.d0.pt")
    for item in "${OVERRIDES[@]}"; do preview_step4+=(--set "$item"); done
    preview_step4+=(--set "STEP4_BASEDIR_NAME=${STEP4_DIR}" --set "STEP4_SUBDIR=${step4_sub}" --set "STEP4_NAME_SUFFIX=${suffix}")
    preview_step4+=("${verbose_args[@]}")
    printf 'Step 4 preview: '; printf '%q ' "${preview_step4[@]}"; echo
    # DIA-NN appends ".predicted" to the --out-lib stem. The original Step-5
    # start command therefore consumes the double-suffixed file.
    lib="${EXPDIR}/${STEP4_DIR}/${step4_sub}/refined_lib_${suffix}.predicted.predicted.speclib"
    preview_step5=(bash "${PIPELINE_ROOT}/bin/step5_quant.sh" --expname "$EXPNAME" --fastaabs "$FASTA" --lib "$lib" --out-sub "${STEP5_SUB}/lr_${lr_tag}")
    if [[ -n "$RUN_LIST" ]]; then
      preview_step5+=(--run-list "$RUN_LIST")
    else
      for raw in "${RAW_FILES[@]}"; do preview_step5+=(--f "$raw"); done
    fi
    for item in "${OVERRIDES[@]}"; do preview_step5+=(--set "$item"); done
    preview_step5+=("${verbose_args[@]}")
    printf 'Step 5 preview: '; printf '%q ' "${preview_step5[@]}"; echo
  done
  echo "OK: full restricted-layer variant plan validated; no files were created."
  exit 0
fi

maybe_run_auto_resume_supervisor
append_variant_manifest

mapfile -t expected_lrs < <(python3 - "$TUNE_LR" <<'PY'
import sys
spec = sys.argv[1].strip()
if ',' in spec:
    values = [float(x.strip()) for x in spec.split(',') if x.strip()]
elif ':' in spec:
    start, stop, step = map(float, spec.split(':'))
    values, value = [], start
    eps = abs(step) * 1e-9
    while (step > 0 and value <= stop + eps) or (step < 0 and value >= stop - eps):
        if value > 0:
            values.append(value)
        value += step
else:
    values = [float(spec)]
for value in values:
    if value > 0:
        print(f"{value:.12g}")
PY
)

step3_complete=1
for lr in "${expected_lrs[@]}"; do
  step3_lr_complete "${STEP3_ABS}/lr_${lr//./p}" || { step3_complete=0; break; }
done
if (( RESUME_MODE == 1 && step3_complete == 1 )); then
  echo "Resume: Step3 is complete; keeping ${STEP3_ABS}"
  append_variant_progress SKIP "step3_tune_models.sh" "variant=${VARIANT};source=${SOURCE_RUN_LABEL};profile=${PROFILE};lr=${TUNE_LR};reason=verified_artifacts"
else
  if (( RESUME_MODE == 1 && step3_complete == 0 )); then
    backup_incomplete_dir "$STEP3_ABS" "step3_tune_models.sh"
  fi
  run_variant_stage "step3_tune_models.sh" "variant=${VARIANT};source=${SOURCE_RUN_LABEL};profile=${PROFILE};lr=${TUNE_LR};threads=${EFFECTIVE_THREADS}" "${step3[@]}"
fi

shopt -s nullglob
lr_dirs=("${STEP3_ABS}"/lr_*)
shopt -u nullglob
(( ${#lr_dirs[@]} > 0 )) || { echo "No restricted-layer LR output created in ${STEP3_ABS}" >&2; exit 1; }

for lr_dir in "${lr_dirs[@]}"; do
  lr_tag="${lr_dir##*/lr_}"
  step4_sub="${PROFILE}_${VARIANT}/lr_${lr_tag}"
  suffix="${PROFILE}_${VARIANT}_lr_${lr_tag}"
  shopt -s nullglob
  token_files=("${lr_dir}"/*.dict.txt)
  rt_models=("${lr_dir}"/*.tuned_rt.pt "${lr_dir}"/*.rt.d0.pt)
  im_models=("${lr_dir}"/*.tuned_im.pt "${lr_dir}"/*.im.d0.pt)
  fr_models=("${lr_dir}"/*.tuned_fr.pt "${lr_dir}"/*.fr.d0.pt)
  shopt -u nullglob
  (( ${#token_files[@]} == 1 )) || { echo "Expected exactly one tokens file in ${lr_dir}" >&2; exit 1; }
  (( ${#rt_models[@]} <= 1 && ${#im_models[@]} <= 1 && ${#fr_models[@]} <= 1 )) || { echo "Ambiguous tuned model files in ${lr_dir}" >&2; exit 1; }
  (( ${#rt_models[@]} + ${#im_models[@]} + ${#fr_models[@]} > 0 )) || { echo "No tuned model files found in ${lr_dir}" >&2; exit 1; }
  step4=(bash "${PIPELINE_ROOT}/bin/step4_create_tuned_predlib.sh" --expname "$EXPNAME" --fastaabs "$FASTA" --tokens "${token_files[0]}")
  (( ${#rt_models[@]} == 1 )) && step4+=(--rt-model "${rt_models[0]}")
  (( ${#im_models[@]} == 1 )) && step4+=(--im-model "${im_models[0]}")
  (( ${#fr_models[@]} == 1 )) && step4+=(--fr-model "${fr_models[0]}")
  for item in "${OVERRIDES[@]}"; do step4+=(--set "$item"); done
  step4+=(--set "STEP4_BASEDIR_NAME=${STEP4_DIR}" --set "STEP4_SUBDIR=${step4_sub}" --set "STEP4_NAME_SUFFIX=${suffix}")
  step4+=("${verbose_args[@]}")
  lib="${EXPDIR}/${STEP4_DIR}/${step4_sub}/refined_lib_${suffix}.predicted.predicted.speclib"
  step4_outdir="${EXPDIR}/${STEP4_DIR}/${step4_sub}"
  if (( RESUME_MODE == 1 )) && is_nonempty_file "$lib"; then
    echo "Resume: Step4 is complete for LR=${lr_tag}; keeping ${step4_outdir}"
    append_variant_progress SKIP "step4_create_tuned_predlib.sh" "variant=${VARIANT};source=${SOURCE_RUN_LABEL};profile=${PROFILE};lr=${lr_tag};outdir=${STEP4_DIR};reason=verified_artifacts"
  else
    if (( RESUME_MODE == 1 )); then backup_incomplete_dir "$step4_outdir" "step4_create_tuned_predlib.sh"; fi
    run_variant_stage "step4_create_tuned_predlib.sh" "variant=${VARIANT};source=${SOURCE_RUN_LABEL};profile=${PROFILE};lr=${lr_tag};models=auto;outdir=${STEP4_DIR}" "${step4[@]}"
  fi
  [[ -f "$lib" ]] || { echo "Expected refined library was not created: $lib" >&2; exit 1; }
  step5=(bash "${PIPELINE_ROOT}/bin/step5_quant.sh" --expname "$EXPNAME" --fastaabs "$FASTA" --lib "$lib" --out-sub "${STEP5_SUB}/lr_${lr_tag}")
  if [[ -n "$RUN_LIST" ]]; then
    step5+=(--run-list "$RUN_LIST")
  else
    for raw in "${RAW_FILES[@]}"; do step5+=(--f "$raw"); done
  fi
  for item in "${OVERRIDES[@]}"; do step5+=(--set "$item"); done
  step5+=("${verbose_args[@]}")
  step5_outdir="${EXPDIR}/05_quant/${STEP5_SUB}/lr_${lr_tag}"
  step5_report="${step5_outdir}/final_report.parquet"
  if (( RESUME_MODE == 1 )) && is_nonempty_file "$step5_report"; then
    echo "Resume: Step5 is complete for LR=${lr_tag}; keeping ${step5_outdir}"
    append_variant_progress SKIP "step5_quant.sh" "variant=${VARIANT};source=${SOURCE_RUN_LABEL};profile=${PROFILE};lr=${lr_tag};mode=tuned;outsub=${STEP5_SUB}/lr_${lr_tag};reason=verified_artifacts"
  else
    if (( RESUME_MODE == 1 )); then backup_incomplete_dir "$step5_outdir" "step5_quant.sh"; fi
    run_variant_stage "step5_quant.sh" "variant=${VARIANT};source=${SOURCE_RUN_LABEL};profile=${PROFILE};lr=${lr_tag};mode=tuned;outsub=${STEP5_SUB}/lr_${lr_tag}" "${step5[@]}"
  fi
done

echo "Restricted-layer variant complete: ${EXPDIR}/${STEP2_VARIANT_REL}, ${EXPDIR}/${STEP3_DIR}/${STEP3_PROFILE_DIR}, ${EXPDIR}/${STEP4_DIR}/${PROFILE}_${VARIANT}, ${EXPDIR}/05_quant/${STEP5_SUB}"
