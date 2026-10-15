#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "ERROR: $*" >&2; exit 1; }

supports_color() {
  [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]
}

if supports_color; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
else
  C_RESET=""
  C_BOLD=""
  C_DIM=""
  C_GREEN=""
  C_YELLOW=""
  C_BLUE=""
fi

print_section() {
  echo
  echo "${C_BOLD}${C_BLUE}== $1 ==${C_RESET}"
}

print_kv() {
  printf '%s%-18s%s %s\n' "$C_DIM" "$1" "$C_RESET" "$2"
}

print_ok() {
  echo "${C_GREEN}✔${C_RESET} $*"
}

print_warn() {
  echo "${C_YELLOW}⚠${C_RESET} $*"
}

print_command() {
  local index=0 current next
  printf '  '
  printf '%q' "${CARAFE_CMD[0]}"
  index=1

  while (( index < ${#CARAFE_CMD[@]} )); do
    current="${CARAFE_CMD[$index]}"
    next="${CARAFE_CMD[$((index + 1))]:-}"
    printf ' \\\n    %q' "$current"
    if [[ "$current" == -* && -n "$next" && "$next" != -* ]]; then
      printf ' %q' "$next"
      ((index += 1))
    fi
    ((index += 1))
  done
  echo
}

print_modification_mapping() {
  print_kv "DIA-NN preset" "$SOURCE_DIANN_MOD_CONFIG"
  print_kv "Fixed: C" "UniMod:4 Carbamidomethyl (+57.021464) -> Carafe #1"
  print_kv "Fixed: K" "UniMod:510 heavy dimethyl (+34.063117) -> Carafe #479"
  print_kv "Variable: M" "UniMod:35 Oxidation (+15.994915) -> Carafe #2"
  print_kv "Variable: n-term" "UniMod:510 heavy dimethyl (+34.063117) -> Carafe #480"
  print_kv "Variable: n-term" "UniMod:1 Acetyl (+42.010565) -> Carafe #31"
  print_kv "Variable limit" "DIA-NN 2 -> Carafe -maxVar ${CARAFE_MAX_VARIABLE_MODS}"

  if [[ "$SOURCE_DIANN_MOD_CONFIG" == "fixed_lysin_dimethyl" \
        && "$CARAFE_FIXED_MODS" == "1,479" \
        && "$CARAFE_VARIABLE_MODS" == "2,480,31" \
        && "$CARAFE_MAX_VARIABLE_MODS" == "2" ]]; then
    print_ok "DIA-NN -> Carafe modification mapping verified for fixed_lysin_dimethyl."
  else
    print_warn "Custom or mismatching modification settings: the fixed_lysin_dimethyl mapping is not verified for this job."
  fi
}

JOB_DIR=""
MACHINE_CONFIG="${CARAFE_MACHINE_CONFIG:-}"
TEST_MODE=0
RESUME_REQUESTED=0
SHOW_COMMAND=0
FINE_TUNE_ONLY=0
FORCE_FINE_TUNE=0
SAVE_VALIDATION_CHECKPOINTS=0
MS2_EPOCHS=""
RT_EPOCHS=""
FINE_TUNE_RUN=""
GPU_REQUEST=""
EXTERNAL_MODEL_DIR=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --job) ((i += 1)); JOB_DIR="${ARGS[$i]:-}" ;;
    --machine-config) ((i += 1)); MACHINE_CONFIG="${ARGS[$i]:-}" ;;
    --machine) ((i += 1)); MACHINE_CONFIG="$SCRIPT_DIR/config/carafe_machines/${ARGS[$i]:-}.cfg" ;;
    --gpu) ((i += 1)); GPU_REQUEST="${ARGS[$i]:-}" ;;
    --test) TEST_MODE=1 ;;
    --resume) RESUME_REQUESTED=1 ;;
    --fine-tune-only) FINE_TUNE_ONLY=1 ;;
    --force-fine-tune) FORCE_FINE_TUNE=1 ;;
    --validation-checkpoints) SAVE_VALIDATION_CHECKPOINTS=1 ;;
    --ms2-epochs) ((i += 1)); MS2_EPOCHS="${ARGS[$i]:-}" ;;
    --rt-epochs) ((i += 1)); RT_EPOCHS="${ARGS[$i]:-}" ;;
    --fine-tune-run) ((i += 1)); FINE_TUNE_RUN="${ARGS[$i]:-}" ;;
    --external-model-dir) ((i += 1)); EXTERNAL_MODEL_DIR="${ARGS[$i]:-}" ;;
    --show-command) SHOW_COMMAND=1 ;;
    -h|--help) sed -n '1,220p' "$0"; exit 0 ;;
    *) die "Unknown argument: ${ARGS[$i]}" ;;
  esac
done

[[ -n "$JOB_DIR" ]] || die "--job is required"
JOB_DIR="$(realpath -e "$JOB_DIR")" || die "Job does not exist: $JOB_DIR"
[[ -f "$JOB_DIR/manifest.yaml" ]] || die "Not a prepared Carafe job: $JOB_DIR"
[[ -n "$MACHINE_CONFIG" ]] || die "Provide --machine-config or --machine"
[[ -f "$MACHINE_CONFIG" ]] || die "Carafe machine config not found: $MACHINE_CONFIG"
[[ "$FINE_TUNE_ONLY" == "0" || "$RESUME_REQUESTED" == "0" ]] || die "--fine-tune-only cannot be combined with --resume"
[[ "$FORCE_FINE_TUNE" == "0" || "$FINE_TUNE_ONLY" == "1" ]] || die "--force-fine-tune requires --fine-tune-only"
[[ -z "$MS2_EPOCHS" || "$MS2_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--ms2-epochs must be a positive integer"
[[ -z "$RT_EPOCHS" || "$RT_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--rt-epochs must be a positive integer"
[[ -z "$FINE_TUNE_RUN" || "$FINE_TUNE_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--fine-tune-run must contain only letters, digits, dot, underscore, or hyphen"
[[ -z "$FINE_TUNE_RUN" || "$FINE_TUNE_ONLY" == "1" ]] || die "--fine-tune-run requires --fine-tune-only"
if [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
  [[ "$FINE_TUNE_ONLY" == "0" ]] || die "--external-model-dir cannot be combined with --fine-tune-only"
  [[ "$FORCE_FINE_TUNE" == "0" ]] || die "--external-model-dir cannot be combined with --force-fine-tune"
  [[ "$RESUME_REQUESTED" == "0" ]] || die "--external-model-dir cannot be combined with --resume"
  [[ -z "$MS2_EPOCHS" && -z "$RT_EPOCHS" && -z "$FINE_TUNE_RUN" ]] \
    || die "External-model inference cannot use fine-tuning epoch/run options"
  EXTERNAL_MODEL_DIR="$(realpath -e "$EXTERNAL_MODEL_DIR")" \
    || die "External model directory does not exist: $EXTERNAL_MODEL_DIR"
  [[ -d "$EXTERNAL_MODEL_DIR" ]] || die "External model path is not a directory: $EXTERNAL_MODEL_DIR"
  [[ -s "$EXTERNAL_MODEL_DIR/rt_model.pt" ]] \
    || die "External Carafe RT checkpoint is missing: $EXTERNAL_MODEL_DIR/rt_model.pt"
fi
if [[ -n "$GPU_REQUEST" ]]; then
  [[ "$GPU_REQUEST" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpu expects one GPU index or a comma-separated list, got '$GPU_REQUEST'"
  [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || die "Use either --gpu or CUDA_VISIBLE_DEVICES, not both"
  export CUDA_VISIBLE_DEVICES="$GPU_REQUEST"
fi

set -a
# shellcheck source=/dev/null
source "$JOB_DIR/meta/carafe_options.env"
# shellcheck source=/dev/null
source "$MACHINE_CONFIG"
set +a
CARAFE_ASSEMBLY_WORKERS="${CARAFE_ASSEMBLY_WORKERS:-2}"
CARAFE_DIANN_REFERENCE_LIBRARY="${CARAFE_DIANN_REFERENCE_LIBRARY:-./shared_data/250_speclib_save/lib.parquet}"
CARAFE_DIANN2_PARQUET="${CARAFE_DIANN2_PARQUET:-1}"
CARAFE_THERMORAW_PARSER="${CARAFE_THERMORAW_PARSER:-thermorawfileparser}"
# Settings validated in thermo_raw_file_parser_test/commands.txt:
# -f=2 writes indexed mzML, -z disables zlib, and omitting -p keeps Thermo's
# native peak picking enabled.
CARAFE_THERMORAW_FORMAT="${CARAFE_THERMORAW_FORMAT:-2}"
CARAFE_THERMORAW_NO_ZLIB="${CARAFE_THERMORAW_NO_ZLIB:-1}"
CARAFE_THERMORAW_PEAK_PICKING="${CARAFE_THERMORAW_PEAK_PICKING:-1}"

case "${CARAFE_LIBRARY_FORMAT,,}" in
  parquet|tsv) ;;
  *) die "CARAFE_LIBRARY_FORMAT must be 'tsv' or 'parquet', got: $CARAFE_LIBRARY_FORMAT" ;;
esac

# CARAFE's parallel assembler always writes Parquet chunks and merges them
# into a Parquet library.  TSV must therefore use its serial writer; otherwise
# -lf_format tsv is silently ignored for jobs with multiple assembly workers.
if [[ "${CARAFE_LIBRARY_FORMAT,,}" == "tsv" && "$CARAFE_ASSEMBLY_WORKERS" != "1" ]]; then
  print_warn "TSV library export uses CARAFE_ASSEMBLY_WORKERS=1; CARAFE's parallel assembler only produces Parquet."
  CARAFE_ASSEMBLY_WORKERS=1
fi

# Jobs prepared before semi-specific Carafe support do not yet contain this
# variable. The workflow-wide default is nevertheless semi-specific.
CARAFE_SEMI_SPECIFIC="${CARAFE_SEMI_SPECIFIC:-1}"
SOURCE_DIANN_MOD_CONFIG="$(awk -F"'" '/^  diann_mod_config:/ {print $2; exit}' "$JOB_DIR/manifest.yaml")"
SOURCE_DIANN_MOD_CONFIG="${SOURCE_DIANN_MOD_CONFIG:-<not recorded>}"

DIGESTION_CONTRACT="$JOB_DIR/meta/digestion_contract.env"
[[ -f "$DIGESTION_CONTRACT" ]] || die "Digestion contract is missing: $DIGESTION_CONTRACT. Re-prepare this job after updating the Carafe workflow."
set -a
# shellcheck source=/dev/null
source "$DIGESTION_CONTRACT"
set +a
[[ "$DIANN_SEMI" == "$CARAFE_SEMI_SPECIFIC" ]] || die "Semi-specificity mismatch in $DIGESTION_CONTRACT"
[[ "$DIANN_MISSED_CLEAVAGES" == "$CARAFE_MISSED_CLEAVAGES" ]] || die "Missed-cleavage mismatch in $DIGESTION_CONTRACT"
[[ "$DIANN_MIN_PEP_LEN" == "$CARAFE_MIN_PEPTIDE_LENGTH" && "$DIANN_MAX_PEP_LEN" == "$CARAFE_MAX_PEPTIDE_LENGTH" ]] || die "Peptide-length mismatch in $DIGESTION_CONTRACT"
[[ "$DIANN_MIN_PR_MZ" == "$CARAFE_MIN_PRECURSOR_MZ" && "$DIANN_MAX_PR_MZ" == "$CARAFE_MAX_PRECURSOR_MZ" ]] || die "Precursor m/z mismatch in $DIGESTION_CONTRACT"
case "$CARAFE_ENZYME" in
  4) [[ "$DIANN_CUT" == 'R*' ]] || die "DIA-NN CUT must be R* for Carafe enzyme 4, got $DIANN_CUT" ;;
  7) [[ "$DIANN_CUT" == 'K*' ]] || die "DIA-NN CUT must be K* for Carafe enzyme 7, got $DIANN_CUT" ;;
  *) die "No DIA-NN cut-rule verification is defined for Carafe enzyme $CARAFE_ENZYME" ;;
esac

CARAFE_JAVA="${CARAFE_JAVA_OVERRIDE:-${CARAFE_JAVA:-}}"
CARAFE_JAR="${CARAFE_JAR_OVERRIDE:-${CARAFE_JAR:-}}"
CARAFE_JAVA_OPTS="${CARAFE_JAVA_OPTS_OVERRIDE:-${CARAFE_JAVA_OPTS:--Xms16G -Xmx96G}}"
: "${CARAFE_JAVA:?CARAFE_JAVA is missing in $MACHINE_CONFIG}"
: "${CARAFE_JAR:?CARAFE_JAR is missing in $MACHINE_CONFIG}"
[[ -x "$CARAFE_JAVA" ]] || die "Java executable is not executable: $CARAFE_JAVA"
[[ -f "$CARAFE_JAR" ]] || die "Carafe jar is missing: $CARAFE_JAR"
read -r -a CARAFE_JAVA_ARGS <<< "$CARAFE_JAVA_OPTS"
(( ${#CARAFE_JAVA_ARGS[@]} > 0 )) || die "CARAFE_JAVA_OPTS must contain at least one Java option"

CARAFE_JAR_SOURCE="$CARAFE_JAR"
CARAFE_RUNTIME_DIR=""
CARAFE_RUNTIME_ROOT="${CARAFE_RUNTIME_ROOT:-/tmp/carafe2-runtime}"

stage_carafe_jar() {
  local source_lib_dir jar_tool
  source_lib_dir="$(dirname "$CARAFE_JAR_SOURCE")/lib"
  [[ -d "$source_lib_dir" ]] || die "Carafe dependency directory is missing: $source_lib_dir"
  mkdir -p "$CARAFE_RUNTIME_ROOT"
  CARAFE_RUNTIME_DIR="$(mktemp -d "$CARAFE_RUNTIME_ROOT/run.XXXXXX")"
  cp -- "$CARAFE_JAR_SOURCE" "$CARAFE_RUNTIME_DIR/carafe-2.2.0.jar"
  ln -s "$source_lib_dir" "$CARAFE_RUNTIME_DIR/lib"

  # Fail before starting a multi-hour job if the source JAR was being rebuilt
  # while it was staged. The staged copy is then immutable for this run.
  jar_tool="$(dirname "$CARAFE_JAVA")/jar"
  [[ -x "$jar_tool" ]] || die "JAR validation tool is not executable: $jar_tool"
  "$jar_tool" tf "$CARAFE_RUNTIME_DIR/carafe-2.2.0.jar" >/dev/null \
    || die "Staged Carafe JAR is invalid; rebuild finished JAR before starting the job"

  CARAFE_JAR="$CARAFE_RUNTIME_DIR/carafe-2.2.0.jar"
}

REPORT="$JOB_DIR/input/diann_step2_report.parquet"
FASTA_FILE="$(<"$JOB_DIR/meta/fasta_source_path.txt")"
TRAIN_DIR="$JOB_DIR/input/train_mzml"
CARAFE_TRAIN_DIR="$TRAIN_DIR"
CONVERTED_TRAIN_DIR="$JOB_DIR/input/train_mzml_converted"
BASE_OUTPUT_DIR="$JOB_DIR/carafe_output"
if [[ -n "$FINE_TUNE_RUN" ]]; then
  OUTPUT_DIR="$BASE_OUTPUT_DIR/fine_tuning_runs/$FINE_TUNE_RUN"
else
  OUTPUT_DIR="$BASE_OUTPUT_DIR"
fi
RAW_LIBRARY_FILE="$OUTPUT_DIR/carafe_spectral_library.parquet"
DIANN2_LIBRARY_FILE="$OUTPUT_DIR/carafe_spectral_library.diann2.parquet"
DIANN2_CONVERTER="$SCRIPT_DIR/bin/convert_carafe_to_diann2_parquet.py"

archive_previous_fine_tuning() {
  local archive_dir timestamp artifact
  timestamp="$(date +%Y%m%dT%H%M%S)"
  archive_dir="$OUTPUT_DIR/previous_finetuning/$timestamp"
  local -a artifacts=(
    ms2_model.pt rt_model.pt ccs_model.pt
    ms2_history.tsv rt_history.tsv ccs_history.tsv
    model_evaluation_metrics.json training_report.html
    training_diagnostics validation_checkpoints
  )

  for artifact in "${artifacts[@]}"; do
    if [[ -e "$OUTPUT_DIR/$artifact" ]]; then
      mkdir -p "$archive_dir"
      mv -- "$OUTPUT_DIR/$artifact" "$archive_dir/"
    fi
  done

  if [[ -d "$archive_dir" ]]; then
    print_warn "Archived previous fine-tuning artifacts to $archive_dir"
  fi
}

if [[ "$FORCE_FINE_TUNE" == "1" && "$TEST_MODE" != "1" ]]; then
  archive_previous_fine_tuning
fi

[[ -f "$REPORT" ]] || die "Step-2 report is missing: $REPORT"
[[ -f "$FASTA_FILE" ]] || die "FASTA is not accessible: $FASTA_FILE"
[[ -d "$TRAIN_DIR" ]] || die "Training acquisition directory is missing: $TRAIN_DIR"
if [[ "$FINE_TUNE_ONLY" != "1" ]]; then
  [[ -f "$CARAFE_DIANN_REFERENCE_LIBRARY" ]] || die "DIA-NN reference library is missing: $CARAFE_DIANN_REFERENCE_LIBRARY"
  [[ -f "$DIANN2_CONVERTER" ]] || die "DIA-NN compatibility converter is missing: $DIANN2_CONVERTER"
fi

TRAINING_FILE_COUNT=0
THERMO_RAW_COUNT=0
MZML_COUNT=0
PENDING_RAW_CONVERSIONS=0
THERMORAW_EXECUTABLE=""
THERMORAW_VERSION=""
declare -a TRAINING_SOURCE_FILES=()
declare -A CARAFE_TRAINING_NAMES=()

mzml_file_looks_valid() {
  local path="$1"
  [[ -s "$path" ]] && grep -a -m 1 -q '<mzML' "$path"
}

raw_cache_metadata() {
  local source="$1"
  printf 'source=%s\nsize=%s\nmtime=%s\nconverter=%s\nconverter_version=%s\nformat=%s\nno_zlib=%s\npeak_picking=%s\n' \
    "$source" "$(stat -c '%s' "$source")" "$(stat -c '%Y' "$source")" \
    "$CARAFE_THERMORAW_PARSER" "$THERMORAW_VERSION" "$CARAFE_THERMORAW_FORMAT" \
    "$CARAFE_THERMORAW_NO_ZLIB" "$CARAFE_THERMORAW_PEAK_PICKING"
}

raw_cache_is_valid() {
  local source="$1" target="$2" metadata="$3" expected
  mzml_file_looks_valid "$target" && [[ -f "$metadata" ]] || return 1
  expected="$(raw_cache_metadata "$source")"
  [[ "$(<"$metadata")" == "$expected" ]]
}

resolve_thermoraw_parser() {
  if [[ "$CARAFE_THERMORAW_PARSER" == */* ]]; then
    [[ -x "$CARAFE_THERMORAW_PARSER" ]] || return 1
    realpath -e "$CARAFE_THERMORAW_PARSER"
  else
    command -v "$CARAFE_THERMORAW_PARSER"
  fi
}

validate_thermoraw_parser() {
  local help_text version_text
  # ThermoRawFileParser 2.0 prints valid help but exits with 255 for --help.
  # Validate its identifying options and require a successful version query.
  help_text="$("$THERMORAW_EXECUTABLE" --help 2>&1 || true)"
  help_text="${help_text,,}"
  [[ "$help_text" == *thermorawfileparser* && "$help_text" == *nopeakpicking* ]] || return 1
  version_text="$("$THERMORAW_EXECUTABLE" --version 2>&1)" || return 1
  [[ -n "$version_text" ]]
}

case "$CARAFE_THERMORAW_FORMAT" in
  2|mzml-indexed|indexed-mzml) ;;
  *) die "CARAFE_THERMORAW_FORMAT must select indexed mzML (2), got: $CARAFE_THERMORAW_FORMAT" ;;
esac
[[ "$CARAFE_THERMORAW_NO_ZLIB" =~ ^[01]$ ]] || die "CARAFE_THERMORAW_NO_ZLIB must be 0 or 1"
[[ "$CARAFE_THERMORAW_PEAK_PICKING" =~ ^[01]$ ]] || die "CARAFE_THERMORAW_PEAK_PICKING must be 0 or 1"

while IFS= read -r raw; do
  [[ -n "$raw" ]] || continue
  [[ -f "$raw" ]] || die "Training acquisition file is not accessible: $raw"
  [[ -L "$TRAIN_DIR/$(basename "$raw")" ]] || die "Training input link is missing: $TRAIN_DIR/$(basename "$raw")"
  TRAINING_SOURCE_FILES+=("$raw")
  ((TRAINING_FILE_COUNT += 1))
  case "${raw,,}" in
    *.raw)
      ((THERMO_RAW_COUNT += 1))
      stem="$(basename "$raw")"
      stem="${stem%.*}"
      converted="$CONVERTED_TRAIN_DIR/${stem}.mzML"
      conversion_metadata="${converted}.source"
      effective_name="${stem,,}.mzml"
      [[ -z "${CARAFE_TRAINING_NAMES[$effective_name]:-}" ]] || \
        die "Training inputs collide after RAW conversion: $raw and ${CARAFE_TRAINING_NAMES[$effective_name]}"
      CARAFE_TRAINING_NAMES[$effective_name]="$raw"
      ;;
    *.mzml)
      ((MZML_COUNT += 1))
      effective_name="$(basename "${raw,,}")"
      [[ -z "${CARAFE_TRAINING_NAMES[$effective_name]:-}" ]] || \
        die "Training inputs have duplicate Carafe mzML names: $raw and ${CARAFE_TRAINING_NAMES[$effective_name]}"
      CARAFE_TRAINING_NAMES[$effective_name]="$raw"
      ;;
    *)
      die "Unsupported Carafe training format: $raw (supported here: .mzML or Thermo .raw)"
      ;;
  esac
done < "$JOB_DIR/meta/train_mzml_source_paths.txt"
((TRAINING_FILE_COUNT > 0)) || die "No training acquisition paths recorded"

if (( THERMO_RAW_COUNT > 0 )) && [[ -z "$EXTERNAL_MODEL_DIR" ]]; then
  CARAFE_TRAIN_DIR="$CONVERTED_TRAIN_DIR"
  THERMORAW_EXECUTABLE="$(resolve_thermoraw_parser || true)"
  [[ -n "$THERMORAW_EXECUTABLE" ]] || die \
    "Thermo RAW input requires ThermoRawFileParser. Set CARAFE_THERMORAW_PARSER in $MACHINE_CONFIG to the executable in the thermoraw conda environment."
  validate_thermoraw_parser || die \
    "Configured CARAFE_THERMORAW_PARSER is not a working ThermoRawFileParser command: $THERMORAW_EXECUTABLE"
  THERMORAW_VERSION="$("$THERMORAW_EXECUTABLE" --version 2>&1 | head -n 1)"
  for raw in "${TRAINING_SOURCE_FILES[@]}"; do
    [[ "${raw,,}" == *.raw ]] || continue
    stem="$(basename "$raw")"
    stem="${stem%.*}"
    converted="$CONVERTED_TRAIN_DIR/${stem}.mzML"
    conversion_metadata="${converted}.source"
    raw_cache_is_valid "$raw" "$converted" "$conversion_metadata" || \
      ((PENDING_RAW_CONVERSIONS += 1))
  done
fi

prepare_carafe_training_directory() {
  local source name stem target metadata temporary_dir converted_candidate expected_metadata
  local -a conversion_cmd
  mkdir -p "$CARAFE_TRAIN_DIR"

  for source in "${TRAINING_SOURCE_FILES[@]}"; do
    name="$(basename "$source")"
    case "${source,,}" in
      *.mzml)
        target="$CARAFE_TRAIN_DIR/$name"
        if [[ -L "$target" ]]; then
          [[ "$(readlink "$target")" == "$source" ]] || die "Converted training link points elsewhere: $target"
        elif [[ -e "$target" ]]; then
          die "Refusing to replace non-symlink training input: $target"
        else
          ln -s "$source" "$target"
        fi
        ;;
      *.raw)
        stem="${name%.*}"
        target="$CARAFE_TRAIN_DIR/${stem}.mzML"
        metadata="${target}.source"
        if raw_cache_is_valid "$source" "$target" "$metadata"; then
          print_ok "Reusing converted mzML: $(basename "$target")"
          continue
        fi

        temporary_dir="$(mktemp -d "$CARAFE_TRAIN_DIR/.thermoraw.XXXXXX")"
        converted_candidate="$temporary_dir/${stem}.mzML"
        conversion_cmd=(
          "$THERMORAW_EXECUTABLE"
          "-i=$source"
          "-b=$converted_candidate"
          "-f=$CARAFE_THERMORAW_FORMAT"
        )
        [[ "$CARAFE_THERMORAW_NO_ZLIB" == "1" ]] && conversion_cmd+=(-z)
        [[ "$CARAFE_THERMORAW_PEAK_PICKING" == "0" ]] && conversion_cmd+=(-p)
        print_kv "RAW conversion" "$name -> ${stem}.mzML"
        if ! "${conversion_cmd[@]}"; then
          rm -rf -- "$temporary_dir"
          die "ThermoRawFileParser failed for Thermo RAW input: $source"
        fi
        mzml_file_looks_valid "$converted_candidate" || {
          rm -rf -- "$temporary_dir"
          die "ThermoRawFileParser did not create a readable mzML: ${stem}.mzML"
        }
        mv -f -- "$converted_candidate" "$target"
        rmdir "$temporary_dir" 2>/dev/null || true
        expected_metadata="$(raw_cache_metadata "$source")"
        printf '%s\n' "$expected_metadata" > "${metadata}.tmp.$$"
        mv -f -- "${metadata}.tmp.$$" "$metadata"
        ;;
    esac
  done
}

compute_run_fingerprint() {
  {
    sha256sum "$REPORT" "$FASTA_FILE" "$JOB_DIR/meta/carafe_options.env" "$CARAFE_JAR_SOURCE"
    if [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
      printf 'external_model_dir=%s\n' "$EXTERNAL_MODEL_DIR"
      for external_artifact in rt_model.pt ms2_model.pt ccs_model.pt model_evaluation_metrics.json; do
        [[ -f "$EXTERNAL_MODEL_DIR/$external_artifact" ]] \
          && sha256sum "$EXTERNAL_MODEL_DIR/$external_artifact"
      done
    fi
    if (( THERMO_RAW_COUNT > 0 )); then
      printf 'thermoraw_parser=%s\nthermoraw_version=%s\nthermoraw_format=%s\nthermoraw_no_zlib=%s\nthermoraw_peak_picking=%s\n' \
        "$CARAFE_THERMORAW_PARSER" "$THERMORAW_VERSION" "$CARAFE_THERMORAW_FORMAT" \
        "$CARAFE_THERMORAW_NO_ZLIB" "$CARAFE_THERMORAW_PEAK_PICKING"
    fi
    while IFS= read -r raw; do
      stat -c '%n|%s|%Y' "$raw"
    done < "$JOB_DIR/meta/train_mzml_source_paths.txt"
  } | sha256sum | awk '{print $1}'
}

RUN_FINGERPRINT="$(compute_run_fingerprint)"
JAR_SHA256="$(sha256sum "$CARAFE_JAR_SOURCE" | awk '{print $1}')"
STATUS_FILE="$JOB_DIR/meta/status.json"
STATUS_MIRROR_FILE="$OUTPUT_DIR/run_status.json"
REPORT_VALIDATION_FILE="$JOB_DIR/meta/report_validation.json"
FAILURE_SUMMARY_FILE="$JOB_DIR/meta/failure_summary.txt"
LOCK_DIR="$JOB_DIR/meta/.carafe_run.lock"
LOCK_HELD=0
RUN_STARTED_AT=""
CARAFE_PID=""
MONITOR_PID=""
RUN_FINALIZED=0
PROGRESS_FILE="$OUTPUT_DIR/progress.txt"
RESUME_FROM_MODELS=0
RESUME_FROM_PREDICTIONS=0
CHECKPOINT_VALIDATION_FILE="$JOB_DIR/meta/prediction_checkpoint_validation.json"
CHECKPOINT_CHUNKS=""
PREDICTION_COMPLETE_LINE=""
STATUS_FINGERPRINT_MATCHES=0
PREDICTION_PROGRESS_RECORDED=0
CHECKPOINT_SOURCE_FINGERPRINT=""
CHECKPOINT_SOURCE_JAR_SHA256=""
if [[ -f "$STATUS_FILE" ]] && grep -Fq "\"run_fingerprint\": \"$RUN_FINGERPRINT\"" "$STATUS_FILE"; then
  STATUS_FINGERPRINT_MATCHES=1
fi
if [[ -f "$PROGRESS_FILE" ]] && grep -Fq '| succeeded | prediction_complete |' "$PROGRESS_FILE"; then
  PREDICTION_PROGRESS_RECORDED=1
  PREDICTION_COMPLETE_LINE="$(grep -F '| succeeded | prediction_complete |' "$PROGRESS_FILE" | tail -n 1)"
fi
if [[ -f "$STATUS_FILE" ]]; then
  read -r CHECKPOINT_SOURCE_FINGERPRINT CHECKPOINT_SOURCE_JAR_SHA256 < <(
    "$CARAFE_PYTHON" - "$STATUS_FILE" <<'PY'
import json
import sys
try:
    status = json.load(open(sys.argv[1]))
    print(status.get("run_fingerprint", ""), status.get("carafe_jar_sha256", ""))
except Exception:
    print(" ")
PY
  )
fi
if [[ "$STATUS_FINGERPRINT_MATCHES" == "1" ]] \
    && [[ -s "$OUTPUT_DIR/ms2_model.pt" ]] \
    && [[ -s "$OUTPUT_DIR/rt_model.pt" ]] \
    && [[ -s "$OUTPUT_DIR/model_evaluation_metrics.json" ]]; then
  RESUME_FROM_MODELS=1
fi
if [[ "$FORCE_FINE_TUNE" == "1" ]]; then
  # Keep --test truthful as well: forced fine-tuning never reuses a model dir.
  RESUME_FROM_MODELS=0
fi

if [[ "$PREDICTION_PROGRESS_RECORDED" == "1" ]] \
      && { [[ "$RESUME_FROM_MODELS" == "1" ]] || [[ "$RESUME_REQUESTED" == "1" ]]; }; then
  CHECKPOINT_ARGS=(--output-dir "$OUTPUT_DIR" --output "$CHECKPOINT_VALIDATION_FILE")
  [[ "${CARAFE_CCS:-0}" == "1" ]] && CHECKPOINT_ARGS+=(--ccs)
  if "$CARAFE_PYTHON" "$SCRIPT_DIR/bin/validate_carafe_prediction_checkpoint.py" "${CHECKPOINT_ARGS[@]}" >/dev/null; then
    CHECKPOINT_CHUNKS="$("$CARAFE_PYTHON" - "$CHECKPOINT_VALIDATION_FILE" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["prediction_chunks"])
PY
)"
    if [[ "$STATUS_FINGERPRINT_MATCHES" == "1" ]]; then
      RESUME_FROM_PREDICTIONS=1
    elif [[ "$RESUME_REQUESTED" == "1" && -f "$STATUS_FILE" ]]; then
      # A rebuilt JAR changes the full run fingerprint. Explicit --resume is the
      # acknowledgement that only final assembly should use that newer JAR.
      RESUME_FROM_PREDICTIONS=1
    fi
  fi
fi

if [[ "$RESUME_REQUESTED" == "1" && "$RESUME_FROM_PREDICTIONS" != "1" ]]; then
  die "--resume requested, but no complete, validated prediction checkpoint is available; see $CHECKPOINT_VALIDATION_FILE"
fi

COMPATIBILITY_ONLY_RESUME=0
if [[ "$CARAFE_DIANN2_PARQUET" != "1" && "$RESUME_REQUESTED" == "1" && -s "$RAW_LIBRARY_FILE" ]] \
    && { [[ ! -s "$DIANN2_LIBRARY_FILE" ]] \
      || [[ ! -f "$PROGRESS_FILE" ]] \
      || ! grep -Fq '| succeeded | diann2_compatibility |' "$PROGRESS_FILE"; }; then
  COMPATIBILITY_ONLY_RESUME=1
fi

write_run_status() {
  local status="$1" stage="$2" message="$3" exit_code="${4:-}" library_path="${5:-}"
  local -a status_args=(
    --output "$STATUS_FILE" --mirror-output "$STATUS_MIRROR_FILE"
    --status "$status" --stage "$stage" --message "$message"
    --fingerprint "$RUN_FINGERPRINT" --job "$JOB_DIR"
    --jar "$CARAFE_JAR_SOURCE" --jar-sha256 "$JAR_SHA256"
    --command-file "$JOB_DIR/meta/carafe_command.sh"
  )
  [[ -n "$RUN_STARTED_AT" ]] && status_args+=(--started-at "$RUN_STARTED_AT")
  [[ -n "$exit_code" ]] && status_args+=(--exit-code "$exit_code")
  [[ -n "$library_path" ]] && status_args+=(--library-path "$library_path")
  if [[ "$COMPATIBILITY_ONLY_RESUME" == "1" ]]; then
    print_kv "Resume mode" "reuse raw CARAFE Parquet; create the missing DIA-NN 2.x-compatible Parquet only"
  elif [[ "$RESUME_FROM_PREDICTIONS" == "1" ]]; then
    [[ -n "$CHECKPOINT_SOURCE_FINGERPRINT" ]] && status_args+=(--resumed-from-fingerprint "$CHECKPOINT_SOURCE_FINGERPRINT")
    [[ -n "$CHECKPOINT_SOURCE_JAR_SHA256" ]] && status_args+=(--resumed-from-jar-sha256 "$CHECKPOINT_SOURCE_JAR_SHA256")
  fi
  "$CARAFE_PYTHON" "$SCRIPT_DIR/bin/write_carafe_status.py" "${status_args[@]}"
  mkdir -p "$OUTPUT_DIR"
  printf '%s | %s | %s | %s\n' "$(date --iso-8601=seconds)" "$status" "$stage" "$message" >> "$PROGRESS_FILE"
}

validate_report() {
  "$CARAFE_PYTHON" "$SCRIPT_DIR/bin/validate_carafe_report.py" \
    --report "$REPORT" \
    --train-paths "$JOB_DIR/meta/train_mzml_source_paths.txt" \
    --output "$REPORT_VALIDATION_FILE"
}

is_valid_diann2_library() {
  local library_file="$1"
  [[ -s "$library_file" ]] || return 1
  "$CARAFE_PYTHON" - "$library_file" "$CARAFE_DIANN_REFERENCE_LIBRARY" <<'PY'
import sys
import pyarrow.parquet as pq

library, reference = map(pq.ParquetFile, sys.argv[1:])
if library.metadata.num_rows <= 0:
    raise SystemExit(1)
if library.schema_arrow != reference.schema_arrow:
    raise SystemExit(1)
PY
}

run_diann2_compatibility() {
  local source_file="$1"
  write_run_status running compatibility 'Converting CARAFE raw Parquet to the DIA-NN 2.x schema.'
  if ! "$CARAFE_PYTHON" "$DIANN2_CONVERTER" \
      --input "$source_file" \
      --output "$DIANN2_LIBRARY_FILE" \
      --reference "$CARAFE_DIANN_REFERENCE_LIBRARY"; then
    write_failure_summary 1 'DIA-NN 2.x Parquet compatibility conversion failed.'
    write_run_status failed compatibility 'DIA-NN 2.x Parquet compatibility conversion failed.' 1
    RUN_FINALIZED=1
    die "DIA-NN compatibility conversion failed; see $FAILURE_SUMMARY_FILE"
  fi
  if ! is_valid_diann2_library "$DIANN2_LIBRARY_FILE"; then
    write_failure_summary 1 'Converted DIA-NN 2.x library failed schema validation.'
    write_run_status failed compatibility 'Converted DIA-NN 2.x library failed schema validation.' 1
    RUN_FINALIZED=1
    die "Converted DIA-NN library failed schema validation; see $FAILURE_SUMMARY_FILE"
  fi
  write_run_status succeeded diann2_compatibility "DIA-NN 2.x library created: $DIANN2_LIBRARY_FILE" 0 "$DIANN2_LIBRARY_FILE"
}

write_failure_summary() {
  local exit_code="$1" reason="$2"
  {
    printf 'Carafe job failed at %s\n' "$(date --iso-8601=seconds)"
    printf 'Reason: %s\nExit code: %s\n\n' "$reason" "$exit_code"
    echo 'Matched error lines:'
    if [[ -f "$OUTPUT_DIR/carafe.log" ]]; then
      grep -Ein 'ERROR|Exception|CUDA out of memory|MS2\.Scan is missing' "$OUTPUT_DIR/carafe.log" || true
      echo
      echo 'Last 80 log lines:'
      tail -n 80 "$OUTPUT_DIR/carafe.log"
    else
      echo '(Carafe log was not created.)'
    fi
  } > "$FAILURE_SUMMARY_FILE"
}

monitor_resources() {
  local pid="$1" interval="${CARAFE_RESOURCE_INTERVAL_SECONDS:-30}"
  local monitor_file="$OUTPUT_DIR/resources.csv" timestamp mem_available mem_used share_available share_used io_read io_write gpu_line
  [[ "$interval" =~ ^[0-9]+$ ]] && (( interval >= 5 )) || interval=30
  if [[ ! -s "$monitor_file" ]]; then
    echo 'timestamp,pid,cuda_visible_devices,gpu_index,gpu_name,gpu_utilization_pct,gpu_memory_used_mib,gpu_memory_total_mib,process_read_bytes,process_write_bytes,mem_used_mib,mem_available_mib,share_available_bytes,share_used_percent' > "$monitor_file"
  fi
  while kill -0 "$pid" 2>/dev/null; do
    timestamp="$(date --iso-8601=seconds)"
    mem_available="$(awk '/MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)"
    mem_used="$(awk '/MemTotal:/ {total=$2} /MemAvailable:/ {available=$2} END {print int((total - available) / 1024)}' /proc/meminfo)"
    share_available="$(df -PB1 "$OUTPUT_DIR" | awk 'NR == 2 {print $4}')"
    share_used="$(df -P "$OUTPUT_DIR" | awk 'NR == 2 {gsub(/%/, "", $5); print $5}')"
    io_read="$(awk '/read_bytes:/ {print $2}' "/proc/$pid/io" 2>/dev/null || echo '')"
    io_write="$(awk '/write_bytes:/ {print $2}' "/proc/$pid/io" 2>/dev/null || echo '')"
    while IFS= read -r gpu_line; do
      IFS=',' read -r gpu_index gpu_name gpu_util gpu_mem_used gpu_mem_total <<< "$gpu_line"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$timestamp" "$pid" "${CUDA_VISIBLE_DEVICES:-all}" "${gpu_index// /}" "${gpu_name// /_}" \
        "${gpu_util// /}" "${gpu_mem_used// /}" "${gpu_mem_total// /}" \
        "$io_read" "$io_write" "$mem_used" "$mem_available" "$share_available" "$share_used" >> "$monitor_file"
    done < <(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    sleep "$interval"
  done
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf 'host=%s\npid=%s\nstarted_at=%s\n' "$(hostname -f 2>/dev/null || hostname)" "$$" "$(date --iso-8601=seconds)" > "$LOCK_DIR/owner.env"
    LOCK_HELD=1
    return 0
  fi
  local owner='unknown owner'
  [[ -f "$LOCK_DIR/owner.env" ]] && owner="$(tr '\n' ' ' < "$LOCK_DIR/owner.env")"
  die "Carafe job is already locked: $LOCK_DIR ($owner). Confirm that the recorded process is no longer running before removing this lock."
}

cleanup_run() {
  local code="$?"
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true
  [[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true
  if [[ "$LOCK_HELD" == "1" ]]; then
    rm -f "$LOCK_DIR/owner.env"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    LOCK_HELD=0
  fi
  if [[ "$RUN_STARTED_AT" != "" && "$RUN_FINALIZED" == "0" ]]; then
    write_failure_summary "$code" 'Runner exited before recording a successful result.'
    write_run_status failed interrupted 'Runner exited before recording a successful result.' "$code" || true
  fi
}

add_value() {
  local flag="$1" variable="$2" value="${!2:-}"
  if [[ -n "$value" ]]; then
    CARAFE_CMD+=("$flag" "$value")
  fi
}
add_bool() {
  local variable="$1" flag="$2"
  if [[ "${!variable:-0}" == "1" ]]; then
    CARAFE_CMD+=("$flag")
  fi
}

CARAFE_CMD=(
  "$CARAFE_JAVA" "${CARAFE_JAVA_ARGS[@]}" -jar "$CARAFE_JAR"
  -db "$FASTA_FILE" -o "$OUTPUT_DIR"
  -fixMod "${CARAFE_FIXED_MODS}" -varMod "${CARAFE_VARIABLE_MODS}"
  -maxVar "${CARAFE_MAX_VARIABLE_MODS}"
  -data_type "${CARAFE_DATA_TYPE}" -lf_type "${CARAFE_LIBRARY_TYPE}"
  -lf_format "${CARAFE_LIBRARY_FORMAT}" -verbose "${CARAFE_VERBOSE}"
)

add_value -itol CARAFE_FRAGMENT_TOLERANCE
add_value -itolu CARAFE_FRAGMENT_TOLERANCE_UNIT
add_value -sg CARAFE_XIC_SMOOTHING_POINTS
add_value -nf CARAFE_MIN_MATCHED_FRAGMENTS
add_value -na CARAFE_ADJACENT_SCANS
add_value -fdr CARAFE_FDR
add_value -cor CARAFE_CORRELATION
add_value -ptm_site_prob CARAFE_PTM_SITE_PROB
add_value -ptm_site_qvalue CARAFE_PTM_SITE_QVALUE
add_value -min_mz CARAFE_MIN_FRAGMENT_MZ
add_value -min_n CARAFE_MIN_HIGH_QUALITY_FRAGMENTS
add_value -enzyme CARAFE_ENZYME
add_value -miss_c CARAFE_MISSED_CLEAVAGES
add_value -minLength CARAFE_MIN_PEPTIDE_LENGTH
add_value -maxLength CARAFE_MAX_PEPTIDE_LENGTH
add_value -min_pep_mz CARAFE_MIN_PRECURSOR_MZ
add_value -max_pep_mz CARAFE_MAX_PRECURSOR_MZ
add_value -min_pep_charge CARAFE_MIN_PRECURSOR_CHARGE
add_value -max_pep_charge CARAFE_MAX_PRECURSOR_CHARGE
add_value -lf_frag_mz_min CARAFE_LIBRARY_MIN_FRAGMENT_MZ
add_value -lf_frag_mz_max CARAFE_LIBRARY_MAX_FRAGMENT_MZ
add_value -lf_top_n_frag CARAFE_LIBRARY_TOP_N_FRAGMENTS
add_value -lf_min_n_frag CARAFE_LIBRARY_MIN_N_FRAGMENTS
add_value -lf_frag_n_min CARAFE_LIBRARY_FRAGMENT_N_MIN
add_value -rt_win_offset CARAFE_RT_WINDOW_OFFSET
add_value -rt_max CARAFE_MAX_RT
add_value -n_ion_min CARAFE_NTERM_ION_MIN
add_value -c_ion_min CARAFE_CTERM_ION_MIN
add_value -se CARAFE_SEARCH_ENGINE
add_value -mode CARAFE_MODE
add_value -tf CARAFE_TUNE_TYPE
add_value -seed CARAFE_RANDOM_SEED
add_value -ai_version CARAFE_AI_VERSION
add_value -ms_instrument CARAFE_MS_INSTRUMENT
add_value -device CARAFE_DEVICE
add_value -python CARAFE_PYTHON
add_value -assembly_workers CARAFE_ASSEMBLY_WORKERS
add_value -nce CARAFE_NCE
if [[ -z "$EXTERNAL_MODEL_DIR" ]]; then
  add_value -model_dir CARAFE_MODEL_DIR
fi
add_value -ms2_model CARAFE_MS2_MODEL
add_value -mod2mass CARAFE_MOD_MASS_OVERRIDES
add_value -user_var_mods CARAFE_USER_VARIABLE_MODS

add_bool CARAFE_NORMALIZE_INTENSITY -nm
add_bool CARAFE_LIMIT_FRAGMENT_CHARGE -cs
add_bool CARAFE_EXPORT_FRAGMENT_MZ -ez
add_bool CARAFE_EXPORT_SKYLINE -skyline
add_bool CARAFE_ONLY_VALID_MATCHES -valid
add_bool CARAFE_USE_ALL_PEAKS -use_all_peaks
add_bool CARAFE_REFINE_PEAK_BOUNDARY -rf
add_bool CARAFE_EXPORT_XIC -xic
add_bool CARAFE_EXPORT_MGF -export_mgf
add_bool CARAFE_NO_PEAK_MASKING -no_masking
add_bool CARAFE_EXCLUDE_Y1 -y1
add_bool CARAFE_I2L -I2L
add_bool CARAFE_CLIP_NTERM_M -clip_n_m
add_bool CARAFE_SEMI_SPECIFIC -semi
add_bool CARAFE_FAST_IO -fast
add_bool CARAFE_TORCH_COMPILE -torch_compile
add_bool CARAFE_CCS -ccs
add_bool CARAFE_DIANN2_PARQUET -diann2_parquet
[[ "$FINE_TUNE_ONLY" == "1" ]] && CARAFE_CMD+=(-fine_tune_only)
[[ -n "$MS2_EPOCHS" ]] && CARAFE_CMD+=(-ms2_epoch "$MS2_EPOCHS")
[[ -n "$RT_EPOCHS" ]] && CARAFE_CMD+=(-rt_epoch "$RT_EPOCHS")
if [[ "$SAVE_VALIDATION_CHECKPOINTS" == "1" || "${CARAFE_SAVE_VALIDATION_CHECKPOINTS:-0}" == "1" ]]; then
  CARAFE_CMD+=(-save_validation_checkpoints)
fi
if [[ -n "${CARAFE_REFINE_RT_WINDOW:-}" ]]; then
  CARAFE_CMD+=(-rf_rt_win "$CARAFE_REFINE_RT_WINDOW")
fi

if [[ "$RESUME_FROM_PREDICTIONS" == "1" ]]; then
  CARAFE_CMD+=(-assemble_existing_predictions)
elif [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
  # Inference-only mode: the external directory supplies RT.  Carafe's Python
  # loader deliberately falls back to its generic MS2/CCS models when the
  # corresponding optional checkpoints are absent.  No Step-2 report or raw
  # training files are passed, so this branch can never fine-tune a model.
  CARAFE_CMD+=(-model_dir "$EXTERNAL_MODEL_DIR")
else
  CARAFE_CMD+=(-i "$REPORT" -ms "$CARAFE_TRAIN_DIR")
  if [[ "$RESUME_FROM_MODELS" == "1" ]]; then
    CARAFE_CMD+=(-model_dir "$OUTPUT_DIR")
  fi
fi

if [[ -z "$EXTERNAL_MODEL_DIR" ]] && ! validate_report; then
  if [[ "$TEST_MODE" != "1" ]]; then
    RUN_STARTED_AT="$(date --iso-8601=seconds)"
    write_run_status failed report_preflight 'DIA-NN report preflight failed; Carafe was not started.' 2
    RUN_FINALIZED=1
  fi
  die "DIA-NN report preflight failed; see $REPORT_VALIDATION_FILE"
fi

if [[ "$TEST_MODE" == "1" ]]; then
  print_section "Test Mode"
  print_ok "Validation successful (no Carafe process started)."

  print_section "Inputs"
  print_kv "Job" "$JOB_DIR"
  print_kv "Step-2 report" "$REPORT"
  print_kv "FASTA" "$FASTA_FILE"
  print_kv "Training inputs" "$TRAINING_FILE_COUNT valid shared files (${THERMO_RAW_COUNT} RAW, ${MZML_COUNT} mzML)"
  if (( THERMO_RAW_COUNT > 0 )); then
    print_kv "Carafe MS folder" "$CARAFE_TRAIN_DIR"
    print_kv "RAW conversion" "$PENDING_RAW_CONVERSIONS pending; $((THERMO_RAW_COUNT - PENDING_RAW_CONVERSIONS)) cached"
    print_kv "RAW parser" "$THERMORAW_EXECUTABLE ($THERMORAW_VERSION)"
    print_kv "Output format" "indexed mzML (-f=$CARAFE_THERMORAW_FORMAT)"
    print_kv "Compression" "$([[ "$CARAFE_THERMORAW_NO_ZLIB" == "1" ]] && echo "zlib disabled (-z)" || echo "zlib enabled")"
    print_kv "Peak picking" "$([[ "$CARAFE_THERMORAW_PEAK_PICKING" == "1" ]] && echo "native Thermo enabled" || echo "disabled (-p)")"
  fi
  if [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
    print_kv "Report validation" "not needed (no fine-tuning)"
  else
    print_kv "Report validation" "$REPORT_VALIDATION_FILE"
  fi
  if [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
    print_kv "External RT model" "$EXTERNAL_MODEL_DIR/rt_model.pt"
    if [[ -s "$EXTERNAL_MODEL_DIR/ms2_model.pt" ]]; then
      print_kv "MS2 model" "$EXTERNAL_MODEL_DIR/ms2_model.pt"
    else
      print_kv "MS2 model" "Carafe generic (fixed fallback)"
    fi
    print_kv "Training" "disabled; inference only"
  fi

  print_section "Carafe Configuration"
  print_kv "Specificity" "$([[ "$CARAFE_SEMI_SPECIFIC" == "1" ]] && echo "semi-specific (-semi)" || echo "specific")"
  print_kv "Enzyme" "${CARAFE_ENZYME}; ${CARAFE_MISSED_CLEAVAGES} missed cleavage(s)"
  print_kv "Peptide space" "${CARAFE_MIN_PEPTIDE_LENGTH}-${CARAFE_MAX_PEPTIDE_LENGTH} aa; ${CARAFE_MIN_PRECURSOR_MZ}-${CARAFE_MAX_PRECURSOR_MZ} m/z; z ${CARAFE_MIN_PRECURSOR_CHARGE}-${CARAFE_MAX_PRECURSOR_CHARGE}"
  print_kv "Library format" "${CARAFE_LIBRARY_TYPE}, ${CARAFE_LIBRARY_FORMAT}"
  print_kv "Model / device" "${CARAFE_MODE}, ${CARAFE_AI_VERSION}, ${CARAFE_DEVICE} (${CARAFE_MS_INSTRUMENT})"
  print_kv "Carafe JAR source" "$CARAFE_JAR_SOURCE"
  print_kv "Runtime staging" "$CARAFE_RUNTIME_ROOT/run.* (created only for a real run)"
  print_kv "Java memory" "$CARAFE_JAVA_OPTS"
  print_kv "Run fingerprint" "$RUN_FINGERPRINT"
  if [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
    print_kv "Resume mode" "external model; library inference without fine-tuning"
  elif [[ "$COMPATIBILITY_ONLY_RESUME" == "1" ]]; then
    print_kv "Resume mode" "reuse raw CARAFE Parquet; create or validate DIA-NN 2.x-compatible Parquet only"
  elif [[ "$RESUME_FROM_PREDICTIONS" == "1" ]]; then
    print_kv "Resume mode" "reuse validated prediction chunks; final library assembly only"
  elif [[ "$RESUME_FROM_MODELS" == "1" ]]; then
    print_kv "Resume mode" "reuse completed fine-tuned models; repeat library inference"
  else
    print_kv "Resume mode" "full data preparation and fine-tuning"
  fi
  print_kv "Prediction checkpoint" "$CHECKPOINT_VALIDATION_FILE"
  print_kv "Assembly workers" "$CARAFE_ASSEMBLY_WORKERS (bounded; final writer remains ordered)"

  print_section "Resume Decision"
  print_kv "Resume requested" "$([[ "$RESUME_REQUESTED" == "1" ]] && echo 'yes' || echo 'no')"
  if [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
    print_kv "Completed stages" "External RT model supplied"
    print_kv "Restart stage" "FASTA digestion and library inference"
    print_kv "Skipped" "Training-data preparation and all fine-tuning"
  elif [[ "$COMPATIBILITY_ONLY_RESUME" == "1" ]]; then
    print_kv "Last durable checkpoint" "CARAFE raw library: $RAW_LIBRARY_FILE"
    print_kv "Completed stages" "CARAFE assembly"
    print_kv "Restart stage" "DIA-NN 2.x schema conversion"
    print_kv "Skipped on resume" "Training-data preparation; fine-tuning; FASTA digestion; model inference; CARAFE assembly"
  elif [[ "$RESUME_FROM_PREDICTIONS" == "1" ]]; then
    print_kv "Last durable checkpoint" "$PREDICTION_COMPLETE_LINE"
    print_kv "Completed stages" "Training-data preparation; fine-tuning; semi-specific digestion; library inference (${CHECKPOINT_CHUNKS} chunks)"
    print_kv "Restart stage" "Final library assembly"
    print_kv "Skipped on resume" "Training-data preparation; fine-tuning; FASTA digestion; model inference"
  elif [[ "$RESUME_FROM_MODELS" == "1" ]]; then
    print_kv "Completed stages" "Training-data preparation and fine-tuning"
    print_kv "Restart stage" "Semi-specific digestion and library inference"
    print_kv "Skipped on resume" "Training-data preparation and fine-tuning"
  else
    print_kv "Completed stages" "No reusable checkpoint"
    print_kv "Restart stage" "Training-data preparation"
    print_kv "Skipped on resume" "None"
  fi

  print_section "Modification Mapping"
  print_modification_mapping
  print_kv "Carafe CLI" "-fixMod ${CARAFE_FIXED_MODS}; -varMod ${CARAFE_VARIABLE_MODS}; -maxVar ${CARAFE_MAX_VARIABLE_MODS}"

  print_section "Execution Plan"
  print_kv "Action" "Create a semi-specific Carafe library from the validated inputs"
  print_kv "Output directory" "$OUTPUT_DIR"
  print_kv "Library record" "$JOB_DIR/meta/carafe_library_path.txt"
  print_kv "Command record" "$JOB_DIR/meta/carafe_command.sh"

  if [[ "$SHOW_COMMAND" == "1" ]]; then
    print_section "Planned Command"
    print_command
  else
    print_kv "Full command" "Hidden for compact output; add --show-command to print it."
  fi
  exit 0
fi

# A prior CARAFE assembly can already have produced the large raw library when
# the schema-upgrade step was interrupted or did not exist yet.  In that case
# --resume performs only the missing conversion, never reassembles predictions.
if [[ "$CARAFE_DIANN2_PARQUET" != "1" && "$RESUME_REQUESTED" == "1" && -s "$RAW_LIBRARY_FILE" ]]; then
  acquire_lock
  RUN_STARTED_AT="$(date --iso-8601=seconds)"
  trap cleanup_run EXIT
  if is_valid_diann2_library "$DIANN2_LIBRARY_FILE"; then
    printf '%s\n' "$DIANN2_LIBRARY_FILE" > "$JOB_DIR/meta/carafe_library_path.txt"
    write_run_status succeeded diann2_compatibility "DIA-NN 2.x library already present: $DIANN2_LIBRARY_FILE" 0 "$DIANN2_LIBRARY_FILE"
    write_run_status succeeded completed "Library created: $DIANN2_LIBRARY_FILE" 0 "$DIANN2_LIBRARY_FILE"
    RUN_FINALIZED=1
    echo "Carafe DIA-NN 2.x library already present: $DIANN2_LIBRARY_FILE"
    exit 0
  fi
  run_diann2_compatibility "$RAW_LIBRARY_FILE"
  printf '%s\n' "$DIANN2_LIBRARY_FILE" > "$JOB_DIR/meta/carafe_library_path.txt"
  write_run_status succeeded completed "Library created: $DIANN2_LIBRARY_FILE" 0 "$DIANN2_LIBRARY_FILE"
  RUN_FINALIZED=1
  echo "Carafe DIA-NN 2.x library: $DIANN2_LIBRARY_FILE"
  exit 0
fi

acquire_lock
RUN_STARTED_AT="$(date --iso-8601=seconds)"
trap cleanup_run EXIT
if (( THERMO_RAW_COUNT > 0 )) && [[ -z "$EXTERNAL_MODEL_DIR" ]]; then
  write_run_status running raw_conversion "Preparing ${THERMO_RAW_COUNT} Thermo RAW training inputs as mzML."
  prepare_carafe_training_directory
fi
write_run_status queued preflight 'Input validation passed; staging immutable Carafe runtime.'

stage_carafe_jar

# Carafe extracts Python resources lazily from its JAR; replace only the JAR
# argument in the already complete command with the immutable staged copy.
for ((i = 0; i < ${#CARAFE_CMD[@]} - 1; i++)); do
  if [[ "${CARAFE_CMD[$i]}" == "-jar" ]]; then
    CARAFE_CMD[$((i + 1))]="$CARAFE_JAR"
    break
  fi
done

mkdir -p "$OUTPUT_DIR"
if [[ -n "$EXTERNAL_MODEL_DIR" ]]; then
  write_run_status running models_ready "Using external RT model without fine-tuning: $EXTERNAL_MODEL_DIR"
elif [[ "$RESUME_FROM_PREDICTIONS" == "1" ]]; then
  write_run_status running assembly 'Reusing validated prediction chunks; assembling final library only.'
elif [[ "$RESUME_FROM_MODELS" == "1" ]]; then
  write_run_status running models_ready 'Reusing validated fine-tuned models; starting library inference.'
else
  write_run_status running training 'Starting training-data preparation and fine-tuning.'
fi
{
  echo '#!/usr/bin/env bash'
  printf '# Source JAR: %q\n' "$CARAFE_JAR_SOURCE"
  printf '# Immutable runtime JAR: %q\n' "$CARAFE_JAR"
  printf '%q ' "${CARAFE_CMD[@]}"
  echo
} > "$JOB_DIR/meta/carafe_command.sh"
chmod +x "$JOB_DIR/meta/carafe_command.sh"

set +e
"${CARAFE_CMD[@]}" > >(tee "$OUTPUT_DIR/carafe.log") 2>&1 &
CARAFE_PID="$!"
monitor_resources "$CARAFE_PID" &
MONITOR_PID="$!"
wait "$CARAFE_PID"
STATUS=$?
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=""
if (( STATUS != 0 )); then
  if [[ -f "$PROGRESS_FILE" ]] && grep -Fq '| succeeded | prediction_complete |' "$PROGRESS_FILE"; then
    write_failure_summary "$STATUS" "Carafe exited during final library assembly after successful inference."
    write_run_status failed assembly "Carafe exited during final library assembly after successful inference." "$STATUS"
  else
    write_failure_summary "$STATUS" "Carafe exited with code $STATUS."
    write_run_status failed carafe "Carafe exited with code $STATUS." "$STATUS"
  fi
  RUN_FINALIZED=1
  die "Carafe failed with exit code $STATUS; see $FAILURE_SUMMARY_FILE"
fi

if [[ "$FINE_TUNE_ONLY" == "1" ]]; then
  [[ -f "$OUTPUT_DIR/model_evaluation_metrics.json" ]] || {
    write_failure_summary 1 "Fine-tuning finished without model_evaluation_metrics.json."
    write_run_status failed validation "Fine-tuning diagnostics are missing." 1
    RUN_FINALIZED=1
    die "Fine-tuning diagnostics are missing: $OUTPUT_DIR/model_evaluation_metrics.json"
  }
  write_run_status succeeded fine_tuned "Fine-tuning completed; library inference and assembly were skipped." 0 "$OUTPUT_DIR"
  RUN_FINALIZED=1
  echo "Fine-tuning completed: $OUTPUT_DIR"
  echo "Report: $OUTPUT_DIR/training_report.html"
  echo "Diagnostics: $OUTPUT_DIR/training_diagnostics"
  exit 0
fi

if [[ "${CARAFE_LIBRARY_FORMAT,,}" == "parquet" ]]; then
  if [[ "$CARAFE_DIANN2_PARQUET" == "1" ]]; then
    LIBRARY_BASENAME='carafe_spectral_library.diann2.parquet'
  else
    LIBRARY_BASENAME='carafe_spectral_library.parquet'
  fi
else
  LIBRARY_BASENAME='carafe_spectral_library.tsv'
fi
mapfile -t LIBRARIES < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "$LIBRARY_BASENAME" -printf '%p\n' | sort)
if (( ${#LIBRARIES[@]} != 1 )); then
  write_failure_summary 1 "Expected one Carafe $CARAFE_LIBRARY_FORMAT output ($LIBRARY_BASENAME), found ${#LIBRARIES[@]}."
  write_run_status failed validation "Expected one Carafe $CARAFE_LIBRARY_FORMAT output ($LIBRARY_BASENAME), found ${#LIBRARIES[@]}." 1
  RUN_FINALIZED=1
  die "Expected one Carafe $CARAFE_LIBRARY_FORMAT output ($LIBRARY_BASENAME), found ${#LIBRARIES[@]}; see $FAILURE_SUMMARY_FILE"
fi
if [[ "${CARAFE_LIBRARY_FORMAT,,}" == "parquet" ]]; then
"$CARAFE_PYTHON" - "${LIBRARIES[0]}" <<'PY'
import sys
import pyarrow.parquet as pq
path = sys.argv[1]
parquet = pq.ParquetFile(path)
if parquet.metadata.num_rows <= 0:
    raise SystemExit(f"Library parquet is empty: {path}")
print(f"Library parquet validated: {parquet.metadata.num_rows} rows, {len(parquet.schema_arrow.names)} columns")
PY
  if [[ "$CARAFE_DIANN2_PARQUET" == "1" ]] && ! is_valid_diann2_library "${LIBRARIES[0]}"; then
    write_failure_summary 1 "Direct DIA-NN 2.x library failed schema validation."
    write_run_status failed validation "Direct DIA-NN 2.x library failed schema validation." 1
    RUN_FINALIZED=1
    die "Direct DIA-NN 2.x library failed schema validation; see $FAILURE_SUMMARY_FILE"
  fi
else
  [[ -s "${LIBRARIES[0]}" ]] || die "Library TSV is empty: ${LIBRARIES[0]}"
  IFS=$'\t' read -r -a LIBRARY_HEADER < <(head -n 1 "${LIBRARIES[0]}")
  (( ${#LIBRARY_HEADER[@]} >= 13 )) || die "Library TSV has an invalid header: ${LIBRARIES[0]}"
  printf 'Library TSV validated: %s columns\n' "${#LIBRARY_HEADER[@]}"
fi
FINAL_LIBRARY_FILE="${LIBRARIES[0]}"
if [[ "${CARAFE_LIBRARY_FORMAT,,}" == "parquet" && "$CARAFE_DIANN2_PARQUET" != "1" ]]; then
  if ! is_valid_diann2_library "$DIANN2_LIBRARY_FILE"; then
    run_diann2_compatibility "${LIBRARIES[0]}"
  elif ! grep -Fq '| succeeded | diann2_compatibility |' "$PROGRESS_FILE" 2>/dev/null; then
    write_run_status succeeded diann2_compatibility "DIA-NN 2.x library already present: $DIANN2_LIBRARY_FILE" 0 "$DIANN2_LIBRARY_FILE"
  fi
  FINAL_LIBRARY_FILE="$DIANN2_LIBRARY_FILE"
fi
printf '%s\n' "$FINAL_LIBRARY_FILE" > "$JOB_DIR/meta/carafe_library_path.txt"
write_run_status succeeded completed "Library created: $FINAL_LIBRARY_FILE" 0 "$FINAL_LIBRARY_FILE"
RUN_FINALIZED=1
echo "Carafe library: $FINAL_LIBRARY_FILE"
