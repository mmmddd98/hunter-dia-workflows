#!/usr/bin/env bash
set -euo pipefail

# Prepare a minimal, fully traceable Carafe job when a purchased dataset is
# available only as a DIA-NN Step-2 report, its mzML files and the matching
# FASTA.  This deliberately produces the same input/ and meta/ contract that
# run_carafe_job.sh consumes for regular pipeline-derived jobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/../diaNN_pipline" && pwd)"
DEFAULT_CONFIG="$PIPELINE_DIR/config/carafe.cfg"

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  bash data_collect/prepare_external_carafe_source.sh \
    --id <permanent-source-id> \
    --report <dia-nn-step2-report.parquet> \
    --mzml-dir <directory-containing-training-mzMLs> \
    --fasta <exact-search-fasta> [options]

This prepares, but does not run, a Carafe fine-tuning source from an external
or purchased DIA-NN dataset. All mzMLs found below --mzml-dir are used as
training files. The source report is copied into the job after Carafe's schema
normalisation; mzMLs are symlinked, never copied.

Options:
  --name <job-name>     Job directory name (default: carafe_source)
  --profile <name>      balanced, fast, or slow (default: balanced)
  --job-root <path>     Parent directory for <profile>/<job-name>
  --experiment-name <text>
                         Original experiment name for provenance (default: --id)
  --config <path>       Carafe config (default: diaNN_pipline/config/carafe.cfg)
  --python <path>       Python for the report normaliser (default: config value)
  --set KEY=VALUE       Override a CARAFE_* config setting; repeatable
  --test                Validate all inputs and print the destination only

The default job location is:
  <CARAFE_SHARE_ROOT>/external_finetune_sources/<id>/carafe/<profile>/<name>

The chosen config must match the original DIA-NN search space and modification
setup. In particular, do not use the Heavy-Dimethyl defaults for an unmodified
or differently modified external search without supplying the matching --set
overrides or a dedicated config.
EOF
}

safe_name() {
  [[ "$2" =~ ^[A-Za-z0-9._-]+$ ]] || die "$1 contains unsupported characters: $2"
}

SOURCE_ID=""
REPORT=""
MZML_DIR=""
FASTA=""
JOB_NAME="carafe_source"
PROFILE="balanced"
JOB_ROOT=""
EXPERIMENT_NAME=""
CARAFE_CONFIG="$DEFAULT_CONFIG"
PYTHON_BIN=""
TEST_MODE=0
OVERRIDES=()
ARGS=("$@")

for ((index = 0; index < ${#ARGS[@]}; index++)); do
  case "${ARGS[$index]}" in
    --id) ((index += 1)); SOURCE_ID="${ARGS[$index]:-}" ;;
    --report) ((index += 1)); REPORT="${ARGS[$index]:-}" ;;
    --mzml-dir) ((index += 1)); MZML_DIR="${ARGS[$index]:-}" ;;
    --fasta) ((index += 1)); FASTA="${ARGS[$index]:-}" ;;
    --name) ((index += 1)); JOB_NAME="${ARGS[$index]:-}" ;;
    --profile) ((index += 1)); PROFILE="${ARGS[$index]:-}" ;;
    --job-root) ((index += 1)); JOB_ROOT="${ARGS[$index]:-}" ;;
    --experiment-name) ((index += 1)); EXPERIMENT_NAME="${ARGS[$index]:-}" ;;
    --config) ((index += 1)); CARAFE_CONFIG="${ARGS[$index]:-}" ;;
    --python) ((index += 1)); PYTHON_BIN="${ARGS[$index]:-}" ;;
    --set) ((index += 1)); OVERRIDES+=("${ARGS[$index]:-}") ;;
    --test) TEST_MODE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: ${ARGS[$index]}" ;;
  esac
done

[[ -n "$SOURCE_ID" && -n "$REPORT" && -n "$MZML_DIR" && -n "$FASTA" ]] || {
  usage >&2
  exit 2
}
safe_name "source id" "$SOURCE_ID"
safe_name "job name" "$JOB_NAME"
PROFILE="${PROFILE,,}"
case "$PROFILE" in balanced|fast|slow) ;; *) die "--profile must be balanced, fast, or slow" ;; esac

REPORT="$(realpath -e "$REPORT")"
MZML_DIR="$(realpath -e "$MZML_DIR")"
FASTA="$(realpath -e "$FASTA")"
[[ -f "$REPORT" ]] || die "Step-2 report is not a file: $REPORT"
[[ -d "$MZML_DIR" ]] || die "mzML directory is not a directory: $MZML_DIR"
[[ -f "$FASTA" ]] || die "FASTA is not a file: $FASTA"
[[ -f "$CARAFE_CONFIG" ]] || die "Carafe config not found: $CARAFE_CONFIG"

# The normaliser and the generated job must use exactly the same defaults as
# a normal pipeline job.  `paths.cfg`/`defaults.cfg` only define variables.
source "$PIPELINE_DIR/config/paths.cfg"
source "$PIPELINE_DIR/config/defaults.cfg"
source "$CARAFE_CONFIG"
for override in "${OVERRIDES[@]}"; do
  [[ "$override" == CARAFE_*=* ]] || die "--set must use a CARAFE_* key, got: $override"
  key="${override%%=*}"
  value="${override#*=}"
  [[ "$key" =~ ^CARAFE_[A-Z0-9_]+$ ]] || die "Invalid --set key: $key"
  printf -v "$key" '%s' "$value"
done

PYTHON_BIN="${PYTHON_BIN:-${CARAFE_PYTHON:-python3}}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$SOURCE_ID}"

# `-L` also supports a curated directory containing symlinks to raw mzMLs.
mapfile -d '' -t MZMLS < <(find -L "$MZML_DIR" -type f \( -iname '*.mzml' \) -print0 | sort -z)
(( ${#MZMLS[@]} > 0 )) || die "No .mzML files found below: $MZML_DIR"
declare -A seen_names=()
for mzml in "${MZMLS[@]}"; do
  name="$(basename "$mzml")"
  [[ -z "${seen_names[$name]:-}" ]] || die "mzML basename occurs more than once: $name"
  seen_names[$name]=1
done

if [[ -n "$JOB_ROOT" ]]; then
  JOB_ROOT="$(realpath -m "$JOB_ROOT")"
else
  : "${CARAFE_SHARE_ROOT:?CARAFE_SHARE_ROOT is required in $CARAFE_CONFIG}"
  JOB_ROOT="${CARAFE_SHARE_ROOT%/}/external_finetune_sources/${SOURCE_ID}/carafe"
fi
JOB_DIR="$JOB_ROOT/$PROFILE/$JOB_NAME"
INPUT_DIR="$JOB_DIR/input"
TRAIN_DIR="$INPUT_DIR/train_mzml"
META_DIR="$JOB_DIR/meta"
OUTPUT_DIR="$JOB_DIR/carafe_output"

echo "External source ID: $SOURCE_ID"
echo "Step-2 report:      $REPORT"
echo "Training mzMLs:     ${#MZMLS[@]} below $MZML_DIR"
echo "FASTA:              $FASTA"
echo "Prepared job:       $JOB_DIR"
if [[ "$TEST_MODE" == "1" ]]; then
  echo "TEST MODE: inputs validated; no files were written."
  exit 0
fi
[[ ! -e "$JOB_DIR" ]] || die "Job already exists and is immutable: $JOB_DIR (choose another --name)"

mkdir -p "$TRAIN_DIR" "$META_DIR" "$OUTPUT_DIR"
"$PYTHON_BIN" "$PIPELINE_DIR/bin/normalize_carafe_report.py" \
  --source "$REPORT" --destination "$INPUT_DIR/diann_step2_report.parquet"
for mzml in "${MZMLS[@]}"; do
  ln -s "$mzml" "$TRAIN_DIR/$(basename "$mzml")"
done
printf '%s\n' "$FASTA" > "$META_DIR/fasta_source_path.txt"
printf '%s\n' "${MZMLS[@]}" > "$META_DIR/train_mzml_source_paths.txt"
printf '%s\n' "External source '$SOURCE_ID'; report=$REPORT" > "$META_DIR/source_step2_command.txt"
printf '%s\n' "" > "$META_DIR/source_step5_baseline_command.txt"

case "${CARAFE_ENZYME:-}" in
  4) DIANN_CUT='R*' ;;
  7) DIANN_CUT='K*' ;;
  *) die "Unsupported CARAFE_ENZYME=${CARAFE_ENZYME:-<unset>}; cannot create a verified digestion contract" ;;
esac
{
  echo '# External source contract; values must match the original DIA-NN search.'
  printf 'DIANN_SEMI=%q\n' "${CARAFE_SEMI_SPECIFIC}"
  printf 'DIANN_CUT=%q\n' "$DIANN_CUT"
  printf 'DIANN_MISSED_CLEAVAGES=%q\n' "${CARAFE_MISSED_CLEAVAGES}"
  printf 'DIANN_MIN_PEP_LEN=%q\n' "${CARAFE_MIN_PEPTIDE_LENGTH}"
  printf 'DIANN_MAX_PEP_LEN=%q\n' "${CARAFE_MAX_PEPTIDE_LENGTH}"
  printf 'DIANN_MIN_PR_MZ=%q\n' "${CARAFE_MIN_PRECURSOR_MZ}"
  printf 'DIANN_MAX_PR_MZ=%q\n' "${CARAFE_MAX_PRECURSOR_MZ}"
  printf 'CARAFE_SEMI_SPECIFIC=%q\n' "${CARAFE_SEMI_SPECIFIC}"
  printf 'CARAFE_ENZYME=%q\n' "${CARAFE_ENZYME}"
  printf 'CARAFE_MISSED_CLEAVAGES=%q\n' "${CARAFE_MISSED_CLEAVAGES}"
  printf 'CARAFE_MIN_PEPTIDE_LENGTH=%q\n' "${CARAFE_MIN_PEPTIDE_LENGTH}"
  printf 'CARAFE_MAX_PEPTIDE_LENGTH=%q\n' "${CARAFE_MAX_PEPTIDE_LENGTH}"
  printf 'CARAFE_MIN_PRECURSOR_MZ=%q\n' "${CARAFE_MIN_PRECURSOR_MZ}"
  printf 'CARAFE_MAX_PRECURSOR_MZ=%q\n' "${CARAFE_MAX_PRECURSOR_MZ}"
} > "$META_DIR/digestion_contract.env"
{
  echo '# Resolved Carafe configuration for an external source'
  echo "# Source: $CARAFE_CONFIG"
  for variable in $(compgen -v | sort); do
    [[ "$variable" == CARAFE_* ]] || continue
    printf '%s=%q\n' "$variable" "${!variable}"
  done
} > "$META_DIR/carafe_options.env"

sha256sum "$REPORT" "$FASTA" > "$META_DIR/source_sha256.txt"
{
  echo 'schema_version: 1'
  printf "created_at: '%s'\n" "$(date --iso-8601=seconds)"
  echo 'job:'
  printf "  name: '%s'\n" "$JOB_NAME"
  printf "  directory: '%s'\n" "$JOB_DIR"
  echo "  stage_mode: 'symlink'"
  echo 'source_experiment:'
  printf "  name: '%s'\n" "$EXPERIMENT_NAME"
  echo "  pipeline_version: 'external'"
  echo "  directory: 'external_input'"
  printf "  profile: '%s'\n" "$PROFILE"
  echo "  diann_mod_config: 'external_unverified'"
  echo 'inputs:'
  printf "  diann_step2_report: '%s'\n" "$REPORT"
  echo "  staged_diann_step2_report: 'input/diann_step2_report.parquet'"
  printf "  fasta: '%s'\n" "$FASTA"
  echo "  source_fasta_path_file: 'meta/fasta_source_path.txt'"
  printf '  training_mzml_count: %s\n' "${#MZMLS[@]}"
  echo '  training_mzml:'
  for mzml in "${MZMLS[@]}"; do printf "    - '%s'\n" "$mzml"; done
  echo 'carafe:'
  echo "  input_report: 'input/diann_step2_report.parquet'"
  echo "  input_mzml_directory: 'input/train_mzml'"
  echo "  input_fasta: 'meta/fasta_source_path.txt'"
  echo "  output_directory: 'carafe_output'"
  echo 'provenance:'
  echo "  source_checksums: 'meta/source_sha256.txt'"
  echo "  source_step2_command_file: 'meta/source_step2_command.txt'"
} > "$JOB_DIR/manifest.yaml"

cat > "$JOB_DIR/README.md" <<EOF
# External Carafe fine-tuning source: $SOURCE_ID

Prepared from an external DIA-NN Step-2 report, ${#MZMLS[@]} training mzML files,
and the matching FASTA. Input mzMLs are symlinks; only the normalised report is
stored in this job.

Run on the GPU machine:

\`\`\`bash
cd /path/to/carafe2/ma
bash diaNN_pipline/run_carafe_job.sh --job "$JOB_DIR" --machine gpu_pc.local --fine-tune-only
\`\`\`

Then register this job with \`data_collect/fine_tune_catalog.py discover-job\`.
EOF

echo "Prepared external Carafe source: $JOB_DIR"
