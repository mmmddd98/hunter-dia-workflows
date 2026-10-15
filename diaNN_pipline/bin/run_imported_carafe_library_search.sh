#!/usr/bin/env bash
# Run or validate a DIA-NN comparison search using an already imported Carafe
# library.  Unlike run_carafe_final_search.sh, this intentionally has no
# dependency on a locally prepared Carafe job or a shared shared filesystem.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/config/paths.cfg"

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  bash bin/run_imported_carafe_library_search.sh \
    --experiment <existing_cluster_experiment> \
    --library <imported_carafe_library.parquet> \
    --search-name <name> \
    --contract <imported_job/meta/digestion_contract.env> \
    [--pipeline-version <directory>] [--threads <n>] [--mode validate|run]

The experiment's local 05_quant/baseline/startcmd.txt supplies FASTA, held-out
mzML files, modifications and ordinary Step-5 settings.  The library is read
in place; no library copy is made.  "validate" invokes Step 5 with --test and
never creates output or starts DIA-NN (the default).
EOF
}

EXPERIMENT=""
LIBRARY=""
SEARCH_NAME=""
CONTRACT=""
PIPELINE_VERSION="${DIANN_VERSION_DIR:-run-diann}"
THREADS=""
MODE="validate"
while (($#)); do
  case "$1" in
    --experiment) EXPERIMENT="${2:-}"; shift 2 ;;
    --library|--lib) LIBRARY="${2:-}"; shift 2 ;;
    --search-name) SEARCH_NAME="${2:-}"; shift 2 ;;
    --contract) CONTRACT="${2:-}"; shift 2 ;;
    --pipeline-version) PIPELINE_VERSION="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ "$EXPERIMENT" =~ ^[A-Za-z0-9._-]+$ ]] || die "--experiment is required and must be a safe name"
[[ "$SEARCH_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--search-name is required and must be a safe name"
[[ "$PIPELINE_VERSION" =~ ^[A-Za-z0-9._-]+$ ]] || die "Invalid --pipeline-version"
[[ "$MODE" == "validate" || "$MODE" == "run" ]] || die "--mode must be validate or run"
[[ -z "$THREADS" || "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"

SOURCE_DIR="${OUTROOT%/}/${PIPELINE_VERSION}/${EXPERIMENT}"
BASELINE_COMMAND_FILE="$SOURCE_DIR/05_quant/baseline/startcmd.txt"
[[ -d "$SOURCE_DIR" ]] || die "Cluster experiment is missing: $SOURCE_DIR"
[[ -s "$BASELINE_COMMAND_FILE" ]] || die "Baseline Step-5 command is missing: $BASELINE_COMMAND_FILE"
LIBRARY="$(realpath -e "$LIBRARY")" || die "Library is missing: $LIBRARY"
[[ -s "$LIBRARY" ]] || die "Library is empty: $LIBRARY"
CONTRACT="$(realpath -e "$CONTRACT")" || die "Digestion contract is missing: $CONTRACT"

# The recorded command can contain shell escaping.  shlex preserves it better
# than read -a and lets this runner use exactly the experiment's test set.
mapfile -t BASELINE_ARGS < <(python3 - "$BASELINE_COMMAND_FILE" <<'PY'
import pathlib, shlex, sys
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if not line.lstrip().startswith("#") and "step5_quant.sh" in line:
        for value in shlex.split(line):
            print(value)
        break
PY
)
(( ${#BASELINE_ARGS[@]} > 0 )) || die "Could not parse a Step-5 command from $BASELINE_COMMAND_FILE"

FASTA=""
VERBOSE="4"
SOURCE_SETS=()
TEST_MZMLS=()
for ((i = 0; i < ${#BASELINE_ARGS[@]}; i++)); do
  case "${BASELINE_ARGS[$i]}" in
    --fastaabs|--fasta) ((i += 1)); FASTA="${BASELINE_ARGS[$i]:-}" ;;
    --f) ((i += 1)); TEST_MZMLS+=("${BASELINE_ARGS[$i]:-}") ;;
    --set) ((i += 1)); SOURCE_SETS+=("${BASELINE_ARGS[$i]:-}") ;;
    --verbose) ((i += 1)); VERBOSE="${BASELINE_ARGS[$i]:-4}" ;;
  esac
done
[[ -f "$FASTA" ]] || die "Local baseline FASTA is missing: $FASTA"
(( ${#TEST_MZMLS[@]} > 0 )) || die "No --f test mzMLs were recorded in $BASELINE_COMMAND_FILE"
for raw in "${TEST_MZMLS[@]}"; do [[ -f "$raw" ]] || die "Local test mzML is missing: $raw"; done

# shellcheck source=/dev/null
source "$CONTRACT"
: "${DIANN_SEMI:?Missing DIANN_SEMI in contract}"
: "${DIANN_CUT:?Missing DIANN_CUT in contract}"
: "${DIANN_MISSED_CLEAVAGES:?Missing DIANN_MISSED_CLEAVAGES in contract}"
: "${DIANN_MIN_PEP_LEN:?Missing DIANN_MIN_PEP_LEN in contract}"
: "${DIANN_MAX_PEP_LEN:?Missing DIANN_MAX_PEP_LEN in contract}"
: "${DIANN_MIN_PR_MZ:?Missing DIANN_MIN_PR_MZ in contract}"
: "${DIANN_MAX_PR_MZ:?Missing DIANN_MAX_PR_MZ in contract}"

RESULT_DIR="$SOURCE_DIR/quant_search/carafe/balanced/$SEARCH_NAME"
TEMP_DIR="${TMPBASE%/}/carafe/${EXPERIMENT}/balanced/${SEARCH_NAME}"
if [[ "$MODE" == "run" && -e "$RESULT_DIR" ]]; then
  die "Refusing to overwrite an existing result directory: $RESULT_DIR"
fi

STEP5=(
  bash "$ROOT/bin/step5_quant.sh"
  --expname "$EXPERIMENT"
  --fastaabs "$FASTA"
  --lib "$LIBRARY"
  --out-dir "$RESULT_DIR"
  --out-sub results
  --temp-dir "$TEMP_DIR"
  --log-dir "$RESULT_DIR/logs"
  --verbose "$VERBOSE"
)
# Retain experiment-specific settings (notably MOD_CONFIG) but don't allow the
# old command to redirect paths, replace the library, or override the contract.
for setting in "${SOURCE_SETS[@]}"; do
  case "${setting%%=*}" in
    THREADS|threads|CUT|MISSED_CLEAVAGES|MIN_PEP_LEN|MAX_PEP_LEN|MIN_PR_MZ|MAX_PR_MZ|enable_SEMI|EXTRA_FLAGS) continue ;;
  esac
  STEP5+=(--set "$setting")
done
[[ -n "$THREADS" ]] && STEP5+=(--set "THREADS=$THREADS")
STEP5+=(
  --set "EXTRA_FLAGS="
  --set "enable_REPORT_DECOYS=0"
  --set "enable_FASTA_SEARCH=0"
  --set "enable_MATRICES=0"
  --set "enable_GEN_SPEC_LIB=0"
  --set "enable_CONT_QUANT_EXQCLUDE_CRAP=0"
  --set "enable_REANALYSE=0"
  --set "enable_RT_PROFILING=0"
  --set "enable_REANNOTATE=1"
  --set "enable_PEPTIDOFORMS=1"
  --set "CUT=$DIANN_CUT"
  --set "MISSED_CLEAVAGES=$DIANN_MISSED_CLEAVAGES"
  --set "MIN_PEP_LEN=$DIANN_MIN_PEP_LEN"
  --set "MAX_PEP_LEN=$DIANN_MAX_PEP_LEN"
  --set "MIN_PR_MZ=$DIANN_MIN_PR_MZ"
  --set "MAX_PR_MZ=$DIANN_MAX_PR_MZ"
  --set "enable_SEMI=$DIANN_SEMI"
)
for raw in "${TEST_MZMLS[@]}"; do STEP5+=(--f "$raw"); done

printf 'Experiment : %s\nLibrary    : %s\nOutput     : %s\nTemp       : %s\nTest mzMLs: %s\nMode       : %s\n' \
  "$SOURCE_DIR" "$LIBRARY" "$RESULT_DIR" "$TEMP_DIR" "${#TEST_MZMLS[@]}" "$MODE"
printf 'Step 5     : '; printf '%q ' "${STEP5[@]}"; echo
if [[ "$MODE" == "validate" ]]; then
  STEP5+=(--test)
fi
"${STEP5[@]}"

if [[ "$MODE" == "run" ]]; then
  [[ -s "$RESULT_DIR/results/final_report.parquet" ]] || die "DIA-NN completed without final_report.parquet"
  echo "Completed: $RESULT_DIR/results/final_report.parquet"
fi
