#!/usr/bin/env bash
# Submit/run DIA-NN Step 5 for an already generated external-Carafe2 library.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$ROOT/config/paths.cfg"
die() { echo "ERROR: $*" >&2; exit 1; }
q() { printf '%q' "$1"; }

usage() {
  cat <<'EOF'
Usage:
  bash run_external_carafe2_library_search.sh \
    --job <prepared_carafe_job> --model-id <registry_id> [options]

Options:
  --library <path>             Override meta/carafe_library_path.txt
  --search-id <id>             Stable result name (default: timestamp)
  --executor cluster|direct   DIA-NN executor (default: machine policy)
  --threads <n>                Override the recorded DIA-NN thread count
  --prepare-only               Write the reproducible worker job but do not start it
  --test                       Read-only validation; print the exact Step-5 command

Only DIA-NN Step 5 is run. The source experiment, its FASTA, held-out raw files,
digestion settings and modification settings remain unchanged.
EOF
}

JOB_DIR=""
MODEL_ID=""
LIBRARY_FILE=""
SEARCH_ID=""
EXECUTOR="${EXTERNAL_MODEL_DEFAULT_EXECUTOR:-cluster}"
THREADS=""
PREPARE_ONLY=0
TEST_MODE=0
while (($#)); do
  case "$1" in
    --job) JOB_DIR="${2:-}"; shift 2 ;;
    --model-id) MODEL_ID="${2:-}"; shift 2 ;;
    --library|--lib) LIBRARY_FILE="${2:-}"; shift 2 ;;
    --search-id) SEARCH_ID="${2:-}"; shift 2 ;;
    --executor) EXECUTOR="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --test) TEST_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$JOB_DIR" ]] || die "--job is required"
JOB_DIR="$(realpath -e "$JOB_DIR")" || die "Prepared Carafe job does not exist: $JOB_DIR"
[[ -f "$JOB_DIR/manifest.yaml" ]] || die "Not a prepared Carafe job: $JOB_DIR"
[[ "$MODEL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--model-id must be a safe registry ID"
[[ -z "$SEARCH_ID" || "$SEARCH_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid --search-id"
[[ "$EXECUTOR" == "cluster" || "$EXECUTOR" == "direct" ]] || die "--executor must be cluster or direct"
[[ -z "$THREADS" || "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"

SOURCE_COMMAND_FILE="$JOB_DIR/meta/source_step5_baseline_command.txt"
[[ -s "$SOURCE_COMMAND_FILE" ]] || die "Recorded baseline Step-5 command is missing: $SOURCE_COMMAND_FILE"
SOURCE_COMMAND="$(<"$SOURCE_COMMAND_FILE")"
[[ -f "$JOB_DIR/meta/source_experiment_dir.txt" ]] || die "Source-experiment metadata is missing"
SOURCE_EXP_DIR="$(<"$JOB_DIR/meta/source_experiment_dir.txt")"
[[ -d "$SOURCE_EXP_DIR" ]] || die "Source experiment is unavailable: $SOURCE_EXP_DIR"
EXPERIMENT="$(basename "$SOURCE_EXP_DIR")"
PROFILE="$(basename "$(dirname "$JOB_DIR")")"
[[ "$PROFILE" == "balanced" || "$PROFILE" == "fast" || "$PROFILE" == "slow" ]] \
  || die "Cannot derive Carafe profile from job path: $JOB_DIR"

FASTA_FILE="$(<"$JOB_DIR/meta/fasta_source_path.txt")"
[[ -f "$FASTA_FILE" ]] || die "Recorded FASTA is unavailable: $FASTA_FILE"
if [[ -z "$LIBRARY_FILE" ]]; then
  [[ -s "$JOB_DIR/meta/carafe_library_path.txt" ]] || die "Carafe library record is missing; run Carafe inference first"
  LIBRARY_FILE="$(<"$JOB_DIR/meta/carafe_library_path.txt")"
fi
LIBRARY_FILE="$(realpath -e "$LIBRARY_FILE")" || die "Carafe library does not exist: $LIBRARY_FILE"
[[ -s "$LIBRARY_FILE" ]] || die "Carafe library is empty: $LIBRARY_FILE"

mapfile -t RECORDED_TEST_FILES < <(python3 - "$SOURCE_COMMAND" <<'PY'
import shlex, sys
args = shlex.split(sys.argv[1])
for i, value in enumerate(args[:-1]):
    if value == "--f":
        print(args[i + 1])
PY
)
mapfile -t LOCAL_TEST_FILES < <(python3 - "$JOB_DIR/meta/test_mzml_source_paths.txt" <<'PY'
import pathlib, shlex, sys
path = pathlib.Path(sys.argv[1])
files = [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
for path in files:
    print(path)
PY
)
(( ${#LOCAL_TEST_FILES[@]} > 0 )) || die "No held-out acquisition files are recorded in the prepared job"
for file in "${LOCAL_TEST_FILES[@]}"; do
  [[ -f "$file" ]] || die "Held-out acquisition file is unavailable: $file"
done
# Locally relocated paths are used on analysis host. For a real cluster executor the
# original /pfs paths are the appropriate namespace, provided they were
# recorded. This keeps both sides reproducible without guessing mount aliases.
if [[ "$EXECUTOR" == "cluster" && ${#RECORDED_TEST_FILES[@]} -gt 0 ]]; then
  TEST_FILES=("${RECORDED_TEST_FILES[@]}")
else
  TEST_FILES=("${LOCAL_TEST_FILES[@]}")
fi

DIGESTION_CONTRACT="$JOB_DIR/meta/digestion_contract.env"
[[ -f "$DIGESTION_CONTRACT" ]] || die "Digestion contract is missing: $DIGESTION_CONTRACT"
# shellcheck source=/dev/null
source "$DIGESTION_CONTRACT"

if [[ "$EXECUTOR" == "cluster" ]]; then
  PIPE_ROOT="${EXTERNAL_MODEL_CLUSTER_PIPELINE_ROOT:-}"
  DIANN_EXECUTABLE="${EXTERNAL_MODEL_CLUSTER_DIANN_BIN:-}"
  SUBMIT_COMMAND="${EXTERNAL_MODEL_CLUSTER_SUBMIT_COMMAND:-}"
else
  PIPE_ROOT="$ROOT"
  DIANN_EXECUTABLE="${EXTERNAL_MODEL_DIRECT_DIANN_BIN:-$DIANN_BIN}"
  SUBMIT_COMMAND=""
fi
[[ -n "$PIPE_ROOT" && -n "$DIANN_EXECUTABLE" ]] || die "Executor '$EXECUTOR' is not fully configured"
[[ "$EXECUTOR" != "cluster" || -n "$SUBMIT_COMMAND" ]] || die "Cluster submit command is not configured"

MOD_CONFIG="$(python3 - "$SOURCE_COMMAND" <<'PY'
import shlex, sys
args = shlex.split(sys.argv[1])
sets = [args[i + 1] for i, value in enumerate(args[:-1]) if value == "--set"]
mods = [value.split("=", 1)[1] for value in sets if value.startswith("MOD_CONFIG=")]
print(mods[-1] if mods else "fixed_lysin_dimethyl")
PY
)"
mapfile -t SOURCE_SETS < <(python3 - "$SOURCE_COMMAND" <<'PY'
import shlex, sys
args = shlex.split(sys.argv[1])
for i, value in enumerate(args[:-1]):
    if value == "--set":
        print(args[i + 1])
PY
)

SEARCH_ID="${SEARCH_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="$SOURCE_EXP_DIR/external_carafe2_searches/$MODEL_ID/$SEARCH_ID"
[[ ! -e "$RESULT_DIR" ]] || die "Search result already exists: $RESULT_DIR"
CENTRAL_PROGRESS_FILE="${EXTERNAL_MODELS_ROOT%/}/carafe2/progress.txt"

STEP5=(
  bash "$PIPE_ROOT/bin/step5_quant.sh"
  --expname "$SEARCH_ID"
  --fastaabs "$FASTA_FILE"
  --lib "$LIBRARY_FILE"
  --out-dir "$RESULT_DIR/05_quant"
  --temp-dir "$RESULT_DIR/tmp"
  --log-dir "$RESULT_DIR/logs"
  --set "DIANN_BIN=$DIANN_EXECUTABLE"
  --set "MOD_CONFIG=$MOD_CONFIG"
  --set "CUT=$DIANN_CUT"
  --set "MISSED_CLEAVAGES=$DIANN_MISSED_CLEAVAGES"
  --set "MIN_PEP_LEN=$DIANN_MIN_PEP_LEN"
  --set "MAX_PEP_LEN=$DIANN_MAX_PEP_LEN"
  --set "MIN_PR_MZ=$DIANN_MIN_PR_MZ"
  --set "MAX_PR_MZ=$DIANN_MAX_PR_MZ"
  --set "enable_SEMI=$DIANN_SEMI"
)
for file in "${TEST_FILES[@]}"; do STEP5+=(--f "$file"); done
for setting in "${SOURCE_SETS[@]}"; do
  case "${setting%%=*}" in
    DIANN_BIN|MOD_CONFIG|CUT|MISSED_CLEAVAGES|MIN_PEP_LEN|MAX_PEP_LEN|MIN_PR_MZ|MAX_PR_MZ|enable_SEMI) continue ;;
    threads) setting="THREADS=${setting#*=}" ;;
  esac
  STEP5+=(--set "$setting")
done
[[ -n "$THREADS" ]] && STEP5+=(--set "THREADS=$THREADS")

if [[ "$TEST_MODE" == "1" ]]; then
  echo "TEST MODE: no files are written and no job is submitted."
  printf 'Model ID      : %s\nExperiment    : %s\nProfile       : %s\nLibrary       : %s\nExecutor      : %s\nOutput        : %s\nHeld-out files: %s\n' \
    "$MODEL_ID" "$SOURCE_EXP_DIR" "$PROFILE" "$LIBRARY_FILE" "$EXECUTOR" "$RESULT_DIR" "${#TEST_FILES[@]}"
  printf 'Step 5: '; printf '%q ' "${STEP5[@]}"; echo
  [[ "$EXECUTOR" == "cluster" ]] && echo "Submit: $SUBMIT_COMMAND <prepared-worker>"
  exit 0
fi

mkdir -p "$RESULT_DIR/commands" "$RESULT_DIR/logs"
ENV_FILE="$RESULT_DIR/resolved_config.env"
{
  printf 'RESULT_DIR='; q "$RESULT_DIR"; echo
  printf 'CENTRAL_PROGRESS_FILE='; q "$CENTRAL_PROGRESS_FILE"; echo
  printf 'MODEL_ID='; q "$MODEL_ID"; echo
  printf 'EXPERIMENT='; q "$EXPERIMENT"; echo
  printf 'SEARCH_ID='; q "$SEARCH_ID"; echo
  printf 'LIBRARY_FILE='; q "$LIBRARY_FILE"; echo
  printf 'STEP5=('; printf ' %q' "${STEP5[@]}"; echo ' )'
} > "$ENV_FILE"
printf '%q ' "${STEP5[@]}" > "$RESULT_DIR/commands/step5.sh"
echo >> "$RESULT_DIR/commands/step5.sh"
cp "$ROOT/bin/run_external_carafe2_library_search_worker.sh" "$RESULT_DIR/worker.sh"
chmod +x "$RESULT_DIR/worker.sh"
python3 - "$RESULT_DIR/manifest.json" "$MODEL_ID" "$EXPERIMENT" "$SOURCE_EXP_DIR" "$PROFILE" "$LIBRARY_FILE" "$FASTA_FILE" "$EXECUTOR" "${TEST_FILES[@]}" <<'PY'
import datetime, json, pathlib, socket, sys
out, model, experiment, source, profile, library, fasta, executor, *files = sys.argv[1:]
data = {
    "model_id": model, "experiment": experiment, "source_experiment": source,
    "profile": profile, "library": library, "fasta": fasta,
    "held_out_files": files, "executor": executor, "host": socket.gethostname(),
    "created_at": datetime.datetime.now().astimezone().isoformat(),
}
pathlib.Path(out).write_text(json.dumps(data, indent=2) + "\n")
PY
printf '{"state":"prepared"}\n' > "$RESULT_DIR/status.json"

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Prepared cluster search: $ENV_FILE"
  exit 0
fi
if [[ "$EXECUTOR" == "direct" || "${EXTERNAL_MODEL_DIRECT_EXECUTION:-0}" == "1" ]]; then
  bash "$RESULT_DIR/worker.sh" "$ENV_FILE"
else
  read -r -a SUBMIT_ARGS <<< "$SUBMIT_COMMAND"
  "${SUBMIT_ARGS[@]}" "$RESULT_DIR/worker.sh" "$ENV_FILE" | tee "$RESULT_DIR/logs/submission.log"
  printf '{"state":"submitted"}\n' > "$RESULT_DIR/status.json"
fi
