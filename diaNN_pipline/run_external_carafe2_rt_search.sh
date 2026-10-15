#!/usr/bin/env bash
# End-to-end launcher: validation -> Carafe RT-only library -> DIA-NN Step 5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$ROOT/config/paths.cfg"
die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  bash run_external_carafe2_rt_search.sh \
    --experiment <name> --model-id <carafe2rt_id> [options]

Options:
  --profile balanced|fast|slow  Existing DIA-NN profile (default: balanced)
  --pipeline-version <version> Experiment version directory (default: 2_5_0)
  --search-id <id>             Stable run name (default: timestamp)
  --executor cluster|direct   DIA-NN Step-5 executor (default: cluster)
  --threads <n>                Override DIA-NN Step-5 threads
  --machine-config <path>      Carafe runtime configuration
  --carafe-device cpu|gpu      Device for the inference-only library (default: cpu)
  --gpu <index[,index...]>     GPU index/list for Carafe; requires --carafe-device gpu
  --validation-root <path>     RT partition root containing <target>/validation
  --skip-rt-validation         Skip the CPU validation gate (recorded in plan)
  --prepare-only               Prepare the portable Carafe job, then stop
  --library-only               Validate, prepare and generate the library; do not search
  --test                       Read-only validation and exact execution plan

Default workflow:
  1. Evaluate the registered RT model on the untouched validation partition.
  2. Prepare inputs from the existing DIA-NN experiment.
  3. Generate a Carafe library on CPU without any fine-tuning. Only the external
     rt_model.pt is exposed; MS2 and CCS use the fixed generic Carafe models.
  4. Submit DIA-NN Step 5 with the original FASTA, held-out files and settings.
EOF
}

EXPERIMENT=""
MODEL_ID=""
PROFILE="balanced"
PIPELINE_VERSION="2_5_0"
SEARCH_ID=""
EXECUTOR="${EXTERNAL_MODEL_DEFAULT_EXECUTOR:-cluster}"
THREADS=""
MACHINE_CONFIG="$ROOT/config/carafe_machines/cpu_server.local.cfg"
CARAFE_DEVICE="cpu"
GPU_REQUEST=""
VALIDATION_ROOT="$(cd "$ROOT/../.." && pwd)/trainingdata/finalrt_4source_trainfit15_validation10_v1"
SKIP_VALIDATION=0
PREPARE_ONLY=0
LIBRARY_ONLY=0
TEST_MODE=0
while (($#)); do
  case "$1" in
    --experiment) EXPERIMENT="${2:-}"; shift 2 ;;
    --model-id) MODEL_ID="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --pipeline-version|--version) PIPELINE_VERSION="${2:-}"; shift 2 ;;
    --search-id) SEARCH_ID="${2:-}"; shift 2 ;;
    --executor) EXECUTOR="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --machine-config) MACHINE_CONFIG="${2:-}"; shift 2 ;;
    --carafe-device) CARAFE_DEVICE="${2:-}"; shift 2 ;;
    --gpu) GPU_REQUEST="${2:-}"; shift 2 ;;
    --validation-root) VALIDATION_ROOT="${2:-}"; shift 2 ;;
    --skip-rt-validation) SKIP_VALIDATION=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --library-only) LIBRARY_ONLY=1; shift ;;
    --test) TEST_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ "$EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--experiment is required and must be a safe name"
[[ "$MODEL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--model-id is required and must be a safe ID"
[[ "$PROFILE" == "balanced" || "$PROFILE" == "fast" || "$PROFILE" == "slow" ]] || die "Invalid --profile"
[[ "$PIPELINE_VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid --pipeline-version"
[[ -z "$SEARCH_ID" || "$SEARCH_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid --search-id"
[[ "$EXECUTOR" == "cluster" || "$EXECUTOR" == "direct" ]] || die "--executor must be cluster or direct"
[[ -z "$THREADS" || "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"
[[ "$PREPARE_ONLY" == "0" || "$LIBRARY_ONLY" == "0" ]] || die "Use either --prepare-only or --library-only"
[[ -f "$MACHINE_CONFIG" ]] || die "Carafe machine config is missing: $MACHINE_CONFIG"
[[ "$CARAFE_DEVICE" == "cpu" || "$CARAFE_DEVICE" == "gpu" ]] || die "--carafe-device must be cpu or gpu"
[[ -z "$GPU_REQUEST" || "$GPU_REQUEST" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpu must be one GPU index or a comma-separated list"
[[ -z "$GPU_REQUEST" || "$CARAFE_DEVICE" == "gpu" ]] || die "--gpu requires --carafe-device gpu"

: "${EXTERNAL_MODELS_ROOT:?EXTERNAL_MODELS_ROOT is not configured}"
REGISTRY_ROOT="${EXTERNAL_MODELS_ROOT%/}/carafe2"
REGISTRY="$REGISTRY_ROOT/models_list.csv"
[[ -f "$REGISTRY" ]] || die "Carafe2 model registry is missing: $REGISTRY"
mapfile -t MODEL_META < <(python3 - "$REGISTRY" "$MODEL_ID" <<'PY'
import csv, sys
registry, wanted = sys.argv[1:]
with open(registry, newline="", encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle) if row.get("model_id") == wanted]
if len(rows) != 1:
    raise SystemExit(f"expected exactly one registry row for {wanted!r}, found {len(rows)}")
row = rows[0]
for key in ("package_name", "relative_path", "target_space", "rt_model_sha256", "status"):
    if not row.get(key):
        raise SystemExit(f"registry row has no {key}")
if row.get("model_family") != "Carafe2":
    raise SystemExit("registry row is not a Carafe2 model")
for key in ("package_name", "relative_path", "target_space", "rt_model_sha256", "status"):
    print(row[key])
PY
) || die "Could not resolve Carafe2 model '$MODEL_ID'"
PACKAGE_NAME="${MODEL_META[0]}"
MODEL_REL="${MODEL_META[1]}"
TARGET_SPACE="${MODEL_META[2]}"
EXPECTED_SHA256="${MODEL_META[3]}"
MODEL_STATUS="${MODEL_META[4]}"
[[ "$MODEL_REL" != /* && "$MODEL_REL" != *".."* ]] || die "Unsafe model registry path: $MODEL_REL"
MODEL_DIR="$REGISTRY_ROOT/models/$MODEL_REL"
[[ -s "$MODEL_DIR/rt_model.pt" ]] || die "Registered RT checkpoint is missing: $MODEL_DIR/rt_model.pt"
ACTUAL_SHA256="$(sha256sum "$MODEL_DIR/rt_model.pt" | awk '{print $1}')"
[[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]] || die "RT checkpoint checksum differs from the registry"

PARTITION_DIR="$VALIDATION_ROOT/$TARGET_SPACE/validation"
ALIGNMENT_FILE="$VALIDATION_ROOT/$TARGET_SPACE/rt_alignment.json"
if [[ "$SKIP_VALIDATION" == "0" ]]; then
  [[ "$MODEL_STATUS" == "ready_for_cpu_validation" ]] \
    || die "Model status is '$MODEL_STATUS', expected ready_for_cpu_validation"
  [[ -f "$PARTITION_DIR/rt_observations.tsv" ]] || die "Validation partition is missing: $PARTITION_DIR"
  [[ -f "$ALIGNMENT_FILE" ]] || die "RT alignment is missing: $ALIGNMENT_FILE"
fi

SEARCH_ID="${SEARCH_ID:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="external_${MODEL_ID}_${SEARCH_ID}"
# shellcheck source=/dev/null
source "$MACHINE_CONFIG"
: "${CARAFE_SHARE_ROOT:?CARAFE_SHARE_ROOT is missing in $MACHINE_CONFIG}"
JOB_DIR="${CARAFE_SHARE_ROOT%/}/$PIPELINE_VERSION/$EXPERIMENT/carafe/$PROFILE/$JOB_NAME"
EVALUATION_DIR="$REGISTRY_ROOT/evaluations/$MODEL_ID/${EXPERIMENT}_${SEARCH_ID}_validation"

PREPARE_CMD=(
  bash "$ROOT/prepare_carafe_job.sh"
  --source-exp "$EXPERIMENT"
  --profile "$PROFILE"
  --pipeline-version "$PIPELINE_VERSION"
  --name "$JOB_NAME"
  --set "CARAFE_DEVICE=$CARAFE_DEVICE"
  --set "CARAFE_TORCH_COMPILE=0"
  --set "CARAFE_CCS=0"
)
RUN_CARAFE_CMD=(
  bash "$ROOT/run_carafe_job.sh"
  --job "$JOB_DIR"
  --machine-config "$MACHINE_CONFIG"
  --external-model-dir "$JOB_DIR/external_rt_model"
)
[[ -n "$GPU_REQUEST" ]] && RUN_CARAFE_CMD+=(--gpu "$GPU_REQUEST")
SEARCH_CMD=(
  bash "$ROOT/run_external_carafe2_library_search.sh"
  --job "$JOB_DIR"
  --model-id "$MODEL_ID"
  --search-id "$SEARCH_ID"
  --executor "$EXECUTOR"
)
[[ -n "$THREADS" ]] && SEARCH_CMD+=(--threads "$THREADS")

if [[ "$TEST_MODE" == "1" ]]; then
  echo "TEST MODE: read-only validation; nothing is created or submitted."
  printf 'Experiment       : %s (%s; %s)\nModel            : %s\nModel package    : %s\nRT target space  : %s\nRT checksum      : %s\nCPU validation   : %s\nCarafe job       : %s\nRT-only runtime  : %s\nDIA-NN executor  : %s\n' \
    "$EXPERIMENT" "$PIPELINE_VERSION" "$PROFILE" "$MODEL_ID" "$MODEL_DIR" "$TARGET_SPACE" "$ACTUAL_SHA256" \
    "$([[ "$SKIP_VALIDATION" == 1 ]] && echo skipped || echo "$PARTITION_DIR")" "$JOB_DIR" \
    "external rt_model.pt + generic MS2/CCS ($CARAFE_DEVICE${GPU_REQUEST:+, GPU $GPU_REQUEST})" "$EXECUTOR"
  echo "Prepare inputs:"
  printf '  %q ' "${PREPARE_CMD[@]}"; echo ' --test'
  echo "Generate library:"
  printf '  %q ' "${RUN_CARAFE_CMD[@]}"; echo ' --test --show-command'
  echo "Submit held-out DIA-NN Step 5 after the library exists:"
  printf '  %q ' "${SEARCH_CMD[@]}"; echo ' --test'
  "${PREPARE_CMD[@]}" --test
  exit 0
fi

if [[ "$SKIP_VALIDATION" == "0" ]]; then
  "$CARAFE_PYTHON" "$ROOT/../data_collect/evaluate_external_carafe2_rt_models.py" \
    --models-list "$REGISTRY" \
    --model "$PACKAGE_NAME" \
    --partition-dir "$PARTITION_DIR" \
    --alignment "$ALIGNMENT_FILE" \
    --target-space "$TARGET_SPACE" \
    --output-dir "$EVALUATION_DIR" \
    --python "$CARAFE_PYTHON"
fi

"${PREPARE_CMD[@]}"
mkdir -p "$JOB_DIR/external_rt_model"
[[ ! -e "$JOB_DIR/external_rt_model/rt_model.pt" ]] \
  || die "Prepared RT-only runtime already contains a checkpoint: $JOB_DIR/external_rt_model/rt_model.pt"
ln -s "$MODEL_DIR/rt_model.pt" "$JOB_DIR/external_rt_model/rt_model.pt"
# Carafe 2.2 reads meta.json unconditionally when --model_dir is supplied.
# This metadata describes the *current library acquisition*, not the external
# checkpoint, so generate it in the per-job runtime rather than packaging a
# stale copy with a model trained on another source.
python3 - "$JOB_DIR/external_rt_model/meta.json" \
  "${CARAFE_MS_INSTRUMENT:-Exploris}" "${CARAFE_NCE:-28}" \
  "${CARAFE_MIN_FRAGMENT_MZ:-200}" "${CARAFE_LIBRARY_MAX_FRAGMENT_MZ:-1800}" \
  "${CARAFE_LIBRARY_MIN_FRAGMENT_MZ:-200}" "${CARAFE_LIBRARY_MAX_FRAGMENT_MZ:-1800}" \
  "${CARAFE_LIBRARY_TOP_N_FRAGMENTS:-20}" "${CARAFE_MAX_RT:-0}" \
  "${CARAFE_MIN_PRECURSOR_MZ:-300}" "${CARAFE_MAX_PRECURSOR_MZ:-1800}" <<'PY'
import json, pathlib, sys
out, instrument, nce, fragment_min, fragment_max, library_min, library_max, top_n, rt_max, precursor_min, precursor_max = sys.argv[1:]
def number(value):
    return float(value) if value else 0.0
data = {"external_runtime": {
    "ms_file": "external_runtime",
    "ms_instrument": instrument or "Exploris",
    "nce": number(nce) or 28.0,
    "min_fragment_ion_mz": number(fragment_min),
    "max_fragment_ion_mz": number(fragment_max),
    "lf_frag_mz_min": number(library_min),
    "lf_frag_mz_max": number(library_max),
    "lf_top_n_fragment_ions": int(float(top_n)),
    "rt_min": 0.0,
    "rt_max": number(rt_max),
    "precursor_ion_mz_min": number(precursor_min),
    "precursor_ion_mz_max": number(precursor_max),
}}
pathlib.Path(out).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
python3 - "$JOB_DIR/meta/external_rt_model.json" "$MODEL_ID" "$MODEL_DIR" "$ACTUAL_SHA256" "$TARGET_SPACE" "$SKIP_VALIDATION" "$EVALUATION_DIR" <<'PY'
import datetime, json, pathlib, sys
out, model_id, model_dir, checksum, target, skipped, evaluation = sys.argv[1:]
data = {
    "model_id": model_id, "source_model_dir": model_dir, "rt_model_sha256": checksum,
    "target_space": target, "rt_validation_skipped": skipped == "1",
    "rt_validation_output": None if skipped == "1" else evaluation,
    "ms2_model": "generic", "ccs_model": "generic", "fine_tuning": False,
    "carafe_runtime_meta": "external_rt_model/meta.json",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
}
pathlib.Path(out).write_text(json.dumps(data, indent=2) + "\n")
PY

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Prepared external Carafe2 RT job: $JOB_DIR"
  exit 0
fi

"${RUN_CARAFE_CMD[@]}"
if [[ "$LIBRARY_ONLY" == "1" ]]; then
  echo "Carafe library ready: $(<"$JOB_DIR/meta/carafe_library_path.txt")"
  exit 0
fi

"${SEARCH_CMD[@]}"
echo "Workflow started/completed for model $MODEL_ID. Job: $JOB_DIR"
