#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for shared Carafe jobs. It resolves the long shared job
# path from the selected profile and delegates all validation/execution to the
# canonical runner.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "ERROR: $*" >&2; exit 1; }

PROFILE=""
EXPERIMENT=""
PIPELINE_VERSION="2_5_0"
JOB_NAME="carafe_v1"
MACHINE_CONFIG="$SCRIPT_DIR/config/carafe_machines/gpu_pc.local.cfg"
TEST_MODE=0
GPU_REQUEST=""
RESUME_REQUESTED=0
SHOW_COMMAND=0
FINE_TUNE_ONLY=0
FORCE_FINE_TUNE=0
SAVE_VALIDATION_CHECKPOINTS=0
MS2_EPOCHS=""
RT_EPOCHS=""
FINE_TUNE_RUN=""
EXTERNAL_MODEL_DIR=""

usage() {
  cat <<EOF
Usage:
  bash run_carafe_profile.sh \\
    [--pipeline-version <version>] \\
    --experiment <name> \\
    --job <balanced|fast|slow>/<job_name> [options]

Options:
  --job <profile>/<job_name>    Unique job selector, e.g. balanced/carafe_v1
  --profile <name>              Profile when --job is not used
  --name <job_name>             Job name when --job is not used (default: ${JOB_NAME})
  --experiment <name>           DIA-NN experiment (required)
  --pipeline-version <version>  Pipeline version (default: ${PIPELINE_VERSION})
  --machine <name>              Machine config name (default: gpu_pc.local)
  --machine-config <path>       Explicit local machine config
  --gpu <index[,index...]>      Set CUDA_VISIBLE_DEVICES for this run
  --resume                      Assemble from a complete validated prediction checkpoint
  --fine-tune-only               Prepare training data and fine-tune; skip library inference and assembly
  --force-fine-tune              Archive prior fine-tuning results and train RT/CCS/MS2 from scratch
  --validation-checkpoints       Save RT/CCS/MS2 snapshots after every validation evaluation
  --ms2-epochs <n>               MS2 fine-tuning epochs (default: Carafe's 20)
  --rt-epochs <n>                RT/CCS fine-tuning epochs (default: Carafe's 40)
  --fine-tune-run <name>         Store this fine-tuning variant in a separate named output directory
  --external-model-dir <path>    Use an existing Carafe RT model for inference; never fine-tune
  --test                        Validate and print a compact execution plan only
  --show-command                With --test, print the full Java command
EOF
}

ARGS=("$@")
JOB_SELECTOR=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --job) ((i += 1)); JOB_SELECTOR="${ARGS[$i]:-}" ;;
    --profile) ((i += 1)); PROFILE="${ARGS[$i]:-}" ;;
    --name) ((i += 1)); JOB_NAME="${ARGS[$i]:-}" ;;
    --experiment) ((i += 1)); EXPERIMENT="${ARGS[$i]:-}" ;;
    --pipeline-version|--version) ((i += 1)); PIPELINE_VERSION="${ARGS[$i]:-}" ;;
    --machine) ((i += 1)); MACHINE_CONFIG="$SCRIPT_DIR/config/carafe_machines/${ARGS[$i]:-}.cfg" ;;
    --machine-config) ((i += 1)); MACHINE_CONFIG="${ARGS[$i]:-}" ;;
    --gpu) ((i += 1)); GPU_REQUEST="${ARGS[$i]:-}" ;;
    --resume) RESUME_REQUESTED=1 ;;
    --fine-tune-only) FINE_TUNE_ONLY=1 ;;
    --force-fine-tune) FORCE_FINE_TUNE=1 ;;
    --validation-checkpoints) SAVE_VALIDATION_CHECKPOINTS=1 ;;
    --ms2-epochs) ((i += 1)); MS2_EPOCHS="${ARGS[$i]:-}" ;;
    --rt-epochs) ((i += 1)); RT_EPOCHS="${ARGS[$i]:-}" ;;
    --fine-tune-run) ((i += 1)); FINE_TUNE_RUN="${ARGS[$i]:-}" ;;
    --external-model-dir) ((i += 1)); EXTERNAL_MODEL_DIR="${ARGS[$i]:-}" ;;
    --show-command) SHOW_COMMAND=1 ;;
    --test) TEST_MODE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: ${ARGS[$i]}" ;;
  esac
done

if [[ -n "$JOB_SELECTOR" ]]; then
  [[ "$JOB_SELECTOR" == */* ]] || die "--job must be <profile>/<job_name>"
  [[ "$JOB_SELECTOR" != */*/* ]] || die "--job must contain exactly one slash"
  [[ -z "$PROFILE" ]] || die "Use either --job or --profile/--name"
  PROFILE="${JOB_SELECTOR%%/*}"
  JOB_NAME="${JOB_SELECTOR#*/}"
fi

case "$PROFILE" in
  balanced|fast|slow) ;;
  *) die "--profile must be balanced, fast, or slow" ;;
esac
[[ -n "$JOB_NAME" ]] || die "Job name is missing"
[[ -n "$EXPERIMENT" ]] || die "--experiment is required"
[[ -f "$MACHINE_CONFIG" ]] || die "Carafe machine config not found: $MACHINE_CONFIG"
[[ "$FINE_TUNE_ONLY" == "0" || "$RESUME_REQUESTED" == "0" ]] || die "--fine-tune-only cannot be combined with --resume"
[[ "$FORCE_FINE_TUNE" == "0" || "$FINE_TUNE_ONLY" == "1" ]] || die "--force-fine-tune requires --fine-tune-only"
[[ -z "$MS2_EPOCHS" || "$MS2_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--ms2-epochs must be a positive integer"
[[ -z "$RT_EPOCHS" || "$RT_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--rt-epochs must be a positive integer"
[[ -z "$FINE_TUNE_RUN" || "$FINE_TUNE_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--fine-tune-run must contain only letters, digits, dot, underscore, or hyphen"
[[ -z "$FINE_TUNE_RUN" || "$FINE_TUNE_ONLY" == "1" ]] || die "--fine-tune-run requires --fine-tune-only"
[[ -z "$EXTERNAL_MODEL_DIR" || "$FINE_TUNE_ONLY" == "0" ]] || die "--external-model-dir cannot be combined with --fine-tune-only"
[[ -z "$EXTERNAL_MODEL_DIR" || "$FORCE_FINE_TUNE" == "0" ]] || die "--external-model-dir cannot be combined with --force-fine-tune"
[[ -z "$EXTERNAL_MODEL_DIR" || "$RESUME_REQUESTED" == "0" ]] || die "--external-model-dir cannot be combined with --resume"

set -a
# shellcheck source=/dev/null
source "$MACHINE_CONFIG"
set +a
: "${CARAFE_SHARE_ROOT:?CARAFE_SHARE_ROOT is missing in $MACHINE_CONFIG}"

JOB_DIR="${CARAFE_SHARE_ROOT%/}/${PIPELINE_VERSION}/${EXPERIMENT}/carafe/${PROFILE}/${JOB_NAME}"
RUNNER_ARGS=(--job "$JOB_DIR" --machine-config "$MACHINE_CONFIG")
[[ -n "$GPU_REQUEST" ]] && RUNNER_ARGS+=(--gpu "$GPU_REQUEST")
[[ "$RESUME_REQUESTED" == "1" ]] && RUNNER_ARGS+=(--resume)
[[ "$FINE_TUNE_ONLY" == "1" ]] && RUNNER_ARGS+=(--fine-tune-only)
[[ "$FORCE_FINE_TUNE" == "1" ]] && RUNNER_ARGS+=(--force-fine-tune)
[[ "$SAVE_VALIDATION_CHECKPOINTS" == "1" ]] && RUNNER_ARGS+=(--validation-checkpoints)
[[ -n "$MS2_EPOCHS" ]] && RUNNER_ARGS+=(--ms2-epochs "$MS2_EPOCHS")
[[ -n "$RT_EPOCHS" ]] && RUNNER_ARGS+=(--rt-epochs "$RT_EPOCHS")
[[ -n "$FINE_TUNE_RUN" ]] && RUNNER_ARGS+=(--fine-tune-run "$FINE_TUNE_RUN")
[[ -n "$EXTERNAL_MODEL_DIR" ]] && RUNNER_ARGS+=(--external-model-dir "$EXTERNAL_MODEL_DIR")
[[ "$SHOW_COMMAND" == "1" ]] && RUNNER_ARGS+=(--show-command)
if [[ "$TEST_MODE" == "1" ]]; then
  RUNNER_ARGS+=(--test)
fi
exec bash "$SCRIPT_DIR/run_carafe_job.sh" "${RUNNER_ARGS[@]}"
