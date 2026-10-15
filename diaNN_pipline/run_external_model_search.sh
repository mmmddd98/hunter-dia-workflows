#!/usr/bin/env bash
# Registry based launcher for an RT-only external DIA-NN model search.
# It never changes the source experiment: Step 4/5 results live below the model.
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config/paths.cfg"
source "$ROOT/bin/_lib.sh"

# Delegate repeated experiments to the bounded-concurrency array launcher.  A
# single experiment keeps the simple direct submission path below.
exp_count=0; parallel_requested=0
for arg in "$@"; do
  [[ "$arg" == --experiment ]] && ((exp_count+=1)) || true
  [[ "$arg" == --parallel ]] && parallel_requested=1
done
if (( exp_count > 1 || parallel_requested )); then
  exec bash "$ROOT/bin/run_external_model_search_batch.sh" "$@"
fi

die(){ echo "ERROR: $*" >&2; exit 1; }
usage(){ cat <<'EOF'
Usage: bash run_external_model_search.sh --experiment NAME --model-id ID
       [--search-id ID] [--executor cluster|direct] [--threads N] [--resume]
       [--parallel N] [--ram-log|--no-ram-log]
       [--ram-log-interval SECONDS] [--test]

Repeat --experiment and pass --parallel N to submit a SLURM job array capped
at N concurrent searches.  One --model-id is shared by all experiments.  To
use different models, repeat --model-id once per --experiment; occurrences are
paired in the same order.  Threads are divided per experiment unless --threads
is specified explicitly.

The experiment must already exist.  The model ID is resolved from the DIA-NN
external-model registry.  Cluster is the default executor.  --test performs
all read-only validation and prints the resolved Step-4/Step-5 commands.
EOF
}
q(){ printf '%q' "$1"; }
json_escape(){ python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }
append_external_progress() {
  local step="$1" state="$2" detail="$3" ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  detail="${detail//$'\t'/ }"; detail="${detail//$'\n'/ }"
  mkdir -p "$(dirname "$CENTRAL_PROGRESS_FILE")"
  (
    flock -x 9
    [[ -s "$CENTRAL_PROGRESS_FILE" ]] || printf '# Central external DIA-NN model-search progress (timestamp<TAB>step<TAB>state<TAB>details)\n' >&9
    printf '%s\t%s\t%s\t%s\n' "$ts" "$step" "$state" "$detail" >&9
  ) 9>>"$CENTRAL_PROGRESS_FILE"
  # Each search retains an isolated record as well.  This makes the progress
  # file inside external_searches self-contained and avoids mixing it with the
  # source experiment's normal pipeline steps.
  if [[ -n "${RESULT_PROGRESS_FILE:-}" ]]; then
    mkdir -p "$(dirname "$RESULT_PROGRESS_FILE")"
    (
      flock -x 9
      [[ -s "$RESULT_PROGRESS_FILE" ]] || printf '# External DIA-NN model-search progress (timestamp<TAB>step<TAB>state<TAB>details)\n' >&9
      printf '%s\t%s\t%s\t%s\n' "$ts" "$step" "$state" "$detail" >&9
    ) 9>>"$RESULT_PROGRESS_FILE"
  fi
}

EXPERIMENT=""; MODEL_ID=""; SEARCH_ID=""; EXECUTOR="${EXTERNAL_MODEL_DEFAULT_EXECUTOR:-cluster}"; RESUME=0; TEST=0; THREADS_OVERRIDE=""; PREPARE_ONLY=0
RAM_LOG_ENABLE="${EXTERNAL_MODEL_RAM_LOG_ENABLE:-1}"
RAM_LOG_INTERVAL="${EXTERNAL_MODEL_RAM_LOG_INTERVAL:-10}"
while (($#)); do
  case "$1" in
    --experiment) EXPERIMENT="${2:-}"; shift 2;;
    --model-id) MODEL_ID="${2:-}"; shift 2;;
    --search-id) SEARCH_ID="${2:-}"; shift 2;;
    --executor) EXECUTOR="${2:-}"; shift 2;;
    --threads) THREADS_OVERRIDE="${2:-}"; shift 2;;
    --ram-log) RAM_LOG_ENABLE=1; shift;;
    --no-ram-log) RAM_LOG_ENABLE=0; shift;;
    --ram-log-interval) RAM_LOG_ENABLE=1; RAM_LOG_INTERVAL="${2:-}"; shift 2;;
    --resume) RESUME=1; shift;;
    --test) TEST=1; shift;;
    --prepare-only) PREPARE_ONLY=1; shift;;
    -h|--help) usage; exit 0;;
    *) die "Unknown argument: $1";;
  esac
done
[[ "$EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--experiment is required and must be a safe name"
[[ "$MODEL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--model-id is required and must be a safe ID"
[[ -z "$SEARCH_ID" || "$SEARCH_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid --search-id"
[[ "$EXECUTOR" == cluster || "$EXECUTOR" == direct ]] || die "--executor must be cluster or direct"
[[ -z "$THREADS_OVERRIDE" || "$THREADS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"
[[ "$RAM_LOG_ENABLE" =~ ^[01]$ ]] || die "RAM logging must be enabled (1) or disabled (0)"
[[ "$RAM_LOG_INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "--ram-log-interval must be a positive integer"

: "${EXTERNAL_MODELS_ROOT:?Set EXTERNAL_MODELS_ROOT in the machine config}"
REGISTRY="$EXTERNAL_MODELS_ROOT/diann/models_list.csv"
[[ -f "$REGISTRY" ]] || die "Model registry is missing: $REGISTRY"
CENTRAL_PROGRESS_FILE="$EXTERNAL_MODELS_ROOT/diann/progress.txt"

# Resolve exactly one CSV row, without assuming a particular column order beyond
# the documented model_id/relative_path headings.
mapfile -t REGISTRY_VALUES < <(python3 - "$REGISTRY" "$MODEL_ID" <<'PY'
import csv, sys
p, wanted = sys.argv[1:]
with open(p, newline='', encoding='utf-8') as f:
    rows=[r for r in csv.DictReader(f) if r.get('model_id') == wanted]
if len(rows) != 1:
    raise SystemExit(f"expected exactly one registry row for {wanted!r}, found {len(rows)}")
r=rows[0]
for k in ('relative_path','model_family'):
    if not r.get(k): raise SystemExit(f"registry row has no {k}")
print(r['relative_path']); print(r['model_family'])
PY
) || die "Could not resolve model '$MODEL_ID' exactly once in $REGISTRY"
MODEL_REL="${REGISTRY_VALUES[0]}"; MODEL_FAMILY="${REGISTRY_VALUES[1]}"
[[ "$MODEL_REL" != /* && "$MODEL_REL" != *".."* ]] || die "Unsafe registry relative_path: $MODEL_REL"
[[ "$MODEL_FAMILY" == "DIA-NN" ]] || die "Model '$MODEL_ID' is not a DIA-NN model (model_family=$MODEL_FAMILY)"
MODEL_DIR="$EXTERNAL_MODELS_ROOT/diann/$MODEL_REL"; PACKAGE="$MODEL_DIR/model_package"
for f in model/rt.d0.finetuned.pt model/dict.txt model_manifest.json checksums.sha256; do
  [[ -f "$PACKAGE/$f" ]] || die "Required model artifact is missing: $PACKAGE/$f"
done
(cd "$PACKAGE" && sha256sum -c checksums.sha256 >/dev/null) || die "Model checksum validation failed: $PACKAGE"
MODEL_VERSION="$(python3 - "$PACKAGE/model_manifest.json" "$MODEL_ID" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f: d=json.load(f)
if d.get('model_id') != sys.argv[2]: raise SystemExit('manifest model_id does not match registry ID')
if d.get('model_family') != 'DIA-NN': raise SystemExit('manifest is not DIA-NN')
v=d.get('diann_version_contract')
if not v: raise SystemExit('manifest has no diann_version_contract')
print(v)
PY
)" || die "Invalid model manifest: $PACKAGE/model_manifest.json"

# The original output may be on analysis host OUTBASE or on a mounted cluster share.
EXPDIR=""; EXP_CANDIDATES=()
[[ -d "${OUTBASE%/}/$EXPERIMENT" ]] && EXP_CANDIDATES+=("${OUTBASE%/}/$EXPERIMENT")
IFS=: read -r -a EXTRA_ROOTS <<< "${DIANN_ADDITIONAL_OUTROOTS:-}"
for root in "${EXTRA_ROOTS[@]}"; do
  [[ -n "$root" && -d "$root" ]] || continue
  while IFS= read -r candidate; do EXP_CANDIDATES+=("$candidate"); done < <(find "$root" -mindepth 2 -maxdepth 2 -type d -name "$EXPERIMENT" -print)
done
[[ ${#EXP_CANDIDATES[@]} -eq 1 ]] || die "Expected exactly one existing experiment '$EXPERIMENT' in configured output roots; found ${#EXP_CANDIDATES[@]}"
EXPDIR="${EXP_CANDIDATES[0]}"
[[ -f "$EXPDIR/run_all.pid.meta" || -f "$EXPDIR/all_startcmds.txt" || -f "$EXPDIR/progress.txt" ]] || die "Not a recorded DIA-NN experiment: $EXPDIR"
[[ -f "$EXPDIR/all_startcmds.txt" ]] || die "Cannot safely derive configuration: missing $EXPDIR/all_startcmds.txt"

# Extract the recorded run_all invocation and its planned baseline Step-5 command.
# Python shlex avoids sourcing experiment-controlled text as shell code.
mapfile -t PROV < <(python3 - "$EXPDIR" <<'PY'
import pathlib, shlex, sys
exp=pathlib.Path(sys.argv[1]); text=(exp/'all_startcmds.txt').read_text(encoding='utf-8',errors='replace').splitlines()
run=next((x for x in text if 'run_all.sh' in x and not x.lstrip().startswith('#')), None)
base=next((x for x in text if 'step5_quant.sh' in x and '--out-sub baseline' in x and not x.lstrip().startswith('#')), None)
if not run or not base: raise SystemExit('recorded run_all invocation or baseline Step-5 command is absent')
def args(line): return shlex.split(line)
ra, ba=args(run), args(base)
def one(a, flag, required=False):
    values=[a[i+1] for i,x in enumerate(a[:-1]) if x==flag]
    if len(values)>1: raise SystemExit(f'ambiguous {flag}')
    if required and len(values)!=1: raise SystemExit(f'missing {flag}')
    return values[0] if values else ''
ds=one(ra,'--ds',True); profile=one(ra,'--profile',True); mods=one(ra,'--mods')
fasta=next((str(p) for p in exp.glob('*.fasta') if p.is_file()), '')
if not fasta: raise SystemExit('no experiment-local FASTA found')
raws=[ba[i+1] for i,x in enumerate(ba[:-1]) if x=='--f']
if not raws: raise SystemExit('baseline Step-5 has no raw files')
sets=[ra[i+1] for i,x in enumerate(ra[:-1]) if x=='--set']
for x in (ds,profile,mods,fasta,*raws): print(x)
print('__END_RAW__')
for x in sets: print(x)
PY
) || die "Cannot derive an unambiguous recorded configuration from $EXPDIR/all_startcmds.txt"
DATASET="${PROV[0]}"; PROFILE="${PROV[1]}"; MODS="${PROV[2]}"; FASTA="${PROV[3]}"
raw_end=0; RAW_FILES=(); SETS=()
for ((i=4;i<${#PROV[@]};i++)); do
  [[ "${PROV[$i]}" == __END_RAW__ ]] && { raw_end=$i; break; }; RAW_FILES+=("${PROV[$i]}")
done
for ((i=raw_end+1;i<${#PROV[@]};i++)); do SETS+=("${PROV[$i]}"); done
[[ -f "$FASTA" ]] || die "Recorded experiment FASTA is unavailable: $FASTA"
for i in "${!RAW_FILES[@]}"; do
  f="${RAW_FILES[$i]}"
  [[ -f "$f" ]] && continue
  # Cluster provenance commonly records its former node-local path.  Resolve the
  # identical named raw from the configured shared dataset root, but only if it
  # is unique; ambiguity is a hard error rather than a guess.
  mapfile -t raw_candidates < <(find "$DATASET_ROOT" -type f -name "$(basename "$f")" -print 2>/dev/null)
  [[ ${#raw_candidates[@]} -eq 1 ]] || die "Recorded Step-5 raw file is unavailable/ambiguous: $f"
  RAW_FILES[$i]="${raw_candidates[0]}"
done

# Preserve the original selected fragment model.  Only the external RT model is replaced.
LR="$(python3 - "$EXPDIR/progress.txt" <<'PY'
import re,sys
try: t=open(sys.argv[1],encoding='utf-8').read()
except FileNotFoundError: raise SystemExit('missing progress.txt')
x=re.findall(r'step4_create_tuned_predlib\.sh.*OK.*profile=([^;]+);lr=([^;]+);.*models=([^;]+)',t)
# A recorded experiment can contain a later RT-only Step-4 attempt.  The
# external search must retain the original fragment model, so select the one
# completed provenance record that explicitly includes FR.
x=[record for record in x if 'fr' in record[2].split('+')]
if len(x)!=1: raise SystemExit(f'expected one completed Step-4 provenance record with an FR model, got {len(x)}')
p,lr,models=x[0]
print(p); print(lr)
PY
)" || die "Cannot derive a unique original fragment-model selection"
ORIG_PROFILE="$(printf '%s\n' "$LR" | sed -n '1p')"; ORIG_LR="$(printf '%s\n' "$LR" | sed -n '2p')"
[[ "$PROFILE" == "$ORIG_PROFILE" ]] || die "Recorded profile conflict: run_all=$PROFILE, Step-4=$ORIG_PROFILE"
lr_dir="${ORIG_LR//./p}"; FR_GLOB="$EXPDIR/03_tune_models/$PROFILE/lr_$lr_dir"/*.fr.d0.pt
shopt -s nullglob; fr_matches=($FR_GLOB); shopt -u nullglob
[[ ${#fr_matches[@]} -eq 1 ]] || die "Expected exactly one original fragment model in ${FR_GLOB%/*}"
FR_MODEL="${fr_matches[0]}"

SEARCH_ID="${SEARCH_ID:-$(date +%Y%m%d_%H%M%S)}"
# External runs belong to their source experiment so that the MA-plot server
# can discover them.  The result directory is also the only runtime directory:
# model package, Step 4, Step 5, logs, and temporary files all stay with the
# original experiment instead of being duplicated in a global scratch tree.
RESULT_DIR="$EXPDIR/external_searches/$MODEL_ID/$SEARCH_ID"
RESULT_PROGRESS_FILE="$RESULT_DIR/progress.txt"
if [[ -e "$RESULT_DIR" ]]; then
  if [[ "$RESUME" != 1 ]]; then
    die "Search already exists (refusing to overwrite): $RESULT_DIR"
  fi
  if [[ -f "$RESULT_DIR/status.json" ]] && grep -q '"complete"' "$RESULT_DIR/status.json"; then
    echo "Completed search already exists: $RESULT_DIR"
    exit 0
  else
    : # a clearly incomplete run may be resumed
  fi
fi

if [[ "$EXECUTOR" == cluster ]]; then PIPE_ROOT="${EXTERNAL_MODEL_CLUSTER_PIPELINE_ROOT:-$ROOT}"; DIANN_EXTERNAL="${EXTERNAL_MODEL_CLUSTER_DIANN_BIN:-}"; SUBMIT="${EXTERNAL_MODEL_CLUSTER_SUBMIT_COMMAND:-}";
else PIPE_ROOT="$ROOT"; DIANN_EXTERNAL="${EXTERNAL_MODEL_DIRECT_DIANN_BIN:-$DIANN_BIN}"; SUBMIT=""; fi
[[ -n "$DIANN_EXTERNAL" ]] || die "External-model DIA-NN setting is incomplete for executor '$EXECUTOR'"
[[ "$MODEL_VERSION" == "${EXTERNAL_MODEL_DIANN_VERSION:-$MODEL_VERSION}" ]] || die "Model requires DIA-NN $MODEL_VERSION but executor is configured for ${EXTERNAL_MODEL_DIANN_VERSION:-unknown}"
[[ "$EXECUTOR" != cluster || -n "$SUBMIT" ]] || die "EXTERNAL_MODEL_CLUSTER_SUBMIT_COMMAND is not configured"
INSTRUMENT="$(grep -hoE -- '--set[[:space:]]+INSTRUMENT=[^[:space:]]+' "$EXPDIR/all_startcmds.txt" | head -1 | sed 's/.*INSTRUMENT=//' || true)"
[[ -n "$INSTRUMENT" ]] || die "Could not derive the original instrument from recorded commands"

step4=(bash "$PIPE_ROOT/bin/step4_create_tuned_predlib.sh" --expname "$SEARCH_ID" --fastaabs "$FASTA" --tokens "$RESULT_DIR/model_package/model/dict.txt" --rt-model "$RESULT_DIR/model_package/model/rt.d0.finetuned.pt" --fr-model "$FR_MODEL" --verbose 4 --set "DIANN_BIN=$DIANN_EXTERNAL" --set "OUTBASE=$(dirname "$RESULT_DIR")" --set "STEP4_BASEDIR_NAME=04_refined_lib" --set "STEP4_SUBDIR=" --set "OUTLIB=$RESULT_DIR/04_refined_lib/refined_lib.speclib" --set "INSTRUMENT=$INSTRUMENT" --set "MOD_CONFIG=$MODS")
for s in "${SETS[@]}"; do
  [[ "$s" == threads=* ]] && s="THREADS=${s#threads=}"
  step4+=(--set "$s")
done
[[ -n "$THREADS_OVERRIDE" ]] && step4+=(--set "THREADS=$THREADS_OVERRIDE")
LIB="$RESULT_DIR/04_refined_lib/refined_lib.predicted.speclib"
step5=(bash "$PIPE_ROOT/bin/step5_quant.sh" --expname "$SEARCH_ID" --fastaabs "$FASTA" --lib "$LIB" --out-dir "$RESULT_DIR/05_quant" --temp-dir "$RESULT_DIR/tmp" --log-dir "$RESULT_DIR/logs" --verbose 4 --set "DIANN_BIN=$DIANN_EXTERNAL" --set "MOD_CONFIG=$MODS")
for f in "${RAW_FILES[@]}"; do step5+=(--f "$f"); done
for s in "${SETS[@]}"; do
  [[ "$s" == threads=* ]] && s="THREADS=${s#threads=}"
  step5+=(--set "$s")
done
[[ -n "$THREADS_OVERRIDE" ]] && step5+=(--set "THREADS=$THREADS_OVERRIDE")

if [[ "$TEST" == 1 ]]; then
  echo "TEST MODE: no files are written and no job is submitted."
  printf 'Model ID       : %s\nModel package  : %s\nExperiment     : %s\nDataset/profile: %s / %s\nFASTA          : %s\nFragment model : %s\nExecutor       : %s\nRAM logging    : %s (%ss)\nOutput         : %s\n' "$MODEL_ID" "$PACKAGE" "$EXPDIR" "$DATASET" "$PROFILE" "$FASTA" "$FR_MODEL" "$EXECUTOR" "$([[ "$RAM_LOG_ENABLE" == 1 ]] && echo on || echo off)" "$RAM_LOG_INTERVAL" "$RESULT_DIR"
  echo "Raw files:"; printf '  %s\n' "${RAW_FILES[@]}"
  echo "Step 4:"; printf '  %q' "${step4[@]}"; echo
  echo "Step 5:"; printf '  %q' "${step5[@]}"; echo
  [[ "$EXECUTOR" == cluster ]] && echo "Submit: $SUBMIT <worker-script>"
  exit 0
fi

mkdir -p "$RESULT_DIR/commands" "$RESULT_DIR/logs" "$RESULT_DIR/comparison"
envfile="$RESULT_DIR/resolved_config.env"
{ printf 'RESULT_DIR='; q "$RESULT_DIR"; printf '\nRESULT_PROGRESS_FILE='; q "$RESULT_PROGRESS_FILE"; printf '\nPACKAGE='; q "$PACKAGE"; printf '\nPIPE_ROOT='; q "$PIPE_ROOT"; printf '\nRAM_LOG_ENABLE='; q "$RAM_LOG_ENABLE"; printf '\nRAM_LOG_INTERVAL='; q "$RAM_LOG_INTERVAL"; printf '\nCENTRAL_PROGRESS_FILE='; q "$CENTRAL_PROGRESS_FILE"; printf '\nMODEL_ID='; q "$MODEL_ID"; printf '\nEXPERIMENT='; q "$EXPERIMENT"; printf '\nSEARCH_ID='; q "$SEARCH_ID"; printf '\n'; printf 'STEP4=('; printf ' %q' "${step4[@]}"; printf ' )\nSTEP5=('; printf ' %q' "${step5[@]}"; printf ' )\n'; } > "$envfile"
printf '%q ' "${step4[@]}" > "$RESULT_DIR/commands/step4.sh"; printf '\n' >> "$RESULT_DIR/commands/step4.sh"
printf '%q ' "${step5[@]}" > "$RESULT_DIR/commands/step5.sh"; printf '\n' >> "$RESULT_DIR/commands/step5.sh"
python3 - "$RESULT_DIR/manifest.json" "$RESULT_DIR" "$MODEL_ID" "$PACKAGE" "$EXPERIMENT" "$EXPDIR" "$FASTA" "$DATASET" "$PROFILE" "$FR_MODEL" "$EXECUTOR" "${RAW_FILES[@]}" <<'PY'
import json,sys,datetime,socket,pathlib
out,result,mid,package,exp,expdir,fasta,ds,profile,fr,executor,*raws=sys.argv[1:]
checksums=(pathlib.Path(package)/'checksums.sha256').read_text(encoding='utf-8')
json.dump(dict(model_id=mid,model_package=package,model_checksums=checksums,experiment=exp,experiment_source=expdir,fasta=fasta,dataset=ds,profile=profile,fragment_model=fr,raw_files=raws,executor=executor,host=socket.gethostname(),created_at=datetime.datetime.now().astimezone().isoformat(),output_dir=result,rt_model='model_package/model/rt.d0.finetuned.pt',tokens='model_package/model/dict.txt',step4_command=(pathlib.Path(result)/'commands/step4.sh').read_text(),step5_command=(pathlib.Path(result)/'commands/step5.sh').read_text()),open(out,'w'),indent=2); print()
PY
printf '{"state":"prepared"}\n' > "$RESULT_DIR/status.json"
append_external_progress "external_model_search.sh" "PREPARED" "model_id=$MODEL_ID;experiment=$EXPERIMENT;search_id=$SEARCH_ID;executor=$EXECUTOR;threads=${THREADS_OVERRIDE:-recorded};result_dir=$RESULT_DIR"
worker="$ROOT/bin/run_external_model_search_worker.sh"
if [[ "$PREPARE_ONLY" == 1 ]]; then
  printf 'PREPARED_ENVFILE=%s\n' "$envfile"
  exit 0
fi
if [[ "$EXECUTOR" == direct || "${EXTERNAL_MODEL_DIRECT_EXECUTION:-0}" == "1" ]]; then
  bash "$worker" "$envfile"
else
  read -r -a submit_cmd <<< "$SUBMIT"
  "${submit_cmd[@]}" "$worker" "$envfile" | tee "$RESULT_DIR/logs/submission.log"
  printf '{"state":"submitted"}\n' > "$RESULT_DIR/status.json"
  append_external_progress "external_model_search.sh" "SUBMITTED" "model_id=$MODEL_ID;experiment=$EXPERIMENT;search_id=$SEARCH_ID;executor=$EXECUTOR;result_dir=$RESULT_DIR"
fi
