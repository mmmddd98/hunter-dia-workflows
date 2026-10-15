#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config/paths.cfg"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config/defaults.cfg"

errors=0

check_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "$path" ]]; then
    printf 'OK   %-22s %s\n' "$label" "$path"
  else
    printf 'MISS %-22s %s\n' "$label" "$path"
    errors=$((errors + 1))
  fi
}

check_exec() {
  local label="$1"
  local executable="$2"
  if command -v "$executable" >/dev/null 2>&1 || [[ -x "$executable" ]]; then
    printf 'OK   %-22s %s\n' "$label" "$executable"
  else
    printf 'MISS %-22s %s\n' "$label" "$executable"
    errors=$((errors + 1))
  fi
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    printf 'OK   %-22s %s\n' "$label" "$path"
  else
    printf 'MISS %-22s %s\n' "$label" "$path"
    errors=$((errors + 1))
  fi
}

warn_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "$path" ]]; then
    printf 'OK   %-22s %s\n' "$label" "$path"
  else
    printf 'WARN %-22s %s (not currently used by the pipeline)\n' "$label" "$path"
  fi
}

check_python_modules() {
  local label="$1"
  local python_bin="$2"
  shift 2
  local missing

  missing="$("$python_bin" - "$@" <<'PY'
import importlib.util
import sys

print(" ".join(name for name in sys.argv[1:] if importlib.util.find_spec(name) is None))
PY
)" || missing="<interpreter failed>"

  if [[ -z "$missing" ]]; then
    printf 'OK   %-22s %s\n' "$label" "$*"
  else
    printf 'MISS %-22s %s (Python: %s)\n' "$label" "$missing" "$python_bin"
    errors=$((errors + 1))
  fi
}

printf 'Machine: %s\n' "$DIANN_MACHINE"
printf 'Config : %s\n\n' "$DIANN_MACHINE_CONFIG"

check_exec "DIA-NN" "$DIANN_BIN"
check_dir "RAWBASE" "$RAWBASE"
check_dir "DATASET_ROOT" "$DATASET_ROOT"
check_dir "FASTABASE" "$FASTABASE"
check_dir "OUTROOT" "$OUTROOT"
warn_dir "TMPBASE" "$TMPBASE"
check_exec "pipeline Python" "$PIPELINE_PYTHON_PATH"
check_exec "MA-plot Python" "$MAPLOT_PYTHON_PATH"
check_file "MA-plot script" "$MAPLOT_REPORT_SCRIPT"
check_exec "evaluation Python" "${EVAL_PYTHON_PATH:-python3}"
check_file "Arabidopsis reference eval script" "${EVAL_ARABIDOPSIS_REFERENCE_PYTHON_FILE:-}"
check_file "E. coli centered eval script" "${EVAL_ECOLI_CENTERED_PYTHON_FILE:-}"
check_python_modules "pipeline modules" "$PIPELINE_PYTHON_PATH" yaml
check_python_modules "evaluation modules" "${EVAL_PYTHON_PATH:-python3}" numpy pandas polars matplotlib pyarrow
check_python_modules "MA-plot modules" "$MAPLOT_PYTHON_PATH" pandas polars matplotlib plotly

if (( errors > 0 )); then
  printf '\nMachine config has %d missing required path(s) or executable(s).\n' "$errors" >&2
  exit 1
fi

printf '\nMachine config is ready.\n'
