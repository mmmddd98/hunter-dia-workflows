#!/usr/bin/env bash
# bin/_lib.sh
set -euo pipefail

VERBOSE="${VERBOSE:-0}"

_ts(){ date +'%Y-%m-%d %H:%M:%S'; }
_log(){ echo "[$(_ts)] [$1] $2" >&2; }
die(){ _log "ERROR" "$*"; exit 1; }
info(){ _log "INFO" "$*"; }
debug(){
  if [[ "${VERBOSE:-0}" =~ ^[0-9]+$ ]] && (( VERBOSE >= 1 )); then
    _log "DEBUG" "$*"
  fi
}

refresh_diann_paths_from_bin() {
  local outbase_overridden=0 key

  for key in "$@"; do
    [[ "${key^^}" == "OUTBASE" ]] && outbase_overridden=1
  done

  DIANN_BIN_NAME="${DIANN_BIN##*/}"
  DIANN_VERSION="${DIANN_BIN_NAME#diann-}"
  DIANN_VERSION_DIR="${DIANN_VERSION//./_}"

  if [[ "${OUTBASE_AUTO:-0}" == "1" && "$outbase_overridden" != "1" ]]; then
    : "${OUTROOT:?OUTROOT not set}"
    OUTBASE="${OUTROOT%/}/${DIANN_VERSION_DIR}"
  fi

  export DIANN_BIN_NAME DIANN_VERSION DIANN_VERSION_DIR OUTBASE
}

load_mod_config() {
  local config_name="${MOD_CONFIG:-}"
  local config_path=""

  unset VAR_MODS_LIST FIXED_MODS_LIST LIB_FIXED_MODS_LIST

  if [[ -z "$config_name" || "$config_name" == "legacy" ]]; then
    debug "Using legacy VAR_MOD_* modification settings"
    return 0
  fi

  if [[ "$config_name" == "none" ]]; then
    VAR_MODS="${VAR_MODS:-0}"
    debug "Using empty modification preset"
    return 0
  fi

  if [[ "$config_name" == */* ]]; then
    config_path="$config_name"
  else
    : "${PIPELINE_ROOT:?PIPELINE_ROOT not set}"
    config_path="${PIPELINE_ROOT}/config/mods/${config_name}.cfg"
  fi

  [[ -f "$config_path" ]] || die "Modification config not found: ${config_path}"
  # shellcheck source=/dev/null
  source "$config_path"
  export MOD_CONFIG_PATH="$config_path"
  debug "Loaded modification config: ${config_path}"
}

append_mod_args() {
  local cmd_array_name="$1"
  local -n __cmd_ref="$cmd_array_name"
  local mod var_mod_key var_mod_value
  local has_var_mods=0

  if declare -p FIXED_MODS_LIST >/dev/null 2>&1; then
    for mod in "${FIXED_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && __cmd_ref+=(--fixed-mod "$mod")
    done
  fi

  if declare -p VAR_MODS_LIST >/dev/null 2>&1; then
    for mod in "${VAR_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && has_var_mods=1
    done
  else
    for var_mod_key in VAR_MOD_1 VAR_MOD_2 VAR_MOD_3 VAR_MOD_4 VAR_MOD_5 VAR_MOD_6 VAR_MOD_7 VAR_MOD_8 VAR_MOD_9; do
      var_mod_value="${!var_mod_key:-}"
      [[ -n "$var_mod_value" ]] && has_var_mods=1
    done
  fi

  if [[ -n "${VAR_MODS:-}" && ( "$has_var_mods" == "1" || "${VAR_MODS}" != "0" ) ]]; then
    __cmd_ref+=(--var-mods "$VAR_MODS")
  fi

  if declare -p VAR_MODS_LIST >/dev/null 2>&1; then
    for mod in "${VAR_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && __cmd_ref+=(--var-mod "$mod")
    done
  else
    for var_mod_key in VAR_MOD_1 VAR_MOD_2 VAR_MOD_3 VAR_MOD_4 VAR_MOD_5 VAR_MOD_6 VAR_MOD_7 VAR_MOD_8 VAR_MOD_9; do
      var_mod_value="${!var_mod_key:-}"
      [[ -n "$var_mod_value" ]] && __cmd_ref+=(--var-mod "$var_mod_value")
    done
  fi

  if declare -p LIB_FIXED_MODS_LIST >/dev/null 2>&1; then
    for mod in "${LIB_FIXED_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && __cmd_ref+=(--lib-fixed-mod "$mod")
    done
  fi
}

mod_config_summary() {
  local config_name="${MOD_CONFIG:-legacy}"
  local fixed_count=0
  local var_count=0
  local lib_fixed_count=0
  local mod var_mod_key var_mod_value

  if declare -p FIXED_MODS_LIST >/dev/null 2>&1; then
    for mod in "${FIXED_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && ((fixed_count+=1))
    done
  fi

  if declare -p VAR_MODS_LIST >/dev/null 2>&1; then
    for mod in "${VAR_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && ((var_count+=1))
    done
  else
    for var_mod_key in VAR_MOD_1 VAR_MOD_2 VAR_MOD_3 VAR_MOD_4 VAR_MOD_5 VAR_MOD_6 VAR_MOD_7 VAR_MOD_8 VAR_MOD_9; do
      var_mod_value="${!var_mod_key:-}"
      [[ -n "$var_mod_value" ]] && ((var_count+=1))
    done
  fi

  if declare -p LIB_FIXED_MODS_LIST >/dev/null 2>&1; then
    for mod in "${LIB_FIXED_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && ((lib_fixed_count+=1))
    done
  fi

  printf '%s' "${config_name}"
  [[ -n "${MOD_CONFIG_PATH:-}" ]] && printf ' (%s)' "${MOD_CONFIG_PATH}"
  printf ': fixed=%s, var=%s, var-mods=%s' "$fixed_count" "$var_count" "${VAR_MODS:-0}"
  if (( lib_fixed_count > 0 )); then
    printf ', lib-fixed=%s' "$lib_fixed_count"
  fi
}

print_mod_config_details() {
  local mod var_mod_key var_mod_value

  echo "Modification config:"
  echo "  Summary: $(mod_config_summary)"

  if declare -p FIXED_MODS_LIST >/dev/null 2>&1; then
    echo "  Fixed mods:"
    for mod in "${FIXED_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && echo "    --fixed-mod ${mod}"
    done
  else
    echo "  Fixed mods: <none>"
  fi

  echo "  Variable mod limit: ${VAR_MODS:-0}"
  if declare -p VAR_MODS_LIST >/dev/null 2>&1; then
    echo "  Variable mods:"
    for mod in "${VAR_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && echo "    --var-mod ${mod}"
    done
  else
    echo "  Variable mods:"
    local any_var_mod=0
    for var_mod_key in VAR_MOD_1 VAR_MOD_2 VAR_MOD_3 VAR_MOD_4 VAR_MOD_5 VAR_MOD_6 VAR_MOD_7 VAR_MOD_8 VAR_MOD_9; do
      var_mod_value="${!var_mod_key:-}"
      if [[ -n "$var_mod_value" ]]; then
        echo "    --var-mod ${var_mod_value}"
        any_var_mod=1
      fi
    done
    (( any_var_mod == 1 )) || echo "    <none>"
  fi

  if declare -p LIB_FIXED_MODS_LIST >/dev/null 2>&1; then
    echo "  Library fixed mods:"
    for mod in "${LIB_FIXED_MODS_LIST[@]}"; do
      [[ -n "$mod" ]] && echo "    --lib-fixed-mod ${mod}"
    done
  fi
}

parse_expname() {
  debug "parse_expname args: $*"
  EXPNAME="auto"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --expname) EXPNAME="${2:-}"; shift 2;;
      *) shift;;
    esac
  done
  [[ -n "$EXPNAME" ]] || die "--expname needs a value (use 'auto' for timestamp)."

  if [[ "$EXPNAME" == "auto" ]]; then
    EXPNAME="exp_$(date +'%Y_%m_%d_%H_%M_%S')"
  fi

  export EXPNAME
  info "Experiment: ${EXPNAME}"
}

set_paths_for_exp() {
  : "${OUTBASE:?OUTBASE not set}"
  outbase="${OUTBASE%/}"
  [[ -n "$outbase" ]] || outbase="/"
  EXPDIR="${outbase}/${EXPNAME}"

  local step4_dir_name
  step4_dir_name="${STEP4_BASEDIR_NAME:-04_refined_lib}"

  S1DIR="${EXPDIR}/01_predlib"
  S2DIR="${EXPDIR}/02_tunelib"
  local step3_dir_name
  step3_dir_name="${STEP3_BASEDIR_NAME:-03_tune_models}"
  [[ "$step3_dir_name" != /* && "$step3_dir_name" != *".."* ]] || die "STEP3_BASEDIR_NAME must be a relative path without '..'"
  S3DIR="${EXPDIR}/${step3_dir_name}"
  S4DIR="${EXPDIR}/${step4_dir_name}"
  S5DIR="${EXPDIR}/05_quant"
  CALDIR="${EXPDIR}/cal_lib"

  mkdir -p "$S1DIR" "$S2DIR" "$S3DIR" "$S4DIR" "$S5DIR" "$CALDIR"
  export EXPDIR S1DIR S2DIR S3DIR S4DIR S5DIR CALDIR
  debug "EXPDIR=${EXPDIR}"
  debug "S1DIR=${S1DIR}"
  debug "S2DIR=${S2DIR}"
  debug "S3DIR=${S3DIR}"
  debug "S4DIR=${S4DIR}"
  debug "S5DIR=${S5DIR}"
  debug "CALDIR=${CALDIR}"
}

# convenience: check required binaries/paths
require_file(){
  debug "Checking file exists: $1"
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_exec(){
  debug "Checking executable: $1"
  command -v "$1" >/dev/null 2>&1 && return 0
  [[ -x "$1" ]] && return 0
  die "Not executable or not in PATH: $1"
}

run_cmd_with_diagnostics(){
  local label="$1"
  local cmd_array_name="$2"
  local log_path="${3:-}"
  local -n __cmd_ref="$cmd_array_name"
  local exit_code

  set +e
  "${__cmd_ref[@]}"
  exit_code=$?
  set -e

  if (( exit_code == 0 )); then
    return 0
  fi

  {
    echo ""
    echo "ERROR: ${label} failed with exit code ${exit_code}."
    if (( exit_code == 132 )); then
      echo "Detected SIGILL/Illegal instruction (exit 132)."
      echo "Likely causes: CPU instruction-set incompatibility for this DIA-NN code path or binary/runtime issue."
      echo "Suggested checks:"
      echo "  1) retry with lower threads (e.g. --set THREADS=1)"
      echo "  2) run one profile only to isolate profile-specific behavior"
      echo "  3) inspect kernel message buffer: dmesg | tail -n 50"
    fi
    if (( exit_code == 139 )); then
      echo "Detected segmentation fault (exit 139 / SIGSEGV)."
      echo "Suggested checks: reduce threads and isolate with a single profile/run first."
    fi
    if [[ -n "$log_path" ]]; then
      echo "Step log file: ${log_path}"
    fi
    echo "Failing command:"
    printf '%q ' "${__cmd_ref[@]}"
    echo
  } >&2

  return "$exit_code"
}

run_cmd_string_with_diagnostics(){
  local label="$1"
  local cmd_string="$2"
  local log_path="${3:-}"
  local exit_code

  set +e
  eval "$cmd_string"
  exit_code=$?
  set -e

  if (( exit_code == 0 )); then
    return 0
  fi

  {
    echo ""
    echo "ERROR: ${label} failed with exit code ${exit_code}."
    if (( exit_code == 132 )); then
      echo "Detected SIGILL/Illegal instruction (exit 132)."
      echo "Likely causes: CPU instruction-set incompatibility for this DIA-NN code path or binary/runtime issue."
      echo "Suggested checks:"
      echo "  1) retry with lower threads (e.g. --set THREADS=1)"
      echo "  2) run one profile only to isolate profile-specific behavior"
      echo "  3) inspect kernel message buffer: dmesg | tail -n 50"
    fi
    if (( exit_code == 139 )); then
      echo "Detected segmentation fault (exit 139 / SIGSEGV)."
      echo "Suggested checks: reduce threads and isolate with a single profile/run first."
    fi
    if [[ -n "$log_path" ]]; then
      echo "Step log file: ${log_path}"
    fi
    echo "Failing command string:"
    echo "$cmd_string"
  } >&2

  return "$exit_code"
}
