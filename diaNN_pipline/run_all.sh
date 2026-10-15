#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ORIGINAL_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${SCRIPT_DIR}"

auto_resume_has_arg() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$needle" ]] && return 0
  done
  return 1
}

is_run_all_process_cmd() {
  local cmd="$1"
  [[ "$cmd" =~ ^([^[:space:]]*/)?run_all\.sh([[:space:]]|$) ]] ||
    [[ "$cmd" =~ ^([^[:space:]]*/)?bash[[:space:]]+([^[:space:]]*/)?run_all\.sh([[:space:]]|$) ]]
}

wait_cmd_option_value() {
  local cmd="$1"
  local option="$2"
  awk -v option="$option" '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == option) {
          print $(i + 1)
          exit
        }
        if (index($i, option "=") == 1) {
          print substr($i, length(option) + 2)
          exit
        }
      }
    }
  ' <<<"$cmd"
}

wait_cmd_set_value() {
  local cmd="$1"
  local key="$2"
  awk -v key="$key" '
    {
      for (i = 1; i <= NF; i++) {
        if ($i != "--set") continue
        split($(i + 1), pair, "=")
        if (pair[1] == key) {
          print substr($(i + 1), length(key) + 2)
          exit
        }
      }
    }
  ' <<<"$cmd"
}

wait_format_duration() {
  local total="${1:-0}"
  local h m s
  (( total < 0 )) && total=0
  h=$(( total / 3600 ))
  m=$(( (total % 3600) / 60 ))
  s=$(( total % 60 ))
  if (( h > 0 )); then
    printf '%dh%02dm%02ds' "$h" "$m" "$s"
  elif (( m > 0 )); then
    printf '%dm%02ds' "$m" "$s"
  else
    printf '%ds' "$s"
  fi
}

wait_progress_has_ok_step() {
  local progress_file="$1"
  local step_name="$2"
  awk -F'\t' -v step="$step_name" '
    $2 == step && $3 == "OK" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$progress_file"
}

wait_progress_extract_runs_field() {
  local progress_file="$1"
  local step_name="$2"
  awk -F'\t' -v step="$step_name" '
    $2 == step && ($3 == "START" || $3 == "OK") {
      if (match($4, /runs=([0-9]+)/, m)) {
        value = m[1]
      }
    }
    END {
      if (value != "") print value
    }
  ' "$progress_file"
}

wait_progress_ok_count() {
  local progress_file="$1"
  local step_name="$2"
  local detail_filter="${3:-}"
  awk -F'\t' -v step="$step_name" -v filter="$detail_filter" '
    $2 == step && $3 == "OK" {
      if (filter != "" && index($4, filter) == 0) next
      count += 1
    }
    END { print count + 0 }
  ' "$progress_file"
}

wait_progress_completed_profiles_csv() {
  local progress_file="$1"
  local step_name="$2"
  awk -F'\t' -v step="$step_name" '
    $2 == step && $3 == "OK" {
      if (match($4, /profile=([^;]+)/, m)) {
        seen[m[1]] = 1
      }
    }
    END {
      first = 1
      for (profile in seen) {
        if (!first) printf ","
        printf "%s", profile
        first = 0
      }
    }
  ' "$progress_file"
}

wait_csv_contains() {
  local csv="$1"
  local needle="$2"
  local item
  IFS=',' read -r -a __wait_csv_items <<< "$csv"
  for item in "${__wait_csv_items[@]}"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

wait_parse_profiles_from_cmd() {
  local cmd="$1"
  local profile_csv item
  local -a profiles=()
  profile_csv="$(wait_cmd_option_value "$cmd" "--profile")"
  profile_csv="${profile_csv,,}"
  profile_csv="${profile_csv//[[:space:]]/}"
  if [[ "$profile_csv" == "all" || -z "$profile_csv" ]]; then
    profiles=(balanced fast slow)
  else
    IFS=',' read -r -a profiles <<< "$profile_csv"
  fi
  printf '%s\n' "${profiles[@]}"
}

wait_render_bar() {
  local filled="$1"
  local total="$2"
  local width="${3:-20}"
  local n_fill=0
  local i

  if (( total > 0 )); then
    n_fill=$(( filled * width / total ))
  fi
  (( n_fill < 0 )) && n_fill=0
  (( n_fill > width )) && n_fill=$width

  printf '['
  for (( i=0; i<width; i++ )); do
    if (( i < n_fill )); then
      printf '#'
    else
      printf '.'
    fi
  done
  printf ']'
}

wait_step_total_eta_from_progress() {
  local progress_file="$1"
  local active_step="$2"
  local active_detail="$3"
  local elapsed="$4"
  local total_count="$5"
  local done_count="$6"
  local avg=""
  local remaining_after_current=0
  local current_remaining=0

  case "$active_step" in
    step2_tunelib.sh|step3_tune_models.sh)
      avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step")"
      ;;
    step4_create_tuned_predlib.sh)
      avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step" "mode=single-run;")"
      ;;
    step5_quant.sh)
      if [[ "$active_detail" == *"mode=tuned;"* ]]; then
        avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step" "mode=tuned;")"
      elif [[ "$active_detail" == *"mode=baseline;"* ]]; then
        avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step" "mode=baseline;")"
      fi
      if [[ -z "$avg" ]]; then
        avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step")"
      fi
      ;;
    *)
      avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step")"
      ;;
  esac

  [[ -n "$avg" ]] || return 0
  current_remaining=$(( avg > elapsed ? avg - elapsed : 0 ))
  remaining_after_current=$(( total_count - done_count - 1 ))
  (( remaining_after_current < 0 )) && remaining_after_current=0
  wait_format_duration $(( current_remaining + remaining_after_current * avg ))
}

wait_resolve_progress_file_from_cmd() {
  local cmd="$1"
  local expname diann_bin outroot default_diann_bin
  local diann_version diann_version_dir

  expname="$(wait_cmd_option_value "$cmd" "--expname")"
  [[ -n "$expname" ]] || return 1

  diann_bin="$(wait_cmd_set_value "$cmd" "DIANN_BIN")"
  outroot="$(
    source "${PIPELINE_ROOT}/config/paths.cfg" >/dev/null 2>&1 || exit 1
    printf '%s\t%s\n' "${OUTROOT:-}" "${DIANN_BIN:-}"
  )" || return 1

  default_diann_bin="${outroot#*$'\t'}"
  outroot="${outroot%%$'\t'*}"
  [[ -n "$outroot" ]] || return 1
  [[ -n "$diann_bin" ]] || diann_bin="$default_diann_bin"
  [[ -n "$diann_bin" ]] || return 1

  diann_bin="${diann_bin##*/}"
  diann_version="${diann_bin#diann-}"
  diann_version_dir="${diann_version//./_}"
  printf '%s/%s/%s/progress.txt\n' "${outroot%/}" "$diann_version_dir" "$expname"
}

wait_last_unfinished_progress_entry() {
  local progress_file="$1"
  awk -F'\t' '
    $3 == "START" {
      key = $2 SUBSEP $4
      seq += 1
      active_seq[key] = seq
      active_ts[key] = $1
      active_step[key] = $2
      active_detail[key] = $4
    }
    $3 == "OK" {
      key = $2 SUBSEP $4
      delete active_seq[key]
      delete active_ts[key]
      delete active_step[key]
      delete active_detail[key]
    }
    END {
      best = 0
      for (key in active_seq) {
        if (active_seq[key] > best) {
          best = active_seq[key]
          best_key = key
        }
      }
      if (best > 0) {
        printf "%s\t%s\t%s\n", active_ts[best_key], active_step[best_key], active_detail[best_key]
      }
    }
  ' "$progress_file"
}

wait_average_elapsed_for_step() {
  local progress_file="$1"
  local step_name="$2"
  local detail_filter_1="${3:-}"
  local detail_filter_2="${4:-}"

  awk -F'\t' -v step="$step_name" -v f1="$detail_filter_1" -v f2="$detail_filter_2" '
    $2 == step && $3 == "OK" {
      if (f1 != "" && index($4, f1) == 0) next
      if (f2 != "" && index($4, f2) == 0) next
      if (match($4, /elapsed=([0-9]+)s/, m)) {
        sum += m[1]
        n += 1
      }
    }
    END {
      if (n > 0) printf "%.0f\n", sum / n
    }
  ' "$progress_file"
}

wait_print_progress_snapshot() {
  local target_cmd="$1"
  local progress_file expname no_cal_lib stop_after_step4
  local active_entry active_ts active_step active_detail elapsed avg same_profile same_mode
  local step_name state line_status line_detail
  local -a step_order=()
  local -a profiles=()
  local profiles_csv=""
  local completed_profiles=""
  local total_profiles=0
  local step4_expected=0 step4_done=0 step5_tuned_expected=0 step5_tuned_done=0
  local step5_total_expected=0 step5_total_done=0 step5_current_remaining=""
  local filled_count total_count current_avg current_remaining current_done
  local baseline_done=0
  local total_eta=""

  progress_file="$(wait_resolve_progress_file_from_cmd "$target_cmd" || true)"
  [[ -n "$progress_file" && -f "$progress_file" ]] || return 0

  expname="$(wait_cmd_option_value "$target_cmd" "--expname")"
  no_cal_lib=0
  stop_after_step4=0
  [[ "$target_cmd" == *"--no-cal-lib"* ]] && no_cal_lib=1
  [[ "$target_cmd" == *"--stop-after-step4"* ]] && stop_after_step4=1

  mapfile -t profiles < <(wait_parse_profiles_from_cmd "$target_cmd")
  total_profiles=${#profiles[@]}
  profiles_csv="$(printf '%s,' "${profiles[@]}")"
  profiles_csv="${profiles_csv%,}"

  step4_expected="$(wait_progress_extract_runs_field "$progress_file" "step4_create_tuned_predlib.sh")"
  step4_expected="${step4_expected:-0}"
  step4_done="$(wait_progress_ok_count "$progress_file" "step4_create_tuned_predlib.sh" "mode=single-run;")"

  step5_tuned_expected="$step4_expected"
  step5_tuned_done="$(wait_progress_ok_count "$progress_file" "step5_quant.sh" "mode=tuned;")"
  if wait_progress_has_ok_step "$progress_file" "step5_quant.sh" && awk -F'\t' '$2 == "step5_quant.sh" && $3 == "OK" && index($4, "mode=baseline;") > 0 { found=1 } END { exit(found ? 0 : 1) }' "$progress_file"; then
    baseline_done=1
  fi
  step5_total_expected=$(( step5_tuned_expected + 1 ))
  step5_total_done=$(( step5_tuned_done + baseline_done ))

  echo "Progress snapshot for ${expname:-<unknown>} (${progress_file}):"

  step_order=(step1_predlib.sh step2_tunelib.sh step3_tune_models.sh step4_create_tuned_predlib.sh step5_quant.sh)
  active_entry="$(wait_last_unfinished_progress_entry "$progress_file")"
  active_ts=""
  active_step=""
  active_detail=""
  if [[ -n "$active_entry" ]]; then
    IFS=$'\t' read -r active_ts active_step active_detail <<< "$active_entry"
  fi

  for step_name in "${step_order[@]}"; do
    if (( stop_after_step4 == 1 )) && [[ "$step_name" == "step5_quant.sh" ]]; then
      printf '  %-30s %s %s\n' "$step_name" "$(wait_render_bar 0 1 18)" "SKIPPED"
      continue
    fi

    if [[ "$step_name" == "$active_step" ]]; then
      state="RUNNING"
    elif wait_progress_has_ok_step "$progress_file" "$step_name"; then
      state="DONE"
    else
      state="PENDING"
    fi

    filled_count=0
    total_count=1
    line_detail="$state"
    case "$step_name" in
      step1_predlib.sh)
        if [[ "$state" == "DONE" ]]; then filled_count=1; fi
        ;;
      step2_tunelib.sh|step3_tune_models.sh)
        total_count=$total_profiles
        completed_profiles="$(wait_progress_completed_profiles_csv "$progress_file" "$step_name")"
        filled_count=0
        for profile in "${profiles[@]}"; do
          if wait_csv_contains "$completed_profiles" "$profile"; then
            filled_count=$((filled_count + 1))
          fi
        done
        if [[ "$state" == "RUNNING" && "$filled_count" -lt "$total_count" ]]; then
          :
        fi
        line_detail="${state} ${filled_count}/${total_count}"
        ;;
      step4_create_tuned_predlib.sh)
        total_count=$(( step4_expected > 0 ? step4_expected : 1 ))
        filled_count=$step4_done
        if [[ "$state" == "DONE" && "$step4_expected" == "0" ]]; then
          filled_count=1
          total_count=1
        fi
        line_detail="${state} ${filled_count}/${total_count}"
        ;;
      step5_quant.sh)
        total_count=$(( step5_total_expected > 0 ? step5_total_expected : 1 ))
        filled_count=$step5_total_done
        if [[ "$state" == "DONE" && "$step5_total_expected" == "0" ]]; then
          filled_count=1
          total_count=1
        fi
        line_detail="${state} ${filled_count}/${total_count}"
        ;;
    esac

    if [[ "$state" == "RUNNING" && "$step_name" == "$active_step" ]]; then
      total_eta="$(wait_step_total_eta_from_progress "$progress_file" "$active_step" "$active_detail" "$(( $(date +%s) - $(date -d "$active_ts" +%s 2>/dev/null || echo 0) ))" "$total_count" "$filled_count")"
      if [[ -n "$total_eta" ]]; then
        line_detail+=" ==> total eta~${total_eta}"
      fi
    fi

    printf '  %-30s %s %s\n' "$step_name" "$(wait_render_bar "$filled_count" "$total_count" 18)" "$line_detail"
  done

  if [[ -n "$active_step" && -n "$active_ts" ]]; then
    elapsed=$(( $(date +%s) - $(date -d "$active_ts" +%s 2>/dev/null || echo 0) ))
    same_profile=""
    same_mode=""
    if [[ "$active_detail" =~ profile=([^;]+) ]]; then
      same_profile="profile=${BASH_REMATCH[1]};"
    fi
    if [[ "$active_detail" =~ mode=([^;]+) ]]; then
      same_mode="mode=${BASH_REMATCH[1]};"
    fi

    avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step" "$same_profile" "$same_mode")"
    if [[ -z "$avg" ]]; then
      avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step" "$same_profile")"
    fi
    if [[ -z "$avg" ]]; then
      avg="$(wait_average_elapsed_for_step "$progress_file" "$active_step")"
    fi

    printf 'Current step: %s | elapsed=%s' "$active_step" "$(wait_format_duration "$elapsed")"
    if [[ -n "$avg" ]]; then
      printf ' | avg(similar)=%s' "$(wait_format_duration "$avg")"
      if (( avg > elapsed )); then
        printf ' | eta~%s' "$(wait_format_duration $((avg - elapsed)))"
      else
        printf ' | eta~soon'
      fi
    fi
    echo
    [[ -n "$active_detail" ]] && echo "Current detail: ${active_detail}"

    if [[ "$active_step" == "step5_quant.sh" && "$step5_total_expected" -gt 0 ]]; then
      current_done=$step5_total_done
      if [[ "$active_detail" == *"mode=baseline;"* ]]; then
        current_avg="$(wait_average_elapsed_for_step "$progress_file" "step5_quant.sh" "mode=baseline;")"
      else
        current_avg="$(wait_average_elapsed_for_step "$progress_file" "step5_quant.sh" "mode=tuned;")"
      fi
      if [[ -n "$current_avg" ]]; then
        if (( current_avg > elapsed )); then
          step5_current_remaining="$(wait_format_duration $((current_avg - elapsed)))"
        else
          step5_current_remaining="soon"
        fi
        current_remaining=$(( current_avg > elapsed ? current_avg - elapsed : 0 ))
        step5_current_remaining="$(wait_format_duration "$current_remaining")"
        if (( step5_tuned_done > 0 )); then
          avg="$(wait_average_elapsed_for_step "$progress_file" "step5_quant.sh" "mode=tuned;")"
        fi
        if [[ -n "$avg" ]]; then
          current_remaining=$(( current_avg > elapsed ? current_avg - elapsed : 0 ))
          same_mode=""
          if [[ "$active_detail" == *"mode=tuned;"* ]]; then
            same_mode="mode=tuned;"
          elif [[ "$active_detail" == *"mode=baseline;"* ]]; then
            same_mode="mode=baseline;"
          fi
          printf 'Step5 subruns: done=%d/%d | current_eta~%s' "$current_done" "$step5_total_expected" "$(wait_format_duration "$current_remaining")"
          if [[ "$active_detail" == *"mode=tuned;"* ]]; then
            local tuned_avg remaining_after_current total_remaining
            tuned_avg="$(wait_average_elapsed_for_step "$progress_file" "step5_quant.sh" "mode=tuned;")"
            if [[ -n "$tuned_avg" ]]; then
              remaining_after_current=$(( step5_total_expected - current_done - 1 ))
              (( remaining_after_current < 0 )) && remaining_after_current=0
              total_remaining=$(( current_remaining + remaining_after_current * tuned_avg ))
              printf ' | remaining_after_this=%d | total_eta~%s' "$remaining_after_current" "$(wait_format_duration "$total_remaining")"
            fi
          fi
          echo
        fi
      fi
    fi
  fi
}

wait_for_existing_run() {
  local wait_mode=""
  local requested_pid=""
  local requested_exp=""
  local test_mode=0
  local i=0
  local arg

  while [[ $i -lt ${#ORIGINAL_ARGS[@]} ]]; do
    arg="${ORIGINAL_ARGS[$i]}"
    case "$arg" in
      --wait-for-pid)
        ((i+=1))
        wait_mode="pid"
        requested_pid="${ORIGINAL_ARGS[$i]:-}"
        ;;
      --wait-for-pid=*)
        wait_mode="pid"
        requested_pid="${arg#*=}"
        ;;
      --wait-for-exp)
        ((i+=1))
        wait_mode="exp"
        requested_exp="${ORIGINAL_ARGS[$i]:-}"
        ;;
      --wait-for-exp=*)
        wait_mode="exp"
        requested_exp="${arg#*=}"
        ;;
      --wait-auto)
        wait_mode="auto"
        ;;
      --test|--test-full)
        test_mode=1
        ;;
    esac
    ((i+=1))
  done

  [[ -n "$wait_mode" ]] || return 0
  [[ "$wait_mode" != "pid" || "$requested_pid" =~ ^[0-9]+$ ]] || {
    echo "--wait-for-pid requires a numeric PID" >&2
    exit 2
  }
  [[ "$wait_mode" != "exp" || -n "$requested_exp" ]] || {
    echo "--wait-for-exp requires an experiment name" >&2
    exit 2
  }

  local target_pid=""
  local target_cmd=""
  local self_pid="${BASHPID:-$$}"
  if [[ "$wait_mode" == "pid" ]]; then
    target_cmd="$(ps -p "$requested_pid" -o args= 2>/dev/null || true)"
    [[ -n "$target_cmd" ]] || {
      echo "Wait target PID $requested_pid is not running; continuing immediately."
      return 0
    }
    [[ "$requested_pid" != "$$" && "$requested_pid" != "$self_pid" ]] && is_run_all_process_cmd "$target_cmd" || {
      echo "PID $requested_pid is not a different run_all.sh process" >&2
      exit 2
    }
    target_pid="$requested_pid"
  else
    local -A run_pids=()
    local -A run_ppids=()
    local -A run_cmds=()
    local pid ppid stat cmd

    while read -r pid ppid stat cmd; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      [[ "$stat" != Z* ]] || continue
      [[ "$pid" != "$$" && "$pid" != "$self_pid" ]] || continue
      is_run_all_process_cmd "$cmd" || continue
      [[ "$cmd" == *"--wait-auto"* || "$cmd" == *"--wait-for-pid"* || "$cmd" == *"--wait-for-exp"* ]] && continue
      if [[ "$wait_mode" == "exp" && "$cmd" != *"--expname $requested_exp"* && "$cmd" != *"--expname=$requested_exp"* ]]; then
        continue
      fi
      run_pids["$pid"]=1
      run_ppids["$pid"]="$ppid"
      run_cmds["$pid"]="$cmd"
    done < <(ps -eo pid=,ppid=,stat=,args=)

    local -a roots=()
    for pid in "${!run_pids[@]}"; do
      ppid="${run_ppids[$pid]}"
      [[ -n "${run_pids[$ppid]:-}" ]] || roots+=("$pid")
    done

    if (( ${#roots[@]} == 0 )); then
      echo "No other running run_all.sh process found for wait mode '$wait_mode'." >&2
      exit 2
    fi
    if (( ${#roots[@]} > 1 )); then
      echo "Multiple running run_all.sh roots found; use --wait-for-pid explicitly:" >&2
      for pid in "${roots[@]}"; do
        echo "  PID $pid: ${run_cmds[$pid]}" >&2
      done
      exit 2
    fi
    target_pid="${roots[0]}"
    target_cmd="${run_cmds[$target_pid]}"
  fi

  if (( test_mode == 1 )); then
    echo "WAIT TEST: would wait for PID $target_pid: $target_cmd"
    return 0
  fi

  echo "Waiting for run_all process PID $target_pid to finish..."
  echo "Target: $target_cmd"
  while kill -0 "$target_pid" 2>/dev/null; do
    printf '[%s] still running; checking again in 60s\n' "$(date -Is)"
    wait_print_progress_snapshot "$target_cmd"
    sleep 60
  done
  echo "[$(date -Is)] Wait target finished. Continuing this run in the same terminal."
  sleep 10
}

maybe_run_auto_resume_supervisor() {
  local enabled=0
  local sleep_seconds="${AUTO_RESUME_SLEEP:-60}"
  local max_attempts="${AUTO_RESUME_MAX_ATTEMPTS:-0}"
  local log_dir="${AUTO_RESUME_LOG_DIR:-${SCRIPT_DIR}/auto_resume_logs}"
  local args=()
  local i=0
  local arg

  while [[ $i -lt ${#ORIGINAL_ARGS[@]} ]]; do
    arg="${ORIGINAL_ARGS[$i]}"
    case "$arg" in
      --auto-resume)
        enabled=1
        ;;
      --auto-resume-sleep)
        ((i+=1))
        sleep_seconds="${ORIGINAL_ARGS[$i]:-}"
        ;;
      --auto-resume-sleep=*)
        sleep_seconds="${arg#*=}"
        ;;
      --auto-resume-max-attempts)
        ((i+=1))
        max_attempts="${ORIGINAL_ARGS[$i]:-}"
        ;;
      --auto-resume-max-attempts=*)
        max_attempts="${arg#*=}"
        ;;
      --auto-resume-log-dir)
        ((i+=1))
        log_dir="${ORIGINAL_ARGS[$i]:-}"
        ;;
      --auto-resume-log-dir=*)
        log_dir="${arg#*=}"
        ;;
      --wait-for-pid)
        ((i+=1))
        ;;
      --wait-for-pid=*)
        ;;
      --wait-for-exp)
        ((i+=1))
        ;;
      --wait-for-exp=*)
        ;;
      --wait-auto)
        ;;
      *)
        args+=("$arg")
        ;;
    esac
    ((i+=1))
  done

  (( enabled == 1 )) || return 0

  if ! [[ "$sleep_seconds" =~ ^[0-9]+$ ]]; then
    echo "--auto-resume-sleep must be a non-negative integer" >&2
    exit 2
  fi
  if ! [[ "$max_attempts" =~ ^[0-9]+$ ]]; then
    echo "--auto-resume-max-attempts must be a non-negative integer" >&2
    exit 2
  fi

  mkdir -p "$log_dir"

  local attempt=1
  local rc=0
  local ts log_file
  local cmd=()

  while true; do
    ts="$(date +'%Y%m%d_%H%M%S')"
    log_file="${log_dir%/}/run_all_auto_resume_attempt_${attempt}_${ts}.log"
    cmd=(bash "${SCRIPT_DIR}/run_all.sh" "${args[@]}")
    if (( attempt > 1 )) && ! auto_resume_has_arg "--resume" "${args[@]}"; then
      cmd+=("--resume")
    fi

    {
      printf '[%s] auto-resume attempt=%s\n' "$(date -Is)" "$attempt"
      printf 'command='
      printf '%q ' "${cmd[@]}"
      printf '\n'
    } | tee "$log_file"

    set +e
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
    set -e

    if (( rc == 0 )); then
      echo "Auto-resume completed successfully on attempt ${attempt}."
      exit 0
    fi

    echo "Auto-resume attempt ${attempt} failed with exit code ${rc}. Log: ${log_file}" >&2
    if (( max_attempts > 0 && attempt >= max_attempts )); then
      echo "Reached --auto-resume-max-attempts=${max_attempts}; stopping." >&2
      exit "$rc"
    fi

    attempt=$((attempt + 1))
    echo "Sleeping ${sleep_seconds}s before retry with --resume..."
    sleep "$sleep_seconds"
  done
}

wait_for_existing_run
maybe_run_auto_resume_supervisor

source "${PIPELINE_ROOT}/config/paths.cfg"
source "${PIPELINE_ROOT}/config/defaults.cfg"
source "${PIPELINE_ROOT}/config/entrapment.cfg"
source "${PIPELINE_ROOT}/config/step4.cfg"
source "${PIPELINE_ROOT}/bin/_lib.sh"

warn(){ _log "WARN" "$*"; }

supports_color() {
  [[ -t 1 ]] || return 1
  [[ -z "${NO_COLOR:-}" ]]
}

if supports_color; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
else
  C_RESET=""
  C_BOLD=""
  C_DIM=""
  C_RED=""
  C_GREEN=""
  C_YELLOW=""
  C_BLUE=""
fi

print_section() {
  local title="$1"
  echo
  echo "${C_BOLD}${C_BLUE}== ${title} ==${C_RESET}"
}

print_kv() {
  local key="$1"
  local value="$2"
  printf '%s%-18s%s %s\n' "$C_DIM" "$key" "$C_RESET" "$value"
}

print_ok() {
  echo "${C_GREEN}✔${C_RESET} $*"
}

print_warn_line() {
  echo "${C_YELLOW}⚠${C_RESET} $*"
}

print_err_line() {
  echo "${C_RED}✖${C_RESET} $*"
}

ram_step_selected() {
  local step_key="$1"
  [[ "${RAM_LOG_ENABLE:-0}" == "1" ]] || return 1

  local steps_csv="${RAM_LOG_STEPS:-all}"
  [[ -n "$steps_csv" ]] || steps_csv="all"
  if [[ "$steps_csv" == "all" ]]; then
    return 0
  fi

  case ",${steps_csv}," in
    *,"${step_key}",*) return 0 ;;
    *) return 1 ;;
  esac
}

run_with_optional_ram_logger() {
  local step_key="$1"
  local log_dir="$2"
  local log_tag="$3"
  local cmd_array_name="$4"
  local -n __cmd_ref="$cmd_array_name"

  if ! ram_step_selected "$step_key"; then
    "${__cmd_ref[@]}"
    return $?
  fi

  mkdir -p "$log_dir"

  local ts safe_tag log_file logger_pid cmd_rc
  ts="$(date +'%Y%m%d_%H%M%S')"
  safe_tag="${log_tag//[^A-Za-z0-9._-]/_}"
  log_file="${log_dir%/}/ram_${safe_tag}_${ts}.txt"

  info "RAM logger enabled for ${step_key}: ${log_file}"

  set +e
  bash "$RAM_LOGGER_SCRIPT" "$RAM_LOG_INTERVAL" "$log_file" "$$" >/dev/null 2>&1 &
  logger_pid=$!

  "${__cmd_ref[@]}"
  cmd_rc=$?

  kill "$logger_pid" >/dev/null 2>&1 || true
  wait "$logger_pid" >/dev/null 2>&1 || true
  set -e

  if [[ "${RAM_PLOT_ENABLE:-0}" == "1" ]]; then
    generate_ram_plot_for_log "$log_file" "$step_key"
  fi

  return "$cmd_rc"
}

generate_ram_plot_for_log() {
  local log_file="$1"
  local step_key="${2:-step}"

  [[ "${RAM_PLOT_ENABLE:-0}" == "1" ]] || return 0
  [[ -f "$log_file" ]] || return 0

  local out_png="${log_file%.txt}.png"
  if [[ "$out_png" == "$log_file" ]]; then
    out_png="${log_file}.png"
  fi

  local py_bin="${RAM_PLOT_PYTHON_BIN:-python3}"

  set +e
  "$py_bin" "$RAM_PLOT_SCRIPT" --input "$log_file" --output "$out_png" --title "RAM usage (${step_key})"
  local plot_rc=$?
  set -e

  if (( plot_rc == 0 )); then
    info "RAM plot written: ${out_png}"
  else
    warn "RAM plot generation failed for ${log_file} (exit ${plot_rc}); continuing"
  fi
}

cleanup_run_pid_file() {
  if [[ -n "${RUN_PID_META_FILE:-}" && -f "$RUN_PID_META_FILE" ]]; then
    printf 'ended_at=%s\n' "$(date -Is)" >> "$RUN_PID_META_FILE"
  fi
}

register_run_pid_file() {
  RUN_PID_FILE="${EXPDIR}/run_all.pid"
  RUN_PID_META_FILE="${EXPDIR}/run_all.pid.meta"
  local start_ticks
  start_ticks="$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null || true)"
  mkdir -p "$EXPDIR"
  printf '%s\n' "$$" > "$RUN_PID_FILE"
  {
    printf 'pid=%s\n' "$$"
    printf 'start_ticks=%s\n' "$start_ticks"
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'expdir=%s\n' "$EXPDIR"
    printf 'cmd='
    printf '%q ' "$0" "$@"
    printf '\n'
  } > "$RUN_PID_META_FILE"
  trap cleanup_run_pid_file EXIT
}

usage() {
  cat <<'EOF'
Usage:
  script.sh --expname <name|auto> --ds <dataset> [--fasta <name_in_FASTABASE>]
  script.sh --expname <name|auto> --ds <dataset> [--fasta-abs <absolute_fasta_path>]

Required arguments:
  --expname <name|auto>      Experiment name (or auto)
  --ds <dataset>             Dataset selector:
                             - dataset id (resolved as datasets/<id>/config.yaml or datasets/<id>.yaml)
                             - or relative path under datasets/
                             - or absolute path to a yaml file

  FASTA (priority order):
  --fasta <filename>         FASTA filename from FASTABASE (highest priority)
  --fasta-abs <abs_path>     Absolute FASTA path (highest priority)
                             If none is provided, the script uses --fasta-kind
                             to choose dataset YAML key fasta.fasta or
                             fasta.fasta_swiss.
  --fasta-kind <full|swiss>  Dataset FASTA selector when --fasta/--fasta-abs
                             is not provided. Default: full
  --fasta-entrap [full|swiss]
                            Use/create a combined original+shuffled entrapped
                            FASTA from the selected dataset YAML FASTA.
  --fasta-entrap-shuffle <fixed-r|simple>
                            Entrapment shuffle mode. Default: fixed-r.
                            fixed-r keeps arginine (R) positions fixed; simple
                            is the previous whole-tail shuffle.

Optional:
  --resume                    Resume same expname run from first incomplete step
                              found in progress.txt (cleans that step and downstream output dirs)
  --wait-for-pid <PID>        Wait for this running run_all.sh process before starting
  --wait-for-exp <name>       Wait for the running run_all.sh process with this experiment name
  --wait-auto                  Auto-detect the single running top-level run_all.sh process
                              (use --wait-for-pid if multiple runs are active)
  --auto-resume               Keep rerunning this command after failures until it
                              exits successfully. Retries add --resume if absent.
  --auto-resume-sleep <sec>   Wait between auto-resume attempts. Default: 60
  --auto-resume-max-attempts <n>
                              Stop after n attempts. Default: unlimited
  --auto-resume-log-dir <dir> Directory for attempt logs.
                              Default: diaNN_pipline/auto_resume_logs
  --stop-after-step4          Run at most through Step4 and always skip Step5
  --profile-parallel          Run multiple selected profiles in parallel and
                              divide THREADS by the number of profiles
  --profile <balanced|fast|slow|all|list>
                            Calibration-lib profile for bin/make_cal_lib.sh
                            Lists are comma-separated, e.g. fast,slow
                            Default: balanced
  --no-cal-lib              Skip bin/make_cal_lib.sh completely and do not pass
                            calibration refs to Step2/Step3
  --mods <preset>           Load config/mods/<preset>.cfg (same as --set MOD_CONFIG=<preset>)
  --set STEP4_DISTILL_LEVELS=...  Distillation level(s) consumed from Step3 models
                            (0|1|2 or comma-separated, default from config: 0)
  --set QVALUE=...          Analysis/report filter threshold. Default: 0.01.
  --set STEP5_REPORT_QVALUE=...
                            DIA-NN Step5 final_report export threshold.
                            Default: 1, so downstream analysis can filter later.
  --set enable_REPORT_DECOYS=0|1
                            Include DIA-NN decoy rows in Step5 final_report.
                            Default: 1.
  --set KEY=VALUE            Override variable (A-Z, a-z, 0-9, _)
  --test                     Validation-only mode (no directories/files created)
  --test-full                Dry-run full output (implies --test)
  --ram-log                  Enable RAM logging for selected step calls (default from config)
  --ram-log-steps <csv>      Steps to log: all|step1,make_cal,step2,step3,step4,step5 (default from config)
  --ram-log-interval <sec>   RAM logger interval in seconds (default from config)
  --ram-plot                 Write one RAM plot per RAM log file into the same folder
  --no-maplot-report         Skip automatic MA plot report after Step 5
  -v | --v | --verbose       Enable debug logs
  -h | --help                Show this help

This script currently implements:
- parse + validate dataset YAML
- validate FASTA input
- print normalized dataset summary
- call bin/step1_predlib.sh with forwarded runtime arguments
- call bin/make_cal_lib.sh as additional calibration helper (first train file)
- call bin/step2_tunelib.sh with profile-aware refs
- call bin/step3_tune_models.sh with profile-aware tuning (instrument-gated IM)
- call bin/step4_create_tuned_predlib.sh for profile/LR/model-target combinations
- call bin/step5_quant.sh for baseline + tuned quant searches
EOF
}

resolve_dataset_cfg() {
  local ds_input="$1"
  local datasets_dir="${PIPELINE_ROOT}/datasets"

  if [[ "$ds_input" == /* || "$ds_input" == ./* || "$ds_input" == ../* ]]; then
    if [[ -f "$ds_input" ]]; then
      echo "$(cd "$(dirname "$ds_input")" && pwd)/$(basename "$ds_input")"
      return 0
    fi
    die "Dataset path does not exist: $ds_input"
  fi

  if [[ -f "${datasets_dir}/${ds_input}/config.yaml" ]]; then
    echo "${datasets_dir}/${ds_input}/config.yaml"
    return 0
  fi

  if [[ -f "${datasets_dir}/${ds_input}.yaml" ]]; then
    echo "${datasets_dir}/${ds_input}.yaml"
    return 0
  fi

  if [[ -f "${datasets_dir}/${ds_input}" ]]; then
    echo "${datasets_dir}/${ds_input}"
    return 0
  fi

  die "Dataset config not found for --ds '${ds_input}'. Tried '${datasets_dir}/${ds_input}/config.yaml' and '${datasets_dir}/${ds_input}.yaml'."
}

resolve_fasta_from_value() {
  local fasta_value="$1"
  local ds_base_dir="${2:-}"
  local fasta_base

  [[ -n "$fasta_value" ]] || return 1

  if [[ "$fasta_value" == /* ]]; then
    echo "$fasta_value"
    return 0
  fi

  if [[ "$fasta_value" != */* ]]; then
    fasta_base="${FASTABASE%/}"
    [[ -n "$fasta_base" ]] || fasta_base="/"
    echo "${fasta_base}/${fasta_value}"
    return 0
  fi

  if [[ -n "$ds_base_dir" ]]; then
    echo "${ds_base_dir%/}/${fasta_value}"
    return 0
  fi

  echo "$fasta_value"
}

entrapment_filename_for_fasta() {
  local fasta_path="$1"
  local shuffle_mode="${2:-${FASTA_ENTRAP_SHUFFLE_MODE:-fixed-r}}"
  local base stem

  base="$(basename "$fasta_path")"
  case "$base" in
    *.fasta) stem="${base%.fasta}" ;;
    *.fa) stem="${base%.fa}" ;;
    *.faa) stem="${base%.faa}" ;;
    *.fas) stem="${base%.fas}" ;;
    *) stem="$base" ;;
  esac

  case "$shuffle_mode" in
    fixed-r|fixed_r|rfixed|r-fixed|preserve-r|preserve_r)
      echo "${stem}.entrapment_rfixed_combined.fasta"
      ;;
    simple|legacy|legacy-simple)
      echo "${stem}.entrapment_combined.fasta"
      ;;
    *)
      die "Invalid FASTA_ENTRAP_SHUFFLE_MODE '${shuffle_mode}' (allowed: fixed-r|simple)"
      ;;
  esac
}

ensure_entrapment_fasta() {
  local source_fasta="$1"
  local entrap_fasta="$2"
  local py_bin="${ENTRAPMENT_PYTHON_BIN:-${PIPELINE_PYTHON_PATH:-python3}}"
  local entrap_dir entrap_name

  require_file "$source_fasta"

  if [[ -f "$entrap_fasta" ]]; then
    FASTA_ENTRAP_ACTION="existing"
    return 0
  fi

  FASTA_ENTRAP_ACTION="create"
  if [[ "$TEST_MODE" == "1" ]]; then
    return 0
  fi

  require_exec "$py_bin"
  entrap_dir="$(dirname "$entrap_fasta")"
  entrap_name="$(basename "$entrap_fasta")"
  mkdir -p "$entrap_dir"

  PYTHONPATH="${PIPELINE_ROOT}/.." "$py_bin" -m entrapment.cli \
    --fasta "$source_fasta" \
    --out-dir "$entrap_dir" \
    --filename "$entrap_name" \
    --shuffle-mode "${FASTA_ENTRAP_SHUFFLE_MODE:-fixed-r}" \
    --seed "${FASTA_ENTRAP_SEED:-1337}" >/dev/null

  require_file "$entrap_fasta"
}

read_env_value() {
  local env_file="$1"
  local wanted_key="$2"
  local line raw key value

  [[ -f "$env_file" ]] || return 1

  while IFS= read -r line || [[ -n "$line" ]]; do
    raw="${line#${line%%[![:space:]]*}}"
    [[ -z "$raw" ]] && continue
    [[ "$raw" == \#* ]] && continue
    [[ "$raw" == *"="* ]] || continue

    key="${raw%%=*}"
    value="${raw#*=}"

    key="${key%${key##*[![:space:]]}}"
    key="${key#${key%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    value="${value#${value%%[![:space:]]*}}"

    if [[ "$key" == "$wanted_key" ]]; then
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      echo "$value"
      return 0
    fi
  done < "$env_file"

  return 1
}

resolve_base_dir_abs() {
  local ds_base_dir_raw="$1"
  local env_file="${PIPELINE_ROOT}/.env"
  local dataset_root="${DATASET_ROOT:-}"
  local base_trimmed

  base_trimmed="${ds_base_dir_raw#${ds_base_dir_raw%%[![:space:]]*}}"
  base_trimmed="${base_trimmed%${base_trimmed##*[![:space:]]}}"
  [[ -n "$base_trimmed" ]] || die "base_dir is empty"

  if [[ "$base_trimmed" == /* ]]; then
    echo "$base_trimmed"
    return 0
  fi

  if [[ -z "$dataset_root" ]]; then
    dataset_root="$(read_env_value "$env_file" "DATASET_ROOT" || true)"
  fi

  if [[ -z "$dataset_root" ]]; then
    dataset_root="${SHARED_STORAGE_ROOT:-}"
  fi
  if [[ -z "$dataset_root" ]]; then
    dataset_root="$(read_env_value "$env_file" "SHARED_STORAGE_ROOT" || true)"
  fi

  [[ -n "$dataset_root" ]] || die "Relative base_dir requires DATASET_ROOT (machine config, environment, or ${env_file})"
  [[ "$dataset_root" == /* ]] || die "DATASET_ROOT must be absolute, got: ${dataset_root}"

  dataset_root="${dataset_root%/}"
  base_trimmed="${base_trimmed#./}"

  echo "${dataset_root}/${base_trimmed}"
}

parse_dataset_yaml() {
  local cfg_file="$1"
  local dataset_type="${2:-prehunter}"
  local py_bin="${PIPELINE_PYTHON_PATH:-python3}"

  require_exec "$py_bin"

  eval "$($py_bin - "$cfg_file" "$dataset_type" <<'PY'
import shlex
import sys

cfg_path = sys.argv[1]
dataset_type = (sys.argv[2] if len(sys.argv) > 2 else "prehunter").strip().lower()

if dataset_type not in {"prehunter", "hunter"}:
    raise ValueError("DATASET_TYPE must be 'prehunter' or 'hunter'")

try:
    import yaml
except Exception:
    print("echo 'ERROR: PyYAML is required for dataset parsing (pip install pyyaml)' >&2", flush=True)
    print("exit 1", flush=True)
    raise SystemExit(0)

with open(cfg_path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}


def get(obj, keys, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def req(obj, keys):
    val = get(obj, keys, None)
    if val in (None, ""):
        dotted = ".".join(keys)
        raise ValueError(f"Missing required key in dataset yaml: {dotted}")
    return val


def q(value):
    return shlex.quote("" if value is None else str(value))


def emit(key, value):
    print(f"{key}={q(value)}")


def normalize_maplot_organism(value):
    token = str(value or "").strip().lower()
    aliases = {
        "arabidopsis thaliana": "arabidopsis",
        "arabidopsis": "arabidopsis",
        "homo sapiens": "human",
        "human": "human",
        "escherichia coli": "ecoli",
        "escherichia coli (strain k12)": "ecoli",
        "e. coli": "ecoli",
        "ecoli": "ecoli",
        "caenorhabditis elegans": "c_elegans",
        "c elegans": "c_elegans",
        "c_elegans": "c_elegans",
        "triplemix": "triplemix",
        "triple_mix": "triplemix",
    }
    return aliases.get(token, token.replace(" ", "_").replace("-", "_"))


def prefixed_dir(source_name, rel_dir):
    if rel_dir in (None, ""):
        return source_name

    rel = str(rel_dir).strip("/")
    if not rel:
        return source_name

    if not source_name:
        return rel

    if rel == source_name or rel.startswith(f"{source_name}/"):
        return rel

    return f"{source_name}/{rel}"


try:
    dataset_id = req(data, ["meta", "dataset_id"])
    organism_name = get(data, ["organism", "name"], "")
    sources = get(data, ["sources"], None)

    if isinstance(sources, dict):
        selected_source_name = dataset_type
        source_block = sources.get(selected_source_name)

        if not isinstance(source_block, dict):
            available_sources = [
                name for name in ("prehunter", "hunter") if isinstance(sources.get(name), dict)
            ]
            available_text = ", ".join(available_sources) if available_sources else "<none>"
            raise ValueError(
                "Dataset yaml does not define sources."
                f"{dataset_type} (available: {available_text}). "
                f"Use --set DATASET_TYPE={available_sources[0] if len(available_sources) == 1 else '<prehunter|hunter>'} "
                "or change DATASET_TYPE in config/defaults.cfg."
            )

        source_data = get(source_block, ["data"], {}) or {}
        if not isinstance(source_data, dict):
            raise ValueError(f"sources.{selected_source_name}.data must be a mapping")

        base_dir = get(sources, ["base_dir"], "")
        if not base_dir:
            base_dir = get(source_data, ["base_dir"], "")
        if not base_dir:
            raise ValueError("sources.base_dir or sources.<type>.data.base_dir must be set")

        dir_train = prefixed_dir(selected_source_name, get(source_data, ["dirs", "train"], ""))
        dir_test = prefixed_dir(selected_source_name, get(source_data, ["dirs", "test"], ""))

        train_files = get(source_data, ["train_files"], []) or []
        test_files = get(source_data, ["test_files"], []) or []

        if not isinstance(train_files, list):
            raise ValueError(f"sources.{selected_source_name}.data.train_files must be a list")
        if not isinstance(test_files, list):
            raise ValueError(f"sources.{selected_source_name}.data.test_files must be a list")

        acquisition = get(source_block, ["acquisition"], {}) or {}
        if not isinstance(acquisition, dict):
            acquisition = {}
    else:
        selected_source_name = ""
        source_data = get(data, ["data"], {}) or {}
        if not isinstance(source_data, dict):
            raise ValueError("data must be a mapping")

        base_dir = req(source_data, ["base_dir"])
        dir_train = get(source_data, ["dirs", "train"], "")
        dir_test = get(source_data, ["dirs", "test"], "")

        train_files = get(source_data, ["train_files"], []) or []
        test_files = get(source_data, ["test_files"], []) or []

        if not isinstance(train_files, list):
            raise ValueError("data.train_files must be a list")
        if not isinstance(test_files, list):
            raise ValueError("data.test_files must be a list")

        acquisition = get(data, ["acquisition"], {}) or {}
        if not isinstance(acquisition, dict):
            acquisition = {}

    emit("DS_META_DATASET_ID", dataset_id)
    emit("DS_ORGANISM_NAME", organism_name)
    emit("DS_MAPLOT_ORGANISM", normalize_maplot_organism(organism_name))
    emit("DS_REQUESTED_DATASET_TYPE", dataset_type)
    emit("DS_DATASET_TYPE", selected_source_name or dataset_type)
    emit("DS_SOURCE_NAME", selected_source_name)
    emit("DS_DATA_BASE_DIR", base_dir)
    emit("DS_DIR_TRAIN", dir_train)
    emit("DS_DIR_TEST", dir_test)
    emit("DS_FASTA", get(data, ["fasta", "fasta"], ""))
    emit("DS_FASTA_SWISS", get(data, ["fasta", "fasta_swiss"], ""))
    emit("DS_CONTAMINANTS_FASTA", get(data, ["fasta", "contaminants_fasta"], ""))
    emit(
        "DS_ACQUISITION_INSTRUMENT",
        get(acquisition, ["instrument"], get(data, ["acquisition", "instrument"], "")),
    )
    emit("DS_TRAIN_FILES", "|".join(str(x) for x in train_files))
    emit("DS_TEST_FILES", "|".join(str(x) for x in test_files))
except Exception as exc:
    print(f"echo {q('ERROR: ' + str(exc))} >&2")
    print("exit 1")
PY
)"
}

resolve_data_path() {
  local base_dir="$1"
  local rel_dir="$2"
  local file_name="$3"

  if [[ "$file_name" == /* ]]; then
    echo "$file_name"
    return 0
  fi

  if [[ -n "$rel_dir" ]]; then
    echo "${base_dir%/}/${rel_dir%/}/${file_name}"
  else
    echo "${base_dir%/}/${file_name}"
  fi
}

resolve_existing_data_file_path() {
  local base_dir="$1"
  local rel_dir="$2"
  local file_name="$3"
  local candidate

  candidate="$(resolve_data_path "$base_dir" "$rel_dir" "$file_name")"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  if [[ "$file_name" != *.* ]]; then
    candidate="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.mzML")"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi

    candidate="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.raw")"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  return 1
}

resolve_attempted_data_paths() {
  local base_dir="$1"
  local rel_dir="$2"
  local file_name="$3"
  local p_exact p_mzml p_raw

  p_exact="$(resolve_data_path "$base_dir" "$rel_dir" "$file_name")"
  p_mzml="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.mzML")"
  p_raw="$(resolve_data_path "$base_dir" "$rel_dir" "${file_name}.raw")"

  printf "exact=%s | mzML=%s | raw=%s" "$p_exact" "$p_mzml" "$p_raw"
}

normalize_split_list() {
  local input="$1"
  local arr_name="$2"
  local -n arr_ref="$arr_name"
  IFS='|' read -r -a arr_ref <<< "$input"
  if [[ ${#arr_ref[@]} -eq 1 && -z "${arr_ref[0]}" ]]; then
    arr_ref=()
  fi
}

check_dataset_group() {
  local label="$1"
  local base_dir="$2"
  local rel_dir="$3"
  local arr_name="$4"
  local -n files_ref="$arr_name"
  local file_path

  echo "${label} dir      : ${rel_dir:-<none>}"
  if [[ -n "$rel_dir" ]]; then
    require_dir_path="${base_dir%/}/${rel_dir%/}"
    [[ -d "$require_dir_path" ]] || die "Missing ${label} directory: ${require_dir_path}"
    if [[ "$TEST_COMPACT" != "1" ]]; then
      echo "${label} dir abs  : ${require_dir_path}"
    fi
  fi

  echo "${label} files    : ${#files_ref[@]}"
  local resolved_count=0
  for file_name in "${files_ref[@]}"; do
    file_path="$(resolve_existing_data_file_path "$base_dir" "$rel_dir" "$file_name" || true)"
    [[ -n "$file_path" ]] || die "Missing ${label} file: ${file_name} (tried exact, .mzML, .raw)"
    require_file "$file_path"
    resolved_count=$(( resolved_count + 1 ))
    if [[ "$TEST_COMPACT" != "1" ]]; then
      echo "  - ${file_name} -> ${file_path}"
    fi
  done

  if [[ "$TEST_COMPACT" == "1" ]]; then
    print_ok "${label}: ${resolved_count}/${#files_ref[@]} files valid"
  fi
}

append_progress_start() {
  local step_name="$1"
  local detail="${2:-}"
  local ts
  ts="$(date +'%Y-%m-%dT%H:%M:%S%z')"
  printf '%s\t%s\tSTART\t%s\n' "$ts" "$step_name" "$detail" >> "$PROGRESS_FILE"
}

append_progress_ok() {
  local step_name="$1"
  local detail="${2:-}"
  local start_ts="${3:-}"
  local ts end_ts elapsed_sec elapsed_str
  ts="$(date +'%Y-%m-%dT%H:%M:%S%z')"

  if [[ -n "$start_ts" ]]; then
    start_sec=$(date -d "$start_ts" +%s 2>/dev/null || echo 0)
    end_sec=$(date -d "$ts" +%s 2>/dev/null || echo 0)
    elapsed_sec=$((end_sec - start_sec))
    if (( elapsed_sec > 0 )); then
      elapsed_str=";elapsed=${elapsed_sec}s"
    else
      elapsed_str=""
    fi
  else
    elapsed_str=""
  fi

  printf '%s\t%s\tOK\t%s%s\n' "$ts" "$step_name" "$detail" "$elapsed_str" >> "$PROGRESS_FILE"
}

get_timestamp() {
  date +'%Y-%m-%dT%H:%M:%S%z'
}

progress_has_ok_step() {
  local step_name="$1"
  [[ -f "$PROGRESS_FILE" ]] || return 1
  awk -F'\t' -v step="$step_name" '
    $2 == step && $3 == "OK" { found=1 }
    END { exit(found ? 0 : 1) }
  ' "$PROGRESS_FILE"
}

progress_has_ok_step_detail_pattern() {
  local step_name="$1"
  local detail_regex="$2"
  [[ -f "$PROGRESS_FILE" ]] || return 1
  awk -F'\t' -v step="$step_name" -v pattern="$detail_regex" '
    $2 == step && $3 == "OK" && $4 ~ pattern { found=1 }
    END { exit(found ? 0 : 1) }
  ' "$PROGRESS_FILE"
}

progress_has_ok_step_detail_fragment() {
  local step_name="$1"
  local fragment="$2"
  [[ -f "$PROGRESS_FILE" ]] || return 1
  awk -F'\t' -v step="$step_name" -v frag="$fragment" '
    $2 == step && $3 == "OK" && index($4, frag) > 0 { found=1 }
    END { exit(found ? 0 : 1) }
  ' "$PROGRESS_FILE"
}

progress_has_ok_step_all_fragments() {
  local step_name="$1"
  shift || true
  local joined=""

  [[ -f "$PROGRESS_FILE" ]] || return 1
  joined="$(printf '%s\034' "$@")"

  awk -F'\t' -v step="$step_name" -v joined="$joined" '
    BEGIN {
      n = split(joined, frags, "\034")
    }
    $2 == step && $3 == "OK" {
      ok = 1
      for (i = 1; i <= n; i++) {
        if (frags[i] == "") continue
        if (index($4, frags[i]) == 0) {
          ok = 0
          break
        }
      }
      if (ok) {
        found = 1
      }
    }
    END {
      exit(found ? 0 : 1)
    }
  ' "$PROGRESS_FILE"
}

progress_has_ok_step_profile() {
  local step_name="$1"
  local profile_name="$2"
  [[ -f "$PROGRESS_FILE" ]] || return 1
  awk -F'\t' -v step="$step_name" -v profile="$profile_name" '
    $2 == step && $3 == "OK" && $4 ~ ("(^|;)profile=" profile "(;|$)") { found=1 }
    END { exit(found ? 0 : 1) }
  ' "$PROGRESS_FILE"
}

step_is_complete() {
  local step_name="$1"
  local profile_name

  case "$step_name" in
    step1_predlib.sh)
      progress_has_ok_step "step1_predlib.sh"
      return $?
      ;;
    make_cal_lib.sh)
      if [[ "${NO_CAL_LIB:-0}" == "1" ]]; then
        return 0
      fi
      if (( MULTI_PROFILE == 1 )); then
        if progress_has_ok_step_profile "make_cal_lib.sh" "all"; then
          return 0
        fi
        for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
          progress_has_ok_step_profile "make_cal_lib.sh" "$profile_name" || return 1
        done
        return 0
      fi
      progress_has_ok_step_profile "make_cal_lib.sh" "$PROFILE"
      return $?
      ;;
    step2_tunelib.sh|step3_tune_models.sh)
      for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
        progress_has_ok_step_profile "$step_name" "$profile_name" || return 1
      done
      return 0
      ;;
    step4_create_tuned_predlib.sh)
      # Always check with current parameters (BASEDIR_NAME, LR_SELECT, etc.)
      # Do not accept old Step4 runs without parameter validation
      step4_legacy_complete
      return $?
      ;;
    step5_quant.sh)
      # Always check with current parameters
      # Do not accept old Step5 runs without parameter validation
      step5_legacy_complete
      return $?
      ;;
    *)
      return 1
      ;;
  esac
}

determine_resume_step() {
  local step_name
  for step_name in step1_predlib.sh make_cal_lib.sh step2_tunelib.sh step3_tune_models.sh step4_create_tuned_predlib.sh step5_quant.sh; do
    if [[ "${NO_CAL_LIB:-0}" == "1" && "$step_name" == "make_cal_lib.sh" ]]; then
      continue
    fi
    if ! step_is_complete "$step_name"; then
      echo "$step_name"
      return 0
    fi
  done
  echo ""
}

step_profile_done() {
  local step_name="$1"
  local profile_name="$2"

  if progress_has_ok_step_profile "$step_name" "$profile_name"; then
    return 0
  fi

  if [[ "$step_name" == "make_cal_lib.sh" ]] && progress_has_ok_step_profile "$step_name" "all"; then
    return 0
  fi

  return 1
}

step5_baseline_complete() {
  progress_has_ok_step_detail_fragment "step5_quant.sh" "mode=baseline;"
}

print_progress_overview() {
  local step_name profile_name state missing_profile missing_item
  echo "Progress status (from ${PROGRESS_FILE}):"

  if [[ ! -f "$PROGRESS_FILE" ]]; then
    echo "  - progress file missing -> all steps are pending"
    return 0
  fi

  for step_name in step1_predlib.sh make_cal_lib.sh step2_tunelib.sh step3_tune_models.sh step4_create_tuned_predlib.sh step5_quant.sh; do
    if step_is_complete "$step_name"; then
      state="DONE"
    else
      state="PENDING"
    fi
    echo "  - ${step_name}: ${state}"

    if [[ "$state" == "PENDING" ]]; then
      case "$step_name" in
        make_cal_lib.sh|step2_tunelib.sh|step3_tune_models.sh)
          if (( MULTI_PROFILE == 1 )); then
            missing_profile=0
            for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
              if ! step_profile_done "$step_name" "$profile_name"; then
                echo "      missing detail: profile=${profile_name}"
                missing_profile=1
              fi
            done
            if (( missing_profile == 0 )); then
              echo "      missing detail: step not complete but no profile-level gap detected"
            fi
          fi
          ;;
        step4_create_tuned_predlib.sh)
          if step4_inputs_ready_for_planning; then
            step4_all_substeps_complete || true
            echo "      missing detail: step4 subruns missing ${STEP4_MISSING_SUBSTEP_COUNT}/${STEP4_EXPECTED_SUBSTEP_COUNT}"
            for missing_item in "${STEP4_MISSING_SUBSTEP_LIST[@]:-}"; do
              echo "        - ${missing_item}"
            done
            if ! progress_has_ok_step_all_fragments "step4_create_tuned_predlib.sh" "aggregate=1;" "profile=${PROFILE};" "" "targets=${STEP4_MODEL_TARGETS:-all};" "lr_select=${STEP4_LR_SELECT:-all};" "outdir=${STEP4_BASEDIR_NAME}"; then
              echo "      missing detail: aggregate step4 OK line for current parameters"
            fi
          else
            echo "      missing detail: step4 combos not resolvable yet (Step3 artifacts missing)"
          fi
          ;;
        step5_quant.sh)
          if step5_baseline_complete; then
            echo "      baseline: OK (will be skipped in resume)"
          else
            echo "      baseline: missing"
          fi

          if step4_inputs_ready_for_planning; then
            STEP5_EXPECTED_TUNED_COUNT=0
            STEP5_MISSING_TUNED_COUNT=0
            STEP5_MISSING_TUNED_LIST=()
            iterate_step4_runs step5_substep_check_callback
            echo "      tuned: missing ${STEP5_MISSING_TUNED_COUNT}/${STEP5_EXPECTED_TUNED_COUNT}"
            for missing_item in "${STEP5_MISSING_TUNED_LIST[@]:-}"; do
              echo "        - ${missing_item}"
            done
          else
            echo "      tuned: unresolved until Step3 artifacts exist"
          fi
          ;;
      esac
    fi

    if (( MULTI_PROFILE == 1 )) && [[ "$step_name" != "step1_predlib.sh" && "$step_name" != "step4_create_tuned_predlib.sh" ]]; then
      if [[ "${NO_CAL_LIB:-0}" == "1" && "$step_name" == "make_cal_lib.sh" ]]; then
        echo "      skipped by --no-cal-lib"
        continue
      fi
      for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
        if step_profile_done "$step_name" "$profile_name"; then
          echo "      ${profile_name}: OK"
        else
          echo "      ${profile_name}: missing"
        fi
      done
    fi
  done
}

print_execution_plan() {
  local -A step_run

  # Map RUN flags to step names
  step_run[step1_predlib.sh]=$RUN_STEP1
  step_run[make_cal_lib.sh]=$RUN_MAKE_CAL
  step_run[step2_tunelib.sh]=$RUN_STEP2
  step_run[step3_tune_models.sh]=$RUN_STEP3
  step_run[step4_create_tuned_predlib.sh]=$RUN_STEP4
  step_run[step5_quant.sh]=$RUN_STEP5

  # Fixed step order
  local -a step_order=(
    "step1_predlib.sh"
    "make_cal_lib.sh"
    "step2_tunelib.sh"
    "step3_tune_models.sh"
    "step4_create_tuned_predlib.sh"
    "step5_quant.sh"
  )

  echo "Execution plan for this invocation:"

  for step in "${step_order[@]}"; do
    if (( step_run[$step] == 1 )); then
      printf '%s▶%s %-40s (will run)\n' "$C_YELLOW" "$C_RESET" "$step"
    else
      if [[ "$step" == "make_cal_lib.sh" && "${NO_CAL_LIB:-0}" == "1" ]]; then
        printf '%s⊘%s %-40s (skipped - --no-cal-lib)\n' "$C_DIM" "$C_RESET" "$step"
        continue
      fi
      if [[ "$step" == "step5_quant.sh" && "$STOP_AFTER_STEP4" == "1" ]]; then
        printf '%s⊘%s %-40s (skipped - stopped after step4)\n' "$C_DIM" "$C_RESET" "$step"
      else
        printf '%s✓%s %-40s (completed)\n' "$C_GREEN" "$C_RESET" "$step"
      fi
    fi
  done
}

cleanup_from_step() {
  local resume_step="$1"
  local -a dirs_to_clean

  case "$resume_step" in
    step1_predlib.sh)
      dirs_to_clean=("$S1DIR" "$CALDIR" "$S2DIR" "$S3DIR" "$S4DIR" "$S5DIR")
      ;;
    make_cal_lib.sh)
      dirs_to_clean=("$CALDIR" "$S2DIR" "$S3DIR" "$S4DIR" "$S5DIR")
      ;;
    step2_tunelib.sh)
      dirs_to_clean=("$S2DIR" "$S3DIR" "$S4DIR" "$S5DIR")
      ;;
    step3_tune_models.sh)
      dirs_to_clean=("$S3DIR" "$S4DIR" "$S5DIR")
      ;;
    step4_create_tuned_predlib.sh)
      dirs_to_clean=("$S4DIR" "$S5DIR")
      ;;
    step5_quant.sh)
      dirs_to_clean=("$S5DIR")
      ;;
    *)
      die "Unknown resume step for cleanup: $resume_step"
      ;;
  esac

  for clean_dir in "${dirs_to_clean[@]}"; do
    if [[ "$TEST_MODE" == "1" ]]; then
      echo "TEST MODE: would remove output dir: ${clean_dir}"
      continue
    fi
    rm -rf "$clean_dir"
    mkdir -p "$clean_dir"
    info "Reset output dir for resume: ${clean_dir}"
  done
}

cleanup_profile_resume_step() {
  local resume_step="$1"
  local profile_name
  local -a downstream_dirs

  case "$resume_step" in
    step2_tunelib.sh)
      for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
        if step_profile_done "step2_tunelib.sh" "$profile_name"; then
          info "Resume mode: preserving completed Step2 profile=${profile_name}"
          continue
        fi
        if [[ "$TEST_MODE" == "1" ]]; then
          echo "TEST MODE: would remove output dir: ${S2DIR}/${profile_name}"
        else
          rm -rf "${S2DIR:?}/${profile_name}"
          mkdir -p "${S2DIR}/${profile_name}"
          info "Reset output dir for resume: ${S2DIR}/${profile_name}"
        fi
      done
      downstream_dirs=("$S3DIR" "$S4DIR" "$S5DIR")
      ;;
    step3_tune_models.sh)
      for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
        if step_profile_done "step3_tune_models.sh" "$profile_name"; then
          info "Resume mode: preserving completed Step3 profile=${profile_name}"
          continue
        fi
        if [[ "$TEST_MODE" == "1" ]]; then
          echo "TEST MODE: would remove output dir: ${S3DIR}/${profile_name}"
        else
          rm -rf "${S3DIR:?}/${profile_name}"
          mkdir -p "${S3DIR}/${profile_name}"
          info "Reset output dir for resume: ${S3DIR}/${profile_name}"
        fi
      done
      downstream_dirs=("$S4DIR" "$S5DIR")
      ;;
    *)
      die "Unknown profile resume step for cleanup: $resume_step"
      ;;
  esac

  for clean_dir in "${downstream_dirs[@]}"; do
    if [[ "$TEST_MODE" == "1" ]]; then
      echo "TEST MODE: would remove output dir: ${clean_dir}"
      continue
    fi
    rm -rf "$clean_dir"
    mkdir -p "$clean_dir"
    info "Reset downstream output dir for resume: ${clean_dir}"
  done
}

truncate_progress_from_step() {
  local resume_step="$1"
  local backup_file tmp_file

  [[ -f "$PROGRESS_FILE" ]] || return 0
  backup_file="${PROGRESS_FILE}.before_resume.$(date +%Y%m%d_%H%M%S).bak"
  cp -f "$PROGRESS_FILE" "$backup_file"
  info "Backed up progress file before resume: ${backup_file}"

  tmp_file="${PROGRESS_FILE}.tmp"
  awk -F'\t' -v resume_step="$resume_step" '
    function rank_step(step) {
      if (step == "step1_predlib.sh") return 1
      if (step == "make_cal_lib.sh") return 2
      if (step == "step2_tunelib.sh") return 3
      if (step == "step3_tune_models.sh") return 4
      if (step == "step4_create_tuned_predlib.sh") return 5
      if (step == "step5_quant.sh") return 6
      return 99
    }
    BEGIN { resume_rank = rank_step(resume_step) }
    {
      if (NF < 3) {
        print $0
        next
      }
      if (rank_step($2) < resume_rank) {
        print $0
      }
    }
  ' "$PROGRESS_FILE" > "$tmp_file"
  mv "$tmp_file" "$PROGRESS_FILE"
}

build_make_cal_cmd() {
  MAKE_CAL_CMD=(
    bash "$MAKE_CAL_SCRIPT"
    --expname "$EXPNAME"
    --fastaabs "$COPIED_FASTA"
    --lib "$STEP1_OUTLIB"
    --profile "$PROFILE"
    --f "$FIRST_TRAIN_FILE_RAW"
  )

  if (( VERBOSE > 0 )); then
    MAKE_CAL_CMD+=(--verbose "$VERBOSE")
  fi

  for key in "${CLI_OVERRIDES[@]}"; do
    MAKE_CAL_CMD+=(--set "${key}=${!key}")
  done
}

build_step2_tunelib_cmd() {
  local profile_name="$1"
  local ref_file="${2:-}"
  local threads_override="${3:-}"

  STEP2_TUNELIB_CMD=(
    bash "$STEP2_SCRIPT"
    --expname "$EXPNAME"
    --fastaabs "$COPIED_FASTA"
    --lib "$STEP1_OUTLIB"
    --profile "$profile_name"
  )

  if [[ -n "$ref_file" ]]; then
    STEP2_TUNELIB_CMD+=(--ref "$ref_file")
  fi

  if [[ "${NO_CAL_LIB:-0}" == "1" ]]; then
    STEP2_TUNELIB_CMD+=(--no-cal-lib)
  fi

  for raw_file in "${STEP2_RAW_FILES_RESOLVED[@]}"; do
    STEP2_TUNELIB_CMD+=(--f "$raw_file")
  done

  if (( VERBOSE > 0 )); then
    STEP2_TUNELIB_CMD+=(--verbose "$VERBOSE")
  fi

  for key in "${CLI_OVERRIDES[@]}"; do
    if [[ -n "$threads_override" && "$key" == "THREADS" ]]; then
      continue
    fi
    STEP2_TUNELIB_CMD+=(--set "${key}=${!key}")
  done

  if [[ -n "$threads_override" ]]; then
    STEP2_TUNELIB_CMD+=(--set "THREADS=${threads_override}")
  fi
}

build_step3_tune_models_cmd() {
  local profile_name="$1"
  local threads_override="${2:-}"

  STEP3_TUNE_MODELS_CMD=(
    bash "$STEP3_SCRIPT"
    --expname "$EXPNAME"
    --fastaabs "$COPIED_FASTA"
    --profile "$profile_name"
  )

  if [[ "${NO_CAL_LIB:-0}" == "1" ]]; then
    STEP3_TUNE_MODELS_CMD+=(--no-cal-lib)
  fi

  if (( VERBOSE > 0 )); then
    STEP3_TUNE_MODELS_CMD+=(--verbose "$VERBOSE")
  fi

  if [[ -n "${DS_DATASET_INSTRUMENT_NORMALIZED:-}" ]]; then
    STEP3_TUNE_MODELS_CMD+=(--set "INSTRUMENT=${DS_DATASET_INSTRUMENT_NORMALIZED}")
  fi

  for key in "${CLI_OVERRIDES[@]}"; do
    if [[ -n "$threads_override" && "$key" == "THREADS" ]]; then
      continue
    fi
    STEP3_TUNE_MODELS_CMD+=(--set "${key}=${!key}")
  done

  if [[ -n "$threads_override" ]]; then
    STEP3_TUNE_MODELS_CMD+=(--set "THREADS=${threads_override}")
  fi
}

expand_step4_lr_values() {
  local spec="$1"
  python3 - "$spec" <<'PY'
import sys

spec = sys.argv[1].strip()
if not spec:
  raise SystemExit("Empty STEP4_LR_SELECT value")

def fmt(v: float) -> str:
  return f"{v:.12g}"

values = []
if "," in spec:
  parts = [p.strip() for p in spec.split(",") if p.strip()]
  if not parts:
    raise SystemExit("Invalid LR list syntax")
  for p in parts:
    values.append(float(p))
elif ":" in spec:
  parts = [p.strip() for p in spec.split(":")]
  if len(parts) != 3:
    raise SystemExit("Range syntax must be start:stop:step")
  start, stop, step = map(float, parts)
  if step == 0:
    raise SystemExit("Range step must not be 0")
  if (stop - start) * step < 0:
    raise SystemExit("Range step direction does not reach stop")
  v = start
  tol = abs(step) * 1e-9
  if step > 0:
    while v <= stop + tol:
      values.append(v)
      v += step
  else:
    while v >= stop - tol:
      values.append(v)
      v += step
else:
  values = [float(spec)]

filtered = []
for v in values:
  if v <= 0:
    continue
  filtered.append(v)

if not filtered:
  raise SystemExit("No LR values > 0 resolved")

for v in filtered:
  print(fmt(v))
PY
}

find_single_required_file_in_dir() {
  local dir="$1"
  local pattern="$2"
  local label="$3"
  local matches=()

  shopt -s nullglob
  matches=("${dir}"/${pattern})
  shopt -u nullglob

  if (( ${#matches[@]} == 1 )); then
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  if (( ${#matches[@]} == 0 )); then
    die "Missing ${label} in ${dir} (pattern: ${pattern})"
  fi
  die "Ambiguous ${label} in ${dir}: found ${#matches[@]} files for pattern ${pattern}"
}

find_single_optional_file_in_dir() {
  local dir="$1"
  local pattern="$2"
  local label="$3"
  local matches=()

  shopt -s nullglob
  matches=("${dir}"/${pattern})
  shopt -u nullglob

  if (( ${#matches[@]} == 1 )); then
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  if (( ${#matches[@]} == 0 )); then
    printf '\n'
    return 0
  fi
  die "Ambiguous ${label} in ${dir}: found ${#matches[@]} files for pattern ${pattern}"
}

resolve_step4_lr_dirs_for_profile() {
  local profile_name="$1"
  local out_array_name="$2"
  local -n out_ref="$out_array_name"
  local profile_s3dir lr_select_raw lr_select
  local lr_dirs=()
  local requested_tags=()
  local selected=()

  profile_s3dir="${S3DIR}/${profile_name}"
  [[ -d "$profile_s3dir" ]] || die "Missing Step3 profile directory for Step4: ${profile_s3dir}"

  shopt -s nullglob
  for d in "${profile_s3dir}"/lr_*; do
    [[ -d "$d" ]] || continue
    lr_dirs+=("$d")
  done
  shopt -u nullglob

  (( ${#lr_dirs[@]} > 0 )) || die "No lr_* directories found for profile '${profile_name}' in ${profile_s3dir}"

  mapfile -t lr_dirs < <(printf '%s\n' "${lr_dirs[@]}" | sort)

  lr_select_raw="${STEP4_LR_SELECT:-all}"
  lr_select="${lr_select_raw,,}"
  lr_select="${lr_select//[[:space:]]/}"

  case "$lr_select" in
    all)
      selected=("${lr_dirs[@]}")
      ;;
    default)
      selected=("${lr_dirs[0]}")
      ;;
    *)
      if [[ "$lr_select" == *"lr_"* ]]; then
        IFS=',' read -r -a requested_tags <<< "$lr_select"
      else
        mapfile -t lr_values < <(expand_step4_lr_values "$lr_select")
        requested_tags=()
        for lr_value in "${lr_values[@]}"; do
          requested_tags+=("lr_$(printf '%s' "$lr_value" | tr '.' 'p')")
        done
      fi

      for req in "${requested_tags[@]}"; do
        req="${req,,}"
        [[ -n "$req" ]] || continue
        if [[ "$req" != lr_* ]]; then
          req="lr_${req}"
        fi
        req_path="${profile_s3dir}/${req}"
        [[ -d "$req_path" ]] || die "Requested Step4 LR directory not found for profile '${profile_name}': ${req_path}"
        selected+=("$req_path")
      done
      ;;
  esac

  (( ${#selected[@]} > 0 )) || die "No Step4 LR directories selected for profile '${profile_name}'"
  out_ref=("${selected[@]}")
}

resolve_step4_distill_levels() {
  local out_array_name="$1"
  local -n out_ref="$out_array_name"
  local raw token
  local requested=()
  local unique=()

  raw="${STEP4_DISTILL_LEVELS:-0}"
  raw="${raw//[[:space:]]/}"
  [[ -n "$raw" ]] || raw="0"

  IFS=',' read -r -a requested <<< "$raw"
  for token in "${requested[@]}"; do
    [[ -n "$token" ]] || continue
    if [[ ! "$token" =~ ^[0-2]$ ]]; then
      die "Invalid STEP4_DISTILL_LEVELS token '${token}' (allowed: 0,1,2)"
    fi
    duplicate=0
    for existing in "${unique[@]}"; do
      if [[ "$existing" == "$token" ]]; then
        duplicate=1
        break
      fi
    done
    if (( duplicate == 0 )); then
      unique+=("$token")
    fi
  done

  (( ${#unique[@]} > 0 )) || die "STEP4_DISTILL_LEVELS='${STEP4_DISTILL_LEVELS:-}' resolved to no usable levels"
  out_ref=("${unique[@]}")
}

resolve_step4_model_for_distill() {
  local lr_dir="$1"
  local dim="$2"
  local distill_level="$3"
  local label="$4"
  local model_file pattern
  local patterns=()

  patterns=(
    "*.${dim}.d${distill_level}.pt"
    "*.tuned_${dim}.d${distill_level}.pt"
    "*.d${distill_level}.tuned_${dim}.pt"
  )

  for pattern in "${patterns[@]}"; do
    model_file="$(find_single_optional_file_in_dir "$lr_dir" "$pattern" "$label (d${distill_level})")"
    if [[ -n "$model_file" ]]; then
      printf '%s\n' "$model_file"
      return 0
    fi
  done

  # Backward compatibility for older DIA-NN outputs without explicit distillation suffix.
  if [[ "$distill_level" == "0" ]]; then
    model_file="$(find_single_optional_file_in_dir "$lr_dir" "*.tuned_${dim}.pt" "$label")"
    if [[ -n "$model_file" ]]; then
      printf '%s\n' "$model_file"
      return 0
    fi
  fi

  printf '\n'
}

resolve_step4_models_for_lr_dir() {
  local lr_dir="$1"
  local distill_level="$2"
  local rt_var_name="$3"
  local im_var_name="$4"
  local fr_var_name="$5"
  local -n rt_ref="$rt_var_name"
  local -n im_ref="$im_var_name"
  local -n fr_ref="$fr_var_name"

  rt_ref="$(resolve_step4_model_for_distill "$lr_dir" "rt" "$distill_level" "Step4 RT model")"
  im_ref="$(resolve_step4_model_for_distill "$lr_dir" "im" "$distill_level" "Step4 IM model")"
  fr_ref="$(resolve_step4_model_for_distill "$lr_dir" "fr" "$distill_level" "Step4 FR model")"
}

resolve_step4_target_sets() {
  local has_rt="$1"
  local has_im="$2"
  local has_fr="$3"
  local out_array_name="$4"
  local -n out_ref="$out_array_name"
  local instrument_norm="${DS_DATASET_INSTRUMENT_NORMALIZED:-}"
  local model_targets_raw model_targets
  local available=()
  local requested=()
  local unique_sets=()

  if [[ "$has_rt" == "1" ]]; then
    available+=("rt")
  fi
  if [[ "$has_fr" == "1" ]]; then
    available+=("fr")
  fi
  if [[ "$has_im" == "1" ]]; then
    if [[ "$instrument_norm" == "timstof" ]]; then
      available+=("im")
    else
      warn "Step4: IM model found but instrument is '${instrument_norm:-unset}', so IM is ignored"
    fi
  fi

  (( ${#available[@]} > 0 )) || die "Step4: no usable tuned model files available (rt/fr/im)"

  model_targets_raw="${STEP4_MODEL_TARGETS:-all}"
  model_targets="${model_targets_raw,,}"
  model_targets="${model_targets//[[:space:]]/}"
  [[ -n "$model_targets" ]] || model_targets="all"

  IFS=',' read -r -a requested <<< "$model_targets"
  out_ref=()

  for target_group in "${requested[@]}"; do
    [[ -n "$target_group" ]] || continue
    target_group="${target_group,,}"

    group_dims=()
    if [[ "$target_group" == "all" ]]; then
      group_dims=("${available[@]}")
    else
      IFS='+' read -r -a target_parts <<< "$target_group"
      for part in "${target_parts[@]}"; do
        [[ -n "$part" ]] || continue
        case "$part" in
          rt)
            [[ "$has_rt" == "1" ]] || die "Step4 target '${target_group}' requested rt, but no matching RT model was found for selected distillation level(s)"
            group_dims+=("rt")
            ;;
          fr)
            [[ "$has_fr" == "1" ]] || die "Step4 target '${target_group}' requested fr, but no matching FR model was found for selected distillation level(s)"
            group_dims+=("fr")
            ;;
          im)
            [[ "$has_im" == "1" ]] || die "Step4 target '${target_group}' requested im, but no matching IM model was found for selected distillation level(s)"
            if [[ "$instrument_norm" == "timstof" ]]; then
              group_dims+=("im")
            else
              warn "Step4 target '${target_group}' requested im but instrument is '${instrument_norm:-unset}'; IM is ignored"
            fi
            ;;
          *)
            die "Invalid STEP4_MODEL_TARGETS token '${part}' in '${target_group}' (allowed dimensions: rt, im, fr; combine with '+', groups with ',')"
            ;;
        esac
      done
    fi

    canonical_dims=()
    for canonical_dim in rt im fr; do
      seen=0
      for d in "${group_dims[@]}"; do
        if [[ "$d" == "$canonical_dim" ]]; then
          seen=1
          break
        fi
      done
      (( seen == 1 )) && canonical_dims+=("$canonical_dim")
    done

    (( ${#canonical_dims[@]} > 0 )) || continue

    combo="${canonical_dims[0]}"
    for (( idx=1; idx<${#canonical_dims[@]}; idx++ )); do
      combo+="+${canonical_dims[$idx]}"
    done

    duplicate=0
    for existing in "${unique_sets[@]}"; do
      if [[ "$existing" == "$combo" ]]; then
        duplicate=1
        break
      fi
    done
    if (( duplicate == 0 )); then
      unique_sets+=("$combo")
    fi
  done

  (( ${#unique_sets[@]} > 0 )) || die "STEP4_MODEL_TARGETS='${model_targets_raw}' resolved to no valid model combination"
  out_ref=("${unique_sets[@]}")
}

build_step4_create_tuned_predlib_cmd() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local tokens_file="$5"
  local rt_model_file="$6"
  local im_model_file="$7"
  local fr_model_file="$8"
  local threads_override="${9:-}"
  local subdir suffix combo_tag

  combo_tag="${target_combo//+/_}"
  subdir="${profile_name}/lr_${lr_tag}/d${distill_level}"
  suffix="${profile_name}_${combo_tag}_lr_${lr_tag}_d${distill_level}"

  STEP4_CREATE_PREDLIB_CMD=(
    bash "$STEP4_SCRIPT"
    --expname "$EXPNAME"
    --fastaabs "$COPIED_FASTA"
    --tokens "$tokens_file"
    --set "STEP4_BASEDIR_NAME=${STEP4_BASEDIR_NAME}"
    --set "STEP4_SUBDIR=${subdir}"
    --set "STEP4_NAME_SUFFIX=${suffix}"
  )

  case "$target_combo" in
    *rt*)
      [[ -n "$rt_model_file" ]] || die "Internal Step4 build error: missing rt model for target ${target_combo} at d${distill_level}"
      STEP4_CREATE_PREDLIB_CMD+=(--rt-model "$rt_model_file")
      ;;
  esac
  case "$target_combo" in
    *im*)
      [[ -n "$im_model_file" ]] || die "Internal Step4 build error: missing im model for target ${target_combo} at d${distill_level}"
      STEP4_CREATE_PREDLIB_CMD+=(--im-model "$im_model_file")
      ;;
  esac
  case "$target_combo" in
    *fr*)
      [[ -n "$fr_model_file" ]] || die "Internal Step4 build error: missing fr model for target ${target_combo} at d${distill_level}"
      STEP4_CREATE_PREDLIB_CMD+=(--fr-model "$fr_model_file")
      ;;
  esac

  if (( VERBOSE > 0 )); then
    STEP4_CREATE_PREDLIB_CMD+=(--verbose "$VERBOSE")
  fi

  if [[ -n "${DS_DATASET_INSTRUMENT_NORMALIZED:-}" ]]; then
    STEP4_CREATE_PREDLIB_CMD+=(--set "INSTRUMENT=${DS_DATASET_INSTRUMENT_NORMALIZED}")
  fi

  for key in "${CLI_OVERRIDES[@]}"; do
    case "$key" in
      STEP4_SUBDIR|STEP4_NAME_SUFFIX)
        continue
        ;;
      THREADS)
        [[ -n "$threads_override" ]] && continue
        ;;
    esac
    STEP4_CREATE_PREDLIB_CMD+=(--set "${key}=${!key}")
  done

  if [[ -n "$threads_override" ]]; then
    STEP4_CREATE_PREDLIB_CMD+=(--set "THREADS=${threads_override}")
  fi
}

iterate_step4_runs() {
  local callback_fn="$1"
  local profile_name lr_dir lr_base lr_tag distill_level
  local tokens_file rt_model_file im_model_file fr_model_file
  local has_rt has_im has_fr
  local step4_lr_dirs=()
  local distill_levels=()
  local target_sets=()

  resolve_step4_distill_levels distill_levels

  for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
    resolve_step4_lr_dirs_for_profile "$profile_name" step4_lr_dirs
    for lr_dir in "${step4_lr_dirs[@]}"; do
      lr_base="$(basename "$lr_dir")"
      lr_tag="${lr_base#lr_}"
      [[ -n "$lr_tag" ]] || lr_tag="default"

      tokens_file="$(find_single_required_file_in_dir "$lr_dir" "*.dict.txt" "Step4 tokens file")"
      for distill_level in "${distill_levels[@]}"; do
        resolve_step4_models_for_lr_dir "$lr_dir" "$distill_level" rt_model_file im_model_file fr_model_file

        if [[ -n "$rt_model_file" ]]; then has_rt=1; else has_rt=0; fi
        if [[ -n "$im_model_file" ]]; then has_im=1; else has_im=0; fi
        if [[ -n "$fr_model_file" ]]; then has_fr=1; else has_fr=0; fi

        resolve_step4_target_sets "$has_rt" "$has_im" "$has_fr" target_sets
        for target_combo in "${target_sets[@]}"; do
          "$callback_fn" "$profile_name" "$lr_tag" "$distill_level" "$target_combo" "$tokens_file" "$rt_model_file" "$im_model_file" "$fr_model_file"
        done
      done
    done
  done
}

iterate_step4_runs_for_profile() {
  local profile_name="$1"
  local callback_fn="$2"
  local lr_dir lr_base lr_tag distill_level
  local tokens_file rt_model_file im_model_file fr_model_file
  local has_rt has_im has_fr
  local step4_lr_dirs=()
  local distill_levels=()
  local target_sets=()

  resolve_step4_distill_levels distill_levels

  resolve_step4_lr_dirs_for_profile "$profile_name" step4_lr_dirs
  for lr_dir in "${step4_lr_dirs[@]}"; do
    lr_base="$(basename "$lr_dir")"
    lr_tag="${lr_base#lr_}"
    [[ -n "$lr_tag" ]] || lr_tag="default"

    tokens_file="$(find_single_required_file_in_dir "$lr_dir" "*.dict.txt" "Step4 tokens file")"
    for distill_level in "${distill_levels[@]}"; do
      resolve_step4_models_for_lr_dir "$lr_dir" "$distill_level" rt_model_file im_model_file fr_model_file

      if [[ -n "$rt_model_file" ]]; then has_rt=1; else has_rt=0; fi
      if [[ -n "$im_model_file" ]]; then has_im=1; else has_im=0; fi
      if [[ -n "$fr_model_file" ]]; then has_fr=1; else has_fr=0; fi

      resolve_step4_target_sets "$has_rt" "$has_im" "$has_fr" target_sets
      for target_combo in "${target_sets[@]}"; do
        "$callback_fn" "$profile_name" "$lr_tag" "$distill_level" "$target_combo" "$tokens_file" "$rt_model_file" "$im_model_file" "$fr_model_file"
      done
    done
  done
}

step4_inputs_ready_for_planning() {
  local profile_name
  for profile_name in "${STEP2_PROFILE_LIST[@]}"; do
    [[ -d "${S3DIR}/${profile_name}" ]] || return 1
    compgen -G "${S3DIR}/${profile_name}/lr_*" > /dev/null || return 1
  done
  return 0
}

step4_plan_print_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local tokens_file="$5"
  local rt_model_file="$6"
  local im_model_file="$7"
  local fr_model_file="$8"

  build_step4_create_tuned_predlib_cmd "$profile_name" "$lr_tag" "$distill_level" "$target_combo" "$tokens_file" "$rt_model_file" "$im_model_file" "$fr_model_file" "$STEP4_THREADS_OVERRIDE"
  printf '%q ' "${STEP4_CREATE_PREDLIB_CMD[@]}"
  echo
}

step4_execute_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local tokens_file="$5"
  local rt_model_file="$6"
  local im_model_file="$7"
  local fr_model_file="$8"
  local step4_run_dir

  build_step4_create_tuned_predlib_cmd "$profile_name" "$lr_tag" "$distill_level" "$target_combo" "$tokens_file" "$rt_model_file" "$im_model_file" "$fr_model_file" "$STEP4_THREADS_OVERRIDE"
  info "Running Step4 for profile=${profile_name}, lr=${lr_tag}, distill=d${distill_level}, models=${target_combo}"
  debug "Step4 wrapper command (${profile_name}, lr=${lr_tag}, distill=d${distill_level}, models=${target_combo}): ${STEP4_CREATE_PREDLIB_CMD[*]}"
  step4_run_dir="${S4DIR}/${profile_name}/lr_${lr_tag}/d${distill_level}"
  run_with_optional_ram_logger "step4" "$step4_run_dir" "step4_${profile_name}_lr_${lr_tag}_d${distill_level}_${target_combo//+/_}" STEP4_CREATE_PREDLIB_CMD
}

step4_count_callback() {
  STEP4_COMBO_COUNT=$(( STEP4_COMBO_COUNT + 1 ))
}

resolve_step4_combo_count() {
  STEP4_COMBO_COUNT=0
  iterate_step4_runs step4_count_callback
  (( STEP4_COMBO_COUNT > 0 )) || die "Step4 combination resolution returned zero runs"
}

step4_execute_and_log_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local tokens_file="$5"
  local rt_model_file="$6"
  local im_model_file="$7"
  local fr_model_file="$8"
  local start_ts
  local fragment

  if (( PARTIAL_RESUME_STEP4 == 1 )); then
    fragment="profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=single-run;outdir=${STEP4_BASEDIR_NAME};"
    if progress_has_ok_step_detail_fragment "step4_create_tuned_predlib.sh" "$fragment"; then
      info "Step4 resume: skipping completed combo profile=${profile_name}, lr=${lr_tag}, distill=d${distill_level}, models=${target_combo} (outdir=${STEP4_BASEDIR_NAME})"
      return 0
    fi
  fi

  start_ts="$(get_timestamp)"
  append_progress_start "step4_create_tuned_predlib.sh" "profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=single-run;parallel=0;targets=${STEP4_MODEL_TARGETS:-all};lr_select=${STEP4_LR_SELECT:-all};distill_select=${STEP4_DISTILL_LEVELS:-0};outdir=${STEP4_BASEDIR_NAME}"
  step4_execute_callback "$profile_name" "$lr_tag" "$distill_level" "$target_combo" "$tokens_file" "$rt_model_file" "$im_model_file" "$fr_model_file"
  append_progress_ok "step4_create_tuned_predlib.sh" "profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=single-run;parallel=0;targets=${STEP4_MODEL_TARGETS:-all};lr_select=${STEP4_LR_SELECT:-all};distill_select=${STEP4_DISTILL_LEVELS:-0};outdir=${STEP4_BASEDIR_NAME}" "$start_ts"
}

step4_print_test_progress_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  printf '%s\n' "<timestamp>\tstep4_create_tuned_predlib.sh\tOK\tprofile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=single-run;parallel=${STEP4_PARALLEL_MODE};targets=${STEP4_MODEL_TARGETS:-all};lr_select=${STEP4_LR_SELECT:-all};distill_select=${STEP4_DISTILL_LEVELS:-0};outdir=${STEP4_BASEDIR_NAME}"
}

step5_print_test_progress_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  printf '%s\n' "<timestamp>\tstep5_quant.sh\tOK\tprofile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;parallel=${STEP5_PARALLEL_MODE};files=${#STEP5_RAW_FILES_RESOLVED[@]}"
}

step4_substep_check_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local fragment

  STEP4_EXPECTED_SUBSTEP_COUNT=$(( STEP4_EXPECTED_SUBSTEP_COUNT + 1 ))
  fragment="profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=single-run;"
  if ! progress_has_ok_step_detail_fragment "step4_create_tuned_predlib.sh" "$fragment"; then
    STEP4_MISSING_SUBSTEP_COUNT=$(( STEP4_MISSING_SUBSTEP_COUNT + 1 ))
    STEP4_MISSING_SUBSTEP_LIST+=("profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo}")
  fi
}

step4_all_substeps_complete() {
  step4_inputs_ready_for_planning || return 1
  STEP4_EXPECTED_SUBSTEP_COUNT=0
  STEP4_MISSING_SUBSTEP_COUNT=0
  STEP4_MISSING_SUBSTEP_LIST=()
  iterate_step4_runs step4_substep_check_callback
  (( STEP4_EXPECTED_SUBSTEP_COUNT > 0 )) || return 1
  (( STEP4_MISSING_SUBSTEP_COUNT == 0 ))
}

step5_substep_check_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local fragment

  STEP5_EXPECTED_TUNED_COUNT=$(( STEP5_EXPECTED_TUNED_COUNT + 1 ))
  fragment="profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;"
  if ! progress_has_ok_step_detail_fragment "step5_quant.sh" "$fragment"; then
    STEP5_MISSING_TUNED_COUNT=$(( STEP5_MISSING_TUNED_COUNT + 1 ))
    STEP5_MISSING_TUNED_LIST+=("profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo}")
  fi
}

step5_all_substeps_complete() {
  step5_baseline_complete || return 1

  step4_inputs_ready_for_planning || return 1
  STEP5_EXPECTED_TUNED_COUNT=0
  STEP5_MISSING_TUNED_COUNT=0
  STEP5_MISSING_TUNED_LIST=()
  iterate_step4_runs step5_substep_check_callback
  (( STEP5_EXPECTED_TUNED_COUNT > 0 )) || return 1
  (( STEP5_MISSING_TUNED_COUNT == 0 ))
}

step4_legacy_complete() {
  # Check for new format first (strict)
  if progress_has_ok_step_all_fragments "step4_create_tuned_predlib.sh" \
    "aggregate=1;" \
    "profile=${PROFILE};" \
    "" \
    "targets=${STEP4_MODEL_TARGETS:-all};" \
    "lr_select=${STEP4_LR_SELECT:-all};" \
    "distill_select=${STEP4_DISTILL_LEVELS:-0};" \
    "outdir=${STEP4_BASEDIR_NAME}"; then
    return 0
  fi

  # Fallback: check for ANY Step4 OK entry (very broad, for old/incomplete runs)
  # If any Step4 entry exists with targets/lr_select and outdir matching, accept it
  if grep -q "step4_create_tuned_predlib.sh.*OK.*targets=${STEP4_MODEL_TARGETS:-all}.*lr_select=${STEP4_LR_SELECT:-all}.*distill_select=${STEP4_DISTILL_LEVELS:-0}.*outdir=${STEP4_BASEDIR_NAME}" "$PROGRESS_FILE" 2>/dev/null; then
    return 0
  fi

  return 1
}

step5_legacy_complete() {
  progress_has_ok_step_all_fragments "step5_quant.sh" \
    "profile=${PROFILE};" \
    "mode=baseline+tuned;" \
    ""
}

step5_tuned_combo_tag() {
  local target_combo="$1"
  echo "${target_combo//+/_}"
}

step5_resolve_tuned_lib() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local combo_tag base_dir candidate

  combo_tag="$(step5_tuned_combo_tag "$target_combo")"
  base_dir="${S4DIR}/${profile_name}/lr_${lr_tag}/d${distill_level}"

  candidate="${base_dir}/refined_lib_${profile_name}_${combo_tag}_lr_${lr_tag}_d${distill_level}.predicted.predicted.speclib"
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  candidate="${base_dir}/refined_lib_${profile_name}_${combo_tag}_lr_${lr_tag}_d${distill_level}.predicted.speclib"
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  # Fallback for legacy naming without explicit distillation suffix.
  candidate="${S4DIR}/${profile_name}/lr_${lr_tag}/refined_lib_${profile_name}_${combo_tag}_lr_${lr_tag}.predicted.predicted.speclib"
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  candidate="${S4DIR}/${profile_name}/lr_${lr_tag}/refined_lib_${profile_name}_${combo_tag}_lr_${lr_tag}.predicted.speclib"
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  printf '%s\n' "$candidate"
}

build_step5_quant_cmd() {
  local lib_file="$1"
  local out_sub="$2"
  local threads_override="${3:-}"

  STEP5_QUANT_CMD=(
    bash "$STEP5_SCRIPT"
    --expname "$EXPNAME"
    --fastaabs "$COPIED_FASTA"
    --lib "$lib_file"
    --out-sub "$out_sub"
  )

  for raw_file in "${STEP5_RAW_FILES_RESOLVED[@]}"; do
    STEP5_QUANT_CMD+=(--f "$raw_file")
  done

  if (( VERBOSE > 0 )); then
    STEP5_QUANT_CMD+=(--verbose "$VERBOSE")
  fi

  for key in "${CLI_OVERRIDES[@]}"; do
    if [[ -n "$threads_override" && "$key" == "THREADS" ]]; then
      continue
    fi
    STEP5_QUANT_CMD+=(--set "${key}=${!key}")
  done

  if [[ -n "$threads_override" ]]; then
    STEP5_QUANT_CMD+=(--set "THREADS=${threads_override}")
  fi
}

run_maplot_report() {
  [[ "$ENABLE_MAPLOT_REPORT" == "1" ]] || return 0

  local diann_out_root
  diann_out_root="${OUTBASE%/${DIANN_VERSION_DIR}}"
  if [[ "$diann_out_root" == "$OUTBASE" ]]; then
    diann_out_root="$(dirname "$OUTBASE")"
  fi

  local -a maplot_cmd
  maplot_cmd=(
    "$MAPLOT_PYTHON_PATH"
    "$MAPLOT_REPORT_SCRIPT"
    --experiment "$EXPNAME"
    --diann-out-root "$diann_out_root"
    --pipeline-version "$DIANN_VERSION_DIR"
    --organism "$MAPLOT_ORGANISM"
    --qvalue "$QVALUE"
    --max-mod-upsets "$MAPLOT_MAX_MOD_UPSETS"
  )

  if [[ "$MAPLOT_REUSE_CACHE" == "1" ]]; then
    maplot_cmd+=(--reuse-cache)
  fi
  if [[ "$MAPLOT_PSSM_PREDICTION" == "1" ]]; then
    maplot_cmd+=(--pssm-prediction)
  fi

  info "Running automatic MA plot report"
  debug "MA plot report command: ${maplot_cmd[*]}"
  "${maplot_cmd[@]}"
}

step5_plan_baseline_cmd() {
  local threads_override="${1:-}"
  build_step5_quant_cmd "$STEP1_OUTLIB" "baseline" "$threads_override"
  printf '%q ' "${STEP5_QUANT_CMD[@]}"
  echo
}

step5_plan_tuned_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local combo_tag tuned_lib tuned_out_sub

  combo_tag="$(step5_tuned_combo_tag "$target_combo")"
  tuned_lib="$(step5_resolve_tuned_lib "$profile_name" "$lr_tag" "$distill_level" "$target_combo")"
  tuned_out_sub="tuned/${profile_name}_${combo_tag}_lr_${lr_tag}_d${distill_level}"

  build_step5_quant_cmd "$tuned_lib" "$tuned_out_sub" "$STEP5_THREADS_OVERRIDE"
  printf '%q ' "${STEP5_QUANT_CMD[@]}"
  echo
}

step5_execute_tuned_callback() {
  local profile_name="$1"
  local lr_tag="$2"
  local distill_level="$3"
  local target_combo="$4"
  local combo_tag tuned_lib tuned_out_sub
  local step5_run_dir

  combo_tag="$(step5_tuned_combo_tag "$target_combo")"
  tuned_lib="$(step5_resolve_tuned_lib "$profile_name" "$lr_tag" "$distill_level" "$target_combo")"
  tuned_out_sub="tuned/${profile_name}_${combo_tag}_lr_${lr_tag}_d${distill_level}"

  require_file "$tuned_lib"
  build_step5_quant_cmd "$tuned_lib" "$tuned_out_sub" "$STEP5_THREADS_OVERRIDE"
  info "Running Step5 tuned quant for profile=${profile_name}, lr=${lr_tag}, distill=d${distill_level}, models=${target_combo}"
  debug "Step5 wrapper command (${profile_name}, lr=${lr_tag}, distill=d${distill_level}, models=${target_combo}): ${STEP5_QUANT_CMD[*]}"
  step5_run_dir="${S5DIR}/${tuned_out_sub}"
  run_with_optional_ram_logger "step5" "$step5_run_dir" "step5_tuned_${profile_name}_lr_${lr_tag}_d${distill_level}_${target_combo//+/_}" STEP5_QUANT_CMD
}

resolve_step1_outlib_for_make_cal() {
  local outlib_parquet="$1"
  local outdir_s1="$2"
  local expname="$3"
  local -a speclib_candidates

  if [[ -f "$outlib_parquet" ]]; then
    echo "$outlib_parquet"
    return 0
  fi

  shopt -s nullglob
  speclib_candidates=("${outdir_s1}/step1_${expname}_predicted"*.speclib)
  shopt -u nullglob

  if (( ${#speclib_candidates[@]} == 0 )); then
    return 1
  fi

  if (( ${#speclib_candidates[@]} > 1 )); then
    warn "Multiple Step1 .speclib candidates found, using first: ${speclib_candidates[0]}"
  fi

  echo "${speclib_candidates[0]}"
}

FASTA_NAME=""
FASTA_ABS=""
FASTA_KIND="${FASTA_KIND:-full}"
FASTA_ENTRAP=0
FASTA_ENTRAP_KIND=""
FASTA_ENTRAP_SOURCE=""
FASTA_ENTRAP_TARGET=""
FASTA_ENTRAP_ACTION="none"
FASTA_ENTRAP_SHUFFLE_MODE="${FASTA_ENTRAP_SHUFFLE_MODE:-fixed-r}"
FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=0
DS_INPUT=""
TEST_MODE=0
TEST_FULL=0
TEST_COMPACT=0
# RAM logging: Use defaults from config, with fallback for old configs
RAM_LOG_ENABLE="${RAM_LOG_ENABLE:-1}"
RAM_LOG_INTERVAL="${RAM_LOG_INTERVAL:-10}"
RAM_LOG_STEPS="${RAM_LOG_STEPS:-all}"
RAM_LOGGER_SCRIPT="${PIPELINE_ROOT}/ram_logger.sh"
RAM_PLOT_ENABLE="${RAM_PLOT_ENABLE:-0}"
RAM_PLOT_SCRIPT="${PIPELINE_ROOT}/plots/step_04_vary_2026_04_01/ramPlotter.py"
RAM_PLOT_PYTHON_BIN="${RAM_PLOT_PYTHON_BIN:-${PIPELINE_PYTHON_PATH:-python3}}"
PROFILE_PARALLEL=0
PROFILE="balanced"
RESUME_MODE=0
STOP_AFTER_STEP4=0
EXPNAME_INPUT=""
CLI_OVERRIDES=()
VERBOSE="${VERBOSE:-0}"
NO_CAL_LIB="${NO_CAL_LIB:-0}"
ENABLE_MAPLOT_REPORT="${ENABLE_MAPLOT_REPORT:-1}"

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --ds)
      ((i+=1))
      DS_INPUT="${ARGS[$i]:-}"
      [[ -n "$DS_INPUT" ]] || die "--ds needs a value"
      ;;
    --fasta)
      ((i+=1))
      FASTA_NAME="${ARGS[$i]:-}"
      [[ -n "$FASTA_NAME" ]] || die "--fasta needs a value"
      ;;
    --fasta-abs|--fastaabs)
      ((i+=1))
      FASTA_ABS="${ARGS[$i]:-}"
      [[ -n "$FASTA_ABS" ]] || die "--fasta-abs needs a value"
      ;;
    --fasta-kind|--fasta-source)
      ((i+=1))
      FASTA_KIND="${ARGS[$i]:-}"
      [[ -n "$FASTA_KIND" ]] || die "--fasta-kind needs a value (full|swiss)"
      ;;
    --fasta-entrap)
      FASTA_ENTRAP=1
      next="${ARGS[$((i+1))]:-}"
      if [[ -n "$next" && "$next" != --* ]]; then
        ((i+=1))
        FASTA_ENTRAP_KIND="$next"
      fi
      ;;
    --fasta-entrap=*)
      FASTA_ENTRAP=1
      FASTA_ENTRAP_KIND="${ARGS[$i]#*=}"
      [[ -n "$FASTA_ENTRAP_KIND" ]] || die "--fasta-entrap needs full|swiss when using --fasta-entrap=<value>"
      ;;
    --fasta-entrap-shuffle)
      ((i+=1))
      FASTA_ENTRAP_SHUFFLE_MODE="${ARGS[$i]:-}"
      [[ -n "$FASTA_ENTRAP_SHUFFLE_MODE" ]] || die "--fasta-entrap-shuffle needs fixed-r|simple"
      FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=1
      ;;
    --fasta-entrap-shuffle=*)
      FASTA_ENTRAP_SHUFFLE_MODE="${ARGS[$i]#*=}"
      [[ -n "$FASTA_ENTRAP_SHUFFLE_MODE" ]] || die "--fasta-entrap-shuffle needs fixed-r|simple"
      FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=1
      ;;
    --profile)
      ((i+=1))
      PROFILE="${ARGS[$i]:-}"
      [[ -n "$PROFILE" ]] || die "--profile needs a value (balanced|fast|slow|all or a comma-separated list)"
      ;;
    --mods)
      ((i+=1))
      MOD_CONFIG="${ARGS[$i]:-}"
      [[ -n "$MOD_CONFIG" ]] || die "--mods needs a preset name from config/mods"
      CLI_OVERRIDES+=("MOD_CONFIG")
      ;;
    --set)
      ((i+=1))
      override="${ARGS[$i]:-}"
      [[ -n "$override" ]] || die "--set needs KEY=VALUE"
      [[ "$override" == *=* ]] || die "--set expects KEY=VALUE"
      key_raw="${override%%=*}"
      key="${key_raw^^}"
      value="${override#*=}"
      [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "Invalid --set key '$key_raw' (use A-Z, a-z, 0-9, _)"
      printf -v "$key" '%s' "$value"
      CLI_OVERRIDES+=("$key")
      if [[ "$key" == "FASTA_ENTRAP_SHUFFLE_MODE" ]]; then
        FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT=1
      fi
      ;;
    --v|-v)
      VERBOSE=1
      ;;
    --verbose)
      next="${ARGS[$((i+1))]:-}"
      if [[ "$next" =~ ^[0-4]$ ]]; then
        VERBOSE="$next"
        ((i+=1))
      else
        VERBOSE=2
      fi
      ;;
    --verbose=*)
      VERBOSE="${ARGS[$i]#*=}"
      ;;
    --test)
      TEST_MODE=1
      ;;
    --test-full)
      TEST_FULL=1
      ;;
    --ram-log)
      RAM_LOG_ENABLE=1
      ;;
    --ram-log-steps)
      ((i+=1))
      RAM_LOG_STEPS="${ARGS[$i]:-}"
      [[ -n "$RAM_LOG_STEPS" ]] || die "--ram-log-steps needs a value"
      RAM_LOG_ENABLE=1
      ;;
    --ram-log-interval)
      ((i+=1))
      RAM_LOG_INTERVAL="${ARGS[$i]:-}"
      [[ -n "$RAM_LOG_INTERVAL" ]] || die "--ram-log-interval needs a value"
      RAM_LOG_ENABLE=1
      ;;
    --ram-plot)
      RAM_PLOT_ENABLE=1
      RAM_LOG_ENABLE=1
      ;;
    --no-maplot-report)
      ENABLE_MAPLOT_REPORT=0
      CLI_OVERRIDES+=("ENABLE_MAPLOT_REPORT")
      ;;
    --profile-parallel)
      PROFILE_PARALLEL=1
      ;;
    --no-cal-lib)
      NO_CAL_LIB=1
      CLI_OVERRIDES+=("NO_CAL_LIB")
      ;;
    --resume)
      RESUME_MODE=1
      ;;
    --wait-for-pid)
      ((i+=1))
      [[ "${ARGS[$i]:-}" =~ ^[0-9]+$ ]] || die "--wait-for-pid needs a numeric PID"
      ;;
    --wait-for-pid=*)
      [[ "${ARGS[$i]#*=}" =~ ^[0-9]+$ ]] || die "--wait-for-pid needs a numeric PID"
      ;;
    --wait-for-exp)
      ((i+=1))
      [[ -n "${ARGS[$i]:-}" ]] || die "--wait-for-exp needs an experiment name"
      ;;
    --wait-for-exp=*)
      [[ -n "${ARGS[$i]#*=}" ]] || die "--wait-for-exp needs an experiment name"
      ;;
    --wait-auto)
      ;;
    --stop-after-step4)
      STOP_AFTER_STEP4=1
      ;;
    --expname)
      ((i+=1))
      EXPNAME_INPUT="${ARGS[$i]:-}"
      [[ -n "$EXPNAME_INPUT" ]] || die "--expname needs a value"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: ${ARGS[$i]}"
      ;;
  esac
  ((i+=1))
done

# --test-full implies test mode
if [[ "$TEST_FULL" == "1" ]]; then
  TEST_MODE=1
fi

RAM_LOG_STEPS="${RAM_LOG_STEPS,,}"
RAM_LOG_STEPS="${RAM_LOG_STEPS// /}"
if [[ "$RAM_LOG_ENABLE" == "1" ]]; then
  [[ "$RAM_LOG_INTERVAL" =~ ^[0-9]+$ ]] || die "--ram-log-interval must be an integer >= 1"
  (( RAM_LOG_INTERVAL >= 1 )) || die "--ram-log-interval must be >= 1"
  [[ -n "$RAM_LOG_STEPS" ]] || RAM_LOG_STEPS="all"

  for token in ${RAM_LOG_STEPS//,/ }; do
    case "$token" in
      all|step1|make_cal|step2|step3|step4|step5)
        ;;
      *)
        die "Invalid --ram-log-steps token '${token}' (allowed: all,step1,make_cal,step2,step3,step4,step5)"
        ;;
    esac
  done
fi

PROFILE="${PROFILE,,}"
PROFILE="${PROFILE//[[:space:]]/}"
[[ "$PROFILE" != ,* && "$PROFILE" != *, && "$PROFILE" != *,,* ]] || \
  die "Invalid --profile '$PROFILE' (empty list element)"
STEP2_PROFILE_LIST=()
declare -A PROFILE_SEEN=()
if [[ "$PROFILE" == "all" ]]; then
  STEP2_PROFILE_LIST=(balanced fast slow)
else
  IFS=',' read -r -a PROFILE_TOKENS <<< "$PROFILE"
  for profile_token in "${PROFILE_TOKENS[@]}"; do
    case "$profile_token" in
      balanced|fast|slow)
        ;;
      *)
        die "Invalid --profile '$PROFILE' (allowed: balanced|fast|slow|all or a comma-separated list)"
        ;;
    esac
    [[ -v "PROFILE_SEEN[$profile_token]" ]] && die "Duplicate profile '$profile_token' in --profile '$PROFILE'"
    PROFILE_SEEN[$profile_token]=1
    STEP2_PROFILE_LIST+=("$profile_token")
  done
fi
(( ${#STEP2_PROFILE_LIST[@]} > 0 )) || die "--profile must select at least one profile"
MULTI_PROFILE=0
(( ${#STEP2_PROFILE_LIST[@]} > 1 )) && MULTI_PROFILE=1

[[ "$VERBOSE" =~ ^[0-4]$ ]] || die "Invalid verbose level '$VERBOSE' (allowed: 0..4)"
[[ "${NO_CAL_LIB:-0}" =~ ^[01]$ ]] || die "Invalid NO_CAL_LIB '${NO_CAL_LIB:-}' (allowed: 0 or 1)"
ENABLE_MAPLOT_REPORT="${ENABLE_MAPLOT_REPORT:-${enable_maplot_report:-1}}"
[[ "$ENABLE_MAPLOT_REPORT" =~ ^[01]$ ]] || die "Invalid ENABLE_MAPLOT_REPORT '${ENABLE_MAPLOT_REPORT}' (allowed: 0 or 1)"
MAPLOT_PYTHON_PATH="${MAPLOT_PYTHON_PATH:-${maplot_python_path:-${EVAL_PYTHON_PATH:-python3}}}"
MAPLOT_REPORT_SCRIPT="${MAPLOT_REPORT_SCRIPT:-${maplot_report_script:-${PIPELINE_ROOT}/../diann_visu/post_run_reports/scripts/run_maplot_report.py}}"
MAPLOT_ORGANISM="${MAPLOT_ORGANISM:-${maplot_organism:-auto}}"
MAPLOT_MAX_MOD_UPSETS="${MAPLOT_MAX_MOD_UPSETS:-${maplot_max_mod_upsets:-6}}"
MAPLOT_REUSE_CACHE="${MAPLOT_REUSE_CACHE:-${maplot_reuse_cache:-0}}"
MAPLOT_PSSM_PREDICTION="${MAPLOT_PSSM_PREDICTION:-${maplot_pssm_prediction:-0}}"
[[ "$MAPLOT_MAX_MOD_UPSETS" =~ ^[0-9]+$ ]] || die "Invalid MAPLOT_MAX_MOD_UPSETS '${MAPLOT_MAX_MOD_UPSETS}' (must be integer)"
[[ "$MAPLOT_REUSE_CACHE" =~ ^[01]$ ]] || die "Invalid MAPLOT_REUSE_CACHE '${MAPLOT_REUSE_CACHE}' (allowed: 0 or 1)"
[[ "$MAPLOT_PSSM_PREDICTION" =~ ^[01]$ ]] || die "Invalid MAPLOT_PSSM_PREDICTION '${MAPLOT_PSSM_PREDICTION}' (allowed: 0 or 1)"

if [[ "$TEST_MODE" == "1" && "$TEST_FULL" != "1" ]]; then
  VERBOSE=0
fi

export VERBOSE

for key in "${CLI_OVERRIDES[@]}"; do
  debug "CLI override applied: ${key}=${!key}"
done
refresh_diann_paths_from_bin "${CLI_OVERRIDES[@]}"
load_mod_config

parse_expname "$@"

if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" != "1" ]]; then
  warn "--profile-parallel needs multiple selected profiles; running sequentially for profile=${PROFILE}."
fi

if [[ "$RAM_LOG_ENABLE" == "1" ]]; then
  require_file "$RAM_LOGGER_SCRIPT"
  info "RAM logging enabled: steps=${RAM_LOG_STEPS}; interval=${RAM_LOG_INTERVAL}s"
fi

if [[ "$RAM_PLOT_ENABLE" == "1" ]]; then
  require_file "$RAM_PLOT_SCRIPT"
  info "RAM plotting enabled: script=${RAM_PLOT_SCRIPT}"
fi

if [[ "$ENABLE_MAPLOT_REPORT" == "1" ]]; then
  require_exec "$MAPLOT_PYTHON_PATH"
  require_file "$MAPLOT_REPORT_SCRIPT"
  info "MA plot report enabled: script=${MAPLOT_REPORT_SCRIPT}"
fi

if [[ "$PROFILE_PARALLEL" == "1" ]]; then
  [[ "${THREADS:-}" =~ ^[0-9]+$ ]] || die "THREADS must be numeric for --profile-parallel (current: '${THREADS:-}')"
  (( THREADS >= 1 )) || die "THREADS must be >= 1 for --profile-parallel"
fi

if [[ -z "$DS_INPUT" ]]; then
  die "--ds is required"
fi

DATASET_TYPE="${DATASET_TYPE:-prehunter}"
DATASET_TYPE="${DATASET_TYPE,,}"
case "$DATASET_TYPE" in
  prehunter|hunter)
    ;;
  *)
    die "DATASET_TYPE must be prehunter or hunter (current: ${DATASET_TYPE})"
    ;;
esac

if [[ -n "$FASTA_NAME" && -n "$FASTA_ABS" ]]; then
  die "Use either --fasta or --fasta-abs, not both"
fi

if [[ "$FASTA_ENTRAP" == "1" && ( -n "$FASTA_NAME" || -n "$FASTA_ABS" ) ]]; then
  die "Use either --fasta-entrap or --fasta/--fasta-abs, not both"
fi

if [[ "$FASTA_ENTRAP" == "1" && -n "$FASTA_ENTRAP_KIND" ]]; then
  FASTA_KIND="$FASTA_ENTRAP_KIND"
fi

FASTA_KIND="${FASTA_KIND,,}"
case "$FASTA_KIND" in
  full|default)
    FASTA_KIND="full"
    ;;
  swiss|swissprot|swiss-prot)
    FASTA_KIND="swiss"
    ;;
  *)
    die "Invalid --fasta-kind '${FASTA_KIND}' (allowed: full|swiss)"
    ;;
esac

FASTA_ENTRAP_SHUFFLE_MODE="${FASTA_ENTRAP_SHUFFLE_MODE,,}"
case "$FASTA_ENTRAP_SHUFFLE_MODE" in
  fixed-r|fixed_r|rfixed|r-fixed|preserve-r|preserve_r)
    FASTA_ENTRAP_SHUFFLE_MODE="fixed-r"
    ;;
  simple|legacy|legacy-simple)
    FASTA_ENTRAP_SHUFFLE_MODE="simple"
    ;;
  *)
    die "Invalid FASTA_ENTRAP_SHUFFLE_MODE '${FASTA_ENTRAP_SHUFFLE_MODE}' (allowed: fixed-r|simple)"
    ;;
esac

if [[ "$TEST_MODE" == "1" ]]; then
  if [[ "$TEST_FULL" == "1" ]]; then
    TEST_COMPACT=0
  else
    TEST_COMPACT=1
  fi

  : "${OUTBASE:?OUTBASE not set}"
  outbase="${OUTBASE%/}"
  [[ -n "$outbase" ]] || outbase="/"
  EXPDIR="${outbase}/${EXPNAME}"
  S1DIR="${EXPDIR}/01_predlib"
  S2DIR="${EXPDIR}/02_tunelib"
  S3DIR="${EXPDIR}/03_tune_models"
  STEP4_BASEDIR_NAME="${STEP4_BASEDIR_NAME:-04_refined_lib}"
  S4DIR="${EXPDIR}/${STEP4_BASEDIR_NAME}"
  S5DIR="${EXPDIR}/05_quant"
  CALDIR="${EXPDIR}/cal_lib"
  export EXPDIR S1DIR S2DIR S3DIR S4DIR S5DIR CALDIR
else
  set_paths_for_exp
  STEP4_BASEDIR_NAME="${STEP4_BASEDIR_NAME:-04_refined_lib}"
  S4DIR="${EXPDIR}/${STEP4_BASEDIR_NAME}"
fi

PROGRESS_FILE="${EXPDIR}/progress.txt"
ALL_STARTCMDS_FILE="${EXPDIR}/all_startcmds.txt"
if [[ "$TEST_MODE" == "1" ]]; then
  echo "TEST MODE: progress file would be written: ${PROGRESS_FILE}"
  echo "TEST MODE: all_startcmds file would be written: ${ALL_STARTCMDS_FILE}"
else
  register_run_pid_file "${ORIGINAL_ARGS[@]}"
  if [[ "$RESUME_MODE" == "1" ]]; then
    [[ -f "$PROGRESS_FILE" ]] || die "--resume requires existing progress file: ${PROGRESS_FILE}"
    info "Resume mode enabled, using existing progress file: ${PROGRESS_FILE}"
  else
    if [[ -f "$PROGRESS_FILE" ]]; then
      local_backup_file="${PROGRESS_FILE}.before_rerun.$(date +%Y%m%d_%H%M%S).bak"
      if [[ -t 0 ]]; then
        printf 'Progress file exists: %s\nOverwrite it for a fresh run? [y/N] ' "$PROGRESS_FILE" >&2
        IFS= read -r overwrite_reply
      elif [[ -r /dev/tty ]]; then
        printf 'Progress file exists: %s\nOverwrite it for a fresh run? [y/N] ' "$PROGRESS_FILE" >/dev/tty
        IFS= read -r overwrite_reply </dev/tty
      else
        die "Progress file already exists and no interactive terminal is available for confirmation: ${PROGRESS_FILE}"
      fi

      case "${overwrite_reply,,}" in
        y|yes)
          mv "$PROGRESS_FILE" "$local_backup_file"
          info "Backed up existing progress file: ${local_backup_file}"
          ;;
        *)
          die "Aborted fresh run to avoid overwriting existing progress file: ${PROGRESS_FILE}"
          ;;
      esac
    fi
    : > "$PROGRESS_FILE"
    info "Initialized progress file: ${PROGRESS_FILE}"
  fi
fi

if [[ "$RESUME_MODE" == "1" && "$FASTA_ENTRAP" == "1" && "$FASTA_ENTRAP_SHUFFLE_MODE_EXPLICIT" == "0" ]]; then
  if grep -q 'step1_predlib.sh.*fasta=.*\.entrapment_combined\.fasta' "$PROGRESS_FILE" 2>/dev/null; then
    FASTA_ENTRAP_SHUFFLE_MODE="simple"
    info "Resume mode: detected legacy simple entrapment FASTA in progress.txt; keeping FASTA_ENTRAP_SHUFFLE_MODE=simple for this run"
  elif grep -q 'step1_predlib.sh.*fasta=.*\.entrapment_rfixed_combined\.fasta' "$PROGRESS_FILE" 2>/dev/null; then
    FASTA_ENTRAP_SHUFFLE_MODE="fixed-r"
    info "Resume mode: detected fixed-r entrapment FASTA in progress.txt; keeping FASTA_ENTRAP_SHUFFLE_MODE=fixed-r for this run"
  fi
fi

DATASET_CFG="$(resolve_dataset_cfg "$DS_INPUT")"
require_file "$DATASET_CFG"
parse_dataset_yaml "$DATASET_CFG" "$DATASET_TYPE"
if [[ "${MAPLOT_ORGANISM:-auto}" == "auto" ]]; then
  MAPLOT_ORGANISM="${DS_MAPLOT_ORGANISM:-arabidopsis}"
fi

# A dataset bundle may carry a canonical Triplemix A/B annotation next to its
# config.yaml.  Copy it into the experiment root, where the MAPLOT server
# discovers it automatically.  This keeps condition labels and paired test
# runs reproducible without embedding dataset-specific EX numbers in code.
DATASET_BUNDLE_DIR="$(cd "$(dirname "$DATASET_CFG")" && pwd)"
TRIPLEMIX_ANNOTATION_SOURCE="${DATASET_BUNDLE_DIR}/triplemix_sample_annotation.tsv"
TRIPLEMIX_ANNOTATION_TARGET="${EXPDIR}/triplemix_sample_annotation.tsv"
if [[ "${DS_MAPLOT_ORGANISM:-}" == "triplemix" && -f "$TRIPLEMIX_ANNOTATION_SOURCE" ]]; then
  if [[ "$TEST_MODE" == "1" ]]; then
    echo "TEST MODE: Triplemix annotation would be copied: ${TRIPLEMIX_ANNOTATION_SOURCE} -> ${TRIPLEMIX_ANNOTATION_TARGET}"
  else
    cp -f "$TRIPLEMIX_ANNOTATION_SOURCE" "$TRIPLEMIX_ANNOTATION_TARGET"
    require_file "$TRIPLEMIX_ANNOTATION_TARGET"
    info "Copied Triplemix A/B annotation to experiment base dir: ${TRIPLEMIX_ANNOTATION_TARGET}"
  fi
fi

DS_DATASET_INSTRUMENT_RAW="${DS_ACQUISITION_INSTRUMENT:-}"
DS_DATASET_INSTRUMENT_NORMALIZED=""
if [[ -n "$DS_DATASET_INSTRUMENT_RAW" ]]; then
  instrument_compact="${DS_DATASET_INSTRUMENT_RAW,,}"
  instrument_compact="${instrument_compact//[[:space:]_-]/}"
  case "$instrument_compact" in
    timstof|timstofpro|timstofpro2)
      DS_DATASET_INSTRUMENT_NORMALIZED="timstof"
      ;;
    elite)
      DS_DATASET_INSTRUMENT_NORMALIZED="elite"
      ;;
    exploris)
      DS_DATASET_INSTRUMENT_NORMALIZED="exploris"
      ;;
    qe|qexactive|qexactivehf|qexactivehfx)
      DS_DATASET_INSTRUMENT_NORMALIZED="qe"
      ;;
    *)
      DS_DATASET_INSTRUMENT_NORMALIZED="$instrument_compact"
      ;;
  esac
fi

BASE_DIR_ABS="$(resolve_base_dir_abs "${DS_DATA_BASE_DIR}")"
[[ -d "$BASE_DIR_ABS" ]] || die "Missing base_dir directory: ${BASE_DIR_ABS}"

if [[ "$FASTA_ENTRAP" == "1" ]]; then
  case "$FASTA_KIND" in
    full)
      FASTA_FROM_YAML="${DS_FASTA:-}"
      FASTA_YAML_KEY="fasta.fasta"
      ;;
    swiss)
      FASTA_FROM_YAML="${DS_FASTA_SWISS:-}"
      FASTA_YAML_KEY="fasta.fasta_swiss"
      ;;
  esac
  [[ -n "$FASTA_FROM_YAML" ]] || die "--fasta-entrap ${FASTA_KIND} requested, but dataset YAML key ${FASTA_YAML_KEY} is empty"
  [[ "$FASTA_FROM_YAML" != */* ]] || die "dataset YAML ${FASTA_YAML_KEY} must be a FASTABASE filename (no path). Use --fastaabs to override."
  fasta_base="${FASTABASE%/}"
  [[ -n "$fasta_base" ]] || fasta_base="/"
  FASTA_ENTRAP_SOURCE="${fasta_base}/${FASTA_FROM_YAML}"
  FASTA_ENTRAP_TARGET="${fasta_base}/$(entrapment_filename_for_fasta "$FASTA_FROM_YAML" "$FASTA_ENTRAP_SHUFFLE_MODE")"
  ensure_entrapment_fasta "$FASTA_ENTRAP_SOURCE" "$FASTA_ENTRAP_TARGET"
  FASTA_FILE="$FASTA_ENTRAP_TARGET"
elif [[ -n "$FASTA_NAME" ]]; then
  [[ "$FASTA_NAME" != */* ]] || die "--fasta expects only a filename from FASTABASE"
  fasta_base="${FASTABASE%/}"
  [[ -n "$fasta_base" ]] || fasta_base="/"
  FASTA_FILE="${fasta_base}/${FASTA_NAME}"
elif [[ -n "$FASTA_ABS" ]]; then
  [[ "$FASTA_ABS" == /* ]] || die "--fasta-abs expects an absolute path"
  FASTA_FILE="$FASTA_ABS"
else
  case "$FASTA_KIND" in
    full)
      FASTA_FROM_YAML="${DS_FASTA:-}"
      FASTA_YAML_KEY="fasta.fasta"
      ;;
    swiss)
      FASTA_FROM_YAML="${DS_FASTA_SWISS:-}"
      FASTA_YAML_KEY="fasta.fasta_swiss"
      ;;
  esac
  [[ -n "$FASTA_FROM_YAML" ]] || die "No FASTA provided via CLI and dataset YAML key ${FASTA_YAML_KEY} is empty"
  [[ "$FASTA_FROM_YAML" != */* ]] || die "dataset YAML ${FASTA_YAML_KEY} must be a FASTABASE filename (no path). Use --fastaabs to override."
  fasta_base="${FASTABASE%/}"
  [[ -n "$fasta_base" ]] || fasta_base="/"
  FASTA_FILE="${fasta_base}/${FASTA_FROM_YAML}"
fi

if [[ "$FASTA_ENTRAP" == "1" && "$TEST_MODE" == "1" && "$FASTA_ENTRAP_ACTION" == "create" ]]; then
  :
else
  require_file "$FASTA_FILE"
fi

COPIED_FASTA="${EXPDIR}/$(basename "$FASTA_FILE")"
if [[ "$TEST_MODE" == "1" ]]; then
  if [[ "$FASTA_ENTRAP" == "1" ]]; then
    case "$FASTA_ENTRAP_ACTION" in
      existing)
        echo "TEST MODE: FASTA entrapment exists and would be used: ${FASTA_ENTRAP_TARGET}"
        echo "TEST MODE: FASTA entrapment shuffle mode: ${FASTA_ENTRAP_SHUFFLE_MODE}"
        ;;
      create)
        echo "TEST MODE: FASTA entrapment would be created: ${FASTA_ENTRAP_SOURCE} -> ${FASTA_ENTRAP_TARGET}"
        echo "TEST MODE: FASTA entrapment shuffle mode: ${FASTA_ENTRAP_SHUFFLE_MODE}"
        ;;
    esac
  fi
  echo "TEST MODE: FASTA would be copied: ${FASTA_FILE} -> ${COPIED_FASTA}"
else
  src_real="$(readlink -f "$FASTA_FILE")"
  dst_real="$(readlink -m "$COPIED_FASTA")"
  if [[ "$src_real" != "$dst_real" ]]; then
    cp -f "$FASTA_FILE" "$COPIED_FASTA"
    info "Copied FASTA to experiment base dir: ${COPIED_FASTA}"
  else
    info "FASTA already located in experiment base dir: ${COPIED_FASTA}"
  fi
  require_file "$COPIED_FASTA"
fi

normalize_split_list "${DS_TRAIN_FILES:-}" DS_TRAIN_FILES_ARR
normalize_split_list "${DS_TEST_FILES:-}" DS_TEST_FILES_ARR

[[ ${#DS_TRAIN_FILES_ARR[@]} -ge 1 ]] || die "Dataset train_files is empty; need at least one file for make_cal_lib step"

CAL_FILE_SOURCE_LABEL="train_files"
CAL_FILE_SOURCE_DIR="${DS_DIR_TRAIN:-}"
CAL_FILE_SOURCE_NAME="${DS_TRAIN_FILES_ARR[0]:-}"

FIRST_TRAIN_FILE_RAW="$(resolve_existing_data_file_path "$BASE_DIR_ABS" "$CAL_FILE_SOURCE_DIR" "$CAL_FILE_SOURCE_NAME" || true)"
[[ -n "$FIRST_TRAIN_FILE_RAW" ]] || die "Could not resolve first file '${CAL_FILE_SOURCE_NAME}' from ${CAL_FILE_SOURCE_LABEL}. Tried: $(resolve_attempted_data_paths "$BASE_DIR_ABS" "$CAL_FILE_SOURCE_DIR" "$CAL_FILE_SOURCE_NAME")"

STEP2_FILE_SOURCE_LABEL="train_files"
STEP2_FILE_SOURCE_DIR="${DS_DIR_TRAIN:-}"
STEP2_FILE_NAMES=("${DS_TRAIN_FILES_ARR[@]}")

STEP2_RAW_FILES_RESOLVED=()
for step2_file_name in "${STEP2_FILE_NAMES[@]}"; do
  step2_raw_file="$(resolve_existing_data_file_path "$BASE_DIR_ABS" "$STEP2_FILE_SOURCE_DIR" "$step2_file_name" || true)"
  [[ -n "$step2_raw_file" ]] || die "Could not resolve Step2 file '${step2_file_name}' from ${STEP2_FILE_SOURCE_LABEL}. Tried: $(resolve_attempted_data_paths "$BASE_DIR_ABS" "$STEP2_FILE_SOURCE_DIR" "$step2_file_name")"
  STEP2_RAW_FILES_RESOLVED+=("$step2_raw_file")
done

STEP5_FILE_SOURCE_LABEL="test_files"
STEP5_FILE_SOURCE_DIR="${DS_DIR_TEST:-}"
STEP5_FILE_NAMES=()
if (( ${#DS_TEST_FILES_ARR[@]} > 0 )); then
  STEP5_FILE_NAMES=("${DS_TEST_FILES_ARR[@]}")
else
  warn "Dataset test_files is empty; Step5 falls back to train_files"
  STEP5_FILE_SOURCE_LABEL="train_files"
  STEP5_FILE_SOURCE_DIR="${DS_DIR_TRAIN:-}"
  STEP5_FILE_NAMES=("${DS_TRAIN_FILES_ARR[@]}")
fi

[[ ${#STEP5_FILE_NAMES[@]} -gt 0 ]] || die "Step5 input file list is empty (${STEP5_FILE_SOURCE_LABEL})."

STEP5_RAW_FILES_RESOLVED=()
for step5_file_name in "${STEP5_FILE_NAMES[@]}"; do
  step5_raw_file="$(resolve_existing_data_file_path "$BASE_DIR_ABS" "$STEP5_FILE_SOURCE_DIR" "$step5_file_name" || true)"
  [[ -n "$step5_raw_file" ]] || die "Could not resolve Step5 file '${step5_file_name}' from ${STEP5_FILE_SOURCE_LABEL}. Tried: $(resolve_attempted_data_paths "$BASE_DIR_ABS" "$STEP5_FILE_SOURCE_DIR" "$step5_file_name")"
  STEP5_RAW_FILES_RESOLVED+=("$step5_raw_file")
done

info "Dataset YAML parsed successfully"
if [[ "$TEST_COMPACT" == "1" ]]; then
  print_section "Inputs"
  print_kv "Dataset" "${DS_META_DATASET_ID}"
  print_kv "Source" "${DS_SOURCE_NAME:-<legacy>}"
  print_kv "Instrument" "${DS_DATASET_INSTRUMENT_NORMALIZED:-<unset>}"
  print_kv "Train/Test" "${#DS_TRAIN_FILES_ARR[@]}/${#DS_TEST_FILES_ARR[@]}"
  print_kv "Profile" "${PROFILE}"
  print_kv "Cal lib" "$([[ "${NO_CAL_LIB:-0}" == "1" ]] && echo "skipped" || echo "enabled")"
  print_kv "FASTA kind" "${FASTA_KIND}"
  if [[ "$FASTA_ENTRAP" == "1" ]]; then
    print_kv "FASTA entrap" "${FASTA_ENTRAP_ACTION} (${FASTA_ENTRAP_TARGET})"
    print_kv "Entrap shuffle" "${FASTA_ENTRAP_SHUFFLE_MODE}"
  fi
  print_kv "Step4 targets" "${STEP4_MODEL_TARGETS:-all}"
  print_kv "Step4 lr" "${STEP4_LR_SELECT:-all}"
  print_kv "MA plot report" "$([[ "$ENABLE_MAPLOT_REPORT" == "1" ]] && echo "on" || echo "off")"
  print_kv "Experiment" "${EXPDIR}"
else
  echo "Dataset config : ${DATASET_CFG}"
  echo "Dataset id     : ${DS_META_DATASET_ID}"
  echo "Dataset type   : ${DS_DATASET_TYPE}"
  echo "Source name    : ${DS_SOURCE_NAME:-<legacy>}"
  echo "Data base dir  : ${DS_DATA_BASE_DIR}"
  echo "Base dir abs   : ${BASE_DIR_ABS}"
  echo "Instrument raw : ${DS_DATASET_INSTRUMENT_RAW:-<unset>}"
  echo "Instrument norm: ${DS_DATASET_INSTRUMENT_NORMALIZED:-<unset>}"
  echo "Train files    : ${#DS_TRAIN_FILES_ARR[@]}"
  echo "Test files     : ${#DS_TEST_FILES_ARR[@]}"
  echo "FASTA src      : ${FASTA_FILE}"
  echo "FASTA kind     : ${FASTA_KIND}"
  if [[ "$FASTA_ENTRAP" == "1" ]]; then
    echo "FASTA entrap   : ${FASTA_ENTRAP_ACTION}"
    echo "Entrap shuffle : ${FASTA_ENTRAP_SHUFFLE_MODE}"
    echo "Entrap source  : ${FASTA_ENTRAP_SOURCE}"
    echo "Entrap target  : ${FASTA_ENTRAP_TARGET}"
  fi
  echo "FASTA run copy : ${COPIED_FASTA}"
  echo "Progress file  : ${PROGRESS_FILE}"
  echo "Profile file -f: ${FIRST_TRAIN_FILE_RAW}"
  echo "Profile file src: ${CAL_FILE_SOURCE_LABEL}"
  echo "Step2 files src : ${STEP2_FILE_SOURCE_LABEL}"
  echo "Step2 files n   : ${#STEP2_RAW_FILES_RESOLVED[@]}"
  echo "Step5 files src : ${STEP5_FILE_SOURCE_LABEL}"
  echo "Step5 files n   : ${#STEP5_RAW_FILES_RESOLVED[@]}"
  echo "Step5 report q  : ${STEP5_REPORT_QVALUE:-1}"
  echo "Analysis qvalue : ${QVALUE}"
  echo "Report decoys   : ${enable_REPORT_DECOYS:-0}"
  echo "Profile        : ${PROFILE}"
  echo "Cal lib        : $([[ "${NO_CAL_LIB:-0}" == "1" ]] && echo "skipped" || echo "enabled")"
  echo "Profile parallel: ${PROFILE_PARALLEL}"
  echo "Step4 out dir  : ${STEP4_BASEDIR_NAME}"
  echo "Step4 targets  : ${STEP4_MODEL_TARGETS:-all}"
  echo "Step4 LR select: ${STEP4_LR_SELECT:-all}"
  if [[ -n "${DS_FASTA:-}" ]]; then
    echo "YAML fasta     : ${DS_FASTA}"
  fi
  if [[ -n "${DS_FASTA_SWISS:-}" ]]; then
    echo "YAML fasta swiss: ${DS_FASTA_SWISS}"
  fi
  echo "Experiment dir : ${EXPDIR}"
fi

echo ""
STEP1_SCRIPT="${PIPELINE_ROOT}/bin/step1_predlib.sh"
require_file "$STEP1_SCRIPT"

STEP1_CMD=(
  bash "$STEP1_SCRIPT"
  --expname "$EXPNAME"
  --fastaabs "$COPIED_FASTA"
)

if (( VERBOSE > 0 )); then
  STEP1_CMD+=(--verbose "$VERBOSE")
fi

for key in "${CLI_OVERRIDES[@]}"; do
  STEP1_CMD+=(--set "${key}=${!key}")
done

STEP1_OUTLIB="${S1DIR}/step1_${EXPNAME}_predicted.parquet"

MAKE_CAL_SCRIPT="${PIPELINE_ROOT}/bin/make_cal_lib.sh"
require_file "$MAKE_CAL_SCRIPT"

STEP2_SCRIPT="${PIPELINE_ROOT}/bin/step2_tunelib.sh"
require_file "$STEP2_SCRIPT"

STEP3_SCRIPT="${PIPELINE_ROOT}/bin/step3_tune_models.sh"
require_file "$STEP3_SCRIPT"

STEP4_SCRIPT="${PIPELINE_ROOT}/bin/step4_create_tuned_predlib.sh"
require_file "$STEP4_SCRIPT"

STEP5_SCRIPT="${PIPELINE_ROOT}/bin/step5_quant.sh"
require_file "$STEP5_SCRIPT"

build_make_cal_cmd

RUN_STEP1=1
RUN_MAKE_CAL=1
RUN_STEP2=1
RUN_STEP3=1
RUN_STEP4=1
RUN_STEP5=1
PARTIAL_RESUME_STEP4=0
PARTIAL_RESUME_STEP5=0
PARTIAL_RESUME_STEP2=0
PARTIAL_RESUME_STEP3=0

if [[ "$RESUME_MODE" == "1" ]]; then
  RESUME_FROM_STEP="$(determine_resume_step)"
  if [[ -z "$RESUME_FROM_STEP" ]]; then
    info "All steps already completed for this configuration according to ${PROGRESS_FILE}. Nothing to resume."
    exit 0
  fi

  case "$RESUME_FROM_STEP" in
    step1_predlib.sh)
      RUN_STEP1=1
      RUN_MAKE_CAL=1
      RUN_STEP2=1
      RUN_STEP3=1
      RUN_STEP4=1
      RUN_STEP5=1
      ;;
    make_cal_lib.sh)
      RUN_STEP1=0
      RUN_MAKE_CAL=1
      RUN_STEP2=1
      RUN_STEP3=1
      RUN_STEP4=1
      RUN_STEP5=1
      ;;
    step2_tunelib.sh)
      RUN_STEP1=0
      RUN_MAKE_CAL=0
      RUN_STEP2=1
      RUN_STEP3=1
      RUN_STEP4=1
      RUN_STEP5=1
      if (( MULTI_PROFILE == 1 )); then
        PARTIAL_RESUME_STEP2=1
      fi
      ;;
    step3_tune_models.sh)
      RUN_STEP1=0
      RUN_MAKE_CAL=0
      RUN_STEP2=0
      RUN_STEP3=1
      RUN_STEP4=1
      RUN_STEP5=1
      if (( MULTI_PROFILE == 1 )); then
        PARTIAL_RESUME_STEP3=1
      fi
      ;;
    step4_create_tuned_predlib.sh)
      RUN_STEP1=0
      RUN_MAKE_CAL=0
      RUN_STEP2=0
      RUN_STEP3=0
      RUN_STEP4=1
      RUN_STEP5=1
      PARTIAL_RESUME_STEP4=1
      PARTIAL_RESUME_STEP5=1
      ;;
    step5_quant.sh)
      RUN_STEP1=0
      RUN_MAKE_CAL=0
      RUN_STEP2=0
      RUN_STEP3=0
      RUN_STEP4=0
      RUN_STEP5=1
      PARTIAL_RESUME_STEP5=1
      ;;
    *)
      die "Unexpected resume step: ${RESUME_FROM_STEP}"
      ;;
  esac

  info "Resume mode: first incomplete step is ${RESUME_FROM_STEP}"
  if (( PARTIAL_RESUME_STEP2 == 1 || PARTIAL_RESUME_STEP3 == 1 )); then
    info "Resume mode: preserving progress log and completed profile output dirs for partial ${RESUME_FROM_STEP} rerun"
    cleanup_profile_resume_step "$RESUME_FROM_STEP"
  elif (( PARTIAL_RESUME_STEP4 == 1 || PARTIAL_RESUME_STEP5 == 1 )); then
    info "Resume mode: preserving progress log and output dirs for partial Step4/Step5 rerun"
  else
    if [[ "$TEST_MODE" == "1" ]]; then
      echo "TEST MODE: would truncate progress entries from ${RESUME_FROM_STEP} onwards"
    else
      truncate_progress_from_step "$RESUME_FROM_STEP"
    fi
    cleanup_from_step "$RESUME_FROM_STEP"
  fi
fi

if [[ "${NO_CAL_LIB:-0}" == "1" ]]; then
  RUN_MAKE_CAL=0
  info "Option --no-cal-lib/NO_CAL_LIB=1 enabled: make_cal_lib.sh is skipped and downstream steps will not use --ref."
fi

if [[ "$STOP_AFTER_STEP4" == "1" ]]; then
  RUN_STEP5=0
  PARTIAL_RESUME_STEP5=0
  LOCAL_STEP5_SKIPPED_AFTER_STEP4=1
  info "Option --stop-after-step4 enabled: step5_quant.sh will be skipped."
  if [[ "$RESUME_MODE" == "1" && "${RESUME_FROM_STEP:-}" == "step5_quant.sh" ]]; then
    info "Resume detected first incomplete step as step5_quant.sh, but it is intentionally skipped."
  fi
else
  LOCAL_STEP5_SKIPPED_AFTER_STEP4=0
fi

PROFILE_PARALLEL_THREADS=""
if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
  PROFILE_PARALLEL_THREADS=$(( THREADS / ${#STEP2_PROFILE_LIST[@]} ))
  if (( PROFILE_PARALLEL_THREADS < 1 )); then
    PROFILE_PARALLEL_THREADS=1
  fi
fi

echo ""
echo "Path checks"
check_dataset_group "Train" "$BASE_DIR_ABS" "${DS_DIR_TRAIN:-}" DS_TRAIN_FILES_ARR
check_dataset_group "Test" "$BASE_DIR_ABS" "${DS_DIR_TEST:-}" DS_TEST_FILES_ARR

STEP4_THREADS_OVERRIDE=""
STEP5_THREADS_OVERRIDE=""
STEP5_PARALLEL_MODE=0
STEP4_PARALLEL_MODE=0
if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
  STEP4_THREADS_OVERRIDE="$PROFILE_PARALLEL_THREADS"
  STEP5_THREADS_OVERRIDE="$PROFILE_PARALLEL_THREADS"
  STEP4_PARALLEL_MODE=1
  STEP5_PARALLEL_MODE=1
fi

if [[ "$TEST_MODE" != "1" ]]; then
  if [[ "$RESUME_MODE" == "1" ]]; then
    if [[ -f "$ALL_STARTCMDS_FILE" ]]; then
      all_startcmds_backup="${ALL_STARTCMDS_FILE}.before_resume.$(date +%Y%m%d_%H%M%S).bak"
      cp -f "$ALL_STARTCMDS_FILE" "$all_startcmds_backup"
      info "Backed up all_startcmds before resume: ${all_startcmds_backup}"
    fi
  fi

  all_startcmds_tmp="${ALL_STARTCMDS_FILE}.tmp.$$"
  {
    if [[ "$RESUME_MODE" == "1" ]]; then
      echo ""
      echo "# ============================================================"
      echo "# RESUME RUN SECTION $(date +'%Y-%m-%dT%H:%M:%S%z')"
      echo "# resume start step: ${RESUME_FROM_STEP}"
      echo "# ============================================================"
      echo ""
    fi

    echo "# run_all invocation command"
    printf '%q ' "$0"
    printf '%q ' "$@"
    echo
    echo ""
    echo "# planned step commands"
    if (( RUN_STEP1 == 1 )); then
      echo "# step1_predlib wrapper"
      printf '%q ' "${STEP1_CMD[@]}"
      echo
    else
      echo "# step1_predlib wrapper (skipped in resume mode)"
    fi

    if (( RUN_MAKE_CAL == 1 )); then
      echo "# calibration helper (make_cal_lib) wrapper"
      echo "# selected profile: ${PROFILE}"
      if (( MULTI_PROFILE == 1 )); then
        if [[ "$PROFILE_PARALLEL" == "1" ]]; then
          echo "# parallel mode: ${#STEP2_PROFILE_LIST[@]} concurrent runs, THREADS per profile: ${PROFILE_PARALLEL_THREADS}"
        else
          echo "# sequential-all mode: 3 runs handled by run_all.sh (balanced, fast, slow)"
        fi
        for cal_profile in "${STEP2_PROFILE_LIST[@]}"; do
          MAKE_CAL_PLAN_CMD=(
            bash "$MAKE_CAL_SCRIPT"
            --expname "$EXPNAME"
            --fastaabs "$COPIED_FASTA"
            --lib "$STEP1_OUTLIB"
            --profile "$cal_profile"
            --f "$FIRST_TRAIN_FILE_RAW"
          )
          if (( VERBOSE > 0 )); then
            MAKE_CAL_PLAN_CMD+=(--verbose "$VERBOSE")
          fi
          for key in "${CLI_OVERRIDES[@]}"; do
            if [[ "$PROFILE_PARALLEL" == "1" && "$key" == "THREADS" ]]; then
              continue
            fi
            MAKE_CAL_PLAN_CMD+=(--set "${key}=${!key}")
          done
          if [[ "$PROFILE_PARALLEL" == "1" ]]; then
            MAKE_CAL_PLAN_CMD+=(--set "THREADS=${PROFILE_PARALLEL_THREADS}")
          fi
          printf '%q ' "${MAKE_CAL_PLAN_CMD[@]}"
          echo
        done
      else
        printf '%q ' "${MAKE_CAL_CMD[@]}"
        echo
      fi
    else
      if [[ "${NO_CAL_LIB:-0}" == "1" ]]; then
        echo "# calibration helper (make_cal_lib) wrapper (skipped by --no-cal-lib)"
      else
        echo "# calibration helper (make_cal_lib) wrapper (skipped in resume mode)"
      fi
    fi

	    if (( RUN_STEP2 == 1 )); then
	      echo "# step2_tunelib wrapper"
	      echo "# selected profile: ${PROFILE}"
	      if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
	        echo "# parallel mode: ${#STEP2_PROFILE_LIST[@]} concurrent runs, THREADS per profile: ${PROFILE_PARALLEL_THREADS}"
	        for step2_profile in "${STEP2_PROFILE_LIST[@]}"; do
	          if (( PARTIAL_RESUME_STEP2 == 1 )) && step_profile_done "step2_tunelib.sh" "$step2_profile"; then
	            info "Step2 resume: skipping completed profile=${step2_profile}"
	            continue
	          fi
	          STEP2_REF_FILE=""
	          if [[ "${NO_CAL_LIB:-0}" != "1" ]]; then
	            STEP2_REF_FILE="${CALDIR}/${step2_profile}/calib_ref_${step2_profile}.parquet"
	          fi
	          build_step2_tunelib_cmd "$step2_profile" "$STEP2_REF_FILE" "$PROFILE_PARALLEL_THREADS"
	          printf '%q ' "${STEP2_TUNELIB_CMD[@]}"
	          echo
	        done
	      else
	        for step2_profile in "${STEP2_PROFILE_LIST[@]}"; do
	          if (( PARTIAL_RESUME_STEP2 == 1 )) && step_profile_done "step2_tunelib.sh" "$step2_profile"; then
	            info "Step2 resume: skipping completed profile=${step2_profile}"
	            continue
	          fi
	          STEP2_REF_FILE=""
	          if [[ "${NO_CAL_LIB:-0}" != "1" ]]; then
	            STEP2_REF_FILE="${CALDIR}/${step2_profile}/calib_ref_${step2_profile}.parquet"
	          fi
	          build_step2_tunelib_cmd "$step2_profile" "$STEP2_REF_FILE"
	          printf '%q ' "${STEP2_TUNELIB_CMD[@]}"
	          echo
	        done
	      fi
	    else
	      echo "# step2_tunelib wrapper (skipped in resume mode)"
	    fi

	    if (( RUN_STEP3 == 1 )); then
	      echo "# step3_tune_models wrapper"
	      echo "# selected profile: ${PROFILE}"
	      if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
	        echo "# parallel mode: ${#STEP2_PROFILE_LIST[@]} concurrent runs, THREADS per profile: ${PROFILE_PARALLEL_THREADS}"
	        for step3_profile in "${STEP2_PROFILE_LIST[@]}"; do
	          if (( PARTIAL_RESUME_STEP3 == 1 )) && step_profile_done "step3_tune_models.sh" "$step3_profile"; then
	            info "Step3 resume: skipping completed profile=${step3_profile}"
	            continue
	          fi
	          build_step3_tune_models_cmd "$step3_profile" "$PROFILE_PARALLEL_THREADS"
	          printf '%q ' "${STEP3_TUNE_MODELS_CMD[@]}"
	          echo
	        done
	      else
	        for step3_profile in "${STEP2_PROFILE_LIST[@]}"; do
	          if (( PARTIAL_RESUME_STEP3 == 1 )) && step_profile_done "step3_tune_models.sh" "$step3_profile"; then
	            info "Step3 resume: skipping completed profile=${step3_profile}"
	            continue
	          fi
	          build_step3_tune_models_cmd "$step3_profile"
	          printf '%q ' "${STEP3_TUNE_MODELS_CMD[@]}"
	          echo
	        done
	      fi
	    else
	      echo "# step3_tune_models wrapper (skipped in resume mode)"
	    fi

    if (( RUN_STEP4 == 1 )); then
      echo "# step4_create_tuned_predlib wrapper"
      echo "# selected profile: ${PROFILE}"
      echo "# model targets: ${STEP4_MODEL_TARGETS:-all}"
      echo "# lr selection: ${STEP4_LR_SELECT:-all}"
      if step4_inputs_ready_for_planning; then
        iterate_step4_runs step4_plan_print_callback
      else
        echo "# step4 commands are resolved after step3 outputs exist (lr_*/model artifacts)."
      fi
    else
      echo "# step4_create_tuned_predlib wrapper (skipped in resume mode)"
    fi

    if (( RUN_STEP5 == 1 )); then
      echo "# step5_quant wrapper"
      echo "# selected profile: ${PROFILE}"
      echo "# baseline + tuned matrix"
      echo "# tuned model targets: ${STEP4_MODEL_TARGETS:-all}"
      echo "# tuned lr selection: ${STEP4_LR_SELECT:-all}"
      if (( PARTIAL_RESUME_STEP5 == 1 )) && step5_baseline_complete; then
        echo "# baseline skipped (already completed in progress.txt)"
      else
        step5_plan_baseline_cmd ""  # baseline: full THREADS; tuned: THREADS/3 per profile in parallel mode
      fi
      if step4_inputs_ready_for_planning; then
        iterate_step4_runs step5_plan_tuned_callback
      else
        echo "# step5 tuned commands are resolved after step3 outputs exist (lr_*/model artifacts)."
      fi
    else
      echo "# step5_quant wrapper (skipped in resume mode)"
    fi
  } > "$all_startcmds_tmp"

  if [[ "$RESUME_MODE" == "1" ]]; then
    cat "$all_startcmds_tmp" >> "$ALL_STARTCMDS_FILE"
    rm -f "$all_startcmds_tmp"
  else
    mv "$all_startcmds_tmp" "$ALL_STARTCMDS_FILE"
  fi
  info "Saved run-all and planned substep commands: ${ALL_STARTCMDS_FILE}"
fi

if [[ "$TEST_MODE" == "1" ]]; then
  print_section "Test Mode"
  print_ok "Validation successful (no step executed)."
  print_progress_overview
  print_section "Execution Plan"
  print_execution_plan
  if [[ "$RESUME_MODE" == "1" ]]; then
    print_kv "Resume start" "${RESUME_FROM_STEP}"
    resume_baseline_done=0
    if step5_baseline_complete; then
      resume_baseline_done=1
    fi

    case "${RESUME_FROM_STEP}" in
      step4_create_tuned_predlib.sh)
        if (( resume_baseline_done == 1 )); then
          print_kv "Resume next" "step4 rerun, then step5 tuned (baseline already completed)"
        else
          print_kv "Resume next" "step4 rerun, then step5 baseline+tuned"
        fi
        ;;
      step5_quant.sh)
        if (( resume_baseline_done == 1 )); then
          print_kv "Resume next" "step5 tuned only (baseline already completed)"
        else
          print_kv "Resume next" "step5 baseline then tuned"
        fi
        ;;
    esac
  fi

  if [[ "$TEST_COMPACT" == "1" ]]; then
    print_section "Summary"
    print_kv "Experiment" "${EXPNAME}"
    print_kv "Dataset" "${DS_META_DATASET_ID}"
    print_kv "Profile" "${PROFILE}"
    print_kv "Cal lib" "$([[ "${NO_CAL_LIB:-0}" == "1" ]] && echo "skipped" || echo "enabled")"
    print_kv "FASTA kind" "${FASTA_KIND}"
    if [[ "$FASTA_ENTRAP" == "1" ]]; then
      print_kv "FASTA entrap" "${FASTA_ENTRAP_ACTION} ($(basename "$FASTA_ENTRAP_TARGET"))"
      print_kv "Entrap shuffle" "${FASTA_ENTRAP_SHUFFLE_MODE}"
    fi
    if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
      print_kv "Parallel mode" "on (${#STEP2_PROFILE_LIST[@]} profiles)"
      print_kv "Threads/profile" "${PROFILE_PARALLEL_THREADS}"
    else
      print_kv "Parallel mode" "off"
    fi
    if [[ "$RAM_LOG_ENABLE" == "1" ]]; then
      print_kv "RAM log" "on (${RAM_LOG_STEPS}; ${RAM_LOG_INTERVAL}s)"
    else
      print_kv "RAM log" "off"
    fi
    if [[ "$RAM_PLOT_ENABLE" == "1" ]]; then
      print_kv "RAM plot" "on (one plot per RAM log)"
    else
      print_kv "RAM plot" "off"
    fi
    if [[ "$ENABLE_MAPLOT_REPORT" == "1" ]]; then
      print_kv "MA plot report" "on (${MAPLOT_ORGANISM})"
    else
      print_kv "MA plot report" "off"
    fi
    print_kv "Mod config" "${MOD_CONFIG:-legacy}"
    print_kv "Mods parsed" "$(mod_config_summary)"
    print_kv "Step4 targets" "${STEP4_MODEL_TARGETS:-all}"
    print_kv "Step4 lr" "${STEP4_LR_SELECT:-all}"
    print_kv "Step2 files" "${#STEP2_RAW_FILES_RESOLVED[@]}"
    print_kv "Step5 files" "${#STEP5_RAW_FILES_RESOLVED[@]}"
    print_kv "Step5 report q" "${STEP5_REPORT_QVALUE:-1}"
    print_kv "Analysis qvalue" "${QVALUE}"
    print_kv "Report decoys" "${enable_REPORT_DECOYS:-0}"

    if step4_inputs_ready_for_planning; then
      resolve_step4_combo_count
      print_kv "Step4 combos" "${STEP4_COMBO_COUNT}"
      print_kv "Step5 tuned runs" "${STEP4_COMBO_COUNT} (+ baseline)"
    else
      print_warn_line "Step4/5 combo expansion deferred until Step3 artifacts exist (lr_*/models)."
    fi

    if [[ "$RESUME_MODE" == "1" ]]; then
      print_ok "Resume check completed; cleanup target shown above."
    fi

    print_kv "Hint" "Use --test-full to print full planned commands and all simulated progress lines."
    # Compact distillation summary line for --test
    print_kv "Step4 distill levels" "${STEP4_DISTILL_LEVELS:-0}"
    exit 0
  fi

  echo ""
  print_section "Mod Config"
  print_mod_config_details

  echo ""
  # Detailed distillation information for --test-full
  print_section "Step4 Distillation"
  print_kv "Requested levels" "${STEP4_DISTILL_LEVELS:-0}"
  if step4_inputs_ready_for_planning; then
    resolve_step4_distill_levels distill_levels
    resolve_step4_combo_count
    # join levels for display
    distill_levels_joined="$(printf '%s,' "${distill_levels[@]:-}" | sed 's/,$//')"
    print_kv "Resolved levels" "${distill_levels_joined:-<none>}"
    print_kv "Step4 combos" "${STEP4_COMBO_COUNT}"
    echo "# Step4 planned commands for distillation combos:"
    iterate_step4_runs step4_plan_print_callback
  else
    print_warn_line "Step4 distillation combos will be resolved after Step3 outputs exist (lr_*/models)."
  fi

  echo ""
  echo "Planned start commands:"
  if [[ "$RAM_LOG_ENABLE" == "1" ]]; then
    echo "# RAM logging enabled: steps=${RAM_LOG_STEPS}; interval=${RAM_LOG_INTERVAL}s"
    echo "# RAM logs are written into per-step/per-run output folders"
    if [[ "$RAM_PLOT_ENABLE" == "1" ]]; then
      echo "# RAM plot enabled: one PNG is written next to each RAM log file"
    fi
  fi
  if (( RUN_STEP1 == 1 )); then
    echo "# step1_predlib wrapper"
    printf '%q ' "${STEP1_CMD[@]}"
    echo
  else
    echo "# step1_predlib wrapper (skipped in resume mode)"
  fi
  if (( RUN_MAKE_CAL == 1 )); then
    echo "# calibration helper (make_cal_lib) wrapper"
    echo "# selected profile: ${PROFILE}"
    if (( MULTI_PROFILE == 1 )); then
      if [[ "$PROFILE_PARALLEL" == "1" ]]; then
        echo "# parallel mode: ${#STEP2_PROFILE_LIST[@]} concurrent runs, THREADS per profile: ${PROFILE_PARALLEL_THREADS}"
      else
        echo "# sequential-all mode: 3 runs handled by run_all.sh (balanced, fast, slow)"
      fi
      for cal_profile in "${STEP2_PROFILE_LIST[@]}"; do
        MAKE_CAL_PLAN_CMD=(
          bash "$MAKE_CAL_SCRIPT"
          --expname "$EXPNAME"
          --fastaabs "$COPIED_FASTA"
          --lib "$STEP1_OUTLIB"
          --profile "$cal_profile"
          --f "$FIRST_TRAIN_FILE_RAW"
        )
        if (( VERBOSE > 0 )); then
          MAKE_CAL_PLAN_CMD+=(--verbose "$VERBOSE")
        fi
        for key in "${CLI_OVERRIDES[@]}"; do
          if [[ "$PROFILE_PARALLEL" == "1" && "$key" == "THREADS" ]]; then
            continue
          fi
          MAKE_CAL_PLAN_CMD+=(--set "${key}=${!key}")
        done
        if [[ "$PROFILE_PARALLEL" == "1" ]]; then
          MAKE_CAL_PLAN_CMD+=(--set "THREADS=${PROFILE_PARALLEL_THREADS}")
        fi
        printf '%q ' "${MAKE_CAL_PLAN_CMD[@]}"
        echo
      done
    else
      printf '%q ' "${MAKE_CAL_CMD[@]}"
      echo
    fi
  else
    if [[ "${NO_CAL_LIB:-0}" == "1" ]]; then
      echo "# calibration helper (make_cal_lib) wrapper (skipped by --no-cal-lib)"
    else
      echo "# calibration helper (make_cal_lib) wrapper (skipped in resume mode)"
    fi
  fi
  if (( RUN_STEP2 == 1 )); then
    echo "# step2_tunelib wrapper"
    echo "# selected profile: ${PROFILE}"
    if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
      echo "# parallel mode: ${#STEP2_PROFILE_LIST[@]} concurrent runs, THREADS per profile: ${PROFILE_PARALLEL_THREADS}"
      for step2_profile in "${STEP2_PROFILE_LIST[@]}"; do
        STEP2_REF_FILE=""
        if [[ "${NO_CAL_LIB:-0}" != "1" ]]; then
          STEP2_REF_FILE="${CALDIR}/${step2_profile}/calib_ref_${step2_profile}.parquet"
        fi
        build_step2_tunelib_cmd "$step2_profile" "$STEP2_REF_FILE" "$PROFILE_PARALLEL_THREADS"
        printf '%q ' "${STEP2_TUNELIB_CMD[@]}"
        echo
      done
    else
      for step2_profile in "${STEP2_PROFILE_LIST[@]}"; do
        STEP2_REF_FILE=""
        if [[ "${NO_CAL_LIB:-0}" != "1" ]]; then
          STEP2_REF_FILE="${CALDIR}/${step2_profile}/calib_ref_${step2_profile}.parquet"
        fi
        build_step2_tunelib_cmd "$step2_profile" "$STEP2_REF_FILE"
        printf '%q ' "${STEP2_TUNELIB_CMD[@]}"
        echo
      done
    fi
  else
    echo "# step2_tunelib wrapper (skipped in resume mode)"
  fi
  if (( RUN_STEP3 == 1 )); then
    echo "# step3_tune_models wrapper"
    echo "# selected profile: ${PROFILE}"
    if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
      echo "# parallel mode: ${#STEP2_PROFILE_LIST[@]} concurrent runs, THREADS per profile: ${PROFILE_PARALLEL_THREADS}"
      for step3_profile in "${STEP2_PROFILE_LIST[@]}"; do
        build_step3_tune_models_cmd "$step3_profile" "$PROFILE_PARALLEL_THREADS"
        printf '%q ' "${STEP3_TUNE_MODELS_CMD[@]}"
        echo
      done
    else
      for step3_profile in "${STEP2_PROFILE_LIST[@]}"; do
        build_step3_tune_models_cmd "$step3_profile"
        printf '%q ' "${STEP3_TUNE_MODELS_CMD[@]}"
        echo
      done
    fi
  else
    echo "# step3_tune_models wrapper (skipped in resume mode)"
  fi
  if (( RUN_STEP4 == 1 )); then
    echo "# step4_create_tuned_predlib wrapper"
    echo "# selected profile: ${PROFILE}"
    echo "# model targets: ${STEP4_MODEL_TARGETS:-all}"
    echo "# lr selection: ${STEP4_LR_SELECT:-all}"
    if step4_inputs_ready_for_planning; then
      iterate_step4_runs step4_plan_print_callback
    else
      echo "# step4 commands are resolved after step3 outputs exist (lr_*/model artifacts)."
    fi
  else
    echo "# step4_create_tuned_predlib wrapper (skipped in resume mode)"
  fi
  if (( RUN_STEP5 == 1 )); then
    echo "# step5_quant wrapper"
    echo "# selected profile: ${PROFILE}"
    echo "# baseline + tuned matrix"
    echo "# tuned model targets: ${STEP4_MODEL_TARGETS:-all}"
    echo "# tuned lr selection: ${STEP4_LR_SELECT:-all}"
    if (( PARTIAL_RESUME_STEP5 == 1 )) && step5_baseline_complete; then
      echo "# baseline skipped (already completed in progress.txt)"
    else
      step5_plan_baseline_cmd ""  # baseline: full THREADS; tuned: THREADS/3 per profile in parallel mode
    fi
    if step4_inputs_ready_for_planning; then
      iterate_step4_runs step5_plan_tuned_callback
    else
      echo "# step5 tuned commands are resolved after step3 outputs exist (lr_*/model artifacts)."
    fi
  else
    echo "# step5_quant wrapper (skipped in resume mode)"
  fi
  echo
  echo "# progress entries that would be written on success"
  if (( RUN_STEP1 == 1 )); then
    printf '%s\n' "<timestamp>\tstep1_predlib.sh\tOK\tfasta=${COPIED_FASTA}"
  fi
  if (( RUN_MAKE_CAL == 1 && MULTI_PROFILE == 1 )); then
    for test_profile in "${STEP2_PROFILE_LIST[@]}"; do
      if [[ "$PROFILE_PARALLEL" == "1" ]]; then
        printf '%s\n' "<timestamp>\tmake_cal_lib.sh\tOK\tprofile=${test_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS};first_f=${FIRST_TRAIN_FILE_RAW}"
      else
        printf '%s\n' "<timestamp>\tmake_cal_lib.sh\tOK\tprofile=${test_profile};mode=sequential-all;first_f=${FIRST_TRAIN_FILE_RAW}"
      fi
    done
  elif (( RUN_MAKE_CAL == 1 )); then
    printf '%s\n' "<timestamp>\tmake_cal_lib.sh\tOK\tprofile=${PROFILE};mode=single;first_f=${FIRST_TRAIN_FILE_RAW}"
  fi
  if (( RUN_STEP2 == 1 && MULTI_PROFILE == 1 )); then
    for test_profile in "${STEP2_PROFILE_LIST[@]}"; do
      if [[ "$PROFILE_PARALLEL" == "1" ]]; then
        printf '%s\n' "<timestamp>\tstep2_tunelib.sh\tOK\tprofile=${test_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS};files=${#STEP2_RAW_FILES_RESOLVED[@]}"
      else
        printf '%s\n' "<timestamp>\tstep2_tunelib.sh\tOK\tprofile=${test_profile};mode=sequential-all;files=${#STEP2_RAW_FILES_RESOLVED[@]}"
      fi
    done
  elif (( RUN_STEP2 == 1 )); then
    printf '%s\n' "<timestamp>\tstep2_tunelib.sh\tOK\tprofile=${PROFILE};mode=single;files=${#STEP2_RAW_FILES_RESOLVED[@]}"
  fi
  if (( RUN_STEP3 == 1 && MULTI_PROFILE == 1 )); then
    for test_profile in "${STEP2_PROFILE_LIST[@]}"; do
      if [[ "$PROFILE_PARALLEL" == "1" ]]; then
        printf '%s\n' "<timestamp>\tstep3_tune_models.sh\tOK\tprofile=${test_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS}"
      else
        printf '%s\n' "<timestamp>\tstep3_tune_models.sh\tOK\tprofile=${test_profile};mode=sequential-all"
      fi
    done
  elif (( RUN_STEP3 == 1 )); then
    printf '%s\n' "<timestamp>\tstep3_tune_models.sh\tOK\tprofile=${PROFILE};mode=single"
  fi
  if (( RUN_STEP4 == 1 )); then
    if step4_inputs_ready_for_planning; then
      iterate_step4_runs step4_print_test_progress_callback
    else
      printf '%s\n' "<timestamp>\tstep4_create_tuned_predlib.sh\tOK\tprofile=<resolved-after-step3>;lr=<resolved-after-step3>;models=<resolved-after-step3>;mode=single-run;parallel=${STEP4_PARALLEL_MODE};targets=${STEP4_MODEL_TARGETS:-all};lr_select=${STEP4_LR_SELECT:-all};outdir=${STEP4_BASEDIR_NAME}"
    fi
    printf '%s\n' "<timestamp>\tstep4_create_tuned_predlib.sh\tOK\taggregate=1;profile=${PROFILE};targets=${STEP4_MODEL_TARGETS:-all};lr_select=${STEP4_LR_SELECT:-all};outdir=${STEP4_BASEDIR_NAME}"
  fi
  if (( RUN_STEP5 == 1 )); then
    if (( PARTIAL_RESUME_STEP5 == 1 )) && step5_baseline_complete; then
      echo "# baseline step5_quant.sh entry would not be written (already completed)"
    else
      printf '%s\n' "<timestamp>\tstep5_quant.sh\tOK\tprofile=${PROFILE};mode=baseline;parallel=0;files=${#STEP5_RAW_FILES_RESOLVED[@]}"
    fi
    if step4_inputs_ready_for_planning; then
      iterate_step4_runs step5_print_test_progress_callback
    else
      printf '%s\n' "<timestamp>\tstep5_quant.sh\tOK\tprofile=<resolved-after-step3>;lr=<resolved-after-step3>;models=<resolved-after-step3>;mode=tuned;parallel=${STEP5_PARALLEL_MODE};files=${#STEP5_RAW_FILES_RESOLVED[@]}"
    fi
    printf '%s\n' "<timestamp>\tstep5_quant.sh\tOK\taggregate=1;profile=${PROFILE};mode=baseline+tuned;files=${#STEP5_RAW_FILES_RESOLVED[@]}"
  fi
  echo ""
  exit 0
fi

if (( RUN_STEP1 == 1 )); then
  info "Running step1_predlib.sh"
  debug "Step1 wrapper command: ${STEP1_CMD[*]}"
  STEP1_START_TS="$(get_timestamp)"
  append_progress_start "step1_predlib.sh" "fasta=${COPIED_FASTA}"
  run_with_optional_ram_logger "step1" "$S1DIR" "step1_predlib" STEP1_CMD
  append_progress_ok "step1_predlib.sh" "fasta=${COPIED_FASTA}" "$STEP1_START_TS"
else
  info "Skipping step1_predlib.sh (resume mode)"
fi

STEP1_OUTLIB_RESOLVED="$(resolve_step1_outlib_for_make_cal "$STEP1_OUTLIB" "$S1DIR" "$EXPNAME" || true)"
[[ -n "$STEP1_OUTLIB_RESOLVED" ]] || die "Step1 output library not found (checked parquet first, then .speclib in ${S1DIR})"
STEP1_OUTLIB="$STEP1_OUTLIB_RESOLVED"
info "Using Step1 library for downstream steps: ${STEP1_OUTLIB}"
build_make_cal_cmd

if (( RUN_MAKE_CAL == 1 )); then
  info "Running make_cal_lib.sh (additional calibration helper, profile=${PROFILE})"
if (( MULTI_PROFILE == 1 )); then
  if [[ "$PROFILE_PARALLEL" == "1" ]]; then
    info "Running make_cal_lib.sh in parallel for profiles balanced/fast/slow (THREADS per profile=${PROFILE_PARALLEL_THREADS})"
    CAL_PIDS=()
    CAL_PROFILES=()
    CAL_START_TIMES=()
    for cal_profile in "${STEP2_PROFILE_LIST[@]}"; do
      MAKE_CAL_PAR_CMD=(
        bash "$MAKE_CAL_SCRIPT"
        --expname "$EXPNAME"
        --fastaabs "$COPIED_FASTA"
        --lib "$STEP1_OUTLIB"
        --profile "$cal_profile"
        --f "$FIRST_TRAIN_FILE_RAW"
      )
      if (( VERBOSE > 0 )); then
        MAKE_CAL_PAR_CMD+=(--verbose "$VERBOSE")
      fi
      for key in "${CLI_OVERRIDES[@]}"; do
        [[ "$key" == "THREADS" ]] && continue
        MAKE_CAL_PAR_CMD+=(--set "${key}=${!key}")
      done
      MAKE_CAL_PAR_CMD+=(--set "THREADS=${PROFILE_PARALLEL_THREADS}")

      CAL_START_TS="$(get_timestamp)"
      append_progress_start "make_cal_lib.sh" "profile=${cal_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS};first_f=${FIRST_TRAIN_FILE_RAW}"
      debug "Calibration helper wrapper command (${cal_profile}): ${MAKE_CAL_PAR_CMD[*]}"
      run_with_optional_ram_logger "make_cal" "${CALDIR}/${cal_profile}" "make_cal_${cal_profile}" MAKE_CAL_PAR_CMD &
      CAL_PIDS+=("$!")
      CAL_PROFILES+=("$cal_profile")
      CAL_START_TIMES+=("$CAL_START_TS")
    done

    cal_failed=0
    for idx in "${!CAL_PIDS[@]}"; do
      cal_pid="${CAL_PIDS[$idx]}"
      cal_profile="${CAL_PROFILES[$idx]}"
      cal_start_ts="${CAL_START_TIMES[$idx]}"
      if wait "$cal_pid"; then
        append_progress_ok "make_cal_lib.sh" "profile=${cal_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS};first_f=${FIRST_TRAIN_FILE_RAW}" "$cal_start_ts"
      else
        cal_failed=1
        warn "Calibration helper failed for profile=${cal_profile}"
      fi
    done

    (( cal_failed == 0 )) || die "At least one parallel make_cal_lib profile run failed"
  else
    for cal_profile in "${STEP2_PROFILE_LIST[@]}"; do
      MAKE_CAL_SEQ_CMD=(
        bash "$MAKE_CAL_SCRIPT"
        --expname "$EXPNAME"
        --fastaabs "$COPIED_FASTA"
        --lib "$STEP1_OUTLIB"
        --profile "$cal_profile"
        --f "$FIRST_TRAIN_FILE_RAW"
      )
      if (( VERBOSE > 0 )); then
        MAKE_CAL_SEQ_CMD+=(--verbose "$VERBOSE")
      fi
      for key in "${CLI_OVERRIDES[@]}"; do
        MAKE_CAL_SEQ_CMD+=(--set "${key}=${!key}")
      done

      CAL_SEQ_START_TS="$(get_timestamp)"
      append_progress_start "make_cal_lib.sh" "profile=${cal_profile};mode=sequential-all;first_f=${FIRST_TRAIN_FILE_RAW}"
      debug "Calibration helper wrapper command (${cal_profile}): ${MAKE_CAL_SEQ_CMD[*]}"
      run_with_optional_ram_logger "make_cal" "${CALDIR}/${cal_profile}" "make_cal_${cal_profile}" MAKE_CAL_SEQ_CMD
      append_progress_ok "make_cal_lib.sh" "profile=${cal_profile};mode=sequential-all;first_f=${FIRST_TRAIN_FILE_RAW}" "$CAL_SEQ_START_TS"
    done
  fi
else
  CAL_SEQ_START_TS="$(get_timestamp)"
  append_progress_start "make_cal_lib.sh" "profile=${PROFILE};mode=single;first_f=${FIRST_TRAIN_FILE_RAW}"
  debug "Calibration helper wrapper command: ${MAKE_CAL_CMD[*]}"
  run_with_optional_ram_logger "make_cal" "${CALDIR}/${PROFILE}" "make_cal_${PROFILE}" MAKE_CAL_CMD
  append_progress_ok "make_cal_lib.sh" "profile=${PROFILE};mode=single;first_f=${FIRST_TRAIN_FILE_RAW}" "$CAL_SEQ_START_TS"
fi
else
  if [[ "${NO_CAL_LIB:-0}" == "1" ]]; then
    info "Skipping make_cal_lib.sh (--no-cal-lib)"
  else
    info "Skipping make_cal_lib.sh (resume mode)"
  fi
fi

if (( RUN_STEP2 == 1 )); then
  info "Running step2_tunelib.sh (profile=${PROFILE})"
if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
  info "Running step2_tunelib.sh in parallel for profiles balanced/fast/slow (THREADS per profile=${PROFILE_PARALLEL_THREADS})"
  STEP2_PIDS=()
  STEP2_PROFILES=()
  STEP2_START_TIMES=()
  for step2_profile in "${STEP2_PROFILE_LIST[@]}"; do
    if (( PARTIAL_RESUME_STEP2 == 1 )) && step_profile_done "step2_tunelib.sh" "$step2_profile"; then
      info "Step2 resume: skipping completed profile=${step2_profile}"
      continue
    fi
    STEP2_REF_FILE=""
    if [[ "${NO_CAL_LIB:-0}" != "1" ]]; then
      STEP2_REF_FILE="${CALDIR}/${step2_profile}/calib_ref_${step2_profile}.parquet"
      [[ -f "$STEP2_REF_FILE" ]] || die "Missing Step2 ref for profile '${step2_profile}': ${STEP2_REF_FILE}"
    fi
    build_step2_tunelib_cmd "$step2_profile" "$STEP2_REF_FILE" "$PROFILE_PARALLEL_THREADS"
    STEP2_START_TS="$(get_timestamp)"
    append_progress_start "step2_tunelib.sh" "profile=${step2_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS};files=${#STEP2_RAW_FILES_RESOLVED[@]}"
    debug "Step2 wrapper command (${step2_profile}): ${STEP2_TUNELIB_CMD[*]}"
    run_with_optional_ram_logger "step2" "${S2DIR}/${step2_profile}" "step2_${step2_profile}" STEP2_TUNELIB_CMD &
    STEP2_PIDS+=("$!")
    STEP2_PROFILES+=("$step2_profile")
    STEP2_START_TIMES+=("$STEP2_START_TS")
  done

  step2_failed=0
  for idx in "${!STEP2_PIDS[@]}"; do
    step2_pid="${STEP2_PIDS[$idx]}"
    step2_profile="${STEP2_PROFILES[$idx]}"
    step2_start_ts="${STEP2_START_TIMES[$idx]}"
    if wait "$step2_pid"; then
      append_progress_ok "step2_tunelib.sh" "profile=${step2_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS};files=${#STEP2_RAW_FILES_RESOLVED[@]}" "$step2_start_ts"
    else
      step2_failed=1
      warn "Step2 failed for profile=${step2_profile}"
    fi
  done

  (( step2_failed == 0 )) || die "At least one parallel step2_tunelib profile run failed"
else
  for step2_profile in "${STEP2_PROFILE_LIST[@]}"; do
    if (( PARTIAL_RESUME_STEP2 == 1 )) && step_profile_done "step2_tunelib.sh" "$step2_profile"; then
      info "Step2 resume: skipping completed profile=${step2_profile}"
      continue
    fi
    STEP2_REF_FILE=""
    if [[ "${NO_CAL_LIB:-0}" != "1" ]]; then
      STEP2_REF_FILE="${CALDIR}/${step2_profile}/calib_ref_${step2_profile}.parquet"
      [[ -f "$STEP2_REF_FILE" ]] || die "Missing Step2 ref for profile '${step2_profile}': ${STEP2_REF_FILE}"
    fi
    build_step2_tunelib_cmd "$step2_profile" "$STEP2_REF_FILE"
    STEP2_SEQ_START_TS="$(get_timestamp)"
    if (( MULTI_PROFILE == 1 )); then
      append_progress_start "step2_tunelib.sh" "profile=${step2_profile};mode=sequential-all;files=${#STEP2_RAW_FILES_RESOLVED[@]}"
    else
      append_progress_start "step2_tunelib.sh" "profile=${step2_profile};mode=single;files=${#STEP2_RAW_FILES_RESOLVED[@]}"
    fi
    debug "Step2 wrapper command (${step2_profile}): ${STEP2_TUNELIB_CMD[*]}"
    run_with_optional_ram_logger "step2" "${S2DIR}/${step2_profile}" "step2_${step2_profile}" STEP2_TUNELIB_CMD

    if (( MULTI_PROFILE == 1 )); then
      append_progress_ok "step2_tunelib.sh" "profile=${step2_profile};mode=sequential-all;files=${#STEP2_RAW_FILES_RESOLVED[@]}" "$STEP2_SEQ_START_TS"
    else
      append_progress_ok "step2_tunelib.sh" "profile=${step2_profile};mode=single;files=${#STEP2_RAW_FILES_RESOLVED[@]}" "$STEP2_SEQ_START_TS"
    fi
  done
fi
else
  info "Skipping step2_tunelib.sh (resume mode)"
fi

if (( RUN_STEP3 == 1 )); then
  info "Running step3_tune_models.sh (profile=${PROFILE})"
if [[ "$PROFILE_PARALLEL" == "1" && "$MULTI_PROFILE" == "1" ]]; then
  info "Running step3_tune_models.sh in parallel for profiles balanced/fast/slow (THREADS per profile=${PROFILE_PARALLEL_THREADS})"
  STEP3_PIDS=()
  STEP3_PROFILES=()
  STEP3_START_TIMES=()
  for step3_profile in "${STEP2_PROFILE_LIST[@]}"; do
    if (( PARTIAL_RESUME_STEP3 == 1 )) && step_profile_done "step3_tune_models.sh" "$step3_profile"; then
      info "Step3 resume: skipping completed profile=${step3_profile}"
      continue
    fi
    build_step3_tune_models_cmd "$step3_profile" "$PROFILE_PARALLEL_THREADS"
    STEP3_START_TS="$(get_timestamp)"
    append_progress_start "step3_tune_models.sh" "profile=${step3_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS}"
    debug "Step3 wrapper command (${step3_profile}): ${STEP3_TUNE_MODELS_CMD[*]}"
    run_with_optional_ram_logger "step3" "${S3DIR}/${step3_profile}" "step3_${step3_profile}" STEP3_TUNE_MODELS_CMD &
    STEP3_PIDS+=("$!")
    STEP3_PROFILES+=("$step3_profile")
    STEP3_START_TIMES+=("$STEP3_START_TS")
  done

  step3_failed=0
  for idx in "${!STEP3_PIDS[@]}"; do
    step3_pid="${STEP3_PIDS[$idx]}"
    step3_profile="${STEP3_PROFILES[$idx]}"
    step3_start_ts="${STEP3_START_TIMES[$idx]}"
    if wait "$step3_pid"; then
      append_progress_ok "step3_tune_models.sh" "profile=${step3_profile};mode=parallel;threads=${PROFILE_PARALLEL_THREADS}" "$step3_start_ts"
    else
      step3_failed=1
      warn "Step3 failed for profile=${step3_profile}"
    fi
  done

  (( step3_failed == 0 )) || die "At least one parallel step3_tune_models profile run failed"
else
  for step3_profile in "${STEP2_PROFILE_LIST[@]}"; do
    if (( PARTIAL_RESUME_STEP3 == 1 )) && step_profile_done "step3_tune_models.sh" "$step3_profile"; then
      info "Step3 resume: skipping completed profile=${step3_profile}"
      continue
    fi
    build_step3_tune_models_cmd "$step3_profile"
    STEP3_SEQ_START_TS="$(get_timestamp)"
    if (( MULTI_PROFILE == 1 )); then
      append_progress_start "step3_tune_models.sh" "profile=${step3_profile};mode=sequential-all"
    else
      append_progress_start "step3_tune_models.sh" "profile=${step3_profile};mode=single"
    fi
    debug "Step3 wrapper command (${step3_profile}): ${STEP3_TUNE_MODELS_CMD[*]}"
    run_with_optional_ram_logger "step3" "${S3DIR}/${step3_profile}" "step3_${step3_profile}" STEP3_TUNE_MODELS_CMD

    if (( MULTI_PROFILE == 1 )); then
      append_progress_ok "step3_tune_models.sh" "profile=${step3_profile};mode=sequential-all" "$STEP3_SEQ_START_TS"
    else
      append_progress_ok "step3_tune_models.sh" "profile=${step3_profile};mode=single" "$STEP3_SEQ_START_TS"
    fi
  done
fi
else
  info "Skipping step3_tune_models.sh (resume mode)"
fi

if (( RUN_STEP4 == 1 )); then
  info "Running step4_create_tuned_predlib.sh (profile=${PROFILE})"
  resolve_step4_combo_count
  if (( STEP4_PARALLEL_MODE == 1 )); then
    info "Running Step4 with max 3 parallel profiles; LR/model combos stay sequential per profile (THREADS per profile=${STEP4_THREADS_OVERRIDE})"
    STEP4_PROFILE_PIDS=()
    STEP4_PROFILE_NAMES=()

    step4_run_profile_worker() {
      local worker_profile="$1"
      iterate_step4_runs_for_profile "$worker_profile" step4_execute_and_log_callback
    }

    for step4_profile in "${STEP2_PROFILE_LIST[@]}"; do
      step4_run_profile_worker "$step4_profile" &
      STEP4_PROFILE_PIDS+=("$!")
      STEP4_PROFILE_NAMES+=("$step4_profile")
    done

    step4_profile_failed=0
    for idx in "${!STEP4_PROFILE_PIDS[@]}"; do
      if ! wait "${STEP4_PROFILE_PIDS[$idx]}"; then
        step4_profile_failed=1
        warn "Step4 failed for profile=${STEP4_PROFILE_NAMES[$idx]}"
      fi
    done
    unset -f step4_run_profile_worker
    (( step4_profile_failed == 0 )) || die "At least one parallel Step4 profile run failed"
  else
    info "Running Step4 sequentially across selected profile(s) and combinations"
    iterate_step4_runs step4_execute_and_log_callback
  fi

  STEP4_AGG_START_TS="$(get_timestamp)"
  append_progress_start "step4_create_tuned_predlib.sh" "aggregate=1;profile=${PROFILE};runs=${STEP4_COMBO_COUNT};parallel=${STEP4_PARALLEL_MODE};threads_per_profile=${STEP4_THREADS_OVERRIDE:-${THREADS}};targets=${STEP4_MODEL_TARGETS:-all};lr_select=${STEP4_LR_SELECT:-all};distill_select=${STEP4_DISTILL_LEVELS:-0};outdir=${STEP4_BASEDIR_NAME}"
  append_progress_ok "step4_create_tuned_predlib.sh" "aggregate=1;profile=${PROFILE};runs=${STEP4_COMBO_COUNT};parallel=${STEP4_PARALLEL_MODE};threads_per_profile=${STEP4_THREADS_OVERRIDE:-${THREADS}};targets=${STEP4_MODEL_TARGETS:-all};lr_select=${STEP4_LR_SELECT:-all};distill_select=${STEP4_DISTILL_LEVELS:-0};outdir=${STEP4_BASEDIR_NAME}" "$STEP4_AGG_START_TS"
else
  info "Skipping step4_create_tuned_predlib.sh (resume mode)"
fi

if (( RUN_STEP5 == 1 )); then
  info "Running step5_quant.sh (baseline + tuned; profile=${PROFILE})"

  resolve_step4_combo_count
  if (( STEP5_PARALLEL_MODE == 1 )); then
    info "Running Step5 tuned with max 3 parallel profiles; LR/model combos stay sequential per profile (THREADS per profile=${STEP5_THREADS_OVERRIDE})"
  else
    info "Running Step5 tuned sequentially across selected profile(s) and combinations"
  fi

  STEP5_BASELINE_START_TS="$(get_timestamp)"
  if (( PARTIAL_RESUME_STEP5 == 1 )) && step5_baseline_complete; then
    info "Step5 resume: skipping completed baseline run for profile=${PROFILE}"
  else
    append_progress_start "step5_quant.sh" "profile=${PROFILE};mode=baseline;files=${#STEP5_RAW_FILES_RESOLVED[@]}"
    build_step5_quant_cmd "$STEP1_OUTLIB" "baseline" ""  # baseline: full THREADS regardless of parallel mode
    debug "Step5 baseline wrapper command: ${STEP5_QUANT_CMD[*]}"
    run_with_optional_ram_logger "step5" "${S5DIR}/baseline" "step5_baseline" STEP5_QUANT_CMD
    append_progress_ok "step5_quant.sh" "profile=${PROFILE};mode=baseline;files=${#STEP5_RAW_FILES_RESOLVED[@]}" "$STEP5_BASELINE_START_TS"
  fi

  if (( STEP5_PARALLEL_MODE == 1 )); then
    step5_run_profile_worker() {
      local worker_profile="$1"
      step5_execute_tuned_and_log() {
        local profile_name="$1"
        local lr_tag="$2"
        local distill_level="$3"
        local target_combo="$4"
        local start_ts
        local fragment

        if (( PARTIAL_RESUME_STEP5 == 1 )); then
          fragment="profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;"
          if progress_has_ok_step_detail_fragment "step5_quant.sh" "$fragment"; then
            info "Step5 resume: skipping completed tuned combo profile=${profile_name}, lr=${lr_tag}, distill=d${distill_level}, models=${target_combo}"
            return 0
          fi
        fi

        start_ts="$(get_timestamp)"
        append_progress_start "step5_quant.sh" "profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;parallel=1;files=${#STEP5_RAW_FILES_RESOLVED[@]}"
        step5_execute_tuned_callback "$profile_name" "$lr_tag" "$distill_level" "$target_combo" "$5" "$6" "$7" "$8"
        append_progress_ok "step5_quant.sh" "profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;parallel=1;files=${#STEP5_RAW_FILES_RESOLVED[@]}" "$start_ts"
      }
      iterate_step4_runs_for_profile "$worker_profile" step5_execute_tuned_and_log
      unset -f step5_execute_tuned_and_log
    }

    STEP5_PROFILE_PIDS=()
    STEP5_PROFILE_NAMES=()
    for step5_profile in "${STEP2_PROFILE_LIST[@]}"; do
      step5_run_profile_worker "$step5_profile" &
      STEP5_PROFILE_PIDS+=("$!")
      STEP5_PROFILE_NAMES+=("$step5_profile")
    done

    step5_profile_failed=0
    for idx in "${!STEP5_PROFILE_PIDS[@]}"; do
      if ! wait "${STEP5_PROFILE_PIDS[$idx]}"; then
        step5_profile_failed=1
        warn "Step5 tuned quant failed for profile=${STEP5_PROFILE_NAMES[$idx]}"
      fi
    done
    unset -f step5_run_profile_worker
    (( step5_profile_failed == 0 )) || die "At least one parallel Step5 profile run failed"
  else
    step5_execute_tuned_and_log() {
      local profile_name="$1"
      local lr_tag="$2"
      local distill_level="$3"
      local target_combo="$4"
      local start_ts
      local fragment

      if (( PARTIAL_RESUME_STEP5 == 1 )); then
        fragment="profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;"
        if progress_has_ok_step_detail_fragment "step5_quant.sh" "$fragment"; then
          info "Step5 resume: skipping completed tuned combo profile=${profile_name}, lr=${lr_tag}, distill=d${distill_level}, models=${target_combo}"
          return 0
        fi
      fi

      start_ts="$(get_timestamp)"
      append_progress_start "step5_quant.sh" "profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;parallel=0;files=${#STEP5_RAW_FILES_RESOLVED[@]}"
      step5_execute_tuned_callback "$profile_name" "$lr_tag" "$distill_level" "$target_combo" "$5" "$6" "$7" "$8"
      append_progress_ok "step5_quant.sh" "profile=${profile_name};lr=${lr_tag};distill=${distill_level};models=${target_combo};mode=tuned;parallel=0;files=${#STEP5_RAW_FILES_RESOLVED[@]}" "$start_ts"
    }
    iterate_step4_runs step5_execute_tuned_and_log
    unset -f step5_execute_tuned_and_log
  fi

  STEP5_AGG_START_TS="$(get_timestamp)"
  append_progress_start "step5_quant.sh" "aggregate=1;profile=${PROFILE};mode=baseline+tuned;runs=${STEP4_COMBO_COUNT};parallel=${STEP5_PARALLEL_MODE};threads_per_profile=${STEP5_THREADS_OVERRIDE:-${THREADS}};distill_select=${STEP4_DISTILL_LEVELS:-0};files=${#STEP5_RAW_FILES_RESOLVED[@]}"
  append_progress_ok "step5_quant.sh" "aggregate=1;profile=${PROFILE};mode=baseline+tuned;runs=${STEP4_COMBO_COUNT};parallel=${STEP5_PARALLEL_MODE};threads_per_profile=${STEP5_THREADS_OVERRIDE:-${THREADS}};distill_select=${STEP4_DISTILL_LEVELS:-0};files=${#STEP5_RAW_FILES_RESOLVED[@]}" "$STEP5_AGG_START_TS"

  MAPLOT_START_TS="$(get_timestamp)"
  if [[ "$ENABLE_MAPLOT_REPORT" == "1" ]]; then
    append_progress_start "maplot_report" "experiment=${EXPNAME};pipeline_version=${DIANN_VERSION_DIR};organism=${MAPLOT_ORGANISM}"
    run_maplot_report
    append_progress_ok "maplot_report" "experiment=${EXPNAME};pipeline_version=${DIANN_VERSION_DIR};organism=${MAPLOT_ORGANISM}" "$MAPLOT_START_TS"
  else
    info "Skipping automatic MA plot report (ENABLE_MAPLOT_REPORT=0)"
  fi
else
  info "Skipping step5_quant.sh (resume mode)"
fi

echo "Step 1 finished via run_all.sh."
echo "Calibration helper (make_cal_lib) finished via run_all.sh."
echo "Step 2 (tunelib) finished via run_all.sh."
echo "Step 3 (tune models) finished via run_all.sh."
echo "Step 4 (create tuned predlib) finished via run_all.sh."
echo "Step 5 (quant search) finished via run_all.sh."
