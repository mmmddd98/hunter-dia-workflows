#!/usr/bin/env python3
"""Reusable manifest-driven GPU sweep runner with resumable jobs and live bars."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("STUDY_GPUS", "0,1,2"))
    parser.add_argument(
        "--jobs-per-gpu", type=int,
        default=int(os.environ.get("STUDY_JOBS_PER_GPU", "2")),
    )
    parser.add_argument(
        "--jobs-per-gpu-map",
        default=os.environ.get("STUDY_JOBS_PER_GPU_MAP", ""),
        help="Optional explicit capacities, e.g. 0:2,1:2,2:1; overrides the memory guard",
    )
    parser.add_argument(
        "--gpu-memory-per-job-mib", type=int,
        default=int(os.environ.get("STUDY_GPU_MEMORY_PER_JOB_MIB", "3200")),
        help="Estimated peak GPU memory per job (default: 3200 MiB)",
    )
    parser.add_argument(
        "--gpu-memory-safety-margin-mib", type=int,
        default=int(os.environ.get("STUDY_GPU_MEMORY_SAFETY_MARGIN_MIB", "256")),
        help="Free GPU memory left unassigned by the default memory guard (default: 256 MiB)",
    )
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument(
        "--shared-slot-root", type=Path,
        default=Path(os.environ["STUDY_SHARED_SLOT_ROOT"]) if os.environ.get("STUDY_SHARED_SLOT_ROOT") else None,
        help="Optional cross-process GPU-slot lock directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A per-process temporary name keeps reporting safe even if an external
    # monitor reads the status file while it is being refreshed.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_free_memory_mib() -> dict[str, int]:
    """Return currently free memory for every visible physical GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        )
        return {
            index.strip(): int(free.strip())
            for index, free in (
                line.split(",", 1) for line in result.stdout.splitlines() if line.strip()
            )
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def resolve_slot_counts(
    gpus: list[str],
    requested: int,
    explicit_map: str,
    memory_per_job_mib: int,
    safety_margin_mib: int,
    free_memory_mib: dict[str, int] | None = None,
) -> dict[str, int]:
    """Cap requested workers using free VRAM observed before jobs are started."""
    if requested < 1:
        raise ValueError("jobs per GPU must be at least one")
    if memory_per_job_mib < 1:
        raise ValueError("GPU memory per job must be at least 1 MiB")
    if safety_margin_mib < 0:
        raise ValueError("GPU memory safety margin must be non-negative")
    if explicit_map.strip():
        try:
            capacities = {
                gpu.strip(): int(count.strip())
                for gpu, count in (
                    item.split(":", 1)
                    for item in explicit_map.split(",")
                    if item.strip()
                )
            }
        except (ValueError, TypeError) as error:
            raise ValueError(
                "jobs-per-gpu-map must look like 0:2,1:2,2:1"
            ) from error
        if set(capacities) != set(gpus) or any(count < 0 for count in capacities.values()):
            raise ValueError(
                f"jobs-per-gpu-map must define every selected GPU exactly once: {gpus}"
            )
        if not any(capacities.values()):
            raise ValueError("jobs-per-gpu-map disables every GPU")
        return capacities

    capacities = {gpu: requested for gpu in gpus}
    free = free_memory_mib if free_memory_mib is not None else gpu_free_memory_mib()
    if not free:
        print(
            "WARNING: could not query free GPU memory; using the requested slot count",
            file=sys.stderr, flush=True,
        )
        return capacities
    for gpu in gpus:
        if gpu not in free:
            continue
        safe_workers = max(0, (free[gpu] - safety_margin_mib) // memory_per_job_mib)
        capacities[gpu] = min(requested, safe_workers)
    if not any(capacities.values()):
        raise ValueError(
            "No selected GPU has enough free memory for one worker: "
            f"free={free}, required/job={memory_per_job_mib} MiB, "
            f"safety margin={safety_margin_mib} MiB"
        )
    return capacities


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("jobs"), list) or not payload["jobs"]:
        raise SystemExit("Manifest must contain a non-empty 'jobs' list")
    seen = set()
    for job in payload["jobs"]:
        job_id = job.get("id")
        commands = job.get("commands")
        if not isinstance(job_id, str) or not job_id or job_id in seen:
            raise SystemExit(f"Every job needs a unique non-empty id; invalid id: {job_id!r}")
        if not isinstance(commands, list) or not commands or not all(
            isinstance(command, list) and command and all(isinstance(item, str) for item in command)
            for command in commands
        ):
            raise SystemExit(f"Job {job_id!r} needs a non-empty list of argv-style commands")
        if not isinstance(job.get("completion_files"), list) or not job["completion_files"]:
            raise SystemExit(f"Job {job_id!r} needs at least one final completion file")
        # Older preparation scripts accidentally serialised ``\\\\d`` rather
        # than ``\\d``.  Accept that legacy spelling so a live-bar typo can
        # never make a valid running job look stalled again.
        if isinstance(job.get("progress_regex"), str):
            job["progress_regex"] = job["progress_regex"].replace("\\\\d", "\\d")
        seen.add(job_id)
    return payload


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "job"


def job_fingerprint(job: dict) -> str:
    relevant = {
        "commands": job["commands"],
        "workdir": job.get("workdir"),
        "environment": job.get("environment", {}),
        "completion_files": job.get("completion_files", []),
    }
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True).encode("utf-8")
    ).hexdigest()


def completion_files_exist(job: dict) -> bool:
    return all(Path(item).is_file() for item in job.get("completion_files", []))


def job_complete(job: dict, marker: Path) -> bool:
    if not marker.is_file() or not completion_files_exist(job):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return payload.get("fingerprint") == job_fingerprint(job)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def read_progress(log_path: Path, regex: str | None, total_hint: int | None) -> tuple[int, int] | None:
    if not regex or not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 250_000))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    if "] START " in text:
        text = text.rsplit("] START ", 1)[-1]
    matches = list(re.finditer(regex, text, re.MULTILINE))
    if not matches:
        return None
    match = matches[-1]
    current = int(match.group(1))
    total = int(match.group(2)) if match.lastindex and match.lastindex >= 2 else int(total_hint or 0)
    return (current, total) if total > 0 else None


def bar(done: int, total: int, width: int = 24) -> str:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, done / total))
    filled = round(width * fraction)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def format_duration(seconds: float | None) -> str:
    """Format a finite duration for a compact terminal status line."""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "–"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def completed_run_durations(jobs: list[dict]) -> list[float]:
    """Read wall-clock durations written by completed jobs in this sweep."""
    durations = []
    for job in jobs:
        marker = job["marker"]
        if not marker.is_file():
            continue
        try:
            value = json.loads(marker.read_text(encoding="utf-8")).get("duration_seconds")
            if isinstance(value, (int, float)) and value > 0:
                durations.append(float(value))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return durations


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    name = str(manifest.get("name") or args.manifest.stem)
    state_root = Path(manifest.get("state_root") or args.manifest.parent / f".{args.manifest.stem}_state")
    status_path = Path(manifest.get("status_file") or state_root / "status.json")
    log_root = Path(manifest.get("log_root") or state_root / "logs")
    marker_root = state_root / "markers"
    lock_root = state_root / "locks"
    # One manifest has one controller.  Job-level locks already prevent
    # duplicated work, but a second controller still overwrites status.json
    # and makes the live UI look stalled.  Hold this lock for the whole run.
    state_root.mkdir(parents=True, exist_ok=True)
    controller_lock_path = state_root / "controller.lock"
    controller_lock = controller_lock_path.open("a+")
    try:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        controller_lock.close()
        raise SystemExit(
            f"Another controller is already running for this manifest: {controller_lock_path}"
        ) from error
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise SystemExit("--gpus must contain distinct comma-separated GPU IDs")
    free_memory_at_start = (
        {} if args.jobs_per_gpu_map.strip() else gpu_free_memory_mib()
    )
    try:
        slot_counts = resolve_slot_counts(
            gpus,
            args.jobs_per_gpu,
            args.jobs_per_gpu_map,
            args.gpu_memory_per_job_mib,
            args.gpu_memory_safety_margin_mib,
            free_memory_mib=free_memory_at_start,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    slot_summary = ", ".join(f"GPU {gpu}×{slot_counts[gpu]}" for gpu in gpus)
    if free_memory_at_start:
        free_summary = ", ".join(
            f"GPU {gpu}={free_memory_at_start.get(gpu, '?')} MiB" for gpu in gpus
        )
        print(f"Free VRAM at launch: {free_summary}", flush=True)

    jobs = []
    for raw_job in manifest["jobs"]:
        job = raw_job.copy()
        job["marker"] = marker_root / f"{safe_name(job['id'])}.complete.json"
        job["lock"] = lock_root / f"{safe_name(job['id'])}.lock"
        job["log"] = Path(job.get("log") or log_root / f"{safe_name(job['id'])}.log")
        jobs.append(job)
    complete_at_start = sum(job_complete(job, job["marker"]) for job in jobs)
    pending = [job for job in jobs if not job_complete(job, job["marker"])]
    print(
        f"{name}: {complete_at_start}/{len(jobs)} complete, {len(pending)} pending; "
        f"workers={slot_summary}",
        flush=True,
    )
    if args.dry_run:
        for job in pending:
            print(f"  - {job['id']}")
        return 0

    work: queue.Queue[dict] = queue.Queue()
    for job in pending:
        work.put(job)
    states: dict[int, dict] = {}
    states_lock = threading.Lock()
    processes: dict[int, subprocess.Popen] = {}
    processes_lock = threading.Lock()
    failures: list[str] = []
    stop_event = threading.Event()

    def set_state(slot: int, **values: object) -> None:
        with states_lock:
            states[slot] = {**states.get(slot, {}), **values}

    def terminate_children() -> None:
        with processes_lock:
            active = list(processes.values())
        for process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    def worker(slot: int, gpu: str, local_slot: int) -> None:
        while not stop_event.is_set():
            try:
                job = work.get_nowait()
            except queue.Empty:
                set_state(slot, phase="idle", job="-")
                return
            shared_handle = None
            if args.shared_slot_root is not None:
                args.shared_slot_root.mkdir(parents=True, exist_ok=True)
                shared_path = args.shared_slot_root / f"gpu_{safe_name(gpu)}_slot_{local_slot + 1}.lock"
                shared_handle = shared_path.open("a+")
                set_state(slot, phase="waiting for shared GPU slot", job=job["id"], log=job["log"])
                while not stop_event.is_set():
                    try:
                        fcntl.flock(shared_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        stop_event.wait(0.5)
                if stop_event.is_set():
                    shared_handle.close()
                    work.task_done()
                    return
            job["lock"].parent.mkdir(parents=True, exist_ok=True)
            lock_handle = job["lock"].open("a+")
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                set_state(slot, phase="active elsewhere", job=job["id"], log=job["log"])
                lock_handle.close()
                if shared_handle is not None:
                    fcntl.flock(shared_handle, fcntl.LOCK_UN)
                    shared_handle.close()
                work.task_done()
                continue
            try:
                job["log"].parent.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment.update({str(k): str(v) for k, v in job.get("environment", {}).items()})
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                started_monotonic = time.monotonic()
                set_state(
                    slot,
                    phase="running",
                    job=job["id"],
                    log=job["log"],
                    progress_regex=job.get("progress_regex"),
                    progress_total=job.get("progress_total"),
                    started_monotonic=started_monotonic,
                )
                with job["log"].open("a", encoding="utf-8") as log:
                    log.write(
                        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START "
                        f"job={job['id']} gpu={gpu} slot={slot}\n"
                    )
                    log.flush()
                    for command in job["commands"]:
                        if stop_event.is_set():
                            raise RuntimeError("sweep interrupted")
                        log.write("$ " + " ".join(command) + "\n")
                        log.flush()
                        process = subprocess.Popen(
                            command,
                            cwd=job.get("workdir"),
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        with processes_lock:
                            processes[slot] = process
                        return_code = process.wait()
                        with processes_lock:
                            processes.pop(slot, None)
                        if return_code:
                            raise RuntimeError(f"command exited with {return_code}")
                    if not completion_files_exist(job):
                        raise RuntimeError("required completion files are missing")
                    atomic_json(job["marker"], {
                        "job": job["id"],
                        "fingerprint": job_fingerprint(job),
                        "gpu": gpu,
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_seconds": time.monotonic() - started_monotonic,
                    })
                    log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE job={job['id']}\n")
                set_state(slot, phase="complete", job=job["id"], log=job["log"])
            except Exception as error:
                failures.append(f"{job['id']}: {error} (log: {job['log']})")
                stop_event.set()
                set_state(slot, phase="FAILED", job=job["id"], log=job["log"], error=str(error))
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                lock_handle.close()
                if shared_handle is not None:
                    fcntl.flock(shared_handle, fcntl.LOCK_UN)
                    shared_handle.close()
                work.task_done()

    slots = [
        (gpu, local_slot)
        for gpu in gpus
        for local_slot in range(slot_counts[gpu])
    ]
    threads = []
    for slot_id, (gpu, local_slot) in enumerate(slots):
        states[slot_id] = {"gpu": gpu, "local_slot": local_slot + 1, "phase": "queued", "job": "-"}
        thread = threading.Thread(target=worker, args=(slot_id, gpu, local_slot), daemon=False)
        threads.append(thread)
        thread.start()

    interrupted = False
    is_tty = sys.stdout.isatty()
    last_print = 0.0
    try:
        while any(thread.is_alive() for thread in threads):
            try:
                complete = sum(job_complete(job, job["marker"]) for job in jobs)
                with states_lock:
                    snapshot = {key: value.copy() for key, value in states.items()}
                serializable = {}
                for slot_id, item in snapshot.items():
                    clean = {key: str(value) if isinstance(value, Path) else value for key, value in item.items()}
                    progress = read_progress(
                        Path(item["log"]),
                        str(item["progress_regex"]) if item.get("log") and item.get("progress_regex") else None,
                        int(item.get("progress_total") or 0) or None,
                    ) if item.get("log") else None
                    if progress:
                        clean["progress"], clean["progress_total"] = progress
                    serializable[slot_id] = clean
                completed_durations = completed_run_durations(jobs)
                active_duration_estimates = []
                active_remaining_seconds = []
                now_monotonic = time.monotonic()
                for item in serializable.values():
                    progress = item.get("progress")
                    progress_total = item.get("progress_total")
                    started = item.get("started_monotonic")
                    if not (isinstance(progress, int) and isinstance(progress_total, int)
                            and progress > 0 and progress_total > 0
                            and isinstance(started, (int, float))):
                        continue
                    elapsed = max(0.0, now_monotonic - started)
                    estimate = elapsed * progress_total / progress
                    active_duration_estimates.append(estimate)
                    active_remaining_seconds.append(max(0.0, estimate - elapsed))
                if completed_durations:
                    typical_run_seconds = statistics.median(completed_durations)
                    eta_basis = f"{len(completed_durations)} completed runs"
                elif active_duration_estimates:
                    typical_run_seconds = statistics.median(active_duration_estimates)
                    eta_basis = "active epoch rate (provisional)"
                else:
                    typical_run_seconds = None
                    eta_basis = None
                eta_seconds = None
                if typical_run_seconds is not None:
                    # Active jobs already occupy all available slots; their
                    # remaining time is averaged across those slots, and the
                    # queued jobs are distributed over the same capacity.
                    active_component = sum(active_remaining_seconds) / max(1, len(slots))
                    queued_component = work.qsize() * typical_run_seconds / max(1, len(slots))
                    eta_seconds = active_component + queued_component
                atomic_json(status_path, {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name, "complete": complete, "total": len(jobs),
                    "queued": work.qsize(), "gpus": gpus,
                    "jobs_per_gpu": slot_counts, "slots": serializable,
                    "free_gpu_memory_mib_at_start": free_memory_at_start,
                    "typical_run_seconds": typical_run_seconds,
                    "eta_seconds": eta_seconds,
                    "eta_basis": eta_basis,
                    "failures": failures,
                })
                now = time.monotonic()
                if is_tty or now - last_print >= 30:
                    lines = [
                        f"{name} · resumable GPU sweep",
                        f"TOTAL {bar(complete, len(jobs))} {complete}/{len(jobs)} complete · {work.qsize()} queued",
                    ]
                    for slot_id in sorted(serializable):
                        item = serializable[slot_id]
                        # A job has a total epoch count before its first log
                        # line is available.  Rendering that state must not
                        # turn a cosmetic KeyError into repeated warnings.
                        if item.get("progress_total") and isinstance(item.get("progress"), int):
                            progress_text = (
                                f"{bar(int(item['progress']), int(item['progress_total']), 12)} "
                                f"{item['progress']}/{item['progress_total']}"
                            )
                        else:
                            progress_text = "[············] --/--"
                        lines.append(
                            f"GPU {item['gpu']} · slot {item['local_slot']}: {progress_text} · "
                            f"{item['phase']} · {item['job']}"
                        )
                    if typical_run_seconds is not None:
                        lines.append(
                            f"RUN median {format_duration(typical_run_seconds)} · "
                            f"ETA {format_duration(eta_seconds)} · {eta_basis}"
                        )
                    print(("\033[2J\033[H" if is_tty else "") + "\n".join(lines), flush=True)
                    last_print = now
            except Exception as error:
                # A reporting failure must never stop or hide active training.
                print(f"WARNING: progress renderer failed; retrying: {error}", file=sys.stderr, flush=True)
            time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        terminate_children()
    finally:
        for thread in threads:
            thread.join()

    complete = sum(job_complete(job, job["marker"]) for job in jobs)
    print(f"{name}: {complete}/{len(jobs)} complete; {len(failures)} failed", flush=True)
    if interrupted:
        print("Interrupted; completed jobs remain resumable.", file=sys.stderr)
        return 130
    if failures:
        print("\n".join(f"FAILED: {item}" for item in failures), file=sys.stderr)
        return 1
    return 0 if complete == len(jobs) else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    raise SystemExit(main())
