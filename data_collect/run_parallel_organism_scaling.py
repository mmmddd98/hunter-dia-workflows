#!/usr/bin/env python3
"""Run organism-scaling jobs with memory-aware independent GPU workers."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import TextIO


ORGANISMS = ("celegans", "triplemix", "human")
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DIANN_ROOT = Path(os.environ.get("DIANN_FINETUNE_ROOT", WORKSPACE_ROOT.parent / "diann-fintune")).expanduser()
CARAFE2_ROOT = Path(os.environ.get("CARAFE2_ROOT", WORKSPACE_ROOT / "carafe2-semi")).expanduser()
DIANN_MODELS = Path(
    os.environ.get("DIANN_MODEL_DIR", WORKSPACE_ROOT.parent / "diann_sc" / "DIA-NN-2.5.0" / "diann-2.5.0" / "models")
).expanduser()
DEFAULT_CARAFE2_PYTHON = Path(
    os.environ.get("CARAFE2_PYTHON", "python")
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("diann", "carafe2"))
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--gpus", default=os.environ.get("STUDY_GPUS", "0,1,2"))
    parser.add_argument("--jobs-per-gpu", type=int, default=int(os.environ.get("STUDY_JOBS_PER_GPU", "2")))
    parser.add_argument(
        "--jobs-per-gpu-map",
        default=os.environ.get("STUDY_JOBS_PER_GPU_MAP", ""),
        help="Optional explicit capacities, e.g. 0:2,1:2,2:1",
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--warmup-epochs", type=int, required=True)
    parser.add_argument("--seeds", default=os.environ.get("STUDY_SEEDS", ""))
    parser.add_argument("--conditions", default=os.environ.get("STUDY_CONDITIONS", ""))
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", "python3"))
    parser.add_argument(
        "--carafe2-python",
        default=str(DEFAULT_CARAFE2_PYTHON),
        help="Python environment containing torch, alphabase, and Carafe2 dependencies",
    )
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected(value: str) -> set[str] | None:
    values = {item.strip() for item in value.split(",") if item.strip()}
    return values or None


def validate_carafe2_python(python: str) -> Path:
    path = Path(python).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Carafe2 Python is not executable: {path}")
    result = subprocess.run(
        [
            str(path), "-c",
            "import sys, torch, alphabase; print(sys.executable)",
        ],
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"exit code {result.returncode}"
        raise ValueError(
            f"Carafe2 Python lacks required dependencies (torch, alphabase): {path}: {reason}"
        )
    return path


def gpu_free_memory_mib() -> dict[str, int]:
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
            for index, free in (line.split(",", 1) for line in result.stdout.splitlines())
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def resolve_slot_counts(
    gpus: list[str],
    requested: int,
    explicit_map: str,
    family: str,
    free_memory_mib: dict[str, int] | None = None,
) -> dict[str, int]:
    if requested < 1:
        raise ValueError("jobs per GPU must be at least one")
    if explicit_map.strip():
        try:
            capacities = {
                gpu.strip(): int(count.strip())
                for gpu, count in (
                    item.split(":", 1) for item in explicit_map.split(",") if item.strip()
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
    if family != "carafe2":
        return capacities
    free = free_memory_mib if free_memory_mib is not None else gpu_free_memory_mib()
    if not free:
        return capacities
    # A Carafe2 RT worker peaks around 3.0 GiB here. Keep 256 MiB unassigned
    # and cap only downward, allowing GPUs occupied by other applications to
    # participate safely with fewer workers.
    memory_per_worker = 3200
    safety_margin = 256
    for gpu in gpus:
        if gpu not in free:
            continue
        safe_workers = max(0, (free[gpu] - safety_margin) // memory_per_worker)
        capacities[gpu] = min(requested, safe_workers)
    if not any(capacities.values()):
        raise ValueError(f"No selected GPU has enough free memory for one Carafe2 worker: {free}")
    return capacities


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_metrics(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return all(
            isinstance(payload.get(organism), dict)
            and isinstance(payload[organism].get("raw_mae_minutes"), (int, float))
            and math.isfinite(float(payload[organism]["raw_mae_minutes"]))
            for organism in ORGANISMS
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def table_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def legacy_training_complete(family: str, out: Path, epochs: int) -> bool:
    if family == "diann":
        return (
            (out / "rt.d0.finetuned.pt").is_file()
            and (out / "dict.txt").is_file()
            and (out / "run_metadata.json").is_file()
            and table_rows(out / "training_history.csv") >= epochs
        )
    metrics_path = out / "model_evaluation_metrics.json"
    if not ((out / "rt_model.pt").is_file() and table_rows(out / "rt_history.tsv") >= epochs):
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return isinstance(metrics.get("rt", {}).get("finetuned"), dict)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def training_complete(family: str, out: Path, epochs: int) -> bool:
    marker = out / "training_complete.json"
    model_exists = (
        (out / "rt.d0.finetuned.pt").is_file()
        if family == "diann"
        else (out / "rt_model.pt").is_file()
    )
    if marker.is_file() and model_exists:
        return True
    if legacy_training_complete(family, out, epochs):
        atomic_json(marker, {
            "family": family,
            "epochs": epochs,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recognized_from_legacy_artifacts": True,
        })
        return True
    return False


def snapshot_tests(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for organism in ORGANISMS:
        shutil.copy2(source / f"{organism}.tsv", destination / f"{organism}.tsv")
    shutil.copy2(source.parent / "rt_alignment.json", destination / "rt_alignment.json")


def command_for_training(args: argparse.Namespace, job: dict, tests: Path) -> list[str]:
    if args.family == "diann":
        return [
            args.diann_python, str(DIANN_ROOT / "train_carafe_rt.py"),
            "--train-split", str(job["train"]), "--validation-split", str(tests / "celegans.tsv"),
            "--model-dir", str(DIANN_MODELS), "--output-dir", str(job["out"]),
            "--device", "cuda", "--trainable", "all", "--epochs", str(args.epochs),
            "--warmup-epochs", str(args.warmup_epochs), "--batch-size", "512",
            "--learning-rate", "0.000280195788", "--loss", "mae",
            "--heavy-nterm-weight", "6.0", "--heavy-lysine-weight", "2.0",
            "--heavy-both-weight", "1.0", "--seed", job["seed"],
        ]
    return [
        args.carafe2_python, str(CARAFE2_ROOT / "src/main/resources/py/v2/ai.py"),
        "--in_dir", str(job["input"]), "--out_dir", str(job["out"]),
        "--device", "gpu", "--tf_type", "rt", "--seed", job["seed"], "--use_best_model",
        "--rt_validation_file", str(tests / "celegans.tsv"), "--rt_epoch", str(args.epochs),
        "--rt_warmup_epoch", str(args.warmup_epochs), "--rt_batch_size", "2048",
        "--rt_lr", "0.000216144694", "--rt_loss", "l1",
        "--rt_heavy_nterm_weight", "6.0", "--rt_heavy_lysine_weight", "2.0",
        "--rt_heavy_both_weight", "2.0", "--no_adjust_rt_batch_size",
    ]


def command_for_evaluation(args: argparse.Namespace, job: dict, tests: Path) -> list[str]:
    if args.family == "diann":
        return [
            args.diann_python, str(DIANN_ROOT / "evaluate_organism_tests.py"),
            "--run-dir", str(job["out"]), "--tests-dir", str(tests), "--device", "cuda",
            "--rt-alignment", str(tests / "rt_alignment.json"),
        ]
    return [
        args.carafe2_python, str(CARAFE2_ROOT / "evaluate_organism_tests.py"),
        "--run-dir", str(job["out"]), "--tests-dir", str(tests), "--device", "gpu",
        "--python", args.carafe2_python, "--rt-alignment", str(tests / "rt_alignment.json"),
    ]


def run_logged(command: list[str], environment: dict[str, str], log: TextIO) -> None:
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}")


def epoch_from_log(path: Path, family: str) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 200_000))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    pattern = r"epoch=(\d+)/(\d+)" if family == "diann" else r"Epoch=(\d+)"
    if "] START " in text:
        text = text.rsplit("] START ", 1)[-1]
    matches = list(re.finditer(pattern, text, re.IGNORECASE if family == "diann" else 0))
    if not matches:
        return None
    match = matches[-1]
    return (int(match.group(1)), int(match.group(2))) if family == "diann" else (int(match.group(1)), 0)


def bar(done: int, total: int, width: int = 28) -> str:
    fraction = 1.0 if total == 0 else min(1.0, done / total)
    filled = round(width * fraction)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def main() -> int:
    args = parse_args()
    if args.family == "carafe2":
        try:
            args.carafe2_python = str(validate_carafe2_python(args.carafe2_python))
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"Carafe2 Python: {args.carafe2_python}", flush=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != 3 or len(set(gpus)) != 3:
        raise SystemExit(f"--gpus must contain exactly three distinct IDs, got: {args.gpus}")
    try:
        slot_counts = resolve_slot_counts(
            gpus, args.jobs_per_gpu, args.jobs_per_gpu_map, args.family
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    slot_summary = ", ".join(f"GPU {gpu}×{slot_counts[gpu]}" for gpu in gpus)
    disabled_gpus = [gpu for gpu in gpus if slot_counts[gpu] == 0]
    if disabled_gpus:
        print(
            "Memory guard disabled worker allocation on GPU(s): " + ", ".join(disabled_gpus),
            flush=True,
        )
    print(f"Worker allocation: {slot_summary}", flush=True)

    seed_filter, condition_filter = selected(args.seeds), selected(args.conditions)
    jobs: list[dict] = []
    for seed_dir in sorted((args.study_root / "prepared").glob("seed_*")):
        seed = seed_dir.name.removeprefix("seed_")
        if seed_filter and seed not in seed_filter:
            continue
        for train in sorted((seed_dir / "trains").glob("*.tsv")):
            condition = train.stem
            if condition_filter and condition not in condition_filter:
                continue
            out = args.study_root / "runs" / f"seed_{seed}" / condition
            jobs.append({
                "seed": seed, "condition": condition, "train": train, "seed_dir": seed_dir,
                "out": out, "input": out / "input", "name": f"seed_{seed}/{condition}",
            })
    if not jobs:
        raise SystemExit("No matching prepared jobs found")

    states: dict[int, dict] = {}
    state_lock = threading.Lock()
    work: queue.Queue[dict] = queue.Queue()
    failures: list[str] = []
    stop_event = threading.Event()
    completed_at_start = sum(valid_metrics(job["out"] / "organism_test_metrics.json") for job in jobs)
    for job in jobs:
        if not valid_metrics(job["out"] / "organism_test_metrics.json"):
            work.put(job)

    if args.dry_run:
        print(
            f"{args.family.upper()} dry run: {completed_at_start}/{len(jobs)} complete, "
            f"{work.qsize()} resumable jobs, GPUs={','.join(gpus)}, "
            f"slots={slot_summary}"
        )
        return 0

    def set_state(slot: int, **values: object) -> None:
        with state_lock:
            states[slot] = {**states.get(slot, {}), **values}

    def worker(slot: int, gpu: str) -> None:
        while True:
            if stop_event.is_set():
                set_state(slot, phase="stopped", name="-")
                return
            try:
                job = work.get_nowait()
            except queue.Empty:
                set_state(slot, phase="idle", name="-")
                return
            out: Path = job["out"]
            out.mkdir(parents=True, exist_ok=True)
            lock_handle = (out / ".organism_scaling.lock").open("a+")
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                set_state(slot, phase="active elsewhere", name=job["name"], log=out / "run.log")
                work.task_done()
                lock_handle.close()
                continue
            log_path = out / "run.log"
            set_state(slot, phase="starting", name=job["name"], log=log_path)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            try:
                tests = out / "test_inputs" if args.family == "diann" else job["input"] / "tests"
                snapshot_tests(job["seed_dir"] / "tests", tests)
                if args.family == "carafe2":
                    job["input"].mkdir(parents=True, exist_ok=True)
                    shutil.copy2(job["train"], job["input"] / "rt_train_data.tsv")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START family={args.family} "
                        f"gpu={gpu} slot={slot} run={job['name']}\n"
                    )
                    if not training_complete(args.family, out, args.epochs):
                        set_state(slot, phase="training", name=job["name"], log=log_path)
                        run_logged(command_for_training(args, job, tests), environment, log)
                        if not legacy_training_complete(args.family, out, args.epochs):
                            raise RuntimeError("training command exited successfully but completion artifacts are invalid")
                        atomic_json(out / "training_complete.json", {
                            "family": args.family, "epochs": args.epochs, "gpu": gpu,
                            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "recognized_from_legacy_artifacts": False,
                        })
                    if not valid_metrics(out / "organism_test_metrics.json"):
                        set_state(slot, phase="evaluation", name=job["name"], log=log_path)
                        run_logged(command_for_evaluation(args, job, tests), environment, log)
                    if not valid_metrics(out / "organism_test_metrics.json"):
                        raise RuntimeError("evaluation did not produce valid organism metrics")
                    log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE run={job['name']}\n")
                set_state(slot, phase="complete", name=job["name"], log=log_path)
            except Exception as error:
                message = f"{job['name']}: {error} (log: {log_path})"
                failures.append(message)
                stop_event.set()
                set_state(slot, phase="FAILED", name=job["name"], log=log_path, error=str(error))
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAILED {error}\n")
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                lock_handle.close()
                work.task_done()

    slots = [
        (gpu, local_slot)
        for gpu in gpus
        for local_slot in range(1, slot_counts[gpu] + 1)
    ]
    threads = [
        threading.Thread(
            target=worker, args=(index, gpu),
            name=f"gpu-{gpu}-slot-{local_slot}", daemon=False,
        )
        for index, (gpu, local_slot) in enumerate(slots)
    ]
    for index, (gpu, local_slot) in enumerate(slots):
        states[index] = {
            "gpu": gpu, "local_slot": local_slot, "phase": "queued", "name": "-"
        }
    for thread in threads:
        thread.start()

    is_tty = sys.stdout.isatty()
    last_print = 0.0
    while any(thread.is_alive() for thread in threads):
        now = time.monotonic()
        complete = sum(valid_metrics(job["out"] / "organism_test_metrics.json") for job in jobs)
        with state_lock:
            snapshot = {key: value.copy() for key, value in states.items()}
        serializable_snapshot = {}
        for slot_id, item in snapshot.items():
            clean_item = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in item.items()
            }
            epoch = epoch_from_log(item["log"], args.family) if item.get("log") else None
            if epoch:
                clean_item["epoch"] = epoch[0]
                clean_item["epochs"] = epoch[1] or args.epochs
            serializable_snapshot[slot_id] = clean_item
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "family": args.family,
            "complete": complete, "total": len(jobs), "queued": work.qsize(),
            "jobs_per_gpu": slot_counts, "gpus": gpus, "slots": serializable_snapshot,
            "failures": failures,
        }
        atomic_json(args.study_root / "parallel_status.json", payload)
        if is_tty or now - last_print >= 30:
            lines = [
                f"{args.family.upper()} · {slot_summary}",
                f"TOTAL {bar(complete, len(jobs))} {complete}/{len(jobs)} complete · {work.qsize()} queued",
            ]
            for index in sorted(serializable_snapshot):
                item = serializable_snapshot[index]
                epoch = (
                    (int(item["epoch"]), int(item["epochs"]))
                    if "epoch" in item and "epochs" in item else None
                )
                progress = ""
                if epoch:
                    total_epochs = epoch[1] or args.epochs
                    progress = f" · {bar(epoch[0], total_epochs, 12)} {epoch[0]}/{total_epochs}"
                lines.append(
                    f"GPU {item['gpu']} slot {item.get('local_slot', '?')}: "
                    f"{item.get('phase', '-'):>10} · {item.get('name', '-')}{progress}"
                )
            text = "\n".join(lines)
            print(("\033[2J\033[H" if is_tty else "") + text, flush=True)
            last_print = now
        time.sleep(args.refresh_seconds)

    for thread in threads:
        thread.join()
    complete = sum(valid_metrics(job["out"] / "organism_test_metrics.json") for job in jobs)
    atomic_json(args.study_root / "parallel_status.json", {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "family": args.family,
        "complete": complete, "total": len(jobs), "queued": work.qsize(),
        "jobs_per_gpu": slot_counts, "gpus": gpus, "failures": failures,
    })
    print(f"{args.family.upper()} finished: {complete}/{len(jobs)} complete "
          f"({complete - completed_at_start} added), {len(failures)} failed", flush=True)
    if failures:
        print("\n".join(f"FAILED: {item}" for item in failures), file=sys.stderr)
        return 1
    return 0 if complete == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
