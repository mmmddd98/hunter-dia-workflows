#!/usr/bin/env python3
"""Resume and monitor the joint DIA-NN/Carafe2 RT normalization study.

The supervisor can be started while legacy single-family launchers are still
running.  It adopts those launchers, waits for them without interrupting their
work, and then fills every missing model/method run.  A compact terminal panel
shows completed and remaining evaluable runs throughout the campaign.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable


METHODS = (
    "global_60min",
    "source_native",
    "anchor_irt",
    "linear",
    "relative_rt",
    "lowess",
    "isotonic",
    "monotonic_spline",
    "calib_rt",
    "kde",
    "chronologer_hi",
)
METHOD_LABELS = {
    "global_60min": "Global 60 min",
    "source_native": "Source-native",
    "anchor_irt": "Joint anchor RT",
    "linear": "Robust linear",
    "relative_rt": "Relative gradient RT",
    "lowess": "LOWESS",
    "isotonic": "Isotonic",
    "monotonic_spline": "Monotonic spline",
    "calib_rt": "Calib-RT",
    "kde": "KDE ridge",
    "chronologer_hi": "Chronologer HI",
}
ORGANISMS = ("celegans", "triplemix", "human")
SEEDS = (2024, 2025, 2026, 2027, 2028)
CONDITIONS = (
    "ce_010",
    "ce_025",
    "ce_050",
    "ce_075",
    "ce_100",
    "ce100_triple_025",
    "ce100_triple_050",
    "ce100_triple_100",
    "ce100_human_025",
    "ce100_human_050",
    "ce100_human_100",
    "ce100_all_025",
    "ce100_all_050",
    "ce100_all_100",
)
RUNS_PER_ARM = len(SEEDS) * len(CONDITIONS)

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIANN_ROOT = Path(
    os.environ.get(
        "DIANN_FINETUNE_OUTPUT_ROOT",
        "./shared_data/diann_fintune_out",
    )
).expanduser() / "rt_normalization_scaling_v1"
DEFAULT_CARAFE2_ROOT = Path(
    os.environ.get(
        "CARAFE2_FINETUNE_OUTPUT_ROOT",
        "./shared_data/carafe2_finetune_out",
    )
).expanduser() / "rt_normalization_scaling_v1"
RUNNERS = {
    "DIA-NN": Path(
        os.environ.get("DIANN_FINETUNE_ROOT", SCRIPT_ROOT.parent / "diann-fintune")
    ).expanduser() / "scripts/start_diann_organism_scaling.sh",
    "Carafe2": SCRIPT_ROOT / "carafe2-semi/scripts/start_carafe2_organism_scaling.sh",
}
LEGACY_LAUNCHERS = {
    "DIA-NN": "ma/data_collect/start_diann_rt_normalization_scaling_study.sh",
    "Carafe2": "ma/data_collect/start_carafe2_rt_normalization_scaling_study.sh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diann-root", type=Path, default=DEFAULT_DIANN_ROOT)
    parser.add_argument("--carafe2-root", type=Path, default=DEFAULT_CARAFE2_ROOT)
    parser.add_argument(
        "--modes",
        default=",".join(METHODS),
        help="Comma-separated target modes (default: the full normalization study)",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(RUNNERS),
        help="Model family to run; repeat for both (default: both families)",
    )
    parser.add_argument(
        "--wait-for-pid",
        action="append",
        default=[],
        type=int,
        help="Additional existing launcher PID to adopt; may be repeated",
    )
    parser.add_argument(
        "--no-adopt-active",
        action="store_true",
        help="Do not automatically wait for active legacy DIA-NN/Carafe2 launchers",
    )
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--event-log", type=Path)
    return parser.parse_args()


def selected_methods(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(modes) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown RT target mode(s): {', '.join(unknown)}")
    if not modes:
        raise SystemExit("At least one RT target mode is required")
    if len(set(modes)) != len(modes):
        raise SystemExit("RT target modes must not be repeated")
    return modes


def valid_metrics(path: Path) -> bool:
    """Match the notebook's definition of one evaluable model run."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    for organism in ORGANISMS:
        item = payload.get(organism)
        if not isinstance(item, dict) or "raw_mae_minutes" not in item:
            return False
        try:
            value = float(item["raw_mae_minutes"])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def completed_runs(root: Path, mode: str) -> int:
    run_root = root / mode / "runs"
    return sum(
        valid_metrics(run_root / f"seed_{seed}" / condition / "organism_test_metrics.json")
        for seed in SEEDS
        for condition in CONDITIONS
    )


def completion_snapshot(
    roots: dict[str, Path], modes: Iterable[str]
) -> dict[str, object]:
    selected = tuple(modes)
    by_family: dict[str, int] = {}
    by_method: dict[str, dict[str, int]] = {}
    for family, root in roots.items():
        by_family[family] = 0
        for mode in selected:
            count = completed_runs(root, mode)
            by_family[family] += count
            by_method.setdefault(mode, {})[family] = count
    total = len(roots) * len(selected) * RUNS_PER_ARM
    complete = sum(by_family.values())
    return {
        "total": total,
        "complete": complete,
        "missing": total - complete,
        "by_family": by_family,
        "by_method": by_method,
    }


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def process_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


def ancestor_pids(pid: int) -> set[int]:
    """Return ancestors so a launcher never accidentally waits for its own shell."""
    ancestors: set[int] = set()
    current = pid
    while current > 1:
        try:
            status = Path(f"/proc/{current}/status").read_text(encoding="utf-8")
            parent_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
            parent = int(parent_line.split()[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration, ValueError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def discover_legacy_launchers(families: Iterable[str]) -> list[int]:
    own_pid = os.getpid()
    excluded = ancestor_pids(own_pid) | {own_pid}
    markers = tuple(LEGACY_LAUNCHERS[family] for family in families)
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        command = process_cmdline(pid)
        if any(marker in command for marker in markers):
            found.append(pid)
    return sorted(found)


def progress_bar(done: int, total: int, width: int) -> str:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, done / total))
    filled = round(width * fraction)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def active_run_progress(log_path: Path, max_bytes: int = 2_000_000) -> dict[str, object]:
    """Read the current queue section and extract run identity and epoch progress."""
    if not log_path.is_file():
        return {"state": "initializing"}
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return {"state": "initializing"}

    marker = "unified queue: start"
    section = text.rsplit(marker, 1)[-1] if marker in text else text
    run_matches = list(
        re.finditer(r"/runs/seed_(\d+)/([^/\s\"']+)", section)
    )
    epoch_section = section[run_matches[-1].start():] if run_matches else section
    epoch_matches = list(
        re.finditer(r"(?i)\bepoch\s*[=: ]\s*(\d+)\s*/\s*(\d+)", epoch_section)
    )
    progress: dict[str, object] = {}
    if run_matches:
        progress["seed"] = int(run_matches[-1].group(1))
        progress["condition"] = run_matches[-1].group(2)
    if epoch_matches:
        epoch = int(epoch_matches[-1].group(1))
        epochs = int(epoch_matches[-1].group(2))
        progress.update(
            {
                "epoch": epoch,
                "epochs": epochs,
                "state": "training" if epoch < epochs else "evaluation",
            }
        )
    elif "Waiting for exclusive all-GPU study lock" in section and not any(
        phrase in section
        for phrase in ("will distribute each experiment", "will use all", "will use GPUs")
    ):
        progress["state"] = "waiting for all-GPU lock"
    elif run_matches:
        progress["state"] = "initializing/evaluation"
    else:
        progress["state"] = "initializing"
    return progress


class Reporter:
    def __init__(
        self,
        roots: dict[str, Path],
        modes: tuple[str, ...],
        status_file: Path,
        event_log: Path,
    ) -> None:
        self.roots = roots
        self.modes = modes
        self.status_file = status_file
        self.event_log = event_log
        self.is_tty = sys.stdout.isatty()
        self.last_non_tty_print = 0.0
        self.last_event: str | None = None

    def event(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{message}\n")
        self.last_event = message

    def render(
        self,
        phase: str,
        detail: str,
        *,
        force: bool = False,
        active_family: str | None = None,
        active_mode: str | None = None,
        active_log: Path | None = None,
        active_status: Path | None = None,
    ) -> dict[str, object]:
        snapshot = completion_snapshot(self.roots, self.modes)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        run_progress = active_run_progress(active_log) if active_log else None
        parallel_progress = None
        if active_status and active_status.is_file():
            try:
                parallel_progress = json.loads(active_status.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        payload = {
            "updated_at": now,
            "phase": phase,
            "detail": detail,
            "modes": list(self.modes),
            "runs_per_model_method": RUNS_PER_ARM,
            "active": {
                "family": active_family,
                "mode": active_mode,
                "run": run_progress,
                "parallel": parallel_progress,
            } if active_family and active_mode else None,
            **snapshot,
        }
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_file.with_suffix(self.status_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.status_file)

        columns = shutil.get_terminal_size(fallback=(120, 30)).columns
        bar_width = max(12, min(42, columns - 66))
        family_total = len(self.modes) * RUNS_PER_ARM
        family_scope = " + ".join(self.roots)
        lines = [
            f"RT normalization study · {family_scope} resumable queue",
            f"Updated: {now}    GPUs: 3 × 2 independent single-GPU runs",
            "",
            (
                f"TOTAL   {progress_bar(snapshot['complete'], snapshot['total'], bar_width)} "
                f"{snapshot['complete']:4d}/{snapshot['total']:4d} complete · "
                f"{snapshot['missing']:4d} missing"
            ),
        ]
        for family in self.roots:
            done = snapshot["by_family"][family]
            lines.append(
                f"{family:<7} {progress_bar(done, family_total, bar_width)} "
                f"{done:4d}/{family_total:4d} complete · {family_total - done:4d} missing"
            )
        if active_family and active_mode:
            method_done = snapshot["by_method"][active_mode][active_family]
            lines.extend(
                (
                    "",
                    (
                        f"METHOD  {progress_bar(method_done, RUNS_PER_ARM, bar_width)} "
                        f"{method_done:2d}/{RUNS_PER_ARM:2d} runs · "
                        f"{METHOD_LABELS[active_mode]}"
                    ),
                )
            )
            if run_progress and not (parallel_progress and parallel_progress.get("slots")):
                run_name = ""
                if "seed" in run_progress and "condition" in run_progress:
                    run_name = f"seed {run_progress['seed']} / {run_progress['condition']} · "
                if "epoch" in run_progress and "epochs" in run_progress:
                    epoch = int(run_progress["epoch"])
                    epochs = int(run_progress["epochs"])
                    lines.append(
                        f"RUN     {progress_bar(epoch, epochs, bar_width)} "
                        f"{epoch:2d}/{epochs:2d} epochs · {run_name}{run_progress['state']}"
                    )
                else:
                    lines.append(f"RUN     {'[' + '·' * bar_width + ']'} {run_name}{run_progress['state']}")
            if parallel_progress and parallel_progress.get("slots"):
                lines.append("")
                for slot_id, item in sorted(
                    parallel_progress["slots"].items(), key=lambda pair: int(pair[0])
                ):
                    epoch = int(item.get("epoch", 0))
                    epochs = int(item.get("epochs", 0))
                    slot_bar = (
                        f"{progress_bar(epoch, epochs, 12)} {epoch:2d}/{epochs:2d} · "
                        if epochs else "[············] --/-- · "
                    )
                    lines.append(
                        f"GPU {item.get('gpu', '?')} · slot {int(slot_id) % 2 + 1}: "
                        f"{slot_bar}{item.get('phase', '-')} · {item.get('name', '-')}"
                    )
        lines.extend(("", f"Phase:   {phase}", f"Current: {detail}"))
        if self.last_event:
            lines.append(f"Last:    {self.last_event}")
        text = "\n".join(lines)
        monotonic = time.monotonic()
        if self.is_tty:
            print("\033[2J\033[H" + text, flush=True)
        elif force or monotonic - self.last_non_tty_print >= 60:
            print(text, flush=True)
            self.last_non_tty_print = monotonic
        return snapshot


def wait_for_adopted_processes(
    pids: Iterable[int], reporter: Reporter, refresh_seconds: float
) -> None:
    remaining = sorted({pid for pid in pids if process_alive(pid)})
    if not remaining:
        return
    reporter.event(f"Adopted active legacy launchers: {', '.join(map(str, remaining))}")
    while remaining:
        labels = [f"{pid} ({Path(process_cmdline(pid).split()[-1]).name})" for pid in remaining]
        reporter.render("waiting for current studies", ", ".join(labels), force=True)
        time.sleep(refresh_seconds)
        remaining = [pid for pid in remaining if process_alive(pid)]
    reporter.event("All adopted legacy launchers finished; starting the unified queue")


def run_arm(
    family: str,
    root: Path,
    counterpart_root: Path,
    mode: str,
    reporter: Reporter,
    refresh_seconds: float,
) -> None:
    if completed_runs(root, mode) >= RUNS_PER_ARM:
        reporter.event(f"Skipped complete arm: {family} / {METHOD_LABELS[mode]}")
        reporter.render("resuming", f"skip complete {family} / {METHOD_LABELS[mode]}", force=True)
        return

    method_root = root / mode
    method_root.mkdir(parents=True, exist_ok=True)
    log_path = method_root / "study.log"
    environment = os.environ.copy()
    environment["RT_TARGET"] = mode
    if family == "DIA-NN":
        environment["CARAFE2_STUDY_ROOT"] = str(counterpart_root / mode)
    else:
        environment["DIANN_STUDY_ROOT"] = str(counterpart_root / mode)
    command = ["bash", str(RUNNERS[family]), str(method_root)]
    detail = f"{family} / {METHOD_LABELS[mode]}"
    reporter.event(f"Starting {detail}; log={log_path}")

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] unified queue: start {detail}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                reporter.render(
                    "training and evaluating",
                    detail,
                    active_family=family,
                    active_mode=mode,
                    active_log=log_path,
                    active_status=method_root / "parallel_status.json",
                )
                time.sleep(refresh_seconds)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise
        return_code = process.wait()
    completed_after_run = completed_runs(root, mode)
    if return_code != 0 and completed_after_run >= RUNS_PER_ARM:
        reporter.event(
            f"Ignored non-zero runner exit for complete arm: {detail} "
            f"({completed_after_run}/{RUNS_PER_ARM} valid runs)"
        )
        return_code = 0
    if return_code != 0:
        reporter.event(f"FAILED {detail} with exit code {return_code}")
        raise RuntimeError(f"{detail} failed with exit code {return_code}; see {log_path}")
    reporter.event(f"Finished {detail}: {completed_after_run}/{RUNS_PER_ARM} evaluable runs")
    reporter.render("arm complete", detail, force=True)


def run_analysis(roots: dict[str, Path], reporter: Reporter) -> None:
    output_dir = roots["DIA-NN"] / "analysis"
    log_path = output_dir / "analysis.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_ROOT / "ma/data_collect/analyze_rt_normalization_scaling.py"),
        "--diann-root",
        str(roots["DIA-NN"]),
        "--carafe2-root",
        str(roots["Carafe2"]),
        "--output-dir",
        str(output_dir),
    ]
    reporter.event(f"Starting final analysis; log={log_path}")
    reporter.render("final analysis", str(output_dir), force=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=SCRIPT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Final analysis failed; see {log_path}")
    reporter.event(f"Final analysis complete: {output_dir}")


def main() -> int:
    args = parse_args()
    modes = selected_methods(args.modes)
    all_roots = {
        "DIA-NN": args.diann_root.resolve(),
        "Carafe2": args.carafe2_root.resolve(),
    }
    families = tuple(args.family or all_roots)
    if len(set(families)) != len(families):
        raise SystemExit("Model families must not be repeated")
    roots = {family: all_roots[family] for family in families}
    primary_root = roots[families[0]]
    status_file = (args.status_file or primary_root / "queue_status.json").resolve()
    event_log = (args.event_log or primary_root / "queue_events.log").resolve()
    reporter = Reporter(roots, modes, status_file, event_log)

    adopted = set(args.wait_for_pid)
    if not args.no_adopt_active:
        adopted.update(discover_legacy_launchers(families))
    snapshot = reporter.render("initializing", "checking existing results", force=True)

    planned = []
    for mode in modes:
        for family in roots:
            if completed_runs(roots[family], mode) < RUNS_PER_ARM:
                planned.append(f"{family}/{mode}")
    if args.dry_run:
        print(f"Adopted active PIDs: {sorted(adopted) or 'none'}")
        print(f"Completed: {snapshot['complete']}/{snapshot['total']}; missing: {snapshot['missing']}")
        print("Incomplete arms in execution order:")
        for item in planned:
            print(f"  - {item}")
        return 0

    family_slug = "_".join(family.lower().replace("-", "") for family in families)
    lock_file = args.lock_file or Path(f"/tmp/rt_normalization_scaling_{family_slug}_queue.lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another normalization queue owns {lock_file}")
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    pid_file = primary_root / "queue_supervisor.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    reporter.event(f"Unified queue supervisor started with PID {os.getpid()}")
    try:
        wait_for_adopted_processes(adopted, reporter, args.refresh_seconds)
        for mode in modes:
            for family, root in roots.items():
                counterpart = all_roots["Carafe2" if family == "DIA-NN" else "DIA-NN"]
                run_arm(
                    family, root, counterpart, mode, reporter, args.refresh_seconds,
                )
        final_snapshot = reporter.render("checking completion", "all queued arms returned", force=True)
        if final_snapshot["missing"]:
            raise RuntimeError(
                f"Queue returned with {final_snapshot['missing']} evaluable runs still missing"
            )
        if len(roots) == 2:
            run_analysis(roots, reporter)
            detail = "all runs and final analysis finished"
        else:
            detail = f"all {families[0]} runs finished"
        reporter.event(f"Normalization queue completed successfully: {detail}")
        reporter.render("complete", detail, force=True)
        return 0
    except KeyboardInterrupt:
        reporter.event("Queue interrupted by user; completed results remain resumable")
        reporter.render("interrupted", "restart the same command to resume", force=True)
        return 130
    except Exception as error:
        reporter.event(f"Queue stopped: {error}")
        reporter.render("stopped with error", str(error), force=True)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
