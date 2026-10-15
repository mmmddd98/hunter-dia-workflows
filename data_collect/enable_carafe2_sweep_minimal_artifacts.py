#!/usr/bin/env python3
"""Switch a stopped Carafe2 RT sweep to numeric-only per-run artifacts.

Completed jobs remain completed: their marker fingerprints are migrated after
the train command gains ``--sweep_minimal_artifacts``.  The program refuses to
operate while the manifest controller lock is held, so it cannot race a live
sweep.  It changes no dataset, model, hyperparameter, validation command, or
selection metric.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Perform the migration; otherwise only report it.")
    return parser.parse_args()


def fingerprint(job: dict) -> str:
    relevant = {
        "commands": job["commands"],
        "workdir": job.get("workdir"),
        "environment": job.get("environment", {}),
        "completion_files": job.get("completion_files", []),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "job"


def add_flag(job: dict) -> bool:
    command = job["commands"][0]
    if "--sweep_minimal_artifacts" in command:
        return False
    command.append("--sweep_minimal_artifacts")
    return True


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.study_root.expanduser().resolve()
    sweep_path, study_path = root / "sweep_manifest.json", root / "study_manifest.json"
    if not sweep_path.is_file() or not study_path.is_file():
        raise FileNotFoundError("Expected sweep_manifest.json and study_manifest.json under --study-root")

    lock_path = root / ".sweep_state" / "controller.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("Sweep controller is still running. Stop it cleanly before changing the manifest.") from error

        sweep, study = load(sweep_path), load(study_path)
        if [job["id"] for job in sweep["jobs"]] != [job["id"] for job in study["jobs"]]:
            raise ValueError("Sweep and study manifests no longer contain the same ordered job IDs")
        changed = sum(add_flag(job) for job in sweep["jobs"])
        for job in study["jobs"]:
            add_flag(job)

        marker_root = root / ".sweep_state" / "markers"
        migrated = 0
        for job in sweep["jobs"]:
            marker = marker_root / f"{safe_name(job['id'])}.complete.json"
            if not marker.is_file() or not all(Path(path).is_file() for path in job["completion_files"]):
                continue
            record = load(marker)
            record["fingerprint"] = fingerprint(job)
            if args.apply:
                dump(marker, record)
            migrated += 1

        print(
            f"{changed} train commands would receive --sweep_minimal_artifacts; "
            f"{migrated} completed markers would be preserved."
        )
        if not args.apply:
            print("Dry run only. Re-run with --apply after reviewing this summary.")
            return
        for path in (sweep_path, study_path):
            backup = path.with_suffix(path.suffix + ".before_minimal_artifacts")
            if not backup.exists():
                shutil.copy2(path, backup)
        dump(sweep_path, sweep)
        dump(study_path, study)
        print("Migration complete. Resume with the normal sweep launcher.")


if __name__ == "__main__":
    main()
