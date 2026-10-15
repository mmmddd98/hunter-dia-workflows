#!/usr/bin/env python3
"""Atomically write reproducible Carafe job status/provenance metadata."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def nvidia_smi() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    keys = ("index", "name", "driver_version", "memory_total_mib")
    return [dict(zip(keys, (part.strip() for part in line.split(",")))) for line in output.splitlines() if line]


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mirror-output", type=Path)
    parser.add_argument("--status", required=True, choices=("queued", "running", "failed", "succeeded"))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--jar", required=True)
    parser.add_argument("--jar-sha256", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--library-path")
    parser.add_argument("--started-at")
    parser.add_argument("--resumed-from-fingerprint")
    parser.add_argument("--resumed-from-jar-sha256")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "status": args.status,
        "stage": args.stage,
        "message": args.message,
        "updated_at": now,
        "started_at": args.started_at or now,
        "ended_at": now if args.status in {"failed", "succeeded"} else None,
        "job": args.job,
        "hostname": os.uname().nodename,
        "pid": os.getppid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "run_fingerprint": args.fingerprint,
        "carafe_jar": args.jar,
        "carafe_jar_sha256": args.jar_sha256,
        "command_file": args.command_file,
        "exit_code": args.exit_code,
        "library_path": args.library_path,
        "resumed_from_fingerprint": args.resumed_from_fingerprint,
        "resumed_from_carafe_jar_sha256": args.resumed_from_jar_sha256,
        "python": sys.executable,
        "torch_version": package_version("torch"),
        "peptdeep_version": package_version("peptdeep"),
        "gpus": nvidia_smi(),
    }
    atomic_write(args.output, payload)
    if args.mirror_output:
        atomic_write(args.mirror_output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
