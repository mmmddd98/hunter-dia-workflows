#!/usr/bin/env python3
"""Split a frozen Heavy-Dimethyl test manifest into finalist-specific manifests.

Each output manifest contains one validation-selected finalist (all seeds) and
its matched within-arm controls.  This is purely a scheduling convenience:
the test predictions and the frozen selection remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-evaluation-dir", type=Path, required=True)
    return parser.parse_args()


def label(meta: dict) -> str:
    raw = "_".join(
        str(meta[key]) for key in ("target", "arm", "profile", "loss", "lr")
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def main() -> None:
    args = parse_args()
    root = args.test_evaluation_dir.resolve()
    manifest_path = root / "test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])
    finalists = [job for job in jobs if job.get("metadata", {}).get("selection_role") == "finalist"]
    if not finalists:
        raise ValueError(f"No finalist jobs in {manifest_path}")
    groups: dict[tuple, list[dict]] = {}
    for job in finalists:
        meta = job["metadata"]
        key = tuple(meta[field] for field in ("target", "arm", "profile", "loss", "lr"))
        groups.setdefault(key, []).append(job)

    output = root / "window_manifests"
    output.mkdir(exist_ok=True)
    index = []
    for key, finalist_jobs in sorted(groups.items()):
        exemplar = finalist_jobs[0]["metadata"]
        control_jobs = [
            job for job in jobs
            if job.get("metadata", {}).get("selection_role") == "matched_control"
            and all(job["metadata"][field] == exemplar[field] for field in ("target", "arm", "loss", "lr"))
        ]
        if len(finalist_jobs) != 3 or len(control_jobs) != 3:
            raise ValueError(
                f"Expected three finalist and three control seeds for {key}; "
                f"found {len(finalist_jobs)} and {len(control_jobs)}"
            )
        name = label(exemplar)
        window_manifest = {
            **{key: value for key, value in manifest.items() if key != "jobs"},
            "name": name,
            "state_root": str(root / ".window_state" / name),
            "log_root": str(root / ".window_logs" / name),
            "jobs": sorted(finalist_jobs + control_jobs, key=lambda job: job["id"]),
        }
        path = output / f"{name}.json"
        path.write_text(json.dumps(window_manifest, indent=2) + "\n", encoding="utf-8")
        index.append({"name": name, "manifest": str(path), "jobs": len(window_manifest["jobs"])})
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"window_manifests": index}, indent=2))


if __name__ == "__main__":
    main()
