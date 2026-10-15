#!/usr/bin/env python3
"""Register an immutable, externally fine-tuned Carafe2 RT model directory."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--target-space", required=True)
    parser.add_argument("--label", required=True, help="Stable folder/registry label, e.g. 2026-08-25_carafe2rt_..._v1")
    parser.add_argument("--source-run", default="")
    parser.add_argument("--status", default="ready_for_cpu_validation")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.registry_root.resolve()
    model_dir = args.model_dir.resolve()
    models_dir = root / "models"
    try:
        relative_path = model_dir.relative_to(models_dir)
    except ValueError as error:
        raise SystemExit(f"--model-dir must be below {models_dir}: {model_dir}") from error
    if len(relative_path.parts) != 1:
        raise SystemExit("A registered model must be one direct child of models/")
    if relative_path.name != args.label:
        raise SystemExit("--label must match the model directory name")
    checkpoint = model_dir / "rt_model.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"Missing required Carafe2 RT checkpoint: {checkpoint}")

    checksum = sha256(checkpoint)
    model_id = f"carafe2rt_{checksum[:12]}"
    registry = root / "models_list.csv"
    if not registry.is_file():
        raise SystemExit(f"Registry missing: {registry}")
    row = {
        "model_id": model_id,
        "created_date": date.today().isoformat(),
        "package_name": args.label,
        "relative_path": str(relative_path),
        "model_family": "Carafe2",
        "target_space": args.target_space,
        "rt_model_sha256": checksum,
        "source_run": args.source_run,
        "status": args.status,
    }
    fields = list(row)
    # A package may be prepared from different GPU hosts at the same time.
    # Keep the registry append serialised and durable, as the DIA-NN exporter does.
    registry_has_header = registry.stat().st_size > 0
    with registry.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = list(csv.DictReader(handle))
        if any(item.get("model_id") == model_id or item.get("package_name") == args.label for item in existing):
            raise SystemExit(f"Model is already registered: {args.label} / {model_id}")
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not registry_has_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)
    (model_dir / "external_model_manifest.json").write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
