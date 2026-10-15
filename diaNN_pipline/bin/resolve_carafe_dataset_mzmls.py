#!/usr/bin/env python3
"""Resolve dataset train/test acquisition paths from pipeline YAML metadata.

Carafe can consume the same vendor ``.raw`` inputs that DIA-NN used.  Prefer
an mzML conversion when it exists, but fall back to the original RAW file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def resolve_input(directory: Path, name: str) -> Path:
    """Return the available acquisition file, preferring mzML."""
    stem = Path(str(name)).stem
    candidates = (directory / f"{stem}.mzML", directory / f"{stem}.raw")
    for candidate in candidates:
        if candidate.is_file():
            # Keep the mounted shared path.  Resolving a RAW symlink can point
            # back to a host-private location and is unnecessary here.
            return candidate.absolute()
    # Keep the usual mzML path in the diagnostic when neither file exists.
    return candidates[0].absolute()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--datasets-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    args = parser.parse_args()

    matches = []
    for path in sorted(args.datasets_dir.glob("*.yaml")):
        document = yaml.safe_load(path.read_text()) or {}
        if document.get("meta", {}).get("dataset_id") != args.dataset_id:
            continue
        source = document.get("sources", {}).get(args.source)
        if not isinstance(source, dict):
            continue
        data = source.get("data", {})
        files = data.get(f"{args.split}_files")
        if not isinstance(files, list):
            continue
        base_dir = document.get("sources", {}).get("base_dir", "")
        split_dir = data.get("dirs", {}).get(args.split, args.split)
        directory = args.data_root / str(base_dir) / args.source / str(split_dir)
        matches.append([resolve_input(directory, str(name)) for name in files])

    if len(matches) != 1:
        raise SystemExit(f"Could not uniquely resolve dataset {args.dataset_id} source {args.source}")
    paths = matches[0]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Resolved dataset acquisition file is missing (tried .mzML, then .raw): " + ", ".join(missing))
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
