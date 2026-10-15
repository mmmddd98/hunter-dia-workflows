#!/usr/bin/env python3
"""Resolve the dataset declaration matching Step-2 inputs.

A directory named ``test`` is not a split declaration: files under it can be
training inputs.  A result is emitted only for one exact match of the complete
Step-2 input set to a dataset's declared ``train_files``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def input_stems(inputs: list[Path]) -> set[str]:
    return {path.stem for path in inputs}


def resolve_paths(inputs: list[Path], data: dict) -> bool:
    train = data.get("train_files")
    test = data.get("test_files")
    if not isinstance(train, list) or not train or not isinstance(test, list):
        return False
    # Cluster commands may use .raw while the dataset declaration lists bare
    # run IDs.  Match run IDs, never the file extension or old cluster root.
    return input_stems(inputs) == {Path(str(name)).stem for name in train}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--datasets-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", action="store_true")
    parser.add_argument("--source", action="store_true", help="Print matching dataset source")
    args = parser.parse_args()
    inputs = [Path(line.strip()).resolve() for line in args.inputs.read_text().splitlines() if line.strip()]
    if not inputs:
        raise SystemExit("No Step-2 mzML inputs supplied")

    matches: list[tuple[str, str]] = []
    for dataset_file in sorted(args.datasets_dir.glob("*.yaml")):
        try:
            document = yaml.safe_load(dataset_file.read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            raise SystemExit(f"Cannot read {dataset_file}: {error}")
        sources = document.get("sources", {})
        if not isinstance(sources, dict):
            continue
        for source_name, source in sources.items():
            if not isinstance(source, dict) or source_name == "base_dir":
                continue
            if resolve_paths(inputs, source.get("data", {})):
                dataset_id = document.get("meta", {}).get("dataset_id")
                if not isinstance(dataset_id, str) or not dataset_id:
                    raise SystemExit(f"Matched dataset lacks meta.dataset_id: {dataset_file}")
                matches.append((dataset_id, str(source_name)))
    if len(matches) != 1:
        found = ", ".join(dataset_id for dataset_id, _ in matches) or "none"
        raise SystemExit(
            "Could not uniquely resolve held-out mzMLs from dataset metadata "
            f"(matches: {found}). Record a Step-5 command or add a matching dataset YAML."
        )
    dataset_id, source = matches[0]
    if args.dataset_id:
        print(dataset_id)
    elif args.source:
        print(source)
    else:
        raise SystemExit("Choose --dataset-id or --source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
