#!/usr/bin/env python3
"""Validate whether a Carafe output directory contains a complete prediction checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON validation record.")
    parser.add_argument("--ccs", action="store_true", help="Require CCS prediction files as well.")
    args = parser.parse_args()

    required_models = ("ms2_model.pt", "rt_model.pt", "model_evaluation_metrics.json")
    missing = [name for name in required_models if not (args.output_dir / name).is_file() or (args.output_dir / name).stat().st_size == 0]
    chunk_dir = args.output_dir / "prediction_chunks"
    # Old jobs stored the same checkpoint files directly in the output root.
    # Accept that layout for resume compatibility, while new runs use the
    # dedicated directory to keep the output root manageable.
    if not chunk_dir.is_dir():
        chunk_dir = args.output_dir
    peptide_forms: dict[int, Path] = {}
    pattern = re.compile(r"^peptide_forms_(\d+)\.parquet$")
    if chunk_dir.is_dir():
        for path in chunk_dir.iterdir():
            match = pattern.match(path.name)
            if match and path.is_file():
                peptide_forms[int(match.group(1))] = path

    errors: list[str] = []
    if missing:
        errors.append("Missing or empty model/checkpoint metadata: " + ", ".join(missing))
    if not peptide_forms:
        errors.append("No peptide_forms_*.parquet files were found.")

    expected_count = max(peptide_forms, default=0)
    if peptide_forms and sorted(peptide_forms) != list(range(1, expected_count + 1)):
        errors.append("Peptide-form chunk numbering is not contiguous from 1.")

    suffixes = ("_ms2_df.parquet", "_ms2_mz_df.parquet", "_ms2_pred.parquet", "_rt_pred.parquet")
    if args.ccs:
        suffixes += ("_ccs_pred.parquet",)
    for peptide_chunk in range(1, expected_count + 1):
        input_path = peptide_forms.get(peptide_chunk)
        if input_path is None or input_path.stat().st_size == 0:
            errors.append(f"Missing or empty peptide_forms_{peptide_chunk}.parquet")
        prediction_chunk = peptide_chunk - 1
        for suffix in suffixes:
            path = chunk_dir / f"{prediction_chunk}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing or empty {path.name}")
                if len(errors) >= 25:
                    errors.append("Further missing files omitted.")
                    break
        if len(errors) >= 25:
            break

    payload = {
        "valid": not errors,
        "output_dir": str(args.output_dir),
        "prediction_chunk_dir": str(chunk_dir),
        "peptide_form_chunks": expected_count,
        "prediction_chunks": expected_count,
        "ccs_required": args.ccs,
        "errors": errors,
    }
    if args.output:
        atomic_write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
