#!/usr/bin/env python3
"""Validate a DIA-NN report before it is handed to Carafe.

The check is deliberately streaming: a production report can contain millions
of rows, so no full pandas table is materialised merely for preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def run_key(value: object) -> str:
    name = Path(str(value)).name
    for suffix in (".mzml", ".raw", ".d"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path | None, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--train-paths", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = args.report.resolve()
    expected_paths = [line.strip() for line in args.train_paths.read_text().splitlines() if line.strip()]
    expected = {run_key(item): Path(item).name for item in expected_paths}
    errors: list[str] = []
    result: dict = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "report": str(report),
        "expected_training_files": len(expected),
        "errors": errors,
    }

    if len(expected) != len(expected_paths):
        errors.append("Training mzML basenames are not unique after normalisation.")

    try:
        parquet = pq.ParquetFile(report)
        columns = parquet.schema_arrow.names
        result["report_rows"] = parquet.metadata.num_rows
        result["report_sha256"] = sha256(report)
        result["schema_columns"] = columns
    except Exception as exc:  # pragma: no cover - defensive reporting path
        errors.append(f"Report is not a readable Parquet file: {exc}")
        result["status"] = "failed"
        write_json(args.output, result)
        print("ERROR: " + errors[-1], file=sys.stderr)
        return 2

    canonical_scan = "MS2.Scan"
    scan_aliases = [column for column in columns if column.casefold() == canonical_scan.casefold()]
    if canonical_scan not in columns and scan_aliases:
        errors.append(
            f"DIA-NN scan column is named {scan_aliases[0]!r}, but Carafe requires {canonical_scan!r}. "
            "Re-prepare or refresh the Carafe job so its staged report is normalized."
        )
    elif canonical_scan not in columns:
        errors.append(
            "Required DIA-NN column 'MS2.Scan' is missing. Re-run Step 2 with --export-quant "
            "(enable_EXPORT_QUANT=1 in this pipeline) and prepare a fresh/updated Carafe job."
        )
    if "Run" not in columns:
        errors.append("Required DIA-NN column 'Run' is missing; cannot match report rows to training mzML files.")
    if result["report_rows"] <= 0:
        errors.append("Report contains no rows.")

    observed: set[str] = set()
    non_null_scans = 0
    if "Run" in columns and canonical_scan in columns:
        for batch in parquet.iter_batches(columns=["Run", canonical_scan], batch_size=100_000):
            run_values = batch.column(0).to_pylist()
            scan_values = batch.column(1).to_pylist()
            observed.update(run_key(value) for value in run_values if value is not None and str(value))
            non_null_scans += sum(value is not None for value in scan_values)
        result["observed_runs"] = sorted(observed)
        result["non_null_ms2_scans"] = non_null_scans
        if non_null_scans == 0:
            errors.append("Column 'MS2.Scan' exists but contains no usable scan values.")

        missing = sorted(expected[key] for key in expected.keys() - observed)
        unexpected = sorted(observed - expected.keys())
        result["missing_training_runs"] = missing
        result["unexpected_report_runs"] = unexpected
        if missing:
            errors.append("Training mzMLs absent from report Run column: " + ", ".join(missing))
        if unexpected:
            errors.append("Report contains runs not staged for Carafe: " + ", ".join(unexpected))

    result["status"] = "passed" if not errors else "failed"
    write_json(args.output, result)
    if errors:
        for error in errors:
            print("ERROR: " + error, file=sys.stderr)
        return 2
    print(
        f"Report validation passed: {result['report_rows']} rows, "
        f"{len(observed)} matching runs, {non_null_scans} non-null MS2.Scan values."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
