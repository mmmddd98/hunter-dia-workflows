#!/usr/bin/env python3
"""Evaluate registered external Carafe2 RT models on CPU, without training."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "ma/data_collect/evaluate_carafe2_trainfit.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-list", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--target-space", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", action="append", help="Optional package_name; repeat to select individual models")
    parser.add_argument("--allow-test-partition", action="store_true", help="Required acknowledgement for any path containing /test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partition = args.partition_dir.resolve()
    alignment = args.alignment.resolve()
    if "test" in partition.parts and not args.allow_test_partition:
        raise SystemExit("Refusing a test partition without --allow-test-partition")
    if not (partition / "rt_observations.tsv").is_file() or not alignment.is_file():
        raise SystemExit("Partition must contain rt_observations.tsv and --alignment must exist")
    with args.models_list.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    requested = set(args.model or [])
    selected = [
        row for row in rows
        if row.get("target_space") == args.target_space
        and row.get("status") == "ready_for_cpu_validation"
        and (not requested or row.get("package_name") in requested)
    ]
    if requested and {row["package_name"] for row in selected} != requested:
        missing = sorted(requested - {row["package_name"] for row in selected})
        raise SystemExit(f"Requested ready models not found for target {args.target_space}: {missing}")
    if not selected:
        raise SystemExit("No ready registered models match this target")
    registry_root = args.models_list.resolve().parent
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing evaluation: {output}")
    output.mkdir(parents=True)
    summary = []
    for row in selected:
        run_dir = registry_root / "models" / row["relative_path"]
        if not (run_dir / "rt_model.pt").is_file():
            raise FileNotFoundError(f"Registered checkpoint missing: {run_dir / 'rt_model.pt'}")
        model_output = output / row["package_name"]
        metrics_path = model_output / "metrics.json"
        command = [
            args.python, str(EVALUATOR), "--run-dir", str(run_dir),
            "--partition-dir", str(partition), "--alignment", str(alignment),
            "--output", str(metrics_path), "--prediction-dir", str(model_output / "predictions"),
            "--device", "cpu", "--python", args.python,
        ]
        subprocess.run(command, check=True)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary.append({
            **row,
            "model_dir": str(run_dir),
            "source_balanced_raw_mae": metrics.get("source_balanced_raw_mae"),
            "normalized_aggregate_mae": metrics.get("aggregate", {}).get("normalized", {}).get("mae"),
        })
    fields = list(summary[0])
    with (output / "model_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    manifest = {
        "device": "cpu", "target_space": args.target_space, "partition": str(partition),
        "alignment": str(alignment), "models": [row["package_name"] for row in selected],
        "test_partition_acknowledged": args.allow_test_partition,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(summary), "output": str(output), "device": "cpu"}, indent=2))


if __name__ == "__main__":
    main()
