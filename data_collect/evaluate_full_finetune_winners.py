#!/usr/bin/env python3
"""Evaluate validation-selected multi-seed winners on the untouched final test."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DIANN_ROOT = Path(os.environ.get("DIANN_FINETUNE_ROOT", WORKSPACE_ROOT.parent / "diann-fintune")).expanduser()
CARAFE_ROOT = Path(os.environ.get("CARAFE2_ROOT", WORKSPACE_ROOT / "carafe2-semi")).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable))
    parser.add_argument("--carafe2-python", default=os.environ.get("CARAFE2_PYTHON", sys.executable))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = pd.read_csv(args.analysis_dir / "sweep_runs.csv")
    summary = pd.read_csv(args.analysis_dir / "sweep_configuration_summary.csv")
    max_seeds = summary.groupby("model").seeds.transform("max")
    eligible = summary[summary.seeds.eq(max_seeds) & summary.seeds.gt(1)]
    if eligible.empty:
        raise SystemExit("No multi-seed finalists found; run confirmation first")
    test_dir = args.prepared_dir / "test"
    winners = {}
    for model in ("DIA-NN", "Carafe2"):
        candidates = eligible[eligible.model.eq(model)].sort_values("composite_mae_mean")
        if candidates.empty:
            raise SystemExit(f"No confirmed finalist for {model}")
        winner = candidates.iloc[0]
        selected_runs = runs[(runs.model.eq(model)) & (runs.config_id.eq(winner.config_id))]
        evaluated = []
        for run in selected_runs.itertuples():
            run_dir = Path(run.run_dir)
            validation_path = run_dir / "validation_metrics.json"
            final_path = run_dir / "final_test_metrics.json"
            if final_path.is_file():
                evaluated.append(str(run_dir))
                continue
            if not validation_path.is_file():
                shutil.copy2(run_dir / "organism_test_metrics.json", validation_path)
            if model == "DIA-NN":
                command = [
                    args.diann_python, str(DIANN_ROOT / "evaluate_organism_tests.py"),
                    "--run-dir", str(run_dir), "--tests-dir", str(test_dir), "--device", "auto",
                    "--rt-alignment", str(test_dir / "rt_alignment.json"),
                ]
                cwd = DIANN_ROOT
            else:
                command = [
                    args.carafe2_python, str(CARAFE_ROOT / "evaluate_organism_tests.py"),
                    "--run-dir", str(run_dir), "--tests-dir", str(test_dir), "--device", "gpu",
                    "--python", args.carafe2_python,
                    "--rt-alignment", str(test_dir / "rt_alignment.json"),
                ]
                cwd = CARAFE_ROOT
            subprocess.run(command, cwd=cwd, check=True)
            shutil.copy2(run_dir / "organism_test_metrics.json", final_path)
            shutil.copy2(validation_path, run_dir / "organism_test_metrics.json")
            evaluated.append(str(run_dir))
        winners[model] = {
            "config_id": winner.config_id,
            "validation_composite_mae": float(winner.composite_mae_mean),
            "seeds": int(winner.seeds),
            "run_dirs": evaluated,
        }
    (args.analysis_dir / "winner_manifest.json").write_text(
        json.dumps(winners, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(winners, indent=2))


if __name__ == "__main__":
    main()
