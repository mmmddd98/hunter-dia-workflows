#!/usr/bin/env python3
"""Select diverse full-data sweep finalists for multi-seed confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def select(summary: pd.DataFrame, model: str, count: int) -> list[str]:
    candidates = summary[summary.model.eq(model)].sort_values("composite_mae_mean")
    selected: list[str] = []

    def add(frame: pd.DataFrame, n: int = 1) -> None:
        for config_id in frame.config_id.head(n):
            if config_id not in selected:
                selected.append(config_id)

    add(candidates, 3)
    unweighted = candidates[
        np.isclose(candidates.nterm_weight.astype(float), 1.0)
        & np.isclose(candidates.lysine_weight.astype(float), 1.0)
        & np.isclose(candidates.both_weight.astype(float), 1.0)
    ]
    add(unweighted)
    add(candidates[candidates.strategy.eq("all")])
    add(candidates[~candidates.strategy.isin(("all", "discriminative"))])
    add(candidates, count)
    return selected[:count]


def main() -> None:
    args = parse_args()
    runs = pd.read_csv(args.analysis_dir / "sweep_runs.csv")
    summary = pd.read_csv(args.analysis_dir / "sweep_configuration_summary.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for model, filename in (("DIA-NN", "diann_finalists.json"), ("Carafe2", "carafe2_finalists.json")):
        selected_ids = select(summary, model, args.top)
        mapping = {}
        for rank, config_id in enumerate(selected_ids, start=1):
            run = runs[(runs.model.eq(model)) & (runs.config_id.eq(config_id))].iloc[0]
            command = json.loads((Path(run.run_dir) / "command.json").read_text(encoding="utf-8"))
            mapping[f"rank{rank:02d}_{config_id}"] = command.get("wandb_config", command.get("config", {}))
        path = args.output_dir / filename
        path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        outputs[model] = {"path": str(path), "configurations": list(mapping)}
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(outputs, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
