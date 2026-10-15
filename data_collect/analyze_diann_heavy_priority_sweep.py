#!/usr/bin/env python3
"""Summarize the five validation-selected checkpoints from the priority sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SELECTIONS = ("100_0", "75_25", "50_50", "25_75", "0_100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mae(record: dict | None) -> float | None:
    if not record or record.get("rows", 0) <= 0:
        return None
    value = record.get("mae")
    return float(value) if value is not None else None


def source_balanced(sources: dict, key: str) -> float | None:
    values = [mae(source.get(key)) for source in sources.values()]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def parse_priority(selection: str) -> tuple[float, float]:
    overall, heavy = selection.split("_", 1)
    return int(overall) / 100.0, int(heavy) / 100.0


def run_rows(sweep_dir: Path, job: dict) -> list[dict]:
    run = Path(job["commands"][0][job["commands"][0].index("--output-dir") + 1])
    provenance_path = run / "checkpoint_selection.json"
    if not provenance_path.is_file():
        return []
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    rows = []
    for selection in SELECTIONS:
        metric_path = run / "validation_metrics" / f"checkpoint_{selection}.json"
        if not metric_path.is_file():
            continue
        result = json.loads(metric_path.read_text(encoding="utf-8"))
        sources = result.get("sources", {})
        values = {
            "overall_mae": result.get("source_balanced_raw_mae"),
            "heavy_nterm_mae": source_balanced(sources, "raw_heavy_heavy_nterm"),
            "heavy_lysine_mae": source_balanced(sources, "raw_heavy_heavy_lysine"),
            "heavy_both_mae": source_balanced(sources, "raw_heavy_heavy_both"),
            "nonheavy_mae": source_balanced(sources, "raw_heavy_normal"),
            "semitryptic_mae": source_balanced(sources, "raw_digestion_semi_tryptic"),
        }
        heavy_values = [values[name] for name in ("heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae")]
        values["heavy_macro_mae"] = sum(heavy_values) / 3.0 if all(value is not None for value in heavy_values) else None
        overall_weight, heavy_weight = parse_priority(selection)
        values["priority_score_minutes"] = (
            overall_weight * values["overall_mae"] + heavy_weight * values["heavy_macro_mae"]
            if values["overall_mae"] is not None and values["heavy_macro_mae"] is not None else None
        )
        selected = provenance.get("selections", {}).get(selection, {})
        rows.append({
            **job["metadata"], "job_id": job["id"], "run_dir": str(run),
            "checkpoint_selection": selection, "overall_weight": overall_weight,
            "heavy_macro_weight": heavy_weight, "checkpoint_epoch": selected.get("best_epoch"),
            "checkpoint_score_rt_norm": selected.get("score"),
            "checkpoint_path": result.get("checkpoint_path"), **values,
        })
    return rows


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.sweep_dir / "sweep_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("validation_only") or manifest.get("test_partition") != "closed":
        raise ValueError("This analyser only accepts a validation-only, test-closed priority manifest")
    rows = [row for job in manifest["jobs"] for row in run_rows(args.sweep_dir, job)]
    if not rows:
        raise FileNotFoundError("No completed checkpoint-grid validation results found")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(rows)
    candidates.to_csv(output / "checkpoint_candidates.csv", index=False)

    metrics = [
        "priority_score_minutes", "overall_mae", "heavy_macro_mae", "heavy_nterm_mae",
        "heavy_lysine_mae", "heavy_both_mae", "nonheavy_mae", "semitryptic_mae",
    ]
    grouping = ["target", "profile", "lr", "oversampling", "checkpoint_selection", "overall_weight", "heavy_macro_weight"]
    summary = candidates.groupby(grouping, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary = summary.reset_index().sort_values(["checkpoint_selection", "priority_score_minutes_mean"])
    summary.to_csv(output / "configuration_summary.csv", index=False)

    control = candidates[candidates.profile.eq("C0")].loc[:, ["lr", "seed", "checkpoint_selection", *metrics]]
    control = control.rename(columns={metric: f"control_{metric}" for metric in metrics})
    delta = candidates.merge(control, on=["lr", "seed", "checkpoint_selection"], how="left")
    for metric in metrics:
        delta[f"delta_{metric}"] = delta[metric] - delta[f"control_{metric}"]
    delta.to_csv(output / "seed_matched_control_deltas.csv", index=False)

    readme = """# Priority-checkpoint validation analysis

Every row is one checkpoint selected from one run using only the fixed
validation partition. `priority_score_minutes` applies that row's stated
Overall/Heavy-Macro priority to source-balanced observed-minute MAE. It is a
readable validation ranking, not a final-test result. Inspect all three exact
Heavy MAEs before selecting a candidate: a good Heavy-Macro average must not
hide a worse Heavy N-term, Heavy Lys, or Heavy-both result.

The test partition remains closed. Freeze one configuration and checkpoint
priority from these files before creating or reading a fresh confirmation
hold-out.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"candidates": len(candidates), "configuration_rows": len(summary), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
