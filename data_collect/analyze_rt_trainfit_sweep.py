#!/usr/bin/env python3
"""Collect, rank and plot a factorial DIA-NN or Carafe2 RT sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from plot_rt_sweep_publication import generate_publication_figures


def metric_value(record: dict, name: str) -> float:
    value = record.get(name, {}).get("mae")
    return float(value) if value is not None else float("nan")


def model_label_for_sweep(sweep_dir: Path, fallback: str = "RT model") -> str:
    """Infer a display name without relying on the entry-point filename."""
    # The directory/study name is authoritative: both implementations may use
    # DIA-NN-formatted inputs and shared evaluation scripts.
    name = sweep_dir.name.lower()
    if "carafe" in name:
        return "Carafe2"
    if "diann" in name or "dia-nn" in name:
        return "DIA-NN"
    evidence = name
    manifest = sweep_dir / "sweep_manifest.json"
    if manifest.exists():
        evidence += " " + manifest.read_text(encoding="utf-8").lower()
    if "carafe" in evidence:
        return "Carafe2"
    if "diann" in evidence or "dia-nn" in evidence:
        return "DIA-NN"
    return fallback


def learning_rates_from_manifest(sweep_dir: Path) -> dict[str, float]:
    """Return absolute primary LR by run id when a manifest records it."""
    manifest = sweep_dir / "sweep_manifest.json"
    if not manifest.exists():
        return {}
    jobs = json.loads(manifest.read_text(encoding="utf-8")).get("jobs", [])
    rates: dict[str, float] = {}
    for job in jobs:
        for command in job.get("commands", []):
            for flag in ("--rt_lr", "--learning-rate"):
                if flag in command:
                    try:
                        rates[job["id"]] = float(command[command.index(flag) + 1])
                    except (IndexError, ValueError):
                        pass
                    break
            if job.get("id") in rates:
                break
    return rates


def collect_runs(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    learning_rates = learning_rates_from_manifest(sweep_dir)
    metric_paths = list(sweep_dir.glob("runs/*/*/seed_*/validation_metrics.json"))
    metric_paths += list(sweep_dir.glob("runs/*/*/*/seed_*/validation_metrics.json"))
    for path in metric_paths:
        run_parts = path.relative_to(sweep_dir / "runs").parts
        if len(run_parts) == 4:
            mode, policy, seed, _ = run_parts
            lr_level = "base"
        elif len(run_parts) == 5:
            mode, policy, lr_level, seed, _ = run_parts
        else:
            raise ValueError(f"Unexpected run metrics path: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        combined = metrics["all_source_observations"]
        row = {
            "mode": mode,
            "policy": policy,
            "lr_level": lr_level,
            "seed": int(seed.removeprefix("seed_")),
            "absolute_learning_rate": learning_rates.get("/".join(run_parts[:-1]), float("nan")),
            "source_balanced_raw_mae": metrics["source_balanced_raw_mae"],
            "max_source_raw_mae": max(metric_value(value, "raw_minutes") for value in metrics["sources"].values()),
            "organism_macro_raw_mae": sum(metric_value(value, "raw_minutes") for value in metrics["organisms"].values()) / len(metrics["organisms"]),
            "aggregate_normalized_mae": metric_value(metrics["aggregate"], "normalized"),
            "all_observations_raw_mae": metric_value(combined, "raw_minutes"),
            "semi_tryptic_raw_mae": metric_value(combined, "raw_digestion_semi_tryptic"),
            "heavy_nterm_raw_mae": metric_value(combined, "raw_heavy_heavy_nterm"),
            "heavy_lysine_raw_mae": metric_value(combined, "raw_heavy_heavy_lysine"),
            "heavy_both_raw_mae": metric_value(combined, "raw_heavy_heavy_both"),
        }
        row.update({f"source_{name}_raw_mae": metric_value(value, "raw_minutes") for name, value in metrics["sources"].items()})
        row.update({f"organism_{name}_raw_mae": metric_value(value, "raw_minutes") for name, value in metrics["organisms"].items()})
        rows.append(row)
    if not rows:
        raise SystemExit("No validation_metrics.json files found")
    return pd.DataFrame(rows).sort_values(["mode", "policy", "lr_level", "seed"])


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in runs if column not in {"mode", "policy", "lr_level", "seed"}]
    grouping = ["mode", "policy", "lr_level"]
    aggregations = {f"{column}_mean": (column, "mean") for column in numeric}
    aggregations["seeds"] = ("seed", "nunique")
    aggregations["source_balanced_raw_mae_std"] = ("source_balanced_raw_mae", "std")
    summary = runs.groupby(grouping, as_index=False).agg(**aggregations)
    summary["rank"] = summary["source_balanced_raw_mae_mean"].rank(method="first").astype(int)
    return summary.sort_values("rank")


def terminal_progress(done: int, total: int, label: str) -> None:
    """Render one compact progress line for the figure-export phase."""
    width = 24
    filled = round(width * done / total) if total else width
    bar = "█" * filled + "·" * (width - filled)
    line = f"  Figures [{bar}] {done:>2}/{total:<2}  {label:<34}"
    if sys.stdout.isatty():
        end = "\n" if done == total else "\r"
        print(line, end=end, flush=True)
    else:
        print(line, flush=True)


def main(default_model_label: str = "RT model") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", help="Display label; inferred from the sweep when omitted")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    model_label = args.model_label or model_label_for_sweep(args.sweep_dir, default_model_label)
    print(f"\n{model_label} sweep analysis")
    print(f"  Input:  {args.sweep_dir}")
    runs = collect_runs(args.sweep_dir)
    summary = summarize_runs(runs)
    print(f"  Found:  {len(runs)} completed seed runs across {len(summary)} configurations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "validation_runs.csv", index=False)
    summary.to_csv(args.output_dir / "validation_summary.csv", index=False)
    selection = (
        "# Validation-only ranking\n\n"
        "Primary ranking: mean source-balanced MAE in observed RT minutes across seeds. "
        "Guardrails include worst source, organism macro-average, semi-tryptic and all three exact Heavy-Dimethyl strata. "
        "The final test partition is not read by this script.\n\n"
        + summary.to_markdown(index=False)
        + "\n"
    )
    (args.output_dir / "selection.md").write_text(selection, encoding="utf-8")
    if not args.skip_plots:
        print("  Exporting publication figures")
        generate_publication_figures(
            runs, args.output_dir / "figures",
            model_label, progress=terminal_progress,
        )
    best = summary.iloc[0]
    print("\nCompleted")
    print(f"  Best validation configuration: {best['mode']} / {best['policy']} / {best['lr_level']} "
          f"({best['source_balanced_raw_mae_mean']:.3f} min)")
    print(f"  Tables and ranking: {args.output_dir}")
    if not args.skip_plots:
        print(f"  Figures and interpretation guides: {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
