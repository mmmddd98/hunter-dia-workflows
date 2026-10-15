#!/usr/bin/env python3
"""Create publication-ready figures for the cross-organism RT scaling study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORGANISMS = ("celegans", "triplemix", "human")
SUBSET_METRICS = (
    ("mae", "All peptides"),
    ("heavy_dimethyl_mae", "Any Heavy-Dimethyl"),
    ("heavy_nterm_dimethyl_mae", "N-term Heavy-Dimethyl"),
)
COLORS = {"diann": "#0072B2", "carafe2": "#D55E00"}
DISPLAY = {"diann": "DIA-NN", "carafe2": "Carafe2"}
CONDITIONS = [
    "ce_010", "ce_025", "ce_050", "ce_075", "ce_100",
    "ce100_triple_025", "ce100_triple_050", "ce100_triple_100",
    "ce100_human_025", "ce100_human_050", "ce100_human_100",
    "ce100_all_025", "ce100_all_050", "ce100_all_100",
]
LABELS = {
    "ce_010": "CE 10%", "ce_025": "CE 25%", "ce_050": "CE 50%",
    "ce_075": "CE 75%", "ce_100": "CE 100%",
    "ce100_triple_025": "+TM 25%", "ce100_triple_050": "+TM 50%",
    "ce100_triple_100": "+TM 100%", "ce100_human_025": "+HU 25%",
    "ce100_human_050": "+HU 50%", "ce100_human_100": "+HU 100%",
    "ce100_all_025": "+Both 25%", "ce100_all_050": "+Both 50%",
    "ce100_all_100": "+Both 100%",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diann-dir", type=Path, required=True)
    parser.add_argument("--carafe2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def collect(root: Path, family: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(root.glob("runs/seed_*/*")):
        metrics_path = run_dir / "organism_test_metrics.json"
        if not metrics_path.is_file() or run_dir.name not in CONDITIONS:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for organism in ORGANISMS:
            item = metrics.get(organism, {})
            if "mae" not in item:
                continue
            rows.append({
                "family": family,
                "seed": int(run_dir.parent.name.removeprefix("seed_")),
                "condition": run_dir.name,
                "condition_order": CONDITIONS.index(run_dir.name),
                "test_organism": organism,
                **{key: item.get(key, np.nan) for key in (
                    "mae", "rmse", "r2", "pearson", "spearman", "rows", "unique_sequences",
                    "heavy_dimethyl_mae", "heavy_dimethyl_rows",
                    "heavy_nterm_dimethyl_mae", "heavy_nterm_dimethyl_rows",
                )},
            })
    return pd.DataFrame(rows)


def save_all(fig: plt.Figure, base: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), dpi=450 if suffix == ".png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = pd.concat([
        collect(args.diann_dir, "diann"), collect(args.carafe2_dir, "carafe2")
    ], ignore_index=True)
    if data.empty:
        raise SystemExit("No completed organism_test_metrics.json files found")
    data.to_csv(output / "organism_scaling_runs.csv", index=False)
    summary = data.groupby(
        ["family", "condition", "condition_order", "test_organism"], as_index=False
    ).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"),
        pearson_mean=("pearson", "mean"), repeats=("seed", "nunique"),
    )
    summary["mae_ci95"] = 1.96 * summary["mae_std"].fillna(0) / np.sqrt(summary["repeats"])
    for metric, _ in SUBSET_METRICS[1:]:
        grouped = data.groupby(
            ["family", "condition", "condition_order", "test_organism"], as_index=False
        ).agg(**{
            f"{metric}_mean": (metric, "mean"),
            f"{metric}_std": (metric, "std"),
            f"{metric}_repeats": (metric, "count"),
        })
        grouped[f"{metric}_ci95"] = (
            1.96 * grouped[f"{metric}_std"].fillna(0)
            / np.sqrt(grouped[f"{metric}_repeats"].clip(lower=1))
        )
        summary = summary.merge(
            grouped,
            on=["family", "condition", "condition_order", "test_organism"], how="left",
        )
    summary.sort_values(["test_organism", "family", "condition_order"]).to_csv(
        output / "organism_scaling_summary.csv", index=False
    )

    # Main figure: direct analogue of the old learning curve, one panel per test organism.
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.35), sharey=False)
    for label, organism, axis in zip("ABC", ORGANISMS, axes):
        subset = summary[summary.test_organism == organism]
        for family, group in subset.groupby("family"):
            group = group.sort_values("condition_order")
            axis.errorbar(group.condition_order, group.mae_mean, yerr=group.mae_ci95,
                          color=COLORS[family], marker="o", linewidth=2, markersize=4.5,
                          capsize=3, label=DISPLAY[family])
        axis.axvline(4.5, color="#999999", linestyle="--", linewidth=.8)
        axis.set_xticks(range(len(CONDITIONS)), [LABELS[item] for item in CONDITIONS], rotation=58, ha="right")
        axis.set(title=f"{label}  {organism.capitalize()} test", xlabel="Nested training condition",
                 ylabel="MAE (normalized RT)")
        axis.grid(axis="y", linestyle=":", alpha=.4)
        axis.legend(frameon=False)
    fig.suptitle("Cross-organism RT data scaling on fixed sequence-disjoint test sets", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_all(fig, output / "fig1_organism_scaling_multitest_mae")

    # Repeated-split evidence: preserve paired seed trajectories, as in the old study.
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), sharex=True)
    for row, family in enumerate(("diann", "carafe2")):
        for col, organism in enumerate(ORGANISMS):
            axis = axes[row, col]
            subset = data[(data.family == family) & (data.test_organism == organism)]
            for _, group in subset.groupby("seed"):
                group = group.sort_values("condition_order")
                axis.plot(group.condition_order, group.mae, color="#9AA0A6", alpha=.45, linewidth=.8)
            mean = summary[(summary.family == family) & (summary.test_organism == organism)].sort_values("condition_order")
            axis.errorbar(mean.condition_order, mean.mae_mean, yerr=mean.mae_ci95,
                          color=COLORS[family], marker="o", linewidth=2.2, capsize=3)
            axis.axvline(4.5, color="#999999", linestyle="--", linewidth=.8)
            axis.set_title(f"{DISPLAY[family]} | {organism.capitalize()}")
            axis.set_ylabel("MAE (normalized RT)")
            axis.grid(axis="y", linestyle=":", alpha=.4)
            axis.set_xticks(range(len(CONDITIONS)), [LABELS[item] for item in CONDITIONS], rotation=58, ha="right")
    fig.suptitle("Paired seed trajectories across every training-data extension", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_all(fig, output / "fig2_organism_scaling_seed_trajectories")

    # Generalization heatmap: improvement relative to CE-only 100% for every model/test pair.
    additions = CONDITIONS[5:]
    heatmap_rows = []
    for family in ("diann", "carafe2"):
        for organism in ORGANISMS:
            base = data[(data.family == family) & (data.test_organism == organism) & (data.condition == "ce_100")].set_index("seed").mae
            for condition in additions:
                candidate = data[(data.family == family) & (data.test_organism == organism) & (data.condition == condition)].set_index("seed").mae
                paired = base.index.intersection(candidate.index)
                gain = (base.loc[paired] - candidate.loc[paired]) if len(paired) else pd.Series(dtype=float)
                heatmap_rows.append({"family": family, "test_organism": organism, "condition": condition,
                                     "mean_mae_gain": gain.mean(), "paired_repeats": len(gain)})
    gains = pd.DataFrame(heatmap_rows)
    gains.to_csv(output / "paired_mae_gain_vs_ce100.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.7), constrained_layout=True)
    finite = gains.mean_mae_gain.dropna().abs()
    limit = max(float(finite.max()) if len(finite) else .01, .005)
    image = None
    for axis, family in zip(axes, ("diann", "carafe2")):
        matrix = gains[gains.family == family].pivot(index="test_organism", columns="condition", values="mean_mae_gain").reindex(index=ORGANISMS, columns=additions)
        image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(additions)), [LABELS[item] for item in additions], rotation=55, ha="right")
        axis.set_yticks(range(len(ORGANISMS)), [item.capitalize() for item in ORGANISMS])
        axis.set_title(f"{DISPLAY[family]}: MAE gain vs CE 100%")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                axis.text(j, i, "NA" if pd.isna(value) else f"{value:+.3f}", ha="center", va="center", fontsize=7)
    if image is not None:
        fig.colorbar(image, ax=axes, label="Positive means added data improves MAE", shrink=.85)
    save_all(fig, output / "fig3_cross_organism_generalization_gain")

    # Heavy-label view over the complete experiment: both overlapping subsets,
    # every organism, both model families, and all nested training conditions.
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.7), sharex=True)
    for row, (metric, subset_label) in enumerate(SUBSET_METRICS[1:]):
        for col, organism in enumerate(ORGANISMS):
            axis = axes[row, col]
            subset = summary[summary.test_organism == organism]
            for family, group in subset.groupby("family"):
                group = group.sort_values("condition_order")
                axis.errorbar(
                    group.condition_order, group[f"{metric}_mean"],
                    yerr=group[f"{metric}_ci95"], color=COLORS[family], marker="o",
                    linewidth=2, markersize=4.5, capsize=3, label=DISPLAY[family],
                )
            axis.axvline(4.5, color="#999999", linestyle="--", linewidth=.8)
            axis.set_xticks(range(len(CONDITIONS)), [LABELS[item] for item in CONDITIONS],
                            rotation=58, ha="right")
            axis.set_title(f"{subset_label} | {organism.capitalize()}")
            axis.set_ylabel("MAE (normalized RT)")
            axis.grid(axis="y", linestyle=":", alpha=.4)
            if row == 0 and col == 0:
                axis.legend(frameon=False)
    fig.suptitle("Heavy-Dimethyl RT performance across the complete organism-scaling experiment",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_all(fig, output / "fig4_heavy_dimethyl_scaling_multitest")

    # Final endpoint summary at the largest mixed-organism training condition.
    endpoint = data[data.condition == "ce100_all_100"]
    endpoint_rows = []
    for family in ("diann", "carafe2"):
        for organism in ORGANISMS:
            group = endpoint[(endpoint.family == family) & (endpoint.test_organism == organism)]
            for metric, subset_label in SUBSET_METRICS:
                values = group[metric].dropna()
                endpoint_rows.append({
                    "family": family, "test_organism": organism, "subset": subset_label,
                    "mae_mean": values.mean(), "mae_std": values.std(), "repeats": len(values),
                    "mae_ci95": 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0,
                })
    endpoint_summary = pd.DataFrame(endpoint_rows)
    endpoint_summary.to_csv(output / "final_endpoint_heavy_dimethyl_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=False)
    width = .36
    for axis, organism in zip(axes, ORGANISMS):
        organism_data = endpoint_summary[endpoint_summary.test_organism == organism]
        positions = np.arange(len(SUBSET_METRICS))
        for offset, family in zip((-.5, .5), ("diann", "carafe2")):
            group = organism_data[organism_data.family == family].set_index("subset")
            order = [label for _, label in SUBSET_METRICS]
            axis.bar(positions + offset * width, group.loc[order, "mae_mean"], width,
                     yerr=group.loc[order, "mae_ci95"], capsize=3,
                     color=COLORS[family], label=DISPLAY[family])
        axis.set_xticks(positions, [label.replace(" ", "\n", 1) for _, label in SUBSET_METRICS])
        axis.set_title(f"{organism.capitalize()} test")
        axis.set_ylabel("MAE (normalized RT)")
        axis.grid(axis="y", linestyle=":", alpha=.4)
        axis.legend(frameon=False)
    fig.suptitle("Final ce100_all_100 performance across global and Heavy-Dimethyl subsets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_all(fig, output / "fig5_final_endpoint_heavy_dimethyl_comparison")
    print(f"Wrote publication figures and CSV summaries to {output}")


if __name__ == "__main__":
    main()
