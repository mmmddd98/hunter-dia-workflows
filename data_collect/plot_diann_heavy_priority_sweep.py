#!/usr/bin/env python3
"""Create publication-style validation plots and a Markdown result summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = {
    "C0 · kein Oversampling": "#4477AA",
    "H24 · Oversampling": "#66CCEE",
    "H32 · Oversampling": "#228833",
    "H40 · Oversampling": "#CC6677",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def setup() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_tradeoff(analysis: Path, plots: Path) -> None:
    data = pd.read_csv(analysis / "report_tradeoff.csv", dtype={"LR": str})
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for profile, part in data.groupby("Profil", sort=False):
        ax.scatter(
            part["Overall MAE"],
            part["Heavy macro MAE"],
            s=48,
            alpha=0.74,
            color=COLORS[profile],
            edgecolor="white",
            linewidth=0.55,
            label=profile,
        )
    highlights = [
        ("C0 · kein Oversampling", "1.5e-4", "50/50", "bester absoluter\nKompromiss"),
        ("H40 · Oversampling", "1.0e-4", "100/0", "beste Heavy-beide-MAE"),
    ]
    for profile, lr, checkpoint, label in highlights:
        point = data[
            data["Profil"].eq(profile)
            & data["LR"].eq(lr)
            & data["Checkpoint"].eq(checkpoint)
        ].iloc[0]
        ax.scatter(
            point["Overall MAE"], point["Heavy macro MAE"], marker="*", s=230,
            color=COLORS[profile], edgecolor="#222222", linewidth=0.8, zorder=5,
        )
        offset = (-68, -33) if profile.startswith("C0") else (12, 8)
        ax.annotate(
            label,
            (point["Overall MAE"], point["Heavy macro MAE"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.8},
        )
    ax.set(
        title="Gesamtfehler versus Heavy-Macro-Fehler",
        xlabel="Source-balanced Overall-MAE (min)",
        ylabel="Heavy-Macro-MAE (min)",
    )
    ax.text(
        0.01, 0.99, "links unten ist besser",
        transform=ax.transAxes, ha="left", va="top", color="#555555", fontsize=9,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, plots / "01_tradeoff_overall_vs_heavy_macro.png")


def plot_strata(analysis: Path, plots: Path) -> None:
    data = pd.read_csv(analysis / "report_key_strata.csv")
    order = [
        "Overall", "Kein Heavy-Dimethyl", "Heavy macro", "Heavy N-term",
        "Heavy Lys", "Heavy N-term + Lys",
    ]
    names = data["Modell"].drop_duplicates().tolist()
    colors = ["#4477AA", "#CC6677"]
    pivot = data.pivot(index="Stratum", columns="Modell", values="MAE Minuten").loc[order]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(order))
    for i, stratum in enumerate(order):
        ax.plot(
            [pivot.loc[stratum, names[0]], pivot.loc[stratum, names[1]]],
            [i, i], color="#B8BEC8", lw=2.2, zorder=1,
        )
    for name, color, marker in zip(names, colors, ("o", "D")):
        ax.scatter(pivot[name], y, s=70, color=color, marker=marker, label=name, zorder=3)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set(
        title="Der Heavy-beide-Gewinn ist stratum-spezifisch",
        xlabel="Source-balanced MAE (min; niedriger ist besser)",
        ylabel="",
    )
    ax.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.58, 1.02),
        ncol=2, fontsize=8.5, handletextpad=0.5, columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, plots / "02_exact_strata_key_models.png")


def plot_oversampling_deltas(analysis: Path, plots: Path) -> None:
    data = pd.read_csv(analysis / "report_oversampling_deltas.csv")
    order = [
        "Overall", "Kein Heavy-Dimethyl", "Heavy macro", "Heavy N-term",
        "Heavy Lys", "Heavy N-term + Lys",
    ]
    matrix = data.pivot(index="Vergleich", columns="Stratum", values="Delta Sekunden")[order]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    limit = float(np.ceil(np.abs(matrix.to_numpy()).max()))
    sns.heatmap(
        matrix,
        annot=False,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "MAE-Delta (s)"},
        ax=ax,
    )
    for row_idx in range(matrix.shape[0]):
        for column_idx in range(matrix.shape[1]):
            ax.text(
                column_idx + 0.5,
                row_idx + 0.5,
                f"{matrix.iloc[row_idx, column_idx]:+.2f}",
                ha="center",
                va="center",
                color="#202020",
                fontsize=9.5,
            )
    ax.set(
        title="Seed-gematchter Oversampling-Effekt gegenüber C0 ohne Oversampling",
        xlabel="",
        ylabel="",
    )
    ax.tick_params(axis="x", rotation=22)
    ax.tick_params(axis="y", rotation=0)
    fig.text(
        0.16, 0.02, "Blau / negativ = Verbesserung · Rot / positiv = Verschlechterung",
        fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, plots / "03_seed_matched_oversampling_effects.png")


def plot_seed_stability(analysis: Path, plots: Path) -> None:
    data = pd.read_csv(analysis / "report_seed_stability.csv")
    names = data["Modell"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(7.7, 4.7))
    for name, color, marker in zip(names, ("#4477AA", "#CC6677"), ("o", "D")):
        part = data[data["Modell"].eq(name)].sort_values("Seed")
        ax.plot(part["Seed"].astype(str), part["Heavy-both MAE"], color=color, marker=marker, lw=2, ms=7, label=name)
    ax.set(
        title="Heavy-beide-Verbesserung ist in allen drei Seeds sichtbar",
        xlabel="Seed",
        ylabel="Heavy N-term + Lys MAE (min)",
    )
    ax.legend(frameon=False, fontsize=8.8)
    fig.tight_layout()
    save(fig, plots / "04_heavy_both_seed_stability.png")


def plot_checkpoint_effect(analysis: Path, plots: Path) -> None:
    data = pd.read_csv(analysis / "report_checkpoint_effect.csv")
    order = ["100/0", "75/25", "50/50", "25/75", "0/100"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    markers = {"Overall": "o", "Heavy macro": "D"}
    colors = {"Overall": "#4477AA", "Heavy macro": "#EE7733"}
    y = np.arange(len(order))
    for metric in ("Overall", "Heavy macro"):
        part = data[data["Metrik"].eq(metric)].set_index("Checkpoint").loc[order]
        offset = -0.08 if metric == "Overall" else 0.08
        ax.scatter(
            part["Delta Sekunden"], y + offset, s=65, marker=markers[metric],
            color=colors[metric], label=metric, zorder=3,
        )
    ax.axvline(0, color="#666666", lw=1, ls="--")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set(
        title="Checkpoint-Priorität hat im Mittel nur einen kleinen Effekt",
        xlabel="Mittleres MAE-Delta zum 100/0-Checkpoint (s)",
        ylabel="Overall/Heavy-Priorität",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, plots / "05_checkpoint_priority_effect.png")


def write_markdown(analysis: Path) -> None:
    body = """# DIA-NN heavy-priority validation plots

The figures below use the CSV tables in this analysis directory. Values and
configuration labels come from those tables; this index does not assert a
winning configuration or prescribe a follow-up experiment.

| Figure | Source table |
| --- | --- |
| [Overall versus heavy-macro MAE](plots/01_tradeoff_overall_vs_heavy_macro.png) | `report_tradeoff.csv` |
| [Modification-stratum comparison](plots/02_exact_strata_key_models.png) | `report_key_strata.csv` |
| [Seed-matched oversampling differences](plots/03_seed_matched_oversampling_effects.png) | `report_oversampling_deltas.csv` |
| [Heavy-both errors by seed](plots/04_heavy_both_seed_stability.png) | `report_seed_stability.csv` |
| [Checkpoint-selection differences](plots/05_checkpoint_priority_effect.png) | `report_checkpoint_effect.csv` |

MAE values are source-balanced validation errors in observed RT minutes;
difference plots use seconds. Lower errors are better. Validation comparisons
do not establish performance on an independent test set.
"""
    (analysis / "SWEEP_RESULTS.md").write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    plots = args.analysis_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    setup()
    plot_tradeoff(args.analysis_dir, plots)
    plot_strata(args.analysis_dir, plots)
    plot_oversampling_deltas(args.analysis_dir, plots)
    plot_seed_stability(args.analysis_dir, plots)
    plot_checkpoint_effect(args.analysis_dir, plots)
    write_markdown(args.analysis_dir)
    print(f"Wrote five plots and {args.analysis_dir / 'SWEEP_RESULTS.md'}")


if __name__ == "__main__":
    main()
