from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_qc_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    specs = [
        ("observed_rt", "RT distribution", "Observed RT"),
        ("q_value", "Q-value distribution", "Q-value"),
        ("charge", "Charge distribution", "Charge"),
    ]
    plot_df = df.copy()
    if "sequence" in plot_df:
        plot_df["peptide_length"] = plot_df["sequence"].fillna("").astype(str).str.len()
        specs.append(("peptide_length", "Peptide-length distribution", "Length"))
    for column, title, xlabel in specs:
        if column not in plot_df:
            continue
        values = pd.to_numeric(plot_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=40, color="#3B6EA8", edgecolor="white")
        ax.set(title=title, xlabel=xlabel, ylabel="Identifications")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        outputs.extend(save_figure(fig, output_dir / column))
        plt.close(fig)

    for group_column, name in (("raw_file_name", "rt_by_raw_file"), ("hunter_status", "rt_by_hunter_status")):
        if {"observed_rt", group_column}.issubset(plot_df):
            grouped = [
                pd.to_numeric(group["observed_rt"], errors="coerce").dropna()
                for _, group in plot_df.groupby(group_column)
            ]
            labels = [str(label) for label, _ in plot_df.groupby(group_column)]
            if grouped:
                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
                ax.boxplot(grouped, tick_labels=labels, showfliers=False)
                ax.set(title=f"RT distribution by {group_column}", ylabel="Observed RT")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                outputs.extend(save_figure(fig, output_dir / name))
                plt.close(fig)
    return outputs


def save_figure(fig, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    return paths
