#!/usr/bin/env python3
"""Create reusable publication figures from a factorial RT sweep summary.

The input contract is the run-level table written by
``analyze_rt_trainfit_sweep.py``. Figures are exported as SVG, PDF and
600-dpi PNG using a compact, colour-blind-safe journal style.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MODE_ORDER = ("anchor_irt", "chronologer_hi", "linear", "global_60min")
LR_ORDER = ("very_low", "low", "base", "high", "five_x", "very_high", "fifteen_x", "twenty_x", "thirty_x")
MODE_LABELS = {
    "anchor_irt": "Anchor iRT",
    "chronologer_hi": "Chronologer HI",
    "linear": "Linear",
    "global_60min": "Global 60 min",
}
POLICY_LABELS = {
    "embeddings": "Embeddings",
    "head": "Output head",
    "rt_head": "RT head",
    "cnn_head": "CNN + head",
    "gru_head": "GRU + head",
    "rnn_head": "RNN + head",
    "discriminative": "Discriminative",
    "all": "All parameters",
}
LR_LABELS = {
    "very_low": "0.1×",
    "low": "0.3×",
    "base": "1×",
    "high": "3×",
    "five_x": "5×",
    "very_high": "10×",
    "fifteen_x": "15×",
    "twenty_x": "20×",
    "thirty_x": "30×",
}
MODE_COLORS = {
    "anchor_irt": "#0072B2",
    "chronologer_hi": "#E69F00",
    "linear": "#CC79A7",
    "global_60min": "#6B7280",
}
ORGANISM_COLORS = {"celegans": "#0072B2", "human": "#E69F00", "triplemix": "#CC79A7"}
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<")
POLICY_MARKERS = {
    "embeddings": "o", "head": "s", "rt_head": "s", "cnn_head": "^",
    "gru_head": "D", "rnn_head": "D", "discriminative": "P", "all": "X",
}
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue", ["#EEF5FA", "#A6CEE3", "#4C93C3", "#075985"]
)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, output: Path, dpi: int = 600) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def configuration_summary(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "policy", "lr_level"]
    metrics = [
        "source_balanced_raw_mae", "max_source_raw_mae", "organism_macro_raw_mae",
        "semi_tryptic_raw_mae", "heavy_nterm_raw_mae", "heavy_lysine_raw_mae",
        "heavy_both_raw_mae",
    ]
    aggregations = {}
    for metric in metrics:
        if metric in runs:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")
    if "absolute_learning_rate" in runs:
        aggregations["absolute_learning_rate"] = ("absolute_learning_rate", "first")
    aggregations["seeds"] = ("seed", "nunique")
    result = runs.groupby(keys, as_index=False).agg(**aggregations)
    result["rank"] = result["source_balanced_raw_mae_mean"].rank(method="min").astype(int)
    return result.sort_values("rank")


def configuration_label(row: pd.Series) -> str:
    return f"{MODE_LABELS.get(row['mode'], row['mode'])} · {POLICY_LABELS.get(row['policy'], row['policy'])} · {LR_LABELS.get(row['lr_level'], row['lr_level'])}"


def learning_rate_levels(frame: pd.DataFrame) -> list[str]:
    """Return only levels present, in the intended multiplier order.

    Keeping absent levels out is important: a focused refinement sweep should
    retain compact, publication-ready panels rather than inherit empty columns
    from the preceding broad screen.
    """
    present = set(frame["lr_level"].dropna())
    ordered = [level for level in LR_ORDER if level in present]
    return ordered + sorted(present - set(ordered))


def policy_levels(frame: pd.DataFrame) -> list[str]:
    """Return known policies first and retain arbitrary sweep-specific profiles."""
    present = set(frame["policy"].dropna())
    ordered = [policy for policy in POLICY_LABELS if policy in present]
    return ordered + sorted(present - set(ordered))


def figure_ranking(runs: pd.DataFrame, summary: pd.DataFrame, output: Path, model_label: str) -> None:
    top = summary.nsmallest(15, "source_balanced_raw_mae_mean").copy().sort_values(
        "source_balanced_raw_mae_mean", ascending=False
    )
    top["label"] = top.apply(configuration_label, axis=1)
    fig, axis = plt.subplots(figsize=(7.1, 5.3))
    y = np.arange(len(top))
    for index, (_, row) in enumerate(top.iterrows()):
        color = MODE_COLORS.get(row["mode"], "#4B5563")
        error = row.get("source_balanced_raw_mae_std", np.nan)
        axis.errorbar(
            row["source_balanced_raw_mae_mean"], index,
            xerr=0 if pd.isna(error) else error,
            fmt="o", color=color, ecolor=color, markeredgecolor="white",
            markeredgewidth=0.5, markersize=5.2, elinewidth=1.0, capsize=2.0, zorder=3,
        )
        selected = runs[
            (runs["mode"] == row["mode"]) & (runs["policy"] == row["policy"])
            & (runs["lr_level"] == row["lr_level"])
        ]
        axis.scatter(selected["source_balanced_raw_mae"], np.full(len(selected), index),
                     s=9, facecolors="none", edgecolors="#374151", linewidths=0.45, zorder=4)
    axis.set_yticks(y, top["label"])
    axis.set_xlabel("Source-balanced validation MAE (min; lower is better)")
    axis.set_title(f"{model_label}: top validation configurations")
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.8)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=MODE_LABELS[mode], markersize=5)
               for mode, color in MODE_COLORS.items() if mode in set(top["mode"])]
    axis.legend(handles=handles, title="RT target space", frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output / "fig1_top_configuration_ranking")


def figure_heatmaps(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    modes = [mode for mode in MODE_ORDER if mode in set(summary["mode"])]
    policies = policy_levels(summary)
    lr_levels = learning_rate_levels(summary)
    values = summary["source_balanced_raw_mae_mean"].to_numpy(float)
    positive = values[np.isfinite(values) & (values > 0)]
    lower, upper = positive.min(), positive.max()
    normalizer = LogNorm(vmin=lower, vmax=upper)
    fig, axes = plt.subplots(
        1, len(modes), figsize=(max(7.1, 0.82 * len(lr_levels) * len(modes)), 3.9),
        sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, mode in zip(axes, modes):
        subset = summary[summary["mode"] == mode]
        matrix = subset.pivot(index="policy", columns="lr_level", values="source_balanced_raw_mae_mean")
        matrix = matrix.reindex(index=policies, columns=lr_levels)
        image = axis.imshow(matrix, aspect="auto", cmap=BLUE_CMAP.reversed(), norm=normalizer)
        axis.set_xticks(range(len(lr_levels)), [LR_LABELS.get(item, item) for item in lr_levels], rotation=45, ha="right")
        axis.set_yticks(range(len(policies)), [POLICY_LABELS.get(item, item) for item in policies])
        axis.set_title(MODE_LABELS.get(mode, mode))
        axis.set_xlabel("Learning-rate multiplier")
        for row in range(len(policies)):
            for column in range(len(lr_levels)):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    tone = "white" if normalizer(value) <= 0.42 else "#111827"
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.2, color=tone)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#9CA3AF")
            spine.set_linewidth(0.5)
    if image is not None:
        bar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.02)
        bar.set_label("Source-balanced MAE (min)")
    fig.suptitle(f"{model_label}: complete layer-policy × learning-rate landscape", fontsize=9.5)
    save_figure(fig, output / "fig2_policy_learning_rate_heatmaps")


def figure_organisms(runs: pd.DataFrame, summary: pd.DataFrame, output: Path, model_label: str) -> None:
    winners = summary.sort_values("source_balanced_raw_mae_mean").groupby("mode", as_index=False).first()
    selected_parts = []
    for _, winner in winners.iterrows():
        selected_parts.append(runs[
            (runs["mode"] == winner["mode"]) & (runs["policy"] == winner["policy"])
            & (runs["lr_level"] == winner["lr_level"])
        ])
    selected = pd.concat(selected_parts, ignore_index=True)
    organisms = [item for item in ("celegans", "human", "triplemix") if f"organism_{item}_raw_mae" in selected]
    fig, axes = plt.subplots(1, len(organisms), figsize=(7.1, 3.0), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    modes = [mode for mode in MODE_ORDER if mode in set(selected["mode"])]
    y = np.arange(len(modes))
    for axis, organism in zip(axes, organisms):
        metric = f"organism_{organism}_raw_mae"
        for index, mode in enumerate(modes):
            values = selected.loc[selected["mode"] == mode, metric].to_numpy(float)
            axis.scatter(values, np.full(len(values), index), s=12, facecolors="none",
                         edgecolors=MODE_COLORS[mode], linewidths=0.65, zorder=3)
            axis.errorbar(values.mean(), index, xerr=values.std(ddof=1) if len(values) > 1 else 0,
                          fmt="o", color=MODE_COLORS[mode], markersize=5, capsize=2, linewidth=1.0, zorder=4)
        axis.set_title({"celegans": "C. elegans", "human": "Human", "triplemix": "TripleMix"}[organism])
        axis.set_xlabel("Validation MAE (min)")
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.5)
        axis.set_yticks(y, [MODE_LABELS[item] for item in modes])
    fig.suptitle(f"{model_label}: organism performance of the best configuration per target space", fontsize=9.5)
    save_figure(fig, output / "fig3_organism_validation_performance")


def figure_organism_parameter_landscape(
    runs: pd.DataFrame, summary: pd.DataFrame, output: Path, model_label: str,
) -> None:
    """Compare policy/LR sensitivity across organisms in the best target space."""
    best_mode = str(summary.iloc[0]["mode"])
    subset = runs[runs["mode"] == best_mode]
    organisms = [
        item for item in ("celegans", "human", "triplemix")
        if f"organism_{item}_raw_mae" in subset
    ]
    policies = policy_levels(subset)
    lr_levels = learning_rate_levels(subset)
    all_values = np.concatenate([
        subset[f"organism_{organism}_raw_mae"].dropna().to_numpy(float)
        for organism in organisms
    ])
    normalizer = Normalize(vmin=float(all_values.min()), vmax=float(all_values.max()))
    fig, axes = plt.subplots(
        1, len(organisms), figsize=(7.1, 3.8), sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, organism in zip(axes, organisms):
        metric = f"organism_{organism}_raw_mae"
        matrix = subset.pivot_table(
            index="policy", columns="lr_level", values=metric, aggfunc="mean",
        ).reindex(index=policies, columns=lr_levels)
        image = axis.imshow(matrix, aspect="auto", cmap=BLUE_CMAP.reversed(), norm=normalizer)
        axis.set_xticks(range(len(lr_levels)), [LR_LABELS.get(item, item) for item in lr_levels], rotation=45, ha="right")
        axis.set_yticks(range(len(policies)), [POLICY_LABELS.get(item, item) for item in policies])
        axis.set_title({"celegans": "C. elegans", "human": "Human", "triplemix": "TripleMix"}[organism])
        axis.set_xlabel("Learning-rate multiplier")
        for row in range(len(policies)):
            for column in range(len(lr_levels)):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    tone = "white" if normalizer(value) <= 0.42 else "#111827"
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.2, color=tone)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#9CA3AF")
            spine.set_linewidth(0.5)
    if image is not None:
        bar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.02)
        bar.set_label("Organism MAE (min)")
    fig.suptitle(
        f"{model_label}: organism-specific parameter landscape in {MODE_LABELS.get(best_mode, best_mode)}",
        fontsize=9.5,
    )
    save_figure(fig, output / "fig6_organism_parameter_landscape")


def pareto_frontier(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    ordered = frame.sort_values([x, y])
    keep = []
    best_y = np.inf
    for index, row in ordered.iterrows():
        if row[y] < best_y:
            keep.append(index)
            best_y = row[y]
    return ordered.loc[keep]


def figure_tradeoffs(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    full_frame = summary.copy()
    # The complete grid is shown in Figure 2. Here we focus on the competitive
    # half so failed/high-error fits do not compress the relevant trade-offs.
    cutoff = full_frame["source_balanced_raw_mae_mean"].median()
    frame = full_frame[full_frame["source_balanced_raw_mae_mean"] <= cutoff].copy()
    heavy_columns = [column for column in (
        "heavy_nterm_raw_mae_mean", "heavy_lysine_raw_mae_mean", "heavy_both_raw_mae_mean"
    ) if column in frame]
    frame["heavy_macro_mae"] = frame[heavy_columns].mean(axis=1)
    policies = policy_levels(frame)
    marker_map = {policy: MARKERS[index % len(MARKERS)] for index, policy in enumerate(policies)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 4.0), constrained_layout=True)
    for mode in MODE_ORDER:
        for policy in policies:
            subset = frame[(frame["mode"] == mode) & (frame["policy"] == policy)]
            if subset.empty:
                continue
            left.scatter(subset["source_balanced_raw_mae_mean"], subset["heavy_macro_mae"],
                         s=23, marker=marker_map[policy], color=MODE_COLORS[mode], alpha=0.72,
                         edgecolor="white", linewidth=0.35)
            right.scatter(subset["source_balanced_raw_mae_mean"], subset["semi_tryptic_raw_mae_mean"],
                          s=23, marker=marker_map[policy], color=MODE_COLORS[mode], alpha=0.72,
                          edgecolor="white", linewidth=0.35)
    for axis, y, title in (
        (left, "heavy_macro_mae", "Heavy-Dimethyl strata"),
        (right, "semi_tryptic_raw_mae_mean", "Semi-tryptic peptides"),
    ):
        frontier = pareto_frontier(frame, "source_balanced_raw_mae_mean", y)
        axis.plot(frontier["source_balanced_raw_mae_mean"], frontier[y], color="#111827", linewidth=0.8, linestyle="--")
        best = frame.nsmallest(1, "source_balanced_raw_mae_mean").iloc[0]
        axis.scatter(best["source_balanced_raw_mae_mean"], best[y], s=62, facecolors="none",
                     edgecolors="#111827", linewidths=1.1, zorder=5)
        axis.set_xlabel("Source-balanced MAE (min)")
        axis.set_ylabel("Stratum MAE (min)")
        axis.set_title(title)
        axis.grid(color="#E5E7EB", linewidth=0.45)
    mode_handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=MODE_LABELS[mode], markersize=4.5)
                    for mode, color in MODE_COLORS.items() if mode in set(frame["mode"])]
    policy_handles = [Line2D([0], [0], marker=marker_map[policy], linestyle="", color="#374151",
                             label=POLICY_LABELS.get(policy, policy), markersize=4.5)
                      for policy in policies]
    left.legend(handles=mode_handles, title="RT target", frameon=False,
                loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
                columnspacing=0.9, handletextpad=0.4)
    right.legend(handles=policy_handles, title="Trainable parameters", frameon=False,
                 loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
                 columnspacing=0.8, handletextpad=0.4)
    fig.suptitle(f"{model_label}: validation trade-offs among competitive configurations", fontsize=9.5)
    save_figure(fig, output / "fig4_stratum_tradeoffs")


def figure_main_effects(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    policies = policy_levels(summary)
    modes = [mode for mode in MODE_ORDER if mode in set(summary["mode"])]
    lr_levels = learning_rate_levels(summary)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 4.0), constrained_layout=True)
    offsets = np.linspace(-0.24, 0.24, len(modes))
    for offset, mode in zip(offsets, modes):
        subset = summary[summary["mode"] == mode]
        marginal = subset.groupby("policy")["source_balanced_raw_mae_mean"].agg(["mean", "sem"]).reindex(policies)
        y = np.arange(len(policies)) + offset
        left.errorbar(marginal["mean"], y, xerr=marginal["sem"], fmt="o", color=MODE_COLORS[mode],
                      markersize=4, linewidth=0.9, capsize=1.8, label=MODE_LABELS[mode])
        lr = subset.groupby("lr_level")["source_balanced_raw_mae_mean"].agg(["mean", "sem"]).reindex(lr_levels)
        x = np.arange(len(lr_levels))
        right.errorbar(x, lr["mean"], yerr=lr["sem"], marker="o", color=MODE_COLORS[mode],
                       markersize=3.5, linewidth=1.1, capsize=1.8, label=MODE_LABELS[mode])
    left.set_yticks(np.arange(len(policies)), [POLICY_LABELS.get(item, item) for item in policies])
    left.set_xlabel("Marginal mean validation MAE (min)")
    left.set_xscale("log")
    left.set_title("Trainable parameter policy")
    left.grid(axis="x", color="#E5E7EB", linewidth=0.5)
    right.set_xticks(np.arange(len(lr_levels)), [LR_LABELS.get(item, item) for item in lr_levels])
    right.set_xlabel("Policy-specific LR multiplier")
    right.set_ylabel("Marginal mean validation MAE (min)")
    right.set_yscale("log")
    right.set_title("Learning-rate level")
    right.grid(color="#E5E7EB", linewidth=0.5)
    right.legend(frameon=False, title="RT target space", loc="upper center",
                 bbox_to_anchor=(0.5, -0.17), ncol=2,
                 columnspacing=0.9, handletextpad=0.4)
    fig.suptitle(f"{model_label}: marginal effects of swept parameters", fontsize=9.5)
    save_figure(fig, output / "fig5_parameter_main_effects")


def figure_performance_vs_seed_stability(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    """Show the accuracy/seed-robustness trade-off and its Pareto frontier."""
    metric = "source_balanced_raw_mae"
    frame = summary.dropna(subset=[f"{metric}_mean", f"{metric}_std"]).copy()
    # A global absolute-RT target is useful context but makes aligned targets unreadable.
    frame = frame[frame["mode"] != "global_60min"]
    policies = policy_levels(frame)
    marker_map = {policy: MARKERS[index % len(MARKERS)] for index, policy in enumerate(policies)}
    fig, axis = plt.subplots(figsize=(7.1, 4.2))
    for policy in policies:
        subset = frame[frame["policy"] == policy]
        axis.scatter(subset[f"{metric}_mean"], subset[f"{metric}_std"], s=30,
                     marker=marker_map[policy], c=[MODE_COLORS.get(mode, "#4B5563") for mode in subset["mode"]],
                     edgecolor="white", linewidth=0.4, alpha=0.85)
    frontier = pareto_frontier(frame, f"{metric}_mean", f"{metric}_std")
    axis.plot(frontier[f"{metric}_mean"], frontier[f"{metric}_std"], "--", color="#111827", linewidth=0.8, zorder=1)
    labels = pd.concat([frontier, frame.nsmallest(3, f"{metric}_mean")]).drop_duplicates()
    for _, row in labels.iterrows():
        axis.annotate(configuration_label(row), (row[f"{metric}_mean"], row[f"{metric}_std"]),
                      xytext=(4, 4), textcoords="offset points", fontsize=6.2)
    axis.set_xlabel("Mean source-balanced validation MAE (min; lower is better)")
    axis.set_ylabel("Seed SD of source-balanced validation MAE (min; lower is better)")
    axis.set_title(f"{model_label}: performance versus seed stability")
    axis.grid(color="#E5E7EB", linewidth=0.5)
    mode_handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=MODE_LABELS.get(mode, mode), markersize=4.5)
                    for mode, color in MODE_COLORS.items() if mode in set(frame["mode"])]
    policy_handles = [Line2D([0], [0], marker=marker_map[policy], linestyle="", color="#374151",
                             label=POLICY_LABELS.get(policy, policy), markersize=4.5) for policy in policies]
    axis.legend(handles=mode_handles + policy_handles, frameon=False, ncol=2, loc="upper right", fontsize=6.5)
    save_figure(fig, output / "fig7_performance_vs_seed_stability")


def figure_guardrail_penalty_heatmap(summary: pd.DataFrame, output: Path, model_label: str, top_n: int = 15) -> None:
    overall = "source_balanced_raw_mae_mean"
    guardrails = {
        "Worst\nsource": "max_source_raw_mae_mean", "Organism\nmacro": "organism_macro_raw_mae_mean",
        "Semi-\ntryptic": "semi_tryptic_raw_mae_mean", "Heavy\nN-term": "heavy_nterm_raw_mae_mean",
        "Heavy\nLys": "heavy_lysine_raw_mae_mean", "Heavy\nboth": "heavy_both_raw_mae_mean",
    }
    available = {label: column for label, column in guardrails.items() if column in summary}
    if not available:
        return
    frame = summary.nsmallest(top_n, overall).copy()
    matrix = pd.DataFrame({label: frame[column] - frame[overall] for label, column in available.items()}, index=frame.index)
    limit = float(np.nanmax(np.abs(matrix.to_numpy(float)))) if matrix.notna().any().any() else 1.0
    limit = max(limit, 0.01)
    fig, axis = plt.subplots(figsize=(7.1, max(3.2, 0.32 * len(frame) + 1.2)))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=Normalize(vmin=-limit, vmax=limit))
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(frame)), frame.apply(configuration_label, axis=1))
    for row in range(len(frame)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=6.3,
                          color="white" if abs(value) > limit * 0.55 else "#111827")
    bar = fig.colorbar(image, ax=axis, pad=0.02)
    bar.set_label("MAE penalty vs. each configuration's overall MAE (min)")
    axis.set_title(f"{model_label}: guardrail penalties for top validation configurations")
    save_figure(fig, output / "fig8_top_config_guardrail_penalties")


def figure_learning_rate_response(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    """Plot actual primary LR, seed uncertainty, and boundary optima per target."""
    if "absolute_learning_rate" not in summary or summary["absolute_learning_rate"].notna().sum() == 0:
        return
    modes = [mode for mode in MODE_ORDER if mode in set(summary["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(max(7.1, 2.3 * len(modes)), 3.5), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, mode in zip(axes, modes):
        subset = summary[(summary["mode"] == mode) & summary["absolute_learning_rate"].notna()]
        for index, (policy, group) in enumerate(subset.groupby("policy")):
            group = group.sort_values("absolute_learning_rate")
            x, y = group["absolute_learning_rate"], group["source_balanced_raw_mae_mean"]
            errors = group["source_balanced_raw_mae_std"].fillna(0.0)
            color = plt.cm.tab10(index % 10)
            axis.errorbar(x, y, yerr=errors, marker=MARKERS[index % len(MARKERS)], color=color,
                          markersize=3.7, linewidth=1.0, capsize=1.8, label=POLICY_LABELS.get(policy, policy))
            best = group.loc[y.idxmin()]
            if best["absolute_learning_rate"] in (x.min(), x.max()):
                axis.annotate("▲ boundary", (best["absolute_learning_rate"], best["source_balanced_raw_mae_mean"]),
                              xytext=(2, 6), textcoords="offset points", fontsize=5.8, color=color)
        axis.set_xscale("log")
        axis.set_title(MODE_LABELS.get(mode, mode))
        axis.set_xlabel("Primary learning rate (log scale)")
        axis.grid(color="#E5E7EB", linewidth=0.5)
    axes[0].set_ylabel("Mean source-balanced validation MAE (min)")
    axes[-1].legend(frameon=False, fontsize=6.2, loc="best")
    fig.suptitle(f"{model_label}: learning-rate response with seed uncertainty", fontsize=9.5)
    save_figure(fig, output / "fig9_learning_rate_response")


def figure_ranking_improved(runs: pd.DataFrame, summary: pd.DataFrame, output: Path, model_label: str) -> None:
    top = summary.nsmallest(15, "source_balanced_raw_mae_mean").copy().sort_values("source_balanced_raw_mae_mean", ascending=False)
    top["label"] = top.apply(configuration_label, axis=1)
    fig, axis = plt.subplots(figsize=(7.1, 5.3))
    rng = np.random.default_rng(20260821)
    for index, (_, row) in enumerate(top.iterrows()):
        color = MODE_COLORS.get(row["mode"], "#4B5563")
        mean, sd = row["source_balanced_raw_mae_mean"], row.get("source_balanced_raw_mae_std", np.nan)
        axis.errorbar(mean, index, xerr=0 if pd.isna(sd) else sd, fmt="D", color=color, ecolor=color,
                      markeredgecolor="white", markeredgewidth=0.5, markersize=4.8, capsize=2, zorder=3)
        selected = runs[(runs["mode"] == row["mode"]) & (runs["policy"] == row["policy"]) & (runs["lr_level"] == row["lr_level"])]
        jitter = rng.uniform(-0.12, 0.12, len(selected))
        axis.scatter(selected["source_balanced_raw_mae"], index + jitter, s=12, facecolors="none", edgecolors="#374151", linewidths=0.45, zorder=4)
        axis.annotate(f"{mean:.3f}", (mean, index), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.4)
    axis.set_yticks(np.arange(len(top)), top["label"])
    axis.set_xlabel("Source-balanced validation MAE (min; lower is better; truncated scale)")
    axis.set_title(f"{model_label}: top validation configurations (mean ± SD; n = 3 seeds)")
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output / "fig1_top_configuration_ranking_improved")


def figure_heatmaps_improved(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    modes = [mode for mode in MODE_ORDER if mode in set(summary["mode"])]
    policies = policy_levels(summary)
    lr_levels = learning_rate_levels(summary)
    fig, axes = plt.subplots(
        1, len(modes), figsize=(max(7.1, 0.82 * len(lr_levels) * len(modes)), 3.9),
        sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    aligned = summary[summary["mode"] != "global_60min"]["source_balanced_raw_mae_mean"]
    global_values = summary[summary["mode"] == "global_60min"]["source_balanced_raw_mae_mean"]
    norms = {
        "aligned": Normalize(vmin=float(aligned.min()), vmax=float(aligned.max())),
        "global": Normalize(vmin=float(global_values.min()), vmax=float(global_values.max())) if len(global_values) else None,
    }
    images = {}
    for axis, mode in zip(axes, modes):
        matrix = summary[summary["mode"] == mode].pivot(index="policy", columns="lr_level", values="source_balanced_raw_mae_mean").reindex(index=policies, columns=lr_levels)
        group = "global" if mode == "global_60min" else "aligned"
        image = axis.imshow(matrix, aspect="auto", cmap=BLUE_CMAP.reversed(), norm=norms[group])
        images[group] = image
        axis.set_xticks(range(len(lr_levels)), [LR_LABELS.get(item, item) for item in lr_levels], rotation=45, ha="right")
        axis.set_yticks(range(len(policies)), [POLICY_LABELS.get(item, item) for item in policies])
        axis.set_title(MODE_LABELS.get(mode, mode) + (" (separate scale)" if group == "global" else ""))
        best_y, best_x = np.unravel_index(np.nanargmin(matrix.to_numpy(float)), matrix.shape)
        axis.add_patch(plt.Rectangle((best_x - .48, best_y - .48), .96, .96, fill=False, edgecolor="#111827", linewidth=1.4))
        for row in range(len(policies)):
            for column in range(len(lr_levels)):
                value = matrix.iloc[row, column]
                if pd.notna(value): axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.1)
    for group, image in images.items():
        bar = fig.colorbar(image, ax=axes, shrink=.72, pad=.02)
        bar.set_label("MAE (min; " + ("Global 60 min" if group == "global" else "aligned targets") + ")")
    fig.suptitle(f"{model_label}: policy × learning-rate landscape (outlined = panel best)", fontsize=9.5)
    save_figure(fig, output / "fig2_policy_learning_rate_heatmaps_improved")


def figure_organisms_improved(runs: pd.DataFrame, summary: pd.DataFrame, output: Path, model_label: str) -> None:
    winners = summary.sort_values("source_balanced_raw_mae_mean").groupby("mode", as_index=False).first()
    selected = pd.concat([runs[(runs["mode"] == row.mode) & (runs["policy"] == row.policy) & (runs["lr_level"] == row.lr_level)] for row in winners.itertuples()], ignore_index=True)
    organisms = [item for item in ("celegans", "human", "triplemix") if f"organism_{item}_raw_mae" in selected]
    values = np.concatenate([selected[f"organism_{item}_raw_mae"].dropna().to_numpy() for item in organisms])
    limits = (max(0, float(values.min()) - .08), float(values.max()) + .12)
    fig, axes = plt.subplots(1, len(organisms), figsize=(7.1, 3.0), sharey=True, sharex=True, constrained_layout=True)
    modes = [mode for mode in MODE_ORDER if mode in set(selected["mode"])]
    for axis, organism in zip(np.atleast_1d(axes), organisms):
        metric = f"organism_{organism}_raw_mae"
        for index, mode in enumerate(modes):
            item = selected[selected["mode"] == mode][metric].dropna()
            axis.scatter(item, np.full(len(item), index), s=12, facecolors="none", edgecolors=MODE_COLORS[mode], linewidths=.65)
            axis.errorbar(item.mean(), index, xerr=item.std(ddof=1) if len(item) > 1 else 0, fmt="D", color=MODE_COLORS[mode], markersize=4, capsize=2)
        axis.set_title({"celegans": "C. elegans", "human": "Human", "triplemix": "TripleMix"}[organism])
        axis.set_xlim(*limits); axis.set_xlabel("Validation MAE (min)"); axis.grid(axis="x", color="#D1D5DB", linewidth=.5)
    axes[0].set_yticks(np.arange(len(modes)), [MODE_LABELS[item] for item in modes])
    fig.suptitle(f"{model_label}: organism performance (common x-axis; mean ± SD, n = 3)", fontsize=9.5)
    save_figure(fig, output / "fig3_organism_validation_performance_improved")


def figure_tradeoffs_improved(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    frame = summary[summary["source_balanced_raw_mae_mean"] <= summary["source_balanced_raw_mae_mean"].median()].copy()
    heavy_columns = [column for column in ("heavy_nterm_raw_mae_mean", "heavy_lysine_raw_mae_mean", "heavy_both_raw_mae_mean") if column in frame]
    frame["heavy_dimethyl_macro_mae"] = frame[heavy_columns].mean(axis=1)
    policies = policy_levels(frame)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.0), constrained_layout=True)
    for axis, y, title in ((axes[0], "heavy_dimethyl_macro_mae", "Heavy-Dimethyl macro-MAE"), (axes[1], "semi_tryptic_raw_mae_mean", "Semi-tryptic MAE")):
        for mode in MODE_ORDER:
            for policy in policies:
                item = frame[(frame["mode"] == mode) & (frame["policy"] == policy)]
                axis.scatter(item["source_balanced_raw_mae_mean"], item[y], s=23, marker=POLICY_MARKERS.get(policy, "o"), color=MODE_COLORS[mode], alpha=.75, edgecolor="white", linewidth=.35)
        lo = min(frame["source_balanced_raw_mae_mean"].min(), frame[y].min()); hi = max(frame["source_balanced_raw_mae_mean"].max(), frame[y].max())
        axis.plot((lo, hi), (lo, hi), color="#9CA3AF", linewidth=.7, linestyle=":", label="equal to overall MAE")
        axis.set_xlabel("Source-balanced MAE (min)"); axis.set_ylabel(f"{title} (min)"); axis.set_title(title); axis.grid(color="#E5E7EB", linewidth=.45)
    fig.suptitle(f"{model_label}: HUNTER-relevant stratum trade-offs", fontsize=9.5)
    save_figure(fig, output / "fig4_stratum_tradeoffs_improved")


def figure_main_effects_improved(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    """Keep the marginal display, clearly labelled as a supplementary diagnostic."""
    policies = policy_levels(summary)
    modes = [mode for mode in MODE_ORDER if mode in set(summary["mode"])]
    lr_levels = learning_rate_levels(summary)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 4.0), constrained_layout=True)
    for mode in modes:
        subset = summary[summary["mode"] == mode]
        marginal = subset.groupby("policy")["source_balanced_raw_mae_mean"].agg(["mean", "sem"]).reindex(policies)
        left.errorbar(marginal["mean"], np.arange(len(policies)), xerr=marginal["sem"], fmt="o", color=MODE_COLORS[mode], markersize=3.5, capsize=1.8, label=MODE_LABELS[mode])
        lr = subset.groupby("lr_level")["source_balanced_raw_mae_mean"].agg(["mean", "sem"]).reindex(lr_levels)
        right.errorbar(np.arange(len(lr_levels)), lr["mean"], yerr=lr["sem"], marker="o", color=MODE_COLORS[mode], markersize=3.5, capsize=1.8)
    left.set_yticks(np.arange(len(policies)), [POLICY_LABELS.get(item, item) for item in policies]); left.set_xlabel("Marginal mean validation MAE (min)")
    right.set_xticks(np.arange(len(lr_levels)), [LR_LABELS.get(item, item) for item in lr_levels]); right.set_xlabel("LR multiplier"); right.set_ylabel("Marginal mean validation MAE (min)")
    fig.suptitle(f"Supplementary — {model_label}: marginal effects (interactions shown in Fig. 2)", fontsize=9.2)
    save_figure(fig, output / "fig5_parameter_main_effects_improved")


def figure_organism_parameter_landscape_improved(runs: pd.DataFrame, summary: pd.DataFrame, output: Path, model_label: str) -> None:
    best_mode = str(summary.iloc[0]["mode"]); subset = runs[runs["mode"] == best_mode]
    organisms = [item for item in ("celegans", "human", "triplemix") if f"organism_{item}_raw_mae" in subset]
    policies = policy_levels(subset)
    lr_levels = learning_rate_levels(subset)
    all_values = np.concatenate([subset[f"organism_{item}_raw_mae"].dropna().to_numpy(float) for item in organisms])
    fig, axes = plt.subplots(1, len(organisms), figsize=(7.1, 3.8), sharey=True, constrained_layout=True)
    image = None
    for axis, organism in zip(np.atleast_1d(axes), organisms):
        matrix = subset.pivot_table(index="policy", columns="lr_level", values=f"organism_{organism}_raw_mae", aggfunc="mean").reindex(index=policies, columns=lr_levels)
        image = axis.imshow(matrix, aspect="auto", cmap=BLUE_CMAP.reversed(), norm=Normalize(vmin=all_values.min(), vmax=all_values.max()))
        best_y, best_x = np.unravel_index(np.nanargmin(matrix.to_numpy(float)), matrix.shape)
        axis.add_patch(plt.Rectangle((best_x - .48, best_y - .48), .96, .96, fill=False, edgecolor="#111827", linewidth=1.4))
        axis.set_xticks(range(len(lr_levels)), [LR_LABELS.get(item, item) for item in lr_levels], rotation=45, ha="right"); axis.set_yticks(range(len(policies)), [POLICY_LABELS.get(item, item) for item in policies])
        axis.set_title({"celegans": "C. elegans", "human": "Human", "triplemix": "TripleMix"}[organism]); axis.set_xlabel("Learning-rate multiplier")
    bar = fig.colorbar(image, ax=np.atleast_1d(axes), shrink=.82, pad=.02); bar.set_label("Organism MAE (min; mean of 3 seeds)")
    fig.suptitle(f"{model_label}: organism-specific landscape in {MODE_LABELS.get(best_mode, best_mode)} (outlined = organism best)", fontsize=9.2)
    save_figure(fig, output / "fig6_organism_parameter_landscape_improved")


PLOT_GUIDES = {
    "fig1_top_configuration_ranking": ("Configuration ranking", "x: source-balanced validation MAE in minutes; y: ranked training configurations. Filled symbols show seed means, open symbols individual seeds, and bars the seed SD.", "Further left is better. Compare both the mean and the spread of the seed points: a small mean advantage may not be practically meaningful when seed variation overlaps."),
    "fig2_policy_learning_rate_heatmaps": ("Policy × learning-rate landscape", "Rows: trainable-parameter policy; columns: learning-rate level; panels: RT target space. Each cell is the mean source-balanced validation MAE.", "Lower-valued cells are better. Look for policy×LR interactions, broad low-error regions (robust choices), and whether the best cell is at a searched boundary."),
    "fig3_organism_validation_performance": ("Organism performance", "x: validation MAE in minutes; y: target space. Panels: organism. Symbols/bars show seed-level estimates and their variation for the best configuration per target space.", "Further left is better. With common axes, horizontal distances can be compared directly across organisms; differing positions indicate organism-specific difficulty."),
    "fig4_stratum_tradeoffs": ("Biological-stratum trade-offs", "x: overall source-balanced MAE; y: either Heavy-Dimethyl macro-MAE or semi-tryptic MAE. Each point is one configuration.", "Bottom-left is preferable. Points below the y=x reference have lower stratum than overall error; points above it show a stratum penalty. A Pareto frontier identifies choices not jointly worse on both axes."),
    "fig5_parameter_main_effects": ("Marginal parameter effects", "Left: marginal mean MAE by training policy; right: marginal mean MAE by LR level. Values average over the other swept factors.", "Use this as a broad descriptive diagnostic only. Strong policy×LR interactions mean that a marginal average may not correspond to any single runnable configuration; inspect the heatmap for selection."),
    "fig6_organism_parameter_landscape": ("Organism-specific parameter landscape", "Rows: trainable-parameter policy; columns: learning-rate level; panels: organism. Cells are mean validation MAE within the selected RT target space.", "Lower cells are better. A common colour scale enables direct comparison of both absolute performance and whether the parameter optimum changes by organism."),
    "fig7_performance_vs_seed_stability": ("Performance versus seed stability", "x: mean source-balanced validation MAE; y: standard deviation of that MAE across seeds. Colours encode target space and markers encode policy.", "Bottom-left is best. Pareto-optimal points have no alternative that is both more accurate and more stable; this plot distinguishes reliable improvements from potentially seed-sensitive ones."),
    "fig8_top_config_guardrail_penalties": ("Guardrail penalties", "Rows: top configurations by overall MAE; columns: source/organism/biological guardrails. Each value is guardrail MAE minus that configuration's overall MAE.", "Zero means similar performance to the overall result. Positive values identify disproportionately difficult subsets; negative values identify subsets that are easier than the overall population."),
    "fig9_learning_rate_response": ("Learning-rate response", "x: absolute primary learning rate on a logarithmic scale; y: mean source-balanced validation MAE. Lines represent policies; bars are seed SD.", "Lower points are better. A best point at either x-axis boundary indicates that the explored range may need expansion; flat low-error regions suggest LR robustness."),
}


def write_plot_guides(output_dir: Path, figure_names: list[str]) -> None:
    """Write data-independent reading guides beside every exported plot family."""
    guide_dir = output_dir / "plot_readmes"
    guide_dir.mkdir(parents=True, exist_ok=True)
    index = ["# Plot interpretation guides", "", "These guides describe how to read each plot in general. They intentionally make no claim about the results of a particular sweep.", ""]
    for name in figure_names:
        base = name.removesuffix("_improved")
        title, axes, interpretation = PLOT_GUIDES[base]
        text = f"# {title}\n\n## What this plot shows\n\n{axes}\n\n## How to interpret it\n\n{interpretation}\n"
        (guide_dir / f"{name}.md").write_text(text, encoding="utf-8")
        index.append(f"- [{title}](./{name}.md)")
    (guide_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def generate_publication_figures(
    runs: pd.DataFrame, output_dir: Path, model_label: str, *, dpi: int = 600,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    del dpi  # The export contract is intentionally fixed at 600 dpi.
    configure_style()
    required = {"mode", "policy", "lr_level", "seed", "source_balanced_raw_mae"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Run table lacks plotting columns: {sorted(missing)}")
    summary = configuration_summary(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[tuple[str, Callable[[], None]]] = [
        ("Fig. 1 ranking", lambda: figure_ranking(runs, summary, output_dir, model_label)),
        ("Fig. 2 heatmaps", lambda: figure_heatmaps(summary, output_dir, model_label)),
        ("Fig. 3 organisms", lambda: figure_organisms(runs, summary, output_dir, model_label)),
        ("Fig. 4 trade-offs", lambda: figure_tradeoffs(summary, output_dir, model_label)),
        ("Fig. 5 main effects", lambda: figure_main_effects(summary, output_dir, model_label)),
        ("Fig. 6 organism landscape", lambda: figure_organism_parameter_landscape(runs, summary, output_dir, model_label)),
        ("Fig. 7 seed stability", lambda: figure_performance_vs_seed_stability(summary, output_dir, model_label)),
        ("Fig. 8 guardrail penalties", lambda: figure_guardrail_penalty_heatmap(summary, output_dir, model_label)),
        ("Fig. 1 improved ranking", lambda: figure_ranking_improved(runs, summary, output_dir, model_label)),
        ("Fig. 2 improved heatmaps", lambda: figure_heatmaps_improved(summary, output_dir, model_label)),
        ("Fig. 3 improved organisms", lambda: figure_organisms_improved(runs, summary, output_dir, model_label)),
        ("Fig. 4 improved trade-offs", lambda: figure_tradeoffs_improved(summary, output_dir, model_label)),
        ("Fig. 5 improved main effects", lambda: figure_main_effects_improved(summary, output_dir, model_label)),
        ("Fig. 6 improved organism landscape", lambda: figure_organism_parameter_landscape_improved(runs, summary, output_dir, model_label)),
    ]
    if "absolute_learning_rate" in summary and summary["absolute_learning_rate"].notna().any():
        steps.insert(8, ("Fig. 9 learning-rate response", lambda: figure_learning_rate_response(summary, output_dir, model_label)))
    for index, (label, create) in enumerate(steps, start=1):
        if progress:
            progress(index - 1, len(steps), label)
        create()
    if progress:
        progress(len(steps), len(steps), "complete")
    names = [
        "fig1_top_configuration_ranking", "fig2_policy_learning_rate_heatmaps",
        "fig3_organism_validation_performance", "fig4_stratum_tradeoffs",
        "fig5_parameter_main_effects",
        "fig6_organism_parameter_landscape",
        "fig7_performance_vs_seed_stability",
        "fig8_top_config_guardrail_penalties",
        "fig1_top_configuration_ranking_improved",
        "fig2_policy_learning_rate_heatmaps_improved",
        "fig3_organism_validation_performance_improved",
        "fig4_stratum_tradeoffs_improved",
        "fig5_parameter_main_effects_improved",
        "fig6_organism_parameter_landscape_improved",
    ]
    if "absolute_learning_rate" in summary and summary["absolute_learning_rate"].notna().any():
        names.append("fig9_learning_rate_response")
    write_plot_guides(output_dir, names)
    manifest = {
        "model_label": model_label,
        "input_runs": int(len(runs)),
        "configurations": int(len(summary)),
        "formats": ["png", "pdf", "svg"],
        "dpi_png": 600,
        "figures": names,
        "metric": "source-balanced validation MAE in observed RT minutes",
        "test_partition_used": False,
        "plot_readmes": "plot_readmes/ contains one data-independent interpretation guide per figure.",
        "display_contract": {
            "fig2_colour_scale": "logarithmic; full factorial grid",
            "fig4_filter": "configurations at or below the median source-balanced validation MAE",
            "fig4_cutoff_minutes": float(summary["source_balanced_raw_mae_mean"].median()),
            "fig5_axes": "logarithmic on the metric axis",
            "fig7_filter": "global_60min excluded to retain resolution among aligned target spaces",
            "fig8_values": "guardrail MAE minus the configuration's source-balanced MAE",
            "fig9_boundary_flag": "triangle label marks a policy optimum at either searched LR boundary",
            "improved_figures": "Original figures are retained; *_improved variants apply publication-focused presentation corrections.",
        },
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    captions = """# RT sweep publication figures

All panels use validation data only. Points denote configurations or individual random seeds as stated; intervals are standard deviations across seeds unless otherwise noted. Lower MAE is better.

1. **Top configuration ranking.** Fifteen configurations with the lowest mean source-balanced validation MAE. Filled points are seed means, open points are individual seeds and error bars are standard deviations.
2. **Policy × learning-rate heatmaps.** Complete factorial landscape separately for each RT target space. Cell values are mean source-balanced MAE in observed minutes; the colour scale is logarithmic so competitive and failed fits remain distinguishable.
3. **Organism performance.** Best validation-selected configuration within each RT target space, evaluated separately for *C. elegans*, Human and TripleMix observations.
4. **Stratum trade-offs.** Overall source-balanced MAE versus equal-weight Heavy-Dimethyl macro-MAE and semi-tryptic MAE, restricted to the best half of configurations by overall validation MAE. This display filter prevents failed fits from compressing the competitive region; the complete grid remains visible in Figure 2. Dashed lines show the empirical Pareto frontier; the outlined point is the overall validation winner.
5. **Parameter main effects.** Marginal means across the remaining swept factors, with standard errors across configurations. These panels describe the explored grid and are not causal estimates.
6. **Organism-specific parameter landscape.** Policy × learning-rate validation MAE for each organism within the RT target space containing the overall validation winner. Values are seed means. This diagnostic tests whether the global parameter optimum behaves consistently across organisms; target-space selection remains validation-only.
7. **Performance versus seed stability.** Each point is a configuration, positioned by its mean validation MAE and seed standard deviation. The dashed line is the empirical Pareto frontier; labels identify Pareto-optimal configurations and the top three by MAE. The global absolute-RT target is omitted so aligned target spaces remain legible.
8. **Guardrail penalties.** For the top fifteen validation configurations, each cell is a stratum MAE minus that configuration's overall source-balanced MAE. Positive values identify disproportionately difficult sources or biological strata.
9. **Learning-rate response.** Mean validation MAE versus the absolute primary learning rate on a logarithmic scale, with seed SD error bars. A triangle annotation marks a policy whose best tested LR is at the boundary of the searched interval. For discriminative policies, this is the primary/head LR; lower active backbone rates are defined by the sweep manifest.

## Improved variants

The original figures are retained unchanged. Files ending in `_improved` are publication-oriented alternatives: Figure 1 explicitly shows mean ± SD (n = 3) and a truncated scale; Figure 2 separates Global 60 min colour normalization and outlines panel optima; Figure 3 uses one common x-axis; Figure 4 defines Heavy-Dimethyl as the unweighted macro-average of N-term, Lys and both strata; Figure 5 is marked supplementary because policy×LR interactions limit marginal-effect interpretation; and Figure 6 outlines the organism-specific optimum and declares seed averaging.
"""
    (output_dir / "README.md").write_text(captions, encoding="utf-8")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    args = parser.parse_args()
    runs = pd.read_csv(args.runs_csv)
    figures = generate_publication_figures(runs, args.output_dir, args.model_label)
    print(json.dumps({"figures": figures, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
