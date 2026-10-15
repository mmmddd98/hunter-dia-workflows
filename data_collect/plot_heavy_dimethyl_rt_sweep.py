"""Reusable figures for controlled Heavy-Dimethyl RT sweeps.

The module consumes the three tables produced by
``analyze_heavy_dimethyl_rt_sweep.py``.  It intentionally knows nothing about
the trainer: DIA-NN and Carafe2 use the same manifest and table contract, so
the complete figure set is directly comparable across model families.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_rt_sweep_publication import configure_style, save_figure


ARM_COLORS = {"weighting": "#D55E00", "oversampling": "#0072B2"}
ARM_LABELS = {"weighting": "Loss weighting", "oversampling": "Oversampling"}
DELTA_METRICS = {
    "delta_heavy_macro_source_balanced_mae": "Heavy macro\n(source-balanced)",
    "delta_heavy_nterm_sb_mae": "Heavy N-term",
    "delta_heavy_lys_sb_mae": "Heavy lysine",
    "delta_heavy_both_sb_mae": "Heavy both",
    "delta_nonheavy_sb_mae": "Non-heavy",
    "delta_overall_mae": "Overall",
    "delta_semitryptic_mae": "Semi-tryptic",
    "delta_worst_source_mae": "Worst source",
}
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "improvement_to_cost", ["#2166AC", "#F7F7F7", "#B2182B"]
)
CONFIGURATION_KEYS = [
    "model", "arm", "target", "policy", "profile", "loss", "lr", "weights", "oversampling",
]


def _profile_order(values: pd.Series) -> list[str]:
    def key(value: object) -> tuple[str, int]:
        text = str(value)
        prefix = "".join(character for character in text if character.isalpha())
        digits = "".join(character for character in text if character.isdigit())
        return prefix, int(digits) if digits else -1

    return sorted(values.dropna().unique(), key=key)


def _lr_order(values: pd.Series) -> list[float]:
    return sorted(float(value) for value in values.dropna().unique())


def _configuration_label(row: pd.Series) -> str:
    values = row.get("weights") if row.get("arm") == "weighting" else row.get("oversampling")
    return f"{row['profile']} · {float(row['lr']):.0e} · {values}"


def _pareto_minimum(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Return the lower-left Pareto frontier; lower MAE deltas are better."""
    ordered = frame.sort_values([x, y])
    best_y = np.inf
    keep: list[int] = []
    for index, row in ordered.iterrows():
        if row[y] < best_y:
            keep.append(index)
            best_y = row[y]
    return ordered.loc[keep]


def _delta_norm(frame: pd.DataFrame, columns: list[str]) -> TwoSlopeNorm:
    values = frame[columns].to_numpy(dtype=float).ravel()
    finite = np.abs(values[np.isfinite(values)])
    limit = float(np.quantile(finite, 0.98)) if len(finite) else 0.01
    return TwoSlopeNorm(vmin=-max(limit, 1e-4), vcenter=0.0, vmax=max(limit, 1e-4))


def figure_tradeoff_frontiers(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    """Plot matched Heavy benefit against matched Non-heavy cost per target."""
    metric_x = "delta_nonheavy_sb_mae"
    metric_y = "delta_heavy_macro_source_balanced_mae"
    frame = summary.dropna(subset=[metric_x, metric_y]).copy()
    targets = list(frame["target"].dropna().unique())
    if frame.empty or not targets:
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(3.65 * len(targets), 3.55), squeeze=False)
    axes = axes[0]
    max_abs = float(np.abs(frame[[metric_x, metric_y]].to_numpy(dtype=float)).max())
    pad = max(0.003, max_abs * 0.08)
    for axis, target in zip(axes, targets):
        panel = frame[frame["target"] == target]
        axis.axhspan(-max_abs - pad, 0, xmin=0, xmax=0.5, color="#E8F3EC", zorder=0)
        axis.axhline(0, color="#6B7280", linewidth=0.7, zorder=1)
        axis.axvline(0, color="#6B7280", linewidth=0.7, zorder=1)
        for arm, group in panel.groupby("arm", sort=False):
            color = ARM_COLORS.get(arm, "#4B5563")
            failures = group.get("guardrail_status", pd.Series("PASS", index=group.index)).ne("PASS")
            axis.scatter(
                group[metric_x], group[metric_y], s=25, color=color, alpha=0.78,
                edgecolor=np.where(failures, "#111827", "white"), linewidth=0.45,
                label=ARM_LABELS.get(arm, arm), zorder=3,
            )
            frontier = _pareto_minimum(group, metric_x, metric_y)
            axis.plot(frontier[metric_x], frontier[metric_y], color=color, linewidth=1.1, zorder=2)
            for _, row in frontier.nsmallest(4, metric_y).iterrows():
                axis.annotate(
                    f"{row['profile']}@{float(row['lr']):.0e}", (row[metric_x], row[metric_y]), xytext=(3, -5),
                    textcoords="offset points", fontsize=6.3, color=color,
                )
        axis.scatter([0], [0], marker="+", s=52, color="#111827", linewidths=1.1, zorder=4)
        axis.set(
            title=str(target).replace("_", " "),
            xlabel="Matched Δ non-heavy SB-MAE (min)",
            ylabel="Matched Δ heavy-macro SB-MAE (min)",
            xlim=(-max_abs - pad, max_abs + pad), ylim=(-max_abs - pad, max_abs + pad),
        )
        axis.grid(color="#E5E7EB", linewidth=0.5, zorder=0)
    axes[0].legend(frameon=False, title="Intervention", loc="upper left")
    fig.suptitle(f"{model_label}: matched intervention trade-offs (lower-left is favourable)", fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, output / "heavy_benefit_vs_nonheavy_cost")


def figure_effect_landscape(summary: pd.DataFrame, output: Path, metric: str, title: str) -> None:
    """Show every tested profile × learning-rate combination as a heatmap."""
    frame = summary.dropna(subset=[metric]).copy()
    targets = list(frame["target"].dropna().unique())
    arms = [arm for arm in ("weighting", "oversampling") if arm in set(frame["arm"])]
    if frame.empty or not targets or not arms:
        return
    normalizer = _delta_norm(frame, [metric])
    fig, axes = plt.subplots(
        len(targets), len(arms), figsize=(3.55 * len(arms), 2.6 * len(targets)),
        squeeze=False, constrained_layout=True,
    )
    image = None
    for row_index, target in enumerate(targets):
        for column_index, arm in enumerate(arms):
            axis = axes[row_index, column_index]
            panel = frame[(frame["target"] == target) & (frame["arm"] == arm)]
            profiles = _profile_order(panel["profile"])
            lrs = _lr_order(panel["lr"])
            matrix = panel.pivot_table(index="profile", columns="lr", values=metric, aggfunc="mean")
            matrix = matrix.reindex(index=profiles, columns=lrs)
            image = axis.imshow(matrix, aspect="auto", cmap=DELTA_CMAP, norm=normalizer)
            axis.set_xticks(range(len(lrs)), [f"{value:.0e}" for value in lrs])
            axis.set_yticks(range(len(profiles)), profiles)
            axis.set_xlabel("Learning rate")
            axis.set_ylabel("Profile" if column_index == 0 else "")
            axis.set_title(f"{str(target).replace('_', ' ')} · {ARM_LABELS.get(arm, arm)}")
            for y in range(len(profiles)):
                for x in range(len(lrs)):
                    value = matrix.iloc[y, x]
                    if pd.notna(value):
                        color = "white" if abs(value) > normalizer.vmax * 0.62 else "#111827"
                        axis.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=6.2, color=color)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#9CA3AF")
                spine.set_linewidth(0.5)
    if image is not None:
        bar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
        bar.set_label("Matched Δ MAE (min; blue = improvement)")
    fig.suptitle(title, fontsize=9.5)
    save_figure(fig, output / f"figures/{metric}_landscape")


def _top_candidates(summary: pd.DataFrame, per_arm: int = 5) -> pd.DataFrame:
    metric = "delta_heavy_macro_source_balanced_mae"
    candidates = summary.dropna(subset=[metric]).copy()
    if "guardrail_status" in candidates:
        candidates["guardrail_priority"] = candidates["guardrail_status"].ne("PASS").astype(int)
    else:
        candidates["guardrail_priority"] = 0
    pieces = []
    for _, group in candidates.groupby(["target", "arm"], sort=False):
        pieces.append(group.sort_values(["guardrail_priority", metric]).head(per_arm))
    return pd.concat(pieces, ignore_index=True) if pieces else candidates.iloc[0:0]


def figure_stratum_effects(summary: pd.DataFrame, output: Path, model_label: str) -> pd.DataFrame:
    """Show which peptide stratum drives the apparent benefit of top candidates."""
    selected = _top_candidates(summary)
    columns = list(DELTA_METRICS)
    targets = list(selected["target"].dropna().unique())
    if selected.empty or not targets:
        return selected
    normalizer = _delta_norm(selected, columns)
    fig, axes = plt.subplots(len(targets), 1, figsize=(7.8, 3.75 * len(targets)), squeeze=False)
    axes = axes[:, 0]
    for axis, target in zip(axes, targets):
        panel = selected[selected["target"] == target].copy()
        panel["label"] = panel.apply(_configuration_label, axis=1)
        matrix = panel.set_index("label")[columns]
        image = axis.imshow(matrix, aspect="auto", cmap=DELTA_CMAP, norm=normalizer)
        axis.set_xticks(range(len(columns)), list(DELTA_METRICS.values()), rotation=42, ha="right")
        axis.set_yticks(range(len(matrix)), matrix.index)
        axis.set_title(str(target).replace("_", " "))
        for y in range(len(matrix)):
            for x in range(len(columns)):
                value = matrix.iloc[y, x]
                if pd.notna(value):
                    color = "white" if abs(value) > normalizer.vmax * 0.62 else "#111827"
                    axis.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=6.0, color=color)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#9CA3AF")
            spine.set_linewidth(0.5)
    bar = fig.colorbar(image, ax=axes, shrink=0.76, pad=0.02)
    bar.set_label("Matched Δ MAE (min; blue = improvement)")
    fig.suptitle(f"{model_label}: stratum-level effects of the leading candidates", fontsize=9.5)
    fig.subplots_adjust(left=0.20, right=0.86, top=0.90, bottom=0.10, hspace=0.70)
    save_figure(fig, output / "figures/stratum_effects_top_candidates")
    return selected


def figure_seed_spread(deltas: pd.DataFrame, selected: pd.DataFrame, output: Path, model_label: str) -> None:
    """Retain every seed for shortlisted configurations rather than hiding spread."""
    if deltas.empty or selected.empty:
        return
    keys = ["model", "arm", "target", "policy", "profile", "loss", "lr", "weights", "oversampling"]
    selected_keys = selected[keys].drop_duplicates()
    frame = deltas.merge(selected_keys, on=keys, how="inner")
    targets = list(frame["target"].dropna().unique())
    metrics = ["delta_heavy_macro_source_balanced_mae", "delta_nonheavy_sb_mae"]
    if frame.empty or not targets:
        return
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(7.1, 2.9 * len(targets)), squeeze=False)
    rng = np.random.default_rng(20260823)
    for row_index, target in enumerate(targets):
        panel = frame[frame["target"] == target].copy()
        order = (
            panel.groupby(["arm", "profile", "lr", "weights", "oversampling"], dropna=False)[metrics[0]]
            .mean().sort_values().index.tolist()
        )
        panel["config"] = list(zip(panel["arm"], panel["profile"], panel["lr"], panel["weights"], panel["oversampling"]))
        labels = [f"{ARM_LABELS.get(key[0], key[0])}: {key[1]} · {key[2]:.0e}" for key in order]
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            for y, configuration in enumerate(order):
                part = panel[panel["config"] == configuration]
                values = part[metric].dropna().to_numpy(dtype=float)
                color = ARM_COLORS.get(configuration[0], "#4B5563")
                axis.scatter(values, np.full(len(values), y) + rng.uniform(-0.08, 0.08, len(values)),
                             s=19, color=color, alpha=0.78, edgecolor="white", linewidth=0.35, zorder=3)
                if len(values):
                    sd = values.std(ddof=1) if len(values) > 1 else 0.0
                    axis.errorbar(values.mean(), y, xerr=sd, fmt="o", color="#111827", markersize=3.2,
                                  capsize=2, linewidth=0.9, zorder=4)
            axis.axvline(0, color="#6B7280", linewidth=0.7)
            axis.set_yticks(range(len(order)), labels if column_index == 0 else [])
            axis.set_xlabel("Matched Δ MAE (min)")
            axis.set_title(DELTA_METRICS[metric])
            axis.grid(axis="x", color="#E5E7EB", linewidth=0.5)
        axes[row_index, 0].set_ylabel(str(target).replace("_", " "))
    fig.suptitle(f"{model_label}: seed-level variation for shortlisted configurations", fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, output / "figures/seed_spread_top_candidates")


def figure_mechanism_distributions(summary: pd.DataFrame, output: Path, model_label: str) -> None:
    """Descriptive distribution of configuration-mean effects by mechanism."""
    metrics = ["delta_heavy_macro_source_balanced_mae", "delta_nonheavy_sb_mae"]
    frame = summary.dropna(subset=metrics).copy()
    targets = list(frame["target"].dropna().unique())
    arms = [arm for arm in ("weighting", "oversampling") if arm in set(frame["arm"])]
    if frame.empty or not targets or not arms:
        return
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(7.1, 2.85 * len(targets)), squeeze=False)
    rng = np.random.default_rng(20260824)
    for row_index, target in enumerate(targets):
        panel = frame[frame["target"] == target]
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            groups = [panel.loc[panel["arm"] == arm, metric].dropna().to_numpy(dtype=float) for arm in arms]
            boxes = axis.boxplot(groups, labels=[ARM_LABELS.get(arm, arm) for arm in arms], patch_artist=True, widths=0.58)
            for patch, arm in zip(boxes["boxes"], arms):
                patch.set_facecolor(ARM_COLORS.get(arm, "#4B5563")); patch.set_alpha(0.32)
            for position, (arm, values) in enumerate(zip(arms, groups), start=1):
                axis.scatter(np.full(len(values), position) + rng.uniform(-0.11, 0.11, len(values)), values,
                             s=10, color=ARM_COLORS.get(arm, "#4B5563"), alpha=0.55, linewidth=0, zorder=3)
            axis.axhline(0, color="#6B7280", linewidth=0.7)
            axis.set_title(DELTA_METRICS[metric])
            axis.set_ylabel(f"{str(target).replace('_', ' ')}\nΔ MAE (min)")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    fig.suptitle(f"{model_label}: descriptive effect distributions across tested configurations", fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, output / "figures/mechanism_effect_distributions")


def organism_outcomes(organism_deltas: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-matched organism effects into the three study outcomes."""
    required = set(CONFIGURATION_KEYS + ["organism", "stratum", "delta_mae"])
    if organism_deltas.empty or not required.issubset(organism_deltas):
        return pd.DataFrame()
    by_stratum = (
        organism_deltas.groupby(CONFIGURATION_KEYS + ["organism", "stratum"], dropna=False)["delta_mae"]
        .mean().reset_index()
    )
    pivot = by_stratum.pivot_table(
        index=CONFIGURATION_KEYS + ["organism"], columns="stratum", values="delta_mae", aggfunc="first",
    ).reset_index()
    heavy_columns = [column for column in ("heavy_nterm", "heavy_lys", "heavy_both") if column in pivot]
    pivot["heavy_macro"] = pivot[heavy_columns].mean(axis=1) if heavy_columns else np.nan
    outcome_columns = [column for column in ("overall", "heavy_macro", "semitryptic") if column in pivot]
    return pivot.melt(
        id_vars=CONFIGURATION_KEYS + ["organism"], value_vars=outcome_columns,
        var_name="outcome", value_name="delta_mae",
    ).dropna(subset=["delta_mae"])


def figure_organism_effect_distributions(outcomes: pd.DataFrame, output: Path, model_label: str) -> None:
    """Show whether effects are broad, organism-specific, or mechanism-specific."""
    if outcomes.empty:
        return
    targets = list(outcomes["target"].dropna().unique())
    outcomes_order = [item for item in ("overall", "heavy_macro", "semitryptic") if item in set(outcomes["outcome"])]
    organisms = [item for item in ("celegans", "human", "triplemix") if item in set(outcomes["organism"])]
    arms = [item for item in ("weighting", "oversampling") if item in set(outcomes["arm"])]
    if not targets or not outcomes_order or not organisms or not arms:
        return
    fig, axes = plt.subplots(
        len(targets), len(outcomes_order), figsize=(3.25 * len(outcomes_order), 2.95 * len(targets)),
        squeeze=False, constrained_layout=True,
    )
    rng = np.random.default_rng(20260825)
    for row_index, target in enumerate(targets):
        for column_index, outcome in enumerate(outcomes_order):
            axis = axes[row_index, column_index]
            panel = outcomes[(outcomes["target"] == target) & (outcomes["outcome"] == outcome)]
            positions: list[float] = []
            values: list[np.ndarray] = []
            colors: list[str] = []
            for organism_index, organism in enumerate(organisms):
                for arm_index, arm in enumerate(arms):
                    positions.append(organism_index + (arm_index - (len(arms) - 1) / 2) * 0.28)
                    values.append(panel.loc[(panel["organism"] == organism) & (panel["arm"] == arm), "delta_mae"].to_numpy(float))
                    colors.append(ARM_COLORS.get(arm, "#4B5563"))
            boxes = axis.boxplot(values, positions=positions, widths=0.23, patch_artist=True, manage_ticks=False)
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.30)
            for position, values_for_box, color in zip(positions, values, colors):
                axis.scatter(
                    np.full(len(values_for_box), position) + rng.uniform(-0.045, 0.045, len(values_for_box)),
                    values_for_box, s=9, color=color, alpha=0.52, linewidth=0, zorder=3,
                )
            axis.axhline(0, color="#6B7280", linewidth=0.7)
            axis.set_xticks(range(len(organisms)), ["C. elegans" if item == "celegans" else item.title() for item in organisms])
            axis.set_title({"overall": "Overall", "heavy_macro": "Heavy macro", "semitryptic": "Semi-tryptic"}[outcome])
            axis.set_ylabel(f"{str(target).replace('_', ' ')}\nMatched Δ MAE (min)" if column_index == 0 else "")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=ARM_COLORS[arm], label=ARM_LABELS[arm], markersize=5)
               for arm in arms]
    axes[0, -1].legend(handles=handles, title="Intervention", frameon=False, loc="upper right")
    fig.suptitle(f"{model_label}: organism-specific effects across every tested configuration", fontsize=9.5)
    save_figure(fig, output / "figures/organism_effect_distributions")


def figure_organism_outcomes_top_candidates(
    outcomes: pd.DataFrame, selected: pd.DataFrame, output: Path, model_label: str,
) -> None:
    """Expose the organism-specific consequences of globally competitive choices."""
    if outcomes.empty or selected.empty:
        return
    selected_keys = selected[CONFIGURATION_KEYS].drop_duplicates()
    frame = outcomes.merge(selected_keys, on=CONFIGURATION_KEYS, how="inner")
    organisms = [item for item in ("celegans", "human", "triplemix") if item in set(frame["organism"])]
    outcomes_order = [item for item in ("overall", "heavy_macro", "semitryptic") if item in set(frame["outcome"])]
    targets = list(frame["target"].dropna().unique())
    if frame.empty or not organisms or not outcomes_order or not targets:
        return
    frame["organism_outcome"] = frame["organism"].map(
        {"celegans": "C. elegans", "human": "Human", "triplemix": "TripleMix"}
    ).fillna(frame["organism"]) + "\n" + frame["outcome"].map(
        {"overall": "Overall", "heavy_macro": "Heavy macro", "semitryptic": "Semi-tryptic"}
    )
    column_order = [
        f"{ {'celegans': 'C. elegans', 'human': 'Human', 'triplemix': 'TripleMix'}.get(organism, organism) }\n"
        f"{ {'overall': 'Overall', 'heavy_macro': 'Heavy macro', 'semitryptic': 'Semi-tryptic'}[outcome] }"
        for organism in organisms for outcome in outcomes_order
    ]
    normalizer = _delta_norm(frame.rename(columns={"delta_mae": "plot_delta"}), ["plot_delta"])
    fig, axes = plt.subplots(len(targets), 1, figsize=(9.7, 3.75 * len(targets)), squeeze=False)
    axes = axes[:, 0]
    for axis, target in zip(axes, targets):
        panel = frame[frame["target"] == target].copy()
        labels = selected[selected["target"] == target].copy()
        labels["label"] = labels.apply(_configuration_label, axis=1)
        ordering = labels["label"].tolist()
        panel["label"] = panel.apply(_configuration_label, axis=1)
        matrix = panel.pivot_table(index="label", columns="organism_outcome", values="delta_mae", aggfunc="first")
        matrix = matrix.reindex(index=ordering, columns=column_order)
        image = axis.imshow(matrix, aspect="auto", cmap=DELTA_CMAP, norm=normalizer)
        axis.set_xticks(range(len(column_order)), column_order, rotation=36, ha="right")
        axis.set_yticks(range(len(matrix)), matrix.index)
        axis.set_title(str(target).replace("_", " "))
        for y in range(len(matrix)):
            for x in range(len(matrix.columns)):
                value = matrix.iloc[y, x]
                if pd.notna(value):
                    color = "white" if abs(value) > normalizer.vmax * 0.62 else "#111827"
                    axis.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=6.0, color=color)
    bar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    bar.set_label("Matched Δ MAE (min; blue = improvement)")
    fig.suptitle(f"{model_label}: organism outcomes for the globally leading candidates", fontsize=9.5)
    fig.subplots_adjust(left=0.17, right=0.79, top=0.90, bottom=0.10, hspace=0.68)
    save_figure(fig, output / "figures/organism_outcomes_top_candidates")


def write_plot_guide(output: Path, model_label: str) -> None:
    text = f"""# {model_label} Heavy-Dimethyl sweep plots

All plots use validation results only. A negative matched delta means lower MAE
than the seed-, target-, loss- and learning-rate-matched control. Blue in the
heatmaps denotes an improvement; red denotes a cost.

| File | Question answered |
|---|---|
| `heavy_benefit_vs_nonheavy_cost.*` | Which configurations improve heavy peptides without harming non-heavy peptides? Lines are lower-left Pareto fronts within each intervention arm. |
| `figures/delta_heavy_macro_source_balanced_mae_landscape.*` | How do every tested profile and learning rate affect the primary Heavy-Macro metric? |
| `figures/delta_nonheavy_sb_mae_landscape.*` | Where is the corresponding cost or benefit for non-heavy peptides? |
| `figures/stratum_effects_top_candidates.*` | Which exact Heavy-Dimethyl stratum drives the apparent gain, and what happens to the guardrails? |
| `figures/seed_spread_top_candidates.*` | Are leading effects stable across the individual seeds? Black points/bars are mean ± SD. |
| `figures/mechanism_effect_distributions.*` | What is the broad, descriptive difference between weighting and oversampling across the entire tested grid? |
| `figures/organism_effect_distributions.*` | Do weighting or oversampling help a particular organism for Overall, Heavy-Macro or Semi-tryptic MAE? Every dot is one configuration mean across its seeds. |
| `figures/organism_outcomes_top_candidates.*` | What are the organism-specific costs and benefits of the globally leading candidates? |

The same script and figure contract are used for DIA-NN and Carafe2. Do not use
these validation plots as a final test result: freeze a shortlist before
reading the held-out test partition.
"""
    (output / "PLOTS.md").write_text(text, encoding="utf-8")


def generate_heavy_dimethyl_figures(
    summary: pd.DataFrame, deltas: pd.DataFrame, output: Path, model_label: str,
    organism_deltas: pd.DataFrame | None = None,
) -> None:
    """Export the standard Heavy-Dimethyl figure suite for either model family."""
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    figure_tradeoff_frontiers(summary, output, model_label)
    figure_effect_landscape(
        summary, output, "delta_heavy_macro_source_balanced_mae",
        f"{model_label}: Heavy-Macro response across the complete sweep",
    )
    figure_effect_landscape(
        summary, output, "delta_nonheavy_sb_mae",
        f"{model_label}: Non-heavy response across the complete sweep",
    )
    selected = figure_stratum_effects(summary, output, model_label)
    figure_seed_spread(deltas, selected, output, model_label)
    figure_mechanism_distributions(summary, output, model_label)
    if organism_deltas is not None:
        outcomes = organism_outcomes(organism_deltas)
        figure_organism_effect_distributions(outcomes, output, model_label)
        figure_organism_outcomes_top_candidates(outcomes, selected, output, model_label)
    write_plot_guide(output, model_label)


def main() -> None:
    """Re-render figures from an existing Heavy-Dimethyl analysis directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", type=Path, required=True,
        help="Directory containing weighting_vs_oversampling.csv and matched_control_deltas.csv",
    )
    parser.add_argument("--model-label", help="Display name; inferred from the analysis table when omitted")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    summary_path = analysis_dir / "weighting_vs_oversampling.csv"
    deltas_path = analysis_dir / "matched_control_deltas.csv"
    if not summary_path.is_file() or not deltas_path.is_file():
        raise SystemExit(
            "Expected weighting_vs_oversampling.csv and matched_control_deltas.csv "
            f"in {analysis_dir}"
        )
    summary = pd.read_csv(summary_path)
    deltas = pd.read_csv(deltas_path)
    organism_deltas_path = analysis_dir / "matched_organism_stratum_deltas.csv"
    organism_deltas = pd.read_csv(organism_deltas_path) if organism_deltas_path.is_file() else None
    inferred = "RT model"
    if "model" in summary and summary["model"].notna().any():
        inferred = str(summary.loc[summary["model"].notna(), "model"].iloc[0])
    model_label = args.model_label or {"diann": "DIA-NN", "carafe2": "Carafe2"}.get(inferred.lower(), inferred)
    generate_heavy_dimethyl_figures(summary, deltas, analysis_dir, model_label, organism_deltas)
    print(f"Wrote Heavy-Dimethyl figures to {analysis_dir}")


if __name__ == "__main__":
    main()
