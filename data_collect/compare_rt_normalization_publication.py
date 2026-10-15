#!/usr/bin/env python3
"""Compare completed DIA-NN and Carafe2 RT-normalization studies.

The analysis is intentionally strict: both studies must contain every expected
method, seed, condition and organism test, and their paired evaluation rows and
training compositions must agree. Identical publication exports are written to
both study roots under ``publication_analysis/model_comparison``.

Primary metric
--------------
``raw_mae_minutes`` after method-specific inverse transformation to observed LC
minutes. Carafe2-minus-DIA-NN differences are paired by seed, condition,
normalization method and test organism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from analyze_rt_normalization_publication import (
    ADDITION_CONDITIONS,
    BRANCHES,
    CONDITIONS,
    CONDITION_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_SHORT,
    METHODS,
    ORGANISM_LABELS,
    ORGANISMS,
    collect_runs,
    collect_training_composition,
    mean_total_training_rows,
    panel_label,
    t_interval,
    validate_runs,
)


MODEL_ORDER = ("diann", "carafe2")
EXPECTED_SEEDS = (2024, 2025, 2026, 2027, 2028)
MODEL_LABELS = {"diann": "DIA-NN", "carafe2": "Carafe2"}
MODEL_COLORS = {"diann": "#0072B2", "carafe2": "#D55E00"}
MODEL_MARKERS = {"diann": "o", "carafe2": "s"}
MODEL_LINESTYLES = {"diann": "-", "carafe2": "--"}
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "carafe_minus_diann", ["#0072B2", "#F7F7F7", "#D55E00"]
)

# Compact chart contract: static Matplotlib exports because these figures are
# intended for manuscript assembly. Every chart uses observed-minute MAE,
# paired seeds, explicit model markers/line styles, quiet scaffolding, and
# PNG/PDF/SVG delivery to both study trees.
CHART_CONTRACTS = {
    "fig1": "Faceted dot-and-interval: overall absolute MAE and paired model difference.",
    "fig2": "Three-panel heatmap: Carafe2 minus DIA-NN over every method and condition.",
    "fig3": "Faceted dumbbell: full mixed-data endpoint by method and test organism.",
    "fig4": "Three-panel heatmap: model difference in within-model data-addition gains.",
    "fig5": "Nine-panel line chart: actual training-row scaling for pooled top methods.",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diann-root", type=Path, required=True)
    parser.add_argument("--carafe2-root", type=Path, required=True)
    parser.add_argument(
        "--output-subdir",
        type=Path,
        default=Path("publication_analysis/model_comparison"),
        help="Relative output path created below both study roots",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_study(root: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = collect_runs(root)
    audit = validate_runs(data, allow_incomplete=False)
    observed_seeds = tuple(sorted(int(seed) for seed in data.seed.unique()))
    if observed_seeds != EXPECTED_SEEDS:
        raise SystemExit(
            f"{MODEL_LABELS[model]} has seeds {observed_seeds}; expected {EXPECTED_SEEDS}"
        )
    expected_records = len(METHODS) * len(EXPECTED_SEEDS) * len(CONDITIONS) * len(ORGANISMS)
    if len(data) != expected_records:
        raise SystemExit(
            f"{MODEL_LABELS[model]} has {len(data)} organism-level records; "
            f"expected {expected_records}"
        )
    composition = collect_training_composition(root, data)
    if composition.empty:
        raise SystemExit(f"Training composition could not be read below {root}")
    data = data.copy()
    data["model"] = model
    data["model_label"] = MODEL_LABELS[model]
    audit = audit.copy()
    audit["model"] = model
    composition = composition.copy()
    composition["model"] = model
    return data, audit, composition


def validate_pairing(
    diann: pd.DataFrame,
    carafe2: pd.DataFrame,
    diann_composition: pd.DataFrame,
    carafe2_composition: pd.DataFrame,
) -> None:
    keys = ["method", "seed", "condition", "test_organism"]
    left_keys = pd.MultiIndex.from_frame(diann[keys]).sort_values()
    right_keys = pd.MultiIndex.from_frame(carafe2[keys]).sort_values()
    if not left_keys.equals(right_keys):
        left_set, right_set = set(left_keys), set(right_keys)
        raise SystemExit(
            "DIA-NN and Carafe2 run keys differ: "
            f"{len(left_set - right_set)} only in DIA-NN, "
            f"{len(right_set - left_set)} only in Carafe2"
        )

    population = diann[keys + ["rows", "unique_sequences"]].merge(
        carafe2[keys + ["rows", "unique_sequences"]],
        on=keys,
        suffixes=("_diann", "_carafe2"),
        validate="one_to_one",
    )
    for column in ("rows", "unique_sequences"):
        left = population[f"{column}_diann"].to_numpy()
        right = population[f"{column}_carafe2"].to_numpy()
        if not np.allclose(left, right, equal_nan=True):
            raise SystemExit(f"Evaluation populations differ between models in {column}")

    comp_keys = ["seed", "condition", "source_organism"]
    composition = diann_composition[comp_keys + ["training_rows"]].merge(
        carafe2_composition[comp_keys + ["training_rows"]],
        on=comp_keys,
        how="outer",
        suffixes=("_diann", "_carafe2"),
        indicator=True,
        validate="one_to_one",
    )
    if not composition._merge.eq("both").all():
        raise SystemExit("Training-data source composition keys differ between models")
    if not np.array_equal(composition.training_rows_diann, composition.training_rows_carafe2):
        raise SystemExit("Training-data row counts differ between models")


def summarize_by_condition(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in data.groupby(
        ["model", "method", "condition", "condition_order", "test_organism"]
    ):
        model, method, condition, order, organism = keys
        mean, sd, low, high, n = t_interval(group.raw_mae_minutes)
        records.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "condition": condition,
                "condition_order": order,
                "test_organism": organism,
                "mean_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
            }
        )
    return pd.DataFrame(records)


def overall_comparison(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_means = (
        data.groupby(["model", "method", "seed"], as_index=False).raw_mae_minutes.mean()
        .rename(columns={"raw_mae_minutes": "seed_mean_mae_minutes"})
    )
    summaries = []
    for keys, group in seed_means.groupby(["model", "method"]):
        model, method = keys
        mean, sd, low, high, n = t_interval(group.seed_mean_mae_minutes)
        summaries.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "mean_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
            }
        )
    summary = pd.DataFrame(summaries)
    pooled_order = (
        summary.groupby("method").mean_mae_minutes.mean().sort_values().index.tolist()
    )
    order_lookup = {method: rank for rank, method in enumerate(pooled_order, start=1)}
    summary["pooled_rank"] = summary.method.map(order_lookup)

    paired = seed_means.pivot(index=["method", "seed"], columns="model", values="seed_mean_mae_minutes").reset_index()
    paired["carafe2_minus_diann_minutes"] = paired.carafe2 - paired.diann
    differences = []
    for method, group in paired.groupby("method"):
        mean, sd, low, high, n = t_interval(group.carafe2_minus_diann_minutes)
        differences.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "pooled_rank": order_lookup[method],
                "mean_carafe2_minus_diann_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
            }
        )
    return summary, seed_means, pd.DataFrame(differences)


def paired_model_effects(diann: pd.DataFrame, carafe2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["method", "seed", "condition", "condition_order", "test_organism"]
    paired = diann[keys + ["raw_mae_minutes"]].merge(
        carafe2[keys + ["raw_mae_minutes"]],
        on=keys,
        suffixes=("_diann", "_carafe2"),
        validate="one_to_one",
    )
    paired["carafe2_minus_diann_minutes"] = paired.raw_mae_minutes_carafe2 - paired.raw_mae_minutes_diann
    records = []
    for keys_value, group in paired.groupby(
        ["method", "condition", "condition_order", "test_organism"]
    ):
        method, condition, order, organism = keys_value
        mean, sd, low, high, n = t_interval(group.carafe2_minus_diann_minutes)
        records.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "condition": condition,
                "condition_order": order,
                "test_organism": organism,
                "mean_carafe2_minus_diann_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(records), paired


def within_model_addition_effects(data: pd.DataFrame) -> pd.DataFrame:
    base = data[data.condition.eq("ce_100")][
        ["model", "method", "seed", "test_organism", "raw_mae_minutes"]
    ].rename(columns={"raw_mae_minutes": "ce100_mae_minutes"})
    additions = data[data.condition.isin(ADDITION_CONDITIONS)].merge(
        base,
        on=["model", "method", "seed", "test_organism"],
        validate="many_to_one",
    )
    additions["addition_delta_mae_minutes"] = additions.raw_mae_minutes - additions.ce100_mae_minutes
    return additions


def difference_in_differences(additions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["method", "seed", "condition", "condition_order", "test_organism"]
    paired = additions[keys + ["model", "addition_delta_mae_minutes"]].pivot(
        index=keys, columns="model", values="addition_delta_mae_minutes"
    ).reset_index()
    paired["carafe2_minus_diann_addition_effect_minutes"] = paired.carafe2 - paired.diann
    records = []
    for key_values, group in paired.groupby(
        ["method", "condition", "condition_order", "test_organism"]
    ):
        method, condition, order, organism = key_values
        mean, sd, low, high, n = t_interval(group.carafe2_minus_diann_addition_effect_minutes)
        records.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "condition": condition,
                "condition_order": order,
                "test_organism": organism,
                "mean_difference_in_differences_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(records), paired


def save_figure(fig: plt.Figure, name: str, outputs: tuple[Path, Path], dpi: int) -> None:
    for output in outputs:
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(
                (output / name).with_suffix(suffix),
                dpi=dpi if suffix == ".png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
    plt.close(fig)


def write_csv(frame: pd.DataFrame, name: str, outputs: tuple[Path, Path]) -> None:
    for output in outputs:
        frame.to_csv(output / name, index=False)


def figure_overall(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    outputs: tuple[Path, Path],
    dpi: int,
) -> None:
    method_order = (
        summary[["method", "pooled_rank"]].drop_duplicates().sort_values("pooled_rank", ascending=False).method.tolist()
    )
    y_lookup = {method: index for index, method in enumerate(method_order)}
    fig, (absolute, delta) = plt.subplots(
        1, 2, figsize=(12.6, 6.1), gridspec_kw={"width_ratios": [1.15, 1]}, constrained_layout=True
    )
    for method in method_order:
        rows = summary[summary.method.eq(method)].set_index("model")
        y = y_lookup[method]
        absolute.plot(
            [rows.loc["diann", "mean_mae_minutes"], rows.loc["carafe2", "mean_mae_minutes"]],
            [y, y],
            color="#C8CDD2",
            linewidth=1.2,
            zorder=1,
        )
        for model in MODEL_ORDER:
            row = rows.loc[model]
            absolute.errorbar(
                row.mean_mae_minutes,
                y,
                xerr=[[row.mean_mae_minutes - row.ci_low], [row.ci_high - row.mean_mae_minutes]],
                fmt=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=6,
                capsize=2.5,
                linewidth=1.2,
                label=MODEL_LABELS[model] if method == method_order[0] else None,
                zorder=3,
            )
    absolute.set_yticks(range(len(method_order)), [METHOD_LABELS[item] for item in method_order])
    absolute.set_xlabel("Mean MAE in observed RT (min)\n14 conditions × 3 organism tests; equal block weight")
    absolute.set_title("Overall absolute performance", loc="left", fontweight="bold")
    absolute.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    absolute.legend(frameon=False, loc="lower right")
    panel_label(absolute, "a")

    diff = differences.set_index("method").loc[method_order].reset_index()
    delta.axvline(0, color="#444444", linewidth=0.9)
    for y, row in diff.iterrows():
        color = MODEL_COLORS["carafe2"] if row.mean_carafe2_minus_diann_minutes > 0 else MODEL_COLORS["diann"]
        delta.errorbar(
            row.mean_carafe2_minus_diann_minutes,
            y,
            xerr=[
                [row.mean_carafe2_minus_diann_minutes - row.ci_low],
                [row.ci_high - row.mean_carafe2_minus_diann_minutes],
            ],
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            capsize=2.5,
            linewidth=1.2,
        )
    delta.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
    delta.set_xlabel("Carafe2 − DIA-NN MAE (min)\nnegative favors Carafe2; paired over seeds")
    delta.set_title("Paired model difference", loc="left", fontweight="bold")
    delta.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    panel_label(delta, "b")
    save_figure(fig, "comparison_fig1_overall_performance", outputs, dpi)


def annotated_heatmaps(
    table: pd.DataFrame,
    value_column: str,
    conditions: tuple[str, ...],
    method_order: list[str],
    composition: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_name: str,
    outputs: tuple[Path, Path],
    dpi: int,
) -> None:
    totals = mean_total_training_rows(composition) / 1000
    labels = [f"{CONDITION_LABELS[item]}\n{totals.get(item, np.nan):.1f}k" for item in conditions]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.6), constrained_layout=True)
    for panel, (axis, organism) in enumerate(zip(axes, ORGANISMS)):
        matrix = (
            table[table.test_organism.eq(organism)]
            .pivot(index="method", columns="condition", values=value_column)
            .reindex(index=method_order, columns=conditions)
        )
        finite = np.abs(matrix.to_numpy()[np.isfinite(matrix.to_numpy())])
        limit = max(float(np.max(finite)) if finite.size else 0.1, 0.1)
        image = axis.imshow(
            matrix,
            cmap=DELTA_CMAP,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            aspect="auto",
        )
        axis.set_xticks(range(len(conditions)), labels, rotation=58, ha="right")
        axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
        axis.set_title(ORGANISM_LABELS[organism], fontweight="bold")
        axis.tick_params(length=0)
        if len(conditions) == 14:
            for boundary in (4.5, 7.5, 10.5):
                axis.axvline(boundary, color="white", linewidth=2.0)
        elif len(conditions) == 9:
            for boundary in (2.5, 5.5):
                axis.axvline(boundary, color="white", linewidth=2.0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                label = "NA" if not np.isfinite(value) else f"{value:+.2f}"
                color = "white" if np.isfinite(value) and abs(value) > 0.52 * limit else "#202020"
                axis.text(j, i, label, ha="center", va="center", fontsize=5.8, color=color)
        colorbar = fig.colorbar(image, ax=axis, shrink=0.72, pad=0.015)
        colorbar.set_label(colorbar_label, fontsize=8)
        panel_label(axis, chr(ord("a") + panel))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    save_figure(fig, output_name, outputs, dpi)


def figure_endpoint(
    summary: pd.DataFrame,
    pooled_order: list[str],
    outputs: tuple[Path, Path],
    dpi: int,
) -> None:
    endpoint = summary[summary.condition.eq("ce100_all_100")]
    method_order = list(reversed(pooled_order))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.0), constrained_layout=True)
    for panel, (axis, organism) in enumerate(zip(axes, ORGANISMS)):
        subset = endpoint[endpoint.test_organism.eq(organism)]
        for y, method in enumerate(method_order):
            rows = subset[subset.method.eq(method)].set_index("model")
            axis.plot(
                [rows.loc["diann", "mean_mae_minutes"], rows.loc["carafe2", "mean_mae_minutes"]],
                [y, y],
                color="#C8CDD2",
                linewidth=1.2,
                zorder=1,
            )
            for model in MODEL_ORDER:
                row = rows.loc[model]
                axis.errorbar(
                    row.mean_mae_minutes,
                    y,
                    xerr=[[row.mean_mae_minutes - row.ci_low], [row.ci_high - row.mean_mae_minutes]],
                    fmt=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model],
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    markersize=5.5,
                    capsize=2.2,
                    linewidth=1.1,
                    label=MODEL_LABELS[model] if y == 0 and panel == 0 else None,
                    zorder=3,
                )
        axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
        axis.set_xlabel("MAE in observed RT (min)")
        axis.set_title(ORGANISM_LABELS[organism], fontweight="bold")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        panel_label(axis, chr(ord("a") + panel))
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Full mixed-data endpoint: DIA-NN versus Carafe2", fontsize=12, fontweight="bold")
    save_figure(fig, "comparison_fig3_full_mixture_endpoint", outputs, dpi)


def figure_scaling(
    summary: pd.DataFrame,
    pooled_order: list[str],
    composition: pd.DataFrame,
    outputs: tuple[Path, Path],
    dpi: int,
) -> None:
    selected = pooled_order[:3]
    totals = mean_total_training_rows(composition)
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 10.0), sharey="row", constrained_layout=True)
    for row, organism in enumerate(ORGANISMS):
        for col, (branch_label, conditions) in enumerate(BRANCHES.items()):
            axis = axes[row, col]
            x = np.array([totals.loc[condition] / 1000 for condition in conditions])
            for method in selected:
                for model in MODEL_ORDER:
                    group = (
                        summary[
                            summary.model.eq(model)
                            & summary.method.eq(method)
                            & summary.test_organism.eq(organism)
                            & summary.condition.isin(conditions)
                        ]
                        .set_index("condition")
                        .reindex(conditions)
                    )
                    label = f"{MODEL_LABELS[model]} · {METHOD_SHORT[method]}"
                    axis.plot(
                        x,
                        group.mean_mae_minutes,
                        color=METHOD_COLORS[method],
                        linestyle=MODEL_LINESTYLES[model],
                        marker=MODEL_MARKERS[model],
                        markerfacecolor="white" if model == "carafe2" else METHOD_COLORS[method],
                        markeredgecolor=METHOD_COLORS[method],
                        markersize=4,
                        linewidth=1.5,
                        label=label,
                    )
                    axis.fill_between(
                        x,
                        group.ci_low,
                        group.ci_high,
                        color=METHOD_COLORS[method],
                        alpha=0.07,
                        linewidth=0,
                    )
            if row == 0:
                axis.set_title(branch_label, fontweight="bold")
            if col == 0:
                axis.set_ylabel(f"{ORGANISM_LABELS[organism]} test\nMAE (min)")
            if row == 2:
                axis.set_xlabel("Training rows (thousands)")
            axis.grid(color="#D9D9D9", linewidth=0.55)
            if row == 0:
                panel_label(axis, chr(ord("a") + col))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    save_figure(fig, "comparison_fig5_top_method_scaling", outputs, dpi)


def write_markdown_report(
    outputs: tuple[Path, Path],
    roots: dict[str, Path],
    overall: pd.DataFrame,
    differences: pd.DataFrame,
    did: pd.DataFrame,
) -> dict[str, object]:
    ranking = overall.sort_values(["model", "mean_mae_minutes"])
    best = ranking.groupby("model", as_index=False).first().set_index("model")
    strongest = differences.iloc[differences.mean_carafe2_minus_diann_minutes.abs().argmax()]
    full_did = did[did.condition.eq("ce100_all_100")]
    report = f"""# DIA-NN versus Carafe2: RT-normalization comparison

## Scope

- Primary metric: MAE after inverse transformation to observed LC minutes.
- Pairing: identical normalization method, seed, training condition and test organism.
- Overall means: equal weight for all 14 conditions and three organism tests within each seed.
- Uncertainty: two-sided 95% Student-t intervals across the five paired seeds.
- Signed model contrast: `Carafe2 - DIA-NN`; negative values favor Carafe2.
- Data-addition difference-in-differences: `(Carafe2 added-data - Carafe2 CE100) - (DIA-NN added-data - DIA-NN CE100)`; negative values mean Carafe2 benefits more from the added data.

## Descriptive summary

- DIA-NN best overall method: **{METHOD_LABELS[best.loc['diann', 'method']]}**, {best.loc['diann', 'mean_mae_minutes']:.3f} min MAE.
- Carafe2 best overall method: **{METHOD_LABELS[best.loc['carafe2', 'method']]}**, {best.loc['carafe2', 'mean_mae_minutes']:.3f} min MAE.
- Largest absolute overall paired model difference: **{METHOD_LABELS[strongest.method]}**, {strongest.mean_carafe2_minus_diann_minutes:+.3f} min Carafe2 minus DIA-NN.

These are benchmark descriptions, not claims that one architecture is universally superior. Both models share source datasets and split seeds, and the five seeds are not independent biological replicates.

## Figures

1. ![Overall performance](comparison_fig1_overall_performance.png)
2. ![Condition-wise model differences](comparison_fig2_condition_model_differences.png)
3. ![Full mixed endpoint](comparison_fig3_full_mixture_endpoint.png)
4. ![Difference in data-addition effects](comparison_fig4_data_addition_difference_in_differences.png)
5. ![Top-method scaling](comparison_fig5_top_method_scaling.png)

## Interpretation guide

- Figures 1 and 3 show absolute MAE and should drive practical model selection.
- Figure 2 shows where model differences arise across data quantity and source composition.
- Figure 4 isolates which model gains more from adding foreign-organism training data; it should not be read as absolute accuracy.
- Figure 5 restricts curves to the three best methods after pooling the two model-level means to keep the visual readable.

## Inputs

- DIA-NN: `{roots['diann']}`
- Carafe2: `{roots['carafe2']}`

Exact run-level values, paired contrasts, confidence intervals and completeness audits are stored as CSV files next to this report.
"""
    for output in outputs:
        (output / "model_comparison_report.md").write_text(report, encoding="utf-8")
    return {
        "best_diann_method": str(best.loc["diann", "method"]),
        "best_carafe2_method": str(best.loc["carafe2", "method"]),
        "largest_absolute_overall_model_difference_method": str(strongest.method),
        "largest_absolute_overall_model_difference_minutes": float(
            strongest.mean_carafe2_minus_diann_minutes
        ),
        "full_endpoint_difference_in_differences_rows": int(len(full_did)),
    }


def main() -> None:
    args = parse_args()
    diann_root = args.diann_root.resolve()
    carafe2_root = args.carafe2_root.resolve()
    if args.output_subdir.is_absolute() or ".." in args.output_subdir.parts:
        raise SystemExit("--output-subdir must be a safe relative path")
    outputs = (
        diann_root / args.output_subdir,
        carafe2_root / args.output_subdir,
    )
    diann, diann_audit, diann_composition = load_study(diann_root, "diann")
    carafe2, carafe2_audit, carafe2_composition = load_study(carafe2_root, "carafe2")
    validate_pairing(diann, carafe2, diann_composition, carafe2_composition)
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)

    data = pd.concat([diann, carafe2], ignore_index=True)
    audit = pd.concat([diann_audit, carafe2_audit], ignore_index=True)
    composition = pd.concat([diann_composition, carafe2_composition], ignore_index=True)
    summary = summarize_by_condition(data)
    overall, overall_seed_values, overall_differences = overall_comparison(data)
    model_effects, paired_model_values = paired_model_effects(diann, carafe2)
    additions = within_model_addition_effects(data)
    did, paired_did_values = difference_in_differences(additions)
    pooled_order = (
        overall[["method", "pooled_rank"]].drop_duplicates().sort_values("pooled_rank").method.tolist()
    )

    write_csv(data, "comparison_run_level_metrics.csv", outputs)
    write_csv(audit, "comparison_completeness_audit.csv", outputs)
    write_csv(composition, "comparison_training_data_composition.csv", outputs)
    write_csv(summary, "comparison_condition_summary.csv", outputs)
    write_csv(overall, "comparison_overall_method_summary.csv", outputs)
    write_csv(overall_seed_values, "comparison_overall_seed_values.csv", outputs)
    write_csv(overall_differences, "comparison_overall_paired_model_differences.csv", outputs)
    write_csv(model_effects, "comparison_condition_paired_model_differences.csv", outputs)
    write_csv(paired_model_values, "comparison_condition_paired_run_values.csv", outputs)
    write_csv(additions, "comparison_within_model_data_addition_run_values.csv", outputs)
    write_csv(did, "comparison_data_addition_difference_in_differences.csv", outputs)
    write_csv(paired_did_values, "comparison_data_addition_paired_run_values.csv", outputs)

    figure_overall(overall, overall_differences, outputs, args.dpi)
    annotated_heatmaps(
        model_effects,
        "mean_carafe2_minus_diann_minutes",
        CONDITIONS,
        pooled_order,
        diann_composition,
        "Carafe2 minus DIA-NN across normalization methods and training conditions",
        "Carafe2 − DIA-NN MAE (min)",
        "comparison_fig2_condition_model_differences",
        outputs,
        args.dpi,
    )
    figure_endpoint(summary, pooled_order, outputs, args.dpi)
    annotated_heatmaps(
        did,
        "mean_difference_in_differences_minutes",
        ADDITION_CONDITIONS,
        pooled_order,
        diann_composition,
        "Which model benefits more from additional cross-organism data?",
        "Difference in data-addition effect (min)",
        "comparison_fig4_data_addition_difference_in_differences",
        outputs,
        args.dpi,
    )
    figure_scaling(summary, pooled_order, diann_composition, outputs, args.dpi)
    report_values = write_markdown_report(
        outputs,
        {"diann": diann_root, "carafe2": carafe2_root},
        overall,
        overall_differences,
        did,
    )

    manifest = {
        "study": "rt_normalization_model_comparison",
        "complete": True,
        "diann_root": str(diann_root),
        "carafe2_root": str(carafe2_root),
        "outputs": [str(path) for path in outputs],
        "primary_metric": "raw_mae_minutes",
        "model_contrast": "Carafe2 minus DIA-NN",
        "pairing_keys": ["method", "seed", "condition", "test_organism"],
        "methods": list(METHODS),
        "conditions": list(CONDITIONS),
        "organisms": list(ORGANISMS),
        "chart_contracts": CHART_CONTRACTS,
        "exports": ["PNG 600 dpi", "vector PDF", "editable SVG", "CSV", "Markdown"],
        **report_values,
    }
    for output in outputs:
        (output / "comparison_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print("Wrote identical model-comparison artifacts to:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
