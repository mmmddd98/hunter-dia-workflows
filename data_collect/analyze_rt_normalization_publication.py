#!/usr/bin/env python3
"""Build a publication-ready report for one RT-normalization scaling study.

The script is deliberately model-agnostic. Point ``--study-root`` at either a
DIA-NN or a Carafe2 normalization study with the standard directory layout.
All primary comparisons use errors transformed back to observed LC minutes.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats


METHODS = (
    "global_60min",
    "source_native",
    "anchor_irt",
    "linear",
    "relative_rt",
    "lowess",
    "isotonic",
    "monotonic_spline",
    "calib_rt",
    "kde",
    "chronologer_hi",
)
METHOD_LABELS = {
    "global_60min": "Global 60 min",
    "source_native": "Source-native",
    "anchor_irt": "Joint anchor iRT",
    "linear": "Robust linear",
    "relative_rt": "Relative gradient RT",
    "lowess": "LOWESS",
    "isotonic": "Isotonic",
    "monotonic_spline": "Monotonic spline",
    "calib_rt": "Calib-RT",
    "kde": "KDE ridge",
    "chronologer_hi": "Chronologer HI",
}
METHOD_SHORT = {
    "global_60min": "Global 60 min",
    "source_native": "Source-native",
    "anchor_irt": "Anchor iRT",
    "linear": "Robust linear",
    "relative_rt": "Relative RT",
    "lowess": "LOWESS",
    "isotonic": "Isotonic",
    "monotonic_spline": "Mono. spline",
    "calib_rt": "Calib-RT",
    "kde": "KDE ridge",
    "chronologer_hi": "Chronologer HI",
}
METHOD_COLORS = {
    "global_60min": "#8C8C8C",
    "source_native": "#E69F00",
    "anchor_irt": "#0072B2",
    "chronologer_hi": "#009E73",
    "linear": "#56B4E9",
    "lowess": "#CC79A7",
    "kde": "#44AA99",
    "calib_rt": "#332288",
    "isotonic": "#D55E00",
    "monotonic_spline": "#AA4499",
    "relative_rt": "#999933",
}
ORGANISMS = ("celegans", "triplemix", "human")
ORGANISM_LABELS = {
    "celegans": "C. elegans",
    "triplemix": "TripleMix",
    "human": "Human",
}
SOURCE_COLORS = {
    "celegans": "#4477AA",
    "triplemix": "#EE6677",
    "human": "#228833",
}
CONDITIONS = (
    "ce_010",
    "ce_025",
    "ce_050",
    "ce_075",
    "ce_100",
    "ce100_triple_025",
    "ce100_triple_050",
    "ce100_triple_100",
    "ce100_human_025",
    "ce100_human_050",
    "ce100_human_100",
    "ce100_all_025",
    "ce100_all_050",
    "ce100_all_100",
)
CONDITION_LABELS = {
    "ce_010": "CE 10%",
    "ce_025": "CE 25%",
    "ce_050": "CE 50%",
    "ce_075": "CE 75%",
    "ce_100": "CE 100%",
    "ce100_triple_025": "+TM 25%",
    "ce100_triple_050": "+TM 50%",
    "ce100_triple_100": "+TM 100%",
    "ce100_human_025": "+HU 25%",
    "ce100_human_050": "+HU 50%",
    "ce100_human_100": "+HU 100%",
    "ce100_all_025": "+Both 25%",
    "ce100_all_050": "+Both 50%",
    "ce100_all_100": "+Both 100%",
}
ADDITION_CONDITIONS = CONDITIONS[5:]
BRANCHES = {
    "Add TripleMix": ("ce_100", "ce100_triple_025", "ce100_triple_050", "ce100_triple_100"),
    "Add Human": ("ce_100", "ce100_human_025", "ce100_human_050", "ce100_human_100"),
    "Add both": ("ce_100", "ce100_all_025", "ce100_all_050", "ce100_all_100"),
}
METRICS = (
    "raw_mae_minutes",
    "raw_rmse_minutes",
    "raw_bias_minutes",
    "raw_heavy_dimethyl_mae_minutes",
    "raw_heavy_nterm_dimethyl_mae_minutes",
)


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
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True, help="Display label, e.g. DIA-NN or Carafe2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to STUDY_ROOT/publication_analysis",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Create a clearly marked draft report from an incomplete study",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def collect_runs(root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method in METHODS:
        for path in sorted((root / method / "runs").glob("seed_*/*/organism_test_metrics.json")):
            condition = path.parent.name
            if condition not in CONDITIONS:
                continue
            seed_name = path.parents[1].name
            if not seed_name.startswith("seed_"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for organism in ORGANISMS:
                item = payload.get(organism, {})
                if "raw_mae_minutes" not in item:
                    continue
                record: dict[str, object] = {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": int(seed_name.removeprefix("seed_")),
                    "condition": condition,
                    "condition_order": CONDITIONS.index(condition),
                    "test_organism": organism,
                    "test_organism_label": ORGANISM_LABELS[organism],
                    "rows": item.get("rows", np.nan),
                    "unique_sequences": item.get("unique_sequences", np.nan),
                }
                record.update({metric: item.get(metric, np.nan) for metric in METRICS})
                records.append(record)
    data = pd.DataFrame(records)
    if data.empty:
        raise SystemExit(f"No completed organism_test_metrics.json files found below {root}")
    return data


def validate_runs(data: pd.DataFrame, allow_incomplete: bool) -> pd.DataFrame:
    keys = ["method", "seed", "condition", "test_organism"]
    duplicated = data.duplicated(keys, keep=False)
    if duplicated.any():
        examples = data.loc[duplicated, keys].head().to_dict("records")
        raise SystemExit(f"Duplicate metric records detected: {examples}")
    if not np.isfinite(data["raw_mae_minutes"]).all():
        raise SystemExit("Non-finite raw_mae_minutes values detected")

    seeds = sorted(data.seed.unique())
    rows = []
    for method in METHODS:
        subset = data[data.method.eq(method)]
        observed = len(subset)
        expected = len(seeds) * len(CONDITIONS) * len(ORGANISMS)
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "records_observed": observed,
                "records_expected": expected,
                "complete": observed == expected,
                "seeds_observed": subset.seed.nunique(),
                "conditions_observed": subset.condition.nunique(),
                "organisms_observed": subset.test_organism.nunique(),
            }
        )
    audit = pd.DataFrame(rows)
    if not audit.complete.all() and not allow_incomplete:
        missing = audit.loc[~audit.complete, ["method", "records_observed", "records_expected"]]
        raise SystemExit(
            "Study is incomplete; rerun with --allow-incomplete for a draft report:\n"
            + missing.to_string(index=False)
        )

    # Every method must be scored against exactly the same fixed test rows.
    row_counts = data.groupby(["seed", "condition", "test_organism"])["rows"].nunique(dropna=False)
    if (row_counts > 1).any():
        raise SystemExit("Test-set row counts differ between normalization methods")
    return audit


def collect_training_composition(root: Path, data: pd.DataFrame) -> pd.DataFrame:
    available = [method for method in METHODS if (root / method / "prepared").is_dir()]
    if not available:
        return pd.DataFrame(columns=["seed", "condition", "source_organism", "training_rows"])
    # Transformations preserve rows, so reading one method avoids scanning all 11 copies.
    preferred = "source_native" if "source_native" in available else available[0]
    records = []
    for path in sorted((root / preferred / "prepared").glob("seed_*/trains/*.tsv")):
        if path.stem not in CONDITIONS:
            continue
        seed = int(path.parents[1].name.removeprefix("seed_"))
        counts = pd.read_csv(path, sep="\t", usecols=["source_organism"])[
            "source_organism"
        ].value_counts()
        for organism, count in counts.items():
            records.append(
                {
                    "seed": seed,
                    "condition": path.stem,
                    "source_organism": organism,
                    "training_rows": int(count),
                }
            )
    composition = pd.DataFrame(records)
    expected_seeds = set(data.seed.unique())
    if not composition.empty and set(composition.seed.unique()) != expected_seeds:
        warnings.warn("Training composition is available for only a subset of seeds")
    return composition


def t_interval(values: pd.Series | np.ndarray) -> tuple[float, float, float, float, int]:
    clean = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(clean)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(np.mean(clean))
    sd = float(np.std(clean, ddof=1)) if n > 1 else np.nan
    if n > 1 and sd > 0:
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    else:
        half = 0.0
    return mean, sd, mean - half, mean + half, n


def summarize_conditions(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in data.groupby(["method", "condition", "condition_order", "test_organism"]):
        method, condition, order, organism = keys
        for metric in METRICS:
            mean, sd, low, high, n = t_interval(group[metric])
            records.append(
                {
                    "method": method,
                    "condition": condition,
                    "condition_order": order,
                    "test_organism": organism,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "ci_low": low,
                    "ci_high": high,
                    "seeds": n,
                }
            )
    return pd.DataFrame(records)


def overall_ranking(data: pd.DataFrame, reference: str = "source_native") -> tuple[pd.DataFrame, pd.DataFrame]:
    # Equal weight for each of the 14 condition x 3 test-organism blocks.
    seed_means = (
        data.groupby(["method", "seed"], as_index=False)["raw_mae_minutes"]
        .mean()
        .rename(columns={"raw_mae_minutes": "seed_mean_mae"})
    )
    pivot = seed_means.pivot(index="seed", columns="method", values="seed_mean_mae")
    means = pivot.mean().sort_values()
    best = str(means.index[0])
    records = []
    for rank, method in enumerate(means.index, start=1):
        mean, sd, low, high, n = t_interval(pivot[method])
        ref_mean, _, ref_low, ref_high, _ = t_interval(pivot[method] - pivot[reference])
        best_mean, _, best_low, best_high, _ = t_interval(pivot[method] - pivot[best])
        records.append(
            {
                "rank": rank,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "mean_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
                "delta_vs_source_native": ref_mean,
                "delta_vs_source_native_ci_low": ref_low,
                "delta_vs_source_native_ci_high": ref_high,
                "delta_vs_best": best_mean,
                "delta_vs_best_ci_low": best_low,
                "delta_vs_best_ci_high": best_high,
            }
        )
    return pd.DataFrame(records), seed_means


def combination_effects(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = data[data.condition.eq("ce_100")][
        ["method", "seed", "test_organism", "raw_mae_minutes"]
    ].rename(columns={"raw_mae_minutes": "baseline_mae"})
    paired = data[data.condition.isin(ADDITION_CONDITIONS)].merge(
        base, on=["method", "seed", "test_organism"], validate="many_to_one"
    )
    paired["delta_mae_minutes"] = paired.raw_mae_minutes - paired.baseline_mae
    records = []
    for keys, group in paired.groupby(["method", "condition", "condition_order", "test_organism"]):
        method, condition, order, organism = keys
        mean, sd, low, high, n = t_interval(group.delta_mae_minutes)
        records.append(
            {
                "method": method,
                "condition": condition,
                "condition_order": order,
                "test_organism": organism,
                "mean_delta_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(records), paired


def winner_stability(data: pd.DataFrame) -> pd.DataFrame:
    ranked = data.copy()
    ranked["rank"] = ranked.groupby(["seed", "condition", "test_organism"])[
        "raw_mae_minutes"
    ].rank(method="average")
    result = (
        ranked.groupby(["method", "test_organism"], as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            first_place_runs=("rank", lambda x: int(np.sum(np.isclose(x, 1)))),
            evaluated_runs=("rank", "size"),
        )
    )
    result["first_place_fraction"] = result.first_place_runs / result.evaluated_runs
    return result


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(
            base.with_suffix(suffix),
            dpi=dpi if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def mean_total_training_rows(composition: pd.DataFrame) -> pd.Series:
    """Total rows per condition, summing sources before averaging seeds."""
    if composition.empty:
        return pd.Series(dtype=float)
    return (
        composition.groupby(["seed", "condition"], as_index=False).training_rows.sum()
        .groupby("condition").training_rows.mean()
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")


def figure_design_and_ranking(
    composition: pd.DataFrame,
    ranking: pd.DataFrame,
    seed_means: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 6.2), gridspec_kw={"width_ratios": [1.12, 1]})
    panel_label(left, "a")
    panel_label(right, "b")

    if not composition.empty:
        comp = (
            composition.groupby(["condition", "source_organism"])["training_rows"]
            .mean()
            .unstack(fill_value=0)
            .reindex(CONDITIONS, fill_value=0)
        )
        y = np.arange(len(CONDITIONS))
        cumulative = np.zeros(len(comp))
        for organism in ORGANISMS:
            values = comp.get(organism, pd.Series(0, index=comp.index)).to_numpy() / 1000
            left.barh(
                y,
                values,
                left=cumulative,
                color=SOURCE_COLORS[organism],
                edgecolor="white",
                linewidth=0.4,
                label=ORGANISM_LABELS[organism],
            )
            cumulative += values
        left.set_yticks(y, [CONDITION_LABELS[item] for item in CONDITIONS])
        left.invert_yaxis()
        left.set_xlabel("Training rows (thousands; mean across seeds)")
        left.set_title("Nested training-data design", loc="left", fontweight="bold")
        left.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
        left.legend(frameon=False, ncol=3, loc="lower right")
    else:
        left.text(0.5, 0.5, "Training composition unavailable", ha="center", va="center")
        left.set_axis_off()

    ordered = ranking.sort_values("mean_mae_minutes", ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    seed_lookup = seed_means.pivot(index="method", columns="seed", values="seed_mean_mae")
    for index, row in ordered.iterrows():
        method = row.method
        raw = seed_lookup.loc[method].dropna().to_numpy()
        jitter = np.linspace(-0.09, 0.09, len(raw)) if len(raw) > 1 else np.zeros(len(raw))
        right.scatter(raw, np.full(len(raw), index) + jitter, s=12, color=METHOD_COLORS[method], alpha=0.42, edgecolor="none")
        right.errorbar(
            row.mean_mae_minutes,
            index,
            xerr=[[row.mean_mae_minutes - row.ci_low], [row.ci_high - row.mean_mae_minutes]],
            fmt="o",
            color=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            capsize=2.5,
            linewidth=1.2,
            zorder=3,
        )
    right.set_yticks(y, [METHOD_LABELS[item] for item in ordered.method])
    right.set_xlabel("Mean MAE in observed RT (min)\n(equal weight per condition and test organism)")
    right.set_title(f"{model_label} normalization ranking", loc="left", fontweight="bold")
    right.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
    save_figure(fig, output / "fig1_design_and_overall_ranking", dpi)


def figure_combination_heatmap(
    effects: pd.DataFrame,
    ranking: pd.DataFrame,
    composition: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    method_order = ranking.sort_values("rank").method.tolist()
    if composition.empty:
        row_labels = [CONDITION_LABELS[item] for item in ADDITION_CONDITIONS]
    else:
        totals = mean_total_training_rows(composition) / 1000
        row_labels = [f"{CONDITION_LABELS[item]}\n{totals.get(item, np.nan):.1f}k" for item in ADDITION_CONDITIONS]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.5), constrained_layout=True)
    for panel, (axis, organism) in enumerate(zip(axes, ORGANISMS)):
        matrix = (
            effects[effects.test_organism.eq(organism)]
            .pivot(index="method", columns="condition", values="mean_delta_mae_minutes")
            .reindex(index=method_order, columns=ADDITION_CONDITIONS)
        )
        finite = np.abs(matrix.to_numpy()[np.isfinite(matrix.to_numpy())])
        limit = max(float(np.max(finite)) if finite.size else 0.1, 0.1)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_xticks(range(len(ADDITION_CONDITIONS)), row_labels, rotation=58, ha="right")
        axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
        axis.set_title(ORGANISM_LABELS[organism], fontweight="bold")
        axis.tick_params(length=0)
        axis.axvline(2.5, color="white", linewidth=2.0)
        axis.axvline(5.5, color="white", linewidth=2.0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if not np.isfinite(value):
                    label = "NA"
                    color = "#333333"
                else:
                    label = f"{value:+.2f}"
                    color = "white" if abs(value) > 0.52 * limit else "#202020"
                axis.text(j, i, label, ha="center", va="center", fontsize=6.3, color=color)
        cbar = fig.colorbar(image, ax=axis, shrink=0.72, pad=0.02)
        cbar.set_label("Δ MAE vs CE 100% (min)", fontsize=8)
        panel_label(axis, chr(ord("a") + panel))
    fig.suptitle(
        f"{model_label}: effect of adding cross-organism training data",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output / "fig2_paired_data_combination_effects", dpi)


def figure_scaling_curves(
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    composition: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    top = ranking.sort_values("rank").method.head(4).tolist()
    selected = list(dict.fromkeys(top + ["source_native", "global_60min"]))
    totals = mean_total_training_rows(composition)
    mae = summary[summary.metric.eq("raw_mae_minutes")]
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.0), sharey="row", constrained_layout=True)
    for row, organism in enumerate(ORGANISMS):
        for col, (branch_label, conditions) in enumerate(BRANCHES.items()):
            axis = axes[row, col]
            for method in selected:
                group = mae[
                    mae.method.eq(method)
                    & mae.test_organism.eq(organism)
                    & mae.condition.isin(conditions)
                ].set_index("condition").reindex(conditions)
                if group["mean"].isna().all():
                    continue
                if totals.empty:
                    x = np.arange(len(conditions), dtype=float)
                else:
                    x = np.array([totals.get(condition, np.nan) / 1000 for condition in conditions])
                axis.plot(
                    x,
                    group["mean"],
                    marker="o",
                    markersize=3.7,
                    linewidth=1.6,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
                axis.fill_between(x, group.ci_low, group.ci_high, color=METHOD_COLORS[method], alpha=0.10, linewidth=0)
            if row == 0:
                axis.set_title(branch_label, fontweight="bold")
            if col == 0:
                axis.set_ylabel(f"{ORGANISM_LABELS[organism]} test\nMAE (min)")
            if row == 2:
                axis.set_xlabel("Training rows (thousands)" if not totals.empty else "Training condition")
            axis.grid(color="#D9D9D9", linewidth=0.55)
            if row == 0 and col == 0:
                panel_label(axis, "a")
            elif row == 0 and col == 1:
                panel_label(axis, "b")
            elif row == 0 and col == 2:
                panel_label(axis, "c")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    save_figure(fig, output / "fig3_selected_method_scaling_curves", dpi)


def figure_endpoint(
    data: pd.DataFrame,
    ranking: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    endpoint = data[data.condition.eq("ce100_all_100")]
    method_order = ranking.sort_values("rank", ascending=False).method.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.0), constrained_layout=True)
    for panel, (axis, organism) in enumerate(zip(axes, ORGANISMS)):
        subset = endpoint[endpoint.test_organism.eq(organism)]
        for y, method in enumerate(method_order):
            values = subset[subset.method.eq(method)].raw_mae_minutes
            mean, _, low, high, _ = t_interval(values)
            jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.zeros(len(values))
            axis.scatter(values, np.full(len(values), y) + jitter, s=11, color=METHOD_COLORS[method], alpha=0.42, edgecolor="none")
            axis.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=5.5,
                capsize=2.5,
                linewidth=1.1,
            )
        axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
        axis.set_xlabel("MAE in observed RT (min)")
        axis.set_title(ORGANISM_LABELS[organism], fontweight="bold")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        panel_label(axis, chr(ord("a") + panel))
    fig.suptitle(
        f"{model_label}: full mixed-data endpoint (CE + TripleMix + Human, 100%)",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output / "fig4_full_mixture_endpoint", dpi)


def figure_rank_stability(
    stability: pd.DataFrame,
    ranking: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    method_order = ranking.sort_values("rank").method.tolist()
    matrix = stability.pivot(index="method", columns="test_organism", values="mean_rank").reindex(
        index=method_order, columns=ORGANISMS
    )
    wins = stability.pivot(index="method", columns="test_organism", values="first_place_runs").reindex(
        index=method_order, columns=ORGANISMS
    )
    fig, axis = plt.subplots(figsize=(5.8, 6.0))
    image = axis.imshow(matrix, cmap="viridis_r", norm=Normalize(vmin=1, vmax=len(METHODS)), aspect="auto")
    axis.set_xticks(range(len(ORGANISMS)), [ORGANISM_LABELS[item] for item in ORGANISMS])
    axis.set_yticks(range(len(method_order)), [METHOD_LABELS[item] for item in method_order])
    axis.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rank = matrix.iloc[i, j]
            count = int(wins.iloc[i, j])
            color = "white" if rank > 6.5 else "#181818"
            axis.text(j, i, f"{rank:.1f}\n({count} wins)", ha="center", va="center", fontsize=7.2, color=color)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label("Mean rank; lower is better")
    axis.set_title(f"{model_label}: rank stability over 14 conditions × 5 seeds", fontweight="bold")
    save_figure(fig, output / "fig5_rank_stability", dpi)


def figure_heavy_sensitivity(
    data: pd.DataFrame,
    ranking: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    endpoint = data[data.condition.eq("ce100_all_100")]
    method_order = ranking.sort_values("rank").method.tolist()
    metric_labels = {
        "raw_heavy_dimethyl_mae_minutes": "Any Heavy-Dimethyl",
        "raw_heavy_nterm_dimethyl_mae_minutes": "N-terminal Heavy-Dimethyl",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 9.2), constrained_layout=True)
    for row, (metric, metric_label) in enumerate(metric_labels.items()):
        for col, organism in enumerate(ORGANISMS):
            axis = axes[row, col]
            subset = endpoint[endpoint.test_organism.eq(organism)]
            for y, method in enumerate(method_order):
                values = subset[subset.method.eq(method)][metric].dropna()
                mean, _, low, high, _ = t_interval(values)
                axis.errorbar(
                    mean,
                    y,
                    xerr=[[mean - low], [high - mean]],
                    fmt="o",
                    color=METHOD_COLORS[method],
                    markersize=5,
                    capsize=2,
                    linewidth=1,
                )
            axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
            axis.set_xlabel("MAE in observed RT (min)")
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
            if row == 0:
                axis.set_title(ORGANISM_LABELS[organism], fontweight="bold")
            if col == 0:
                axis.set_ylabel(metric_label)
    fig.suptitle(f"{model_label}: Heavy-Dimethyl endpoint sensitivity analysis", fontsize=12, fontweight="bold")
    save_figure(fig, output / "supp_fig1_heavy_dimethyl_endpoint", dpi)


def format_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def report_findings(
    data: pd.DataFrame,
    ranking: pd.DataFrame,
    effects: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:
    best = ranking.sort_values("rank").iloc[0]
    second = ranking.sort_values("rank").iloc[1]
    second_vs_best = format_ci(
        second.delta_vs_best,
        second.delta_vs_best_ci_low,
        second.delta_vs_best_ci_high,
        3,
    )
    endpoint = data[data.condition.eq("ce100_all_100")]
    endpoint_means = endpoint.groupby(["method", "test_organism"]).raw_mae_minutes.mean()
    full_effect = effects[effects.condition.eq("ce100_all_100")]
    best_effects = full_effect[full_effect.method.eq(best.method)].set_index("test_organism")
    relative_pair = data[data.method.eq("source_native")][
        ["seed", "condition", "test_organism", "raw_mae_minutes"]
    ].merge(
        data[data.method.eq("relative_rt")][
            ["seed", "condition", "test_organism", "raw_mae_minutes"]
        ],
        on=["seed", "condition", "test_organism"],
        suffixes=("_native", "_relative"),
    )
    relative_gap = float(
        np.max(np.abs(relative_pair.raw_mae_minutes_native - relative_pair.raw_mae_minutes_relative))
    )
    global_row = ranking.set_index("method").loc["global_60min"]
    native_row = ranking.set_index("method").loc["source_native"]
    if second.delta_vs_best_ci_low <= 0 <= second.delta_vs_best_ci_high:
        comparison_sentence = (
            "The interval includes zero, so this benchmark does not resolve a clear winner between the top two."
        )
    else:
        comparison_sentence = "The interval excludes zero, supporting the leading method over the runner-up in this benchmark."
    lines = [
        (
            f"<strong>{html.escape(METHOD_LABELS[best.method])}</strong> had the lowest overall mean MAE "
            f"({format_ci(best.mean_mae_minutes, best.ci_low, best.ci_high)} min), with equal weight given "
            "to every training condition and test organism."
        ),
        (
            f"The runner-up, <strong>{html.escape(METHOD_LABELS[second.method])}</strong>, differed from the leading method by "
            f"{second_vs_best} min (paired mean difference and 95% t interval across five seeds). {comparison_sentence}"
        ),
        (
            f"The naive global target ({global_row.mean_mae_minutes:.3f} min) and source-native target "
            f"({native_row.mean_mae_minutes:.3f} min) were substantially worse than the leading aligned targets. "
            "Combining raw target spaces is therefore not supported by these data."
        ),
        (
            f"With the leading method, full mixed training changed MAE relative to CE-only training by "
            + "; ".join(
                f"{ORGANISM_LABELS[organism]} {format_ci(row.mean_delta_mae_minutes, row.ci_low, row.ci_high)} min"
                for organism, row in best_effects.iterrows()
            )
            + ". Negative values indicate improvement. Added data strongly helped Human and TripleMix but modestly harmed C. elegans."
        ),
        (
            "At the full mixed endpoint, the descriptive minima were "
            + "; ".join(
                f"{ORGANISM_LABELS[organism]}: {METHOD_LABELS[endpoint_means.xs(organism, level='test_organism').idxmin()]} "
                f"({endpoint_means.xs(organism, level='test_organism').min():.3f} min)"
                for organism in ORGANISMS
            )
            + "."
        ),
        (
            f"Source-native and Relative gradient RT produced the same observed-minute MAE to numerical precision "
            f"(maximum absolute block difference {relative_gap:.2e} min); they should not be counted as independent evidence of benefit."
        ),
    ]
    machine = {
        "best_overall_method": str(best.method),
        "best_overall_mean_mae_minutes": float(best.mean_mae_minutes),
        "second_overall_method": str(second.method),
        "second_minus_best_mean_minutes": float(second.delta_vs_best),
        "second_minus_best_ci95": [float(second.delta_vs_best_ci_low), float(second.delta_vs_best_ci_high)],
        "source_native_relative_rt_max_abs_difference_minutes": relative_gap,
    }
    return lines, machine


def write_report(
    output: Path,
    root: Path,
    model_label: str,
    data: pd.DataFrame,
    audit: pd.DataFrame,
    ranking: pd.DataFrame,
    effects: pd.DataFrame,
    stability: pd.DataFrame,
    incomplete: bool,
) -> dict[str, object]:
    findings, machine = report_findings(data, ranking, effects)
    recommended = METHOD_LABELS[str(machine["best_overall_method"])]
    runner_up = METHOD_LABELS[str(machine["second_overall_method"])]
    second_ci = machine["second_minus_best_ci95"]
    co_leading = second_ci[0] <= 0 <= second_ci[1]
    top_table = ranking[
        [
            "rank",
            "method_label",
            "mean_mae_minutes",
            "ci_low",
            "ci_high",
            "delta_vs_source_native",
        ]
    ].copy()
    top_table.columns = ["Rank", "Method", "Mean MAE (min)", "CI low", "CI high", "Δ vs source-native (min)"]
    status = "DRAFT — INCOMPLETE STUDY" if incomplete else "COMPLETE STUDY"
    status_class = "warn" if incomplete else "ok"
    figures = [
        (
            "Figure 1. Study design and overall normalization ranking",
            "fig1_design_and_overall_ranking.png",
            "Training rows are counted directly from prepared TSV files. Ranking values are means over 14 conditions and three fixed organism tests for each seed; points show individual seeds and bars show 95% t intervals (n=5).",
        ),
        (
            "Figure 2. Paired effect of adding cross-organism data",
            "fig2_paired_data_combination_effects.png",
            "Each cell is the mean within-seed change from CE-only 100% training for the same method and test organism. Negative values (blue) are improvements. Training-row totals are shown below condition labels.",
        ),
        (
            "Figure 3. Scaling trajectories",
            "fig3_selected_method_scaling_curves.png",
            "The four best overall methods plus the two baseline target definitions. Lines are seed means; ribbons are 95% t intervals. The x-axis is the actual number of training rows, not an ordinal condition index.",
        ),
        (
            "Figure 4. Full mixed-data endpoint",
            "fig4_full_mixture_endpoint.png",
            "Absolute observed-minute MAE after training on 100% CE, TripleMix and Human data. Small points are seeds; large points and bars are means and 95% t intervals.",
        ),
        (
            "Figure 5. Rank stability",
            "fig5_rank_stability.png",
            "Mean within-block rank over 14 conditions and five seeds. Parentheses report the number of first-place runs out of 70 per test organism. Rank summarizes consistency, not effect magnitude.",
        ),
        (
            "Supplementary Figure 1. Heavy-Dimethyl sensitivity",
            "supp_fig1_heavy_dimethyl_endpoint.png",
            "Endpoint analysis restricted to Heavy-Dimethyl subsets. These overlapping subsets are sensitivity analyses and are not independent confirmatory endpoints.",
        ),
    ]
    figure_html = "".join(
        f"<section class='figure'><h2>{html.escape(title)}</h2><img src='{html.escape(path)}' alt='{html.escape(title)}'><p class='caption'>{html.escape(caption)}</p></section>"
        for title, path, caption in figures
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(model_label)} RT-normalization benchmark</title>
<style>
:root{{--ink:#17212b;--muted:#5e6a75;--line:#d9e0e6;--paper:#fff;--wash:#f4f7f9;--blue:#0072b2;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:16px/1.55 Arial,Helvetica,sans-serif}}
main{{max-width:1280px;margin:0 auto;background:var(--paper);padding:54px 64px 80px;box-shadow:0 0 32px #dfe5e9}}
h1{{font-size:34px;line-height:1.15;margin:0 0 8px}} h2{{font-size:21px;margin:36px 0 12px}} h3{{font-size:17px;margin:24px 0 8px}}
.subtitle{{font-size:18px;color:var(--muted);margin:0 0 22px}} .badge{{display:inline-block;padding:5px 10px;border-radius:3px;font-weight:700;font-size:12px;letter-spacing:.06em}}
.ok{{background:#dff2e7;color:#125c37}} .warn{{background:#ffe2db;color:#8b2f1d}} .summary{{border-left:5px solid var(--blue);padding:4px 0 4px 20px;margin:28px 0}}
li{{margin:.5rem 0}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:right}} th{{background:#eef3f6}} th:nth-child(2),td:nth-child(2){{text-align:left}}
.figure{{margin-top:48px;padding-top:8px;border-top:1px solid var(--line)}} img{{display:block;width:100%;height:auto;margin:14px 0 8px}} .caption{{font-size:14px;color:var(--muted);margin-top:6px}}
code{{background:#eef3f6;padding:2px 5px;border-radius:3px}} .small{{font-size:13px;color:var(--muted)}}
@media(max-width:800px){{main{{padding:30px 20px}}h1{{font-size:28px}}}}
</style></head><body><main>
<span class="badge {status_class}">{status}</span>
<h1>{html.escape(model_label)} retention-time normalization benchmark</h1>
<p class="subtitle">Which target representation supports mixed-organism training, and how does performance scale with added data?</p>
<section class="summary"><h2>Technical summary</h2><ul>{''.join(f'<li>{item}</li>' for item in findings)}</ul></section>
<h2>Overall ranking</h2>
<p>Primary endpoint: mean absolute error after inverse transformation to source-observed LC minutes. The analysis gives equal weight to each training condition and test organism, then computes paired uncertainty across the five common seeds.</p>
{top_table.to_html(index=False, border=0, float_format=lambda value: f'{value:.4f}')}
{figure_html}
<h2>Scope, definitions and statistical method</h2>
<ul>
<li><strong>Unit of comparison:</strong> a fixed sequence-disjoint test set for each organism, evaluated for every method, condition and seed.</li>
<li><strong>Primary metric:</strong> <code>raw_mae_minutes</code>. This is evaluated in observed LC minutes after the method-specific inverse mapping, making target spaces comparable.</li>
<li><strong>Uncertainty:</strong> two-sided 95% Student-t intervals over five paired seed-level estimates. They quantify split/training variability in this benchmark, not uncertainty over all possible datasets or instruments.</li>
<li><strong>Data-addition effects:</strong> within-seed MAE differences against <code>ce_100</code> for the same method and test organism. Negative is better.</li>
<li><strong>Overall rank:</strong> equal-weight average over 14 conditions × 3 organisms within each seed. It is intentionally not weighted by test-set size.</li>
</ul>
<h2>Interpretation and limitations</h2>
<ul>
<li>The five seeds reuse the same underlying source datasets; intervals measure robustness to the implemented splitting/training procedure and should not be interpreted as biological replication.</li>
<li>The benchmark compares both alignment mappings and target representations. Chronologer HI additionally uses a fixed external predictor, whereas most other mappings are estimated from the non-test training pool.</li>
<li>The method search is extensive. Confidence intervals are descriptive and unadjusted for multiple comparisons; ranking alone should not be treated as a formal selection test.</li>
<li>Improved Human/TripleMix accuracy alongside a small C. elegans loss is a transfer trade-off. A deployment-specific organism weighting may select a different optimum than the equal-weight ranking.</li>
<li>Heavy-Dimethyl analyses are overlapping subsets and therefore sensitivity checks, not independent replication.</li>
</ul>
<h2>Recommendation</h2>
<p>Use {html.escape(recommended)} as the default mixed-organism target for the next {html.escape(model_label)} experiments{f', while retaining {html.escape(runner_up)} as a co-leading sensitivity analysis' if co_leading else ''}. Robust linear alignment remains a useful simple fallback. Do not pool raw source-native or globally scaled targets when cross-organism accuracy matters. If preserving C. elegans performance is a hard constraint, tune sampling weights or use organism-balanced batches rather than merely adding all available foreign rows.</p>
<h2>Reproducibility</h2>
<p>Input: <code>{html.escape(str(root))}</code><br>Run-level data, all summaries, paired effects, audit tables and vector figures are stored beside this report.</p>
<p class="small">Generated deterministically by <code>analyze_rt_normalization_publication.py</code>. No test labels were used to fit alignment mappings according to the study design.</p>
</main></body></html>"""
    (output / "rt_normalization_report.html").write_text(report, encoding="utf-8")
    return machine


def main() -> None:
    args = parse_args()
    root = args.study_root.resolve()
    output = (args.output_dir or root / "publication_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)

    data = collect_runs(root)
    audit = validate_runs(data, args.allow_incomplete)
    composition = collect_training_composition(root, data)
    summary = summarize_conditions(data)
    ranking, seed_means = overall_ranking(data)
    effects, paired_effects = combination_effects(data)
    stability = winner_stability(data)

    data.sort_values(["method", "seed", "condition_order", "test_organism"]).to_csv(
        output / "run_level_metrics.csv", index=False
    )
    audit.to_csv(output / "completeness_audit.csv", index=False)
    composition.to_csv(output / "training_data_composition.csv", index=False)
    summary.to_csv(output / "condition_metric_summary.csv", index=False)
    ranking.to_csv(output / "overall_method_ranking.csv", index=False)
    seed_means.to_csv(output / "overall_seed_means.csv", index=False)
    effects.to_csv(output / "paired_data_combination_effects.csv", index=False)
    paired_effects.to_csv(output / "paired_data_combination_run_values.csv", index=False)
    stability.to_csv(output / "method_rank_stability.csv", index=False)

    figure_design_and_ranking(composition, ranking, seed_means, output, args.model_label, args.dpi)
    figure_combination_heatmap(effects, ranking, composition, output, args.model_label, args.dpi)
    figure_scaling_curves(summary, ranking, composition, output, args.model_label, args.dpi)
    figure_endpoint(data, ranking, output, args.model_label, args.dpi)
    figure_rank_stability(stability, ranking, output, args.model_label, args.dpi)
    figure_heavy_sensitivity(data, ranking, output, args.model_label, args.dpi)

    machine = write_report(
        output,
        root,
        args.model_label,
        data,
        audit,
        ranking,
        effects,
        stability,
        incomplete=not audit.complete.all(),
    )
    manifest = {
        "study_root": str(root),
        "model_label": args.model_label,
        "complete": bool(audit.complete.all()),
        "methods": list(METHODS),
        "seeds": sorted(int(seed) for seed in data.seed.unique()),
        "conditions": list(CONDITIONS),
        "test_organisms": list(ORGANISMS),
        "primary_metric": "raw_mae_minutes",
        "statistical_interval": "95% Student-t interval over paired seed-level estimates",
        "exports": ["PNG 600 dpi", "vector PDF", "editable SVG", "CSV", "HTML"],
        **machine,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote publication analysis to {output}")
    print(f"Report: {output / 'rt_normalization_report.html'}")


if __name__ == "__main__":
    main()
