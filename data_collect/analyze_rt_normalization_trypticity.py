#!/usr/bin/env python3
"""Stratify a completed RT-normalization study by FASTA-backed trypticity.

The script is model-family agnostic and supports both DIA-NN and Carafe2
prediction filenames. Existing run outputs are read only. Results are written
to ``publication_analysis/trypticity_analysis`` below the supplied study root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_rt_normalization_publication import (
    BRANCHES,
    CONDITIONS,
    CONDITION_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_SHORT,
    METHODS,
    ORGANISM_LABELS,
    ORGANISMS,
    panel_label,
    t_interval,
)
from rt_alignment import predictions_to_observed_rt


STRATA = ("fully_tryptic", "semi_tryptic")
STRATUM_LABELS = {
    "fully_tryptic": "Fully tryptic",
    "semi_tryptic": "Semi-tryptic",
    "ambiguous": "Ambiguous",
}
STRATUM_COLORS = {
    "fully_tryptic": "#0072B2",
    "semi_tryptic": "#D55E00",
    "ambiguous": "#8C8C8C",
}
STRATUM_MARKERS = {"fully_tryptic": "o", "semi_tryptic": "s"}
DETAIL_ORDER = ("fully_tryptic", "semi_nterm", "semi_cterm", "ambiguous")
DETAIL_LABELS = {
    "fully_tryptic": "Fully tryptic",
    "semi_nterm": "Semi-tryptic N-terminus",
    "semi_cterm": "Semi-tryptic C-terminus",
    "ambiguous": "Ambiguous protein context",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
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
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--annotation-catalog",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "trainingdata/rt_trypticity_annotated_v1/rt_trypticity_annotations.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to STUDY_ROOT/publication_analysis/trypticity_analysis",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def prediction_path(run_dir: Path, organism: str) -> Path:
    candidates = (
        run_dir / f"test_predictions_{organism}.tsv",
        run_dir / f"test_predictions_{organism}_rt_pred.tsv",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected one prediction file for {organism} below {run_dir}; found {existing}"
        )
    return existing[0]


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    delta = prediction - target
    return {
        "mae_minutes": float(np.mean(np.abs(delta))),
        "rmse_minutes": float(np.sqrt(np.mean(delta * delta))),
        "bias_minutes": float(np.mean(delta)),
    }


def collect_metrics(root: Path, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["source_organism", "sequence"]
    if catalog.duplicated(keys).any():
        raise SystemExit("Annotation catalog has duplicate organism/sequence keys")
    required = {
        "source_organism",
        "sequence",
        "digestion_class",
        "trypticity_detail",
    }
    missing = required - set(catalog.columns)
    if missing:
        raise SystemExit(f"Annotation catalog lacks columns: {sorted(missing)}")

    binary_records: list[dict[str, object]] = []
    detail_records: list[dict[str, object]] = []
    alignment_cache: dict[tuple[str, int], dict] = {}
    for method in METHODS:
        run_root = root / method / "runs"
        run_dirs = sorted(run_root.glob("seed_*/*"))
        if len(run_dirs) != 70:
            raise SystemExit(f"{method}: expected 70 completed run directories, found {len(run_dirs)}")
        for run_dir in run_dirs:
            condition = run_dir.name
            if condition not in CONDITIONS:
                continue
            seed = int(run_dir.parent.name.removeprefix("seed_"))
            cache_key = (method, seed)
            if cache_key not in alignment_cache:
                alignment_path = root / method / "prepared" / f"seed_{seed}" / "rt_alignment.json"
                alignment_cache[cache_key] = json.loads(alignment_path.read_text(encoding="utf-8"))
            alignment = alignment_cache[cache_key]
            metrics_path = run_dir / "organism_test_metrics.json"
            expected_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            for organism in ORGANISMS:
                frame = pd.read_csv(
                    prediction_path(run_dir, organism),
                    sep="\t",
                    usecols=lambda column: column in {"sequence", "source_organism", "rt", "rt_pred"},
                )
                if set(frame.columns) != {"sequence", "source_organism", "rt", "rt_pred"}:
                    raise RuntimeError(f"Unexpected prediction schema in {prediction_path(run_dir, organism)}")
                if not frame.source_organism.astype(str).eq(organism).all():
                    raise RuntimeError(f"Mixed source labels in {run_dir}/{organism}")
                annotated = frame.merge(catalog, on=keys, how="left", validate="many_to_one")
                if annotated.digestion_class.isna().any():
                    raise RuntimeError(
                        f"Missing annotations for {int(annotated.digestion_class.isna().sum())} rows "
                        f"in {run_dir}/{organism}"
                    )
                observed = pd.to_numeric(annotated.rt, errors="raise").to_numpy(float)
                predicted = predictions_to_observed_rt(
                    pd.to_numeric(annotated.rt_pred, errors="raise").to_numpy(float),
                    annotated.source_organism,
                    alignment,
                )
                reconstructed = regression_metrics(observed, predicted)
                expected = float(expected_metrics[organism]["raw_mae_minutes"])
                # Prediction TSVs are text-rounded; tolerate sub-microsecond
                # reconstruction noise while still catching scale/unit errors.
                if not np.isclose(reconstructed["mae_minutes"], expected, rtol=0, atol=1e-6):
                    raise RuntimeError(
                        f"Observed-minute MAE mismatch in {run_dir}/{organism}: "
                        f"{reconstructed['mae_minutes']} versus {expected}"
                    )
                base = {
                    "model_label": "",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed,
                    "condition": condition,
                    "condition_order": CONDITIONS.index(condition),
                    "test_organism": organism,
                }
                for stratum in (*STRATA, "ambiguous"):
                    mask = annotated.digestion_class.eq(stratum).to_numpy()
                    if not mask.any():
                        continue
                    binary_records.append(
                        {
                            **base,
                            "digestion_class": stratum,
                            "rows": int(mask.sum()),
                            "unique_sequences": int(annotated.loc[mask, "sequence"].nunique()),
                            **regression_metrics(observed[mask], predicted[mask]),
                        }
                    )
                for detail in DETAIL_ORDER:
                    mask = annotated.trypticity_detail.eq(detail).to_numpy()
                    if not mask.any():
                        continue
                    detail_records.append(
                        {
                            **base,
                            "trypticity_detail": detail,
                            "rows": int(mask.sum()),
                            "unique_sequences": int(annotated.loc[mask, "sequence"].nunique()),
                            **regression_metrics(observed[mask], predicted[mask]),
                        }
                    )
    return pd.DataFrame(binary_records), pd.DataFrame(detail_records)


def summarize(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    records = []
    group_keys = ["method", "condition", "condition_order", "test_organism", group_column]
    for keys, group in data.groupby(group_keys):
        method, condition, order, organism, stratum = keys
        mean, sd, low, high, n = t_interval(group.mae_minutes)
        records.append(
            {
                "method": method,
                "condition": condition,
                "condition_order": order,
                "test_organism": organism,
                group_column: stratum,
                "mean_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
                "mean_rows": float(group.rows.mean()),
            }
        )
    return pd.DataFrame(records)


def overall_ranking(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = data[data.digestion_class.isin(STRATA)]
    seed_means = (
        eligible.groupby(["method", "digestion_class", "seed"], as_index=False).mae_minutes.mean()
        .rename(columns={"mae_minutes": "seed_mean_mae_minutes"})
    )
    records = []
    for keys, group in seed_means.groupby(["method", "digestion_class"]):
        method, stratum = keys
        mean, sd, low, high, n = t_interval(group.seed_mean_mae_minutes)
        records.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "digestion_class": stratum,
                "mean_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
            }
        )
    ranking = pd.DataFrame(records)
    pooled = ranking.groupby("method").mean_mae_minutes.mean().sort_values()
    ranking["pooled_rank"] = ranking.method.map({method: rank for rank, method in enumerate(pooled.index, 1)})

    pivot = seed_means.pivot(index=["method", "seed"], columns="digestion_class", values="seed_mean_mae_minutes").reset_index()
    pivot["semi_minus_fully_mae_minutes"] = pivot.semi_tryptic - pivot.fully_tryptic
    gap_records = []
    for method, group in pivot.groupby("method"):
        mean, sd, low, high, n = t_interval(group.semi_minus_fully_mae_minutes)
        gap_records.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "pooled_rank": int(ranking.loc[ranking.method.eq(method), "pooled_rank"].iloc[0]),
                "mean_semi_minus_fully_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
            }
        )
    return ranking, pd.DataFrame(gap_records)


def full_addition_effects(data: pd.DataFrame) -> pd.DataFrame:
    eligible = data[data.digestion_class.isin(STRATA)]
    base = eligible[eligible.condition.eq("ce_100")][
        ["method", "seed", "test_organism", "digestion_class", "mae_minutes"]
    ].rename(columns={"mae_minutes": "ce100_mae_minutes"})
    full = eligible[eligible.condition.eq("ce100_all_100")].merge(
        base,
        on=["method", "seed", "test_organism", "digestion_class"],
        validate="one_to_one",
    )
    full["delta_mae_minutes"] = full.mae_minutes - full.ce100_mae_minutes
    records = []
    for keys, group in full.groupby(["method", "test_organism", "digestion_class"]):
        method, organism, stratum = keys
        mean, sd, low, high, n = t_interval(group.delta_mae_minutes)
        records.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "test_organism": organism,
                "digestion_class": stratum,
                "mean_delta_mae_minutes": mean,
                "sd_across_seeds": sd,
                "ci_low": low,
                "ci_high": high,
                "seeds": n,
            }
        )
    return pd.DataFrame(records)


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(
            base.with_suffix(suffix),
            dpi=dpi if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def figure_composition(catalog: pd.DataFrame, quality: dict, output: Path, dpi: int) -> None:
    sequence_counts = (
        catalog.groupby(["source_organism", "digestion_class"]).size().unstack(fill_value=0).reindex(ORGANISMS)
    )
    row_records = []
    for source in quality["sources"]:
        for stratum, count in source["digestion_class_rows"].items():
            row_records.append(
                {"source_organism": source["organism"], "digestion_class": stratum, "count": count}
            )
    row_counts = pd.DataFrame(row_records).pivot(
        index="source_organism", columns="digestion_class", values="count"
    ).fillna(0).reindex(ORGANISMS)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    for panel, (axis, counts, title) in enumerate(
        zip(axes, (row_counts, sequence_counts), ("Training peptidoform rows", "Unique peptide sequences"))
    ):
        fractions = counts.div(counts.sum(axis=1), axis=0) * 100
        left = np.zeros(len(fractions))
        for stratum in (*STRATA, "ambiguous"):
            values = fractions.get(stratum, pd.Series(0, index=fractions.index)).to_numpy()
            axis.barh(
                range(len(fractions)),
                values,
                left=left,
                color=STRATUM_COLORS[stratum],
                edgecolor="white",
                linewidth=0.5,
                label=STRATUM_LABELS[stratum],
            )
            for y, (start, value) in enumerate(zip(left, values)):
                if value >= 3:
                    axis.text(start + value / 2, y, f"{value:.1f}%", ha="center", va="center", color="white", fontsize=8)
            left += values
        axis.set_yticks(range(len(ORGANISMS)), [ORGANISM_LABELS[item] for item in ORGANISMS])
        axis.set_xlim(0, 100)
        axis.set_xlabel("Composition (%)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        panel_label(axis, chr(ord("a") + panel))
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(1.05, -0.30))
    save_figure(fig, output / "trypticity_fig1_annotation_composition", dpi)


def figure_ranking(
    ranking: pd.DataFrame,
    gaps: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    method_order = (
        ranking[["method", "pooled_rank"]].drop_duplicates().sort_values("pooled_rank", ascending=False).method.tolist()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.1), constrained_layout=True)
    absolute_limits = (
        float(ranking.ci_low.min() - 0.05),
        float(ranking.ci_high.max() + 0.05),
    )
    for panel, stratum in enumerate(STRATA):
        axis = axes[panel]
        subset = ranking[ranking.digestion_class.eq(stratum)].set_index("method").loc[method_order]
        for y, row in enumerate(subset.itertuples()):
            axis.errorbar(
                row.mean_mae_minutes,
                y,
                xerr=[[row.mean_mae_minutes - row.ci_low], [row.ci_high - row.mean_mae_minutes]],
                fmt=STRATUM_MARKERS[stratum],
                color=METHOD_COLORS[row.Index],
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=6,
                capsize=2.5,
                linewidth=1.2,
            )
        axis.set_yticks(range(len(method_order)), [METHOD_LABELS[item] for item in method_order])
        axis.set_xlim(*absolute_limits)
        axis.set_xlabel("Mean MAE in observed RT (min)")
        axis.set_title(STRATUM_LABELS[stratum], loc="left", fontweight="bold")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        panel_label(axis, chr(ord("a") + panel))

    axis = axes[2]
    gap = gaps.set_index("method").loc[method_order]
    axis.axvline(0, color="#444444", linewidth=0.9)
    for y, row in enumerate(gap.itertuples()):
        axis.errorbar(
            row.mean_semi_minus_fully_mae_minutes,
            y,
            xerr=[
                [row.mean_semi_minus_fully_mae_minutes - row.ci_low],
                [row.ci_high - row.mean_semi_minus_fully_mae_minutes],
            ],
            fmt="o",
            color="#5B5B5B",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            capsize=2.5,
            linewidth=1.2,
        )
    axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
    axis.set_xlabel("Semi-tryptic − fully tryptic MAE (min)")
    axis.set_title("Paired difficulty gap", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    panel_label(axis, "c")
    fig.suptitle(f"{model_label}: normalization ranking by digestion class", fontsize=12, fontweight="bold")
    save_figure(fig, output / "trypticity_fig2_method_ranking", dpi)


def figure_endpoint(
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    endpoint = summary[
        summary.condition.eq("ce100_all_100") & summary.digestion_class.isin(STRATA)
    ]
    method_order = (
        ranking[["method", "pooled_rank"]].drop_duplicates().sort_values("pooled_rank", ascending=False).method.tolist()
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6), sharex="col", constrained_layout=True)
    for row, stratum in enumerate(STRATA):
        for col, organism in enumerate(ORGANISMS):
            axis = axes[row, col]
            subset = endpoint[
                endpoint.digestion_class.eq(stratum) & endpoint.test_organism.eq(organism)
            ].set_index("method").loc[method_order]
            for y, item in enumerate(subset.itertuples()):
                axis.errorbar(
                    item.mean_mae_minutes,
                    y,
                    xerr=[[item.mean_mae_minutes - item.ci_low], [item.ci_high - item.mean_mae_minutes]],
                    fmt=STRATUM_MARKERS[stratum],
                    color=METHOD_COLORS[item.Index],
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    markersize=5.5,
                    capsize=2.2,
                    linewidth=1.1,
                )
            axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
            if row == 0:
                axis.set_title(ORGANISM_LABELS[organism], fontweight="bold")
            if col == 0:
                axis.set_ylabel(STRATUM_LABELS[stratum])
            if row == 1:
                axis.set_xlabel("MAE in observed RT (min)")
    fig.suptitle(
        f"{model_label}: full mixed-data endpoint by digestion class",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output / "trypticity_fig3_full_mixture_endpoint", dpi)


def figure_addition_effects(
    effects: pd.DataFrame,
    ranking: pd.DataFrame,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    method_order = (
        ranking[["method", "pooled_rank"]].drop_duplicates().sort_values("pooled_rank", ascending=False).method.tolist()
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.1), constrained_layout=True)
    for panel, (axis, organism) in enumerate(zip(axes, ORGANISMS)):
        subset = effects[effects.test_organism.eq(organism)]
        axis.axvline(0, color="#444444", linewidth=0.9)
        for y, method in enumerate(method_order):
            rows = subset[subset.method.eq(method)].set_index("digestion_class")
            values = [rows.loc[stratum, "mean_delta_mae_minutes"] for stratum in STRATA]
            axis.plot(values, [y, y], color="#C8CDD2", linewidth=1.1, zorder=1)
            for stratum, offset in zip(STRATA, (-0.08, 0.08)):
                item = rows.loc[stratum]
                axis.errorbar(
                    item.mean_delta_mae_minutes,
                    y + offset,
                    xerr=[
                        [item.mean_delta_mae_minutes - item.ci_low],
                        [item.ci_high - item.mean_delta_mae_minutes],
                    ],
                    fmt=STRATUM_MARKERS[stratum],
                    color=STRATUM_COLORS[stratum],
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    markersize=5.2,
                    capsize=2.1,
                    linewidth=1.0,
                    label=STRATUM_LABELS[stratum] if y == 0 and panel == 0 else None,
                    zorder=3,
                )
        axis.set_yticks(range(len(method_order)), [METHOD_SHORT[item] for item in method_order])
        axis.set_xlabel("Δ MAE: full mix − CE 100% (min)\nnegative means added data improves")
        axis.set_title(ORGANISM_LABELS[organism], fontweight="bold")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        panel_label(axis, chr(ord("a") + panel))
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        f"{model_label}: effect of full cross-organism training by digestion class",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output / "trypticity_fig4_full_data_addition_effect", dpi)


def figure_best_method_scaling(
    summary: pd.DataFrame,
    best_method: str,
    training_rows: pd.Series,
    output: Path,
    model_label: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.8), sharey="row", constrained_layout=True)
    for row, organism in enumerate(ORGANISMS):
        for col, (branch_label, conditions) in enumerate(BRANCHES.items()):
            axis = axes[row, col]
            x = np.array([training_rows.loc[condition] / 1000 for condition in conditions])
            for stratum in STRATA:
                group = (
                    summary[
                        summary.method.eq(best_method)
                        & summary.test_organism.eq(organism)
                        & summary.digestion_class.eq(stratum)
                        & summary.condition.isin(conditions)
                    ]
                    .set_index("condition")
                    .reindex(conditions)
                )
                axis.plot(
                    x,
                    group.mean_mae_minutes,
                    color=STRATUM_COLORS[stratum],
                    marker=STRATUM_MARKERS[stratum],
                    markersize=4,
                    linewidth=1.6,
                    label=STRATUM_LABELS[stratum],
                )
                axis.fill_between(
                    x,
                    group.ci_low,
                    group.ci_high,
                    color=STRATUM_COLORS[stratum],
                    alpha=0.10,
                    linewidth=0,
                )
            if row == 0:
                axis.set_title(branch_label, fontweight="bold")
                panel_label(axis, chr(ord("a") + col))
            if col == 0:
                axis.set_ylabel(f"{ORGANISM_LABELS[organism]} test\nMAE (min)")
            if row == 2:
                axis.set_xlabel("Training rows (thousands)")
            axis.grid(color="#D9D9D9", linewidth=0.55)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"{model_label}: {METHOD_LABELS[best_method]} scaling by digestion class",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output / "trypticity_fig5_best_method_scaling", dpi)


def training_row_totals(study_root: Path) -> pd.Series:
    """Return actual prepared training rows, averaged across the five seeds."""
    records = []
    paths = sorted(
        (study_root / "source_native/prepared").glob("seed_*/trains/*.tsv")
    )
    if len(paths) != 5 * len(CONDITIONS):
        raise RuntimeError(
            "Expected 70 source-native prepared training files, "
            f"found {len(paths)} below {study_root}"
        )
    for path in paths:
        records.append(
            {
                "seed": int(path.parents[1].name.removeprefix("seed_")),
                "condition": path.stem,
                "rows": len(pd.read_csv(path, sep="\t", usecols=["sequence"])),
            }
        )
    counts = pd.DataFrame(records)
    if set(counts.condition) != set(CONDITIONS):
        raise RuntimeError("Prepared training conditions do not match the benchmark design")
    return counts.groupby("condition").rows.mean().reindex(CONDITIONS)


def write_summary(
    output: Path,
    model_label: str,
    ranking: pd.DataFrame,
    gaps: pd.DataFrame,
    quality: dict,
) -> None:
    best = (
        ranking.groupby("method").mean_mae_minutes.mean().sort_values().index[0]
    )
    best_rows = ranking[ranking.method.eq(best)].set_index("digestion_class")
    best_gap = gaps.set_index("method").loc[best]
    ambiguous = {
        source["organism"]: source["ambiguous_sequence_fraction"]
        for source in quality["sources"]
    }
    report = f"""# {model_label} RT-normalization benchmark by trypticity

## Primary result

- Best equal-weight method across fully and semi-tryptic strata: **{METHOD_LABELS[best]}**.
- Fully tryptic mean MAE: **{best_rows.loc['fully_tryptic', 'mean_mae_minutes']:.3f} min**.
- Semi-tryptic mean MAE: **{best_rows.loc['semi_tryptic', 'mean_mae_minutes']:.3f} min**.
- Paired semi-minus-fully gap: **{best_gap.mean_semi_minus_fully_mae_minutes:+.3f} min** \
  (95% t interval {best_gap.ci_low:+.3f} to {best_gap.ci_high:+.3f}; n=5 seeds).

## Annotation validity

- Classification uses the exact recorded `R*` semi-specific digestion contract.
- Every training/test sequence maps to at least one DIA-NN-reported FASTA occurrence.
- Conflicting protein contexts are excluded from the binary comparison and retained as `ambiguous`.
- Ambiguous sequence fractions: {', '.join(f'{ORGANISM_LABELS[key]} {value:.2%}' for key, value in ambiguous.items())}.
- Modifications do not change the digestion class; annotation grain is source organism plus unmodified sequence.

## Statistical unit

MAE is computed after method-specific inverse transformation to source-observed LC minutes. Peptidoform rows are equally weighted within each test stratum. Overall method estimates first average the 14 condition × 3 organism blocks within each seed, then report two-sided 95% Student-t intervals across five seeds.

## Files

- `trypticity_fig1_annotation_composition`: annotation coverage and composition.
- `trypticity_fig2_method_ranking`: method ranking and paired semi-versus-fully difficulty gap.
- `trypticity_fig3_full_mixture_endpoint`: full mixed-data endpoint by organism and stratum.
- `trypticity_fig4_full_data_addition_effect`: full-mix versus CE-only data-addition effect.
- `trypticity_fig5_best_method_scaling`: best-method scaling over actual training rows.
"""
    (output / "trypticity_analysis_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.study_root.resolve()
    catalog_path = args.annotation_catalog.resolve()
    annotation_root = catalog_path.parent
    output = (
        args.output_dir or root / "publication_analysis/trypticity_analysis"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(catalog_path, sep="\t")
    quality = json.loads((annotation_root / "quality_report.json").read_text(encoding="utf-8"))

    binary, detail = collect_metrics(root, catalog)
    binary["model_label"] = args.model_label
    detail["model_label"] = args.model_label
    binary_summary = summarize(binary, "digestion_class")
    detail_summary = summarize(detail, "trypticity_detail")
    ranking, gaps = overall_ranking(binary)
    effects = full_addition_effects(binary)
    training_rows = training_row_totals(root)

    binary.to_csv(output / "trypticity_run_level_metrics.csv", index=False)
    detail.to_csv(output / "trypticity_detail_run_level_metrics.csv", index=False)
    binary_summary.to_csv(output / "trypticity_condition_summary.csv", index=False)
    detail_summary.to_csv(output / "trypticity_detail_condition_summary.csv", index=False)
    ranking.to_csv(output / "trypticity_overall_method_ranking.csv", index=False)
    gaps.to_csv(output / "trypticity_paired_difficulty_gaps.csv", index=False)
    effects.to_csv(output / "trypticity_full_data_addition_effects.csv", index=False)

    figure_composition(catalog, quality, output, args.dpi)
    figure_ranking(ranking, gaps, output, args.model_label, args.dpi)
    figure_endpoint(binary_summary, ranking, output, args.model_label, args.dpi)
    figure_addition_effects(effects, ranking, output, args.model_label, args.dpi)
    best_method = ranking.groupby("method").mean_mae_minutes.mean().sort_values().index[0]
    figure_best_method_scaling(
        binary_summary, best_method, training_rows, output, args.model_label, args.dpi
    )
    write_summary(output, args.model_label, ranking, gaps, quality)

    manifest = {
        "study_root": str(root),
        "model_label": args.model_label,
        "annotation_catalog": str(catalog_path),
        "annotation_catalog_sha256": quality["catalog_sha256"],
        "primary_strata": list(STRATA),
        "ambiguous_policy": "retained in CSV; excluded from binary figures and rankings",
        "primary_metric": "MAE after inverse transformation to observed LC minutes",
        "best_equal_weight_method": best_method,
        "exports": ["PNG 600 dpi", "vector PDF", "editable SVG", "CSV", "Markdown"],
    }
    (output / "trypticity_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote trypticity-stratified analysis to {output}")


if __name__ == "__main__":
    main()
