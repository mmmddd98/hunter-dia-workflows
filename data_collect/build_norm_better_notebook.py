#!/usr/bin/env python3
"""Build the publication-oriented RT alignment comparison notebook."""

import os
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(
    os.environ.get("CARAFE2_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])
).expanduser()
OUTPUT = Path(
    os.environ.get("NORM_NOTEBOOK_OUTPUT", PROJECT_ROOT / "trainingdata" / "norm_better.ipynb")
).expanduser()


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
notebook["cells"] = [
    md(r"""
# Retention-time alignment and target normalization across heterogeneous LC gradients

**A reproducible, publication-oriented comparison of raw RT, normalization baselines,
run-to-reference alignment, and external hydrophobic-index alignment.**

The notebook uses the immutable Carafe2 source tables and the five-seed
`rt_normalization_scaling_v1` study. Representation geometry is computed directly from a
sequence-disjoint seed-2024 calibration split, so all methods can be inspected before every
neural-network run has completed. Model-performance sections automatically include only
available DIA-NN and Carafe2 results. Figures are shown inline and exported as editable SVG,
vector PDF, and 600-dpi PNG to `trainingdata/norm_better_figures/`.
"""),
    md(r"""
## tl;dr

- DIA-NN and Carafe2 both receive one scalar `rt_norm` target. The scientifically meaningful
  operation is first to align every source into a common RT or HI coordinate; the final affine
  scaling to an approximately 0–1 neural-network target is a separate normalization step.
- Twelve representations are visualized: raw observed minutes plus eleven trained target
  definitions. These include the original baselines and all eight requested alignment classes.
- Alignment fits and target-to-observed inverse maps use training rows only. Test rows are
  sequence-disjoint and are used solely for representation checks and model evaluation.
- Trained comparisons remain valid while the study is incomplete: completion is shown
  explicitly, paired statistics require at least three matched seeds, and missing runs never
  enter a denominator silently.
"""),
    md(r"""
## Context & Methods

### Analytical questions

1. How do the candidate transformations change cross-source target geometry?
2. Which transformations most consistently place shared training peptidoforms in the same
   coordinate, without confusing calibration diagnostics with held-out prediction error?
3. As runs become available, which target representation minimizes observed-minute error for
   DIA-NN and Carafe2 under identical sequence splits and scaling conditions?

### Representation levels

- **Descriptive baseline:** raw observed RT in minutes; not supplied to the benchmark models.
- **Normalization baselines:** fixed 60-minute scaling and source-native/relative-gradient RT.
- **Human-reference alignment:** joint affine anchors, direct robust linear, LOWESS, isotonic,
  monotonic PCHIP, Calib-RT, and a Chronologer-style KDE ridge.
- **External reference:** measured RT mapped to Chronologer hydrophobic index (HI).

`anchor_irt` is retained as the legacy benchmark identifier. Because Human measured RT is the
identity reference rather than a standardized iRT peptide panel, its precise interpretation is
**joint anchor RT**, not canonical iRT.

### Key assumptions and safeguards

- Seed 2024 uses the same deterministic 10% sequence hold-out logic as the scaling study.
- The union of test sequences from all organisms is excluded from every calibration pool.
- Primary model comparisons use MAE/RMSE after the training-only inverse map back to observed
  LC minutes. Errors in differently scaled `rt_norm` spaces are not compared directly.
- Shared-anchor residuals are in-calibration geometry diagnostics, not generalization metrics.
- Chronologer-HI uses compatible unmodified training peptides and a fixed external predictor;
  it is therefore an external-reference strategy, not merely another smoother.
- Split seeds measure split/training sensitivity, not independent biological replication.
"""),
    code(r"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from scipy import stats
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(os.environ.get("CARAFE2_WORKSPACE_ROOT", Path.cwd())).expanduser().resolve()
TRAINING_ROOT = PROJECT_ROOT / "trainingdata"
ALIGNMENT_MODULE_ROOT = PROJECT_ROOT / "ma/data_collect"
if str(ALIGNMENT_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_MODULE_ROOT))

from rt_alignment import (
    apply_rt_target,
    fit_target_alignment,
    predictions_to_observed_rt,
)
from rt_alignment_methods import transform_alignment_state

MODEL_ROOTS = {
    "DIA-NN": Path(os.environ.get(
        "DIANN_NORMALIZATION_STUDY_ROOT", PROJECT_ROOT.parent / "diann_finetune_out" / "rt_normalization_scaling_v1"
    )).expanduser(),
    "Carafe2": Path(os.environ.get(
        "CARAFE2_NORMALIZATION_STUDY_ROOT", PROJECT_ROOT / "finetune_output" / "rt_normalization_scaling_v1"
    )).expanduser(),
}
FIGURE_DIR = TRAINING_ROOT / "norm_better_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_MINUTES = 60.101620738856
SEEDS = tuple(range(2024, 2029))
METHODS = (
    "global_60min", "source_native", "anchor_irt", "linear", "relative_rt",
    "lowess", "isotonic", "monotonic_spline", "calib_rt", "kde",
    "chronologer_hi",
)
METHOD_LABELS = {
    "raw": "Raw RT (none)",
    "global_60min": "Global 60 min",
    "source_native": "Source-native",
    "anchor_irt": "Joint anchor RT",
    "linear": "Robust linear",
    "relative_rt": "Relative gradient RT",
    "lowess": "LOWESS",
    "isotonic": "Isotonic",
    "monotonic_spline": "Monotonic spline",
    "calib_rt": "Calib-RT",
    "kde": "KDE ridge",
    "chronologer_hi": "Chronologer HI",
}
METHOD_FAMILIES = {
    "global_60min": "Scaling baseline",
    "source_native": "Scaling baseline",
    "anchor_irt": "Affine reference alignment",
    "linear": "Affine reference alignment",
    "relative_rt": "Scaling baseline",
    "lowess": "Smooth reference alignment",
    "isotonic": "Monotonic reference alignment",
    "monotonic_spline": "Monotonic reference alignment",
    "calib_rt": "Robust reference alignment",
    "kde": "Density-ridge reference alignment",
    "chronologer_hi": "External HI alignment",
}
ORGANISMS = ("celegans", "triplemix", "human")
ORGANISM_LABELS = {
    "celegans": r"$C.\ elegans$", "triplemix": "Triple mix", "human": "Human",
}
PLAIN_ORGANISM_LABELS = {
    "celegans": "C. elegans", "triplemix": "Triple mix", "human": "Human",
}
SOURCE_PATHS = {
    "celegans": TRAINING_ROOT / "celegans_exploris_44min_rtms2_v1/diann_rt_input/rt_train_data.tsv",
    "triplemix": TRAINING_ROOT / "triplemix_source_a_entrap_neu_filtered_v1/diann_rt_input/rt_train_data.tsv",
    "human": TRAINING_ROOT / "human_source_hunter_human_rtms2_v1/diann_rt_input/rt_train_data.tsv",
}
CONDITION_ORDER = (
    "ce_010", "ce_025", "ce_050", "ce_075", "ce_100",
    "ce100_triple_025", "ce100_triple_050", "ce100_triple_100",
    "ce100_human_025", "ce100_human_050", "ce100_human_100",
    "ce100_all_025", "ce100_all_050", "ce100_all_100",
)
CONDITION_LABELS = {
    "ce_010": "CE 10%", "ce_025": "CE 25%", "ce_050": "CE 50%",
    "ce_075": "CE 75%", "ce_100": "CE 100%",
    "ce100_triple_025": "+TM 25%", "ce100_triple_050": "+TM 50%",
    "ce100_triple_100": "+TM 100%", "ce100_human_025": "+HU 25%",
    "ce100_human_050": "+HU 50%", "ce100_human_100": "+HU 100%",
    "ce100_all_025": "+both 25%", "ce100_all_050": "+both 50%",
    "ce100_all_100": "+both 100%",
}

# Three source roots and two model roots; method identity is carried by facets/labels.
SOURCE_COLORS = {"celegans": "#3B6FB6", "triplemix": "#D89B2B", "human": "#64733B"}
SOURCE_MARKERS = {"celegans": "o", "triplemix": "s", "human": "^"}
SOURCE_LINES = {"celegans": "-", "triplemix": "--", "human": ":"}
MODEL_COLORS = {"DIA-NN": "#3B6FB6", "Carafe2": "#D89B2B"}
INK, MUTED, GRID, PAPER, MISSING = "#20242A", "#687078", "#D9DEE3", "#FFFFFF", "#ECEFF1"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide", category=RuntimeWarning,
    module="statsmodels",
)
sns.set_theme(style="ticks", context="paper")
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.3, "axes.titlesize": 9.2,
    "axes.labelsize": 8.3, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.7, "lines.linewidth": 1.35, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": PAPER,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
})

def panel_letters(axes, x=-0.16, y=1.08):
    for letter, axis in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes, dtype=object).ravel()):
        if axis.get_visible():
            axis.text(x, y, letter, transform=axis.transAxes, fontsize=10,
                      fontweight="bold", va="top", ha="left")

def finish_axis(axis, *, grid_axis="y"):
    sns.despine(ax=axis)
    if grid_axis:
        axis.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.75, zorder=0)
    axis.set_axisbelow(True)

def export_figure(figure, stem):
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        figure.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    return figure

def mean_t_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, np.nan, np.nan
    half = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * stats.sem(values)
    return mean, mean - half, mean + half

def bootstrap_mean_ci(values, *, iterations=20_000, seed=17):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    ordered = valid[np.argsort(p_values[valid])]
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted

def rank_biserial(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())

print(f"Figure exports: {FIGURE_DIR}")
"""),
    md(r"""
## Data

### 1. Load and validate immutable source tables

The source tables are the controlling inputs. The checks below verify row grain, numeric
targets, missingness, and each source's original RT scale.
"""),
    code(r"""
raw_frames = {}
quality_records = []
anchor_key = ["sequence", "mods", "mod_sites"]

for organism, path in SOURCE_PATHS.items():
    frame = pd.read_csv(path, sep="\t", dtype={"mods": str, "mod_sites": str})
    frame["mods"] = frame["mods"].fillna("").astype(str)
    frame["mod_sites"] = frame["mod_sites"].fillna("").astype(str)
    frame["rt"] = pd.to_numeric(frame["rt"], errors="raise")
    frame["rt_norm"] = pd.to_numeric(frame["rt_norm"], errors="raise")
    frame["source_rt_norm"] = frame["rt_norm"]
    frame["source_organism"] = organism
    assert np.isfinite(frame[["rt", "rt_norm"]].to_numpy()).all()
    assert (frame["rt"] > 0).all()
    raw_frames[organism] = frame
    quality_records.append({
        "source": PLAIN_ORGANISM_LABELS[organism],
        "rows": len(frame),
        "unique_peptidoforms": frame[anchor_key].drop_duplicates().shape[0],
        "duplicate_peptidoform_keys": int(frame.duplicated(anchor_key).sum()),
        "missing_RT": int(frame["rt"].isna().sum()),
        "RT_min": frame["rt"].min(),
        "RT_median": frame["rt"].median(),
        "RT_max": frame["rt"].max(),
        "native_gradient_scale": np.polyfit(frame["rt_norm"], frame["rt"], 1)[0],
    })

raw_data = pd.concat(raw_frames.values(), ignore_index=True)
quality = pd.DataFrame(quality_records)
display(quality.style.format({
    "RT_min": "{:.2f}", "RT_median": "{:.2f}", "RT_max": "{:.2f}",
    "native_gradient_scale": "{:.4f}",
}).hide(axis="index"))
"""),
    code(r"""
# Figure 1 — raw/no-normalization baseline: distribution and robust range summaries.
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.35, 1]})

axis = axes[0]
for organism in ORGANISMS:
    values = np.sort(raw_frames[organism]["rt"].to_numpy(float))
    probability = np.arange(1, len(values) + 1) / len(values)
    axis.plot(values, probability, color=SOURCE_COLORS[organism],
              linestyle=SOURCE_LINES[organism],
              label=f"{ORGANISM_LABELS[organism]}  (n={len(values):,})")
axis.set(title="Observed RT distributions", xlabel="Observed RT (min)", ylabel="Cumulative fraction")
axis.legend(frameon=False, loc="lower right", handlelength=2.5)
finish_axis(axis, grid_axis="both")

axis = axes[1]
for position, organism in enumerate(ORGANISMS):
    values = raw_frames[organism]["rt"].to_numpy(float)
    q05, q25, median, q75, q95 = np.quantile(values, [.05, .25, .50, .75, .95])
    axis.plot([q05, q95], [position, position], color=SOURCE_COLORS[organism], linewidth=1.2)
    axis.plot([q25, q75], [position, position], color=SOURCE_COLORS[organism],
              linewidth=5, solid_capstyle="butt")
    axis.scatter(median, position, s=28, marker=SOURCE_MARKERS[organism], color=PAPER,
                 edgecolor=SOURCE_COLORS[organism], linewidth=1.1, zorder=3)
    axis.text(q95 + 1.0, position, f"median {median:.1f}", va="center", fontsize=7.2)
axis.set_yticks(range(len(ORGANISMS)), [ORGANISM_LABELS[item] for item in ORGANISMS])
axis.set_xlim(0, max(raw_data["rt"].max() * 1.08, 64))
axis.set(title="Robust RT ranges", xlabel="Observed RT (min)", ylabel="")
axis.invert_yaxis()
finish_axis(axis, grid_axis="x")

panel_letters(axes)
figure.suptitle("Observed retention times span different LC ranges", y=1.04, fontsize=11)
figure.tight_layout(w_pad=2.2)
export_figure(figure, "fig1_raw_rt_baseline")
plt.show()
"""),
    md(r"""
### 2. Fit every target representation on one leak-free calibration pool

For seed 2024, 10% of sequences are selected independently per source as test rows. The union
of all test sequences is removed from every source's calibration pool. Each method sees the
same pool and transforms the same held-out rows. The cell can take several seconds because it
fits official Calib-RT and predicts external Chronologer HI anchors.
"""),
    code(r"""
def nested_sequence_subset(frame, fraction, seed):
    sequences = np.asarray(sorted(frame["sequence"].unique()))
    rng = np.random.default_rng(seed)
    selected = set(rng.permutation(sequences)[:round(len(sequences) * fraction)])
    return frame.loc[frame["sequence"].isin(selected)].copy()

TEST_SEED = 2024
test_frames = {
    organism: nested_sequence_subset(frame, 0.10, TEST_SEED)
    for organism, frame in raw_frames.items()
}
forbidden_sequences = set().union(*(set(frame["sequence"]) for frame in test_frames.values()))
calibration_frames = {
    organism: frame.loc[~frame["sequence"].isin(forbidden_sequences)].copy()
    for organism, frame in raw_frames.items()
}

target_metadata = {}
transformed_tests = {}
transformed_calibration = {}
fit_progress = []

for method in METHODS:
    alignment = fit_target_alignment(calibration_frames, mode=method)
    transformed_tests[method], metadata = apply_rt_target(
        test_frames, mode=method, reference_minutes=REFERENCE_MINUTES, alignment=alignment
    )
    transformed_calibration[method], _ = apply_rt_target(
        calibration_frames, mode=method, reference_minutes=REFERENCE_MINUTES,
        alignment=alignment,
        inverse_coefficients=metadata["target_to_observed_rt"],
    )
    target_metadata[method] = metadata
    fit_progress.append({
        "target": METHOD_LABELS[method], "family": METHOD_FAMILIES[method],
        "calibration_rows": sum(len(item) for item in calibration_frames.values()),
        "test_rows": sum(len(item) for item in test_frames.values()),
        "fit_scope": metadata.get("alignment", {}).get("fit_scope", "fixed transformation"),
    })

display(pd.DataFrame(fit_progress).style.hide(axis="index"))
"""),
    code(r"""
# Verify finite targets and quantify the forward/inverse roundtrip used for observed-minute evaluation.
target_summary_rows = []
for method in METHODS:
    metadata = target_metadata[method]
    for organism in ORGANISMS:
        frame = transformed_tests[method][organism]
        target = frame["rt_norm"].to_numpy(float)
        restored = predictions_to_observed_rt(
            target, np.full(len(frame), organism), metadata
        )
        observed = frame["rt"].to_numpy(float)
        assert np.isfinite(target).all() and np.isfinite(restored).all()
        target_summary_rows.append({
            "target": METHOD_LABELS[method], "method": method,
            "source": PLAIN_ORGANISM_LABELS[organism],
            "rows": len(frame), "target_min": target.min(),
            "target_median": np.median(target), "target_max": target.max(),
            "inverse_roundtrip_MAE_min": np.mean(np.abs(restored - observed)),
            "inverse_roundtrip_P95_min": np.quantile(np.abs(restored - observed), .95),
        })

target_summary = pd.DataFrame(target_summary_rows)
display(target_summary.style.format({
    "target_min": "{:.3f}", "target_median": "{:.3f}", "target_max": "{:.3f}",
    "inverse_roundtrip_MAE_min": "{:.3f}", "inverse_roundtrip_P95_min": "{:.3f}",
}).hide(axis="index"))
"""),
    md(r"""
## Results

### 3. Transformation geometry

The following curves show the exact seed-2024 mapping from observed minutes to each model
target. Source identity is encoded redundantly by color and line style. Raw RT is included as
a descriptive reference and has a different y-axis unit.
"""),
    code(r"""
def target_from_observed_rt(method, organism, values):
    values = np.asarray(values, dtype=float)
    metadata = target_metadata[method]
    if method == "global_60min":
        return values / REFERENCE_MINUTES
    if method == "source_native":
        inverse = metadata["target_to_observed_rt"][organism]
        return (values - inverse["intercept"]) / inverse["slope"]
    alignment = metadata["alignment"]
    if method == "anchor_irt":
        coefficient = alignment["coefficients"][organism]
        return (values * coefficient["slope"] + coefficient["intercept"]) / REFERENCE_MINUTES
    aligned = transform_alignment_state(alignment["models"][organism], values)
    if method == "relative_rt":
        return aligned
    offset = float(alignment.get("target_offset", 0.0))
    scale = float(alignment.get("target_scale", REFERENCE_MINUTES))
    return (aligned - offset) / scale

representations = ("raw", *METHODS)
figure, axes = plt.subplots(3, 4, figsize=(7.2, 7.0))
for axis, method in zip(axes.ravel(), representations):
    for organism in ORGANISMS:
        values = np.linspace(0, raw_frames[organism]["rt"].max(), 260)
        target = values if method == "raw" else target_from_observed_rt(method, organism, values)
        axis.plot(values, target, color=SOURCE_COLORS[organism],
                  linestyle=SOURCE_LINES[organism], label=ORGANISM_LABELS[organism])
    axis.set_title(METHOD_LABELS[method])
    axis.set_xlabel("Observed RT (min)")
    axis.set_ylabel("RT (min)" if method == "raw" else "Model target, rt_norm")
    if method != "raw":
        axis.axhline(0.5, color=MUTED, linewidth=0.6, alpha=0.55)
    finish_axis(axis, grid_axis="both")

axes[0, 0].legend(frameon=False, loc="upper left", handlelength=2.5)
panel_letters(axes, x=-0.20, y=1.10)
figure.suptitle("Observed RT is mapped into distinct model target coordinates", y=1.005, fontsize=11)
figure.tight_layout(h_pad=1.45, w_pad=1.2)
export_figure(figure, "fig2_all_target_transformations")
plt.show()
"""),
    code(r"""
# Figure 3 — held-out target distributions; ECDF avoids bandwidth-dependent density artifacts.
figure, axes = plt.subplots(3, 4, figsize=(7.2, 6.65), sharey=True)
for axis, method in zip(axes.ravel(), METHODS):
    for organism in ORGANISMS:
        values = np.sort(transformed_tests[method][organism]["rt_norm"].to_numpy(float))
        probability = np.arange(1, len(values) + 1) / len(values)
        axis.plot(values, probability, color=SOURCE_COLORS[organism],
                  linestyle=SOURCE_LINES[organism], label=ORGANISM_LABELS[organism])
    axis.set_title(METHOD_LABELS[method])
    axis.set_xlabel("Model target, rt_norm")
    axis.set_ylabel("Cumulative fraction")
    finish_axis(axis, grid_axis="both")
for axis in axes.ravel()[len(METHODS):]:
    axis.set_visible(False)
axes[0, 0].legend(frameon=False, loc="lower right", handlelength=2.5)
panel_letters(axes, x=-0.20, y=1.10)
figure.suptitle("Held-out target distributions after alignment and scaling", y=1.005, fontsize=11)
figure.tight_layout(h_pad=1.45, w_pad=1.1)
export_figure(figure, "fig3_all_target_distributions")
plt.show()
"""),
    md(r"""
### 4. Cross-source consistency on shared calibration peptidoforms

For each source pair, identical training peptidoforms are compared after transformation.
Values are median absolute differences in `rt_norm`; lower values mean that a shared
peptidoform receives a more similar target across sources. These rows contributed to fitting
for anchor-based methods, so this measures calibration geometry rather than held-out accuracy.
"""),
    code(r"""
pair_labels = {
    ("celegans", "triplemix"): "CE–TM",
    ("celegans", "human"): "CE–HU",
    ("triplemix", "human"): "TM–HU",
}
consistency_rows = []
for method in METHODS:
    medians = {
        organism: transformed_calibration[method][organism][anchor_key + ["rt_norm"]]
        .groupby(anchor_key, as_index=False, dropna=False)["rt_norm"].median()
        for organism in ORGANISMS
    }
    for left, right in combinations(ORGANISMS, 2):
        overlap = medians[left].merge(
            medians[right], on=anchor_key, suffixes=("_left", "_right")
        )
        differences = overlap["rt_norm_left"] - overlap["rt_norm_right"]
        consistency_rows.append({
            "method": method, "target": METHOD_LABELS[method],
            "pair": pair_labels[(left, right)], "anchors": len(overlap),
            "median_abs_target_difference": np.median(np.abs(differences)),
            "mean_abs_target_difference": np.mean(np.abs(differences)),
            "Spearman_rho": stats.spearmanr(
                overlap["rt_norm_left"], overlap["rt_norm_right"]
            ).statistic,
        })

alignment_consistency = pd.DataFrame(consistency_rows)
consistency_summary = alignment_consistency.groupby(
    ["method", "target"], as_index=False
).agg(
    median_pairwise_gap=("median_abs_target_difference", "median"),
    worst_pairwise_gap=("median_abs_target_difference", "max"),
    minimum_shared_anchors=("anchors", "min"),
    median_spearman=("Spearman_rho", "median"),
).sort_values("median_pairwise_gap")
display(consistency_summary.style.format({
    "median_pairwise_gap": "{:.4f}", "worst_pairwise_gap": "{:.4f}",
    "median_spearman": "{:.3f}",
}).hide(axis="index"))
"""),
    code(r"""
# Figure 4 — method-by-source-pair consistency matrix.
matrix = alignment_consistency.pivot(
    index="method", columns="pair", values="median_abs_target_difference"
).reindex(index=METHODS, columns=list(pair_labels.values()))
anchor_counts = alignment_consistency.groupby("pair")["anchors"].first().reindex(matrix.columns)
column_labels = [f"{pair}\n(n={int(anchor_counts[pair]):,})" for pair in matrix.columns]

figure, axis = plt.subplots(figsize=(7.2, 4.65))
cmap = sns.light_palette("#3B6FB6", as_cmap=True)
sns.heatmap(
    matrix, ax=axis, cmap=cmap, annot=True, fmt=".3f", linewidths=.55,
    linecolor=PAPER, cbar_kws={"label": "Median |Δ target|"},
    yticklabels=[METHOD_LABELS[item] for item in matrix.index],
    xticklabels=column_labels,
)
axis.set(xlabel="Shared training peptidoform source pair", ylabel="Target representation",
         title="Cross-source target consistency on calibration anchors")
axis.tick_params(axis="x", rotation=0)
figure.tight_layout()
export_figure(figure, "fig4_cross_source_consistency")
plt.show()
"""),
    md(r"""
### 5. Distribution overlap and inverse-map fidelity

Distribution overlap is descriptive because organisms contain different peptide populations.
Wasserstein distances are calculated in normalized target units. Inverse roundtrip MAE tests
whether the separately fitted target-to-observed mapping can recover measured minutes; it is
not a neural-network error.
"""),
    code(r"""
overlap_rows = []
for method in METHODS:
    for left, right in combinations(ORGANISMS, 2):
        left_values = transformed_tests[method][left]["rt_norm"].to_numpy(float)
        right_values = transformed_tests[method][right]["rt_norm"].to_numpy(float)
        overlap_rows.append({
            "method": method, "pair": pair_labels[(left, right)],
            "wasserstein_target": wasserstein_distance(left_values, right_values),
        })
distribution_overlap = pd.DataFrame(overlap_rows)
representation_diagnostics = distribution_overlap.groupby("method", as_index=False).agg(
    mean_pairwise_wasserstein=("wasserstein_target", "mean"),
    max_pairwise_wasserstein=("wasserstein_target", "max"),
)
roundtrip = target_summary.groupby("method", as_index=False).agg(
    mean_inverse_roundtrip_MAE_min=("inverse_roundtrip_MAE_min", "mean"),
    max_inverse_roundtrip_P95_min=("inverse_roundtrip_P95_min", "max"),
)
representation_diagnostics = representation_diagnostics.merge(roundtrip, on="method")
representation_diagnostics["target"] = representation_diagnostics.method.map(METHOD_LABELS)
representation_diagnostics = representation_diagnostics.sort_values("mean_pairwise_wasserstein")
display(representation_diagnostics.style.format({
    "mean_pairwise_wasserstein": "{:.4f}", "max_pairwise_wasserstein": "{:.4f}",
    "mean_inverse_roundtrip_MAE_min": "{:.3f}", "max_inverse_roundtrip_P95_min": "{:.3f}",
}).hide(axis="index"))
"""),
    code(r"""
# Figure 5 — compact diagnostic comparison without assigning eleven unrelated colors.
plot_data = representation_diagnostics.set_index("method").reindex(METHODS).reset_index()
positions = np.arange(len(plot_data))
figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), sharey=True)

for axis, column, title, xlabel in (
    (axes[0], "mean_pairwise_wasserstein", "Held-out distribution separation",
     "Mean pairwise Wasserstein distance (rt_norm)"),
    (axes[1], "mean_inverse_roundtrip_MAE_min", "Forward/inverse map fidelity",
     "Mean roundtrip MAE (min)"),
):
    values = plot_data[column].to_numpy(float)
    axis.hlines(positions, 0, values, color=GRID, linewidth=1.0)
    axis.scatter(values, positions, color="#3B6FB6", s=24, zorder=3)
    axis.set_xlim(left=0)
    axis.set(title=title, xlabel=xlabel)
    finish_axis(axis, grid_axis="x")
axes[0].set_yticks(positions, [METHOD_LABELS[item] for item in plot_data.method])
axes[0].invert_yaxis()
axes[1].tick_params(labelleft=False)
panel_letters(axes, x=-0.18, y=1.08)
figure.tight_layout(w_pad=1.6)
export_figure(figure, "fig5_representation_diagnostics")
plt.show()
"""),
    md(r"""
## Trained DIA-NN and Carafe2 benchmark

### 6. Completion and metric ingestion

A planned method contains 14 data conditions × 5 seeds = 70 runs per model family. A run is
counted as evaluable only when its JSON contains `raw_mae_minutes`, ensuring all target spaces
are compared after inverse transformation to observed LC minutes.
"""),
    code(r"""
metric_records = []
inventory_rows = []
expected_runs = len(SEEDS) * len(CONDITION_ORDER)

for family, root in MODEL_ROOTS.items():
    for method in METHODS:
        metric_files = 0
        evaluable_runs = 0
        for seed in SEEDS:
            for condition in CONDITION_ORDER:
                path = root / method / "runs" / f"seed_{seed}" / condition / "organism_test_metrics.json"
                if not path.is_file():
                    continue
                metric_files += 1
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not all("raw_mae_minutes" in payload.get(item, {}) for item in ORGANISMS):
                    continue
                evaluable_runs += 1
                for organism in ORGANISMS:
                    item = payload[organism]
                    metric_records.append({
                        "model": family, "method": method, "seed": seed,
                        "condition": condition, "condition_order": CONDITION_ORDER.index(condition),
                        "test_organism": organism,
                        "MAE_min": item.get("raw_mae_minutes", np.nan),
                        "RMSE_min": item.get("raw_rmse_minutes", np.nan),
                        "bias_min": item.get("raw_bias_minutes", np.nan),
                        "heavy_MAE_min": item.get("raw_heavy_dimethyl_mae_minutes", np.nan),
                        "rows": item.get("rows", np.nan),
                    })
        inventory_rows.append({
            "model": family, "method": method, "target": METHOD_LABELS[method],
            "metrics_files": metric_files, "evaluable_runs": evaluable_runs,
            "planned_runs": expected_runs, "completion": evaluable_runs / expected_runs,
        })

run_inventory = pd.DataFrame(inventory_rows)
model_metrics = pd.DataFrame(metric_records)
snapshot_time = pd.Timestamp.now(tz="Europe/Berlin").strftime("%Y-%m-%d %H:%M %Z")
display(Markdown(f"**Benchmark snapshot:** {snapshot_time}"))
display(run_inventory.style.format({"completion": "{:.0%}"}).hide(axis="index"))
"""),
    code(r"""
# Figure 6 — completion matrix.
completion = run_inventory.pivot(index="model", columns="method", values="evaluable_runs").reindex(
    index=list(MODEL_ROOTS), columns=METHODS
)
figure, axis = plt.subplots(figsize=(7.2, 1.8))
cmap = sns.light_palette("#3B6FB6", as_cmap=True)
sns.heatmap(
    completion, ax=axis, cmap=cmap, vmin=0, vmax=expected_runs,
    annot=True, fmt=".0f", linewidths=.7, linecolor=PAPER,
    cbar_kws={"label": f"Evaluable runs (of {expected_runs})"},
    xticklabels=[METHOD_LABELS[item] for item in completion.columns],
)
axis.set(xlabel="Target representation", ylabel="Model family", title="Benchmark completion")
axis.tick_params(axis="x", rotation=48)
axis.tick_params(axis="y", rotation=0)
figure.tight_layout()
export_figure(figure, "fig6_benchmark_completion")
plt.show()
"""),
    md(r"""
### 7. Full multi-organism endpoint

Points are seed means and intervals are 95% Student-*t* confidence intervals when at least two
seeds exist. Every panel starts at zero. Missing methods are left blank rather than imputed.
"""),
    code(r"""
if model_metrics.empty:
    endpoint_summary = pd.DataFrame()
else:
    endpoint_data = model_metrics.loc[model_metrics.condition.eq("ce100_all_100")].copy()
    endpoint_summary = endpoint_data.groupby(
        ["model", "method", "test_organism"], as_index=False
    ).agg(mean_MAE=("MAE_min", "mean"), sd_MAE=("MAE_min", "std"), seeds=("seed", "nunique"))
    endpoint_summary["ci95"] = endpoint_summary.apply(
        lambda row: stats.t.ppf(.975, row.seeds - 1) * row.sd_MAE / math.sqrt(row.seeds)
        if row.seeds > 1 else np.nan, axis=1
    )

if endpoint_summary.empty:
    display(Markdown("**No evaluable full-endpoint runs are available yet.**"))
else:
    endpoint_table = endpoint_summary.copy()
    endpoint_table["target"] = endpoint_table.method.map(METHOD_LABELS)
    endpoint_table["organism"] = endpoint_table.test_organism.map(PLAIN_ORGANISM_LABELS)
    display(endpoint_table[[
        "model", "target", "organism", "seeds", "mean_MAE", "ci95"
    ]].sort_values(["model", "organism", "mean_MAE"]).style.format({
        "mean_MAE": "{:.3f}", "ci95": "{:.3f}",
    }).hide(axis="index"))
"""),
    code(r"""
# Figure 7 — endpoint dot-and-interval comparison.
figure, axes = plt.subplots(2, 3, figsize=(7.2, 7.15), sharey=True, squeeze=False)
positions = np.arange(len(METHODS))
for row, family in enumerate(MODEL_ROOTS):
    for column, organism in enumerate(ORGANISMS):
        axis = axes[row, column]
        subset = endpoint_summary.loc[
            endpoint_summary.model.eq(family) & endpoint_summary.test_organism.eq(organism)
        ] if not endpoint_summary.empty else pd.DataFrame()
        for position, method in enumerate(METHODS):
            item = subset.loc[subset.method.eq(method)] if not subset.empty else pd.DataFrame()
            if item.empty:
                continue
            value = item.iloc[0]
            xerr = value.ci95 if np.isfinite(value.ci95) else None
            axis.errorbar(value.mean_MAE, position, xerr=xerr, fmt="o",
                          color=MODEL_COLORS[family], markersize=4.2, capsize=2.2,
                          linewidth=1.0)
            axis.text(value.mean_MAE, position - .22, f"n={int(value.seeds)}",
                      color=MUTED, fontsize=6.3, ha="center")
        axis.set_xlim(left=0)
        axis.set_title(f"{family} | {ORGANISM_LABELS[organism]}")
        axis.set_xlabel("MAE (observed RT, min)")
        finish_axis(axis, grid_axis="x")
axes[0, 0].set_yticks(positions, [METHOD_LABELS[item] for item in METHODS])
axes[0, 0].invert_yaxis()
for axis in axes[:, 1:].ravel():
    axis.tick_params(labelleft=False)
panel_letters(axes, x=-0.24, y=1.07)
figure.suptitle("Prediction error at the full multi-organism endpoint", y=.995, fontsize=11)
figure.tight_layout(h_pad=1.3, w_pad=1.0)
export_figure(figure, "fig7_full_endpoint_mae")
plt.show()
"""),
    md(r"""
### 8. Paired effects relative to Source-native

For each model × organism endpoint, candidate methods are paired with Source-native by seed.
Positive gain means lower candidate MAE. Confidence intervals use a deterministic paired
bootstrap; Wilcoxon tests require at least three matched seeds and are Holm-adjusted across
all displayed endpoint contrasts. With only five seeds, effect magnitude and interval are more
informative than the discrete p-values.
"""),
    code(r"""
paired_rows = []
if not model_metrics.empty:
    endpoint_data = model_metrics.loc[model_metrics.condition.eq("ce100_all_100")]
    for (family, organism), group in endpoint_data.groupby(["model", "test_organism"]):
        pivot = group.pivot_table(index="seed", columns="method", values="MAE_min", aggfunc="first")
        if "source_native" not in pivot:
            continue
        for method in METHODS:
            if method == "source_native" or method not in pivot:
                continue
            paired = pivot[["source_native", method]].dropna()
            row = {"model": family, "test_organism": organism, "method": method,
                   "matched_seeds": len(paired)}
            if len(paired) >= 3:
                gains = paired["source_native"].to_numpy() - paired[method].to_numpy()
                low, high = bootstrap_mean_ci(gains)
                nonzero = gains[gains != 0]
                p_value = stats.wilcoxon(nonzero, method="auto").pvalue if len(nonzero) else 1.0
                row.update({
                    "mean_gain_min": gains.mean(), "CI95_low": low, "CI95_high": high,
                    "rank_biserial": rank_biserial(gains), "p_value": p_value,
                })
            paired_rows.append(row)

paired_effects = pd.DataFrame(paired_rows)
if paired_effects.empty or "p_value" not in paired_effects:
    display(Markdown("**No endpoint contrast currently has three matched seeds.**"))
else:
    paired_effects["p_holm"] = holm_adjust(paired_effects["p_value"])
    valid_effects = paired_effects.loc[paired_effects.p_value.notna()].copy()
    valid_effects["target"] = valid_effects.method.map(METHOD_LABELS)
    valid_effects["organism"] = valid_effects.test_organism.map(PLAIN_ORGANISM_LABELS)
    display(valid_effects[[
        "model", "target", "organism", "matched_seeds", "mean_gain_min",
        "CI95_low", "CI95_high", "rank_biserial", "p_value", "p_holm",
    ]].sort_values(["model", "organism", "mean_gain_min"], ascending=[True, True, False])
      .style.format({
          "mean_gain_min": "{:+.3f}", "CI95_low": "{:+.3f}", "CI95_high": "{:+.3f}",
          "rank_biserial": "{:+.2f}", "p_value": "{:.3g}", "p_holm": "{:.3g}",
      }).hide(axis="index"))
"""),
    code(r"""
# Figure 8 — paired endpoint gains; positive favors the candidate over Source-native.
figure, axes = plt.subplots(2, 3, figsize=(7.2, 6.7), sharey=True, squeeze=False)
candidates = tuple(method for method in METHODS if method != "source_native")
positions = np.arange(len(candidates))
for row, family in enumerate(MODEL_ROOTS):
    for column, organism in enumerate(ORGANISMS):
        axis = axes[row, column]
        subset = valid_effects.loc[
            valid_effects.model.eq(family) & valid_effects.test_organism.eq(organism)
        ] if "valid_effects" in globals() else pd.DataFrame()
        for position, method in enumerate(candidates):
            item = subset.loc[subset.method.eq(method)] if not subset.empty else pd.DataFrame()
            if item.empty:
                continue
            value = item.iloc[0]
            axis.errorbar(
                value.mean_gain_min, position,
                xerr=[[value.mean_gain_min - value.CI95_low], [value.CI95_high - value.mean_gain_min]],
                fmt="o", color=MODEL_COLORS[family], markersize=4.2, capsize=2.2,
            )
        axis.axvline(0, color=INK, linestyle="--", linewidth=.75)
        axis.set_title(f"{family} | {ORGANISM_LABELS[organism]}")
        axis.set_xlabel("MAE gain vs Source-native (min)")
        finish_axis(axis, grid_axis="x")
axes[0, 0].set_yticks(positions, [METHOD_LABELS[item] for item in candidates])
axes[0, 0].invert_yaxis()
for axis in axes[:, 1:].ravel():
    axis.tick_params(labelleft=False)
panel_letters(axes, x=-0.24, y=1.07)
figure.suptitle("Paired full-endpoint effects relative to Source-native", y=.995, fontsize=11)
figure.tight_layout(h_pad=1.25, w_pad=1.0)
export_figure(figure, "fig8_paired_endpoint_effects")
plt.show()
"""),
    md(r"""
### 9. Effects across the complete data-scaling grid

Each heatmap cell is the paired mean percentage reduction in observed-minute MAE relative to
Source-native. Positive values favor the candidate. Blank cells lack matched runs; separate
condition branches are divided by vertical rules and should not be read as one continuous
time series.
"""),
    code(r"""
scaling_gain_rows = []
if not model_metrics.empty:
    for (family, condition, condition_order, organism), group in model_metrics.groupby(
        ["model", "condition", "condition_order", "test_organism"]
    ):
        pivot = group.pivot_table(index="seed", columns="method", values="MAE_min", aggfunc="first")
        if "source_native" not in pivot:
            continue
        for method in METHODS:
            if method == "source_native" or method not in pivot:
                continue
            paired = pivot[["source_native", method]].dropna()
            if paired.empty:
                continue
            gain_pct = 100 * (paired["source_native"] - paired[method]) / paired["source_native"]
            scaling_gain_rows.append({
                "model": family, "condition": condition, "condition_order": condition_order,
                "test_organism": organism, "method": method,
                "matched_seeds": len(paired), "mean_gain_pct": gain_pct.mean(),
            })
scaling_gains = pd.DataFrame(scaling_gain_rows)

figure, axes = plt.subplots(2, 3, figsize=(7.2, 6.15), squeeze=False)
available_values = scaling_gains["mean_gain_pct"].to_numpy(float) if not scaling_gains.empty else np.array([1.0])
limit = max(10.0, float(np.nanquantile(np.abs(available_values), .95)))
cmap = sns.diverging_palette(35, 230, s=70, l=55, as_cmap=True)
for row, family in enumerate(MODEL_ROOTS):
    for column, organism in enumerate(ORGANISMS):
        axis = axes[row, column]
        subset = scaling_gains.loc[
            scaling_gains.model.eq(family) & scaling_gains.test_organism.eq(organism)
        ] if not scaling_gains.empty else pd.DataFrame()
        matrix = subset.pivot(index="method", columns="condition", values="mean_gain_pct").reindex(
            index=[item for item in METHODS if item != "source_native"], columns=CONDITION_ORDER
        ) if not subset.empty else pd.DataFrame(
            np.nan, index=[item for item in METHODS if item != "source_native"], columns=CONDITION_ORDER
        )
        sns.heatmap(matrix, ax=axis, cmap=cmap, center=0, vmin=-limit, vmax=limit,
                    mask=matrix.isna(), cbar=False, linewidths=.2, linecolor=PAPER,
                    yticklabels=[METHOD_LABELS[item] for item in matrix.index],
                    xticklabels=[CONDITION_LABELS[item] for item in matrix.columns])
        axis.set_facecolor(MISSING)
        for boundary in (5, 8, 11):
            axis.axvline(boundary, color=INK, linewidth=.65)
        axis.set_title(f"{family} | {ORGANISM_LABELS[organism]}")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=58)
        if column:
            axis.tick_params(labelleft=False)
norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                           orientation="horizontal", fraction=.035, pad=.10, aspect=45)
colorbar.set_label("Paired MAE reduction vs Source-native (%)")
panel_letters(axes, x=-0.25, y=1.07)
figure.suptitle("Alignment effects across nested data-scaling conditions", y=.995, fontsize=11)
figure.subplots_adjust(left=.17, right=.99, top=.94, bottom=.20, hspace=.42, wspace=.12)
export_figure(figure, "fig9_scaling_gain_heatmaps")
plt.show()
"""),
    md(r"""
## DIA-NN report

The report below ranks only methods with at least three full-endpoint seeds for each organism.
It is descriptive while the expanded benchmark is running; formal effects remain in the paired
table above.
"""),
    code(r"""
def display_model_report(family):
    if endpoint_summary.empty:
        display(Markdown(f"**{family}:** no full-endpoint results are available."))
        return pd.DataFrame()
    data = endpoint_summary.loc[
        endpoint_summary.model.eq(family) & endpoint_summary.seeds.ge(3)
    ].copy()
    if data.empty:
        display(Markdown(f"**{family}:** no method currently has three endpoint seeds."))
        return data
    winners = data.sort_values("mean_MAE").groupby("test_organism", as_index=False).first()
    statements = [
        f"{PLAIN_ORGANISM_LABELS[row.test_organism]}: {METHOD_LABELS[row.method]} "
        f"({row.mean_MAE:.3f} min, n={int(row.seeds)})"
        for row in winners.itertuples()
    ]
    complete_methods = run_inventory.loc[
        run_inventory.model.eq(family) & run_inventory.evaluable_runs.eq(expected_runs), "method"
    ].tolist()
    display(Markdown(
        f"**{family} snapshot.** Lowest available full-endpoint mean MAE by organism: "
        + "; ".join(statements)
        + f". Fully complete target arms: **{len(complete_methods)}/{len(METHODS)}**."
    ))
    table = data.assign(
        target=lambda x: x.method.map(METHOD_LABELS),
        organism=lambda x: x.test_organism.map(PLAIN_ORGANISM_LABELS),
    )[["target", "organism", "seeds", "mean_MAE", "ci95"]]
    display(table.sort_values(["organism", "mean_MAE"]).style.format({
        "mean_MAE": "{:.3f}", "ci95": "{:.3f}",
    }).hide(axis="index"))
    return data

dia_report = display_model_report("DIA-NN")
"""),
    md(r"""
## Carafe2 report

Carafe2 is evaluated with the same target definitions, sequence splits, conditions, and
observed-minute metrics as DIA-NN.
"""),
    code(r"""
carafe_report = display_model_report("Carafe2")
"""),
    md(r"""
## DIA-NN versus Carafe2: matched synthesis

Model-family differences are calculated only for identical method × seed × condition × test
organism rows. Positive values mean Carafe2 MAE minus DIA-NN MAE, so positive values favor
DIA-NN. This compares the complete observed training recipes, not architecture in isolation.
"""),
    code(r"""
head_to_head_rows = []
if not model_metrics.empty:
    pivot = model_metrics.pivot_table(
        index=["method", "seed", "condition", "condition_order", "test_organism"],
        columns="model", values="MAE_min", aggfunc="first",
    ).dropna(subset=list(MODEL_ROOTS))
    pivot["DIA_NN_gain_min"] = pivot["Carafe2"] - pivot["DIA-NN"]
    endpoint_h2h = pivot.reset_index().loc[lambda x: x.condition.eq("ce100_all_100")]
    for (method, organism), group in endpoint_h2h.groupby(["method", "test_organism"]):
        gains = group["DIA_NN_gain_min"].to_numpy(float)
        row = {"method": method, "test_organism": organism, "matched_seeds": len(gains),
               "DIA_NN_MAE": group["DIA-NN"].mean(), "Carafe2_MAE": group["Carafe2"].mean(),
               "DIA_NN_gain_min": gains.mean()}
        if len(gains) >= 3:
            low, high = bootstrap_mean_ci(gains)
            row.update({"CI95_low": low, "CI95_high": high})
        head_to_head_rows.append(row)
head_to_head = pd.DataFrame(head_to_head_rows)
if head_to_head.empty:
    display(Markdown("**No fully matched model-family endpoint is available.**"))
else:
    table = head_to_head.assign(
        target=lambda x: x.method.map(METHOD_LABELS),
        organism=lambda x: x.test_organism.map(PLAIN_ORGANISM_LABELS),
    )
    display(table[[
        "target", "organism", "matched_seeds", "DIA_NN_MAE", "Carafe2_MAE",
        "DIA_NN_gain_min", "CI95_low", "CI95_high",
    ]].style.format({
        "DIA_NN_MAE": "{:.3f}", "Carafe2_MAE": "{:.3f}",
        "DIA_NN_gain_min": "{:+.3f}", "CI95_low": "{:+.3f}", "CI95_high": "{:+.3f}",
    }).hide(axis="index"))
"""),
    code(r"""
# Figure 10 — matched model-family differences at the full endpoint.
figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.75), sharey=True)
positions = np.arange(len(METHODS))
for axis, organism in zip(axes, ORGANISMS):
    subset = head_to_head.loc[head_to_head.test_organism.eq(organism)] if not head_to_head.empty else pd.DataFrame()
    for position, method in enumerate(METHODS):
        item = subset.loc[subset.method.eq(method)] if not subset.empty else pd.DataFrame()
        if item.empty:
            continue
        value = item.iloc[0]
        if np.isfinite(value.get("CI95_low", np.nan)):
            xerr = [[value.DIA_NN_gain_min - value.CI95_low],
                    [value.CI95_high - value.DIA_NN_gain_min]]
        else:
            xerr = None
        axis.errorbar(value.DIA_NN_gain_min, position, xerr=xerr, fmt="o",
                      color="#3B6FB6", markersize=4.2, capsize=2.2)
    axis.axvline(0, color=INK, linestyle="--", linewidth=.75)
    axis.set_title(ORGANISM_LABELS[organism])
    axis.set_xlabel("Carafe2 MAE − DIA-NN MAE (min)")
    finish_axis(axis, grid_axis="x")
axes[0].set_yticks(positions, [METHOD_LABELS[item] for item in METHODS])
axes[0].invert_yaxis()
for axis in axes[1:]:
    axis.tick_params(labelleft=False)
panel_letters(axes, x=-0.23, y=1.08)
figure.suptitle("Matched model-family differences at the full endpoint", y=1.02, fontsize=11)
figure.tight_layout(w_pad=1.1)
export_figure(figure, "fig10_model_family_synthesis")
plt.show()
"""),
    md(r"""
## Takeaways

The cell below is deliberately generated from the executed snapshot. It distinguishes
representation diagnostics from trained-model evidence and updates automatically as new runs
finish.
"""),
    code(r"""
best_consistency = consistency_summary.iloc[0]
complete_count = int(run_inventory.evaluable_runs.eq(expected_runs).sum())
total_arms = len(run_inventory)
available_methods = sorted(model_metrics.method.unique()) if not model_metrics.empty else []

lines = [
    f"1. **Common target geometry:** {best_consistency.target} has the smallest median "
    f"cross-source calibration-anchor gap in this split "
    f"({best_consistency.median_pairwise_gap:.4f} rt_norm). This is a calibration diagnostic, "
    "not evidence that it predicts unseen peptides best.",
    f"2. **Inverse-map QA:** all eleven targets produced finite held-out values and finite "
    f"target-to-observed reconstructions; nonlinear roundtrip error is reported explicitly "
    "rather than assumed to be zero.",
    f"3. **Benchmark maturity:** {complete_count}/{total_arms} model × target arms are fully "
    f"complete at this snapshot. Evaluable methods currently present: "
    + (", ".join(METHOD_LABELS[item] for item in available_methods) if available_methods else "none") + ".",
]
if not endpoint_summary.empty:
    qualified = endpoint_summary.loc[endpoint_summary.seeds.ge(3)]
    if not qualified.empty:
        winners = qualified.sort_values("mean_MAE").groupby(
            ["model", "test_organism"], as_index=False
        ).first()
        winner_text = "; ".join(
            f"{row.model}/{PLAIN_ORGANISM_LABELS[row.test_organism]}: "
            f"{METHOD_LABELS[row.method]} ({row.mean_MAE:.3f} min)"
            for row in winners.itertuples()
        )
        lines.append(f"4. **Lowest currently available endpoint means:** {winner_text}. "
                     "These rankings can change until all target arms are complete.")
lines.append(
    "5. **Interpretation:** choose the production coordinate from observed-minute prediction "
    "performance and calibration robustness together. Distribution overlap alone is not a "
    "selection criterion because organism peptide compositions differ."
)
display(Markdown("\n".join(lines)))
"""),
    md(r"""
### Publication and reproducibility notes

- Method identity is carried mainly by facets and labels; colors encode only three sources or
  two model families, avoiding an eleven-color legend.
- Absolute-error axes start at zero. Signed effect plots show an explicit zero reference.
- PDF and SVG are preferred manuscript formats; PNG is exported at 600 dpi.
- Re-run the notebook after benchmark completion to refresh completion, endpoint, paired-effect,
  scaling, and model-family sections without changing analysis code.
"""),
    code(r"""
exported = sorted(path.name for path in FIGURE_DIR.iterdir() if path.is_file())
figure_stems = sorted({Path(name).stem for name in exported})
figure_manifest = pd.DataFrame({"figure": figure_stems})
figure_manifest["PNG_600dpi"] = figure_manifest.figure.map(
    lambda stem: (FIGURE_DIR / f"{stem}.png").is_file()
)
figure_manifest["vector_PDF"] = figure_manifest.figure.map(
    lambda stem: (FIGURE_DIR / f"{stem}.pdf").is_file()
)
figure_manifest["editable_SVG"] = figure_manifest.figure.map(
    lambda stem: (FIGURE_DIR / f"{stem}.svg").is_file()
)
assert figure_manifest[["PNG_600dpi", "vector_PDF", "editable_SVG"]].all().all()
display(figure_manifest.style.hide(axis="index"))

manifest_payload = {
    "notebook": str(TRAINING_ROOT / "norm_better.ipynb"),
    "executed_snapshot": snapshot_time,
    "methods": list(METHODS),
    "figures": figure_stems,
    "exports": ["PNG 600 dpi", "vector PDF", "editable SVG"],
}
(FIGURE_DIR / "figure_manifest.json").write_text(
    json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
)
print(f"Notebook source: {TRAINING_ROOT / 'norm_better.ipynb'}")
print(f"Figure directory: {FIGURE_DIR}")
"""),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} cells")
