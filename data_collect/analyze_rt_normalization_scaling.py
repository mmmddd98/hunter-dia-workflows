#!/usr/bin/env python3
"""Create publication-ready comparisons for the RT-alignment scaling study."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_organism_scaling_study import CONDITIONS, DISPLAY, LABELS, ORGANISMS


MODES = (
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
MODE_LABELS = {
    "global_60min": "Global 60 min",
    "source_native": "Source-native",
    "anchor_irt": "Joint anchor iRT",
    "linear": "Direct robust linear",
    "relative_rt": "Relative gradient RT",
    "lowess": "LOWESS",
    "isotonic": "Isotonic",
    "monotonic_spline": "Monotonic spline",
    "calib_rt": "Calib-RT",
    "kde": "KDE ridge",
    "chronologer_hi": "Chronologer HI",
}
MODE_SHORT_LABELS = {
    "global_60min": "Global",
    "source_native": "Native",
    "anchor_irt": "Joint iRT",
    "linear": "Linear",
    "relative_rt": "Relative",
    "lowess": "LOWESS",
    "isotonic": "Isotonic",
    "monotonic_spline": "Mono. spline",
    "calib_rt": "Calib-RT",
    "kde": "KDE",
    "chronologer_hi": "Chrono. HI",
}
MODE_COLORS = {
    "global_60min": "#7A7A7A",
    "source_native": "#E69F00",
    "anchor_irt": "#009E73",
    "linear": "#0072B2",
    "relative_rt": "#CC79A7",
    "lowess": "#56B4E9",
    "isotonic": "#D55E00",
    "monotonic_spline": "#6F4C9B",
    "calib_rt": "#222222",
    "kde": "#44AA99",
    "chronologer_hi": "#AA4499",
}
FAMILY_COLORS = {"diann": "#0072B2", "carafe2": "#D55E00"}
# All primary comparisons are transformed back to observed LC minutes using
# training-only calibration.  This prevents a merely smaller numeric target
# range from looking like an improvement.
SUBSETS = (("raw_mae_minutes", "All peptides"),
           ("raw_heavy_dimethyl_mae_minutes", "Any Heavy-Dimethyl"),
           ("raw_heavy_nterm_dimethyl_mae_minutes", "N-term Heavy-Dimethyl"))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diann-root", type=Path, required=True)
    parser.add_argument("--carafe2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def collect(root: Path, family: str) -> pd.DataFrame:
    records = []
    for mode in MODES:
        for run_dir in sorted((root / mode / "runs").glob("seed_*/*")):
            if run_dir.name not in CONDITIONS:
                continue
            metrics_path = run_dir / "organism_test_metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for organism in ORGANISMS:
                item = metrics.get(organism, {})
                if "raw_mae_minutes" not in item:
                    continue
                records.append({
                    "family": family, "mode": mode,
                    "seed": int(run_dir.parent.name.removeprefix("seed_")),
                    "condition": run_dir.name, "condition_order": CONDITIONS.index(run_dir.name),
                    "test_organism": organism,
                    **{metric: item.get(metric, np.nan) for metric, _ in SUBSETS},
                    "rows": item.get("rows", np.nan),
                })
    return pd.DataFrame(records)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "mode", "condition", "condition_order", "test_organism"]
    result = data.groupby(keys, as_index=False).agg(
        repeats=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric, _ in SUBSETS},
        **{f"{metric}_std": (metric, "std") for metric, _ in SUBSETS},
    )
    for metric, _ in SUBSETS:
        result[f"{metric}_ci95"] = 1.96 * result[f"{metric}_std"].fillna(0) / np.sqrt(result.repeats)
    return result


def save_all(figure: plt.Figure, base: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(base.with_suffix(suffix), dpi=450 if suffix == ".png" else None,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def figure_endpoint(summary: pd.DataFrame, output: Path) -> None:
    endpoint = summary[summary["condition"].eq("ce100_all_100")]
    figure, axes = plt.subplots(2, 3, figsize=(15.4, 8.2), sharey="row")
    positions = np.arange(len(MODES))
    for row, family in enumerate(("diann", "carafe2")):
        for col, organism in enumerate(ORGANISMS):
            axis = axes[row, col]
            subset = endpoint[
                endpoint["family"].eq(family) & endpoint["test_organism"].eq(organism)
            ]
            for index, mode in enumerate(MODES):
                values = subset[subset["mode"].eq(mode)]
                if values.empty:
                    continue
                value = values.iloc[0]
                axis.bar(index, value.raw_mae_minutes_mean, .72,
                         yerr=value.raw_mae_minutes_ci95, capsize=3,
                         color=MODE_COLORS[mode], edgecolor="#263238", linewidth=.4,
                         label=MODE_LABELS[mode] if row == 0 and col == 0 else None)
            axis.set_xticks(
                positions,
                [MODE_SHORT_LABELS[item] for item in MODES],
                rotation=48,
                ha="right",
            )
            axis.set_title(f"{DISPLAY[family]} | {organism.capitalize()}")
            axis.set_ylabel("MAE (observed RT, min)")
            axis.grid(axis="y", linestyle=":", alpha=.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    figure.suptitle("RT alignment at the full multi-organism endpoint", fontsize=13, fontweight="bold", y=.995)
    figure.tight_layout(rect=(0, 0, 1, .90))
    save_all(figure, output / "fig1_normalization_endpoint_global_mae")


def figure_trajectories(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.4, 8.2), sharex=True)
    for row, family in enumerate(("diann", "carafe2")):
        for col, organism in enumerate(ORGANISMS):
            axis = axes[row, col]
            subset = summary[
                summary["family"].eq(family) & summary["test_organism"].eq(organism)
            ]
            for mode in MODES:
                group = subset[subset["mode"].eq(mode)]
                if group.empty:
                    continue
                group = group.sort_values("condition_order")
                axis.errorbar(group.condition_order, group.raw_mae_minutes_mean,
                              yerr=group.raw_mae_minutes_ci95,
                              color=MODE_COLORS[mode], marker="o", markersize=3.8,
                              linewidth=1.8, capsize=2, label=MODE_LABELS[mode])
            axis.axvline(4.5, color="#999999", linestyle="--", linewidth=.8)
            axis.set_title(f"{DISPLAY[family]} | {organism.capitalize()}")
            axis.set_ylabel("MAE (observed RT, min)")
            axis.set_xticks(range(len(CONDITIONS)), [LABELS[item] for item in CONDITIONS], rotation=58, ha="right")
            axis.grid(axis="y", linestyle=":", alpha=.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    figure.suptitle("Data-scaling trajectories across RT target definitions", fontsize=13, fontweight="bold", y=.995)
    figure.tight_layout(rect=(0, 0, 1, .90))
    save_all(figure, output / "fig2_normalization_scaling_trajectories")


def figure_heavy_endpoint(summary: pd.DataFrame, output: Path) -> None:
    endpoint = summary[summary["condition"].eq("ce100_all_100")]
    figure, axes = plt.subplots(2, 3, figsize=(15.4, 8.2), sharey="row")
    width = .34
    for row, (metric, label) in enumerate(SUBSETS[1:]):
        for col, organism in enumerate(ORGANISMS):
            axis = axes[row, col]
            for family, offset in (("diann", -.18), ("carafe2", .18)):
                subset = endpoint[
                    endpoint["family"].eq(family) & endpoint["test_organism"].eq(organism)
                ]
                for index, mode in enumerate(MODES):
                    values = subset[subset["mode"].eq(mode)]
                    if values.empty:
                        continue
                    value = values.iloc[0]
                    axis.bar(index + offset, value[f"{metric}_mean"], width,
                             yerr=value[f"{metric}_ci95"], capsize=2,
                             color=FAMILY_COLORS[family], alpha=.82,
                             edgecolor="#263238", linewidth=.35,
                             label=DISPLAY[family] if row == 0 and col == 0 and index == 0 else None)
            axis.set_xticks(
                range(len(MODES)),
                [MODE_SHORT_LABELS[item] for item in MODES],
                rotation=48,
                ha="right",
            )
            axis.set_title(f"{label} | {organism.capitalize()}")
            axis.set_ylabel("MAE (observed RT, min)")
            axis.grid(axis="y", linestyle=":", alpha=.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Heavy-Dimethyl endpoints: model family and RT alignment", fontsize=13, fontweight="bold", y=.995)
    figure.tight_layout(rect=(0, 0, 1, .92))
    save_all(figure, output / "fig3_normalization_endpoint_heavy_mae")


def write_report(summary: pd.DataFrame, output: Path) -> None:
    endpoint = summary[summary["condition"].eq("ce100_all_100")].sort_values(
        ["family", "test_organism", "mode"]
    )
    figures = ["fig1_normalization_endpoint_global_mae.png", "fig2_normalization_scaling_trajectories.png", "fig3_normalization_endpoint_heavy_mae.png"]
    images = "".join(f"<h2>{html.escape(Path(item).stem.replace('_', ' ').title())}</h2><img src='{html.escape(item)}' alt='{html.escape(item)}'>" for item in figures)
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>RT alignment scaling study</title>
<style>body{{font-family:DejaVu Sans,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#1f2937}}table{{border-collapse:collapse;font-size:.84rem}}th,td{{border:1px solid #d1d5db;padding:.35rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #e5e7eb;margin-bottom:1.5rem}}</style></head>
<body><h1>RT alignment and multi-organism data scaling</h1><h2>Study definition</h2><p>The same sequence-disjoint splits, nested data conditions and seeds are compared across eleven RT target definitions: two normalization baselines, joint affine anchor iRT, direct robust linear alignment, relative gradient RT, LOWESS, isotonic regression, monotonic PCHIP splines, Calib-RT, a Chronologer-style run-to-run KDE ridge and RT-to-Chronologer hydrophobic-index alignment. Alignment and inverse mappings are fitted only on the non-test pool. All primary metrics are transformed back to source-observed LC minutes, so errors remain directly comparable across target definitions.</p><p><strong>Interpretation caveat:</strong> run-to-run methods use shared training peptidoforms with Human as reference, whereas Chronologer-HI uses all compatible unmodified training peptides plus the fixed external Chronologer predictor. Its result therefore measures an external-reference strategy, not only a different curve-fitting algorithm.</p>
<h2>Full multi-organism endpoint</h2>{endpoint.to_html(index=False, float_format=lambda value: f'{value:.6g}')}{images}</body></html>"""
    (output / "normalization_scaling_report.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.concat([collect(args.diann_root, "diann"), collect(args.carafe2_root, "carafe2")], ignore_index=True)
    if data.empty:
        raise SystemExit("No completed normalization-scaling runs found")
    summary = summarize(data)
    data.to_csv(args.output_dir / "normalization_scaling_runs.csv", index=False)
    summary.to_csv(args.output_dir / "normalization_scaling_summary.csv", index=False)
    figure_endpoint(summary, args.output_dir)
    figure_trajectories(summary, args.output_dir)
    figure_heavy_endpoint(summary, args.output_dir)
    write_report(summary, args.output_dir)
    (args.output_dir / "figure_manifest.json").write_text(json.dumps({
        "study": "rt_alignment_scaling_v2", "modes": list(MODES),
        "figures": ["fig1_normalization_endpoint_global_mae", "fig2_normalization_scaling_trajectories", "fig3_normalization_endpoint_heavy_mae"],
        "exports": ["PNG 450 dpi", "vector PDF", "editable SVG", "CSV", "HTML"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote normalization-scaling analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
