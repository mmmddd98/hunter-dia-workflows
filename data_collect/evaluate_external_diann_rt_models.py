#!/usr/bin/env python3
"""Evaluate exported DIA-NN RT packages on one fixed partition and plot them.

This is the final-test entry point for external DIA-NN RT packages.  It does
no fitting, selection, alignment fitting, or checkpoint mutation.  It accepts
any partition following the train-fit layout (``diann_rt_input`` plus
``sources``) and an alignment that was fitted outside that partition.

Besides JSON metrics, the command writes peptidoform-level prediction tables
and a compact plot suite covering model-scale and observed-minute performance,
Heavy-Dimethyl strata, digestion classes, sources, and organisms.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_diann_trainfit import evaluate_run


HEAVY_ORDER = ("normal", "heavy_nterm", "heavy_lysine", "heavy_both")
HEAVY_LABELS = {
    "normal": "No Heavy-Dimethyl", "heavy_nterm": "Heavy N-term",
    "heavy_lysine": "Heavy Lys", "heavy_both": "Heavy N-term + Lys",
}
DIGESTION_ORDER = ("fully_tryptic", "semi_tryptic", "ambiguous")
DIGESTION_LABELS = {
    "fully_tryptic": "Fully tryptic", "semi_tryptic": "Semi-tryptic", "ambiguous": "Ambiguous digestion",
}
MODEL_COLORS = ("#2B6CB0", "#D97706", "#7C3AED", "#5F7A3B", "#C2416C", "#0F766E")
STRATUM_COLORS = {
    "normal": "#6B7280", "heavy_nterm": "#3B82B6", "heavy_lysine": "#D97706", "heavy_both": "#A855F7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument(
        "--model", type=Path, action="append",
        help="Package directory, model_package directory, or directory containing rt.d0.finetuned.pt. Repeatable.",
    )
    models.add_argument(
        "--models-list", type=Path,
        help="Registry CSV with relative_path entries; evaluates exactly its registered packages.",
    )
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Reuse existing prediction tables below --output-dir and regenerate analysis only.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 10, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#4B5563", "axes.linewidth": 0.7,
        "xtick.color": "#374151", "ytick.color": "#374151",
        "grid.color": "#E5E7EB", "grid.linewidth": 0.7,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output.with_suffix(suffix), dpi=320 if suffix == ".png" else None,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def manifest_for(package: Path, model_dir: Path) -> tuple[Path | None, dict]:
    candidates = (
        package / "model_package" / "model_manifest.json",
        package / "model_manifest.json",
        model_dir.parent / "model_manifest.json",
        model_dir.parent.parent / "model_manifest.json",
    )
    for path in candidates:
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None, {}


def resolve_model(package: Path) -> dict:
    package = package.resolve()
    candidates = (package / "model_package" / "model", package / "model", package)
    model_dir = next((path for path in candidates if (path / "rt.d0.finetuned.pt").is_file()), None)
    if model_dir is None or not (model_dir / "dict.txt").is_file():
        raise FileNotFoundError(
            f"{package} is not an exported DIA-NN RT package; expected model/rt.d0.finetuned.pt and dict.txt"
        )
    manifest_path, manifest = manifest_for(package, model_dir)
    label = str(manifest.get("package_name") or package.name)
    return {
        "package": package, "model_dir": model_dir, "manifest_path": manifest_path,
        "manifest": manifest, "label": label, "slug": safe_slug(label),
    }


def resolve_models(args: argparse.Namespace) -> list[dict]:
    if args.model:
        items = [(path, {}) for path in args.model]
    else:
        with args.models_list.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "relative_path" not in rows[0]:
            raise ValueError(f"{args.models_list} must be a non-empty models registry with a relative_path column")
        items = [(args.models_list.parent / row["relative_path"], row) for row in rows if row.get("relative_path")]
    models = []
    for path, registry in items:
        model = resolve_model(path)
        model["registry"] = registry
        models.append(model)
    labels = [model["label"] for model in models]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Model labels must be unique: {labels}")
    return models


def metric(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return float(np.mean(np.abs(values))) if len(values) else float("nan")


def source_balanced_mae(frame: pd.DataFrame, residual_column: str, mask: pd.Series | None = None) -> float:
    selected = frame if mask is None else frame.loc[mask]
    if selected.empty:
        return float("nan")
    per_source = selected.groupby("source_dataset_id", sort=True)[residual_column].apply(metric)
    return float(per_source.dropna().mean()) if not per_source.dropna().empty else float("nan")


def heavy_macro(records: dict[str, float]) -> float:
    values = [records.get(name, float("nan")) for name in HEAVY_ORDER[1:]]
    return float(np.mean(values)) if all(np.isfinite(value) for value in values) else float("nan")


def raw_stratum_records(frame: pd.DataFrame, *, organism: str | None = None) -> list[dict]:
    if organism is not None:
        frame = frame.loc[frame["source_organism"].eq(organism)]
    records: dict[str, float] = {
        "Overall": source_balanced_mae(frame, "raw_minutes_residual"),
    }
    for label in HEAVY_ORDER:
        records[HEAVY_LABELS[label]] = source_balanced_mae(
            frame, "raw_minutes_residual", frame["heavy_dimethyl_stratum"].eq(label),
        )
    records["Heavy-Dimethyl macro"] = heavy_macro({
        label: records[HEAVY_LABELS[label]] for label in HEAVY_ORDER[1:]
    })
    for label in DIGESTION_ORDER:
        records[DIGESTION_LABELS[label]] = source_balanced_mae(
            frame, "raw_minutes_residual", frame["digestion_class"].eq(label),
        )
    return [{"metric": name, "mae_minutes": value} for name, value in records.items()]


def load_payload(model_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metrics_path = model_dir / "metrics.json"
    prediction_dir = model_dir / "predictions"
    aggregate_path = prediction_dir / "aggregate_predictions.tsv"
    source_path = prediction_dir / "source_observation_predictions.tsv"
    for path in (metrics_path, aggregate_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed evaluation artifact: {path}")
    return (
        json.loads(metrics_path.read_text(encoding="utf-8")),
        pd.read_csv(aggregate_path, sep="\t", dtype={"mods": str, "mod_sites": str}),
        pd.read_csv(source_path, sep="\t", dtype={"mods": str, "mod_sites": str}),
    )


def summarize(models: list[dict], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    model_rows: list[dict] = []
    stratum_rows: list[dict] = []
    organism_rows: list[dict] = []
    predictions: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for model in models:
        run_dir = output_dir / model["slug"]
        payload, aggregate, sources = load_payload(run_dir)
        predictions[model["label"]] = (aggregate, sources)
        config = model["manifest"].get("configuration", {})
        registry = model.get("registry", {})
        common = {
            "model": model["label"], "model_id": model["manifest"].get("model_id", ""),
            "target": config.get("target") or registry.get("target_space", ""),
            "profile": config.get("profile", ""), "arm": config.get("arm", ""),
            "loss": config.get("loss", ""),
            "lr": config.get("lr", registry.get("lr_level", np.nan)),
        }
        model_rows.append({
            **common,
            "normalized_aggregate_mae": payload["aggregate"]["normalized"]["mae"],
            "observed_minutes_source_balanced_mae": payload["source_balanced_raw_mae"],
            "observed_minutes_all_observations_mae": payload["all_source_observations"]["raw_minutes"]["mae"],
            "aggregate_peptidoforms": payload["aggregate"]["normalized"]["rows"],
            "source_observations": payload["all_source_observations"]["raw_minutes"]["rows"],
        })
        stratum_rows.extend({**common, **row} for row in raw_stratum_records(sources))
        for organism in sorted(sources["source_organism"].dropna().astype(str).unique()):
            organism_rows.extend({**common, "organism": organism, **row}
                                 for row in raw_stratum_records(sources, organism=organism))
    return pd.DataFrame(model_rows), pd.DataFrame(stratum_rows), pd.DataFrame(organism_rows), predictions


def model_color_map(models: Iterable[str]) -> dict[str, str]:
    return {model: MODEL_COLORS[index % len(MODEL_COLORS)] for index, model in enumerate(models)}


def display_model_label(model: str) -> str:
    """Keep legends legible while preserving the full package name in tables."""
    lowered = model.lower()
    if "heavy_dimethyl_w" in lowered:
        profile = model.split("heavy_dimethyl_", 1)[1].split("_", 1)[0].upper()
        return f"{profile} · weighting"
    if "heavy_dimethyl_o" in lowered:
        profile = model.split("heavy_dimethyl_", 1)[1].split("_", 1)[0].upper()
        return f"{profile} · oversampling"
    if "diannrt_" in lowered:
        return "Unweighted full fine-tuning"
    return model


def display_organism(value: str) -> str:
    return {"celegans": "C. elegans", "human": "Human", "triplemix": "TripleMix"}.get(value, value)


def dot_comparison(axis: plt.Axes, frame: pd.DataFrame, *, categories: list[str], colors: dict[str, str], title: str, xlabel: str) -> None:
    models = list(colors)
    width = 0.70 / max(1, len(models))
    for index, model in enumerate(models):
        subset = frame.loc[frame["model"].eq(model)].set_index("metric").reindex(categories)
        offsets = np.arange(len(categories)) + (index - (len(models) - 1) / 2) * width
        axis.scatter(subset["mae_minutes"], offsets, s=30, color=colors[model], label=display_model_label(model), zorder=3)
    axis.set_yticks(np.arange(len(categories)), categories)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x")


def plot_stratum_mae(strata: pd.DataFrame, plots: Path, colors: dict[str, str]) -> None:
    categories = ["Overall", "No Heavy-Dimethyl", "Heavy-Dimethyl macro", "Heavy N-term", "Heavy Lys", "Heavy N-term + Lys", "Semi-tryptic"]
    figure, axis = plt.subplots(figsize=(10.3, 4.9), constrained_layout=True)
    dot_comparison(axis, strata, categories=categories, colors=colors,
                   title="Observed-minute error by peptide stratum",
                   xlabel="Source-balanced MAE (minutes; lower is better)")
    axis.legend(title="RT model", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.3, title_fontsize=7.5)
    save_figure(figure, plots / "01_observed_minutes_mae_by_stratum")


def plot_two_scales(summary: pd.DataFrame, plots: Path, colors: dict[str, str]) -> None:
    labels = list(colors)
    display_labels = [display_model_label(label) for label in labels]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, max(2.8, 0.55 * len(labels) + 1.6)), constrained_layout=True)
    specs = (
        ("normalized_aggregate_mae", "Model-coordinate MAE", "MAE on rt_norm scale"),
        ("observed_minutes_source_balanced_mae", "Observed-minute MAE", "Source-balanced MAE (minutes)"),
    )
    for axis, (column, title, xlabel) in zip(axes, specs):
        panel = summary.set_index("model").reindex(labels)
        axis.scatter(panel[column], np.arange(len(labels)), s=38, color=[colors[label] for label in labels], zorder=3)
        axis.set_yticks(np.arange(len(labels)), display_labels if axis is axes[0] else [])
        axis.invert_yaxis(); axis.grid(axis="x")
        axis.set_title(title); axis.set_xlabel(xlabel)
    figure.suptitle("Final-test performance on the two reported scales", fontsize=11)
    save_figure(figure, plots / "02_model_mae_on_two_scales")


def plot_residuals(predictions: dict[str, tuple[pd.DataFrame, pd.DataFrame]], plots: Path, colors: dict[str, str]) -> None:
    present = [label for label in HEAVY_ORDER if any((sources["heavy_dimethyl_stratum"] == label).any() for _, sources in predictions.values())]
    if not present:
        return
    columns = min(2, len(present)); rows = math.ceil(len(present) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.4 * rows), squeeze=False, constrained_layout=True)
    for axis, label in zip(axes.flat, present):
        values = []
        for model, (_, sources) in predictions.items():
            values.append(sources.loc[sources["heavy_dimethyl_stratum"].eq(label), "raw_minutes_residual"].dropna().to_numpy(float))
        boxes = axis.boxplot(values, patch_artist=True, labels=[display_model_label(model) for model in colors], showfliers=False, widths=0.62)
        for box, model in zip(boxes["boxes"], colors):
            box.set(facecolor=colors[model], alpha=0.62, edgecolor="#374151", linewidth=0.7)
        axis.axhline(0, color="#374151", linewidth=0.8)
        axis.set_title(HEAVY_LABELS[label]); axis.set_ylabel("Prediction − observed RT (minutes)")
        axis.tick_params(axis="x", rotation=25, labelsize=7); axis.grid(axis="y")
    for axis in axes.flat[len(present):]:
        axis.remove()
    figure.suptitle("Observed-minute residual distributions by Heavy-Dimethyl stratum", fontsize=11)
    save_figure(figure, plots / "03_observed_minutes_residuals_by_dimethyl_stratum")


def plot_organisms(organisms: pd.DataFrame, plots: Path, colors: dict[str, str]) -> None:
    if organisms.empty:
        return
    names = sorted(organisms["organism"].unique())
    columns = min(2, len(names)); rows = math.ceil(len(names) / columns)
    categories = ["Overall", "No Heavy-Dimethyl", "Heavy-Dimethyl macro", "Semi-tryptic"]
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.55 * rows), squeeze=False, constrained_layout=True)
    for axis, organism in zip(axes.flat, names):
        dot_comparison(axis, organisms.loc[organisms["organism"].eq(organism)], categories=categories,
                       colors=colors, title=display_organism(str(organism)), xlabel="Source-balanced MAE (minutes)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unused = list(axes.flat[len(names):])
    if unused:
        legend_axis = unused[0]
        legend_axis.axis("off")
        legend_axis.legend(handles, labels, title="RT model", loc="center", fontsize=7.2, title_fontsize=7.4)
        for axis in unused[1:]:
            axis.remove()
    else:
        figure.legend(handles, labels, title="RT model", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.2, title_fontsize=7.4)
    figure.suptitle("Observed-minute error by organism", fontsize=11)
    save_figure(figure, plots / "04_observed_minutes_mae_by_organism")


def plotting_sample(frame: pd.DataFrame, *, maximum_normal: int = 3500) -> pd.DataFrame:
    pieces = []
    for stratum, part in frame.groupby("heavy_dimethyl_stratum", sort=False):
        if stratum == "normal" and len(part) > maximum_normal:
            part = part.sample(maximum_normal, random_state=20260825)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else frame


def scatter_panel(axis: plt.Axes, frame: pd.DataFrame, *, x: str, y: str, title: str, xlabel: str, ylabel: str) -> None:
    panel = plotting_sample(frame)
    bounds = pd.concat([pd.to_numeric(panel[x], errors="coerce"), pd.to_numeric(panel[y], errors="coerce")]).dropna()
    if bounds.empty:
        axis.set_visible(False); return
    low, high = float(bounds.min()), float(bounds.max())
    padding = max((high - low) * 0.035, 1e-4)
    for stratum in HEAVY_ORDER:
        part = panel.loc[panel["heavy_dimethyl_stratum"].eq(stratum)]
        if not part.empty:
            axis.scatter(part[x], part[y], s=5.5, alpha=0.30, linewidths=0,
                         color=STRATUM_COLORS[stratum], label=HEAVY_LABELS[stratum], rasterized=True)
    axis.plot([low - padding, high + padding], [low - padding, high + padding], color="#111827", linewidth=0.8, linestyle="--", zorder=1)
    axis.set(xlim=(low - padding, high + padding), ylim=(low - padding, high + padding),
             title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.55)


def plot_model_calibration(model: str, aggregate: pd.DataFrame, sources: pd.DataFrame, plots: Path) -> None:
    source_names = sorted(sources["source_dataset_id"].astype(str).unique())
    columns = min(3, max(1, len(source_names)))
    rows = 1 + math.ceil(len(source_names) / columns)
    figure = plt.figure(figsize=(4.65 * columns, 3.85 * rows), constrained_layout=True)
    grid = figure.add_gridspec(rows, columns)
    normalized_axis = figure.add_subplot(grid[0, :])
    scatter_panel(normalized_axis, aggregate, x="rt_norm", y="prediction",
                  title="Model-coordinate prediction", xlabel="Test target (rt_norm)", ylabel="Prediction (rt_norm)")
    for index, source in enumerate(source_names):
        axis = figure.add_subplot(grid[1 + index // columns, index % columns])
        part = sources.loc[sources["source_dataset_id"].astype(str).eq(source)]
        organism = str(part["source_organism"].iloc[0]) if not part.empty else ""
        scatter_panel(axis, part, x="rt", y="observed_prediction",
                      title=f"{source} · {organism}", xlabel="Observed RT (minutes)", ylabel="Back-mapped prediction (minutes)")
    handles, labels = normalized_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=7.2)
    figure.suptitle(f"{model}: calibration on held-out peptides", fontsize=11)
    save_figure(figure, plots / "calibration" / f"{safe_slug(model)}_two_scale_calibration")


def write_readme(plots: Path) -> None:
    (plots / "PLOTS.md").write_text(
        """# External DIA-NN RT final-test plots

Every figure is produced from the fixed partition supplied to the evaluation
command.  The command only predicts and summarizes; it never trains a model or
fits an alignment.

| File | Primary question |
|---|---|
| `01_observed_minutes_mae_by_stratum.*` | Which model has the lowest source-balanced observed-minute error for non-heavy, every Heavy-Dimethyl stratum, and semi-tryptic peptides? |
| `02_model_mae_on_two_scales.*` | How do model-coordinate error and observed-minute error compare across packages? Units are intentionally separate. |
| `03_observed_minutes_residuals_by_dimethyl_stratum.*` | Are errors centred on zero and how wide are their distributions for each Heavy-Dimethyl class? |
| `04_observed_minutes_mae_by_organism.*` | Do overall, non-heavy, heavy-macro, or semi-tryptic errors differ by organism? |
| `calibration/*_two_scale_calibration.*` | For each model: prediction versus target on the model scale, then per-source prediction versus observed RT in minutes. The dashed line is ideal prediction. |

Source-balanced MAE assigns equal weight to every source with observations in
the relevant stratum.  Residual distributions and calibration panels retain
individual source observations and are therefore descriptive rather than a
replacement for the source-balanced headline metric.
""", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    models = resolve_models(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        for model in models:
            run_dir = output_dir / model["slug"]
            prediction_dir = run_dir / "predictions"
            metrics, aggregate, sources = evaluate_run(
                run_dir=model["model_dir"], partition_dir=args.partition_dir,
                alignment_path=args.alignment, device_name=args.device, batch_size=args.batch_size,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            prediction_dir.mkdir(exist_ok=True)
            aggregate.to_csv(prediction_dir / "aggregate_predictions.tsv", sep="\t", index=False)
            sources.to_csv(prediction_dir / "source_observation_predictions.tsv", sep="\t", index=False)

    summary, strata, organisms, predictions = summarize(models, output_dir)
    analysis = output_dir / "analysis"; analysis.mkdir(exist_ok=True)
    summary.to_csv(analysis / "model_summary.csv", index=False)
    strata.to_csv(analysis / "source_balanced_stratum_metrics.csv", index=False)
    organisms.to_csv(analysis / "source_balanced_organism_metrics.csv", index=False)
    colors = model_color_map(summary["model"].tolist())
    plot_stratum_mae(strata, analysis / "plots", colors)
    plot_two_scales(summary, analysis / "plots", colors)
    plot_residuals(predictions, analysis / "plots", colors)
    plot_organisms(organisms, analysis / "plots", colors)
    for model, (aggregate, sources) in predictions.items():
        plot_model_calibration(model, aggregate, sources, analysis / "plots")
    write_readme(analysis / "plots")
    run_manifest = {
        "pipeline": "external_diann_rt_final_test/v1", "partition_dir": str(args.partition_dir.resolve()),
        "alignment": str(args.alignment.resolve()), "device": args.device, "batch_size": args.batch_size,
        "models": [{
            "label": model["label"], "package": str(model["package"]), "model_dir": str(model["model_dir"]),
            "manifest": str(model["manifest_path"]) if model["manifest_path"] else None,
        } for model in models],
        "selection_note": "Evaluation and plotting only; no model selection, training, or alignment fitting occurs.",
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "models": len(models), "output_dir": str(output_dir), "analysis_dir": str(analysis),
        "plots": str(analysis / "plots"),
    }, indent=2))


if __name__ == "__main__":
    main()
