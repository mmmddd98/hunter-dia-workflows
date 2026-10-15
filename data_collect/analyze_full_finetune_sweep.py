#!/usr/bin/env python3
"""Create thesis-ready diagnostics for full multi-organism RT strategy sweeps."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORGANISMS = ("celegans", "triplemix", "human")
MODEL_COLORS = {"DIA-NN": "#0072B2", "Carafe2": "#D55E00"}
STRATEGY_COLORS = {
    "head": "#0072B2", "cnn_head": "#E69F00", "rnn_head": "#CC79A7",
    "gru_head": "#CC79A7", "all": "#6B7280", "discriminative": "#009E73",
}
PARAMETER_ALIASES = {
    "DIA-NN": {
        "loss": "loss", "strategy": "trainable", "learning_rate": "learning-rate",
        "backbone_lr": "backbone-learning-rate", "weight_decay": "weight-decay",
        "nterm_weight": "heavy-nterm-weight", "lysine_weight": "heavy-lysine-weight",
        "both_weight": "heavy-both-weight", "huber_delta": "huber-delta",
    },
    "Carafe2": {
        "loss": "rt_loss", "strategy": "rt_trainable", "learning_rate": "rt_lr",
        "backbone_lr": "rt_backbone_lr", "weight_decay": "rt_weight_decay",
        "nterm_weight": "rt_heavy_nterm_weight", "lysine_weight": "rt_heavy_lysine_weight",
        "both_weight": "rt_heavy_both_weight", "huber_delta": "rt_huber_delta",
    },
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diann-dir", type=Path, required=True)
    parser.add_argument("--carafe2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar_config(command_path: Path) -> dict:
    payload = json.loads(command_path.read_text(encoding="utf-8"))
    return payload.get("wandb_config", payload.get("config", {}))


def trainable_parameters(run_dir: Path, model: str) -> float:
    if model == "DIA-NN":
        path = run_dir / "run_metadata.json"
        if path.is_file():
            return float(json.loads(path.read_text(encoding="utf-8"))["result"].get(
                "trainable_parameters", np.nan
            ))
    else:
        path = run_dir / "training_diagnostics" / "data" / "rt_metadata.json"
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            return float(metadata.get("finetuning", {}).get("trainable_parameters", np.nan))
    return float("nan")


def collect(root: Path, model: str) -> pd.DataFrame:
    rows = []
    for command_path in sorted(root.glob("**/command.json")):
        run_dir = command_path.parent
        metrics_path = run_dir / "validation_metrics.json"
        if not metrics_path.is_file():
            metrics_path = run_dir / "organism_test_metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not all(name in metrics for name in ORGANISMS):
            continue
        config = scalar_config(command_path)
        aliases = PARAMETER_ALIASES[model]
        row = {
            "model": model,
            "origin_id": run_dir.parent.name,
            "seed": int(run_dir.name.removeprefix("seed.")),
            "run_dir": str(run_dir),
            "trainable_parameters": trainable_parameters(run_dir, model),
        }
        for canonical, source in aliases.items():
            row[canonical] = config.get(source, np.nan)
        signature_values = {
            canonical: row[canonical] for canonical in aliases
        }
        row["config_id"] = "cfg_" + hashlib.sha1(
            json.dumps(signature_values, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:10]
        for organism in ORGANISMS:
            item = metrics[organism]
            for metric in ("mae", "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae"):
                row[f"{organism}_{metric}"] = float(item[metric])
            row[f"{organism}_rows"] = int(item.get("rows", 0))
            row[f"{organism}_heavy_rows"] = int(item.get("heavy_dimethyl_rows", 0))
            row[f"{organism}_nterm_rows"] = int(item.get("heavy_nterm_dimethyl_rows", 0))
        row["macro_global_mae"] = np.mean([row[f"{item}_mae"] for item in ORGANISMS])
        row["macro_heavy_mae"] = np.mean([row[f"{item}_heavy_dimethyl_mae"] for item in ORGANISMS])
        row["macro_nterm_mae"] = np.mean([row[f"{item}_heavy_nterm_dimethyl_mae"] for item in ORGANISMS])
        row["composite_mae"] = (
            0.4 * row["macro_global_mae"]
            + 0.3 * row["macro_heavy_mae"]
            + 0.3 * row["macro_nterm_mae"]
        )
        test_path = run_dir / "final_test_metrics.json"
        if test_path.is_file():
            test_metrics = json.loads(test_path.read_text(encoding="utf-8"))
            for organism in ORGANISMS:
                for metric in ("mae", "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae"):
                    row[f"test_{organism}_{metric}"] = float(test_metrics[organism][metric])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["model", "config_id"]
    parameter_columns = [
        "loss", "strategy", "learning_rate", "backbone_lr", "weight_decay",
        "nterm_weight", "lysine_weight", "both_weight", "huber_delta",
        "trainable_parameters",
    ]
    metric_columns = [
        "composite_mae", "macro_global_mae", "macro_heavy_mae", "macro_nterm_mae",
        *[f"{organism}_{metric}" for organism in ORGANISMS for metric in (
            "mae", "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae"
        )],
        *[f"test_{organism}_{metric}" for organism in ORGANISMS for metric in (
            "mae", "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae"
        )],
    ]
    records = []
    for keys, group in runs.groupby(group_keys, sort=False):
        record = dict(zip(group_keys, keys))
        record["seeds"] = int(group.seed.nunique())
        for column in parameter_columns:
            values = group[column].dropna()
            record[column] = values.iloc[0] if len(values) else np.nan
        for column in metric_columns:
            values = (
                group[column].dropna().astype(float)
                if column in group.columns else pd.Series(dtype=float)
            )
            record[f"{column}_mean"] = values.mean() if len(values) else np.nan
            record[f"{column}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0 if len(values) else np.nan
            record[f"{column}_ci95"] = (
                1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1
                else 0.0 if len(values) else np.nan
            )
        records.append(record)
    result = pd.DataFrame(records)
    if not result.empty:
        result["rank_within_model"] = result.groupby("model")["composite_mae_mean"].rank(
            method="first"
        ).astype(int)
    return result.sort_values(["model", "composite_mae_mean"])


def save_all(figure: plt.Figure, base: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(base.with_suffix(suffix), dpi=450 if suffix == ".png" else None,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_ranking(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), sharex=False)
    for axis, model in zip(axes, ("DIA-NN", "Carafe2")):
        subset = summary[summary.model.eq(model)].nsmallest(15, "composite_mae_mean").sort_values(
            "composite_mae_mean", ascending=False
        )
        labels = [f"{row.strategy} | {row.loss} | {row.config_id.replace('trial.', '')[:7]}"
                  for row in subset.itertuples()]
        axis.barh(np.arange(len(subset)), subset.composite_mae_mean,
                  xerr=subset.composite_mae_ci95, color=MODEL_COLORS[model], alpha=.82,
                  edgecolor="#263238", linewidth=.4, capsize=2)
        axis.set_yticks(np.arange(len(subset)), labels)
        axis.set_xlabel("Validation composite MAE (lower is better)")
        axis.set_title(f"{model}: top configurations")
        axis.grid(axis="x", linestyle=":", alpha=.4)
    figure.suptitle("Full multi-organism fine-tuning configuration ranking", fontweight="bold")
    figure.tight_layout()
    save_all(figure, output / "fig1_configuration_ranking")


def plot_pareto(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True, sharey=True)
    for axis, model in zip(axes, ("DIA-NN", "Carafe2")):
        subset = summary[summary.model.eq(model)]
        for strategy, group in subset.groupby("strategy", dropna=False):
            key = str(strategy)
            axis.scatter(group.macro_global_mae_mean, group.macro_nterm_mae_mean,
                         s=42, alpha=.78, color=STRATEGY_COLORS.get(key, "#6B7280"),
                         edgecolor="#263238", linewidth=.35, label=key)
        best = subset.nsmallest(1, "composite_mae_mean")
        if len(best):
            axis.scatter(best.macro_global_mae_mean, best.macro_nterm_mae_mean,
                         s=150, marker="*", facecolor="none", edgecolor="#111827", linewidth=1.2)
        axis.set_title(model)
        axis.set_xlabel("Macro global MAE")
        axis.set_ylabel("Macro N-term Heavy-Dimethyl MAE")
        axis.grid(linestyle=":", alpha=.4)
        axis.legend(frameon=False, title="Trainable policy", fontsize=7)
    figure.suptitle("Global versus N-term Heavy-Dimethyl validation performance", fontweight="bold")
    figure.tight_layout()
    save_all(figure, output / "fig2_global_nterm_pareto")


def plot_strategy_loss(runs: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for axis, model in zip(axes, ("DIA-NN", "Carafe2")):
        subset = runs[runs.model.eq(model)].copy()
        subset["policy_loss"] = subset.strategy.astype(str) + "\n" + subset.loss.astype(str).str.upper()
        categories = sorted(subset.policy_loss.unique())
        values = [subset.loc[subset.policy_loss.eq(category), "composite_mae"] for category in categories]
        boxes = axis.boxplot(values, labels=categories, patch_artist=True, showfliers=False)
        for box in boxes["boxes"]:
            box.set(facecolor=MODEL_COLORS[model], alpha=.38, edgecolor="#263238")
        for index, values_for_category in enumerate(values, start=1):
            jitter = np.linspace(-.12, .12, len(values_for_category)) if len(values_for_category) else []
            axis.scatter(index + jitter, values_for_category, s=15, color=MODEL_COLORS[model], alpha=.75)
        axis.set_title(model)
        axis.set_ylabel("Validation composite MAE")
        axis.tick_params(axis="x", rotation=55)
        axis.grid(axis="y", linestyle=":", alpha=.4)
    figure.suptitle("Fine-tuning policy and optimizer loss", fontweight="bold")
    figure.tight_layout()
    save_all(figure, output / "fig3_policy_loss_comparison")


def plot_weights(runs: pd.DataFrame, output: Path) -> None:
    specifications = (
        ("nterm_weight", "macro_nterm_mae", "N-term weight", "Macro N-term MAE"),
        ("lysine_weight", "macro_heavy_mae", "Lysine weight", "Macro Heavy MAE"),
        ("both_weight", "macro_nterm_mae", "Both-sites weight", "Macro N-term MAE"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(11.8, 11.2))
    for row_index, (weight, metric, xlabel, ylabel) in enumerate(specifications):
        for column_index, model in enumerate(("DIA-NN", "Carafe2")):
            axis = axes[row_index, column_index]
            subset = runs[runs.model.eq(model)]
            for strategy, group in subset.groupby("strategy", dropna=False):
                key = str(strategy)
                axis.scatter(group[weight], group[metric], s=32, alpha=.72,
                             color=STRATEGY_COLORS.get(key, "#6B7280"),
                             edgecolor="#263238", linewidth=.3, label=key)
            axis.axvline(1.0, color="#4B5563", linestyle="--", linewidth=.9)
            axis.set_xscale("log")
            axis.set_title(f"{model} | {xlabel}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(linestyle=":", alpha=.35)
            if row_index == 0:
                axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle("Heavy-Dimethyl loss-weight response", fontweight="bold")
    figure.tight_layout()
    save_all(figure, output / "fig4_heavy_weight_response")


def plot_seed_confirmation(summary: pd.DataFrame, output: Path) -> None:
    confirmed = summary[summary.seeds.gt(1)].copy()
    if confirmed.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.6), sharex=False)
    for axis, model in zip(axes, ("DIA-NN", "Carafe2")):
        subset = confirmed[confirmed.model.eq(model)].nsmallest(10, "composite_mae_mean").sort_values(
            "composite_mae_mean", ascending=False
        )
        axis.errorbar(subset.composite_mae_mean, np.arange(len(subset)),
                      xerr=subset.composite_mae_ci95, fmt="o", color=MODEL_COLORS[model], capsize=3)
        axis.set_yticks(np.arange(len(subset)), [
            f"{row.strategy} | {row.loss} | n={row.seeds}" for row in subset.itertuples()
        ])
        axis.set_xlabel("Composite MAE, mean ± 95% CI")
        axis.set_title(model)
        axis.grid(axis="x", linestyle=":", alpha=.4)
    figure.suptitle("Multi-seed finalist confirmation", fontweight="bold")
    figure.tight_layout()
    save_all(figure, output / "fig5_multiseed_confirmation")


def plot_final_test(summary: pd.DataFrame, output: Path) -> None:
    test_columns = [column for column in summary.columns if column.startswith("test_") and column.endswith("_mean")]
    if not test_columns or summary[test_columns].notna().sum().sum() == 0:
        return
    tested = summary[summary[test_columns].notna().any(axis=1)]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=False)
    subsets = (
        ("mae", "All peptides"),
        ("heavy_dimethyl_mae", "Any Heavy-Dimethyl"),
        ("heavy_nterm_dimethyl_mae", "N-term Heavy-Dimethyl"),
    )
    width = .36
    for axis, organism in zip(axes, ORGANISMS):
        positions = np.arange(len(subsets))
        for offset, model in zip((-.5, .5), ("DIA-NN", "Carafe2")):
            row = tested[tested.model.eq(model)].nsmallest(1, "composite_mae_mean")
            if row.empty:
                continue
            means = [float(row[f"test_{organism}_{metric}_mean"].iloc[0]) for metric, _ in subsets]
            intervals = [float(row[f"test_{organism}_{metric}_ci95"].iloc[0]) for metric, _ in subsets]
            axis.bar(positions + offset * width, means, width, yerr=intervals, capsize=3,
                     color=MODEL_COLORS[model], edgecolor="#263238", linewidth=.4, label=model)
        axis.set_xticks(positions, [label.replace(" ", "\n", 1) for _, label in subsets])
        axis.set_title(f"{organism.capitalize()} final test")
        axis.set_ylabel("MAE (normalized RT)")
        axis.grid(axis="y", linestyle=":", alpha=.4)
        axis.legend(frameon=False)
    figure.suptitle("Final held-out test performance of validation-selected winners", fontweight="bold")
    figure.tight_layout()
    save_all(figure, output / "fig6_final_heldout_test")


def write_report(runs: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    figures = sorted(path.name for path in output.glob("fig*.png"))
    image_html = "\n".join(
        f"<h2>{html.escape(path.removesuffix('.png').replace('_', ' ').title())}</h2>"
        f"<img src='{html.escape(path)}' style='max-width:100%' alt='{html.escape(path)}'>"
        for path in figures
    )
    top = summary.groupby("model", group_keys=False).head(10)
    report = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Full multi-organism RT fine-tuning sweep</title>
<style>body{{font-family:DejaVu Sans,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;color:#1f2937}}
table{{border-collapse:collapse;font-size:.82rem}}th,td{{border:1px solid #d1d5db;padding:.35rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{border:1px solid #e5e7eb;margin-bottom:1.5rem}}</style></head>
<body><h1>Full multi-organism RT fine-tuning strategy sweep</h1>
<p>Composite validation metric = 0.4 × macro global MAE + 0.3 × macro Heavy-Dimethyl MAE +
0.3 × macro N-term Heavy-Dimethyl MAE. All validation subsets are sequence-disjoint from training and final test.</p>
<p>Completed runs: {len(runs)}; unique configurations: {len(summary)}.</p>
<h2>Top configurations</h2>{top.to_html(index=False, float_format=lambda value: f'{value:.6g}')}
{image_html}</body></html>"""
    (output / "sweep_report.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.concat([
        collect(args.diann_dir, "DIA-NN"), collect(args.carafe2_dir, "Carafe2")
    ], ignore_index=True)
    if runs.empty:
        raise SystemExit("No completed sweep runs with organism_test_metrics.json found")
    summary = summarize(runs)
    runs.to_csv(args.output_dir / "sweep_runs.csv", index=False)
    summary.to_csv(args.output_dir / "sweep_configuration_summary.csv", index=False)
    plot_ranking(summary, args.output_dir)
    plot_pareto(summary, args.output_dir)
    plot_strategy_loss(runs, args.output_dir)
    plot_weights(runs, args.output_dir)
    plot_seed_confirmation(summary, args.output_dir)
    plot_final_test(summary, args.output_dir)
    chart_map = {
        "fig1": {"question": "Which configurations minimize the prespecified composite MAE?", "family": "ranked bar with uncertainty"},
        "fig2": {"question": "What is the global versus N-term-heavy trade-off?", "family": "faceted scatter"},
        "fig3": {"question": "How do freeze policy and loss relate to performance?", "family": "box plot with observations"},
        "fig4": {"question": "How does each exact-stratum weight relate to heavy-subset MAE?", "family": "faceted scatter with unit-weight reference"},
        "fig5": {"question": "Which finalists reproduce across seeds?", "family": "dot and 95% interval"},
        "fig6": {"question": "How do validation-selected winners perform on the untouched final test?", "family": "grouped bar with 95% interval"},
        "palette": {"models": MODEL_COLORS, "strategies": STRATEGY_COLORS},
        "exports": ["PNG 450 dpi", "vector PDF", "editable SVG", "CSV", "HTML"],
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(chart_map, indent=2) + "\n", encoding="utf-8"
    )
    write_report(runs, summary, args.output_dir)
    print(f"Wrote full-data sweep analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
