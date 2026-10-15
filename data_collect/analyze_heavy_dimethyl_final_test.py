#!/usr/bin/env python3
"""Summarize a frozen Heavy-Dimethyl final-test evaluation without reselection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heavy_dimethyl_sweep_common import heavy_macro


METRICS = (
    "heavy_macro_source_balanced_mae", "heavy_nterm_sb_mae", "heavy_lys_sb_mae",
    "heavy_both_sb_mae", "nonheavy_sb_mae", "overall_mae", "semitryptic_mae", "worst_source_mae",
)
GROUP = ("model", "arm", "target", "policy", "profile", "loss", "lr", "weights", "oversampling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-evaluation-dir", type=Path, required=True)
    return parser.parse_args()


def metric(record: dict | None) -> float | None:
    return record.get("mae") if record and record.get("rows", 0) > 0 else None


def extract_metrics(payload: dict) -> dict[str, float | None]:
    sources = payload.get("sources", {})
    result = {}
    for key, raw_key in (
        ("heavy_nterm_sb_mae", "raw_heavy_heavy_nterm"),
        ("heavy_lys_sb_mae", "raw_heavy_heavy_lysine"),
        ("heavy_both_sb_mae", "raw_heavy_heavy_both"),
        ("nonheavy_sb_mae", "raw_heavy_normal"),
    ):
        values = [metric(source.get(raw_key, {})) for source in sources.values()]
        usable = [value for value in values if value is not None]
        result[key] = sum(usable) / len(usable) if usable else None
    result["heavy_macro_source_balanced_mae"] = heavy_macro(result)
    result["overall_mae"] = payload.get("source_balanced_raw_mae")
    semi = [metric(source.get("raw_digestion_semi_tryptic", {})) for source in sources.values()]
    usable_semi = [value for value in semi if value is not None]
    result["semitryptic_mae"] = sum(usable_semi) / len(usable_semi) if usable_semi else None
    overall = [metric(source.get("raw_minutes", {})) for source in sources.values()]
    usable_overall = [value for value in overall if value is not None]
    result["worst_source_mae"] = max(usable_overall) if usable_overall else None
    return result


def candidate_label(row: pd.Series) -> str:
    return f"{row['target']}\n{row['profile']} · {row['loss']} · {row['lr']:.1e}"


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_plots(finalists: pd.DataFrame, output_dir: Path) -> None:
    """Render compact final-test figures from the frozen finalist comparison."""
    if finalists.empty:
        return
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    labels = [candidate_label(row) for _, row in finalists.iterrows()]
    colors = ["#e17c05", "#2474a4", "#5b9f42", "#a8559f"][:len(finalists)]
    metric_sets = (
        (
            "Heavy-Dimethyl strata",
            (
                ("delta_heavy_macro_source_balanced_mae", "Heavy macro"),
                ("delta_heavy_nterm_sb_mae", "N-term only"),
                ("delta_heavy_lys_sb_mae", "Lys only"),
                ("delta_heavy_both_sb_mae", "Both labels"),
            ),
        ),
        (
            "Guardrail metrics",
            (
                ("delta_nonheavy_sb_mae", "Non-heavy"),
                ("delta_overall_mae", "Overall"),
                ("delta_semitryptic_mae", "Semi-tryptic"),
                ("delta_worst_source_mae", "Worst source"),
            ),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False)
    width = 0.75 / len(finalists)
    for axis, (title, metrics) in zip(axes, metric_sets):
        positions = list(range(len(metrics)))
        for index, (_, row) in enumerate(finalists.iterrows()):
            values = [float(row[field]) for field, _ in metrics]
            errors = [float(row.get(f"{field}_std", 0.0) or 0.0) for field, _ in metrics]
            offset = (index - (len(finalists) - 1) / 2) * width
            axis.bar(
                [position + offset for position in positions], values, width=width,
                yerr=errors, capsize=3, color=colors[index], label=labels[index], alpha=0.9,
            )
        axis.axhline(0, color="#222", linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(positions, [label for _, label in metrics], rotation=18, ha="right")
        axis.set_ylabel("Matched Δ MAE (min; negative improves)")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(title="Validation-selected finalist", fontsize=8, title_fontsize=8, frameon=False)
    figure.suptitle("Carafe2 Heavy-Dimethyl: final-test matched deltas", y=1.02, fontsize=14)
    save_figure(figure, plots / "final_test_matched_metric_deltas")

    figure, axis = plt.subplots(figsize=(6.7, 5.7))
    for index, (_, row) in enumerate(finalists.iterrows()):
        x = float(row["delta_nonheavy_sb_mae"])
        y = float(row["delta_heavy_macro_source_balanced_mae"])
        axis.errorbar(
            x, y,
            xerr=float(row.get("delta_nonheavy_sb_mae_std", 0.0) or 0.0),
            yerr=float(row.get("delta_heavy_macro_source_balanced_mae_std", 0.0) or 0.0),
            fmt="o", color=colors[index], markersize=8, capsize=3,
        )
        axis.annotate(labels[index].replace("\n", " · "), (x, y), xytext=(7, -8), textcoords="offset points")
    axis.axhline(0, color="#222", linewidth=0.8)
    axis.axvline(0, color="#222", linewidth=0.8)
    axis.axhspan(min(axis.get_ylim()[0], -0.001), 0, xmin=0, xmax=0.5, color="#dff0df", alpha=0.55)
    axis.set_xlabel("Matched Δ non-heavy SB-MAE (min)")
    axis.set_ylabel("Matched Δ Heavy-Macro SB-MAE (min)")
    axis.set_title("Final-test trade-off (lower-left is favourable)")
    axis.grid(alpha=0.25)
    save_figure(figure, plots / "final_test_heavy_vs_nonheavy_tradeoff")


def main() -> None:
    args = parse_args()
    evaluation_dir = args.test_evaluation_dir.resolve()
    manifest = json.loads((evaluation_dir / "test_manifest.json").read_text(encoding="utf-8"))
    rows = []
    missing = []
    for job in manifest["jobs"]:
        output = Path(job["completion_files"][0])
        if not output.is_file():
            missing.append(job["id"])
            continue
        rows.append({**job["metadata"], **extract_metrics(json.loads(output.read_text(encoding="utf-8")))})
    if missing:
        raise SystemExit(f"Final test evaluation is incomplete; missing {len(missing)} jobs: {', '.join(missing[:5])}")
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit("No final-test metrics found")
    output_dir = evaluation_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    runs.sort_values(["selection_role", *GROUP, "seed"]).to_csv(output_dir / "test_run_summary.csv", index=False)
    summary = runs.groupby(list(GROUP), dropna=False)[list(METRICS)].agg(["mean", "std", "median", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    roles = runs.groupby(list(GROUP), dropna=False)["selection_role"].first().reset_index()
    summary = summary.merge(roles, on=list(GROUP), how="left")

    controls = {
        (row.arm, row.target, row.loss, float(row.lr), int(row.seed)): row
        for row in runs.itertuples(index=False)
        if row.selection_role == "matched_control"
    }
    deltas = []
    for row in runs.itertuples(index=False):
        if row.selection_role != "finalist":
            continue
        control = controls.get((row.arm, row.target, row.loss, float(row.lr), int(row.seed)))
        if control is None:
            raise ValueError(f"No matched final-test control for {row.source_job_id}")
        record = {field: getattr(row, field) for field in GROUP}
        record["seed"] = row.seed
        record["source_job_id"] = row.source_job_id
        for name in METRICS:
            record[f"delta_{name}"] = getattr(row, name) - getattr(control, name)
        deltas.append(record)
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(output_dir / "test_matched_control_deltas.csv", index=False)
    if not delta_frame.empty:
        delta_columns = [f"delta_{name}" for name in METRICS]
        delta_stats = delta_frame.groupby(list(GROUP), dropna=False)[delta_columns].agg(["mean", "std"])
        delta_stats.columns = [
            field if aggregate == "mean" else f"{field}_{aggregate}"
            for field, aggregate in delta_stats.columns
        ]
        summary = summary.merge(delta_stats.reset_index(), on=list(GROUP), how="left")
    finalists = summary[summary["selection_role"].eq("finalist")].sort_values(
        "heavy_macro_source_balanced_mae_mean"
    )
    summary.to_csv(output_dir / "test_configuration_summary.csv", index=False)
    finalists.to_csv(output_dir / "test_heavy_macro_ranking.csv", index=False)
    make_plots(finalists, output_dir)
    (output_dir / "README.md").write_text(
        "# Final-test summary\n\n"
        "These metrics are descriptive results for the finalist set frozen in "
        "`../selection_manifest.json`; this script does not select, filter, or rerank new candidates.\n\n"
        "`plots/` contains matched final-test metric deltas and the Heavy-vs-Non-heavy trade-off.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "runs": len(runs), "finalist_configurations": len(finalists),
        "analysis_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
