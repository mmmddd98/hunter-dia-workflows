#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.express as px
import polars as pl


SCRIPT_PATH = Path(__file__).resolve()
MAPLOT_SRC = SCRIPT_PATH.parents[1] / "src"
if str(MAPLOT_SRC) not in sys.path:
    sys.path.insert(0, str(MAPLOT_SRC))

from maplot.entrapment import entrapment_qvalue_curve, entrapment_summary  # noqa: E402


PREFERRED_REPORT_FILES = ("final_report-first-pass.parquet", "final_report.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Entrapment FDR CSV/HTML plots for one DIA-NN experiment.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--diann-out-root", default="./diann_out")
    parser.add_argument("--pipeline-version", default="2_5_0")
    return parser.parse_args()


def discover_reports(experiment_root: Path) -> dict[str, Path]:
    quant_dir = experiment_root / "05_quant"
    reports: dict[str, Path] = {}
    baseline = preferred_report_path(quant_dir / "baseline")
    if baseline is not None:
        reports["baseline"] = baseline
    tuned_root = quant_dir / "tuned"
    if tuned_root.exists():
        for search_dir in sorted(tuned_root.iterdir()):
            report = preferred_report_path(search_dir)
            if search_dir.is_dir() and report is not None:
                reports[f"tuned/{search_dir.name}"] = report
    if reports:
        return reports

    tune_root = experiment_root / "02_tunelib"
    if tune_root.exists():
        for profile_dir in sorted(tune_root.iterdir()):
            profile = profile_dir.name
            report = profile_dir / f"tune_search_report_{profile}.parquet"
            if profile_dir.is_dir() and report.exists():
                reports[f"tunelib/{profile}"] = report
    return reports


def preferred_report_path(search_dir: Path) -> Path | None:
    for filename in PREFERRED_REPORT_FILES:
        candidate = search_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def write_report(tables: dict[str, pl.DataFrame], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    summary = entrapment_summary(tables, compare_diann_decoys=True)
    summary_path = output_dir / "entrapment_fdr_summary.csv"
    summary.write_csv(summary_path)
    paths.append(summary_path)

    if not summary.is_empty():
        summary_pd = summary.to_pandas()
        hover_cols = ["target_ids", "entrapment_ids", "entrapment_fraction"]
        if "diann_decoy_ids" in summary_pd.columns:
            hover_cols.extend(["diann_decoy_ids", "diann_decoy_fdr_estimate"])
        fig = px.bar(
            summary_pd,
            x="search",
            y="entrapment_fdr_estimate",
            color="level",
            barmode="group",
            hover_data=hover_cols,
            title="Entrapment FDR estimate by search",
            labels={"entrapment_fdr_estimate": "Entrapment IDs / target IDs", "search": "Search"},
            template="plotly_white",
        )
        fig.update_yaxes(tickformat=".2%")
        html_path = output_dir / "entrapment_fdr_summary.html"
        fig.write_html(html_path)
        paths.append(html_path)

    for level in ("precursor", "peptide", "protein_group"):
        curve = entrapment_qvalue_curve(tables, level=level, compare_diann_decoys=True)
        curve_path = output_dir / f"entrapment_qvalue_curve_{level}.csv"
        curve.write_csv(curve_path)
        paths.append(curve_path)
        if curve.is_empty():
            continue
        curve_pd = curve.to_pandas()
        fig = px.line(
            curve_pd,
            x="qvalue",
            y="fdr_estimate",
            color="search",
            line_dash="method" if "method" in curve_pd.columns else None,
            hover_data=[col for col in ["method", "cum_target", "cum_entrapment", "cum_diann_decoy", "rank"] if col in curve_pd.columns],
            title=f"Entrapment FDR vs DIA-NN reported Q.Value ({level})",
            labels={"qvalue": "DIA-NN Q.Value threshold", "fdr_estimate": "FDR estimate"},
            template="plotly_white",
        )
        fig.update_yaxes(tickformat=".2%")
        fig.update_xaxes(range=[0, min(0.05, max(float(curve_pd["qvalue"].max()), 0.01))])
        html_path = output_dir / f"entrapment_qvalue_curve_{level}.html"
        fig.write_html(html_path)
        paths.append(html_path)
    return paths


def main() -> int:
    args = parse_args()
    experiment_root = Path(args.diann_out_root) / args.pipeline_version / args.experiment
    reports = discover_reports(experiment_root)
    if not reports:
        raise FileNotFoundError(f"No final_report.parquet or tune_search_report_*.parquet files found in {experiment_root}")
    output_dir = Path(args.output_dir) if args.output_dir else experiment_root / "ma_plots" / "experiment" / "entrapment"
    tables = {name: pl.read_parquet(path) for name, path in reports.items()}
    paths = write_report(tables, output_dir)
    print(f"Wrote {len(paths)} entrapment report files to {output_dir}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
