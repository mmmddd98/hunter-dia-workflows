#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import polars as pl
import plotly.express as px


SCRIPT_PATH = Path(__file__).resolve()
MAPLOT_PROJECT = SCRIPT_PATH.parents[1]
MAPLOT_SRC = MAPLOT_PROJECT / "src"
if str(MAPLOT_SRC) not in sys.path:
    sys.path.insert(0, str(MAPLOT_SRC))

from maplot import (  # noqa: E402
    CategoryCompositionPlotter,
    DiannAdapter,
    ExperimentDataHandler,
    Experiment,
    ExperimentSearch,
    LabelEfficiencyPlotter,
    MassErrorPlotter,
    MatplotlibRawFilePresenceUpSetPlotter,
    MatplotlibSearchPresenceUpSetPlotter,
    ModificationSignatureFractionPlotter,
    MultiSearchCategoryCompositionPlotter,
    PeptideLengthDistributionPlotter,
    PrecursorChargeDistributionPlotter,
    QuantCompletenessPlotter,
    QuantCorrelationPlotter,
    RawFileCountPlotter,
    ResultDataLoader,
    SearchDataHandler,
    RtHistogramPerRawFilePlotter,
    RtPredictionErrorPlotter,
    ScoreDistributionPlotter,
    SearchIntersectionQualityPlotter,
    SearchSummaryPlotter,
    TerminiAnnotator,
    add_entrapment_columns,
    entrapment_qvalue_curve,
    entrapment_summary,
    exclude_entrapment,
)


RAW_MOD_PATTERNS = {
    "Dimethyl": r"UniMod:36|UniMod:510|28\.031|34\.063|36\.075",
    "Acetyl": r"UniMod:1|42\.010|Acetyl_delta_after_fixed_UniMod510",
    "Oxidation": r"UniMod:35|15\.994",
    "Carbamidomethyl": r"UniMod:4|57\.021",
    "Pyro-Glu": r"UniMod:26|UniMod:27|-17\.026|-18\.010",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MA-plot reports for one DIA-NN pipeline experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment folder name below <diann-out-root>/<pipeline-version>.")
    parser.add_argument("--diann-out-root", default="./diann_out")
    parser.add_argument("--pipeline-version", default="2_5_0")
    parser.add_argument("--output-dir", default=None, help="Defaults to <experiment-root>/ma_plots.")
    parser.add_argument(
        "--report-file",
        default=None,
        help="Load one DIA-NN report directly, including non-standard layouts such as Carafe output.",
    )
    parser.add_argument(
        "--fasta-file",
        default=None,
        help="FASTA used for termini annotation when --report-file is used; inferred from the job when possible.",
    )
    parser.add_argument(
        "--search-name",
        default=None,
        help="MA-plot label for --report-file (defaults to the report's parent directory name).",
    )
    parser.add_argument(
        "--search-manifest",
        default=None,
        help=(
            "TSV with name, report_file and optional fasta_file/quant_group columns. "
            "Builds a custom multi-search comparison, e.g. baseline versus external RT models."
        ),
    )
    parser.add_argument("--organism", default="arabidopsis")
    parser.add_argument("--qvalue", type=float, default=0.01)
    parser.add_argument("--max-mod-upsets", type=int, default=6)
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse existing adapted/termini parquet files.")
    parser.add_argument("--pssm-prediction", action="store_true", help="Enable slower MEROPS PSSM prediction.")
    parser.add_argument(
        "--focus-nterm-peptide",
        action="store_true",
        help="Only create cross-search N-terminus/peptide result plots (skips broad per-search QC).",
    )
    return parser.parse_args()


def infer_fasta_file(report_file: Path) -> Path | None:
    """Infer the FASTA from a non-standard DIA-NN/Carafe report location."""
    for parent in (report_file.parent, *report_file.parents):
        candidates = sorted(parent.glob("*.fasta")) + sorted(parent.glob("*.fa"))
        if len(candidates) == 1:
            return candidates[0].resolve()

    # Carafe searches keep the original step5 command next to the report.
    startcmd = report_file.parent.parent / "logs" / "startcmd.txt"
    if startcmd.is_file():
        tokens = startcmd.read_text(encoding="utf-8").split()
        for flag in ("--fastaabs", "--fasta"):
            if flag in tokens:
                candidate = Path(tokens[tokens.index(flag) + 1]).expanduser()
                if candidate.is_file():
                    return candidate.resolve()
    return None


def direct_report_data_handler(args: argparse.Namespace, output_dir: Path) -> ExperimentDataHandler:
    """Build a one-search handler for reports outside the standard 05_quant layout."""
    report_file = Path(args.report_file).expanduser().resolve()
    fasta_file = Path(args.fasta_file).expanduser().resolve() if args.fasta_file else infer_fasta_file(report_file)
    if fasta_file is None:
        raise FileNotFoundError(
            "Could not infer a FASTA for --report-file. Provide it with --fasta-file."
        )

    search_name = args.search_name or report_file.parent.parent.name
    search = ExperimentSearch(name=search_name, report_file=report_file, quant_group="baseline")
    experiment_root = Path(args.diann_out_root).expanduser().resolve() / args.pipeline_version / args.experiment
    experiment = Experiment(
        name=args.experiment,
        root_dir=experiment_root,
        searches={search_name: search},
        fasta_file=fasta_file,
    )
    handler = SearchDataHandler(
        name=search_name,
        raw_file=report_file,
        fasta_file=fasta_file,
        output_dir=output_dir,
        qvalue_threshold=args.qvalue,
    )
    return ExperimentDataHandler({search_name: handler}, experiment=experiment)


def manifest_data_handler(args: argparse.Namespace, output_dir: Path) -> ExperimentDataHandler:
    """Build a multi-search handler from explicit report paths.

    This is intentionally layout-independent: external-model reports live below
    ``external_searches/`` while baseline and tuning results have a different
    directory structure.
    """
    manifest = Path(args.search_manifest).expanduser().resolve()
    entries = pl.read_csv(manifest, separator="\t").to_dicts()
    required = {"name", "report_file"}
    if not entries or not required.issubset(entries[0]):
        raise ValueError(f"{manifest} must have TSV columns: name and report_file")

    searches: dict[str, ExperimentSearch] = {}
    handlers: dict[str, SearchDataHandler] = {}
    fasta_files: set[Path] = set()
    for entry in entries:
        name = str(entry["name"])
        report_file = Path(str(entry["report_file"])).expanduser().resolve()
        if not report_file.is_file():
            raise FileNotFoundError(f"Report for {name!r} is missing: {report_file}")
        fasta_value = entry.get("fasta_file") or args.fasta_file
        fasta_file = Path(str(fasta_value)).expanduser().resolve() if fasta_value else infer_fasta_file(report_file)
        if fasta_file is None or not fasta_file.is_file():
            raise FileNotFoundError(f"No readable FASTA for {name!r}; add fasta_file to {manifest}")
        fasta_files.add(fasta_file)
        quant_group = str(entry.get("quant_group") or "external")
        searches[name] = ExperimentSearch(name=name, report_file=report_file, quant_group=quant_group)
        handlers[name] = SearchDataHandler(
            name=name,
            raw_file=report_file,
            fasta_file=fasta_file,
            output_dir=output_dir,
            qvalue_threshold=args.qvalue,
        )
    if len(fasta_files) != 1:
        raise ValueError(f"All comparison reports must use one FASTA; found: {sorted(map(str, fasta_files))}")
    experiment_root = Path(args.diann_out_root).expanduser().resolve() / args.pipeline_version / args.experiment
    experiment = Experiment(name=args.experiment, root_dir=experiment_root, searches=searches, fasta_file=fasta_files.pop())
    return ExperimentDataHandler(handlers, experiment=experiment)


def safe_name(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "_").replace(":", "_")


class ProgressLogger:
    def __init__(self, total: int, *, width: int = 28):
        self.total = max(total, 0)
        self.width = width
        self.current = 0

    def step(self, label: str) -> None:
        self.current += 1
        done = self.width if self.total == 0 else round(self.width * self.current / self.total)
        bar = "#" * done + "-" * (self.width - done)
        print(f"[{bar}] {self.current}/{self.total} {label}", flush=True)


def plotter_label(plotter: object) -> str:
    name = plotter.__class__.__name__
    output_name = getattr(plotter, "output_name", None)
    if output_name:
        return f"{name}:{output_name}"
    entity_col = getattr(plotter, "entity_col", None)
    category_col = getattr(plotter, "category_col", None)
    if entity_col and category_col:
        return f"{name}:{category_col}/{entity_col}"
    if entity_col:
        return f"{name}:{entity_col}"
    return name


def run_single_plotters(plotters: list, df: pl.DataFrame, output_dir: Path, progress: ProgressLogger, context: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for plotter in plotters:
        progress.step(f"{context} -> {plotter_label(plotter)}")
        out = plotter.plot(df, output_dir=output_dir)
        if out is not None:
            paths.append(out)
    return paths


def run_multi_plotters(plotters: list, tables: dict[str, pl.DataFrame], output_dir: Path, progress: ProgressLogger, context: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for plotter in plotters:
        progress.step(f"{context} -> {plotter_label(plotter)}")
        out = plotter.plot_many(tables, output_dir=output_dir)
        if out is not None:
            paths.append(out)
    return paths


def raw_mod_label_expr() -> pl.Expr:
    mod = pl.col("Modified.Sequence").cast(pl.Utf8).str.extract_all(r"\(([^()]*)\)")
    joined = mod.list.join(";")
    return (
        pl.when(joined.str.contains(RAW_MOD_PATTERNS["Dimethyl"]))
        .then(pl.lit("Dimethyl"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Acetyl"]))
        .then(pl.lit("Acetyl"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Oxidation"]))
        .then(pl.lit("Oxidation"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Carbamidomethyl"]))
        .then(pl.lit("Carbamidomethyl"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Pyro-Glu"]))
        .then(pl.lit("Pyro-Glu"))
        .when(mod.list.len() == 0)
        .then(pl.lit("Unmodified"))
        .otherwise(pl.lit("Other named mod"))
        .alias("mod_label")
    )


def with_raw_mod_label(df: pl.DataFrame) -> pl.DataFrame:
    if "Modified.Sequence" not in df.columns:
        return df
    return df.with_columns(raw_mod_label_expr())


def with_model_qc_metrics(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    if "RT" in df.columns and "Predicted.RT" in df.columns:
        rt_delta = pl.col("RT").cast(pl.Float64, strict=False) - pl.col("Predicted.RT").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                rt_delta.alias("RT.Delta"),
                rt_delta.abs().alias("Abs.RT.Delta"),
            ]
        )
    for source_col, target_col in (
        ("Best.Fr.Mz.Delta", "Abs.Best.Fr.Mz.Delta"),
        ("Ms1.Apex.Mz.Delta", "Abs.Ms1.Apex.Mz.Delta"),
    ):
        if source_col in df.columns:
            exprs.append(pl.col(source_col).cast(pl.Float64, strict=False).abs().alias(target_col))
    if not exprs:
        return df
    return df.with_columns(exprs)


def write_entrapment_report(raw_tables: dict[str, pl.DataFrame], output_dir: Path) -> list[Path]:
    entrapment_dir = output_dir / "experiment" / "entrapment"
    entrapment_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    summary = entrapment_summary(raw_tables, compare_diann_decoys=True)
    summary_path = entrapment_dir / "entrapment_fdr_summary.csv"
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
        html_path = entrapment_dir / "entrapment_fdr_summary.html"
        fig.write_html(html_path)
        paths.append(html_path)

    for level in ("precursor", "peptide", "protein_group"):
        curve = entrapment_qvalue_curve(raw_tables, level=level, compare_diann_decoys=True)
        curve_path = entrapment_dir / f"entrapment_qvalue_curve_{level}.csv"
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
        fig.update_xaxes(range=[0, min(0.05, max(curve_pd["qvalue"].max(), 0.01))])
        html_path = entrapment_dir / f"entrapment_qvalue_curve_{level}.html"
        fig.write_html(html_path)
        paths.append(html_path)

    return paths


def filter_raw_by_mod_pattern(df: pl.DataFrame, mod_label: str) -> pl.DataFrame:
    return df.filter(pl.col("Modified.Sequence").cast(pl.Utf8).str.contains(RAW_MOD_PATTERNS[mod_label]))


def top_raw_mods_by_pattern(tables: dict[str, pl.DataFrame], max_values: int) -> list[str]:
    rows: list[tuple[str, int]] = []
    for mod_label in RAW_MOD_PATTERNS:
        count = 0
        for df in tables.values():
            if "Modified.Sequence" in df.columns:
                count += filter_raw_by_mod_pattern(df, mod_label).select("Stripped.Sequence").drop_nulls().unique().height
        if count > 0:
            rows.append((mod_label, count))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [label for label, _ in rows[:max_values]]


def filter_tables_by_col_value(tables: dict[str, pl.DataFrame], col: str, value: str) -> dict[str, pl.DataFrame]:
    return {name: df.filter(pl.col(col).cast(pl.Utf8) == str(value)) for name, df in tables.items() if col in df.columns}


def top_values(
    tables: dict[str, pl.DataFrame],
    col: str,
    *,
    max_values: int,
    drop_values: tuple[str, ...] = ("n", "Unmodified", "NA"),
) -> list[str]:
    frames = [df.select(pl.col(col).cast(pl.Utf8).alias(col)).drop_nulls() for df in tables.values() if col in df.columns]
    if not frames:
        return []
    counts = pl.concat(frames).filter(~pl.col(col).is_in(list(drop_values))).group_by(col).len().sort("len", descending=True)
    return counts.head(max_values).get_column(col).to_list()


def count_raw_runs(raw_tables: dict[str, pl.DataFrame]) -> int:
    count = 0
    for df in raw_tables.values():
        if "Run" in df.columns:
            count += df.select("Run").drop_nulls().unique().height
    return count


def materialize_tables(
    experiment_data: ExperimentDataHandler,
    output_dir: Path,
    *,
    qvalue: float,
    organism: str,
    reuse_cache: bool,
    pssm_prediction: bool,
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    adapter = DiannAdapter(proteotypic=True, summarization="SUM", qvalue_threshold=qvalue)
    annotator = TerminiAnnotator(
        organism=organism,
        sense="C",
        specificity="K|R",
        n_residues_area=10,
        pssm_prediction=pssm_prediction,
    )

    adapted_tables: dict[str, pl.DataFrame] = {}
    termini_tables: dict[str, pl.DataFrame] = {}
    print(f"Materializing adapted/termini tables for {len(experiment_data.search_names)} searches.", flush=True)
    for search_name in experiment_data.search_names:
        search = experiment_data.get_search(search_name)
        data_dir = output_dir / "data" / search_name
        adapted_path = data_dir / "adapted.parquet"
        termini_path = data_dir / "termini.parquet"

        if adapted_path.exists() and reuse_cache:
            print(f"[tables] {search_name}: loading cached adapted table", flush=True)
            search.adapted_file = adapted_path
            adapted_df = search.adapted()
        else:
            print(f"[tables] {search_name}: building adapted table", flush=True)
            adapted_df = search.adapted(adapter=adapter, force=True, prefer_file=False)
            ResultDataLoader.write_table(adapted_df, adapted_path)
            search.adapted_file = adapted_path
        adapted_tables[search_name] = adapted_df

        if termini_path.exists() and reuse_cache:
            print(f"[tables] {search_name}: loading cached termini table", flush=True)
            search.termini_file = termini_path
            termini_df = search.termini(annotator=annotator)
        else:
            print(f"[tables] {search_name}: building termini table", flush=True)
            termini_df = search.termini(annotator=annotator, force=True, prefer_file=False)
            ResultDataLoader.write_table(termini_df, termini_path)
            search.termini_file = termini_path
        termini_tables[search_name] = termini_df

    return adapted_tables, termini_tables


def build_report(args: argparse.Namespace) -> list[Path]:
    diann_out_root = Path(args.diann_out_root)
    experiment_root = diann_out_root / args.pipeline_version / args.experiment
    output_dir = Path(args.output_dir) if args.output_dir else experiment_root / "ma_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.search_manifest:
        print(f"Loading custom comparison manifest: {Path(args.search_manifest).expanduser().resolve()}", flush=True)
        experiment_data = manifest_data_handler(args, output_dir)
    elif args.report_file:
        print(f"Loading direct DIA-NN report: {Path(args.report_file).expanduser().resolve()}", flush=True)
        experiment_data = direct_report_data_handler(args, output_dir)
    else:
        experiment_data = ExperimentDataHandler.from_experiment(
            experiment=args.experiment,
            diann_out_root=diann_out_root,
            pipeline_version=args.pipeline_version,
            output_dir=output_dir,
            qvalue_threshold=args.qvalue,
            quant_group="all",
        )
    raw_tables = experiment_data.raw()
    full_raw_tables = {name: add_entrapment_columns(df) for name, df in raw_tables.items()}
    raw_tables = {name: exclude_entrapment(df) for name, df in raw_tables.items()}
    model_qc_tables = {name: with_model_qc_metrics(df) for name, df in raw_tables.items()}
    adapted_tables, termini_tables = materialize_tables(
        experiment_data,
        output_dir,
        qvalue=args.qvalue,
        organism=args.organism,
        reuse_cache=args.reuse_cache,
        pssm_prediction=args.pssm_prediction,
    )

    if args.focus_nterm_peptide:
        # Deliberately small, interpretable panel for model comparisons.  The
        # identity definitions retain protein/start context for N-termini and
        # include N-terminal modification status (see maplot's nterm_* fields).
        progress = ProgressLogger(total=16)
        paths: list[Path] = []
        paths.extend(
            run_multi_plotters(
                [
                    SearchSummaryPlotter(output_name="adapted_search_summary.png", summary_name="adapted_search_summary.csv"),
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="nterm_modif_peptide", presence_cols="auto", max_intersections=35, output_name="upset_nterm_modif_peptide_by_search.png"),
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="peptide", presence_cols="auto", max_intersections=35, output_name="upset_peptide_by_search.png"),
                    MultiSearchCategoryCompositionPlotter(category_col="nterm_modif", entity_col="nterm_modif_peptide"),
                    QuantCompletenessPlotter(),
                    ModificationSignatureFractionPlotter(entity_col="nterm_modif_peptide", category_col="nterm_modif", output_name="ntermini_mod_signature_fraction.png", summary_name="ntermini_mod_signature_fraction.csv", title="N-terminal modification signatures", signature_label="N-term mod"),
                    ModificationSignatureFractionPlotter(entity_col="peptide", category_col="nterm_modif", output_name="peptide_nterm_mod_signature_fraction.png", summary_name="peptide_nterm_mod_signature_fraction.csv", title="N-terminal modification signatures: peptide level", signature_label="N-term mod"),
                ],
                adapted_tables,
                output_dir / "experiment" / "adapted",
                progress,
                "experiment/adapted",
            )
        )
        paths.extend(
            run_multi_plotters(
                [
                    SearchSummaryPlotter(output_name="termini_search_summary.png", summary_name="termini_search_summary.csv"),
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="nterm_modif_peptide", presence_cols="auto", max_intersections=35, output_name="upset_termini_nterm_modif_peptide_by_search.png"),
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="peptide", presence_cols="auto", max_intersections=35, output_name="upset_termini_peptide_by_search.png"),
                    MultiSearchCategoryCompositionPlotter(category_col="specificity", entity_col="nterm_modif_peptide"),
                    MultiSearchCategoryCompositionPlotter(category_col="uniprot_processing_type", entity_col="nterm_modif_peptide"),
                    MultiSearchCategoryCompositionPlotter(category_col="predicted_protease_activity_names", entity_col="nterm_modif_peptide"),
                    ModificationSignatureFractionPlotter(entity_col="nterm_modif_peptide", category_col="specificity", output_name="ntermini_specificity_fraction.png", summary_name="ntermini_specificity_fraction.csv", title="Specific vs semi-specific: N-termini level", signature_label="Specificity"),
                    ModificationSignatureFractionPlotter(entity_col="peptide", category_col="specificity", output_name="peptide_specificity_fraction.png", summary_name="peptide_specificity_fraction.csv", title="Specific vs semi-specific: peptide level", signature_label="Specificity"),
                ],
                termini_tables,
                output_dir / "experiment" / "termini",
                progress,
                "experiment/termini",
            )
        )
        print(f"Completed focused N-terminus/peptide report: wrote {len(paths)} plots under {output_dir}", flush=True)
        return paths

    raw_mod_labels = top_raw_mods_by_pattern(raw_tables, args.max_mod_upsets)
    nterm_mod_labels = top_values(adapted_tables, "nterm_modif", max_values=args.max_mod_upsets)
    total_plots = (
        len(experiment_data.search_names) * (14 + 2 + 3)
        + count_raw_runs(raw_tables)
        + 7
        + 9
        + 5
        + 6
        + 2
        + 3
        + 3
        + len(raw_mod_labels) * 4
        + len(nterm_mod_labels) * 3
    )
    progress = ProgressLogger(total_plots)
    print(
        f"Starting MA plot report: {total_plots} plot tasks "
        f"({len(experiment_data.search_names)} searches, {count_raw_runs(raw_tables)} raw-file histograms, "
        f"{len(raw_mod_labels)} raw mod groups, {len(nterm_mod_labels)} N-term mod groups).",
        flush=True,
    )

    paths: list[Path] = []
    entrapment_paths = write_entrapment_report(full_raw_tables, output_dir)
    if entrapment_paths:
        print(f"Entrapment report: wrote {len(entrapment_paths)} files under {output_dir / 'experiment' / 'entrapment'}", flush=True)
        paths.extend(entrapment_paths)
    for search_name in experiment_data.search_names:
        raw_df = raw_tables[search_name]
        adapted_df = adapted_tables[search_name]
        termini_df = termini_tables[search_name]
        search_base = output_dir / "searches" / search_name

        paths.extend(
            run_single_plotters(
                [
                    RawFileCountPlotter(),
                    PeptideLengthDistributionPlotter(),
                    PrecursorChargeDistributionPlotter(),
                    ScoreDistributionPlotter(score_col="Q.Value"),
                    ScoreDistributionPlotter(score_col="PEP"),
                    ScoreDistributionPlotter(score_col="PG.Q.Value"),
                    RtPredictionErrorPlotter(),
                    MassErrorPlotter(error_col="Best.Fr.Mz.Delta"),
                    MassErrorPlotter(
                        error_col="Ms1.Apex.Mz.Delta",
                        output_name="Ms1_Apex_Mz_Delta_distribution.png",
                        summary_name="Ms1_Apex_Mz_Delta_distribution.csv",
                    ),
                    LabelEfficiencyPlotter(aa="K", mod="UniMod:36", mode="partially"),
                    LabelEfficiencyPlotter(aa="K", mod="UniMod:36", mode="fully"),
                    MatplotlibRawFilePresenceUpSetPlotter(entity_col="Stripped.Sequence", max_intersections=35),
                    MatplotlibRawFilePresenceUpSetPlotter(
                        entity_col="Protein.Group",
                        max_intersections=35,
                        output_name="upset_Protein_Group_matplotlib.png",
                    ),
                    CategoryCompositionPlotter(
                        category_col="mod_label",
                        entity_col="Stripped.Sequence",
                        output_name="raw_mod_label_peptide_counts.png",
                        summary_name="raw_mod_label_peptide_counts.csv",
                    ),
                ],
                with_raw_mod_label(raw_df),
                search_base / "raw",
                progress,
                f"{search_name}/raw",
            )
        )

        for raw_name in raw_df.select("Run").drop_nulls().unique().get_column("Run").to_list():
            one_raw = raw_df.filter(pl.col("Run") == raw_name)
            paths.extend(
                run_single_plotters(
                    [RtHistogramPerRawFilePlotter()],
                    one_raw,
                    search_base / "raw" / "single_raw_file" / safe_name(raw_name),
                    progress,
                    f"{search_name}/raw/{raw_name}",
                )
            )

        paths.extend(
            run_single_plotters(
                [
                    CategoryCompositionPlotter(category_col="nterm_modif", entity_col="nterm_modif_peptide"),
                    QuantCorrelationPlotter(),
                ],
                adapted_df,
                search_base / "adapted",
                progress,
                f"{search_name}/adapted",
            )
        )
        paths.extend(
            run_single_plotters(
                [
                    CategoryCompositionPlotter(category_col="specificity", entity_col="nterm_modif_peptide"),
                    CategoryCompositionPlotter(category_col="uniprot_processing_type", entity_col="nterm_modif_peptide"),
                    CategoryCompositionPlotter(category_col="predicted_protease_activity_names", entity_col="nterm_modif_peptide"),
                ],
                termini_df,
                search_base / "termini",
                progress,
                f"{search_name}/termini",
            )
        )

    paths.extend(
        run_multi_plotters(
            [
                SearchSummaryPlotter(),
                MatplotlibSearchPresenceUpSetPlotter(entity_col="Stripped.Sequence", max_intersections=35),
                MatplotlibSearchPresenceUpSetPlotter(entity_col="Modified.Sequence", max_intersections=35, output_name="upset_Modified_Sequence_matplotlib.png"),
                MatplotlibSearchPresenceUpSetPlotter(
                    entity_col="Protein.Group",
                    max_intersections=35,
                    output_name="upset_Protein_Group_matplotlib.png",
                ),
                SearchIntersectionQualityPlotter(entity_col="Stripped.Sequence", quality_col="Q.Value", output_name="peptide_Q_Value_intersection_quality.png"),
                SearchIntersectionQualityPlotter(entity_col="Protein.Group", quality_col="PG.Q.Value", output_name="protein_group_PG_Q_Value_intersection_quality.png"),
                SearchIntersectionQualityPlotter(entity_col="Stripped.Sequence", quality_col="PEP", output_name="peptide_PEP_intersection_quality.png"),
            ],
            raw_tables,
            output_dir / "experiment" / "raw",
            progress,
            "experiment/raw",
        )
    )
    paths.extend(
        run_multi_plotters(
            [
                SearchIntersectionQualityPlotter(
                    entity_col="Stripped.Sequence",
                    quality_col="Abs.RT.Delta",
                    higher_is_better=False,
                    title="RT model error by search intersection",
                    output_name="peptide_abs_rt_delta_by_intersection.png",
                    summary_name="peptide_abs_rt_delta_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Modified.Sequence",
                    quality_col="Abs.RT.Delta",
                    higher_is_better=False,
                    title="RT model error by peptidoform intersection",
                    output_name="peptidoform_abs_rt_delta_by_intersection.png",
                    summary_name="peptidoform_abs_rt_delta_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Stripped.Sequence",
                    quality_col="Abs.Best.Fr.Mz.Delta",
                    higher_is_better=False,
                    title="Fragment-ion mass error by search intersection",
                    output_name="peptide_abs_best_fr_mz_delta_by_intersection.png",
                    summary_name="peptide_abs_best_fr_mz_delta_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Modified.Sequence",
                    quality_col="Abs.Best.Fr.Mz.Delta",
                    higher_is_better=False,
                    title="Fragment-ion mass error by peptidoform intersection",
                    output_name="peptidoform_abs_best_fr_mz_delta_by_intersection.png",
                    summary_name="peptidoform_abs_best_fr_mz_delta_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Stripped.Sequence",
                    quality_col="Abs.Ms1.Apex.Mz.Delta",
                    higher_is_better=False,
                    title="MS1 mass error by search intersection",
                    output_name="peptide_abs_ms1_apex_mz_delta_by_intersection.png",
                    summary_name="peptide_abs_ms1_apex_mz_delta_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Stripped.Sequence",
                    quality_col="Lib.Q.Value",
                    title="Library/fragment evidence q-value by search intersection",
                    output_name="peptide_Lib_Q_Value_by_intersection.png",
                    summary_name="peptide_Lib_Q_Value_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Modified.Sequence",
                    quality_col="Lib.Peptidoform.Q.Value",
                    title="Library peptidoform q-value by search intersection",
                    output_name="peptidoform_Lib_Peptidoform_Q_Value_by_intersection.png",
                    summary_name="peptidoform_Lib_Peptidoform_Q_Value_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Stripped.Sequence",
                    quality_col="Evidence",
                    title="DIA-NN evidence score by search intersection",
                    output_name="peptide_Evidence_by_intersection.png",
                    summary_name="peptide_Evidence_by_intersection.csv",
                ),
                SearchIntersectionQualityPlotter(
                    entity_col="Stripped.Sequence",
                    quality_col="Mass.Evidence",
                    title="DIA-NN mass evidence score by search intersection",
                    output_name="peptide_Mass_Evidence_by_intersection.png",
                    summary_name="peptide_Mass_Evidence_by_intersection.csv",
                ),
            ],
            model_qc_tables,
            output_dir / "experiment" / "model_qc",
            progress,
            "experiment/model_qc",
        )
    )
    paths.extend(
        run_multi_plotters(
            [
                SearchSummaryPlotter(output_name="adapted_search_summary.png", summary_name="adapted_search_summary.csv"),
                MatplotlibSearchPresenceUpSetPlotter(entity_col="nterm_modif_peptide", presence_cols="auto", max_intersections=35, output_name="upset_nterm_modif_peptide_by_search.png"),
                MatplotlibSearchPresenceUpSetPlotter(entity_col="peptide", presence_cols="auto", max_intersections=35, output_name="upset_peptide_by_search.png"),
                MultiSearchCategoryCompositionPlotter(category_col="nterm_modif", entity_col="nterm_modif_peptide"),
                QuantCompletenessPlotter(),
            ],
            adapted_tables,
            output_dir / "experiment" / "adapted",
            progress,
            "experiment/adapted",
        )
    )
    paths.extend(
        run_multi_plotters(
            [
                SearchSummaryPlotter(output_name="termini_search_summary.png", summary_name="termini_search_summary.csv"),
                MatplotlibSearchPresenceUpSetPlotter(entity_col="nterm_modif_peptide", presence_cols="auto", max_intersections=35, output_name="upset_termini_nterm_modif_peptide_by_search.png"),
                MatplotlibSearchPresenceUpSetPlotter(entity_col="peptide", presence_cols="auto", max_intersections=35, output_name="upset_termini_peptide_by_search.png"),
                MultiSearchCategoryCompositionPlotter(category_col="specificity", entity_col="nterm_modif_peptide"),
                MultiSearchCategoryCompositionPlotter(category_col="uniprot_processing_type", entity_col="nterm_modif_peptide"),
                MultiSearchCategoryCompositionPlotter(category_col="predicted_protease_activity_names", entity_col="nterm_modif_peptide"),
            ],
            termini_tables,
            output_dir / "experiment" / "termini",
            progress,
            "experiment/termini",
        )
    )

    paths.extend(
        run_multi_plotters(
            [
                ModificationSignatureFractionPlotter(entity_col="Stripped.Sequence", output_name="peptide_mod_signature_fraction.png", summary_name="peptide_mod_signature_fraction.csv", title="Modification signatures: peptide level"),
                ModificationSignatureFractionPlotter(entity_col="Protein.Group", output_name="protein_group_mod_signature_fraction.png", summary_name="protein_group_mod_signature_fraction.csv", title="Modification signatures: protein-group level"),
            ],
            raw_tables,
            output_dir / "experiment" / "mod_fractions" / "raw",
            progress,
            "experiment/mod_fractions/raw",
        )
    )
    paths.extend(
        run_multi_plotters(
            [
                ModificationSignatureFractionPlotter(entity_col="nterm_modif_peptide", category_col="nterm_modif", output_name="ntermini_mod_signature_fraction.png", summary_name="ntermini_mod_signature_fraction.csv", title="N-terminal modification signatures", signature_label="N-term mod"),
                ModificationSignatureFractionPlotter(entity_col="peptide", category_col="nterm_modif", output_name="peptide_nterm_mod_signature_fraction.png", summary_name="peptide_nterm_mod_signature_fraction.csv", title="N-terminal modification signatures: peptide level", signature_label="N-term mod"),
                ModificationSignatureFractionPlotter(entity_col="protein", category_col="nterm_modif", output_name="protein_nterm_mod_signature_fraction.png", summary_name="protein_nterm_mod_signature_fraction.csv", title="N-terminal modification signatures: protein level", signature_label="N-term mod"),
            ],
            adapted_tables,
            output_dir / "experiment" / "mod_fractions" / "adapted",
            progress,
            "experiment/mod_fractions/adapted",
        )
    )
    paths.extend(
        run_multi_plotters(
            [
                ModificationSignatureFractionPlotter(entity_col="nterm_modif_peptide", category_col="specificity", output_name="ntermini_specificity_fraction.png", summary_name="ntermini_specificity_fraction.csv", title="Specific vs semi-specific: N-termini level", signature_label="Specificity"),
                ModificationSignatureFractionPlotter(entity_col="peptide", category_col="specificity", output_name="peptide_specificity_fraction.png", summary_name="peptide_specificity_fraction.csv", title="Specific vs semi-specific: peptide level", signature_label="Specificity"),
                ModificationSignatureFractionPlotter(entity_col="protein", category_col="specificity", output_name="protein_specificity_fraction.png", summary_name="protein_specificity_fraction.csv", title="Specific vs semi-specific: protein level", signature_label="Specificity"),
            ],
            termini_tables,
            output_dir / "experiment" / "specificity_fractions",
            progress,
            "experiment/specificity_fractions",
        )
    )

    for mod_label in raw_mod_labels:
        mod_tables = {name: filter_raw_by_mod_pattern(df, mod_label) for name, df in raw_tables.items()}
        mod_dir = output_dir / "experiment" / "mod_specific_upsets" / "raw" / safe_name(mod_label)
        paths.extend(
            run_multi_plotters(
                [
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="Stripped.Sequence", max_intersections=30, output_name=f"upset_peptide_{safe_name(mod_label)}.png"),
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="Protein.Group", max_intersections=30, output_name=f"upset_protein_group_{safe_name(mod_label)}.png"),
                    SearchIntersectionQualityPlotter(entity_col="Stripped.Sequence", quality_col="Q.Value", output_name=f"id_quality_peptide_Q_Value_{safe_name(mod_label)}.png", summary_name=f"id_quality_peptide_Q_Value_{safe_name(mod_label)}.csv"),
                    SearchIntersectionQualityPlotter(entity_col="Protein.Group", quality_col="PG.Q.Value", output_name=f"id_quality_protein_group_PG_Q_Value_{safe_name(mod_label)}.png", summary_name=f"id_quality_protein_group_PG_Q_Value_{safe_name(mod_label)}.csv"),
                ],
                mod_tables,
                mod_dir,
                progress,
                f"experiment/mod_specific_upsets/raw/{mod_label}",
            )
        )

    for mod_label in nterm_mod_labels:
        mod_tables = filter_tables_by_col_value(adapted_tables, "nterm_modif", mod_label)
        mod_dir = output_dir / "experiment" / "mod_specific_upsets" / "ntermini" / safe_name(mod_label)
        paths.extend(
            run_multi_plotters(
                [
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="nterm_modif_peptide", presence_cols="auto", max_intersections=30, output_name=f"upset_nterm_peptidoform_{safe_name(mod_label)}.png"),
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="peptide", presence_cols="auto", max_intersections=30, output_name=f"upset_peptide_nterm_{safe_name(mod_label)}.png"),
                    MatplotlibSearchPresenceUpSetPlotter(entity_col="protein", presence_cols="auto", max_intersections=30, output_name=f"upset_protein_nterm_{safe_name(mod_label)}.png"),
                ],
                mod_tables,
                mod_dir,
                progress,
                f"experiment/mod_specific_upsets/ntermini/{mod_label}",
            )
        )

    print(f"Finished plotting: {len(paths)} primary output files from {progress.current}/{progress.total} plot tasks.", flush=True)
    return paths


def main() -> int:
    args = parse_args()
    paths = build_report(args)
    print(f"Created {len(paths)} primary MA plot outputs.")
    for path in paths[:50]:
        print(path)
    if len(paths) > 50:
        print(f"... {len(paths) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
