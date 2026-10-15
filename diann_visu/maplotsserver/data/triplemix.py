from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from .figures import PLOTLY_TEMPLATE, display_search_label, empty_figure, style_axes


TRIPLEMIX_ORGANISMS = (
    "Arabidopsis thaliana",
    "Escherichia coli",
    "Homo sapiens",
)
TRIPLEMIX_COLORS = {
    "Arabidopsis thaliana": "#8ED081",
    "Escherichia coli": "#035E7B",
    "Homo sapiens": "#EF5B5B",
}
TRIPLEMIX_MODIFICATION_COLORS = {
    "Dimethyl": "#7C3AED",
    "Acetyl": "#F59E0B",
    "Pyro-Glu": "#EC4899",
    "n": "#94A3B8",
}
TRIPLEMIX_CLEAVAGE_COLORS = {
    "Fully-specific": "#64748B",
    "P1 or P2": "#F59E0B",
    "Semi-specific": "#10B981",
    "Other": "#8B5CF6",
    "Ambiguous": "#DC2626",
}
TRIPLEMIX_SHORT_NAMES = {
    "Arabidopsis thaliana": "Arabidopsis",
    "Escherichia coli": "E. coli",
    "Homo sapiens": "Human",
}
TRIPLEMIX_SCATTER_POINTS_PER_ORGANISM = 2_500
# A = 20:65:15, B = 5:65:30 (Arabidopsis:E. coli:Human), as in the source QMD.
TRIPLEMIX_EXPECTED_FC = {
    "Arabidopsis thaliana": 2.0,
    "Escherichia coli": 0.0,
    "Homo sapiens": -1.0,
}
TRIPLEMIX_ARABIDOPSIS_REFERENCE_EXPECTED_FC = {
    organism: expected - TRIPLEMIX_EXPECTED_FC["Arabidopsis thaliana"]
    for organism, expected in TRIPLEMIX_EXPECTED_FC.items()
}
TRIPLEMIX_ANALYSIS_LABELS = {
    "ecoli_centered": "E. coli fold-change centering",
    "arabidopsis_reference": "Arabidopsis upper-quartile reference normalization",
}
TRIPLEMIX_FILTER_OPTIONS = [
    {"label": "All termini", "value": "all"},
    {"label": "Modified termini", "value": "modified"},
    {"label": "HUNTER termini (modified, semi-specific, no Pyro-Glu)", "value": "modified_semi"},
]
TRIPLEMIX_QVALUE_THRESHOLD = 0.01
# The reference-normalization example handled a phosphopeptide spike-in separately.
# The HUNTER Triplemix experiment has no such separate standard.
TRIPLEMIX_PHOS_STD_GENES: tuple[str, ...] = ()


@dataclass(frozen=True)
class TriplemixPlotData:
    """Full diagnostic data plus an unbiased browser-sized scatter sample."""

    full: pd.DataFrame
    sampled: pd.DataFrame

# These run names are documented in H_DIA_sample_metadata_annotation.tsv used by the QMD.
KNOWN_SAMPLE_CONDITIONS = {
    "EX009759": "A",
    "EX009760": "A",
    "EX009761": "A",
    "EX009766": "B",
    "EX009767": "B",
    "EX009768": "B",
}

def sample_condition_map(sample_columns: list[str], experiment_root: Path | None = None) -> dict[str, str]:
    """Resolve A/B samples from optional local metadata, known QMD runs, or A/B names."""
    resolved: dict[str, str] = {}
    if experiment_root is not None:
        for filename in ("triplemix_sample_annotation.tsv", "sample_metadata_annotation.tsv"):
            path = experiment_root / filename
            if not path.exists():
                continue
            try:
                annotation = pd.read_csv(path, sep="\t")
            except (OSError, pd.errors.ParserError):
                continue
            if "condition" not in annotation.columns:
                continue
            id_column = "run" if "run" in annotation.columns else "sample" if "sample" in annotation.columns else None
            if id_column:
                resolved.update(
                    (str(row[id_column]), str(row["condition"]).upper())
                    for _, row in annotation.iterrows()
                    if str(row["condition"]).upper() in {"A", "B"}
                )

    for sample in sample_columns:
        clean = str(sample).split(".", maxsplit=1)[0]
        if sample in resolved:
            continue
        if clean in KNOWN_SAMPLE_CONDITIONS:
            resolved[sample] = KNOWN_SAMPLE_CONDITIONS[clean]
        elif clean.upper().startswith("A"):
            resolved[sample] = "A"
        elif clean.upper().startswith("B"):
            resolved[sample] = "B"
    return resolved


def build_triplemix_metrics(
    termini: pl.DataFrame,
    *,
    condition_map: Mapping[str, str],
) -> pl.DataFrame:
    """Build one unambiguous quantitative row per terminus using the QMD formulas."""
    a_columns = sorted(column for column, condition in condition_map.items() if condition == "A" and column in termini.columns)
    b_columns = sorted(column for column, condition in condition_map.items() if condition == "B" and column in termini.columns)
    if not a_columns or not b_columns:
        raise ValueError("Triplemix requires at least one sample assigned to both condition A and condition B.")

    required = {"nterm_modif_peptide", "nterm_modif", "peptide", "protein", "protein_organism", "specificity"}
    missing = sorted(required.difference(termini.columns))
    if missing:
        raise ValueError(f"Triplemix termini table is missing required columns: {', '.join(missing)}")

    sample_columns = a_columns + b_columns
    work = termini.with_columns(
        [pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in sample_columns]
    )
    if "p1_position" not in work.columns:
        work = work.with_columns(pl.lit(None, dtype=pl.Int64).alias("p1_position"))

    # A quantitative terminus may map to several proteins. Count it once only if
    # every mapped FASTA record agrees on one canonical Triplemix organism.
    organism_mapping = work.group_by("nterm_modif_peptide").agg(
        pl.col("protein_organism").drop_nulls().unique().sort().alias("_organisms")
    ).with_columns(
        pl.col("_organisms").list.len().alias("organism_mapping_count")
    )
    valid_mapping = organism_mapping.filter(
        (pl.col("organism_mapping_count") == 1)
        & pl.col("_organisms").list.first().is_in(TRIPLEMIX_ORGANISMS)
    ).select("nterm_modif_peptide", "organism_mapping_count")
    excluded_cross_organism = organism_mapping.filter(pl.col("organism_mapping_count") > 1).height
    excluded_unmapped_or_noncanonical = organism_mapping.height - valid_mapping.height - excluded_cross_organism

    work = (
        work.join(valid_mapping, on="nterm_modif_peptide", how="inner")
        .filter(pl.col("protein_organism").is_in(TRIPLEMIX_ORGANISMS))
        .with_columns(
            pl.when(pl.col("specificity") == "specific").then(pl.lit("Fully-specific"))
            .when((pl.col("specificity") == "semi_Nterm") & pl.col("p1_position").is_in([0, 1, 2])).then(pl.lit("P1 or P2"))
            .when(pl.col("specificity") == "semi_Nterm").then(pl.lit("Semi-specific"))
            .otherwise(pl.lit("Other")).alias("cleavage_class")
        )
        .group_by("nterm_modif_peptide")
        .agg(
            pl.col("nterm_modif").drop_nulls().first(),
            pl.col("peptide").drop_nulls().first(),
            pl.col("protein").drop_nulls().unique().sort().first(),
            pl.col("protein_organism").drop_nulls().unique().first(),
            pl.col("protein").drop_nulls().n_unique().alias("protein_mapping_count"),
            pl.col("specificity").drop_nulls().unique().sort().alias("_specificities"),
            pl.col("cleavage_class").drop_nulls().unique().sort().alias("_cleavage_classes"),
            *[pl.col(column).drop_nulls().first().alias(column) for column in sample_columns],
            *[pl.col(column).drop_nulls().n_unique().alias(f"_quant_count_{column}") for column in sample_columns],
        )
        .with_columns(
            pl.when(pl.col("_specificities").list.len() == 1).then(pl.col("_specificities").list.first()).otherwise(pl.lit("ambiguous")).alias("specificity"),
            pl.when(pl.col("_cleavage_classes").list.len() == 1).then(pl.col("_cleavage_classes").list.first()).otherwise(pl.lit("Ambiguous")).alias("cleavage_class"),
            (pl.col("_specificities").list.len() > 1).alias("annotation_ambiguous"),
            pl.any_horizontal([pl.col(f"_quant_count_{column}") > 1 for column in sample_columns]).alias("quant_mapping_conflict"),
            pl.col("peptide").str.replace(r"R$", "").str.count_matches("R").alias("missed_cleavages"),
        )
        .drop("_specificities", "_cleavage_classes", *[f"_quant_count_{column}" for column in sample_columns])
    )
    work, arabidopsis_reference_center = _add_arabidopsis_reference_normalization(work, sample_columns)
    arabidopsis_reference_a_columns = [f"_arabidopsis_reference_{column}" for column in a_columns]
    arabidopsis_reference_b_columns = [f"_arabidopsis_reference_{column}" for column in b_columns]
    arabidopsis_reference_sample_columns = arabidopsis_reference_a_columns + arabidopsis_reference_b_columns

    work = (
        work.with_columns(
            pl.mean_horizontal([pl.col(column).cast(pl.Float64, strict=False) for column in a_columns]).alias("abundance_A"),
            pl.mean_horizontal([pl.col(column).cast(pl.Float64, strict=False) for column in b_columns]).alias("abundance_B"),
            pl.mean_horizontal(arabidopsis_reference_a_columns).alias("arabidopsis_reference_abundance_A"),
            pl.mean_horizontal(arabidopsis_reference_b_columns).alias("arabidopsis_reference_abundance_B"),
            pl.any_horizontal([pl.col(column).is_finite().fill_null(False) for column in a_columns]).alias("detected_A"),
            pl.any_horizontal([pl.col(column).is_finite().fill_null(False) for column in b_columns]).alias("detected_B"),
            pl.all_horizontal([pl.col(column).is_finite().fill_null(False) for column in a_columns]).alias("complete_A"),
            pl.all_horizontal([pl.col(column).is_finite().fill_null(False) for column in b_columns]).alias("complete_B"),
            _lognormal_cv_expr(a_columns).alias("cv_A"),
            _lognormal_cv_expr(b_columns).alias("cv_B"),
        )
        .with_columns(
            (pl.col("complete_A") & pl.col("complete_B")).alias("complete_all_replicates"),
            (pl.col("abundance_A") - pl.col("abundance_B")).alias("fold_change"),
            (pl.col("arabidopsis_reference_abundance_A") - pl.col("arabidopsis_reference_abundance_B")).alias("arabidopsis_reference_fold_change_normalized"),
            pl.mean_horizontal("abundance_A", "abundance_B").alias("mean_abundance"),
            # The reference-normalized scatter uses the arithmetic mean of the log10
            # replicate abundances. Adapter quantities are log2, so convert
            # the row-wise log2 mean instead of averaging linear intensities.
            (
                pl.mean_horizontal(
                    [pl.col(column).cast(pl.Float64, strict=False) for column in sample_columns]
                )
                * np.log10(2.0)
            ).alias("mean_log10_abundance"),
            (
                pl.mean_horizontal(arabidopsis_reference_sample_columns) * np.log10(2.0)
            ).alias("arabidopsis_reference_mean_log10_abundance"),
            (pl.col("abundance_A") + pl.col("abundance_B")).alias("mean_A_plus_mean_B"),
            pl.when(pl.col("detected_A") & pl.col("detected_B")).then(pl.lit("Both"))
            .when(pl.col("detected_A")).then(pl.lit("A only"))
            .when(pl.col("detected_B")).then(pl.lit("B only"))
            .otherwise(pl.lit("Neither")).alias("detection_class"),
        )
    )

    if work.filter(pl.col("quant_mapping_conflict")).height:
        raise ValueError("Conflicting quantitative values were found across protein mappings for the same terminus.")

    complete = work.filter(pl.col("complete_all_replicates") & pl.col("fold_change").is_finite())
    abundance_cutoff = complete.get_column("mean_A_plus_mean_B").quantile(0.90, interpolation="linear") if complete.height else None
    anchor = complete.filter(
        (pl.col("protein_organism") == "Escherichia coli")
        & (pl.col("nterm_modif") != "n")
        & (pl.col("mean_A_plus_mean_B") > abundance_cutoff)
    ) if abundance_cutoff is not None else complete.head(0)
    if anchor.is_empty():
        anchor = complete.filter(pl.col("protein_organism") == "Escherichia coli")
    offset = anchor.get_column("fold_change").median() if not anchor.is_empty() else 0.0
    offset = float(offset) if offset is not None and np.isfinite(offset) else 0.0

    return work.with_columns(
        (pl.col("fold_change") - offset).alias("fold_change_normalized"),
        pl.col("protein_organism").replace_strict(TRIPLEMIX_EXPECTED_FC, default=None).cast(pl.Float64).alias("expected_fold_change"),
        pl.col("protein_organism").replace_strict(TRIPLEMIX_ARABIDOPSIS_REFERENCE_EXPECTED_FC, default=None).cast(pl.Float64).alias("arabidopsis_reference_expected_fold_change"),
        pl.lit(offset).alias("ecoli_fc_offset"),
        pl.lit(arabidopsis_reference_center).alias("arabidopsis_reference_center"),
        pl.lit(", ".join(a_columns)).alias("samples_A"),
        pl.lit(", ".join(b_columns)).alias("samples_B"),
        pl.lit(len(a_columns)).alias("replicates_A"),
        pl.lit(len(b_columns)).alias("replicates_B"),
        pl.lit(excluded_cross_organism).alias("excluded_cross_organism_termini"),
        pl.lit(excluded_unmapped_or_noncanonical).alias("excluded_unmapped_or_noncanonical_termini"),
    ).with_columns(
        (pl.col("fold_change_normalized") - pl.col("expected_fold_change")).alias("fold_change_error"),
        (pl.col("arabidopsis_reference_fold_change_normalized") - pl.col("arabidopsis_reference_expected_fold_change")).alias("arabidopsis_reference_fold_change_error"),
    ).drop(arabidopsis_reference_sample_columns)


def _add_arabidopsis_reference_normalization(
    work: pl.DataFrame,
    sample_columns: list[str],
) -> tuple[pl.DataFrame, float]:
    """Apply the per-run upper-quartile Arabidopsis reference shift."""
    with_mean = work.with_columns(
        pl.mean_horizontal(sample_columns).alias("_arabidopsis_reference_mean_log2")
    )
    threshold = with_mean.get_column("_arabidopsis_reference_mean_log2").quantile(0.75, interpolation="linear")
    reference = with_mean.filter(
        (pl.col("protein_organism") == "Arabidopsis thaliana")
        & (pl.col("_arabidopsis_reference_mean_log2") > threshold)
    ) if threshold is not None else with_mean.head(0)
    if reference.is_empty():
        reference = with_mean.filter(pl.col("protein_organism") == "Arabidopsis thaliana")

    medians: dict[str, float] = {}
    for column in sample_columns:
        median = reference.get_column(column).median() if not reference.is_empty() else None
        medians[column] = float(median) if median is not None and np.isfinite(median) else 0.0
    finite_medians = [value for value in medians.values() if np.isfinite(value)]
    center = float(np.mean(finite_medians)) if finite_medians else 0.0
    normalized = with_mean.with_columns(
        [
            (pl.col(column) - (medians[column] - center)).alias(f"_arabidopsis_reference_{column}")
            for column in sample_columns
        ]
    ).drop("_arabidopsis_reference_mean_log2")
    return normalized, center


def _lognormal_cv_expr(columns: list[str]) -> pl.Expr:
    """Reproduce the QMD CV from log2 abundances; undefined for one replicate."""
    if len(columns) < 2:
        return pl.lit(None, dtype=pl.Float64)
    sd_ln = pl.concat_list([pl.col(column) * np.log(2.0) for column in columns]).list.drop_nulls().list.std(ddof=1)
    return ((sd_ln.pow(2).exp() - 1.0).sqrt() * 100.0)


def filter_triplemix_metrics(metrics: pl.DataFrame, evidence_filter: str | None) -> pl.DataFrame:
    if evidence_filter == "modified":
        return metrics.filter(pl.col("nterm_modif") != "n")
    if evidence_filter == "modified_semi":
        return metrics.filter(
            (pl.col("nterm_modif") != "n")
            & ~pl.col("nterm_modif").str.to_lowercase().str.starts_with("pyro")
            & (pl.col("specificity") == "semi_Nterm")
        )
    return metrics


def triplemix_identifications_figure(metrics_by_search: Mapping[str, pl.DataFrame], evidence_filter: str) -> go.Figure:
    rows = []
    for search, table in metrics_by_search.items():
        data = filter_triplemix_metrics(table, evidence_filter)
        sample_map = _samples_from_metrics(data)
        for condition, sample_names in sample_map.items():
            detected_column = f"detected_{condition}"
            counts = data.filter(pl.col(detected_column)).group_by("protein_organism").len()
            rows.extend({"search": search, "condition": condition, "sample": sample_names, "organism": row["protein_organism"], "count": row["len"]} for row in counts.to_dicts())
    if not rows:
        return empty_figure("No Triplemix identifications found.")
    frame = pd.DataFrame(rows)
    single_search = len(metrics_by_search) == 1
    fig = px.bar(
        frame,
        x="condition" if single_search else "search",
        y="count",
        color="organism",
        facet_col=None if single_search else "condition",
        barmode="stack",
        color_discrete_map=TRIPLEMIX_COLORS,
        template=PLOTLY_TEMPLATE,
        hover_data=["sample"],
    )
    search_suffix = f" — {display_search_label(next(iter(metrics_by_search)))}" if single_search else ""
    fig.update_layout(title=f"Identifications per condition and organism{search_suffix}", xaxis_title="Condition" if single_search else "Search", yaxis_title="N-termini", height=460)
    return _style_categorical_figure(style_axes(fig))


def triplemix_completeness_figure(metrics_by_search: Mapping[str, pl.DataFrame], evidence_filter: str) -> go.Figure:
    rows = []
    for search, table in metrics_by_search.items():
        counts = filter_triplemix_metrics(table, evidence_filter).group_by("protein_organism", "detection_class").len()
        rows.extend({"search": search, "organism": row["protein_organism"], "detection": row["detection_class"], "count": row["len"]} for row in counts.to_dicts())
    if not rows:
        return empty_figure("No A/B completeness data found.")
    single_search = len(metrics_by_search) == 1
    fig = px.bar(pd.DataFrame(rows), x="organism", y="count", color="detection", facet_col=None if single_search else "search", barmode="stack", category_orders={"organism": list(TRIPLEMIX_ORGANISMS), "detection": ["Both", "A only", "B only", "Neither"]}, color_discrete_map={"Both": "#334155", "A only": "#F59E0B", "B only": "#8B5CF6", "Neither": "#CBD5E1"}, template=PLOTLY_TEMPLATE)
    search_suffix = f" — {display_search_label(next(iter(metrics_by_search)))}" if single_search else ""
    fig.update_layout(title=f"A/B detection overlap{search_suffix}", xaxis_title="Organism", yaxis_title="N-termini", height=460)
    return _style_categorical_figure(style_axes(fig), organism_axis=True)


def triplemix_modification_figure(metrics: pl.DataFrame, search: str, evidence_filter: str) -> go.Figure:
    """Reconstruct the QMD identification counts by N-terminal modification."""
    rows = _condition_composition_rows(metrics, evidence_filter, "nterm_modif")
    if not rows:
        return empty_figure("No N-terminal modification counts available.")
    fig = px.bar(
        pd.DataFrame(rows),
        x="organism",
        y="count",
        color="category",
        facet_col="condition",
        barmode="stack",
        category_orders={"organism": list(TRIPLEMIX_ORGANISMS), "condition": ["A", "B"]},
        color_discrete_map=TRIPLEMIX_MODIFICATION_COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    fig.update_layout(title=f"N-terminal modifications — {display_search_label(search)}", xaxis_title="Organism", yaxis_title="Unique termini", height=500)
    return _style_categorical_figure(style_axes(fig), organism_axis=True, faceted=True)


def triplemix_specificity_figure(metrics: pl.DataFrame, search: str, evidence_filter: str) -> go.Figure:
    """Reconstruct the QMD cleavage-specificity composition by condition."""
    rows = _condition_composition_rows(metrics, evidence_filter, "cleavage_class")
    if not rows:
        return empty_figure("No cleavage-specificity counts available.")
    fig = px.bar(
        pd.DataFrame(rows),
        x="organism",
        y="count",
        color="category",
        facet_col="condition",
        barmode="stack",
        category_orders={
            "organism": list(TRIPLEMIX_ORGANISMS),
            "condition": ["A", "B"],
            "category": ["Fully-specific", "P1 or P2", "Semi-specific", "Other", "Ambiguous"],
        },
        color_discrete_map=TRIPLEMIX_CLEAVAGE_COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    fig.update_layout(title=f"Cleavage specificity — {display_search_label(search)}", xaxis_title="Organism", yaxis_title="Unique termini", height=500)
    return _style_categorical_figure(style_axes(fig), organism_axis=True, faceted=True)


def _condition_composition_rows(metrics: pl.DataFrame, evidence_filter: str, category_column: str) -> list[dict]:
    data = filter_triplemix_metrics(metrics, evidence_filter)
    if data.is_empty() or category_column not in data.columns:
        return []
    rows: list[dict] = []
    for condition in ("A", "B"):
        counts = (
            data.filter(pl.col(f"detected_{condition}"))
            .group_by("protein_organism", category_column)
            .len()
        )
        rows.extend(
            {
                "condition": condition,
                "organism": row["protein_organism"],
                "category": row[category_column],
                "count": row["len"],
            }
            for row in counts.to_dicts()
        )
    return rows


def _style_categorical_figure(
    fig: go.Figure,
    *,
    organism_axis: bool = False,
    faceted: bool = False,
) -> go.Figure:
    """Use compact labels and a horizontal legend without stealing plot width."""
    for trace in fig.data:
        if trace.name in TRIPLEMIX_SHORT_NAMES:
            trace.name = TRIPLEMIX_SHORT_NAMES[trace.name]
    if organism_axis:
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(TRIPLEMIX_ORGANISMS),
            ticktext=[TRIPLEMIX_SHORT_NAMES[name] for name in TRIPLEMIX_ORGANISMS],
            automargin=True,
        )
    fig.update_layout(
        legend={
            "orientation": "h",
            "x": 1.0,
            "xanchor": "right",
            "y": 1.10 if faceted else 1.02,
            "yanchor": "bottom",
            "title_text": "",
            "font": {"size": 11},
            "itemsizing": "constant",
        },
        margin={"l": 58, "r": 18, "t": 116 if faceted else 92, "b": 64},
        title={"x": 0.0, "xanchor": "left", "font": {"size": 18}},
    )
    return fig


def triplemix_ab_scatter_figure(
    metrics: pl.DataFrame,
    search: str,
    evidence_filter: str,
    plot_data: TriplemixPlotData | None = None,
    analysis: str = "ecoli_centered",
) -> go.Figure:
    prepared = plot_data or prepare_triplemix_plot_data(metrics, evidence_filter)
    data = _analysis_frame(prepared.sampled, analysis)
    if data.empty:
        return empty_figure("No features quantified in both A and B.")
    fig = px.scatter(
        data,
        x="abundance_A",
        y="abundance_B",
        color="protein_organism",
        color_discrete_map=TRIPLEMIX_COLORS,
        opacity=0.28,
        render_mode="webgl",
        hover_data=["peptide", "nterm_modif", "missed_cleavages", "fold_change_normalized"],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(marker={"size": 4}, selector={"type": "scattergl"})
    diagnostic_data = _analysis_frame(prepared.full, analysis)
    add_scatter_diagnostics(
        fig,
        diagnostic_data,
        x_column="abundance_A",
        y_column="abundance_B",
        show_metrics=False,
    )
    low = float(min(diagnostic_data["abundance_A"].min(), diagnostic_data["abundance_B"].min()))
    high = float(max(diagnostic_data["abundance_A"].max(), diagnostic_data["abundance_B"].max()))
    offset = float(metrics.item(0, "ecoli_fc_offset")) if metrics.height and analysis == "ecoli_centered" else 0.0
    expected_fc = TRIPLEMIX_EXPECTED_FC if analysis == "ecoli_centered" else TRIPLEMIX_ARABIDOPSIS_REFERENCE_EXPECTED_FC
    for organism, expected in expected_fc.items():
        y_values = [expected_ab_line_y(x, expected, offset) for x in (low, high)]
        fig.add_trace(go.Scatter(x=[low, high], y=y_values, mode="lines", name="Target", line={"color": TRIPLEMIX_COLORS[organism], "dash": "dash"}, hoverinfo="skip", showlegend=organism == TRIPLEMIX_ORGANISMS[0]))
    for index, (organism, observed) in enumerate(observed_fold_change_medians(diagnostic_data).items()):
        y_values = [expected_ab_line_y(x, observed, offset) for x in (low, high)]
        fig.add_trace(go.Scatter(x=[low, high], y=y_values, mode="lines", name="Observed median", line={"color": TRIPLEMIX_COLORS[organism], "dash": "dot", "width": 3}, hoverinfo="skip", showlegend=index == 0))
    fig.update_layout(title=f"A vs B · {TRIPLEMIX_ANALYSIS_LABELS[analysis]} — {display_search_label(search)}", xaxis_title="A abundance (log2)", yaxis_title="B abundance (log2)", height=760)
    style_triplemix_scatter_layout(fig)
    return style_axes(fig)


def triplemix_raw_ma_figure(
    metrics: pl.DataFrame,
    search: str,
    evidence_filter: str,
    plot_data: TriplemixPlotData | None = None,
) -> go.Figure:
    """Show raw Precursor.Quantity before DIA-NN, MAD, or reference normalization."""
    required = {"raw_fold_change", "raw_mean_log10_abundance", "raw_complete_all_replicates"}
    if not required.issubset(metrics.columns):
        return empty_figure("Raw precursor quantities are not available yet.")
    prepared = plot_data or prepare_triplemix_plot_data(metrics, evidence_filter)
    data = prepared.sampled.copy()
    diagnostic_data = prepared.full.copy()
    for frame in (data, diagnostic_data):
        if frame.empty or not required.issubset(frame.columns):
            continue
        frame.drop(
            frame.index[
                ~frame["raw_complete_all_replicates"].fillna(False)
                | ~np.isfinite(pd.to_numeric(frame["raw_fold_change"], errors="coerce"))
                | ~np.isfinite(pd.to_numeric(frame["raw_mean_log10_abundance"], errors="coerce"))
            ],
            inplace=True,
        )
    if data.empty or diagnostic_data.empty:
        return empty_figure("No complete raw A/B precursor quantities available.")

    fig = px.scatter(
        data,
        x="raw_fold_change",
        y="raw_mean_log10_abundance",
        color="protein_organism",
        color_discrete_map=TRIPLEMIX_COLORS,
        opacity=0.28,
        render_mode="webgl",
        hover_data=["peptide", "nterm_modif", "missed_cleavages"],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(marker={"size": 4}, selector={"type": "scattergl"})
    add_scatter_diagnostics(
        fig,
        diagnostic_data,
        x_column="raw_fold_change",
        y_column="raw_mean_log10_abundance",
        show_metrics=False,
    )
    fig.add_vline(x=0.0, line_dash="dot", line_color="#475569", line_width=1)
    low = float(diagnostic_data["raw_mean_log10_abundance"].min())
    high = float(diagnostic_data["raw_mean_log10_abundance"].max())
    median_labels: list[str] = []
    for index, organism in enumerate(TRIPLEMIX_ORGANISMS):
        values = pd.to_numeric(
            diagnostic_data.loc[diagnostic_data["protein_organism"] == organism, "raw_fold_change"],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        median = float(values.median())
        median_labels.append(
            f"<span style='color:{TRIPLEMIX_COLORS[organism]}'><b>{TRIPLEMIX_SHORT_NAMES[organism]}</b></span> {median:+.2f}"
        )
        fig.add_trace(
            go.Scatter(
                x=[median, median],
                y=[low, high],
                mode="lines",
                name="Observed raw median",
                line={"color": TRIPLEMIX_COLORS[organism], "dash": "dot", "width": 3},
                hovertemplate=f"{TRIPLEMIX_SHORT_NAMES[organism]} raw median: {median:+.2f}<extra></extra>",
                showlegend=index == 0,
            )
        )
    if median_labels:
        fig.add_annotation(
            x=0.99,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<b>Observed raw median log2(A/B)</b><br>" + " · ".join(median_labels),
            showarrow=False,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.84)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            borderpad=5,
            font={"size": 10, "color": "#111827"},
        )
    fig.update_layout(
        title=(
            f"Unnormalized fold change vs mean abundance — {display_search_label(search)}"
            "<br><sup>Precursor.Quantity; no DIA-NN quantity normalization, MAD scaling, or organism centering</sup>"
        ),
        xaxis_title="Raw log2 fold change A / B",
        yaxis_title="Mean log10 raw abundance across A and B replicates",
        height=820,
    )
    style_triplemix_scatter_layout(fig)
    return style_axes(fig)


def triplemix_ma_figure(
    metrics: pl.DataFrame,
    search: str,
    evidence_filter: str,
    plot_data: TriplemixPlotData | None = None,
    analysis: str = "ecoli_centered",
) -> go.Figure:
    prepared = plot_data or prepare_triplemix_plot_data(metrics, evidence_filter)
    data = _analysis_frame(prepared.sampled, analysis)
    if data.empty:
        return empty_figure("No fold changes available.")
    fig = px.scatter(
        data,
        x="fold_change_normalized",
        y="mean_log10_abundance",
        color="protein_organism",
        color_discrete_map=TRIPLEMIX_COLORS,
        opacity=0.28,
        render_mode="webgl",
        hover_data=["peptide", "nterm_modif", "missed_cleavages", "fold_change_normalized", "fold_change_error"],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(marker={"size": 4}, selector={"type": "scattergl"})
    diagnostic_data = _analysis_frame(prepared.full, analysis)
    add_scatter_diagnostics(
        fig,
        diagnostic_data,
        x_column="fold_change_normalized",
        y_column="mean_log10_abundance",
    )
    low = float(diagnostic_data["mean_log10_abundance"].min())
    high = float(diagnostic_data["mean_log10_abundance"].max())
    expected_fc = TRIPLEMIX_EXPECTED_FC if analysis == "ecoli_centered" else TRIPLEMIX_ARABIDOPSIS_REFERENCE_EXPECTED_FC
    for organism, expected in expected_fc.items():
        fig.add_trace(go.Scatter(x=[expected, expected], y=[low, high], mode="lines", name="Target", line={"color": TRIPLEMIX_COLORS[organism], "dash": "dash"}, hoverinfo="skip", showlegend=organism == TRIPLEMIX_ORGANISMS[0]))
    for index, (organism, observed) in enumerate(observed_fold_change_medians(diagnostic_data).items()):
        fig.add_trace(go.Scatter(x=[observed, observed], y=[low, high], mode="lines", name="Observed median", line={"color": TRIPLEMIX_COLORS[organism], "dash": "dot", "width": 3}, hoverinfo="skip", showlegend=index == 0))
    fig.update_layout(
        title=f"Fold change vs mean abundance · {TRIPLEMIX_ANALYSIS_LABELS[analysis]} — {display_search_label(search)}",
        xaxis_title="Normalized log2 fold change A / B",
        yaxis_title="Mean log10 abundance across A and B replicates",
        height=760,
    )
    style_triplemix_scatter_layout(fig)
    return style_axes(fig)


def triplemix_fc_distribution_figure(
    metrics_by_search: Mapping[str, pl.DataFrame],
    evidence_filter: str,
    plot_data_by_search: Mapping[str, TriplemixPlotData] | None = None,
    analysis: str = "ecoli_centered",
) -> go.Figure:
    rows: list[dict] = []
    single_search = len(metrics_by_search) == 1
    for search, table in metrics_by_search.items():
        prepared = plot_data_by_search.get(search) if plot_data_by_search else None
        data = _analysis_frame((prepared or prepare_triplemix_plot_data(table, evidence_filter)).full, analysis)
        for organism in TRIPLEMIX_ORGANISMS:
            values = pd.to_numeric(
                data.loc[data["protein_organism"] == organism, "fold_change_normalized"],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            q1, median, q3 = (float(value) for value in values.quantile([0.25, 0.50, 0.75]))
            iqr = q3 - q1
            lower_values = values.loc[values >= q1 - 1.5 * iqr]
            upper_values = values.loc[values <= q3 + 1.5 * iqr]
            expected_fc = TRIPLEMIX_EXPECTED_FC if analysis == "ecoli_centered" else TRIPLEMIX_ARABIDOPSIS_REFERENCE_EXPECTED_FC
            expected = float(expected_fc[organism])
            label = TRIPLEMIX_SHORT_NAMES[organism]
            if not single_search:
                label = f"{label}<br>{display_search_label(search)}"
            rows.append(
                {
                    "search": search,
                    "organism": organism,
                    "label": label,
                    "n": int(values.size),
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "lower": float(lower_values.min()) if not lower_values.empty else float(values.min()),
                    "upper": float(upper_values.max()) if not upper_values.empty else float(values.max()),
                    "expected": expected,
                    "error": abs(median - expected),
                }
            )
    if not rows:
        return empty_figure("No fold-change distributions available.")
    fig = go.Figure()
    for row in rows:
        color = TRIPLEMIX_COLORS[row["organism"]]
        fig.add_trace(
            go.Box(
                x=[row["label"]],
                q1=[row["q1"]],
                median=[row["median"]],
                q3=[row["q3"]],
                lowerfence=[row["lower"]],
                upperfence=[row["upper"]],
                name=TRIPLEMIX_SHORT_NAMES[row["organism"]],
                marker_color=color,
                fillcolor=color,
                opacity=0.68,
                line={"color": color, "width": 2},
                boxpoints=False,
                showlegend=False,
                customdata=[[row["n"], row["expected"], row["median"], row["median"] - row["expected"]]],
                hovertemplate=(
                    f"<b>{TRIPLEMIX_SHORT_NAMES[row['organism']]}</b><br>"
                    "n=%{customdata[0]:,}<br>median=%{customdata[2]:+.2f}<br>"
                    "target=%{customdata[1]:+.2f}<br>bias=%{customdata[3]:+.2f}<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[row["label"] for row in rows],
            y=[row["expected"] for row in rows],
            mode="markers",
            name="Expected",
            marker={
                "symbol": "diamond",
                "size": 10,
                "color": [TRIPLEMIX_COLORS[row["organism"]] for row in rows],
                "line": {"color": "white", "width": 1.5},
            },
            hovertemplate="Expected: %{y:+.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#94A3B8", line_width=1)
    mae = float(np.median([row["error"] for row in rows]))
    search_suffix = f" — {display_search_label(next(iter(metrics_by_search)))}" if single_search else ""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Fold-change accuracy · {TRIPLEMIX_ANALYSIS_LABELS[analysis]}{search_suffix} · median |bias| {mae:.2f}",
        xaxis_title="Organism",
        yaxis_title="Normalized log2 fold change A − B",
        height=520,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=[row["label"] for row in rows])
    return _style_categorical_figure(style_axes(fig))


def expected_ab_line_y(abundance_a: float, expected_normalized_fc: float, ecoli_offset: float) -> float:
    """Return B for an expected normalized FC on raw MAD-normalized A/B axes.

    normalized FC = (A - B) - E. coli offset, therefore
    B = A - (expected normalized FC + E. coli offset).
    """
    return float(abundance_a) - (float(expected_normalized_fc) + float(ecoli_offset))


def observed_fold_change_medians(data: pd.DataFrame) -> dict[str, float]:
    """Return finite organism-wise medians of the displayed normalized fold changes."""
    if data.empty or not {"protein_organism", "fold_change_normalized"}.issubset(data.columns):
        return {}
    medians: dict[str, float] = {}
    for organism in TRIPLEMIX_ORGANISMS:
        values = pd.to_numeric(
            data.loc[data["protein_organism"] == organism, "fold_change_normalized"],
            errors="coerce",
        ).dropna()
        if not values.empty:
            medians[organism] = float(values.median())
    return medians


def normalized_log2_fc_to_log10_ratio_2_over_1(fold_change: float) -> float:
    """Convert normalized log2(A/B) to log10(B/A), with 1=A and 2=B."""
    converted = -float(fold_change) * float(np.log10(2.0))
    return 0.0 if converted == 0.0 else converted


def add_ratio_average_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add log10(B/A) and log10((B + A)/2) from log2 A/B abundances."""
    if data.empty:
        return data.copy()
    required = {"abundance_A", "abundance_B", "fold_change_normalized"}
    if not required.issubset(data.columns):
        raise ValueError(f"Ratio/average scatter requires columns: {sorted(required)}")

    transformed = data.copy()
    abundance_a = pd.to_numeric(transformed["abundance_A"], errors="coerce").to_numpy(dtype=float)
    abundance_b = pd.to_numeric(transformed["abundance_B"], errors="coerce").to_numpy(dtype=float)
    normalized_fc = pd.to_numeric(transformed["fold_change_normalized"], errors="coerce").to_numpy(dtype=float)
    ln2 = float(np.log(2.0))
    ln10 = float(np.log(10.0))
    transformed["log10_ratio_2_over_1"] = -normalized_fc * float(np.log10(2.0))
    transformed["log10_average_2_and_1"] = (
        np.logaddexp(abundance_a * ln2, abundance_b * ln2) - ln2
    ) / ln10
    return transformed


def fold_change_distribution_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute organism-wise bias, moment skewness and robust Bowley asymmetry."""
    rows = []
    if data.empty or "fold_change_normalized" not in data.columns:
        return pd.DataFrame(columns=["organism", "n", "median", "expected", "bias", "skewness", "quartile_asymmetry"])
    for organism in TRIPLEMIX_ORGANISMS:
        values = pd.to_numeric(
            data.loc[data["protein_organism"] == organism, "fold_change_normalized"],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        q1, median, q3 = (float(value) for value in values.quantile([0.25, 0.50, 0.75]))
        spread = q3 - q1
        quartile_asymmetry = (q3 + q1 - 2.0 * median) / spread if spread > 0 else 0.0
        expected = float(TRIPLEMIX_EXPECTED_FC[organism])
        rows.append(
            {
                "organism": organism,
                "n": int(values.size),
                "median": median,
                "expected": expected,
                "bias": median - expected,
                "skewness": float(values.skew()) if values.size >= 3 else float("nan"),
                "quartile_asymmetry": float(quartile_asymmetry),
            }
        )
    return pd.DataFrame(rows)


def covariance_ellipse_points(
    x_values,
    y_values,
    *,
    probability: float = 0.90,
    trim_quantile: float = 0.025,
    point_count: int = 121,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a robust central covariance ellipse for a two-dimensional cloud."""
    points = np.column_stack(
        [
            pd.to_numeric(pd.Series(x_values), errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(pd.Series(y_values), errors="coerce").to_numpy(dtype=float),
        ]
    )
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return None
    lower = np.quantile(points, trim_quantile, axis=0)
    upper = np.quantile(points, 1.0 - trim_quantile, axis=0)
    central = points[((points >= lower) & (points <= upper)).all(axis=1)]
    if len(central) < max(3, int(np.ceil(len(points) * 0.50))):
        central = points
    covariance = np.cov(central, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if not np.isfinite(eigenvalues).all() or eigenvalues.max() <= 0:
        return None
    # For chi-square(df=2), the quantile is -2*ln(1-p).
    radius = np.sqrt(-2.0 * np.log(1.0 - min(max(float(probability), 0.01), 0.999)))
    angles = np.linspace(0.0, 2.0 * np.pi, point_count)
    unit_circle = np.vstack((np.cos(angles), np.sin(angles)))
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues) * radius)
    ellipse = central.mean(axis=0)[:, None] + transform @ unit_circle
    return ellipse[0], ellipse[1]


def add_scatter_diagnostics(
    fig: go.Figure,
    data: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    ratio_axis: bool = False,
    show_metrics: bool = True,
) -> go.Figure:
    """Add organism envelopes and compact fold-change shape metrics to a scatter."""
    envelope_legend_shown = False
    for organism in TRIPLEMIX_ORGANISMS:
        current = data.loc[data["protein_organism"] == organism]
        ellipse = covariance_ellipse_points(current[x_column], current[y_column])
        if ellipse is None:
            continue
        ellipse_x, ellipse_y = ellipse
        fig.add_trace(
            go.Scatter(
                x=ellipse_x,
                y=ellipse_y,
                mode="lines",
                name="90% envelope",
                line={"color": TRIPLEMIX_COLORS[organism], "width": 2.25},
                hovertemplate=f"90% central covariance envelope<br>{organism}<extra></extra>",
                showlegend=not envelope_legend_shown,
            )
        )
        envelope_legend_shown = True

    metric_rows = fold_change_distribution_metrics(data) if show_metrics else pd.DataFrame()
    if not metric_rows.empty:
        labels = []
        short_names = {**TRIPLEMIX_SHORT_NAMES, "Arabidopsis thaliana": "Ara"}
        for row in metric_rows.to_dict("records"):
            color = TRIPLEMIX_COLORS[row["organism"]]
            target = row["expected"]
            median = row["median"]
            bias = row["bias"]
            skewness = row["skewness"]
            quartile_asymmetry = row["quartile_asymmetry"]
            if ratio_axis:
                target = normalized_log2_fc_to_log10_ratio_2_over_1(target)
                median = normalized_log2_fc_to_log10_ratio_2_over_1(median)
                bias = median - target
                skewness = -skewness
                quartile_asymmetry = -quartile_asymmetry
            labels.append(
                f"<span style='color:{color}'><b>{short_names[row['organism']]}</b></span> · "
                f"M {median:+.2f} · Δ {bias:+.2f} · skew {skewness:+.2f} · asym {quartile_asymmetry:+.2f}"
            )
        fig.add_annotation(
            x=0.99,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<b>median · bias · shape</b><br>" + "<br>".join(labels),
            showarrow=False,
            align="left",
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            borderpad=5,
            font={"size": 9, "color": "#111827"},
            name="triplemix-diagnostics",
        )
    return fig


def style_triplemix_scatter_layout(fig: go.Figure) -> go.Figure:
    """Keep the organism key compact and reserve the card width for the plot."""
    for trace in fig.data:
        if trace.name in TRIPLEMIX_SHORT_NAMES:
            trace.name = TRIPLEMIX_SHORT_NAMES[trace.name]
    fig.update_layout(
        legend={
            "title_text": "",
            "orientation": "h",
            "x": 1.0,
            "xanchor": "right",
            "y": 1.03,
            "yanchor": "bottom",
            "font": {"size": 11},
            "itemsizing": "constant",
        },
        margin={"l": 70, "r": 25, "t": 95, "b": 70},
        title={"x": 0.0, "xanchor": "left", "font": {"size": 18}},
    )
    return fig


def _samples_from_metrics(metrics: pl.DataFrame) -> dict[str, str]:
    if metrics.is_empty():
        return {"A": "", "B": ""}
    return {"A": metrics.item(0, "samples_A"), "B": metrics.item(0, "samples_B")}


def _analysis_frame(data: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """Expose one normalization method through the canonical plotting columns."""
    if analysis == "ecoli_centered":
        return data
    if analysis != "arabidopsis_reference":
        raise ValueError(f"Unknown Triplemix analysis: {analysis}")
    renamed = data.copy()
    replacements = {
        "arabidopsis_reference_abundance_A": "abundance_A",
        "arabidopsis_reference_abundance_B": "abundance_B",
        "arabidopsis_reference_mean_log10_abundance": "mean_log10_abundance",
        "arabidopsis_reference_fold_change_normalized": "fold_change_normalized",
        "arabidopsis_reference_fold_change_error": "fold_change_error",
        "arabidopsis_reference_expected_fold_change": "expected_fold_change",
    }
    missing = sorted(set(replacements).difference(renamed.columns))
    if missing:
        raise ValueError(f"Arabidopsis reference analysis columns are missing: {', '.join(missing)}")
    for source, target in replacements.items():
        renamed[target] = renamed[source]
    return renamed


def prepare_triplemix_plot_data(
    metrics: pl.DataFrame,
    evidence_filter: str,
    *,
    max_per_organism: int = TRIPLEMIX_SCATTER_POINTS_PER_ORGANISM,
) -> TriplemixPlotData:
    """Filter and transform quantitative rows once per search/filter selection.

    Diagnostics use every complete row. Browser scatter traces use a seeded,
    organism-stratified random sample so they remain representative and compact.
    """
    if max_per_organism <= 0:
        raise ValueError("max_per_organism must be positive")
    data = filter_triplemix_metrics(metrics, evidence_filter).filter(
        pl.col("complete_all_replicates") & pl.col("fold_change_normalized").is_finite()
    )
    if data.is_empty():
        empty = pd.DataFrame()
        return TriplemixPlotData(full=empty, sampled=empty.copy())

    plot_columns = [
        "nterm_modif_peptide",
        "peptide",
        "protein_organism",
        "nterm_modif",
        "specificity",
        "cleavage_class",
        "missed_cleavages",
        "abundance_A",
        "abundance_B",
        "mean_log10_abundance",
        "fold_change_normalized",
        "fold_change_error",
        "expected_fold_change",
        "arabidopsis_reference_abundance_A",
        "arabidopsis_reference_abundance_B",
        "arabidopsis_reference_mean_log10_abundance",
        "arabidopsis_reference_fold_change_normalized",
        "arabidopsis_reference_fold_change_error",
        "arabidopsis_reference_expected_fold_change",
        "raw_abundance_A",
        "raw_abundance_B",
        "raw_fold_change",
        "raw_mean_log10_abundance",
        "raw_complete_all_replicates",
    ]
    data = data.select([column for column in plot_columns if column in data.columns])
    sampled: list[pl.DataFrame] = []
    for index, organism in enumerate(TRIPLEMIX_ORGANISMS):
        current = data.filter(pl.col("protein_organism") == organism)
        if current.height > max_per_organism:
            current = current.sample(
                n=max_per_organism,
                with_replacement=False,
                shuffle=True,
                seed=1729 + index,
            )
        sampled.append(current.sort("nterm_modif_peptide"))

    sampled_data = pl.concat(sampled, how="vertical_relaxed")
    full_frame = add_ratio_average_columns(data.to_pandas())
    sampled_frame = add_ratio_average_columns(sampled_data.to_pandas())
    return TriplemixPlotData(full=full_frame, sampled=sampled_frame)


def _plot_frame(
    metrics: pl.DataFrame,
    evidence_filter: str,
    max_per_organism: int | None = TRIPLEMIX_SCATTER_POINTS_PER_ORGANISM,
) -> pd.DataFrame:
    """Compatibility wrapper for callers that need a full or sampled frame."""
    if max_per_organism is None:
        prepared = prepare_triplemix_plot_data(metrics, evidence_filter)
        return prepared.full
    return prepare_triplemix_plot_data(
        metrics,
        evidence_filter,
        max_per_organism=max_per_organism,
    ).sampled
