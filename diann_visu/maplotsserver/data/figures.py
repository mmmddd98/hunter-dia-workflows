from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from .datasets import summarize_quant_tables

TEMPLE_SRC = Path(__file__).resolve().parents[1] / "external" / "temple_plot_repo"
if TEMPLE_SRC.exists() and str(TEMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(TEMPLE_SRC))

try:
    from temple_plot import TemplePlot
except ImportError:  # pragma: no cover - keeps the server usable if the optional external is absent.
    TemplePlot = None


PLOTLY_TEMPLATE = "plotly_white"
SELECTION_PLOT_HEIGHT = 300
SELECTION_OVERVIEW_HEIGHT = 320
TEMPLE_PLOT_HEIGHT = 340
FOLLOWUP_BAR_WIDTH = 0.18
ENTRAPMENT_COLOR = "#E4572E"
TARGET_COLOR = "#202124"
PROGRAM_DECOY_COLOR = "#8A8F98"
QVALUE_REFERENCE_COLOR = "#4B5563"
PUBLICATION_GRAPH_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "modeBarButtonsToRemove": ["toImage"],
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "maplot_publication_figure",
        "scale": 1,
    },
}
LABEL_MODE_OPTIONS = [
    {"label": "Short labels", "value": "short"},
    {"label": "Compact labels", "value": "compact"},
    {"label": "Full labels", "value": "full"},
]
DEFAULT_LABEL_MODE = "short"
PLOT_STYLE_OPTIONS = [
    {"label": "Interactive", "value": "default"},
    {"label": "Paper / publication", "value": "publication"},
]
DEFAULT_PLOT_STYLE = "default"
DEFAULT_LABEL_SCALE = 1.0


def publication_graph_config(filename: str) -> dict:
    config = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in PUBLICATION_GRAPH_CONFIG.items()
    }
    config["toImageButtonOptions"]["filename"] = filename
    return config


def normalize_label_mode(mode: str | None) -> str:
    values = {option["value"] for option in LABEL_MODE_OPTIONS}
    return str(mode) if mode in values else DEFAULT_LABEL_MODE


def normalize_plot_style(style: str | None) -> str:
    values = {option["value"] for option in PLOT_STYLE_OPTIONS}
    return str(style) if style in values else DEFAULT_PLOT_STYLE


def normalize_label_scale(value) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LABEL_SCALE
    return min(max(scale, 0.8), 1.8)


def apply_plot_display(fig: go.Figure, plot_style: str | None = None, label_scale=None) -> go.Figure:
    style = normalize_plot_style(plot_style)
    scale = normalize_label_scale(label_scale)
    if style == "publication":
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial, Helvetica, sans-serif", color="#111827"),
            margin=dict(l=72, r=36, t=82, b=82),
            legend=dict(itemsizing="constant"),
            hoverlabel=dict(bgcolor="white", font_size=max(int(12 * scale), 10), font_color="#111827", bordercolor="#CBD5E1"),
        )
        fig.update_xaxes(showline=True, linecolor="#111827", linewidth=1.2, mirror=False, gridcolor="#E5E7EB", zeroline=False, automargin=True)
        fig.update_yaxes(showline=True, linecolor="#111827", linewidth=1.2, mirror=False, gridcolor="#E5E7EB", zeroline=False, automargin=True)
        fig.update_traces(selector={"type": "bar"}, marker_line_width=0)
        fig.update_traces(selector={"type": "scatter"}, line_width=3)

    title_size = max(int(18 * scale), 13)
    axis_title_size = max(int(14 * scale), 10)
    tick_size = max(int(12 * scale), 9)
    legend_size = max(int(12 * scale), 9)
    annotation_size = max(int(12 * scale), 9)
    fig.update_layout(
        font=dict(size=tick_size),
        title_font=dict(size=title_size),
        legend=dict(font=dict(size=legend_size), title_font=dict(size=legend_size)),
    )
    fig.update_xaxes(title_font=dict(size=axis_title_size), tickfont=dict(size=tick_size))
    fig.update_yaxes(title_font=dict(size=axis_title_size), tickfont=dict(size=tick_size))
    for annotation in fig.layout.annotations or []:
        font = annotation.font.to_plotly_json() if annotation.font else {}
        if getattr(annotation, "name", None) == "triplemix-diagnostics":
            font["size"] = max(int(9 * scale), 8)
        else:
            font["size"] = annotation_size
        annotation.font = font
    return fig


def display_search_label(label: str, mode: str | None = None) -> str:
    mode = normalize_label_mode(mode)
    text = str(label or "")
    if mode == "full":
        return text
    clean = text.removeprefix("tuned/").replace("__", "/")
    external_label = external_search_display_label(clean)
    if external_label is not None:
        return external_label
    restricted_label = restricted_layers_display_label(clean)
    if restricted_label is not None:
        return restricted_label
    if clean.lower().startswith("carafe/"):
        parts = [part for part in clean.split("/") if part]
        if mode == "full":
            return clean
        if len(parts) >= 3:
            return f"Carafe {parts[-2]} · {parts[-1]}"
        return f"Carafe {parts[-1]}" if parts else "Carafe"
    clean = re.sub(r"\s*[·-]\s*\d{8}\s+\d{6}\s*$", "", clean)
    clean = re.sub(r"[_\s-]+\d{8}[_\s-]*\d{6}\s*$", "", clean)
    clean = clean.replace("_", " ")
    clean = re.sub(r"\brt fr lr\b", "rt+fr", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\brt fr\b", "rt+fr", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bfr lr\b", "fr", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b0p(\d+)\b", r"0.\1", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if mode == "compact":
        return clean

    lowered = clean.lower()
    profile = next((value for value in ("baseline", "balanced", "fast", "slow") if value in lowered), None)
    model = "rt+fr" if "rt+fr" in lowered else "rt" if re.search(r"(^| )rt($| )", lowered) else "fr" if re.search(r"(^| )fr($| )", lowered) else None
    q_match = re.search(r"(?:q|qvalue)\s*([.]?\d+(?:\.\d+)?)", lowered)
    lib_match = re.search(r"(?:lib|outlibq|out-lib)\s*([.]?\d+(?:\.\d+)?)", lowered)
    bits = [bit for bit in [profile, model, f"q{q_match.group(1)}" if q_match else None, f"lib{lib_match.group(1)}" if lib_match else None] if bit]
    if bits:
        return " ".join(bits)
    return clean if len(clean) <= 28 else f"{clean[:25]}..."


def external_search_display_label(search: str) -> str | None:
    """Return a readable label for ``external/<model>/<run>`` searches."""
    parts = [part for part in str(search).strip("/").split("/") if part]
    if len(parts) != 3 or parts[0].lower() != "external":
        return None
    model_id, run_id = parts[1:]
    return f"External model · {model_id.replace('_', ' ')} · {run_id.replace('_', ' ')}"


def restricted_layers_display_label(search: str) -> str | None:
    """Give a restricted-layer Step-5 directory one readable run label."""
    parts = [part for part in str(search).strip("/").split("/") if part]
    if len(parts) < 3:
        return None
    variant, profile, lr_dir = parts[:3]
    if not (
        variant.startswith("tuned_")
        and "restricted" in variant.lower()
        and "layer" in variant.lower()
        and lr_dir.startswith("lr_")
    ):
        return None
    lr = lr_dir.removeprefix("lr_").replace("p", ".")
    profile_label = profile.replace("_", " ")
    return f"Restricted layers · {profile_label} · LR {lr}"


def display_search_label_map(labels, mode: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    used: dict[str, int] = {}
    for label in labels or []:
        text = display_search_label(str(label), mode)
        count = used.get(text, 0) + 1
        used[text] = count
        out[str(label)] = text if count == 1 else f"{text} ({count})"
    return out


def apply_search_label_mode(data: pd.DataFrame, mode: str | None, column: str = "search", display_column: str | None = None) -> pd.DataFrame:
    if data.empty or column not in data.columns:
        return data
    out = data.copy()
    display_column = display_column or column
    label_map = display_search_label_map(out[column].dropna().astype(str).unique(), mode)
    out[display_column] = out[column].astype(str).map(label_map).fillna(out[column].astype(str))
    return out


def empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=SELECTION_PLOT_HEIGHT, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def cap_followup_bar_width(fig: go.Figure, width: float = FOLLOWUP_BAR_WIDTH) -> go.Figure:
    """Keep single-selection follow-up bars from filling the whole plot width."""
    fig.update_traces(width=width, selector={"type": "bar"})
    return fig


def search_summary_figure(raw_tables: dict[str, pl.DataFrame], label_mode: str | None = None) -> go.Figure:
    rows = []
    for search, df in raw_tables.items():
        rows.append(
            {
                "search": search,
                "rows": df.height,
                "precursors": n_unique(df, "Precursor.Id"),
                "peptides": n_unique(df, "Stripped.Sequence"),
                "protein_groups": n_unique(df, "Protein.Group"),
            }
        )
    if not rows:
        return empty_figure("No raw search tables found")
    data = apply_search_label_mode(pd.DataFrame(rows), label_mode, display_column="search_label").melt(
        id_vars=["search", "search_label"],
        var_name="metric",
        value_name="count",
    )
    fig = px.bar(data, x="search_label", y="count", color="metric", barmode="group", template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Search summary", xaxis_title="Search", yaxis_title="Count", height=460)
    return style_axes(fig)


def intensity_histogram(adapted_tables: dict[str, pl.DataFrame], raw_tables: dict[str, pl.DataFrame], label_mode: str | None = None) -> go.Figure:
    rows = []
    for search, df in adapted_tables.items():
        sample_cols = [col for col in df.columns if col not in {"nterm_modif_peptide", "nterm_modif", "peptide", "protein"}]
        if not sample_cols:
            continue
        values = (
            df.select(pl.concat_list([pl.col(col).cast(pl.Float64, strict=False) for col in sample_cols]).alias("values"))
            .explode("values")
            .drop_nulls()
            .head(25000)
            .to_pandas()["values"]
        )
        for value in values:
            if value and value > 0:
                rows.append({"search": search, "log10_intensity": math.log10(value)})
    if not rows:
        for search, df in raw_tables.items():
            for candidate in ("Precursor.Normalised", "Precursor.Quantity", "Quantity"):
                if candidate in df.columns:
                    values = df.select(pl.col(candidate).cast(pl.Float64, strict=False).alias("value")).drop_nulls().head(25000).to_pandas()["value"]
                    rows.extend({"search": search, "log10_intensity": math.log10(v)} for v in values if v and v > 0)
                    break
    if not rows:
        return empty_figure("No intensity columns found")
    data = apply_search_label_mode(pd.DataFrame(rows), label_mode, display_column="search_label")
    fig = px.histogram(data, x="log10_intensity", color="search_label", nbins=55, template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Intensity distribution", xaxis_title="log10 intensity", yaxis_title="Rows", height=460)
    return style_axes(fig)


def nterm_overview(termini_tables: dict[str, pl.DataFrame], label_mode: str | None = None) -> go.Figure:
    rows = []
    for search, df in termini_tables.items():
        if "specificity" in df.columns:
            counts = (
                df.select(pl.col("specificity").cast(pl.Utf8).fill_null("NA").alias("specificity"), pl.col("nterm_modif_peptide").cast(pl.Utf8).alias("entity"))
                .drop_nulls("entity")
                .unique()
                .group_by("specificity")
                .len()
                .rename({"len": "count"})
                .to_dicts()
            )
            rows.extend({"search": search, **row} for row in counts)
    if not rows:
        return empty_figure("No termini.parquet tables found")
    data = apply_search_label_mode(pd.DataFrame(rows), label_mode, display_column="search_label")
    fig = px.bar(data, x="search_label", y="count", color="specificity", barmode="stack", template=PLOTLY_TEMPLATE)
    fig.update_layout(title="N-termini specificity overview", xaxis_title="Search", yaxis_title="N-termini", height=460)
    return style_axes(fig)


def quant_completeness_figure(adapted_tables: dict[str, pl.DataFrame], label_mode: str | None = None) -> go.Figure:
    data = summarize_quant_tables(adapted_tables)
    if data.empty:
        return empty_figure("No adapted matrices found")
    label_map = display_search_label_map(data["search"].dropna().astype(str).unique(), label_mode)
    data = data.copy()
    data["search_label"] = data["search"].astype(str).map(label_map).fillna(data["search"].astype(str))
    table = data.pivot(index="sample", columns="search_label", values="fraction_present").fillna(0)
    fig = px.imshow(table, zmin=0, zmax=1, color_continuous_scale="Greys", template=PLOTLY_TEMPLATE, aspect="auto")
    fig.update_layout(title="Quant completeness", xaxis_title="Search", yaxis_title="Sample", height=max(430, 220 + 18 * len(table)))
    return fig


def entrapment_summary_figure(
    summary: pl.DataFrame,
    fdr_definition: str = "rate",
    qvalue_threshold: float | None = None,
    group_label: str = "specificity",
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
) -> go.Figure:
    if summary.is_empty():
        return empty_figure("No entrapment hits found for this dataset.")
    data = apply_search_label_mode(summary.to_pandas(), label_mode, display_column="search_label")
    data["search_label"] = data["search"].astype(str).map(custom_labels or {}).fillna(data["search_label"])
    if "level" in data.columns:
        data["level_label"] = data["level"].map(entrapment_level_label)
    y_col = entrapment_metric_column(fdr_definition)
    hover_cols = ["target_ids", "entrapment_ids", "entrapment_fdr_estimate", "entrapment_fraction"]
    if "diann_decoy_ids" in data.columns:
        hover_cols.extend(["diann_decoy_ids", "diann_decoy_fdr_estimate"])
    facet_col = "level_label" if "specificity_group" in data.columns and data["level"].nunique() > 1 else None
    facet_kwargs = {"facet_col": facet_col, "facet_col_wrap": 2} if facet_col else {}
    group_phrase = "" if group_label == "specificity" else f", split by {group_label}"
    fig = px.bar(
        data,
        x="search_label",
        y=y_col,
        color="specificity_group" if "specificity_group" in data.columns else "level",
        barmode="group",
        hover_data=[col for col in [*hover_cols, "level", "specificity_group"] if col in data.columns],
        template=PLOTLY_TEMPLATE,
        labels={"specificity_group": group_label, "level": "level", "level_label": "level"},
        **facet_kwargs,
    )
    fig.update_layout(
        title=(
            f"Empirical entrapment rate across entity levels{group_phrase} at DIA-NN Q.Value <= {float(qvalue_threshold):g}"
            if qvalue_threshold is not None
            else f"Empirical entrapment rate across entity levels{group_phrase}"
        ),
        xaxis_title="Search",
        yaxis_title=entrapment_metric_label(fdr_definition),
        height=620 if facet_col else 480,
        legend_title_text=group_label,
    )
    fig.update_yaxes(tickformat=".2%")
    return style_publication_axes(fig)


def entrapment_curve_figure(
    curve: pl.DataFrame,
    *,
    fdr_definition: str = "rate",
    qvalue_threshold: float | None = None,
    max_qvalue: float | None = None,
    show_qvalue_reference: bool = True,
    group_label: str = "specificity",
    split_groups: bool = True,
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
) -> go.Figure:
    if curve.is_empty():
        return empty_figure("No Q.Value curve can be computed for this dataset.")
    data = apply_search_label_mode(curve.to_pandas(), label_mode, display_column="search_label")
    data["search_label"] = data["search"].astype(str).map(custom_labels or {}).fillna(data["search_label"])
    if "level" in data.columns:
        data["level_label"] = data["level"].map(entrapment_level_label)
    has_methods = "method" in data.columns
    n_levels = int(data["level"].nunique()) if "level" in data.columns else 0
    level_label = entrapment_level_label(data["level"].dropna().iloc[0]) if n_levels == 1 and data["level"].notna().any() else "Selected levels"
    entrapment_data = data
    if has_methods:
        entrapment_data = data.loc[data["method"] == "ENT_ entrapment"].copy()
    group_facet_col = "specificity_group" if "specificity_group" in entrapment_data.columns and entrapment_data["specificity_group"].nunique() > 1 else None
    if not split_groups:
        group_facet_col = None
    level_facet_row = "level_label" if "level_label" in entrapment_data.columns and entrapment_data["level"].nunique() > 1 else None
    facet_kwargs = {}
    if group_facet_col:
        facet_kwargs["facet_col"] = group_facet_col
    if level_facet_row:
        facet_kwargs["facet_row"] = level_facet_row
    elif not group_facet_col and "level_label" in entrapment_data.columns and entrapment_data["level"].nunique() > 1:
        facet_kwargs = {"facet_col": "level_label", "facet_col_wrap": 2}
    group_phrase = "" if group_label == "specificity" else f", split by {group_label}"
    fig = px.line(
        entrapment_data,
        x="qvalue",
        y="entrapment_fraction" if fdr_definition == "fraction" else "fdr_estimate",
        color="search_label",
        line_dash="specificity_group" if not split_groups and "specificity_group" in entrapment_data.columns else None,
        hover_data=[
            col
            for col in [
                "specificity_group",
                "level",
                "cum_target",
                "cum_entrapment",
                "cum_diann_decoy",
                "entrapment_fdr_estimate",
                "entrapment_fraction",
                "rank",
            ]
            if col in entrapment_data.columns
        ],
        template=PLOTLY_TEMPLATE,
        labels={"specificity_group": group_label, "level": "level", "level_label": "level"},
        **facet_kwargs,
    )
    fig.update_traces(line={"width": 3.0}, opacity=0.95)
    max_q = max(float(max_qvalue) if max_qvalue is not None else float(data["qvalue"].max()), 0.01)
    if show_qvalue_reference:
        fig.add_trace(
            go.Scatter(
                x=[0, max_q],
                y=[0, max_q],
                mode="lines",
                name="DIA-NN Q.Value threshold reference",
                line={"color": QVALUE_REFERENCE_COLOR, "dash": "dash", "width": 2.2},
                hovertemplate="DIA-NN Q.Value threshold reference<br>Q.Value: %{x:.4f}<extra></extra>",
            )
        )
        add_qvalue_reference_shapes(fig, max_q)
    if qvalue_threshold is not None:
        fig.add_vline(
            x=float(qvalue_threshold),
            line_width=1.5,
            line_dash="dot",
            line_color="#111111",
            annotation_text=f"selected <= {float(qvalue_threshold):g}",
            annotation_position="top left",
        )
    fig.update_layout(
        title=(
            f"{level_label}: Cumulative empirical FDR vs DIA-NN Q.Value threshold{group_phrase} "
            f"({entrapment_metric_label(fdr_definition)}, selected <= {float(qvalue_threshold):g})"
            if qvalue_threshold is not None
            else f"{level_label}: Cumulative empirical FDR vs DIA-NN Q.Value threshold{group_phrase} ({entrapment_metric_label(fdr_definition)})"
        ),
        xaxis_title="DIA-NN Q.Value threshold",
        yaxis_title=entrapment_metric_label(fdr_definition),
        height=curve_figure_height(entrapment_data, group_facet_col, level_facet_row),
        legend_title_text="Search",
    )
    fig.update_xaxes(range=[0, max_q])
    y_col = "entrapment_fraction" if fdr_definition == "fraction" else "fdr_estimate"
    visible_y = entrapment_data.loc[entrapment_data["qvalue"].le(max_q), y_col]
    visible_y = visible_y.dropna()
    if not visible_y.empty:
        y_max = max(float(visible_y.max()) * 1.12, max_q if show_qvalue_reference else 0.0, 0.01)
        fig.update_yaxes(tickformat=".2%", range=[0, y_max])
    else:
        fig.update_yaxes(tickformat=".2%")
    return style_publication_axes(fig)


def add_qvalue_reference_shapes(fig: go.Figure, max_q: float) -> None:
    """Draw y=x reference lines in every facet subplot."""
    xaxis_names = sorted(
        [name for name in fig.layout if re.fullmatch(r"xaxis\d*", name)],
        key=lambda name: int(name.removeprefix("xaxis") or "1"),
    )
    for name in xaxis_names:
        xaxis = fig.layout[name]
        yref = getattr(xaxis, "anchor", None)
        if not isinstance(yref, str) or not yref.startswith("y"):
            continue
        xref = name.replace("axis", "")
        fig.add_shape(
            type="line",
            x0=0,
            x1=max_q,
            y0=0,
            y1=max_q,
            xref=xref,
            yref=yref,
            line={"color": QVALUE_REFERENCE_COLOR, "dash": "dash", "width": 1.7},
            layer="below",
        )


def curve_figure_height(data: pd.DataFrame, group_facet_col: str | None, level_facet_row: str | None) -> int:
    if level_facet_row:
        n_rows = max(int(data[level_facet_row].nunique()), 1)
        return max(520, 360 * n_rows)
    if group_facet_col:
        return 540
    return 480


def entrapment_distribution_figure(
    distribution: pl.DataFrame,
    *,
    fdr_definition: str = "rate",
    show_program_decoys: bool = True,
    label_mode: str | None = None,
) -> go.Figure:
    if distribution.is_empty():
        return empty_figure("No entrapment Q.Value distribution can be computed for this dataset.")
    data = apply_search_label_mode(distribution.to_pandas(), label_mode, display_column="search_label")
    level_label = entrapment_level_label(data["level"].dropna().iloc[0]) if "level" in data.columns and data["level"].notna().any() else "selected level"
    data["qvalue_bin"] = data["qvalue_bin_start"].map(lambda value: f"{value:.2f}")
    has_diann_decoys = show_program_decoys and "diann_decoy" in data.columns and data["diann_decoy"].sum() > 0
    fdr_data = data.copy()
    fdr_data["local_empirical_fdr"] = fdr_data["bin_entrapment_fraction" if fdr_definition == "fraction" else "bin_entrapment_fdr_estimate"]
    fdr_fig = px.bar(
        fdr_data,
        x="qvalue_bin",
        y="local_empirical_fdr",
        color="search_label",
        barmode="group",
        hover_data={
            "qvalue_bin": False,
            "qvalue_bin_start": ":.4f",
            "qvalue_bin_end": ":.4f",
            "target": ":,",
            "entrapment": ":,",
            "diann_decoy": ":," if has_diann_decoys else False,
            "total_ids": ":,",
            "bin_entrapment_fdr_estimate": ":.2%",
            "bin_entrapment_fraction": ":.2%",
            "local_empirical_fdr": ":.2%",
        },
        template=PLOTLY_TEMPLATE,
    )
    count_values = ["target", "entrapment", "diann_decoy"] if has_diann_decoys else ["target", "entrapment"]
    count_data = data.melt(
        id_vars=["search_label", "qvalue_bin", "qvalue_bin_start", "qvalue_bin_end"],
        value_vars=count_values,
        var_name="origin",
        value_name="ids",
    )
    count_fig = px.bar(
        count_data,
        x="qvalue_bin",
        y="ids",
        color="origin",
        facet_row="search_label",
        barmode="stack",
        color_discrete_map={"target": TARGET_COLOR, "entrapment": ENTRAPMENT_COLOR, "diann_decoy": PROGRAM_DECOY_COLOR},
        hover_data={"qvalue_bin": False, "qvalue_bin_start": ":.4f", "qvalue_bin_end": ":.4f", "ids": ":,"},
        template=PLOTLY_TEMPLATE,
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Local/bin empirical FDR, not cumulative", "Target, ENT_, and program-decoy IDs per Q.Value bin" if has_diann_decoys else "Target and ENT_ IDs per Q.Value bin"),
    )
    for trace in fdr_fig.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in count_fig.data:
        trace.showlegend = False
        fig.add_trace(trace, row=2, col=1)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"{level_label}: Local empirical FDR per DIA-NN Q.Value bin ({entrapment_metric_label(fdr_definition)})",
        height=max(720, 520 + 80 * data["search_label"].nunique()),
        legend_title_text="Search",
    )
    fig.update_xaxes(title_text="DIA-NN Q.Value bin", row=2, col=1)
    fig.update_yaxes(title_text=entrapment_metric_label(fdr_definition), tickformat=".2%", row=1, col=1)
    fig.update_yaxes(title_text="IDs", row=2, col=1)
    return style_axes(fig)


def entrapment_gain_figure(
    metrics: pl.DataFrame,
    baseline_search: str | None,
    fdr_definition: str = "rate",
    qvalue_threshold: float | None = None,
    label_mode: str | None = None,
) -> go.Figure:
    if metrics.is_empty() or not baseline_search:
        return empty_figure("No threshold metrics available for ID gain.")
    data = apply_search_label_mode(metrics.to_pandas(), label_mode, display_column="search_label")
    baseline = data.loc[data["search"] == baseline_search]
    if baseline.empty:
        return empty_figure("Selected baseline search is not available.")
    base_target = int(baseline.iloc[0]["n_target"])
    base_entrapment = int(baseline.iloc[0]["n_entrapment"])
    plot_data = data.loc[data["search"] != baseline_search].copy()
    if plot_data.empty:
        return empty_figure("No tuned search to compare against baseline.")
    plot_data["target_gain_vs_baseline"] = plot_data["n_target"] - base_target
    plot_data["entrapment_gain_vs_baseline"] = plot_data["n_entrapment"] - base_entrapment
    color_col = "entrapment_fraction" if fdr_definition == "fraction" else "entrapment_rate"
    fig = px.scatter(
        plot_data,
        x="target_gain_vs_baseline",
        y="entrapment_gain_vs_baseline",
        color=color_col,
        size="n_target",
        hover_name="search_label",
        hover_data={
            "n_target": ":,",
            "n_entrapment": ":,",
            "n_program_decoy": ":,",
            "entrapment_rate": ":.2%",
            "entrapment_fraction": ":.2%",
            "target_gain_vs_baseline": ":,",
            "entrapment_gain_vs_baseline": ":,",
        },
        color_continuous_scale="OrRd",
        template=PLOTLY_TEMPLATE,
    )
    fig.add_hline(y=0, line_width=1.4, line_dash="dash", line_color=QVALUE_REFERENCE_COLOR)
    fig.add_vline(x=0, line_width=1.4, line_dash="dash", line_color=QVALUE_REFERENCE_COLOR)
    fig.update_layout(
        title=f"ID gain vs entrapment gain{f' at Q.Value <= {float(qvalue_threshold):g}' if qvalue_threshold is not None else ''}",
        xaxis_title=f"Target ID gain vs {baseline_search}",
        yaxis_title=f"ENT_ ID gain vs {baseline_search}",
        coloraxis_colorbar_title=entrapment_metric_label(fdr_definition),
        height=480,
    )
    return style_axes(fig)


def program_decoy_vs_entrapment_figure(metrics: pl.DataFrame, qvalue_threshold: float | None = None) -> go.Figure:
    if metrics.is_empty():
        return empty_figure("No threshold metrics available for program-decoy comparison.")
    data = metrics.to_pandas()
    max_value = max(float(data["program_decoy_rate"].max()), float(data["entrapment_rate"].max()), 0.01)
    fig = px.scatter(
        data,
        x="program_decoy_rate",
        y="entrapment_rate",
        color="search",
        size="n_target",
        hover_name="search",
        hover_data={
            "n_target": ":,",
            "n_entrapment": ":,",
            "n_program_decoy": ":,",
            "entrapment_rate": ":.2%",
            "program_decoy_rate": ":.2%",
        },
        template=PLOTLY_TEMPLATE,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            name="y = x",
            line={"color": QVALUE_REFERENCE_COLOR, "dash": "dash", "width": 1.8},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"Program decoys vs entrapment{f' at Q.Value <= {float(qvalue_threshold):g}' if qvalue_threshold is not None else ''}",
        xaxis_title="Program-decoy rate / target IDs",
        yaxis_title="ENT_ / target IDs",
        height=480,
    )
    fig.update_xaxes(tickformat=".2%", range=[0, max_value * 1.08])
    fig.update_yaxes(tickformat=".2%", range=[0, max_value * 1.08])
    return style_axes(fig)


def entrapment_metric_column(fdr_definition: str) -> str:
    return "entrapment_fraction" if fdr_definition == "fraction" else "entrapment_fdr_estimate"


def entrapment_metric_label(fdr_definition: str) -> str:
    return "Empirical ENT_ fraction (ENT_ / [target + ENT_])" if fdr_definition == "fraction" else "Empirical FDR (ENT_ / target IDs)"


def entrapment_level_label(level: str) -> str:
    labels = {
        "precursor": "Precursor level",
        "peptidoform": "Peptidoform level",
        "peptide": "Peptide level",
        "protein_group": "Protein-group level",
        "nterm_precursor": "N-term precursor level",
        "nterm_peptide": "N-term peptide level",
        "nterm_site": "N-term site level",
        "nterm_protein": "N-term protein level",
    }
    return labels.get(str(level), f"{level} level")


def ram_usage_figure(
    ram_tables: list[tuple[dict, pd.DataFrame]],
    mode: str = "pipeline",
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
    step_prefix: str = "step",
) -> go.Figure:
    frames = []
    for meta, df in ram_tables:
        if df.empty:
            continue
        value_col = "pipeline_gb" if mode == "pipeline" else "system_used_gb"
        frame = df[["elapsed_min", "timestamp", value_col]].copy()
        frame = frame.rename(columns={value_col: "ram_gb"})
        frame["log"] = ram_label(meta, label_mode, custom_labels, step_prefix)
        frame["file"] = meta["label"]
        frame["step"] = meta["step"]
        frames.append(frame)
    if not frames:
        return empty_figure("No RAM log data for this selection.")
    data = pd.concat(frames, ignore_index=True)
    color_map = ram_color_map(data["log"].unique())
    fig = px.line(
        data,
        x="elapsed_min",
        y="ram_gb",
        color="log",
        color_discrete_map=color_map,
        hover_data=["timestamp", "step"],
        template=PLOTLY_TEMPLATE,
    )
    title = "Pipeline RAM over time" if mode == "pipeline" else "System used RAM over time"
    fig.update_layout(title=title, xaxis_title="Elapsed time (min)", yaxis_title="RAM (GB)", height=560)
    return style_ram_legend(style_axes(fig))


def ram_summary_figure(
    ram_tables: list[tuple[dict, pd.DataFrame]],
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
    step_prefix: str = "step",
) -> go.Figure:
    rows = []
    for meta, df in ram_tables:
        if df.empty:
            continue
        rows.append(
            {
                "log": meta["label"],
                "label": ram_label(meta, label_mode, custom_labels, step_prefix),
                "step": meta["step"],
                "peak_pipeline_gb": df["pipeline_gb"].max(),
                "median_pipeline_gb": df["pipeline_gb"].median(),
                "duration_min": df["elapsed_min"].max(),
            }
        )
    if not rows:
        return empty_figure("No RAM summary available.")
    data = pd.DataFrame(rows).sort_values("peak_pipeline_gb", ascending=False)
    color_map = ram_color_map(data["label"].unique())
    fig = px.bar(
        data,
        x="peak_pipeline_gb",
        y="label",
        color="label",
        color_discrete_map=color_map,
        orientation="h",
        hover_data={"log": True, "step": True, "median_pipeline_gb": ":.2f", "duration_min": ":.1f", "label": False},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title="Peak pipeline RAM by run", xaxis_title="Peak pipeline RAM (GB)", yaxis_title="", height=max(420, 90 + 28 * len(data)))
    fig.update_yaxes(autorange="reversed")
    return style_ram_legend(style_axes(fig))


def ram_duration_timeline_figure(
    ram_tables: list[tuple[dict, pd.DataFrame]],
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
    step_prefix: str = "step",
) -> go.Figure:
    data = ram_duration_table(ram_tables, label_mode, custom_labels, step_prefix)
    if data.empty:
        return empty_figure("No duration data available.")
    color_map = ram_color_map(data["label"].unique())
    fig = px.timeline(
        data,
        x_start="start",
        x_end="end",
        y="label",
        color="label",
        color_discrete_map=color_map,
        hover_data={"log": True, "step": True, "duration_min": ":.1f", "peak_pipeline_gb": ":.2f", "label": False},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title="RAM log timeline",
        xaxis_title="Wall time",
        yaxis_title="",
        height=max(460, 110 + 30 * len(data)),
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return style_ram_legend(style_axes(fig))


def ram_duration_summary_figure(
    ram_tables: list[tuple[dict, pd.DataFrame]],
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
    step_prefix: str = "step",
) -> go.Figure:
    data = ram_duration_table(ram_tables, label_mode, custom_labels, step_prefix)
    if data.empty:
        return empty_figure("No duration data available.")
    data = data.sort_values("duration_min", ascending=False)
    color_map = ram_color_map(data["label"].unique())
    fig = px.bar(
        data,
        x="duration_min",
        y="label",
        color="label",
        color_discrete_map=color_map,
        orientation="h",
        hover_data={"log": True, "step": True, "start": True, "end": True, "peak_pipeline_gb": ":.2f", "label": False},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title="Duration by run",
        xaxis_title="Duration (min)",
        yaxis_title="",
        height=max(460, 110 + 30 * len(data)),
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return style_ram_legend(style_axes(fig))


def ram_duration_table(
    ram_tables: list[tuple[dict, pd.DataFrame]],
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
    step_prefix: str = "step",
) -> pd.DataFrame:
    rows = []
    for meta, df in ram_tables:
        if df.empty:
            continue
        rows.append(
            {
                "log": meta["label"],
                "label": ram_label(meta, label_mode, custom_labels, step_prefix),
                "step": meta["step"],
                "profile": ram_profile(meta),
                "model": ram_model(meta),
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
                "duration_min": df["elapsed_min"].max(),
                "peak_pipeline_gb": df["pipeline_gb"].max(),
            }
        )
    return pd.DataFrame(rows)


def progress_timeline_figure(events: list[dict]) -> go.Figure:
    rows = progress_duration_rows(events)
    if not rows:
        return empty_figure("No progress events yet.")
    data = pd.DataFrame(rows)
    color_map = {"OK": "#2ca02c", "START": "#1f77b4", "FAIL": "#d62728", "open": "#9467bd"}
    fig = px.timeline(
        data,
        x_start="start",
        x_end="end",
        y="label",
        color="state",
        color_discrete_map=color_map,
        hover_data=["step", "details", "duration_min"],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title="Pipeline step timeline",
        xaxis_title="Wall time",
        yaxis_title="",
        height=max(460, 120 + 30 * len(data)),
        showlegend=True,
    )
    fig.update_yaxes(autorange="reversed")
    return style_axes(fig)


def progress_state_figure(events: list[dict]) -> go.Figure:
    if not events:
        return empty_figure("No progress events yet.")
    rows = []
    for event in events:
        rows.append({"step": event["step_label"], "state": event["state"]})
    data = pd.DataFrame(rows).groupby(["step", "state"], as_index=False).size()
    fig = px.bar(data, x="step", y="size", color="state", barmode="stack", template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Progress event counts", xaxis_title="Step", yaxis_title="Events", height=430)
    fig.update_xaxes(tickangle=30)
    return style_axes(fig)


def live_status_cards(summary: dict) -> list[dict[str, str]]:
    return [
        {"label": "Status", "value": summary.get("status") or "unknown"},
        {"label": "PID", "value": str(summary.get("pid") or "none")},
        {"label": "Current step", "value": summary.get("current_step") or "none open"},
        {"label": "Started", "value": summary.get("started_at") or "unknown"},
        {"label": "Last activity", "value": summary.get("last_activity") or "unknown"},
        {"label": "RAM logs", "value": str(summary.get("ram_log_count", 0))},
    ]


def progress_duration_rows(events: list[dict]) -> list[dict]:
    rows = []
    open_events: dict[tuple[str, tuple[tuple[str, str], ...]], dict] = {}
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    for event in events:
        start = event.get("timestamp_dt")
        if start is None:
            continue
        key_details = tuple(sorted((key, value) for key, value in event["details"].items() if key not in {"elapsed"}))
        key = (event["step"], key_details)
        if event["state"] == "START":
            open_events[key] = event
            continue
        if event["state"] not in {"OK", "FAIL"}:
            continue
        start_event = open_events.pop(key, None)
        if start_event is None:
            continue
        end = start
        rows.append(
            {
                "label": start_event["step_label"],
                "step": start_event["step"],
                "state": event["state"],
                "start": timeline_time(start_event["timestamp_dt"]),
                "end": timeline_time(end),
                "duration_min": (end - start_event["timestamp_dt"]).total_seconds() / 60,
                "details": "; ".join(f"{k}={v}" for k, v in start_event["details"].items()),
            }
        )
    for event in open_events.values():
        start = event.get("timestamp_dt")
        if start is None:
            continue
        rows.append(
            {
                "label": event["step_label"],
                "step": event["step"],
                "state": "open",
                "start": timeline_time(start),
                "end": timeline_time(now),
                "duration_min": (now - start.astimezone(now.tzinfo)).total_seconds() / 60,
                "details": "; ".join(f"{k}={v}" for k, v in event["details"].items()),
            }
        )
    return rows


def timeline_time(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.to_pydatetime()


def ram_color_map(labels) -> dict:
    return {label: ram_color_for_label(label) for label in sorted(labels)}


def ram_color_for_label(label: str) -> str:
    text = str(label).lower()
    if "out-lib" in text:
        return "#118d8d"
    if "baseline" in text:
        return "#d62728"
    if "balanced" in text:
        return "#1f77b4" if "rt+fr" in text else "#7db7e8" if " rt" in text else "#4f92cf"
    if "fast" in text:
        return "#2ca02c" if "rt+fr" in text else "#8fd18b" if " rt" in text else "#43b56b"
    if "slow" in text:
        return "#7b61ff" if "rt+fr" in text else "#d85ca3" if " rt" in text else "#9b6bd3"
    if "step 1" in text or "predlib" in text:
        return "#4c5bdc"
    if "step 2" in text:
        return "#36a187"
    if "step 3" in text:
        return "#f28e2b"
    return "#666666"


def ram_profile(meta: dict) -> str:
    label = ram_label(meta).lower()
    if "out-lib" in label:
        return "out-lib sweep"
    for profile in ("balanced", "fast", "slow"):
        if profile in label:
            return profile
    if "baseline" in label:
        return "baseline"
    return "other"


def ram_model(meta: dict) -> str:
    label = ram_label(meta).lower()
    if "out-lib" in label:
        return "q sweep"
    if "rt+fr" in label:
        return "rt+fr"
    if re.search(r"(^| )rt($| )", label):
        return "rt"
    return ""


def ram_label(
    meta: dict,
    label_mode: str | None = None,
    custom_labels: dict[str, str] | None = None,
    step_prefix: str = "step",
) -> str:
    key = str(meta.get("path") or "")
    custom = str((custom_labels or {}).get(key, "")).strip()
    label = custom or ram_base_label(meta, label_mode)
    prefix = ram_step_prefix(meta.get("step"), step_prefix)
    return f"{prefix} · {label}" if prefix and label else prefix or label


def ram_base_label(meta: dict, label_mode: str | None = None) -> str:
    label = meta.get("short_label") or meta.get("label") or meta.get("run") or "RAM log"
    label = re.sub(r"^(?:Step\s*)?\d+\s*·\s*", "", str(label), flags=re.IGNORECASE)
    return display_search_label(label, label_mode)


def ram_step_prefix(step: str | None, style: str | None = "step") -> str:
    if style == "none":
        return ""
    match = re.search(r"step\s*([1-5])", str(step or ""), flags=re.IGNORECASE)
    if not match:
        return "Cal" if str(step) == "make_cal_lib" else ""
    number = match.group(1)
    return number if style == "number" else f"Step {number}"


def style_ram_legend(fig: go.Figure) -> go.Figure:
    if fig.layout.showlegend is False:
        fig.update_layout(margin=dict(l=65, r=35, t=80, b=90))
        fig.update_xaxes(automargin=True, title_standoff=18)
        fig.update_yaxes(automargin=True, title_standoff=14)
        return fig

    labels = [trace.name for trace in fig.data if getattr(trace, "showlegend", None) is not False and trace.name]
    n_labels = max(len(set(labels)), 1)
    legend_rows = min(5, (n_labels + 1) // 2)
    bottom_margin = max(160, 110 + 30 * legend_rows)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.26,
            xanchor="left",
            x=0,
            font=dict(size=10),
            title=None,
            tracegroupgap=6,
        ),
        margin=dict(l=65, r=35, t=80, b=bottom_margin),
    )
    fig.update_xaxes(automargin=True, title_standoff=18)
    fig.update_yaxes(automargin=True, title_standoff=14)
    return fig


def selected_items_plot(selected_df: pd.DataFrame, title: str = "Selected items") -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select an intersection or set to inspect its items.")
    data = selected_df.copy()
    if "protein" not in data.columns:
        data["protein"] = "unknown"
    if "intensity" not in data.columns:
        data["intensity"] = pd.NA
    if "score" not in data.columns:
        data["score"] = pd.NA
    data["intensity"] = pd.to_numeric(data["intensity"], errors="coerce")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    protein_level = "evidence_count" in data.columns and "peptide_start" not in data.columns
    if protein_level:
        data["evidence_count"] = pd.to_numeric(data["evidence_count"], errors="coerce").fillna(0)
    grouped = (
        data.assign(protein=data["protein"].replace("", "unknown"))
        .groupby("protein", as_index=False)
        .agg(
            n_items=("item_id", "count"),
            evidence_count=("evidence_count", "sum") if protein_level else ("item_id", "count"),
            median_intensity=("intensity", "median"),
            mean_score=("score", "mean"),
        )
        .sort_values("evidence_count" if protein_level else "n_items", ascending=False)
        .head(30)
    )
    y_col = "evidence_count" if protein_level else "n_items"
    y_title = "Evidence rows" if protein_level else "Selected items"
    fig = go.Figure(
        go.Bar(
            x=grouped["protein"],
            y=grouped[y_col],
            marker=dict(color="#111111"),
            customdata=grouped[["n_items", "median_intensity", "mean_score"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                + ("Evidence rows: %{y:,.0f}<br>Protein items: %{customdata[0]:,}<br>" if protein_level else "Items: %{y}<br>")
                + "Median intensity: %{customdata[1]:.2e}<br>Mean score: %{customdata[2]:.3g}<extra></extra>"
            ),
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title, xaxis_title="Protein", yaxis_title=y_title, height=SELECTION_OVERVIEW_HEIGHT)
    return style_axes(cap_followup_bar_width(fig))


def selected_specificity_plot(selected_df: pd.DataFrame, set_columns: list[str], label_mode: str | None = None) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select an intersection or set to inspect specificity composition.")
    return selected_category_composition_plot(
        selected_df,
        set_columns,
        ["specificity"],
        empty_message="No specificity column for this selection.",
        label_mode=label_mode,
    )


def selected_modification_plot(
    selected_df: pd.DataFrame,
    set_columns: list[str],
    evidence_filter: list[str] | str | None = None,
    label_mode: str | None = None,
) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select an intersection or set to inspect N-term modification composition.")
    modification_order = enabled_modification_order(evidence_filter)
    return selected_category_composition_plot(
        selected_df,
        set_columns,
        ["nterm_status", "nterm_modif"],
        empty_message="No N-term modification columns for this selection.",
        category_mapper=nterm_site_modification_class,
        category_order=modification_order,
        allowed_categories=set(modification_order),
        label_mode=label_mode,
    )


def selected_termini_temple_plot(
    selected_df: pd.DataFrame,
    set_columns: list[str],
    evidence_filter: list[str] | str | None = None,
    label_mode: str | None = None,
) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select an intersection or set to inspect termini and modification composition.")

    specificity_col = first_existing(selected_df, ["specificity"])
    modification_col = first_existing(selected_df, ["nterm_status", "nterm_modif"])
    if not specificity_col or not modification_col:
        return empty_figure("No termini specificity and N-term modification columns for this selection.")

    groups = selection_groups(selected_df, set_columns, label_mode)
    specificity_order = ["tryptic", "semi_Nterm", "semi_Cterm"]
    modification_order = enabled_modification_order(evidence_filter)
    specificity_colors = {
        "tryptic": "#2F6F9F",
        "semi_Nterm": "#C44536",
        "semi_Cterm": "#3F8F64",
    }
    modification_colors = {
        "acetyl": "#F28E2B",
        "free": "#9CA3AF",
        "dimethyl": "#6B5B95",
    }

    class_data = []
    for set_name, subset in groups:
        data = subset.copy()
        data["_specificity_group"] = data[specificity_col].map(normalize_specificity_label)
        total = len(data)
        if total == 0:
            continue
        specificity_counts = data["_specificity_group"].value_counts().to_dict()
        specificity_items = []
        for specificity in specificity_order:
            specificity_df = data.loc[data["_specificity_group"] == specificity]
            if specificity_df.empty:
                continue
            mod_counts = nterm_site_modification_counts(specificity_df[modification_col], modification_order)
            specificity_items.append(
                {
                    "name": specificity,
                    "count": int(specificity_counts.get(specificity, 0)),
                    "mod_counts": {mod: int(mod_counts.get(mod, 0)) for mod in modification_order},
                }
            )
        if specificity_items:
            class_data.append({"name": str(set_name), "total": total, "specificities": specificity_items})

    if not class_data:
        return empty_figure("No termini composition values for this selection.")

    fig = go.Figure()
    class_gap = 0.92
    bar_gap = 0.18
    bar_width = 0.22
    master_bar_width = 0.28
    master_width = len(specificity_order) * master_bar_width + max(len(specificity_order) - 1, 0) * bar_gap
    master_y0 = 118.0
    master_y1 = 143.0
    lower_scale = 100.0
    x_tick_values = []
    x_tick_labels = []
    x_max = 0.0
    shown_modifications = set()
    shown_specificities = set()

    for class_index, cls in enumerate(class_data):
        group_x0 = x_max + (class_gap if class_index else 0)
        present_specificities = [item for item in cls["specificities"] if item["count"] > 0]
        if not present_specificities:
            continue
        group_width = master_width
        master_x0 = group_x0
        master_x1 = group_x0 + group_width
        cursor = master_x0
        master_segments = []
        for item in present_specificities:
            width = group_width * item["count"] / cls["total"] if cls["total"] else 0
            if width <= 0:
                continue
            segment_x0 = cursor
            segment_x1 = cursor + width
            master_segments.append((item, segment_x0, segment_x1))
            show_specificity_legend = item["name"] not in shown_specificities
            fig.add_trace(
                go.Bar(
                    x=[width],
                    y=[(master_y0 + master_y1) / 2],
                    base=[cursor],
                    width=[master_y1 - master_y0],
                    orientation="h",
                    marker={"color": specificity_colors[item["name"]], "line": {"color": "white", "width": 0.8}},
                    name=f"Master: {short_specificity_label(item['name'])}",
                    legendgroup=f"specificity:{item['name']}",
                    legendgrouptitle={"text": "Specificity"} if show_specificity_legend and not shown_specificities else None,
                    legendrank=0,
                    showlegend=show_specificity_legend,
                    customdata=[[cls["name"], item["name"], item["count"], item["count"] / cls["total"]]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Termini class: %{customdata[1]}<br>"
                        "Items: %{customdata[2]:,}<br>"
                        "Share: %{customdata[3]:.1%}<extra></extra>"
                    ),
                )
            )
            shown_specificities.add(item["name"])
            cursor += width

        fig.add_annotation(
            text=f"<b>{cls['name']}</b>",
            x=(master_x0 + master_x1) / 2,
            y=master_y1 + 6,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font={"size": 12, "color": "#111827"},
        )

        desired_centers = [(segment_x0 + segment_x1) / 2 for _item, segment_x0, segment_x1 in master_segments]
        lower_centers = bounded_separated_positions(
            desired_centers,
            bar_width + bar_gap,
            master_x0 + bar_width / 2,
            master_x1 - bar_width / 2,
        )

        for (item, segment_x0, segment_x1), x in zip(master_segments, lower_centers):
            base = 0.0
            specificity_total = sum(item["mod_counts"].values())
            for modification in modification_order:
                count = item["mod_counts"].get(modification, 0.0)
                if count <= 0 or specificity_total == 0:
                    continue
                show_modification_legend = modification not in shown_modifications
                height = lower_scale * count / specificity_total
                fig.add_trace(
                    go.Bar(
                        x=[x],
                        y=[height],
                        base=[base],
                        width=[bar_width],
                        marker={"color": modification_colors[modification], "line": {"color": "white", "width": 0.8}},
                        name=f"Mod: {modification}",
                        legendgroup=f"mod:{modification}",
                        legendgrouptitle={"text": "N-term modification"} if show_modification_legend and not shown_modifications else None,
                        legendrank=100,
                        showlegend=show_modification_legend,
                        customdata=[[cls["name"], item["name"], count, specificity_total, count / specificity_total]],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Termini class: %{customdata[1]}<br>"
                            f"Modification: {modification}<br>"
                            "Items: %{customdata[2]:,}<br>"
                            "Share within class: %{customdata[4]:.1%}<extra></extra>"
                        ),
                    )
                )
                shown_modifications.add(modification)
                base += height
            x_tick_values.append(x)
            x_tick_labels.append(item["name"])

            lower_x0 = x - bar_width / 2
            lower_x1 = x + bar_width / 2
            connector_line = {"color": "rgba(100, 116, 139, 0.55)", "dash": "dot", "width": 1.2}
            for x0, x1 in [(segment_x0, lower_x0), (segment_x1, lower_x1)]:
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[master_y0, lower_scale],
                        mode="lines",
                        line=connector_line,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        lower_x1 = (lower_centers[-1] + bar_width / 2) if lower_centers else master_x1
        x_max = max(master_x1, lower_x1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={"text": "Termini specificity x N-term modification", "font": {"size": 18}},
        height=max(TEMPLE_PLOT_HEIGHT, 300 + 6 * len(class_data)),
        barmode="overlay",
        bargap=0,
        xaxis_title="Class",
        yaxis_title="Mod. share",
        legend_title_text="",
        legend={
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
            "traceorder": "normal",
            "tracegroupgap": 8,
            "y": 0.86,
            "yanchor": "top",
        },
        font={"size": 12},
        margin=dict(l=72, r=148, t=58, b=66),
        plot_bgcolor="white",
        paper_bgcolor="white",
        meta={
            "plot_type": "termini_temple",
            "temple_model": {
                "classes": class_data,
                "specificity_order": specificity_order,
                "modification_order": modification_order,
                "specificity_colors": specificity_colors,
                "modification_colors": modification_colors,
                "class_gap": class_gap,
                "bar_gap": bar_gap,
                "bar_width": bar_width,
                "master_bar_width": master_bar_width,
                "master_width": master_width,
                "master_y0": master_y0,
                "master_y1": master_y1,
                "lower_scale": lower_scale,
            },
        },
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_tick_values,
        ticktext=[short_specificity_label(label) for label in x_tick_labels],
        tickangle=30,
        range=[-0.18, x_max + 0.24],
        showline=False,
        gridcolor="rgba(0,0,0,0)",
        zeroline=False,
    )
    fig.update_yaxes(
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["0%", "25%", "50%", "75%", "100%"],
        range=[0, master_y1 + 14],
        showline=False,
        gridcolor="#E8E8E8",
        zeroline=False,
    )
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=0,
        yref="y",
        y0=0,
        y1=lower_scale,
        line={"color": "#111111", "width": 1},
    )
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=0.018,
        yref="y",
        y0=lower_scale,
        y1=lower_scale,
        line={"color": "#111111", "width": 1},
    )
    fig.add_annotation(
        text="Specificity",
        x=-0.06,
        y=(master_y0 + master_y1) / 2,
        xref="paper",
        showarrow=False,
        textangle=-90,
        xanchor="center",
        yanchor="middle",
        font={"size": 14, "color": "#111827"},
    )
    return fig


def separated_positions(desired_positions: list[float], min_distance: float, min_center: float) -> list[float]:
    if len(desired_positions) <= 1:
        return list(desired_positions)

    blocks = []
    for index, value in enumerate(desired_positions):
        blocks.append({"start": index, "end": index, "weight": 1, "mean": value - index * min_distance})
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            last = blocks.pop()
            previous = blocks[-1]
            weight = previous["weight"] + last["weight"]
            previous["mean"] = (previous["mean"] * previous["weight"] + last["mean"] * last["weight"]) / weight
            previous["weight"] = weight
            previous["end"] = last["end"]

    positions = [0.0] * len(desired_positions)
    for block in blocks:
        for index in range(block["start"], block["end"] + 1):
            positions[index] = block["mean"] + index * min_distance

    if positions[0] < min_center:
        offset = min_center - positions[0]
        positions = [position + offset for position in positions]

    return positions


def bounded_separated_positions(desired_positions: list[float], min_distance: float, min_center: float, max_center: float) -> list[float]:
    if len(desired_positions) <= 1:
        if not desired_positions:
            return []
        return [min(max(desired_positions[0], min_center), max_center)]

    available = max_center - min_center
    required = (len(desired_positions) - 1) * min_distance
    if available + 1e-9 < required:
        return separated_positions(desired_positions, min_distance, min_center)

    positions = separated_positions(desired_positions, min_distance, min_center)
    if positions[-1] <= max_center + 1e-9:
        return positions

    span = positions[-1] - positions[0]
    if span <= available + 1e-9:
        shift = min(positions[-1] - max_center, positions[0] - min_center)
        return [position - shift for position in positions]

    desired_midpoint = (desired_positions[0] + desired_positions[-1]) / 2
    start = desired_midpoint - required / 2
    start = min(max(start, min_center), max_center - required)
    return [start + index * min_distance for index in range(len(desired_positions))]


def selected_decoy_fraction_plot(
    selected_df: pd.DataFrame,
    set_columns: list[str],
    qvalue_threshold: float | None = None,
    label_mode: str | None = None,
) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select an intersection or set to inspect ENT_ decoys.")
    if "is_decoy" not in selected_df.columns:
        return empty_figure("ENT_ decoys are excluded for this view.")

    data = selected_df.copy()
    data["is_decoy"] = data["is_decoy"].fillna(False).astype(bool)
    rows = []
    for set_name in set_columns:
        if set_name not in data.columns:
            continue
        present = data[set_name].fillna(False).astype(bool)
        total = int(present.sum())
        if total == 0:
            continue
        decoys = int((present & data["is_decoy"]).sum())
        rows.append(
            {
                "search": display_search_label(set_name, label_mode),
                "target_items": total - decoys,
                "decoy_items": decoys,
                "total_items": total,
                "decoy_fraction": decoys / total,
            }
        )

    if not rows:
        total = len(data)
        decoys = int(data["is_decoy"].sum())
        rows = [
            {
                "search": "selection",
                "target_items": total - decoys,
                "decoy_items": decoys,
                "total_items": total,
                "decoy_fraction": decoys / total if total else 0,
            }
        ]

    plot_df = pd.DataFrame(rows)
    fig = go.Figure(
        go.Bar(
            x=plot_df["search"],
            y=plot_df["decoy_fraction"],
            marker=dict(color="#d62728"),
            customdata=plot_df[["decoy_items", "target_items", "total_items"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "ENT_ decoys: %{customdata[0]:,}<br>"
                "Targets: %{customdata[1]:,}<br>"
                "Total: %{customdata[2]:,}<br>"
                "Decoy fraction: %{y:.2%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"ENT_ decoy fraction{f' at DIA-NN Q.Value <= {qvalue_threshold:g}' if qvalue_threshold is not None else ''}",
        xaxis_title="Search",
        yaxis_title="ENT_ decoys / selected items",
        height=SELECTION_PLOT_HEIGHT,
    )
    fig.update_yaxes(tickformat=".2%", range=[0, max(0.01, float(plot_df["decoy_fraction"].max()) * 1.15)])
    return style_axes(cap_followup_bar_width(fig))


def selected_decoy_temple_plot(selected_df: pd.DataFrame) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select an intersection or set to inspect ENT_ decoys.")
    if "is_decoy" not in selected_df.columns:
        return empty_figure("ENT_ decoys are excluded for this view.")

    modification_col = first_existing(selected_df, ["nterm_status", "nterm_modif"])
    if not modification_col:
        return empty_figure("No N-term modification column for ENT_ decoy Temple plot.")

    data = selected_df.copy()
    data["is_decoy"] = data["is_decoy"].fillna(False).astype(bool)
    modification_order = ["acetyl", "dimethyl"]
    modification_colors = {
        "acetyl": "#F28E2B",
        "dimethyl": "#6B5B95",
    }
    class_specs = [
        ("Target", False, TARGET_COLOR),
        ("ENT_ decoy", True, ENTRAPMENT_COLOR),
    ]
    class_data = []
    for label, is_decoy, color in class_specs:
        subset = data.loc[data["is_decoy"] == is_decoy]
        if subset.empty:
            continue
        mod_counts = nterm_site_modification_counts(subset[modification_col], modification_order)
        class_data.append(
            {
                "name": label,
                "color": color,
                "total": len(subset),
                "mod_counts": {mod: int(mod_counts.get(mod, 0)) for mod in modification_order},
            }
        )

    if not class_data:
        return empty_figure("No target or ENT_ decoy items in this selection.")

    fig = go.Figure()
    class_gap = 0.52
    bar_gap = 0.22
    bar_width = 0.22
    master_y0 = 118.0
    master_y1 = 143.0
    lower_scale = 100.0
    group_width = max(len(class_data) * bar_width + max(len(class_data) - 1, 0) * bar_gap, bar_width)
    master_x0 = 0.0
    master_x1 = group_width
    total_items = sum(item["total"] for item in class_data)
    cursor = master_x0
    master_segments = []
    shown_modifications = set()

    for item in class_data:
        width = group_width * item["total"] / total_items if total_items else 0
        if width <= 0:
            continue
        segment_x0 = cursor
        segment_x1 = cursor + width
        master_segments.append((item, segment_x0, segment_x1))
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[(master_y0 + master_y1) / 2],
                base=[cursor],
                width=[master_y1 - master_y0],
                orientation="h",
                marker={"color": item["color"], "line": {"color": "white", "width": 0.8}},
                name=item["name"],
                legendgroup=f"decoy:{item['name']}",
                legendgrouptitle={"text": "ENT_ decoy fraction"} if cursor == master_x0 else None,
                legendrank=0,
                showlegend=True,
                customdata=[[item["name"], item["total"], item["total"] / total_items if total_items else 0]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Items: %{customdata[1]:,}<br>"
                    "Share: %{customdata[2]:.1%}<extra></extra>"
                ),
            )
        )
        cursor += width

    desired_centers = [(segment_x0 + segment_x1) / 2 for _item, segment_x0, segment_x1 in master_segments]
    lower_centers = bounded_separated_positions(
        desired_centers,
        bar_width + bar_gap,
        master_x0 + bar_width / 2,
        master_x1 - bar_width / 2,
    )
    x_tick_values = []
    x_tick_labels = []
    for (item, segment_x0, segment_x1), x in zip(master_segments, lower_centers):
        base = 0.0
        modification_total = sum(item["mod_counts"].values())
        for modification in modification_order:
            count = item["mod_counts"].get(modification, 0)
            if count <= 0 or modification_total == 0:
                continue
            show_mod_legend = modification not in shown_modifications
            height = lower_scale * count / modification_total
            fig.add_trace(
                go.Bar(
                    x=[x],
                    y=[height],
                    base=[base],
                    width=[bar_width],
                    marker={"color": modification_colors[modification], "line": {"color": "white", "width": 0.8}},
                    name=modification,
                    legendgroup=f"mod:{modification}",
                    legendgrouptitle={"text": "N-term modification"} if show_mod_legend and not shown_modifications else None,
                    legendrank=100,
                    showlegend=show_mod_legend,
                    customdata=[[item["name"], modification, count, modification_total, count / modification_total]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Modification: %{customdata[1]}<br>"
                        "Items: %{customdata[2]:,}<br>"
                        "Share within shown modifications: %{customdata[4]:.1%}<extra></extra>"
                    ),
                )
            )
            shown_modifications.add(modification)
            base += height
        lower_x0 = x - bar_width / 2
        lower_x1 = x + bar_width / 2
        for x0, x1 in [(segment_x0, lower_x0), (segment_x1, lower_x1)]:
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[master_y0, lower_scale],
                    mode="lines",
                    line={"color": "rgba(100, 116, 139, 0.55)", "dash": "dot", "width": 1.2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        x_tick_values.append(x)
        x_tick_labels.append(item["name"])

    x_max = max(master_x1, (lower_centers[-1] + bar_width / 2) if lower_centers else master_x1)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={"text": "ENT_ decoy fraction x N-term modification", "font": {"size": 18}},
        height=TEMPLE_PLOT_HEIGHT,
        barmode="overlay",
        bargap=0,
        xaxis_title="Group",
        yaxis_title="Mod. share",
        legend_title_text="",
        legend={"itemclick": "toggle", "itemdoubleclick": "toggleothers", "traceorder": "normal", "tracegroupgap": 8, "y": 0.86, "yanchor": "top"},
        font={"size": 12},
        margin=dict(l=72, r=148, t=58, b=66),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_tick_values,
        ticktext=x_tick_labels,
        tickangle=0,
        range=[-0.18, x_max + 0.24],
        showline=False,
        gridcolor="rgba(0,0,0,0)",
        zeroline=False,
    )
    fig.update_yaxes(
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["0%", "25%", "50%", "75%", "100%"],
        range=[0, master_y1 + 14],
        showline=False,
        gridcolor="#E8E8E8",
        zeroline=False,
    )
    fig.add_shape(type="line", xref="paper", x0=0, x1=0, yref="y", y0=0, y1=lower_scale, line={"color": "#111111", "width": 1})
    fig.add_shape(type="line", xref="paper", x0=0, x1=0.018, yref="y", y0=lower_scale, y1=lower_scale, line={"color": "#111111", "width": 1})
    fig.add_annotation(
        text="Decoy status",
        x=-0.06,
        y=(master_y0 + master_y1) / 2,
        xref="paper",
        showarrow=False,
        textangle=-90,
        xanchor="center",
        yanchor="middle",
        font={"size": 14, "color": "#111827"},
    )
    return fig


def selection_groups(selected_df: pd.DataFrame, set_columns: list[str], label_mode: str | None = None) -> list[tuple[str, pd.DataFrame]]:
    groups = []
    label_map = display_search_label_map(set_columns, label_mode)
    for set_name in set_columns:
        if set_name not in selected_df.columns:
            continue
        subset = selected_df.loc[selected_df[set_name].fillna(False).astype(bool)]
        if not subset.empty:
            groups.append((label_map.get(set_name, set_name), subset))
    if groups:
        return groups
    return [("selection", selected_df)]


def normalize_specificity_label(value) -> str:
    label = "NA" if pd.isna(value) else str(value).strip()
    aliases = {
        "specific": "tryptic",
        "tryptic": "tryptic",
        "semi_nterm": "semi_Nterm",
        "semi-nterm": "semi_Nterm",
        "semi_Nterm": "semi_Nterm",
        "semi_cterm": "semi_Cterm",
        "semi-cterm": "semi_Cterm",
        "semi_Cterm": "semi_Cterm",
    }
    return aliases.get(label, aliases.get(label.lower(), label or "NA"))


def short_specificity_label(value: str) -> str:
    labels = {
        "tryptic": "Tryptic",
        "semi_Nterm": "Semi-N",
        "semi_Cterm": "Semi-C",
    }
    return labels.get(str(value), str(value))


def normalize_nterm_modification_label(value) -> str:
    label = "NA" if pd.isna(value) else str(value).strip()
    lowered = label.lower()
    if "dimethyl" in lowered or "unimod:36" in lowered or "unimod:510" in lowered:
        return "dimethyl"
    if "acetyl" in lowered or "unimod:1" in lowered:
        return "acetyl"
    if lowered in {"free", "free nh3", "unmodified", "none", "no_mod", "no mod", "n", "", "na", "nan"}:
        return "free"
    return label or "free"


def nterm_modification_labels(value) -> list[str]:
    if pd.isna(value):
        return ["free"]
    text = str(value).strip()
    if not text:
        return ["free"]
    parts = re.split(r"\s*(?:;|,|\+|\|)\s*", text)
    labels = []
    for part in parts:
        if not part:
            continue
        label = normalize_nterm_modification_label(part)
        if label and label not in labels:
            labels.append(label)
    return labels or [normalize_nterm_modification_label(text)]


def nterm_site_modification_class(value) -> str:
    labels = set(nterm_modification_labels(value))
    non_free_labels = {label for label in labels if label != "free"}
    if len(non_free_labels) == 1:
        return next(iter(non_free_labels))
    if not non_free_labels:
        return "free"
    return "other"


def enabled_modification_order(evidence_filter: list[str] | str | None = None) -> list[str]:
    if isinstance(evidence_filter, str):
        selected = {evidence_filter}
    else:
        selected = set(evidence_filter or [])
    order = ["acetyl", "dimethyl"]
    if "free" in selected:
        order.insert(1, "free")
    return order


def nterm_site_modification_counts(values, allowed_labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in allowed_labels}
    for value in values:
        label = nterm_site_modification_class(value)
        if label in counts:
            counts[label] += 1
    return counts


def specificity_groups(selected_df: pd.DataFrame, specificity_col: str) -> list[tuple[str, pd.DataFrame]]:
    label_order = ["tryptic", "semi_Nterm", "semi_Cterm"]
    data = selected_df.copy()
    data["_specificity_group"] = data[specificity_col].map(normalize_specificity_label)
    ordered = []
    for label in label_order:
        subset = data.loc[data["_specificity_group"] == label]
        if not subset.empty:
            ordered.append((label, subset))
    for label in sorted(set(data["_specificity_group"]) - set(label_order)):
        subset = data.loc[data["_specificity_group"] == label]
        if not subset.empty:
            ordered.append((str(label), subset))
    return ordered


def ordered_value_counts(values: pd.Series, category_order: list[str] | None = None) -> pd.Series:
    counts = values.fillna("NA").replace("", "NA").value_counts()
    if not category_order:
        return counts
    ordered_index = [category for category in category_order if category in counts.index]
    ordered_index.extend([category for category in counts.index if category not in ordered_index])
    return counts.reindex(ordered_index)


def selected_category_composition_plot(
    selected_df: pd.DataFrame,
    set_columns: list[str],
    category_candidates: list[str],
    *,
    empty_message: str,
    category_mapper=None,
    category_order: list[str] | None = None,
    allowed_categories: set[str] | None = None,
    label_mode: str | None = None,
) -> go.Figure:
    category_col = first_existing(selected_df, category_candidates)
    if not category_col:
        return empty_figure(empty_message)

    rows = []
    label_map = display_search_label_map(set_columns, label_mode)
    for set_name in set_columns:
        if set_name not in selected_df.columns:
            continue
        subset = selected_df.loc[selected_df[set_name].fillna(False).astype(bool)]
        if subset.empty:
            continue
        values = subset[category_col].map(category_mapper) if category_mapper else subset[category_col].fillna("NA").replace("", "NA")
        if allowed_categories is not None:
            values = values[values.astype(str).isin(allowed_categories)]
        counts = ordered_value_counts(values, category_order).head(12)
        total = counts.sum()
        for category, count in counts.items():
            rows.append(
                {
                    "search": label_map.get(set_name, set_name),
                    "category": str(category),
                    "count": int(count),
                    "fraction": 100 * int(count) / total if total else 0,
                }
            )
    if not rows:
        values = selected_df[category_col].map(category_mapper) if category_mapper else selected_df[category_col].fillna("NA").replace("", "NA")
        if allowed_categories is not None:
            values = values[values.astype(str).isin(allowed_categories)]
        counts = ordered_value_counts(values, category_order).head(12)
        total = counts.sum()
        if total == 0:
            return empty_figure(empty_message)
        rows = [
            {"search": "selection", "category": str(category), "count": int(count), "fraction": 100 * int(count) / total}
            for category, count in counts.items()
        ]

    fig = px.bar(
        pd.DataFrame(rows),
        x="search",
        y="fraction",
        color="category",
        text="count",
        barmode="stack",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title=f"{readable_column(category_col)} composition",
        xaxis_title="Search",
        yaxis_title="Selected entities (%)",
        height=SELECTION_PLOT_HEIGHT,
        legend_title=readable_column(category_col),
    )
    fig.update_yaxes(range=[0, 100])
    return style_axes(cap_followup_bar_width(fig))


def selected_category_comparison_plot(
    selections: list[tuple],
    category_candidates: list[str],
    *,
    title: str,
    empty_message: str,
    category_mapper=None,
    category_order: list[str] | None = None,
    allowed_categories: set[str] | None = None,
) -> go.Figure:
    rows = []
    for entry in selections:
        if len(entry) == 3:
            label, full_label, selected_df = entry
        else:
            label, selected_df = entry
            full_label = label
        if selected_df.empty:
            continue
        category_col = first_existing(selected_df, category_candidates)
        if not category_col:
            continue
        values = selected_df[category_col].map(category_mapper) if category_mapper else selected_df[category_col].fillna("NA").replace("", "NA")
        if allowed_categories is not None:
            values = values[values.astype(str).isin(allowed_categories)]
        counts = ordered_value_counts(values, category_order).head(12)
        total = counts.sum()
        if total == 0:
            continue
        for category, count in counts.items():
            rows.append(
                {
                    "selection": label,
                    "selection_full_label": full_label,
                    "category": str(category),
                    "count": int(count),
                    "fraction": 100 * int(count) / total,
                }
            )

    if not rows:
        return empty_figure(empty_message)

    fig = px.bar(
        pd.DataFrame(rows),
        x="selection",
        y="fraction",
        color="category",
        text="count",
        barmode="stack",
        custom_data=["selection_full_label", "category", "count"],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{customdata[0]}<br>"
            "Category: %{customdata[1]}<br>"
            "Count: %{customdata[2]:,}<br>"
            "Share: %{y:.1f}%<extra></extra>"
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Selection",
        yaxis_title="Selected entities (%)",
        height=TEMPLE_PLOT_HEIGHT,
        legend_title="Category",
        margin=dict(l=72, r=148, t=58, b=66),
    )
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(range=[0, 100])
    return style_axes(cap_followup_bar_width(fig))


def selected_rt_plot(selected_df: pd.DataFrame) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select entities to inspect RT prediction.")
    if {"RT", "Predicted_RT"}.issubset(selected_df.columns):
        cols = ["item_id", "RT", "Predicted_RT"]
        if "Abs_RT_Delta" in selected_df.columns:
            cols.append("Abs_RT_Delta")
        data = selected_df[cols].copy()
        data["RT"] = pd.to_numeric(data["RT"], errors="coerce")
        data["Predicted_RT"] = pd.to_numeric(data["Predicted_RT"], errors="coerce")
        if "Abs_RT_Delta" in data.columns:
            data["Abs_RT_Delta"] = pd.to_numeric(data["Abs_RT_Delta"], errors="coerce")
        else:
            data["Abs_RT_Delta"] = (data["RT"] - data["Predicted_RT"]).abs()
        data = data.dropna(subset=["RT", "Predicted_RT"])
        if not data.empty:
            r2 = rt_prediction_r2(data)
            title = "RT prediction for selection"
            if r2 is not None:
                title = f"{title} (R<sup>2</sup> = {r2:.3f})"
            fig = px.scatter(
                data.head(8000),
                x="Predicted_RT",
                y="RT",
                color="Abs_RT_Delta",
                color_continuous_scale="Greys",
                template=PLOTLY_TEMPLATE,
                hover_data=["item_id"],
            )
            fig.update_layout(title=title, xaxis_title="Predicted RT", yaxis_title="Observed RT", height=SELECTION_PLOT_HEIGHT)
            if r2 is not None:
                fig.add_annotation(
                    text=f"R<sup>2</sup> = {r2:.3f}",
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.78)",
                    bordercolor="#D1D5DB",
                    borderwidth=1,
                    font={"size": 13, "color": "#111827"},
                )
            return style_axes(fig)
    if "Abs_RT_Delta" in selected_df.columns:
        values = pd.to_numeric(selected_df["Abs_RT_Delta"], errors="coerce").dropna()
        if not values.empty:
            fig = px.histogram(values.to_frame("Abs_RT_Delta"), x="Abs_RT_Delta", nbins=40, template=PLOTLY_TEMPLATE)
            fig.update_layout(title="Absolute RT delta", xaxis_title="|RT - predicted RT|", yaxis_title="Entities", height=SELECTION_PLOT_HEIGHT)
            return style_axes(fig)
    return empty_figure("No RT / predicted RT columns for this selection.")


def selected_rt_modification_plot(selected_df: pd.DataFrame) -> go.Figure:
    required = {"RT", "Predicted_RT"}
    if selected_df.empty:
        return empty_figure("Select entities to inspect RT prediction by modification.")
    if not required.issubset(selected_df.columns):
        return empty_figure("No RT / predicted RT columns for this selection.")

    cols = ["item_id", "RT", "Predicted_RT"]
    cols.extend(col for col in ("modified_sequence", "nterm_modif", "nterm_status") if col in selected_df.columns)
    data = selected_df[cols].copy()
    data["RT"] = pd.to_numeric(data["RT"], errors="coerce")
    data["Predicted_RT"] = pd.to_numeric(data["Predicted_RT"], errors="coerce")
    data = data.dropna(subset=["RT", "Predicted_RT"])
    if data.empty:
        return empty_figure("No RT / predicted RT values for this selection.")

    data["Modification"] = data.apply(rt_modification_label, axis=1)
    modification_order = [label for label in ("Free", "Acetyl", "Dimethyl") if label in set(data["Modification"])]
    color_map = {
        "Free": "#9CA3AF",
        "Dimethyl": "#7C3AED",
        "Acetyl": "#2563EB",
    }
    fig = px.scatter(
        data.head(8000),
        x="Predicted_RT",
        y="RT",
        color="Modification",
        category_orders={"Modification": modification_order},
        color_discrete_map=color_map,
        template=PLOTLY_TEMPLATE,
        hover_data=["item_id", "Modification"],
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.72})
    fig.update_layout(
        title="RT prediction by N-terminal modification",
        xaxis_title="Predicted RT",
        yaxis_title="Observed RT",
        height=SELECTION_PLOT_HEIGHT,
        legend_title="N-terminus",
        legend={"itemdoubleclick": "toggleothers"},
    )
    r2_lines = []
    for label in ("Free", "Acetyl", "Dimethyl"):
        group = data.loc[data["Modification"] == label]
        r2 = rt_prediction_r2(group)
        if r2 is not None:
            r2_lines.append(f"{label}: R<sup>2</sup> = {r2:.3f} (n={len(group)})")
    if r2_lines:
        fig.add_annotation(
            text="<br>".join(r2_lines),
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            font={"size": 13, "color": "#111827"},
        )
    return style_axes(fig)


def selected_rt_modification_residual_plot(selected_df: pd.DataFrame) -> go.Figure:
    required = {"RT", "Predicted_RT"}
    if selected_df.empty:
        return empty_figure("Select entities to inspect RT residuals by modification.")
    if not required.issubset(selected_df.columns):
        return empty_figure("No RT / predicted RT columns for this selection.")

    cols = ["item_id", "RT", "Predicted_RT"]
    cols.extend(col for col in ("modified_sequence", "nterm_modif", "nterm_status") if col in selected_df.columns)
    data = selected_df[cols].copy()
    data["RT"] = pd.to_numeric(data["RT"], errors="coerce")
    data["Predicted_RT"] = pd.to_numeric(data["Predicted_RT"], errors="coerce")
    data = data.dropna(subset=["RT", "Predicted_RT"])
    if data.empty:
        return empty_figure("No RT / predicted RT values for this selection.")

    data["RT residual"] = data["RT"] - data["Predicted_RT"]
    data["Modification"] = data.apply(rt_modification_label, axis=1)
    order = [label for label in ("Free", "Acetyl", "Dimethyl") if label in set(data["Modification"])]

    fig = px.box(
        data.head(8000),
        x="Modification",
        y="RT residual",
        color="Modification",
        points="outliers",
        category_orders={"Modification": order},
        color_discrete_map={
            "Free": "#9CA3AF",
            "Dimethyl": "#7C3AED",
            "Acetyl": "#2563EB",
        },
        hover_data=["item_id"],
        template=PLOTLY_TEMPLATE,
    )
    fig.add_hline(y=0, line_color="#374151", line_dash="dash", line_width=1)
    fig.update_traces(boxmean=True, marker={"opacity": 0.72})
    fig.update_layout(
        title="RT prediction residuals by N-terminal modification",
        xaxis_title="N-terminus",
        yaxis_title="Observed RT - predicted RT",
        height=SELECTION_PLOT_HEIGHT,
        showlegend=True,
        legend_title="N-terminus",
        legend={"itemdoubleclick": "toggleothers"},
    )
    fig.update_xaxes(tickangle=25)
    return style_axes(fig)


def rt_modification_label(row: pd.Series) -> str:
    sequence = str(row.get("modified_sequence") or "").strip()
    leading_mods = re.match(r"^(?:\([^()]*\))+", sequence)
    nterm_text = leading_mods.group(0).lower() if leading_mods else ""

    # Only annotations before the first amino acid describe the peptide
    # N-terminus. Internal oxidation, carbamidomethylation, acetylation, etc.
    # deliberately do not affect this three-class grouping.
    if "unimod:36" in nterm_text or "unimod:510" in nterm_text or "dimethyl" in nterm_text:
        return "Dimethyl"
    if "unimod:1" in nterm_text or "acetyl" in nterm_text:
        return "Acetyl"

    # Prefer the explicit N-terminal annotations when the modified sequence
    # does not carry a leading DIA-NN modification token.
    explicit_nterm = " ".join(
        str(row.get(column) or "").strip().lower()
        for column in ("nterm_modif", "nterm_status")
    )
    if "dimethyl" in explicit_nterm:
        return "Dimethyl"
    if "acetyl" in explicit_nterm:
        return "Acetyl"
    return "Free"


def rt_prediction_r2(data: pd.DataFrame) -> float | None:
    observed = pd.to_numeric(data["RT"], errors="coerce")
    predicted = pd.to_numeric(data["Predicted_RT"], errors="coerce")
    valid = pd.DataFrame({"observed": observed, "predicted": predicted}).dropna()
    if len(valid) < 2:
        return None
    ss_res = ((valid["observed"] - valid["predicted"]) ** 2).sum()
    ss_tot = ((valid["observed"] - valid["observed"].mean()) ** 2).sum()
    if ss_tot == 0:
        return None
    return float(1 - ss_res / ss_tot)


def selected_mass_error_plot(selected_df: pd.DataFrame) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select entities to inspect fragment and MS1 deltas.")
    cols = [
        ("Abs_Best_Fr_Mz_Delta", "Abs fragment m/z delta"),
        ("Abs_Ms1_Apex_Mz_Delta", "Abs MS1 apex m/z delta"),
        ("Best_Fr_Mz_Delta", "Fragment m/z delta"),
        ("Ms1_Apex_Mz_Delta", "MS1 apex m/z delta"),
    ]
    rows = []
    for col, label in cols:
        if col not in selected_df.columns:
            continue
        values = pd.to_numeric(selected_df[col], errors="coerce").dropna()
        rows.extend({"metric": label, "value": value} for value in values)
    if not rows:
        return empty_figure("No fragment or MS1 delta columns for this selection.")
    data = pd.DataFrame(rows)
    fig = px.box(data, x="metric", y="value", points="outliers", template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Mass-error deltas", xaxis_title="", yaxis_title="Delta", height=SELECTION_PLOT_HEIGHT)
    return style_axes(fig)


def selected_peptide_properties_plot(selected_df: pd.DataFrame) -> go.Figure:
    if selected_df.empty:
        return empty_figure("Select entities to inspect peptide properties.")
    has_length = "peptide_length" in selected_df.columns
    has_gravy = "gravy" in selected_df.columns
    if not has_length and not has_gravy:
        return empty_figure("No peptide length or GRAVY values for this selection.")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Length", "Hydrophobicity"))
    if has_length:
        values = pd.to_numeric(selected_df["peptide_length"], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=values, nbinsx=30, marker_color="#111111", name="Length"), row=1, col=1)
        add_distribution_summary(fig, values, row=1, col=1, value_label="aa", color="#111111")
    if has_gravy:
        values = pd.to_numeric(selected_df["gravy"], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=values, nbinsx=30, marker_color="#777777", name="GRAVY"), row=1, col=2)
        add_distribution_summary(fig, values, row=1, col=2, value_label="GRAVY", color="#555555")
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Peptide properties", height=SELECTION_PLOT_HEIGHT, showlegend=False)
    fig.update_xaxes(title_text="aa", row=1, col=1)
    fig.update_xaxes(title_text="GRAVY", row=1, col=2)
    fig.update_yaxes(title_text="Entities", row=1, col=1)
    return style_axes(fig)


AA_MASS = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}
AA_PKA_POS = {"K": 10.5, "R": 12.4, "H": 6.0}
AA_PKA_NEG = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
GRAVY_SCALE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


def add_ms_properties(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    sequence_col = "stripped_sequence" if "stripped_sequence" in data.columns else "sequence"
    if sequence_col not in data.columns:
        data["stripped_sequence"] = ""
        sequence_col = "stripped_sequence"
    data["sequence_for_properties"] = data[sequence_col].fillna("").astype(str).map(lambda seq: re.sub(r"[^A-Z]", "", seq))
    data["peptide_length"] = data["sequence_for_properties"].str.len()
    data["molecular_weight"] = data["sequence_for_properties"].map(molecular_weight)
    data["gravy"] = data["sequence_for_properties"].map(gravy_sequence)
    data["pI"] = data["sequence_for_properties"].map(isoelectric_point)
    data["c_terminal_residue"] = data["sequence_for_properties"].str[-1:].replace("", pd.NA)
    internal = data["sequence_for_properties"].map(lambda seq: seq[:-1] if seq else "")
    data["missed_cleavages"] = internal.map(lambda seq: sum(1 for aa in seq if aa in {"K", "R"}))
    data["internal_R_count"] = internal.str.count("R")
    data["internal_K_count"] = internal.str.count("K")
    data["aromatic_fraction"] = data["sequence_for_properties"].map(lambda seq: aa_fraction(seq, {"F", "W", "Y"}))
    data["acidic_fraction"] = data["sequence_for_properties"].map(lambda seq: aa_fraction(seq, {"D", "E"}))
    data["basic_fraction"] = data["sequence_for_properties"].map(lambda seq: aa_fraction(seq, {"K", "R", "H"}))
    return data


def ms_property_figure(data: pd.DataFrame, plot_id: str) -> go.Figure:
    if data.empty:
        return empty_figure("No peptide evidence rows for this property plot.")
    if plot_id == "peptide_length":
        return property_violin(data, "peptide_length", "Peptide length")
    if plot_id == "molecular_weight":
        return property_violin(data, "molecular_weight", "Molecular weight")
    if plot_id == "gravy":
        return property_violin(data, "gravy", "GRAVY / hydrophobicity")
    if plot_id == "pI":
        return property_violin(data, "pI", "Isoelectric point")
    if plot_id == "c_terminal_residue":
        return property_fraction_bar(data, "c_terminal_residue", "C-terminal residue", "c_terminal_residue")
    if plot_id == "missed_cleavages":
        return property_fraction_bar(data, "missed_cleavages", "Missed cleavages", "missed_cleavages")
    if plot_id == "internal_rk_counts":
        return internal_rk_counts_plot(data)
    if plot_id == "aromatic_fraction":
        return property_violin(data, "aromatic_fraction", "Aromatic fraction")
    if plot_id == "acidic_basic_fraction":
        return acidic_basic_fraction_plot(data)
    return empty_figure("Unknown property plot.")


def property_violin(data: pd.DataFrame, y_col: str, title: str) -> go.Figure:
    plot = data[["group", y_col]].copy()
    plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
    plot = plot.dropna(subset=[y_col])
    if plot.empty:
        return empty_figure(f"No values for {title}.")
    fig = px.violin(
        plot,
        x="group",
        y=y_col,
        color="group",
        box=True,
        points=False,
        color_discrete_map={"Selected": "#111111", "Background": "#CFCFCF"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title=title_with_group_counts(title, data), xaxis_title="", yaxis_title=y_col, height=SELECTION_PLOT_HEIGHT, showlegend=False)
    return style_axes(fig)


def property_fraction_bar(data: pd.DataFrame, col: str, title: str, x_title: str) -> go.Figure:
    plot = data[["group", col]].copy().dropna()
    if plot.empty:
        return empty_figure(f"No values for {title}.")
    counts = plot.groupby(["group", col], as_index=False).size().rename(columns={"size": "count"})
    totals = counts.groupby("group")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    counts[col] = counts[col].astype(str)
    fig = px.bar(
        counts,
        x=col,
        y="fraction",
        color="group",
        barmode="group",
        text="count",
        color_discrete_map={"Selected": "#111111", "Background": "#CFCFCF"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title=title_with_group_counts(title, data), xaxis_title=x_title, yaxis_title="Fraction within group", height=SELECTION_PLOT_HEIGHT)
    fig.update_yaxes(tickformat=".0%")
    return style_axes(cap_followup_bar_width(fig))


def internal_rk_counts_plot(data: pd.DataFrame) -> go.Figure:
    rows = []
    for feature, label in (("internal_R_count", "Internal R"), ("internal_K_count", "Internal K")):
        subset = data[["group", feature]].copy()
        subset[feature] = pd.to_numeric(subset[feature], errors="coerce")
        subset = subset.dropna()
        counts = subset.groupby(["group", feature], as_index=False).size().rename(columns={"size": "count", feature: "count_value"})
        if counts.empty:
            continue
        counts["feature"] = label
        totals = counts.groupby(["group", "feature"])["count"].transform("sum")
        counts["fraction"] = counts["count"] / totals
        rows.append(counts)
    if not rows:
        return empty_figure("No internal R/K counts available.")
    plot = pd.concat(rows, ignore_index=True)
    plot["count_value"] = plot["count_value"].astype(int).astype(str)
    fig = px.bar(
        plot,
        x="count_value",
        y="fraction",
        color="group",
        facet_col="feature",
        barmode="group",
        text="count",
        color_discrete_map={"Selected": "#111111", "Background": "#CFCFCF"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title=title_with_group_counts("Internal R/K counts", data), xaxis_title="Count", yaxis_title="Fraction within group", height=SELECTION_PLOT_HEIGHT)
    fig.update_yaxes(tickformat=".0%")
    return style_axes(cap_followup_bar_width(fig))


def acidic_basic_fraction_plot(data: pd.DataFrame) -> go.Figure:
    rows = []
    for col, label in (("acidic_fraction", "Acidic D/E"), ("basic_fraction", "Basic K/R/H")):
        subset = data[["group", col]].copy()
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna()
        if subset.empty:
            continue
        subset = subset.rename(columns={col: "fraction"})
        subset["feature"] = label
        rows.append(subset)
    if not rows:
        return empty_figure("No acidic/basic fraction values available.")
    plot = pd.concat(rows, ignore_index=True)
    fig = px.violin(
        plot,
        x="feature",
        y="fraction",
        color="group",
        box=True,
        points=False,
        color_discrete_map={"Selected": "#111111", "Background": "#CFCFCF"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title=title_with_group_counts("Acidic/basic fraction", data), xaxis_title="", yaxis_title="Fraction", height=SELECTION_PLOT_HEIGHT)
    return style_axes(fig)


def title_with_group_counts(title: str, data: pd.DataFrame) -> str:
    counts = data.groupby("group")["item_id"].count().to_dict() if "item_id" in data.columns else {}
    detail = ", ".join(f"{group} n={int(count)}" for group, count in counts.items())
    return f"{title} ({detail})" if detail else title


def molecular_weight(sequence: str) -> float | None:
    if not sequence:
        return None
    mass = 18.01056
    for aa in sequence:
        value = AA_MASS.get(aa)
        if value is None:
            return None
        mass += value
    return mass


def gravy_sequence(sequence: str) -> float | None:
    if not sequence:
        return None
    values = [GRAVY_SCALE.get(aa) for aa in sequence]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def aa_fraction(sequence: str, residues: set[str]) -> float | None:
    if not sequence:
        return None
    return sum(sequence.count(aa) for aa in residues) / len(sequence)


def isoelectric_point(sequence: str) -> float | None:
    if not sequence:
        return None
    low, high = 0.0, 14.0
    for _ in range(40):
        mid = (low + high) / 2
        if peptide_charge(sequence, mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def peptide_charge(sequence: str, ph: float) -> float:
    charge = 1 / (1 + 10 ** (ph - 9.69))
    charge -= 1 / (1 + 10 ** (2.34 - ph))
    for aa, pka in AA_PKA_POS.items():
        charge += sequence.count(aa) / (1 + 10 ** (ph - pka))
    for aa, pka in AA_PKA_NEG.items():
        charge -= sequence.count(aa) / (1 + 10 ** (pka - ph))
    return charge


def add_distribution_summary(
    fig: go.Figure,
    values: pd.Series,
    *,
    row: int,
    col: int,
    value_label: str,
    color: str,
) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return

    mean = float(values.mean())
    median = float(values.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1

    fig.add_vrect(
        x0=q1,
        x1=q3,
        fillcolor=color,
        opacity=0.10,
        line_width=0,
        row=row,
        col=col,
    )
    fig.add_vline(
        x=median,
        line_color=color,
        line_width=2.5,
        row=row,
        col=col,
    )
    fig.add_vline(
        x=mean,
        line_color=color,
        line_dash="dash",
        line_width=2,
        row=row,
        col=col,
    )

    value_fmt = ".1f" if value_label == "aa" else ".2f"
    iqr_fmt = ".1f" if value_label == "aa" else ".2f"
    xref = "x domain" if col == 1 else f"x{col} domain"
    yref = "y domain" if col == 1 else f"y{col} domain"
    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref=xref,
        yref=yref,
        xanchor="right",
        yanchor="top",
        align="right",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.82)",
        bordercolor="rgba(0,0,0,0.18)",
        borderwidth=1,
        font=dict(size=11, color="#243b5e"),
        text=(
            f"n={len(values)}<br>"
            f"median={median:{value_fmt}} {value_label}<br>"
            f"mean={mean:{value_fmt}} {value_label}<br>"
            f"IQR={iqr:{iqr_fmt}}"
        ),
    )


def style_axes(fig: go.Figure) -> go.Figure:
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=50, r=30, t=70, b=70))
    fig.update_xaxes(showline=True, linecolor="#111111", ticks="outside", gridcolor="#E8E8E8")
    fig.update_yaxes(showline=True, linecolor="#111111", ticks="outside", gridcolor="#E8E8E8", zeroline=False)
    return fig


def style_publication_axes(fig: go.Figure) -> go.Figure:
    style_axes(fig)
    fig.update_layout(
        font=dict(size=16, color="#111827"),
        title=dict(font=dict(size=22)),
        legend=dict(font=dict(size=14), title=dict(font=dict(size=15))),
        margin=dict(l=80, r=45, t=90, b=85),
    )
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=15), automargin=True)
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=15), automargin=True)
    for annotation in fig.layout.annotations or []:
        annotation.font = dict(size=15, color="#111827")
    return fig


def n_unique(df: pl.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return df.select(pl.col(col)).drop_nulls().unique().height


def first_existing(df: pd.DataFrame, columns: list[str]) -> str | None:
    for col in columns:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def readable_column(column: str) -> str:
    labels = {
        "nterm_modif": "N-term modification",
        "specificity": "Termini specificity",
        "nterm_residue": "N-terminal residue",
        "cterm_residue": "C-terminal residue",
    }
    return labels.get(column, column.replace("_", " "))
