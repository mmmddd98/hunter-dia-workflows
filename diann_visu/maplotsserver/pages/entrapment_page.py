from __future__ import annotations

from pathlib import Path

from dash import dcc, html

from data.figures import (
    empty_figure,
    publication_graph_config,
)
from pages.dataset_overview import plot_topbar


DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "entrapment_fdr.md"
ENTRAPMENT_LEVEL_OPTIONS = [
    {"label": "Precursor", "value": "precursor"},
    {"label": "Peptidoform", "value": "peptidoform"},
    {"label": "Peptide", "value": "peptide"},
    {"label": "Protein group", "value": "protein_group"},
]
NTERM_ENTRAPMENT_LEVEL_OPTIONS = [
    {"label": "Precursor", "value": "nterm_precursor"},
    {"label": "N-term peptide", "value": "nterm_peptide"},
    {"label": "N-term site", "value": "nterm_site"},
    {"label": "Protein", "value": "nterm_protein"},
]
ENTRAPMENT_AGGREGATION_OPTIONS = [
    {"label": "DIA-NN reported levels", "value": "diann"},
    {"label": "N-terminal evidence levels", "value": "nterm"},
]
QVALUE_THRESHOLD_OPTIONS = [
    {"label": "<= 0.001", "value": 0.001},
    {"label": "<= 0.005", "value": 0.005},
    {"label": "<= 0.01", "value": 0.01},
    {"label": "<= 0.02", "value": 0.02},
    {"label": "<= 0.05", "value": 0.05},
    {"label": "<= 0.10", "value": 0.10},
]
BIN_WIDTH_OPTIONS = [
    {"label": "0.0025", "value": 0.0025},
    {"label": "0.005", "value": 0.005},
    {"label": "0.01", "value": 0.01},
    {"label": "0.02", "value": 0.02},
]
MAX_QVALUE_OPTIONS = [
    {"label": "0.02", "value": 0.02},
    {"label": "0.05", "value": 0.05},
    {"label": "0.10", "value": 0.10},
    {"label": "All observed", "value": "__all__"},
]
FDR_DEFINITION_OPTIONS = [
    {"label": "ENT_ / target", "value": "rate"},
    {"label": "ENT_ / (target + ENT_)", "value": "fraction"},
]
DISPLAY_OPTIONS = [
    {"label": "Q.Value threshold reference", "value": "qvalue_reference"},
    {"label": "Program decoys", "value": "program_decoys"},
]
CURVE_GROUPING_OPTIONS = [
    {"label": "Split into panels", "value": "split"},
    {"label": "All lines in one plot", "value": "combined"},
]


def graph_download_buttons(graph_id: str, filename: str) -> html.Div:
    return html.Div(
        className="plot-download-row",
        children=[
            html.Button(
                "SVG",
                className="plot-download-button",
                **{
                    "data-plot-download": "svg",
                    "data-graph-id": graph_id,
                    "data-filename": filename,
                },
            ),
            html.Button(
                "PNG",
                className="plot-download-button",
                **{
                    "data-plot-download": "png",
                    "data-graph-id": graph_id,
                    "data-filename": filename,
                    "data-width": "2400",
                    "data-scale": "1",
                },
            ),
            html.Button(
                "HTML",
                className="plot-download-button",
                **{
                    "data-plot-download": "html",
                    "data-graph-id": graph_id,
                    "data-filename": filename,
                },
            ),
        ],
    )


def layout(dataset: dict, registry) -> html.Div:
    default_levels = ["nterm_peptide", "nterm_site"]
    runs = list(dataset.get("set_columns") or [])
    return html.Div(
        children=[
            plot_topbar(dataset, "entrapment"),
            html.H1("Entrapment FDR"),
            html.Div(
                className="search-list",
                children=[
                    html.Span("target vs ENT_ shuffled proteins", className="pill"),
                    html.Span("peptide evidence specificity", className="pill"),
                    html.Span("DIA-NN Q.Value", className="pill"),
                    dcc.Link("Calculation notes", href=f"/dataset/{dataset['id']}/entrapment-method", className="pill link-pill"),
                ],
            ),
            html.Div(
                className="toolbar-row entrapment-control-row",
                children=[
                    html.Div(
                        className="compact-control source-control",
                        children=[
                            html.Label("Aggregation"),
                            dcc.Dropdown(
                                id="entrapment-aggregation-dropdown",
                                options=ENTRAPMENT_AGGREGATION_OPTIONS,
                                value="nterm",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control entrapment-run-control",
                        children=[
                            html.Label("Runs"),
                            dcc.Dropdown(
                                id="entrapment-run-dropdown",
                                options=[{"label": run, "value": run} for run in runs],
                                value=runs,
                                multi=True,
                                searchable=True,
                            ),
                        ],
                    ),
                    html.Details(
                        className="entrapment-label-manager",
                        children=[
                            html.Summary("Edit run labels"),
                            html.Div(
                                className="entrapment-label-fields",
                                children=[
                                    html.Div(
                                        className="upset-label-row",
                                        children=[
                                            html.Span(run, className="upset-label-raw", title=run),
                                            dcc.Input(
                                                id={"type": "entrapment-custom-label", "run": run},
                                                value="",
                                                placeholder=run,
                                                type="text",
                                                debounce=True,
                                            ),
                                        ],
                                    )
                                    for run in runs
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control source-control",
                        children=[
                            html.Label("Level(s)"),
                            dcc.Dropdown(
                                id="entrapment-level-dropdown",
                                options=NTERM_ENTRAPMENT_LEVEL_OPTIONS,
                                value=default_levels,
                                multi=True,
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control source-control",
                        children=[
                            html.Label("Q.Value threshold"),
                            dcc.Dropdown(
                                id="entrapment-threshold-dropdown",
                                options=QVALUE_THRESHOLD_OPTIONS,
                                value=0.01,
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control source-control",
                        children=[
                            html.Label("Curve layout"),
                            dcc.Dropdown(
                                id="entrapment-curve-grouping-dropdown",
                                options=CURVE_GROUPING_OPTIONS,
                                value="split",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        id="entrapment-level-status",
                        className="status-line compact-status",
                    ),
                ],
            ),
            graph_download_buttons("entrapment-summary-graph", "entrapment_fdr_summary"),
            dcc.Graph(
                id="entrapment-summary-graph",
                figure=empty_figure("Loading entrapment summary..."),
                config=publication_graph_config("entrapment_fdr_summary"),
            ),
            graph_download_buttons("entrapment-curve-graph", "entrapment_fdr_curve"),
            dcc.Graph(
                id="entrapment-curve-graph",
                figure=empty_figure("Loading entrapment curve..."),
                config=publication_graph_config("entrapment_fdr_curve"),
            ),
            graph_download_buttons("entrapment-modification-summary-graph", "entrapment_fdr_by_nterm_modification_summary"),
            dcc.Graph(
                id="entrapment-modification-summary-graph",
                figure=empty_figure("Loading N-term modification split..."),
                config=publication_graph_config("entrapment_fdr_by_nterm_modification_summary"),
            ),
            graph_download_buttons("entrapment-modification-curve-graph", "entrapment_fdr_by_nterm_modification_curve"),
            dcc.Graph(
                id="entrapment-modification-curve-graph",
                figure=empty_figure("Loading N-term modification curve..."),
                config=publication_graph_config("entrapment_fdr_by_nterm_modification_curve"),
            ),
        ],
    )


def method_layout(dataset: dict) -> html.Div:
    text = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else "Entrapment FDR documentation is missing."
    return html.Div(
        children=[
            plot_topbar(dataset, "entrapment"),
            html.H1("Entrapment FDR calculation"),
            html.Div(
                className="search-list",
                children=[
                    dcc.Link("Back to plots", href=f"/dataset/{dataset['id']}/entrapment", className="pill link-pill"),
                ],
            ),
            dcc.Markdown(text, className="method-doc"),
        ],
    )
