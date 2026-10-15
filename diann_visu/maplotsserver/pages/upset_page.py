from __future__ import annotations

from dash import dash_table, dcc, html

from data.datasets import ITEM_LEVEL_OPTIONS, TRIPLEMIX_ORGANISM_OPTIONS, UPSET_AGGREGATION_SOURCE_OPTIONS
from data.figures import empty_figure, publication_graph_config, selected_items_plot
from pages.dataset_overview import graph_download_buttons, plot_topbar


BASE_MULTI_DETAIL_OPTIONS = [
    {"label": "Specificity", "value": "specificity"},
    {"label": "N-term modification", "value": "modification"},
    {"label": "Termini Temple", "value": "termini_temple"},
    {"label": "RT", "value": "rt"},
    {"label": "Mass error", "value": "mass_error"},
    {"label": "Peptide properties", "value": "peptide_properties"},
    {"label": "MS property follow-up", "value": "ms_property_followup"},
    {"label": "Position & motif plots", "value": "position_motif"},
    {"label": "Selected items table", "value": "table"},
    {"label": "Overview", "value": "overview"},
]
DECOY_DETAIL_OPTION = {"label": "ENT_ decoy fraction", "value": "decoy_fraction"}
DECOY_TEMPLE_DETAIL_OPTION = {"label": "ENT_ decoy Temple", "value": "decoy_temple"}
MULTI_DETAIL_OPTIONS = [*BASE_MULTI_DETAIL_OPTIONS, DECOY_DETAIL_OPTION, DECOY_TEMPLE_DETAIL_OPTION]

DEFAULT_MULTI_DETAILS = ["specificity", "modification", "termini_temple", "rt", "mass_error", "peptide_properties", "table"]


def graph_config(filename: str, **overrides) -> dict:
    config = publication_graph_config(filename)
    config.update(overrides)
    return config


def layout(dataset: dict, source: str = "nterm_site") -> html.Div:
    available_searches = list(dataset.get("set_columns", []))
    standard_carafe = next(
        (
            search
            for search in available_searches
            if search.startswith("carafe/") and "carafe_standard_baseline_v1" in search
        ),
        None,
    )
    initial_searches = (
        ["baseline", standard_carafe]
        if "baseline" in available_searches and standard_carafe is not None
        else available_searches
    )
    topbar_controls = [
        html.Div(
            className="upset-filter-shell",
            children=[
                html.Details(
                    className="upset-filter-drawer",
                    open=False,
                    children=[
                        html.Summary(
                            className="upset-filter-summary",
                            children=[
                                html.Span("Filters & view", className="upset-filter-title"),
                                html.Div(id="upset-source-status", className="status-line compact-status upset-inline-status"),
                            ],
                        ),
                        html.Div(
                            className="upset-topbar-block upset-topbar-filter-row",
                            children=[
                                html.Div(
                                    id="upset-entrapment-control",
                                    className="compact-control entrapment-control",
                                    children=[
                                        html.Label("Entrapment"),
                                        dcc.Dropdown(
                                            id="upset-entrapment-dropdown",
                                            options=[
                                                {"label": "Exclude ENT_", "value": "exclude"},
                                                {"label": "Include ENT_", "value": "include"},
                                            ],
                                            value="exclude",
                                            clearable=False,
                                            searchable=False,
                                            className="entrapment-dropdown",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="upset-organism-control",
                                    className="compact-control evidence-control",
                                    style={} if dataset.get("organism") == "Mixed" else {"display": "none"},
                                    children=[
                                        html.Label("Organism"),
                                        dcc.Dropdown(
                                            id="upset-organism-dropdown",
                                            options=TRIPLEMIX_ORGANISM_OPTIONS,
                                            value="__all__",
                                            clearable=False,
                                            searchable=False,
                                            className="evidence-dropdown",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control source-control",
                                    children=[
                                        html.Label("Item level"),
                                        dcc.Dropdown(
                                            id="upset-source-dropdown",
                                            options=ITEM_LEVEL_OPTIONS,
                                            value=source,
                                            clearable=False,
                                            searchable=False,
                                            className="source-dropdown",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control source-control",
                                    children=[
                                        html.Label("Aggregation"),
                                        dcc.Dropdown(
                                            id="upset-aggregation-source-dropdown",
                                            options=UPSET_AGGREGATION_SOURCE_OPTIONS,
                                            value="materialized",
                                            clearable=False,
                                            searchable=False,
                                            className="source-dropdown",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control evidence-control",
                                    children=[
                                        html.Label("N-term evidence"),
                                        dcc.Checklist(
                                            id="upset-evidence-filter-dropdown",
                                            options=[
                                                {"label": "Dimethyl", "value": "dimethyl"},
                                                {"label": "Acetyl", "value": "acetyl"},
                                                {"label": "Free NH3", "value": "free"},
                                            ],
                                            value=["dimethyl", "acetyl"],
                                            inline=True,
                                            className="evidence-chip-selector",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="upset-fdr-control",
                                    className="compact-control evidence-control",
                                    children=[
                                        html.Label("DIA-NN Q.Value"),
                                        dcc.Dropdown(
                                            id="upset-fdr-dropdown",
                                            options=[],
                                            value=None,
                                            clearable=False,
                                            searchable=False,
                                            className="evidence-dropdown",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control number-control",
                                    children=[
                                        html.Label("Min"),
                                        dcc.Input(id="min-size-input", type="number", value=1, min=1, step=1),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control number-control",
                                    children=[
                                        html.Label("Top N"),
                                        dcc.Input(id="top-n-input", type="number", value=35, min=1, step=1),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control multi-mode-control",
                                    children=[
                                        html.Label("Multi intersections"),
                                        dcc.RadioItems(
                                            id="multi-intersection-mode",
                                            options=[
                                                {"label": "Off", "value": "off"},
                                                {"label": "On", "value": "on"},
                                            ],
                                            value="on",
                                            inline=True,
                                            className="view-toggle",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control view-control",
                                    children=[
                                        html.Label("View"),
                                        dcc.RadioItems(
                                            id="upset-layout-mode",
                                            options=[
                                                {"label": "Split", "value": "split"},
                                                {"label": "Selection focus", "value": "focus"},
                                                {"label": "Vertical", "value": "vertical"},
                                            ],
                                            value="split",
                                            inline=True,
                                            className="view-toggle",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="compact-control upset-search-control",
                                    children=[
                                        html.Label("Shown searches"),
                                            dcc.Dropdown(
                                                id="upset-search-filter-dropdown",
                                                options=[{"label": search, "value": search} for search in available_searches],
                                                value=initial_searches,
                                            multi=True,
                                            clearable=False,
                                            searchable=True,
                                            placeholder="Choose searches",
                                            className="upset-search-filter-dropdown",
                                        ),
                                    ],
                                ),
                                html.Details(
                                    className="upset-label-manager",
                                    children=[
                                        html.Summary("Labels"),
                                        html.Div(
                                            className="upset-label-manager-body",
                                            children=[
                                                html.Div(
                                                    className="upset-label-actions",
                                                    children=[
                                                        html.Button("💾", id="upset-label-save-button", n_clicks=0, className="icon-action-button save", title="Save labels"),
                                                        html.Button("Reset", id="upset-label-reset-button", n_clicks=0, className="upset-label-reset"),
                                                    ],
                                                ),
                                                html.Div(id="upset-label-fields", className="upset-label-fields"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Button("Clear", id="clear-selection-button", n_clicks=0, className="upset-filter-clear"),
            ],
        ),
    ]

    return html.Div(
        children=[
            dcc.Store(id="item-store"),
            dcc.Store(id="set-columns-store"),
            dcc.Store(id="selected-store"),
            dcc.Store(id="multi-selected-store"),
            dcc.Store(id="multi-selection-edit-store", data={}),
            dcc.Store(id="upset-label-map-store", data={}),
            dcc.Store(id="raw-item-store"),
            dcc.Store(id="available-set-columns-store", data=available_searches),
            dcc.Store(id="all-set-columns-store"),
            dcc.Store(id="upset-base-status-store"),
            dcc.Store(id="custom-label-store", data={}),
            plot_topbar(dataset, "upset", extra_controls=topbar_controls, extra_class="upset-onebar"),
            html.H1("Interactive UpSet Plot"),
            graph_download_buttons("upset-plot", "upset_plot"),
            html.Div(
                id="upset-workspace",
                className="upset-workspace split",
                children=[
                    html.Section(
                        className="upset-main-panel",
                        children=[
                            dcc.Loading(
                                dcc.Graph(
                                    id="upset-plot",
                                    className="upset-plot",
                                    config=graph_config(
                                        "upset_plot",
                                        scrollZoom=False,
                                        doubleClick=False,
                                        modeBarButtonsToRemove=[
                                            "zoom2d",
                                            "pan2d",
                                            "select2d",
                                            "lasso2d",
                                            "zoomIn2d",
                                            "zoomOut2d",
                                            "autoScale2d",
                                            "resetScale2d",
                                        ],
                                    ),
                                ),
                                type="circle",
                            ),
                        ],
                    ),
                    html.Section(
                        id="multi-detail-panel",
                        className="multi-detail-panel collapsed",
                        children=[
                            html.Button("Plot menu", id="multi-detail-toggle", n_clicks=0, className="multi-detail-toggle"),
                            html.Div(
                                id="multi-detail-body",
                                className="multi-detail-body",
                                children=[
                                    html.H2("Multi columns"),
                                    html.Div(id="multi-detail-status", className="status-line compact-status"),
                                    dcc.Checklist(
                                        id="multi-detail-checklist",
                                        options=MULTI_DETAIL_OPTIONS,
                                        value=DEFAULT_MULTI_DETAILS,
                                        className="multi-detail-checklist",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Section(
                        id="single-selection-panel",
                        className="selection-panel",
                        children=[
                            html.H2(id="selected-title", children="No selection"),
                            html.Div(
                                id="single-selection-analysis-grid",
                                className="analysis-grid",
                                children=[
                                    dcc.Graph(id="selected-specificity-plot", figure=empty_analysis_plot(), config=graph_config("selected_specificity")),
                                    dcc.Graph(id="selected-modification-plot", figure=empty_analysis_plot(), config=graph_config("selected_modification")),
                                    dcc.Graph(id="selected-termini-temple-plot", figure=empty_analysis_plot(), config=graph_config("selected_termini_temple")),
                                    dcc.Graph(id="selected-rt-plot", figure=empty_analysis_plot(), config=graph_config("selected_rt")),
                                    dcc.Graph(id="selected-rt-modification-plot", figure=empty_analysis_plot(), config=graph_config("selected_rt_modifications")),
                                    dcc.Graph(id="selected-rt-modification-residual-plot", figure=empty_analysis_plot(), config=graph_config("selected_rt_modification_residuals")),
                                    dcc.Graph(id="selected-mass-error-plot", figure=empty_analysis_plot(), config=graph_config("selected_mass_error")),
                                    dcc.Graph(id="selected-peptide-properties-plot", figure=empty_analysis_plot(), config=graph_config("selected_peptide_properties")),
                                ],
                            ),
                            html.H3("Selected items"),
                            dash_table.DataTable(
                                id="selected-items-table",
                                columns=[{"name": "item_id", "id": "item_id"}],
                                data=[],
                                page_size=15,
                                sort_action="native",
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontFamily": "Arial, sans-serif",
                                    "fontSize": "13px",
                                    "padding": "8px",
                                    "textAlign": "left",
                                    "border": "1px solid #E5E5E5",
                                },
                                style_header={"backgroundColor": "#F5F5F5", "fontWeight": "bold", "border": "1px solid #D0D0D0"},
                            ),
                            html.Details(
                                id="selected-decoy-fraction-panel",
                                className="property-panel",
                                open=True,
                                style={"display": "none"},
                                children=[
                                    html.Summary("ENT_ decoy fraction"),
                                    dcc.Graph(id="selected-decoy-fraction-plot", figure=empty_analysis_plot(), config=graph_config("selected_entrapment_fraction")),
                                ],
                            ),
                            html.Section(
                                id="ms-property-panel",
                                className="property-panel",
                                children=[
                                    html.H3("MS property follow-up"),
                                    html.Div(
                                        className="property-controls",
                                        children=[
                                            html.Div(
                                                className="compact-control property-mode-control",
                                                children=[
                                                    html.Label("Comparison"),
                                                    dcc.RadioItems(
                                                        id="property-comparison-mode",
                                                        options=[
                                                            {"label": "Selected only", "value": "selected"},
                                                            {"label": "Selected vs background", "value": "background"},
                                                        ],
                                                        value="selected",
                                                        inline=True,
                                                        className="view-toggle",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="compact-control property-background-control",
                                                children=[
                                                    html.Label("Background"),
                                                    dcc.Dropdown(
                                                        id="property-background-mode",
                                                        options=[
                                                            {"label": "All current items except selected", "value": "all_except_selected"},
                                                            {"label": "Baseline items except selected", "value": "baseline_except_selected"},
                                                            {"label": "Shared/core items except selected", "value": "core_except_selected"},
                                                        ],
                                                        value="all_except_selected",
                                                        clearable=False,
                                                        searchable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="compact-control property-plot-control",
                                                children=[
                                                    html.Label("Shown property plots"),
                                                    dcc.Checklist(
                                                        id="property-plot-checklist",
                                                        options=[
                                                            {"label": "Peptide length", "value": "peptide_length"},
                                                            {"label": "Molecular weight", "value": "molecular_weight"},
                                                            {"label": "GRAVY", "value": "gravy"},
                                                            {"label": "pI", "value": "pI"},
                                                            {"label": "C-terminal residue", "value": "c_terminal_residue"},
                                                            {"label": "Missed cleavages", "value": "missed_cleavages"},
                                                            {"label": "Internal R/K counts", "value": "internal_rk_counts"},
                                                            {"label": "Aromatic fraction", "value": "aromatic_fraction"},
                                                            {"label": "Acidic/basic fraction", "value": "acidic_basic_fraction"},
                                                        ],
                                                        value=[
                                                            "peptide_length",
                                                            "molecular_weight",
                                                            "gravy",
                                                            "pI",
                                                            "c_terminal_residue",
                                                            "missed_cleavages",
                                                            "internal_rk_counts",
                                                        ],
                                                        inline=True,
                                                        className="property-checklist",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(id="property-plot-status", className="status-line compact-status"),
                                    html.Div(id="property-plot-grid", className="property-plot-grid"),
                                ],
                            ),
                            html.Section(
                                id="motif-property-panel",
                                className="property-panel",
                                children=[
                                    html.H3("Position & motif plots"),
                                    html.Div(
                                        className="property-controls",
                                        children=[
                                            html.Div(
                                                className="compact-control property-mode-control",
                                                children=[
                                                    html.Label("Comparison"),
                                                    dcc.RadioItems(
                                                        id="motif-comparison-mode",
                                                        options=[
                                                            {"label": "Selected only", "value": "selected"},
                                                            {"label": "Selected vs background", "value": "background"},
                                                        ],
                                                        value="selected",
                                                        inline=True,
                                                        className="view-toggle",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="compact-control property-background-control",
                                                children=[
                                                    html.Label("Background"),
                                                    dcc.Dropdown(
                                                        id="motif-background-mode",
                                                        options=[
                                                            {"label": "All identified", "value": "all_identified"},
                                                            {"label": "Baseline identified", "value": "baseline_identified"},
                                                            {"label": "Shared/core identified", "value": "core_identified"},
                                                        ],
                                                        value="all_identified",
                                                        clearable=False,
                                                        searchable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="compact-control property-background-control",
                                                children=[
                                                    html.Label("Logo mode"),
                                                    dcc.Dropdown(
                                                        id="sequence-logo-mode",
                                                        options=[
                                                            {"label": "Selected only", "value": "selected"},
                                                            {"label": "Selected vs all identified", "value": "all_identified"},
                                                            {"label": "Selected vs baseline identified", "value": "baseline_identified"},
                                                            {"label": "Selected vs shared/core identified", "value": "core_identified"},
                                                        ],
                                                        value="selected",
                                                        clearable=False,
                                                        searchable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="compact-control property-plot-control",
                                                children=[
                                                    html.Label("Shown position & motif plots"),
                                                    dcc.Checklist(
                                                        id="motif-plot-checklist",
                                                        options=[
                                                            {"label": "Relative N-terminus position", "value": "relative_nterm_position"},
                                                            {"label": "P1 residue distribution", "value": "p1_distribution"},
                                                            {"label": "P1' residue distribution", "value": "p1_prime_distribution"},
                                                            {"label": "Cleavage-site sequence logo", "value": "sequence_logo"},
                                                        ],
                                                        value=[
                                                            "relative_nterm_position",
                                                            "p1_distribution",
                                                            "p1_prime_distribution",
                                                            "sequence_logo",
                                                        ],
                                                        inline=True,
                                                        className="property-checklist",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(id="motif-plot-status", className="status-line compact-status"),
                                    html.Div(id="motif-plot-grid", className="property-plot-grid"),
                                ],
                            ),
                            html.Details(
                                className="property-panel",
                                open=False,
                                children=[
                                    html.Summary("Overview bar plot"),
                                    dcc.Graph(id="selected-items-plot", figure=selected_items_plot_empty(), config=graph_config("selected_items")),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="multi-selection-strip",
                className="multi-selection-strip",
            ),
            html.Div(
                id="multi-selection-results",
                className="multi-selection-results",
            ),
        ],
    )


def selected_items_plot_empty():
    import pandas as pd

    return selected_items_plot(pd.DataFrame())


def empty_analysis_plot():
    return empty_figure("Select an intersection or set.")
