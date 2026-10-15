from __future__ import annotations

from dash import dcc, html

from data.figures import empty_figure, publication_graph_config
from data.triplemix import TRIPLEMIX_FILTER_OPTIONS
from pages.dataset_overview import graph_download_buttons, plot_topbar


PLOT_ROWS = (
    ("identifications", "triplemix_identifications", "1. Identifications by organism", 500),
    ("completeness", "triplemix_ab_overlap", "2. A/B detection overlap", 500),
    ("ab-scatter", "triplemix_a_vs_b", "3. A vs B abundance", 820),
    ("ma", "triplemix_fold_change_abundance", "4. Fold change vs mean abundance", 820),
    ("fc-distribution", "triplemix_fold_change_distribution", "5. Fold-change accuracy", 560),
    ("modifications", "triplemix_nterm_modifications", "6. N-terminal modification profile", 560),
    ("specificity", "triplemix_cleavage_specificity", "7. Cleavage specificity profile", 560),
)


def layout(dataset: dict) -> html.Div:
    searches = dataset.get("set_columns", [])
    selected_search = searches[0] if searches else None
    controls = [
        html.Div(
            className="compact-control",
            children=[
                html.Label("Search used for both methods"),
                dcc.Dropdown(
                    id="triplemix-search-dropdown",
                    options=[{"label": search, "value": search} for search in searches],
                    value=selected_search,
                    clearable=False,
                ),
            ],
        ),
        html.Div(
            className="compact-control",
            children=[
                html.Label("Termini subset"),
                dcc.Dropdown(
                    id="triplemix-filter-dropdown",
                    options=TRIPLEMIX_FILTER_OPTIONS,
                    value="all",
                    clearable=False,
                    searchable=False,
                ),
            ],
        ),
    ]
    return html.Div(
        children=[
            plot_topbar(dataset, "triplemix", controls, extra_class="triplemix-toolbar"),
            html.H1("Triplemix A/B performance"),
            html.P(
                "QMD-compatible analysis of the final Step-5 report at 1% precursor FDR. "
                "The raw diagnostic uses Precursor.Quantity before DIA-NN quantity normalization, MAD scaling, "
                "or organism centering and shows observed medians without target assumptions. "
                "Both methods use the same complete A/B features. E. coli centering adjusts fold changes on high-abundance "
                "modified E. coli; Arabidopsis reference normalization shifts every replicate to the upper-quartile Arabidopsis reference. "
                "The separate phosphopeptide standard is disabled.",
                className="muted",
            ),
            html.Div(id="triplemix-summary-cards", className="metric-card-grid"),
            html.Div(
                className="triplemix-comparison-row triplemix-raw-row",
                children=[
                    html.H2("0. Unnormalized diagnostic", className="triplemix-row-title"),
                    html.Div(
                        className="triplemix-plot-card",
                        children=[
                            html.Div(
                                "Raw Precursor.Quantity · observed medians only",
                                className="triplemix-method-label triplemix-method-raw",
                            ),
                            html.Div(
                                className="triplemix-card-actions",
                                children=graph_download_buttons(
                                    "triplemix-raw-ma-graph",
                                    "triplemix_raw_fold_change_abundance",
                                ),
                            ),
                            dcc.Graph(
                                id="triplemix-raw-ma-graph",
                                className="triplemix-graph",
                                figure=empty_figure("Loading raw Triplemix diagnostic."),
                                config=publication_graph_config("triplemix_raw_fold_change_abundance"),
                                responsive=True,
                                style={"height": "820px", "minHeight": "820px"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="triplemix-comparison",
                children=[
                    html.Div(
                        className="triplemix-comparison-row",
                        children=[
                            html.H2(title, className="triplemix-row-title"),
                            html.Div(
                                className="triplemix-comparison-columns",
                                children=[
                                    html.Div(
                                        className="triplemix-plot-card",
                                        children=[
                                            html.Div(
                                                "E. coli fold-change centering" if side == "left" else "Arabidopsis reference normalization",
                                                className=f"triplemix-method-label triplemix-method-{side}",
                                            ),
                                            html.Div(
                                                className="triplemix-card-actions",
                                                children=graph_download_buttons(
                                                    f"triplemix-{side}-{slug}-graph",
                                                    f"{filename}_{side}",
                                                ),
                                            ),
                                            dcc.Graph(
                                                id=f"triplemix-{side}-{slug}-graph",
                                                className="triplemix-graph",
                                                figure=empty_figure("Loading Triplemix analysis."),
                                                config=publication_graph_config(f"{filename}_{side}"),
                                                responsive=True,
                                                style={"height": f"{height}px", "minHeight": f"{height}px"},
                                            ),
                                        ],
                                    )
                                    for side in ("left", "right")
                                ],
                            ),
                        ],
                    )
                    for slug, filename, title, height in PLOT_ROWS
                ],
            ),
        ]
    )
