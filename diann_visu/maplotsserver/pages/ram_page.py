from __future__ import annotations

from dash import dcc, html

from data.figures import empty_figure, publication_graph_config
from pages.dataset_overview import plot_topbar


RAM_MODE_OPTIONS = [
    {"label": "Pipeline RAM", "value": "pipeline"},
    {"label": "System used RAM", "value": "system"},
]

RAM_LAYOUT_OPTIONS = [
    {"label": "Horizontal", "value": "horizontal"},
    {"label": "Vertical", "value": "vertical"},
]


def layout(dataset: dict) -> html.Div:
    return html.Div(
        children=[
            plot_topbar(dataset, "ram"),
            html.Div(
                className="plot-page-heading compact-plot-heading",
                children=[
                    html.H1("RAM Usage"),
                    html.Div(className="status-line compact-status", id="ram-status"),
                ],
            ),
            html.Div(
                className="ram-controls",
                children=[
                    html.Div(
                        className="compact-control ram-step-control",
                        children=[
                            html.Label("Step"),
                            dcc.Dropdown(id="ram-step-dropdown", clearable=False, searchable=False),
                        ],
                    ),
                    html.Div(
                        className="compact-control ram-log-control",
                        children=[
                            html.Label("Visible runs / logs"),
                            dcc.Dropdown(id="ram-log-dropdown", multi=True, searchable=True),
                        ],
                    ),
                    html.Div(
                        className="compact-control ram-prefix-control",
                        children=[
                            html.Label("Step prefix"),
                            dcc.Dropdown(
                                id="ram-step-prefix-dropdown",
                                options=[
                                    {"label": "Step 1", "value": "step"},
                                    {"label": "1", "value": "number"},
                                    {"label": "None", "value": "none"},
                                ],
                                value="step",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control ram-mode-control",
                        children=[
                            html.Label("Metric"),
                            dcc.Dropdown(
                                id="ram-mode-dropdown",
                                options=RAM_MODE_OPTIONS,
                                value="pipeline",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control ram-layout-control",
                        children=[
                            html.Label("Layout"),
                            dcc.Dropdown(
                                id="ram-layout-dropdown",
                                options=RAM_LAYOUT_OPTIONS,
                                value="horizontal",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                ],
            ),
            html.Details(
                className="ram-label-editor",
                children=[
                    html.Summary("Edit plot labels"),
                    html.P("These labels are used in every RAM and runtime plot on this page.", className="muted"),
                    html.Div(id="ram-label-fields", className="ram-label-fields"),
                ],
            ),
            html.Div(
                id="ram-plot-grid",
                className="ram-plot-grid horizontal",
                children=[
                    dcc.Graph(id="ram-usage-graph", figure=empty_figure("Choose a RAM log."), config=publication_graph_config("ram_usage")),
                    dcc.Graph(id="ram-summary-graph", figure=empty_figure("Choose a RAM log."), config=publication_graph_config("ram_summary")),
                ],
            ),
            html.H2("Step duration"),
            html.Div(
                id="duration-plot-grid",
                className="ram-plot-grid horizontal",
                children=[
                    dcc.Graph(id="ram-duration-timeline-graph", figure=empty_figure("Choose a RAM log."), config=publication_graph_config("ram_duration_timeline")),
                    dcc.Graph(id="ram-duration-summary-graph", figure=empty_figure("Choose a RAM log."), config=publication_graph_config("ram_duration_summary")),
                ],
            ),
        ],
    )
