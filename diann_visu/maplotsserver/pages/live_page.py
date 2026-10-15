from __future__ import annotations

from dash import dcc, html

from data.figures import empty_figure, live_status_cards, publication_graph_config
from pages.dataset_overview import plot_display_settings


REFRESH_OPTIONS = [
    {"label": "5 s", "value": 5},
    {"label": "15 s", "value": 15},
    {"label": "30 s", "value": 30},
    {"label": "Off", "value": 0},
]


def layout(experiments: list[str], selected: str | None, summary: dict | None = None) -> html.Div:
    summary = summary or {}
    return html.Div(
        children=[
            html.Div(
                className="plot-page-heading compact-plot-heading",
                children=[
                    html.Div(
                        children=[
                            html.H1("Live Run Monitor"),
                            html.P("Progress and resource traces for experiments while the pipeline is running.", className="muted"),
                        ]
                    ),
                    html.Div(id="live-status", className="status-line compact-status"),
                ],
            ),
            html.Div(
                className="ram-controls",
                children=[
                    plot_display_settings(),
                    html.Div(
                        className="compact-control live-experiment-control",
                        children=[
                            html.Label("Experiment"),
                            dcc.Dropdown(
                                id="live-experiment-dropdown",
                                options=[{"label": exp, "value": exp} for exp in experiments],
                                value=selected,
                                clearable=False,
                                searchable=True,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control ram-mode-control",
                        children=[
                            html.Label("Metric"),
                            dcc.Dropdown(
                                id="live-ram-mode-dropdown",
                                options=[
                                    {"label": "Pipeline RAM", "value": "pipeline"},
                                    {"label": "System used RAM", "value": "system"},
                                ],
                                value="pipeline",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control live-log-control",
                        children=[
                            html.Label("Visible RAM logs"),
                            dcc.Dropdown(id="live-log-dropdown", multi=True, searchable=True),
                        ],
                    ),
                    html.Div(
                        className="compact-control ram-prefix-control",
                        children=[
                            html.Label("Step prefix"),
                            dcc.Dropdown(
                                id="live-step-prefix-dropdown",
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
                        className="compact-control ram-layout-control",
                        children=[
                            html.Label("Refresh"),
                            dcc.Dropdown(
                                id="live-refresh-dropdown",
                                options=REFRESH_OPTIONS,
                                value=15,
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
                    html.Summary("Edit RAM / runtime plot labels"),
                    html.Div(id="live-label-fields", className="ram-label-fields"),
                ],
            ),
            dcc.Interval(id="live-refresh-interval", interval=15_000, n_intervals=0),
            html.Div(id="live-card-row", className="live-card-row", children=status_cards(summary)),
            dcc.Graph(id="live-progress-timeline", figure=empty_figure("Choose an experiment."), config=publication_graph_config("live_progress_timeline")),
            html.Div(
                className="ram-plot-grid horizontal",
                children=[
                    dcc.Graph(id="live-ram-usage", figure=empty_figure("No RAM logs yet."), config=publication_graph_config("live_ram_usage")),
                    dcc.Graph(id="live-progress-states", figure=empty_figure("No progress events yet."), config=publication_graph_config("live_progress_states")),
                ],
            ),
            html.Div(
                className="ram-plot-grid horizontal",
                children=[
                    dcc.Graph(id="live-ram-duration-timeline", figure=empty_figure("No duration data yet."), config=publication_graph_config("live_ram_duration_timeline")),
                    dcc.Graph(id="live-ram-duration-summary", figure=empty_figure("No duration data yet."), config=publication_graph_config("live_ram_duration_summary")),
                ],
            ),
        ],
    )


def status_cards(summary: dict) -> list[html.Div]:
    return [
        html.Div(
            className="live-card",
            children=[
                html.Span(card["label"], className="live-card-label"),
                html.Strong(card["value"], className="live-card-value"),
            ],
        )
        for card in live_status_cards(summary)
    ]
