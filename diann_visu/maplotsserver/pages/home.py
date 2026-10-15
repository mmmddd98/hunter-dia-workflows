from __future__ import annotations

from dash import dcc, html


def layout(datasets: dict[str, dict]) -> html.Div:
    pipeline_versions = sorted({dataset["pipeline_version"] for dataset in datasets.values()})
    organisms = sorted({dataset["organism"] for dataset in datasets.values()})
    experiment_types = sorted({dataset["experiment_type"] for dataset in datasets.values()})
    clusters = sorted({dataset["cluster"] for dataset in datasets.values()})
    return html.Div(
        children=[
            dcc.Store(id="home-multi-mode-store", data=False),
            html.Div(
                className="home-filter-row",
                children=[
                    html.Div(
                        className="home-filter-control",
                        children=[
                            html.Label("Cluster", htmlFor="cluster-filter"),
                            dcc.Dropdown(
                                id="cluster-filter",
                                options=[{"label": "All clusters", "value": "__all__"}]
                                + [{"label": cluster, "value": cluster} for cluster in clusters],
                                value="__all__",
                                clearable=False,
                                searchable=False,
                                className="cluster-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="home-filter-control",
                        children=[
                            html.Label("Pipeline version", htmlFor="pipeline-version-filter"),
                            dcc.Dropdown(
                                id="pipeline-version-filter",
                                options=[{"label": "All versions", "value": "__all__"}]
                                + [{"label": version, "value": version} for version in pipeline_versions],
                                value="__all__",
                                clearable=False,
                                searchable=False,
                                className="pipeline-version-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="home-filter-control",
                        children=[
                            html.Label("Organism", htmlFor="organism-filter"),
                            dcc.Dropdown(
                                id="organism-filter",
                                options=[{"label": "All organisms", "value": "__all__"}]
                                + [{"label": organism, "value": organism} for organism in organisms],
                                value="__all__",
                                clearable=False,
                                searchable=False,
                                className="organism-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="home-filter-control",
                        children=[
                            html.Label("Experiment type", htmlFor="experiment-type-filter"),
                            dcc.Dropdown(
                                id="experiment-type-filter",
                                options=[{"label": "All experiment types", "value": "__all__"}]
                                + [{"label": experiment_type, "value": experiment_type} for experiment_type in experiment_types],
                                value="__all__",
                                clearable=False,
                                searchable=False,
                                className="experiment-type-dropdown",
                            ),
                        ],
                    ),
                    html.Div(id="home-result-summary", className="status-line"),
                ],
            ),
            html.Div(
                className="home-multi-row",
                children=[
                    html.Button("Multi experiment", id="home-multi-toggle", n_clicks=0, className="action-button secondary-action"),
                    html.Button("Choose plots", id="home-multi-confirm", n_clicks=0, className="action-button", disabled=True),
                    html.Div(id="home-multi-status", className="status-line compact-status"),
                ],
            ),
            html.Div(
                id="home-card-grid",
                className="card-grid",
                children=dataset_cards(datasets),
            ),
        ],
    )


def dataset_cards(
    datasets: dict[str, dict],
    pipeline_version: str = "__all__",
    organism: str = "__all__",
    experiment_type: str = "__all__",
    cluster: str = "__all__",
    *,
    multi_mode: bool = False,
    selected_ids: list[str] | None = None,
) -> list:
    selected = set(selected_ids or [])
    cards = [
        dataset_card(dataset_id, dataset, multi_mode=multi_mode, selected=dataset_id in selected)
        for dataset_id, dataset in datasets.items()
        if pipeline_version == "__all__" or dataset["pipeline_version"] == pipeline_version
        if organism == "__all__" or dataset["organism"] == organism
        if experiment_type == "__all__" or dataset["experiment_type"] == experiment_type
        if cluster == "__all__" or dataset["cluster"] == cluster
    ]
    show_live_card = not multi_mode and pipeline_version == "__all__" and organism == "__all__" and experiment_type == "__all__"
    return ([live_card()] if show_live_card else []) + cards


def dataset_card(dataset_id: str, dataset: dict, *, multi_mode: bool = False, selected: bool = False):
    card = html.Div(
        className=f"card dataset-card{' selected' if selected else ''}{' multi-mode' if multi_mode else ''}",
        children=[
            html.H2(dataset["title"]),
            html.P(dataset["description"], className="muted"),
            html.Div(
                [
                    html.Span(dataset["pipeline_version"], className="pill pipeline-version-pill"),
                    html.Span(dataset["cluster"], className="pill cluster-pill"),
                    html.Span(dataset["organism"], className="pill"),
                    html.Span(dataset["experiment_type"], className="pill experiment-type-pill"),
                    html.Span(f"{len(dataset['set_columns'])} searches", className="pill"),
                    *running_badge(dataset),
                ],
                className="card-meta-row",
            ),
        ],
    )
    if multi_mode:
        return html.Div(
            className="card-link multi-card-link",
            children=[
                dcc.Checklist(
                    id={"type": "home-dataset-select", "dataset_id": dataset_id},
                    options=[{"label": "", "value": dataset_id}],
                    value=[dataset_id] if selected else [],
                    className="home-card-selector",
                ),
                card,
            ],
        )
    return dcc.Link(href=f"/dataset/{dataset_id}", className="card-link", children=card)


def running_badge(dataset: dict) -> list[html.Span]:
    if not dataset.get("is_running"):
        return []
    return [html.Span("running", className="pill running-pill")]


def live_card():
    return dcc.Link(
        href="/live",
        className="card-link",
        children=html.Div(
            className="card",
            children=[
                html.H2("Live Run Monitor"),
                html.P("Follow progress.txt, RAM traces and step timing for the current pipeline run.", className="muted"),
                html.Div([html.Span("auto-refresh", className="pill")], className="card-meta-row"),
            ],
        ),
    )
