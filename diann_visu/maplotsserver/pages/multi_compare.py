from __future__ import annotations

from dash import dcc, html


MULTI_PLOT_CARDS = [
    (
        "Interactive UpSet",
        "Compare shared and experiment-specific entities across all selected searches.",
        "upset",
        True,
    ),
    (
        "QC Overview",
        "Place search-level identification counts from selected experiments side by side.",
        "qc",
        True,
    ),
    (
        "Intensity Distribution",
        "Compare intensity shapes across the combined searches.",
        "intensity",
        True,
    ),
    (
        "N-Termini Overview",
        "Compare N-terminal specificity and modification evidence across experiments.",
        "nterm",
        True,
    ),
    (
        "Entrapment FDR",
        "Needs a dedicated cross-experiment FDR design before enabling.",
        "entrapment",
        False,
    ),
    (
        "RAM Usage",
        "Run-monitoring traces stay tied to individual pipeline runs.",
        "ram",
        False,
    ),
]


def layout(dataset: dict, query_string: str) -> html.Div:
    href_prefix = "/multi"
    return html.Div(
        children=[
            html.Div(
                [
                    html.H1("Multi Experiment Compare"),
                    html.P(dataset["description"], className="muted"),
                ],
                className="page-heading",
            ),
            html.Div(
                className="search-list",
                children=[
                    html.Span(dataset["organism"], className="pill"),
                    html.Span(f"{len(dataset.get('multi_dataset_ids', []))} experiments", className="pill"),
                    html.Span(f"{len(dataset.get('set_columns', []))} searches", className="pill"),
                ],
            ),
            html.Div(
                className="multi-experiment-list",
                children=[
                    html.Span(experiment_id.replace("::", " / "), className="pill experiment-type-pill")
                    for experiment_id in dataset.get("multi_dataset_ids", [])
                ],
            ),
            html.Div(
                className="card-grid multi-plot-grid",
                children=[
                    plot_card(
                        title,
                        description,
                        href=f"{href_prefix}/{slug}{query_string}",
                        enabled=enabled,
                    )
                    for title, description, slug, enabled in MULTI_PLOT_CARDS
                ],
            ),
        ],
    )


def plot_card(title: str, description: str, *, href: str, enabled: bool):
    card = html.Div(
        className=f"card multi-plot-card{' disabled-card' if not enabled else ''}",
        children=[
            html.H2(title),
            html.P(description, className="muted"),
            html.Div(
                [html.Span("available" if enabled else "not enabled", className="pill")],
                className="card-meta-row",
            ),
        ],
    )
    if not enabled:
        return html.Div(className="card-link disabled-card-link", children=card)
    return dcc.Link(href=href, className="card-link", children=card)
