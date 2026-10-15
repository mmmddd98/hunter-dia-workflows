from __future__ import annotations

from dash import dcc, html

from data.figures import (
    DEFAULT_LABEL_MODE,
    DEFAULT_LABEL_SCALE,
    DEFAULT_PLOT_STYLE,
    LABEL_MODE_OPTIONS,
    PLOT_STYLE_OPTIONS,
    display_search_label,
)


PLOT_CARDS = [
    ("Live Run", "Progress, RAM and step timing while the pipeline is running.", "live"),
    ("UpSet Plot", "Interactive JS-built Plotly UpSet across searches.", "upset"),
    ("QC Overview", "Search-level counts from raw DIA-NN final reports.", "qc"),
    ("Intensity Distribution", "Intensity distributions from adapted matrices or raw quantities.", "intensity"),
    ("N-Termini Overview", "Termini specificity overview from maplot termini tables.", "nterm"),
    ("Triplemix A/B", "Organism-aware A/B fold-change performance for Mixed FASTA datasets.", "triplemix"),
    ("Entrapment FDR", "Target vs entrapped protein hits for empirical FDR checks.", "entrapment"),
    ("RAM Usage", "Memory traces across pipeline steps and individual runs.", "ram"),
    ("Loss Plots", "Saved DIA-NN tuning loss plots, including separate distillation rounds.", "loss"),
]

PLOT_OPTIONS = [{"label": title, "value": slug} for title, _description, slug in PLOT_CARDS]


def layout(dataset: dict) -> html.Div:
    dataset_id = dataset["id"]
    return html.Div(
        children=[
            html.Div(
                [
                    html.H1(dataset["title"]),
                    html.P(dataset["description"], className="muted"),
                ],
                className="page-heading",
            ),
            html.Div(className="search-list", children=[html.Span(dataset["organism"], className="pill")]),
            html.Div(
                className="search-list",
                children=[
                    html.Span(display_search_label(search, "compact"), className="pill")
                    for search in dataset["set_columns"]
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    dcc.Link(
                        href=f"/dataset/{dataset_id}/{slug}",
                        className="card-link",
                        children=html.Div(
                            className="card",
                            children=[html.H2(title), html.P(description, className="muted")],
                        ),
                    )
                    for title, description, slug in visible_plot_cards(dataset)
                ],
            ),
            reimport_panel(),
        ],
    )


def reimport_panel() -> html.Div:
    return html.Div(
        className="reimport-panel",
        children=[
            html.Div(
                className="reimport-heading",
                children=[
                    html.H2("Daten neu einlesen"),
                    html.P(
                        "Rohdaten erneut einlesen und die maplot-Tabellen fuer diesen Datensatz neu anlegen.",
                        className="muted",
                    ),
                ],
            ),
            html.Button("Neu einlesen", id="reimport-button", n_clicks=0, className="action-button"),
            html.Div(id="reimport-status", className="status-line"),
            html.Pre(id="reimport-log", className="terminal-log", children="Noch kein Lauf gestartet."),
            dcc.Interval(id="reimport-log-interval", interval=1000, n_intervals=0, disabled=True),
        ],
    )


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


def plot_topbar(dataset: dict, active: str, extra_controls: list | None = None, extra_class: str = "") -> html.Div:
    class_name = "plot-topbar plot-controls-bar"
    if extra_class:
        class_name = f"{class_name} {extra_class}"

    right_children = []
    right_children.append(plot_display_settings())
    if extra_controls:
        right_children.append(html.Div(className="plot-extra-controls", children=extra_controls))

    return html.Div(
        className=class_name,
        children=[
            html.Div(className="plot-topbar-right", children=right_children),
        ],
    )


def plot_display_settings() -> html.Details:
    return html.Details(
        className="plot-settings-drawer",
        open=False,
        children=[
            html.Summary(className="plot-settings-summary", children="Plot settings"),
            html.Div(
                className="plot-settings-body",
                children=[
                    html.Div(
                        className="compact-control label-mode-control",
                        children=[
                            html.Label("Search labels"),
                            dcc.Dropdown(
                                id="global-label-mode-dropdown",
                                options=LABEL_MODE_OPTIONS,
                                value=DEFAULT_LABEL_MODE,
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control plot-style-control",
                        children=[
                            html.Label("Plot style"),
                            dcc.RadioItems(
                                id="global-plot-style-radio",
                                options=PLOT_STYLE_OPTIONS,
                                value=DEFAULT_PLOT_STYLE,
                                inline=True,
                                className="view-toggle",
                            ),
                        ],
                    ),
                    html.Div(
                        className="compact-control label-size-control",
                        children=[
                            html.Label("Text & legend size"),
                            dcc.Slider(
                                id="global-label-scale-slider",
                                min=0.8,
                                max=1.8,
                                step=0.1,
                                value=DEFAULT_LABEL_SCALE,
                                marks={0.8: "80%", 1.0: "100%", 1.4: "140%", 1.8: "180%"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def visible_plot_cards(dataset: dict) -> list[tuple[str, str, str]]:
    cards = PLOT_CARDS
    if not dataset.get("is_running"):
        cards = [card for card in cards if card[2] != "live"]
    if not dataset.get("has_entrapment"):
        cards = [card for card in cards if card[2] != "entrapment"]
    if dataset.get("organism") != "Mixed" or dataset.get("multi_dataset_ids"):
        cards = [card for card in cards if card[2] != "triplemix"]
    return cards
