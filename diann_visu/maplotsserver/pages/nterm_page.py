from __future__ import annotations

from dash import dcc, html

from data.figures import empty_figure, publication_graph_config
from pages.dataset_overview import graph_download_buttons, plot_topbar


def layout(dataset: dict, registry) -> html.Div:
    return html.Div(
        children=[
            plot_topbar(dataset, "nterm"),
            html.H1("N-Termini Overview"),
            graph_download_buttons("nterm-overview-graph", "n_termini_overview"),
            dcc.Graph(id="nterm-overview-graph", figure=empty_figure("Loading N-termini overview."), config=publication_graph_config("n_termini_overview")),
        ],
    )
