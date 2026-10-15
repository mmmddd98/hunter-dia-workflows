from __future__ import annotations

from dash import dcc, html

from data.figures import empty_figure, publication_graph_config
from pages.dataset_overview import graph_download_buttons, plot_topbar


def layout(dataset: dict, registry) -> html.Div:
    return html.Div(
        children=[
            plot_topbar(dataset, "qc"),
            html.H1("QC Overview"),
            graph_download_buttons("qc-overview-graph", "qc_overview"),
            dcc.Graph(id="qc-overview-graph", figure=empty_figure("Loading QC overview."), config=publication_graph_config("qc_overview")),
        ],
    )
