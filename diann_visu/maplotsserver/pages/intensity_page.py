from __future__ import annotations

from dash import dcc, html

from data.figures import empty_figure, publication_graph_config
from pages.dataset_overview import graph_download_buttons, plot_topbar


def layout(dataset: dict, registry) -> html.Div:
    return html.Div(
        children=[
            plot_topbar(dataset, "intensity"),
            html.H1("Intensity Distribution"),
            graph_download_buttons("intensity-distribution-graph", "intensity_distribution"),
            dcc.Graph(id="intensity-distribution-graph", figure=empty_figure("Loading intensity distribution."), config=publication_graph_config("intensity_distribution")),
        ],
    )
