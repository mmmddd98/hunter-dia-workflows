from __future__ import annotations

from collections import defaultdict
from urllib.parse import quote

import pandas as pd
import plotly.express as px
from dash import dcc, html

from data.figures import PLOTLY_TEMPLATE, publication_graph_config, style_axes
from pages.dataset_overview import plot_topbar


SECTION_META = {
    "combined": ("Combined base losses", "Combined LR overview plots for the base distillation round d0."),
    "distillate": ("Distillation losses", "Combined LR overview plots for distillation rounds d1 and higher."),
    "single_lr": ("Single LR runs", "Per-learning-rate plots; when present, distillation rounds are shown inside each image."),
}


def layout(dataset: dict, repository) -> html.Div:
    plots = repository.list_loss_plots(dataset["id"])
    curves = repository.load_loss_curves(dataset["id"])
    return html.Div(
        children=[
            plot_topbar(dataset, "loss"),
            html.Div(
                className="plot-page-heading compact-plot-heading",
                children=[
                    html.H1("Loss Plots"),
                    html.Div(
                        f"{len(plots)} saved loss plot image(s)" if plots else "No saved loss plot images found.",
                        className="status-line compact-status",
                    ),
                ],
            ),
            html.Div(
                className="loss-view-toolbar",
                children=[
                    html.Div(
                        className="compact-control",
                        children=[
                            html.Label("View"),
                            dcc.RadioItems(
                                id="loss-view-mode",
                                options=[
                                    {"label": "Interactive plots", "value": "interactive"},
                                    {"label": "Saved images", "value": "images"},
                                ],
                                value="interactive",
                                inline=True,
                                className="view-toggle",
                            ),
                        ],
                    ),
                    html.Div(
                        f"{len(curves):,} loss points loaded" if not curves.empty else "No loss CSV data found.",
                        className="status-line compact-status",
                    ),
                ],
            ),
            html.Div(
                id="loss-interactive-view",
                children=interactive_loss_sections(curves),
            ),
            html.Div(
                id="loss-image-view",
                style={"display": "none"},
                children=loss_sections(dataset["id"], plots),
            ),
        ],
    )


def interactive_loss_sections(curves: pd.DataFrame) -> list:
    if curves.empty:
        return [
            html.Div(
                className="empty-state",
                children=[
                    html.H2("No interactive loss data available"),
                    html.P("Switch to Saved images to view existing rendered plots.", className="muted"),
                ],
            )
        ]

    sections = []
    for profile in sorted(curves["profile"].unique()):
        profile_data = curves.loc[curves["profile"] == profile]
        graphs = []
        for (model, distill), data in profile_data.groupby(["model", "distill"], sort=True):
            graphs.append(
                html.Div(
                    className="loss-interactive-card",
                    children=dcc.Graph(
                        figure=interactive_loss_figure(data, profile, model, int(distill)),
                        config=publication_graph_config(f"loss_{profile}_{model}_d{int(distill)}"),
                    ),
                )
            )
        sections.append(
            html.Section(
                className="loss-section loss-interactive-section",
                children=[
                    html.Div(
                        className="loss-section-heading",
                        children=[
                            html.H2(profile),
                            html.P("Click legend entries to hide or isolate learning rates.", className="muted"),
                        ],
                    ),
                    html.Div(className="loss-interactive-grid", children=graphs),
                ],
            )
        )
    return sections


def interactive_loss_figure(data: pd.DataFrame, profile: str, model: str, distill: int):
    plot_data = data.copy()
    plot_data["learning_rate"] = plot_data["lr_tag"].map(format_learning_rate)
    fig = px.line(
        plot_data,
        x="percent",
        y="loss",
        color="learning_rate",
        markers=True,
        render_mode="webgl",
        template=PLOTLY_TEMPLATE,
        hover_data={"lr_tag": True, "learning_rate": False, "percent": ":.1f", "loss": ":.6g"},
        labels={"percent": "Training progress (%)", "loss": "Loss", "learning_rate": "Learning rate"},
    )
    fig.update_traces(line={"width": 2}, marker={"size": 4})
    fig.update_layout(
        title=f"{profile} · {model} loss · d{distill}",
        height=520,
        hovermode="closest",
        legend_title_text="Learning rate",
    )
    return style_axes(fig)


def format_learning_rate(value: str) -> str:
    text = str(value)
    if "p" in text:
        text = text.replace("p", ".")
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def loss_sections(dataset_id: str, plots: list[dict]) -> list:
    if not plots:
        return [
            html.Div(
                className="empty-state",
                children=[
                    html.H2("No loss plots available"),
                    html.P(
                        "Expected images under 03_tune_models/**/loss_plots after Step 3 loss extraction.",
                        className="muted",
                    ),
                ],
            )
        ]

    grouped = defaultdict(list)
    for plot in plots:
        grouped[plot["kind"]].append(plot)

    sections = []
    for kind in ("combined", "distillate", "single_lr"):
        kind_plots = grouped.get(kind, [])
        if not kind_plots:
            continue
        title, description = SECTION_META[kind]
        sections.append(
            html.Section(
                className=f"loss-section loss-section-{kind}",
                children=[
                    html.Div(
                        className="loss-section-heading",
                        children=[
                            html.H2(title),
                            html.P(description, className="muted"),
                        ],
                    ),
                    *profile_groups(dataset_id, kind_plots),
                ],
            )
        )
    return sections


def profile_groups(dataset_id: str, plots: list[dict]) -> list:
    by_profile = defaultdict(list)
    for plot in plots:
        by_profile[plot["profile"]].append(plot)

    groups = []
    for profile in sorted(by_profile):
        groups.append(
            html.Div(
                className="loss-profile-group",
                children=[
                    html.Div(
                        className="loss-profile-heading",
                        children=[
                            html.H3(profile),
                            html.Span(f"{len(by_profile[profile])} plots", className="pill"),
                        ],
                    ),
                    html.Div(
                        className="loss-plot-grid",
                        children=[loss_plot_card(dataset_id, plot) for plot in by_profile[profile]],
                    ),
                ],
            )
        )
    return groups


def loss_plot_card(dataset_id: str, plot: dict) -> html.Figure:
    src = f"/loss-plot-image?dataset={quote(dataset_id, safe='')}&path={quote(plot['relative_path'], safe='')}"
    meta = [
        html.Span(plot["model"], className="pill"),
    ]
    if plot.get("distill") is not None:
        meta.append(html.Span(f"d{plot['distill']}", className="pill"))
    if plot.get("lr_tag"):
        meta.append(html.Span(f"LR {plot['lr_tag'].replace('p', '.')}", className="pill"))

    return html.Figure(
        className="loss-plot-card",
        children=[
            html.Div(
                className="loss-plot-frame",
                children=html.Img(src=src, alt=plot["title"]),
            ),
            html.Figcaption(
                children=[
                    html.Div(plot["title"], className="loss-plot-title"),
                    html.Div(meta, className="loss-plot-meta"),
                    html.Code(plot["relative_path"], className="loss-plot-path"),
                ]
            ),
        ],
    )
