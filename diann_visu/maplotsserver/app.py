from __future__ import annotations

import copy
import sys
import re
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, quote

import pandas as pd
import polars as pl
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from flask import abort, request, send_file

from data.config import AppConfig


CONFIG = AppConfig.from_env()
if CONFIG.maplot_src.exists():
    sys.path.insert(0, str(CONFIG.maplot_src))
if CONFIG.dash_plotly_upset_src.exists():
    sys.path.insert(0, str(CONFIG.dash_plotly_upset_src))

from data.datasets import (  # noqa: E402
    DatasetRegistry,
    create_multi_dataset_id,
    filter_upset_items_by_organism,
    is_multi_dataset_id,
    multi_dataset_ids,
)
from data.figures import (  # noqa: E402
    empty_figure,
    entrapment_curve_figure,
    entrapment_distribution_figure,
    entrapment_gain_figure,
    intensity_histogram,
    display_search_label_map,
    entrapment_summary_figure,
    nterm_overview,
    program_decoy_vs_entrapment_figure,
    progress_state_figure,
    progress_timeline_figure,
    ram_duration_summary_figure,
    ram_duration_timeline_figure,
    ram_base_label,
    ram_summary_figure,
    ram_usage_figure,
    search_summary_figure,
    selected_decoy_fraction_plot,
    selected_decoy_temple_plot,
    selected_items_plot,
    selected_mass_error_plot,
    add_ms_properties,
    apply_plot_display,
    ms_property_figure,
    selected_category_comparison_plot,
    selected_modification_plot,
    nterm_site_modification_class,
    publication_graph_config,
    normalize_label_mode,
    normalize_label_scale,
    normalize_plot_style,
    selected_peptide_properties_plot,
    selected_rt_modification_plot,
    selected_rt_modification_residual_plot,
    selected_rt_plot,
    selected_specificity_plot,
    selected_termini_temple_plot,
)
from data.motif import (  # noqa: E402
    cached_background_windows,
    load_fasta_sequences,
    position_motif_figure,
    site_table_from_evidence,
)
from data.triplemix import (  # noqa: E402
    TRIPLEMIX_ORGANISMS,
    TRIPLEMIX_SHORT_NAMES,
    filter_triplemix_metrics,
    prepare_triplemix_plot_data,
    triplemix_ab_scatter_figure,
    triplemix_completeness_figure,
    triplemix_fc_distribution_figure,
    triplemix_identifications_figure,
    triplemix_ma_figure,
    triplemix_modification_figure,
    triplemix_raw_ma_figure,
    triplemix_specificity_figure,
)
from data.repository import ExperimentRepository  # noqa: E402
from dash_plotly_upset import attach_clientside_assets, clientside_script_urls, selected_items as selected_upset_items  # noqa: E402
from pages import dataset_overview, entrapment_page, home, intensity_page, live_page, loss_page, multi_compare, nterm_page, qc_page, ram_page, triplemix_page, upset_page  # noqa: E402


repository = ExperimentRepository(CONFIG)
registry = DatasetRegistry(repository)
DATASET_SNAPSHOT_TTL_SECONDS = 60
REIMPORT_JOBS: dict[str, dict] = {}
REIMPORT_LOCK = threading.Lock()

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="maplot server",
    external_scripts=clientside_script_urls(),
)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/png" sizes="32x32" href="/assets/maplot-icon-32.png">
        <link rel="apple-touch-icon" href="/assets/maplot-icon-180.png">
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
server = app.server
attach_clientside_assets(app)


@server.route("/loss-plot-image")
def loss_plot_image():
    dataset_id = request.args.get("dataset", "")
    relative_path = request.args.get("path", "")
    if not dataset_id or not relative_path:
        abort(400)

    root = repository.experiment_root(dataset_id).resolve()
    image_path = (root / relative_path).resolve()
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
    if image_path.suffix.lower() not in allowed_suffixes:
        abort(404)
    try:
        image_path.relative_to(root)
    except ValueError:
        abort(404)
    if not image_path.exists() or not image_path.is_file():
        abort(404)
    return send_file(image_path)


@lru_cache(maxsize=4)
def cached_dataset_snapshot(_bucket: int) -> dict[str, dict]:
    return registry.all()


def all_datasets() -> dict[str, dict]:
    return cached_dataset_snapshot(int(time.time() // DATASET_SNAPSHOT_TTL_SECONDS))


def global_header() -> html.Header:
    return html.Header(
        className="global-header",
        children=[
            dcc.Link(
                href="/",
                className="brand-link",
                children=[
                    html.Img(src="/assets/maplot_final-header.svg", alt="maplot server", className="brand-banner"),
                ],
            ),
            html.Div(
                className="header-context",
                children=[
                    html.Div(
                        className="header-context-control dataset-context-control",
                        children=[
                            html.Label("Dataset", htmlFor="dataset-page-dropdown"),
                            dcc.Dropdown(
                                id="dataset-page-dropdown",
                                options=[],
                                value=None,
                                placeholder="Choose dataset",
                                clearable=False,
                                searchable=True,
                                className="dataset-page-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        id="plot-header-control",
                        className="header-context-control plot-switcher",
                        children=[
                            html.Label("Plot", htmlFor="plot-page-dropdown"),
                            dcc.Dropdown(
                                id="plot-page-dropdown",
                                options=[],
                                value=None,
                                placeholder="Choose plot",
                                clearable=False,
                                searchable=False,
                                disabled=True,
                                className="plot-page-dropdown",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

app.layout = html.Div(
    children=[
        dcc.Location(id="url"),
        dcc.Store(id="dataset-id-store"),
        dcc.Store(id="current-page-store"),
        dcc.Store(id="reimport-job-store"),
        dcc.Store(id="label-mode-store", data="short"),
        dcc.Store(id="plot-style-store", data="default"),
        dcc.Store(id="label-scale-store", data=1.0),
        global_header(),
        html.Div(id="page-content"),
    ],
    id="app-shell",
    className="app-shell compact",
)


def followup_graph(
    figure,
    filename: str,
    class_name: str | None = None,
    plot_style: str | None = None,
    label_scale=None,
    className: str | None = None,
) -> dcc.Graph:
    return dcc.Graph(
        figure=apply_plot_display(figure, plot_style, label_scale),
        config=publication_graph_config(plot_filename(filename)),
        className=class_name or className,
    )


def plot_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned or "maplot_plot"


def evidence_filter_cache_key(evidence_filter) -> tuple[str, ...]:
    if evidence_filter in (None, "", "__all__"):
        return ()
    if isinstance(evidence_filter, (list, tuple, set)):
        return tuple(str(value) for value in evidence_filter if value not in (None, "", "__all__"))
    return (str(evidence_filter),)


@lru_cache(maxsize=32)
def _cached_upset_evidence_rows(
    dataset_id: str,
    item_level: str | None,
    evidence_filter_key: tuple[str, ...],
    include_entrapment: bool,
    qvalue_threshold: float | None,
) -> pd.DataFrame:
    evidence_filter = list(evidence_filter_key) if evidence_filter_key else None
    return registry.build_upset_evidence_rows(
        dataset_id,
        item_level,
        evidence_filter,
        include_entrapment=include_entrapment,
        qvalue_threshold=qvalue_threshold,
    )


def cached_upset_evidence_rows(
    dataset_id: str,
    item_level: str | None,
    evidence_filter,
    *,
    include_entrapment: bool = False,
    qvalue_threshold: float | None = None,
) -> pd.DataFrame:
    cached = _cached_upset_evidence_rows(
        dataset_id,
        item_level,
        evidence_filter_cache_key(evidence_filter),
        bool(include_entrapment),
        qvalue_threshold,
    )
    return cached.copy()


def parse_pathname(pathname: str | None) -> tuple[str | None, str | None]:
    parts = [part for part in (pathname or "/").strip("/").split("/") if part]
    if not parts:
        return None, None
    if parts[0] == "live":
        return "__live__", None
    if parts[0] == "multi-upset":
        return "__multi__", "upset"
    if parts[0] == "multi":
        return "__multi__", parts[1] if len(parts) >= 2 else None
    if len(parts) >= 2 and parts[0] == "dataset":
        return parts[1], parts[2] if len(parts) >= 3 else None
    return None, None


def multi_dataset_id_from_search(search: str | None) -> str | None:
    values = parse_qs((search or "").lstrip("?")).get("datasets", [])
    if not values:
        return None
    dataset_ids = [part for part in values[0].split(",") if part]
    return create_multi_dataset_id(dataset_ids) if len(dataset_ids) >= 2 else None


def multi_search_for_dataset_ids(dataset_ids: list[str]) -> str:
    return f"?datasets={quote(','.join(dataset_ids), safe='')}"


def selected_dataset_ids(selection_values, selection_ids) -> list[str]:
    selected = []
    for values, pattern_id in zip(selection_values or [], selection_ids or []):
        dataset_id = (pattern_id or {}).get("dataset_id")
        if dataset_id and dataset_id in (values or []):
            selected.append(dataset_id)
    return selected


def not_found(message: str) -> html.Div:
    return html.Div(
        children=[
            html.H1("Page not found"),
            html.P(message, className="muted"),
        ],
    )


def no_quant_results_yet(dataset_id: str) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                [
                    html.H1(dataset_id.replace("_", " ")),
                    html.P(
                        "This run is currently active, but no completed quant search result is available yet.",
                        className="muted",
                    ),
                ],
                className="page-heading",
            ),
            html.Div(
                className="card-grid",
                children=[
                    dcc.Link(
                        href="/live",
                        className="card-link",
                        children=html.Div(
                            className="card",
                            children=[
                                html.H2("Live Run Monitor"),
                                html.P("Follow progress, RAM traces and step timing while the quant results are still missing.", className="muted"),
                                html.Div([html.Span("running", className="pill running-pill")], className="card-meta-row"),
                            ],
                        ),
                    )
                ],
            ),
            dataset_overview.reimport_panel(),
        ],
    )


def item_records_for(
    dataset_id: str,
    item_level: str | None = None,
    evidence_filter: str | None = None,
    include_entrapment: bool = False,
    qvalue_threshold: float | None = None,
    aggregation_source: str | None = None,
) -> tuple[list[dict], list[str]]:
    item_level = item_level or registry.default_item_level(dataset_id)
    item_df, set_cols, _source_label = registry.build_upset_items(
        dataset_id,
        item_level,
        evidence_filter,
        include_entrapment=include_entrapment,
        qvalue_threshold=qvalue_threshold,
        aggregation_source=aggregation_source,
    )
    item_df = item_df.where(pd.notna(item_df), None)
    return item_df.to_dict("records"), set_cols


def start_reimport_job(dataset_id: str) -> str:
    with REIMPORT_LOCK:
        for job_id, job in REIMPORT_JOBS.items():
            if job["dataset_id"] == dataset_id and job["status"] == "running":
                return job_id

        job_id = uuid.uuid4().hex
        REIMPORT_JOBS[job_id] = {
            "dataset_id": dataset_id,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "logs": deque(maxlen=1000),
        }

    def emit(message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with REIMPORT_LOCK:
            job = REIMPORT_JOBS.get(job_id)
            if job is not None:
                job["logs"].append(f"[{timestamp}] {message}")

    def run() -> None:
        emit("Starting table rebuild.")
        try:
            repository.rebuild_maplot_cache(dataset_id, log=emit)
            _cached_upset_evidence_rows.cache_clear()
        except Exception:
            emit("ERROR while rebuilding tables:")
            for line in traceback.format_exc().rstrip().splitlines():
                emit(line)
            with REIMPORT_LOCK:
                if job_id in REIMPORT_JOBS:
                    REIMPORT_JOBS[job_id]["status"] = "failed"
                    REIMPORT_JOBS[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            return
        with REIMPORT_LOCK:
            if job_id in REIMPORT_JOBS:
                REIMPORT_JOBS[job_id]["status"] = "done"
                REIMPORT_JOBS[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")

    threading.Thread(target=run, daemon=True).start()
    return job_id


def reimport_job_snapshot(job_id: str | None) -> dict | None:
    if not job_id:
        return None
    with REIMPORT_LOCK:
        job = REIMPORT_JOBS.get(job_id)
        if job is None:
            return None
        return {
            "dataset_id": job["dataset_id"],
            "status": job["status"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "logs": list(job["logs"]),
        }


def normalize_qvalue_threshold(value) -> float | None:
    if value in (None, "", "__all__"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_float(value, default: float | None = None) -> float | None:
    if value in (None, "", "__all__"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def qvalue_filter_options(max_qvalue: float) -> list[dict]:
    max_qvalue = max(float(max_qvalue), 0.0)
    candidates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    values = [value for value in candidates if value <= max_qvalue + 1e-12]
    if not values or abs(values[-1] - max_qvalue) > 1e-9:
        values.append(max_qvalue)
    values = sorted(set(round(value, 6) for value in values if value >= 0.001 or value == max_qvalue))
    return [
        {
            "label": f"<= {value:g}" if abs(value - max_qvalue) > 1e-9 else f"<= {value:g} (max)",
            "value": value,
        }
        for value in values
    ]


def default_qvalue_filter(max_qvalue: float) -> float:
    if max_qvalue <= 0.01:
        return round(float(max_qvalue), 6)
    return 0.01


def active_qvalue_threshold(dataset_id: str, value) -> float | None:
    qvalue_threshold = normalize_qvalue_threshold(value)
    if qvalue_threshold is not None:
        return qvalue_threshold
    return 0.01


@app.callback(
    Output("label-mode-store", "data"),
    Input("global-label-mode-dropdown", "value"),
)
def update_label_mode_store(label_mode):
    return normalize_label_mode(label_mode)


@app.callback(
    Output("plot-style-store", "data"),
    Input("global-plot-style-radio", "value"),
)
def update_plot_style_store(plot_style):
    return normalize_plot_style(plot_style)


@app.callback(
    Output("label-scale-store", "data"),
    Input("global-label-scale-slider", "value"),
)
def update_label_scale_store(label_scale):
    return normalize_label_scale(label_scale)


@app.callback(
    Output("upset-label-map-store", "data"),
    Input("set-columns-store", "data"),
)
def update_upset_label_map(set_columns):
    return {str(column): str(column) for column in (set_columns or [])}


def entrapment_summary_cards(metrics, distribution, level_label: str, qvalue_threshold: float, baseline_search: str | None):
    if metrics is None or metrics.is_empty():
        return []
    data = metrics.to_pandas()
    target_total = int(data["n_target"].sum()) if "n_target" in data.columns else 0
    entrapment_total = int(data["n_entrapment"].sum()) if "n_entrapment" in data.columns else 0
    decoy_total = int(data["n_program_decoy"].sum()) if "n_program_decoy" in data.columns else 0
    entrapment_rate = entrapment_total / target_total if target_total else 0.0
    entrapment_fraction = entrapment_total / (target_total + entrapment_total) if target_total + entrapment_total else 0.0
    worst_bin = "n/a"
    if distribution is not None and not distribution.is_empty():
        dist = distribution.to_pandas()
        if not dist.empty and "bin_entrapment_fdr_estimate" in dist.columns:
            row = dist.sort_values("bin_entrapment_fdr_estimate", ascending=False).iloc[0]
            worst_bin = f"{row['qvalue_bin_start']:.3g}-{row['qvalue_bin_end']:.3g} ({row['bin_entrapment_fdr_estimate']:.1%})"

    target_gain = "n/a"
    entrapment_gain = "n/a"
    if baseline_search and baseline_search in set(data["search"]):
        baseline = data.loc[data["search"] == baseline_search].iloc[0]
        tuned = data.loc[data["search"] != baseline_search]
        if not tuned.empty:
            best = tuned.sort_values("n_target", ascending=False).iloc[0]
            target_gain = f"{int(best['n_target'] - baseline['n_target']):+,}"
            entrapment_gain = f"{int(best['n_entrapment'] - baseline['n_entrapment']):+,}"

    cards = [
        ("Entity level", level_label),
        ("Q.Value threshold", f"<= {qvalue_threshold:g}"),
        ("Target IDs", f"{target_total:,}"),
        ("ENT_ IDs", f"{entrapment_total:,}"),
        ("Program decoys", f"{decoy_total:,}"),
        ("ENT_ / target", f"{entrapment_rate:.2%}"),
        ("ENT_ / (target + ENT_)", f"{entrapment_fraction:.2%}"),
        ("Worst local bin", worst_bin),
        ("Target gain vs baseline", target_gain),
        ("ENT_ gain vs baseline", entrapment_gain),
    ]
    card_grid = html.Div(
        className="metric-card-grid",
        children=[
            html.Div(
                className="metric-card",
                children=[
                    html.Div(label, className="metric-card-label"),
                    html.Div(value, className="metric-card-value"),
                ],
            )
            for label, value in cards
        ],
    )

    table = data.copy()
    if baseline_search and baseline_search in set(table["search"]):
        baseline = table.loc[table["search"] == baseline_search].iloc[0]
        table["target_gain_vs_baseline"] = table["n_target"] - baseline["n_target"]
        table["entrapment_gain_vs_baseline"] = table["n_entrapment"] - baseline["n_entrapment"]
    else:
        table["target_gain_vs_baseline"] = 0
        table["entrapment_gain_vs_baseline"] = 0
    table = table[
        [
            "search",
            "n_target",
            "n_entrapment",
            "n_program_decoy",
            "entrapment_rate",
            "entrapment_fraction",
            "target_gain_vs_baseline",
            "entrapment_gain_vs_baseline",
        ]
    ].copy()
    for col in ("entrapment_rate", "entrapment_fraction"):
        table[col] = table[col].map(lambda value: f"{value:.2%}")
    for col in ("n_target", "n_entrapment", "n_program_decoy", "target_gain_vs_baseline", "entrapment_gain_vs_baseline"):
        table[col] = table[col].map(lambda value: f"{int(value):+,}" if "gain" in col else f"{int(value):,}")
    metric_table = dash_table.DataTable(
        columns=[
            {"name": "Search", "id": "search"},
            {"name": "Target IDs", "id": "n_target"},
            {"name": "ENT_ IDs", "id": "n_entrapment"},
            {"name": "Program decoys", "id": "n_program_decoy"},
            {"name": "ENT_/target", "id": "entrapment_rate"},
            {"name": "ENT_/(target+ENT_)", "id": "entrapment_fraction"},
            {"name": "Target gain", "id": "target_gain_vs_baseline"},
            {"name": "ENT_ gain", "id": "entrapment_gain_vs_baseline"},
        ],
        data=table.to_dict("records"),
        page_size=8,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Arial, sans-serif",
            "fontSize": "12px",
            "padding": "7px",
            "textAlign": "left",
            "border": "1px solid #E5E5E5",
        },
        style_header={"backgroundColor": "#F5F5F5", "fontWeight": "bold", "border": "1px solid #D0D0D0"},
    )
    return [
        card_grid,
        html.Div(metric_table, className="entrapment-threshold-table"),
    ]


@app.callback(
    Output("app-shell", "className"),
    Output("page-content", "children"),
    Output("dataset-id-store", "data"),
    Output("current-page-store", "data"),
    Input("url", "pathname"),
    Input("url", "search"),
)
def route(pathname, search):
    dataset_id, page = parse_pathname(pathname)
    if dataset_id == "__live__":
        selected = repository.latest_live_experiment()
        summary = repository.live_summary(selected) if selected else {}
        return "app-shell wide", live_page.layout(repository.list_live_experiments(), selected, summary), None, "live"

    if dataset_id == "__multi__":
        multi_dataset_id = multi_dataset_id_from_search(search)
        dataset = registry.get(multi_dataset_id) if multi_dataset_id else None
        if dataset is None:
            return "app-shell compact", not_found("Select at least two experiments with the same organism for Multi UpSet."), None, None
        if page is None:
            return "app-shell compact", multi_compare.layout(dataset, search or ""), multi_dataset_id, None
        if page == "upset":
            content = upset_page.layout(dataset, source=registry.default_item_level(multi_dataset_id))
        elif page == "qc":
            content = qc_page.layout(dataset, registry)
        elif page == "intensity":
            content = intensity_page.layout(dataset, registry)
        elif page == "nterm":
            content = nterm_page.layout(dataset, registry)
        else:
            content = not_found(f"Unknown multi experiment page: {page}")
        return "app-shell wide plot-shell", content, multi_dataset_id, page

    if dataset_id is None:
        return "app-shell compact", home.layout(all_datasets()), None, None

    dataset = registry.get(dataset_id)
    if dataset is None:
        if repository.run_pid_status(dataset_id)["running"]:
            return "app-shell compact", no_quant_results_yet(dataset_id), dataset_id, None
        return "app-shell compact", not_found(f"Unknown dataset: {dataset_id}"), None, None

    default_source = "nterm_site"
    if page == "upset":
        try:
            default_source = registry.default_item_level(dataset_id)
        except Exception:
            default_source = "nterm_site"

    if page is None:
        content = dataset_overview.layout(dataset)
    elif page == "upset":
        content = upset_page.layout(dataset, source=default_source)
    elif page == "qc":
        content = qc_page.layout(dataset, registry)
    elif page == "intensity":
        content = intensity_page.layout(dataset, registry)
    elif page == "nterm":
        content = nterm_page.layout(dataset, registry)
    elif page == "triplemix" and dataset.get("organism") == "Mixed":
        content = triplemix_page.layout(dataset)
    elif page == "entrapment" and dataset.get("has_entrapment"):
        content = entrapment_page.layout(dataset, registry)
    elif page == "entrapment-method" and dataset.get("has_entrapment"):
        content = entrapment_page.method_layout(dataset)
    elif page == "live" and dataset.get("is_running"):
        summary = repository.live_summary(dataset_id)
        content = live_page.layout(repository.list_live_experiments(), dataset_id, summary)
    elif page == "ram":
        content = ram_page.layout(dataset)
    elif page == "loss":
        content = loss_page.layout(dataset, repository)
    else:
        content = not_found(f"Unknown page for {dataset['title']}: {page}")

    shell_class = "app-shell wide plot-shell" if page in {"upset", "qc", "intensity", "nterm", "triplemix", "entrapment", "entrapment-method", "live", "ram", "loss"} else "app-shell compact"
    return shell_class, content, dataset_id, page


@app.callback(
    Output("dataset-page-dropdown", "options"),
    Output("dataset-page-dropdown", "value"),
    Output("plot-page-dropdown", "options"),
    Output("plot-page-dropdown", "value"),
    Output("plot-page-dropdown", "disabled"),
    Output("plot-header-control", "style"),
    Output("url", "pathname"),
    Input("url", "pathname"),
    Input("dataset-page-dropdown", "value"),
    Input("plot-page-dropdown", "value"),
    State("current-page-store", "data"),
)
def header_dropdown_state(pathname, selected_dataset_id, plot_page, current_page):
    dataset_options = [
        {"label": dataset["title"], "value": dataset_id}
        for dataset_id, dataset in all_datasets().items()
    ]

    current_dataset_id, current_url_page = parse_pathname(pathname)

    if ctx.triggered_id == "dataset-page-dropdown":
        if not selected_dataset_id or selected_dataset_id == current_dataset_id:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        dataset = registry.get(selected_dataset_id)
        if dataset is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        visible_pages = [slug for _title, _description, slug in dataset_overview.visible_plot_cards(dataset)]
        page = current_page if current_page in visible_pages else (visible_pages[0] if visible_pages else None)
        url = f"/dataset/{selected_dataset_id}/{page}" if page else f"/dataset/{selected_dataset_id}"
        return no_update, no_update, no_update, no_update, no_update, no_update, url

    if ctx.triggered_id == "plot-page-dropdown":
        if not plot_page or not current_dataset_id or current_dataset_id in {"__live__", "__multi__"} or plot_page == current_url_page:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        return no_update, no_update, no_update, no_update, no_update, no_update, f"/dataset/{current_dataset_id}/{plot_page}"

    dataset_id, page = current_dataset_id, current_url_page
    dataset_value = None if dataset_id in (None, "__live__", "__multi__") else dataset_id
    if not dataset_id or dataset_id in {"__live__", "__multi__"}:
        return dataset_options, dataset_value, [], None, True, {"display": "none"}, no_update

    dataset = registry.get(dataset_id)
    if dataset is None or page is None:
        return dataset_options, dataset_value, [], None, True, {"display": "none"}, no_update

    plot_options = [{"label": title, "value": slug} for title, _description, slug in dataset_overview.visible_plot_cards(dataset)]
    visible_values = {option["value"] for option in plot_options}
    plot_value = page if page in visible_values else (plot_options[0]["value"] if plot_options else None)
    return dataset_options, dataset_value, plot_options, plot_value, False, {}, no_update


@app.callback(
    Output("reimport-job-store", "data"),
    Input("reimport-button", "n_clicks"),
    State("dataset-id-store", "data"),
    prevent_initial_call=True,
)
def launch_reimport(n_clicks, dataset_id):
    if not n_clicks or not dataset_id:
        return no_update
    return {"job_id": start_reimport_job(dataset_id)}


@app.callback(
    Output("reimport-status", "children"),
    Output("reimport-log", "children"),
    Output("reimport-button", "disabled"),
    Output("reimport-log-interval", "disabled"),
    Input("reimport-job-store", "data"),
    Input("reimport-log-interval", "n_intervals"),
    State("dataset-id-store", "data"),
)
def update_reimport_log(job_data, _n_intervals, dataset_id):
    job_id = (job_data or {}).get("job_id")
    snapshot = reimport_job_snapshot(job_id)
    if snapshot is None or snapshot["dataset_id"] != dataset_id:
        return "Bereit.", "Noch kein Lauf gestartet.", False, True

    status = snapshot["status"]
    started = snapshot.get("started_at") or "unknown"
    finished = snapshot.get("finished_at")
    logs = "\n".join(snapshot["logs"]) or "Starting..."
    if status == "running":
        return f"Laeuft seit {started}.", logs, True, False
    if status == "done":
        return f"Fertig. Gestartet {started}, beendet {finished}.", logs, False, True
    return f"Fehlgeschlagen. Gestartet {started}, beendet {finished}.", logs, False, True


@app.callback(
    Output("live-refresh-interval", "interval"),
    Output("live-refresh-interval", "disabled"),
    Input("live-refresh-dropdown", "value"),
)
def update_live_refresh(seconds):
    seconds = int(seconds or 0)
    return max(seconds, 1) * 1000, seconds <= 0


def ram_label_fields(logs: list[dict], selected_paths, field_type: str, label_mode: str | None):
    selected = set(selected_paths or [])
    return [
        html.Div(
            className="ram-label-field",
            children=[
                html.Label(log.get("short_label", log["label"])),
                dcc.Input(
                    id={"type": field_type, "path": log["path"]},
                    value=ram_base_label(log, label_mode),
                    type="text",
                    debounce=True,
                ),
            ],
        )
        for log in logs
        if log["path"] in selected
    ]


@app.callback(
    Output("live-log-dropdown", "options"),
    Output("live-log-dropdown", "value"),
    Input("live-experiment-dropdown", "value"),
    Input("live-refresh-interval", "n_intervals"),
    State("live-log-dropdown", "value"),
)
def update_live_log_options(experiment, _n_intervals, current_selection):
    logs = repository.list_ram_logs(experiment) if experiment else []
    options = [{"label": log.get("short_label", log["label"]), "value": log["path"]} for log in logs]
    available = {log["path"] for log in logs}
    if ctx.triggered_id == "live-refresh-interval" and current_selection is not None:
        selected = [path for path in current_selection if path in available]
    else:
        selected = [log["path"] for log in logs]
    return options, selected


@app.callback(
    Output("live-label-fields", "children"),
    Input("live-log-dropdown", "value"),
    Input("label-mode-store", "data"),
    State("live-experiment-dropdown", "value"),
)
def update_live_label_fields(selected_paths, label_mode, experiment):
    logs = repository.list_ram_logs(experiment) if experiment else []
    return ram_label_fields(logs, selected_paths, "live-custom-ram-label", label_mode)


@app.callback(
    Output("live-status", "children"),
    Output("live-card-row", "children"),
    Output("live-progress-timeline", "figure"),
    Output("live-progress-states", "figure"),
    Output("live-ram-usage", "figure"),
    Output("live-ram-duration-timeline", "figure"),
    Output("live-ram-duration-summary", "figure"),
    Input("live-experiment-dropdown", "value"),
    Input("live-ram-mode-dropdown", "value"),
    Input("live-log-dropdown", "value"),
    Input("live-step-prefix-dropdown", "value"),
    Input({"type": "live-custom-ram-label", "path": ALL}, "value"),
    Input({"type": "live-custom-ram-label", "path": ALL}, "id"),
    Input("label-mode-store", "data"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
    Input("live-refresh-interval", "n_intervals"),
)
def update_live_monitor(
    experiment, ram_mode, selected_logs, step_prefix, custom_values, custom_ids,
    label_mode, plot_style, label_scale, _n_intervals,
):
    custom_labels = {
        item["path"]: value
        for item, value in zip(custom_ids or [], custom_values or [])
        if str(value or "").strip()
    }
    if not experiment:
        empty = apply_plot_display(progress_timeline_figure([]), plot_style, label_scale)
        return (
            "No live experiment selected.",
            live_page.status_cards({}),
            empty,
            apply_plot_display(progress_state_figure([]), plot_style, label_scale),
            apply_plot_display(ram_usage_figure([]), plot_style, label_scale),
            apply_plot_display(ram_duration_timeline_figure([]), plot_style, label_scale),
            apply_plot_display(ram_duration_summary_figure([]), plot_style, label_scale),
        )
    summary = repository.live_summary(experiment)
    events = repository.load_progress_events(experiment)
    logs = repository.list_ram_logs(experiment)
    selected = set(selected_logs or [])
    logs = [log for log in logs if log["path"] in selected]
    ram_tables = [(log, repository.load_ram_log(log["path"])) for log in logs]
    status = (
        f"{summary.get('status', 'unknown')} - "
        f"{summary.get('event_count', 0)} progress events, "
        f"{len(logs)}/{summary.get('ram_log_count', 0)} RAM logs visible"
    )
    return (
        status,
        live_page.status_cards(summary),
        apply_plot_display(progress_timeline_figure(events), plot_style, label_scale),
        apply_plot_display(progress_state_figure(events), plot_style, label_scale),
        apply_plot_display(ram_usage_figure(ram_tables, ram_mode or "pipeline", label_mode, custom_labels, step_prefix), plot_style, label_scale),
        apply_plot_display(ram_duration_timeline_figure(ram_tables, label_mode, custom_labels, step_prefix), plot_style, label_scale),
        apply_plot_display(ram_duration_summary_figure(ram_tables, label_mode, custom_labels, step_prefix), plot_style, label_scale),
    )


@app.callback(
    Output("home-multi-mode-store", "data"),
    Input("home-multi-toggle", "n_clicks"),
    State("home-multi-mode-store", "data"),
    prevent_initial_call=True,
)
def toggle_home_multi_mode(_n_clicks, enabled):
    return not bool(enabled)


@app.callback(
    Output("home-card-grid", "children"),
    Output("home-result-summary", "children"),
    Output("home-multi-toggle", "children"),
    Input("pipeline-version-filter", "value"),
    Input("organism-filter", "value"),
    Input("experiment-type-filter", "value"),
    Input("cluster-filter", "value"),
    Input("home-multi-mode-store", "data"),
)
def filter_home_datasets(pipeline_version, organism, experiment_type, cluster, multi_mode):
    pipeline_version = pipeline_version or "__all__"
    organism = organism or "__all__"
    experiment_type = experiment_type or "__all__"
    cluster = cluster or "__all__"
    multi_mode = bool(multi_mode)
    datasets = all_datasets()
    cards = home.dataset_cards(
        datasets,
        pipeline_version,
        organism,
        experiment_type,
        cluster,
        multi_mode=multi_mode,
    )
    version_label = "all versions" if pipeline_version == "__all__" else pipeline_version
    organism_label = "all organisms" if organism == "__all__" else organism
    type_label = "all experiment types" if experiment_type == "__all__" else experiment_type
    cluster_label = "all clusters" if cluster == "__all__" else cluster
    visible_dataset_count = len(cards)
    toggle_label = "Cancel multi experiment" if multi_mode else "Multi experiment"
    return cards, f"{visible_dataset_count} dataset(s), {version_label}, {organism_label}, {type_label}, {cluster_label}", toggle_label


@app.callback(
    Output("home-multi-confirm", "disabled"),
    Output("home-multi-status", "children"),
    Input("home-multi-mode-store", "data"),
    Input({"type": "home-dataset-select", "dataset_id": ALL}, "value"),
    State({"type": "home-dataset-select", "dataset_id": ALL}, "id"),
)
def update_home_multi_status(multi_mode, selection_values, selection_ids):
    multi_mode = bool(multi_mode)
    datasets = all_datasets()
    selected_ids = selected_dataset_ids(selection_values, selection_ids)
    selected_datasets = [datasets[dataset_id] for dataset_id in selected_ids if dataset_id in datasets]
    selected_organisms = {dataset["organism"] for dataset in selected_datasets}
    confirm_disabled = len(selected_datasets) < 2 or len(selected_organisms) != 1
    if not multi_mode:
        status = "Enable multi experiment mode to compare searches across experiments."
        confirm_disabled = True
    elif len(selected_datasets) < 2:
        status = "Select at least two experiments."
    elif len(selected_organisms) != 1:
        status = "Selected experiments must use the same organism."
    else:
        status = f"{len(selected_datasets)} experiments selected for {next(iter(selected_organisms))}."
    return confirm_disabled, status


@app.callback(
    Output("url", "href", allow_duplicate=True),
    Input("home-multi-confirm", "n_clicks"),
    State({"type": "home-dataset-select", "dataset_id": ALL}, "value"),
    State({"type": "home-dataset-select", "dataset_id": ALL}, "id"),
    prevent_initial_call=True,
)
def open_multi_upset(n_clicks, selection_values, selection_ids):
    if not n_clicks:
        return no_update
    dataset_ids = selected_dataset_ids(selection_values, selection_ids)
    datasets = all_datasets()
    selected_datasets = [datasets[dataset_id] for dataset_id in dataset_ids if dataset_id in datasets]
    organisms = {dataset["organism"] for dataset in selected_datasets}
    if len(selected_datasets) < 2 or len(organisms) != 1:
        return no_update
    return f"/multi{multi_search_for_dataset_ids(dataset_ids)}"


@app.callback(
    Output("raw-item-store", "data"),
    Output("all-set-columns-store", "data"),
    Output("upset-base-status-store", "data"),
    Input("upset-source-dropdown", "value"),
    Input("upset-aggregation-source-dropdown", "value"),
    Input("upset-evidence-filter-dropdown", "value"),
    Input("upset-entrapment-dropdown", "value"),
    Input("upset-organism-dropdown", "value"),
    Input("upset-fdr-dropdown", "value"),
    Input("upset-search-filter-dropdown", "value"),
    State("dataset-id-store", "data"),
)
def load_upset_source(
    item_level,
    aggregation_source,
    evidence_filter,
    entrapment_mode,
    organism_filter,
    qvalue_threshold,
    selected_searches,
    dataset_id,
):
    if not dataset_id:
        return [], [], "No dataset selected."
    try:
        include_entrapment = entrapment_mode == "include" and registry.has_entrapment_hits(dataset_id)
        qvalue_threshold = active_qvalue_threshold(dataset_id, qvalue_threshold)
        item_df, set_cols, source_label = registry.build_upset_items(
            dataset_id,
            item_level,
            evidence_filter,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
            aggregation_source=aggregation_source,
            searches=selected_searches,
        )
        item_df = filter_upset_items_by_organism(item_df, organism_filter)
        all_set_cols = list(set_cols)
        item_df = item_df.where(pd.notna(item_df), None)
        entrapment_label = "targets + ENT_" if include_entrapment else "targets only"
        fdr_label = f"DIA-NN Q.Value <= {qvalue_threshold:g}" if qvalue_threshold is not None else "all reported DIA-NN Q.Value"
        aggregation_label = "materialized" if aggregation_source != "on_the_fly" else "on the fly"
        organism_label = "all organisms" if organism_filter in (None, "", "__all__") else str(organism_filter)
        status_prefix = f"{source_label}, {organism_label}, {entrapment_label}, {fdr_label}, {aggregation_label}"
        return item_df.to_dict("records"), all_set_cols, status_prefix
    except Exception as exc:
        return [], [], f"Could not load source: {exc}"


app.clientside_callback(
    """
    function(rawItems, allSetCols, selectedSearches, customLabels, labelMode, statusPrefix) {
        const allSets = Array.isArray(allSetCols) ? allSetCols : [];
        if (!Array.isArray(rawItems) || rawItems.length === 0 || allSets.length === 0) {
            return [[], [], statusPrefix || "No current UpSet items available."];
        }

        const allSetLookup = new Set(allSets);
        let selected = Array.isArray(selectedSearches)
            ? selectedSearches.filter(function (name) { return allSetLookup.has(name); })
            : [];
        if (selected.length === 0) {
            selected = allSets.slice();
        }

        const autoLabelMap = makeAutoLabelMap(selected, labelMode || "short");
        const custom = customLabels || {};
        const reserved = new Set(Object.keys(rawItems[0] || {}).filter(function (key) {
            return !allSetLookup.has(key);
        }));
        const used = new Set();
        const labelMap = {};
        selected.forEach(function (setName) {
            const proposed = String(custom[setName] || "").trim() || autoLabelMap[setName] || setName;
            let label = proposed;
            let suffix = 2;
            while (used.has(label) || reserved.has(label)) {
                label = proposed + " (" + suffix + ")";
                suffix += 1;
            }
            used.add(label);
            labelMap[setName] = label;
        });

        const selectedLookup = new Set(selected);
        const filteredItems = [];
        rawItems.forEach(function (row) {
            let keep = false;
            for (let i = 0; i < selected.length; i += 1) {
                if (Boolean(row[selected[i]])) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                return;
            }
            const next = {};
            Object.keys(row).forEach(function (key) {
                if (!allSetLookup.has(key)) {
                    next[key] = row[key];
                }
            });
            selected.forEach(function (setName) {
                next[labelMap[setName]] = Boolean(row[setName]);
            });
            filteredItems.push(next);
        });

        const setCols = selected.map(function (setName) { return labelMap[setName]; });
        const status = (statusPrefix || "Current source") + ", " + setCols.length + "/" + allSets.length +
            " searches shown: " + filteredItems.length.toLocaleString("en-US") + " entities.";
        return [filteredItems, setCols, status];

        function makeAutoLabelMap(labels, mode) {
            const out = {};
            const usedLabels = {};
            labels.forEach(function (label) {
                const text = displaySearchLabel(String(label || ""), mode);
                const count = (usedLabels[text] || 0) + 1;
                usedLabels[text] = count;
                out[String(label)] = count === 1 ? text : text + " (" + count + ")";
            });
            return out;
        }

        function displaySearchLabel(label, mode) {
            if (mode === "full") {
                return label;
            }
            let clean = label.replace(/^tuned\\//, "").replace(/__/g, "/");
            clean = clean.replace(/\\s*[·-]\\s*\\d{8}\\s+\\d{6}\\s*$/, "");
            clean = clean.replace(/[_\\s-]+\\d{8}[_\\s-]*\\d{6}\\s*$/, "");
            clean = clean.replace(/_/g, " ");
            clean = clean.replace(/\\brt fr lr\\b/ig, "rt+fr");
            clean = clean.replace(/\\brt fr\\b/ig, "rt+fr");
            clean = clean.replace(/\\bfr lr\\b/ig, "fr");
            clean = clean.replace(/\\b0p(\\d+)\\b/g, "0.$1");
            clean = clean.replace(/\\s+/g, " ").trim();
            if (mode === "compact") {
                return clean;
            }
            const lowered = clean.toLowerCase();
            const profile = ["baseline", "balanced", "fast", "slow"].find(function (value) {
                return lowered.indexOf(value) !== -1;
            });
            let model = null;
            if (lowered.indexOf("rt+fr") !== -1) {
                model = "rt+fr";
            } else if (/(^| )rt($| )/.test(lowered)) {
                model = "rt";
            } else if (/(^| )fr($| )/.test(lowered)) {
                model = "fr";
            }
            const qMatch = lowered.match(/(?:q|qvalue)\\s*([.]?\\d+(?:\\.\\d+)?)/);
            const libMatch = lowered.match(/(?:lib|outlibq|out-lib)\\s*([.]?\\d+(?:\\.\\d+)?)/);
            const bits = [];
            if (profile) bits.push(profile);
            if (model) bits.push(model);
            if (qMatch) bits.push("q" + qMatch[1]);
            if (libMatch) bits.push("lib" + libMatch[1]);
            if (bits.length > 0) {
                return bits.join(" ");
            }
            return clean.length <= 28 ? clean : clean.slice(0, 25) + "...";
        }
    }
    """,
    Output("item-store", "data"),
    Output("set-columns-store", "data"),
    Output("upset-source-status", "children"),
    Input("raw-item-store", "data"),
    Input("all-set-columns-store", "data"),
    Input("upset-search-filter-dropdown", "value"),
    Input("custom-label-store", "data"),
    Input("label-mode-store", "data"),
    State("upset-base-status-store", "data"),
)


def filter_upset_searches(item_df: pd.DataFrame, set_cols: list[str], selected_searches) -> tuple[pd.DataFrame, list[str]]:
    selected = [search for search in (selected_searches or []) if search in set_cols]
    if not selected:
        selected = list(set_cols)
    if not selected:
        return item_df, []

    filtered = item_df.copy()
    dropped_cols = [col for col in set_cols if col not in selected and col in filtered.columns]
    if dropped_cols:
        filtered = filtered.drop(columns=dropped_cols)
    existing_selected = [col for col in selected if col in filtered.columns]
    if existing_selected:
        mask = filtered[existing_selected].fillna(False).astype(bool).any(axis=1)
        filtered = filtered.loc[mask].copy()
    return filtered, existing_selected


def effective_search_label_map(set_columns: list[str], label_mode: str | None, custom_labels, reserved_labels: list[str] | None = None) -> dict[str, str]:
    auto_labels = display_search_label_map(set_columns, label_mode)
    custom_labels = custom_labels or {}
    reserved = {str(label) for label in (reserved_labels or [])}
    used: set[str] = set()
    out: dict[str, str] = {}
    for set_name in set_columns:
        custom = str(custom_labels.get(set_name, "")).strip()
        base_label = custom or auto_labels.get(set_name, set_name)
        label = base_label
        suffix = 2
        while label in used or label in reserved:
            label = f"{base_label} ({suffix})"
            suffix += 1
        used.add(label)
        out[set_name] = label
    return out


@app.callback(
    Output("upset-search-filter-dropdown", "options"),
    Output("upset-search-filter-dropdown", "value"),
    Input("available-set-columns-store", "data"),
    Input("custom-label-store", "data"),
    State("upset-search-filter-dropdown", "value"),
    State("label-mode-store", "data"),
)
def update_upset_search_filter_options(set_columns, custom_labels, current_value, label_mode):
    set_columns = set_columns or []
    label_map = effective_search_label_map(set_columns, label_mode, custom_labels)
    options = [{"label": label_map.get(search, search), "value": search} for search in set_columns]
    current = [search for search in (current_value or []) if search in set_columns]
    if current:
        return options, current

    # Start the integrated Carafe2 comparison with its paired DIA-NN baseline.
    # Other searches remain available in the selector for broader comparisons.
    standard_carafe = next(
        (
            search
            for search in set_columns
            if search.startswith("carafe/") and "carafe_standard_baseline_v1" in search
        ),
        None,
    )
    if "baseline" in set_columns and standard_carafe is not None:
        return options, ["baseline", standard_carafe]
    return options, set_columns


@app.callback(
    Output("upset-search-filter-dropdown", "value", allow_duplicate=True),
    Input("upset-search-filter-dropdown", "value"),
    State("available-set-columns-store", "data"),
    prevent_initial_call=True,
)
def keep_upset_search_selection_nonempty(value, set_columns):
    if value:
        return value
    return set_columns or []


@app.callback(
    Output("upset-label-fields", "children"),
    Input("available-set-columns-store", "data"),
    Input("upset-search-filter-dropdown", "value"),
    Input("custom-label-store", "data"),
    State("label-mode-store", "data"),
)
def update_upset_label_fields(set_columns, selected_searches, custom_labels, label_mode):
    set_columns = set_columns or []
    selected_lookup = set(selected_searches or [])
    visible_columns = [set_name for set_name in set_columns if set_name in selected_lookup]
    if not selected_searches:
        visible_columns = list(set_columns)
    custom_labels = custom_labels or {}
    auto_labels = display_search_label_map(set_columns, label_mode)
    if not visible_columns:
        return html.Div("No selected searches available.", className="status-line compact-status")
    return [
        html.Div(
            className="upset-label-row",
            children=[
                html.Div(
                    className="upset-label-row-source",
                    title=set_name,
                    children=[
                        html.Span(auto_labels.get(set_name, set_name), className="upset-label-auto"),
                        html.Span(set_name, className="upset-label-raw"),
                    ],
                ),
                dcc.Input(
                    id={"type": "upset-label-input", "set": set_name},
                    type="text",
                    value=str(custom_labels.get(set_name, "")),
                    placeholder=auto_labels.get(set_name, set_name),
                    debounce=True,
                    className="upset-label-input",
                ),
            ],
        )
        for set_name in visible_columns
    ]


@app.callback(
    Output("custom-label-store", "data"),
    Input("upset-label-save-button", "n_clicks"),
    Input("upset-label-reset-button", "n_clicks"),
    State({"type": "upset-label-input", "set": ALL}, "id"),
    State({"type": "upset-label-input", "set": ALL}, "value"),
    prevent_initial_call=True,
)
def update_custom_label_store(_save_clicks, _reset_clicks, ids, values):
    if ctx.triggered_id == "upset-label-reset-button":
        return {}
    if ctx.triggered_id != "upset-label-save-button":
        return no_update
    labels = {}
    for value, pattern_id in zip(values or [], ids or []):
        set_name = (pattern_id or {}).get("set")
        label = str(value or "").strip()
        if set_name and label:
            labels[set_name] = label
    return labels


app.clientside_callback(
    """
    function(_itemLevel, _aggregationSource, _evidenceFilter, _entrapmentMode, _organismFilter, _qvalueThreshold, _selectedSearches, _customLabels, currentSelection, currentMulti) {
        const hasSingle = Boolean(currentSelection);
        const hasMulti = Array.isArray(currentMulti) && currentMulti.length > 0;
        if (!hasSingle && !hasMulti) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }
        return [null, []];
    }
    """,
    Output("selected-store", "data", allow_duplicate=True),
    Output("multi-selected-store", "data", allow_duplicate=True),
    Input("upset-source-dropdown", "value"),
    Input("upset-aggregation-source-dropdown", "value"),
    Input("upset-evidence-filter-dropdown", "value"),
    Input("upset-entrapment-dropdown", "value"),
    Input("upset-organism-dropdown", "value"),
    Input("upset-fdr-dropdown", "value"),
    Input("upset-search-filter-dropdown", "value"),
    Input("custom-label-store", "data"),
    State("selected-store", "data"),
    State("multi-selected-store", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("upset-entrapment-dropdown", "options"),
    Output("upset-entrapment-dropdown", "value"),
    Output("upset-entrapment-control", "style"),
    Input("dataset-id-store", "data"),
)
def update_entrapment_options(dataset_id):
    options = [{"label": "Exclude ENT_", "value": "exclude"}]
    if dataset_id and registry.has_entrapment_hits(dataset_id):
        options.append({"label": "Include ENT_", "value": "include"})
        return options, "exclude", {}
    return options, "exclude", {"display": "none"}


@app.callback(
    Output("multi-detail-checklist", "options"),
    Output("multi-detail-checklist", "value"),
    Input("upset-entrapment-dropdown", "value"),
    Input("dataset-id-store", "data"),
    Input("upset-source-dropdown", "value"),
)
def update_multi_detail_options(entrapment_mode, dataset_id, item_level):
    if item_level == "protein":
        options = [
            {"label": "Overview", "value": "overview"},
            {"label": "Selected items table", "value": "table"},
        ]
        value = ["overview", "table"]
    else:
        options = list(upset_page.BASE_MULTI_DETAIL_OPTIONS)
        value = list(upset_page.DEFAULT_MULTI_DETAILS)
    if entrapment_mode == "include" and dataset_id and registry.has_entrapment_hits(dataset_id):
        options.append(upset_page.DECOY_DETAIL_OPTION)
        value.append("decoy_fraction")
        if item_level != "protein":
            options.append(upset_page.DECOY_TEMPLE_DETAIL_OPTION)
            value.append("decoy_temple")
    return options, value


@app.callback(
    Output("upset-fdr-dropdown", "options"),
    Output("upset-fdr-dropdown", "value"),
    Output("upset-fdr-control", "style"),
    Input("dataset-id-store", "data"),
)
def update_upset_fdr_options(dataset_id):
    if not dataset_id:
        return [], None, {"display": "none"}
    max_qvalue = registry.max_reported_qvalue(dataset_id)
    if max_qvalue is None:
        return [], None, {"display": "none"}
    options = qvalue_filter_options(max_qvalue)
    default_value = default_qvalue_filter(max_qvalue)
    style = {} if max_qvalue > 0.011 else {"display": "none"}
    return options, default_value, style


@app.callback(
    Output("upset-evidence-filter-dropdown", "value"),
    Input("upset-evidence-filter-dropdown", "value"),
    prevent_initial_call=True,
)
def keep_upset_evidence_selection_nonempty(value):
    if not value:
        return ["dimethyl", "acetyl"]
    if isinstance(value, str):
        return [value]
    return value


@app.callback(
    Output("qc-overview-graph", "figure"),
    Input("dataset-id-store", "data"),
    Input("label-mode-store", "data"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
)
def update_qc_overview(dataset_id, label_mode, plot_style, label_scale):
    if not dataset_id:
        return apply_plot_display(empty_figure("No dataset selected."), plot_style, label_scale)
    fig = search_summary_figure(registry.raw_tables(dataset_id), label_mode)
    return apply_plot_display(fig, plot_style, label_scale)


@app.callback(
    Output("intensity-distribution-graph", "figure"),
    Input("dataset-id-store", "data"),
    Input("label-mode-store", "data"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
)
def update_intensity_distribution(dataset_id, label_mode, plot_style, label_scale):
    if not dataset_id:
        return apply_plot_display(empty_figure("No dataset selected."), plot_style, label_scale)
    fig = intensity_histogram(registry.adapted_tables(dataset_id), registry.raw_tables(dataset_id), label_mode)
    return apply_plot_display(fig, plot_style, label_scale)


@app.callback(
    Output("nterm-overview-graph", "figure"),
    Input("dataset-id-store", "data"),
    Input("label-mode-store", "data"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
)
def update_nterm_overview(dataset_id, label_mode, plot_style, label_scale):
    if not dataset_id:
        return apply_plot_display(empty_figure("No dataset selected."), plot_style, label_scale)
    fig = nterm_overview(registry.termini_tables(dataset_id), label_mode)
    return apply_plot_display(fig, plot_style, label_scale)


def _triplemix_source_signature(dataset_id: str) -> tuple:
    """Track every file that can invalidate the derived Triplemix metrics."""
    experiment_root = repository.experiment_root(dataset_id)
    fasta_path = repository.fasta_path_for_experiment(dataset_id)
    common_paths = [
        path
        for path in (
            fasta_path,
            experiment_root / "triplemix_sample_annotation.tsv",
            experiment_root / "sample_metadata_annotation.tsv",
        )
        if path is not None and path.exists()
    ]
    signature: list[tuple] = [
        ("source", str(path), path.stat().st_mtime_ns)
        for path in common_paths
    ]
    for search in repository.list_searches(dataset_id):
        report_path = repository.triplemix_report_path(dataset_id, search)
        cache_path = repository.triplemix_metrics_path(dataset_id, search)
        signature.append(
            (
                "search",
                search,
                str(report_path),
                report_path.stat().st_mtime_ns if report_path.exists() else -1,
                cache_path.stat().st_mtime_ns if cache_path.exists() else -1,
            )
        )
    return tuple(signature)


@lru_cache(maxsize=16)
def _triplemix_metrics_tables(dataset_id: str, _source_signature: tuple) -> dict[str, pl.DataFrame]:
    """Avoid re-reading unchanged metrics on display-only callbacks."""
    return repository.load_triplemix_metrics(dataset_id)


@lru_cache(maxsize=48)
def _triplemix_figure_templates(
    dataset_id: str,
    search: str,
    evidence_filter: str,
    analysis: str,
    source_signature: tuple,
) -> tuple:
    """Cache expensive, display-style-independent figures per metrics file."""
    metrics = _triplemix_metrics_tables(dataset_id, source_signature).get(search)
    if metrics is None:
        return tuple(empty_figure("No Triplemix A/B data available.") for _ in triplemix_page.PLOT_ROWS)
    plot_data = prepare_triplemix_plot_data(metrics, evidence_filter)
    tables = {search: metrics}
    plot_data_by_search = {search: plot_data}
    return (
        triplemix_identifications_figure(tables, evidence_filter),
        triplemix_completeness_figure(tables, evidence_filter),
        triplemix_ab_scatter_figure(metrics, search, evidence_filter, plot_data, analysis),
        triplemix_ma_figure(metrics, search, evidence_filter, plot_data, analysis),
        triplemix_fc_distribution_figure(tables, evidence_filter, plot_data_by_search, analysis),
        triplemix_modification_figure(metrics, search, evidence_filter),
        triplemix_specificity_figure(metrics, search, evidence_filter),
    )


@lru_cache(maxsize=48)
def _triplemix_raw_figure_template(
    dataset_id: str,
    search: str,
    evidence_filter: str,
    source_signature: tuple,
):
    metrics = _triplemix_metrics_tables(dataset_id, source_signature).get(search)
    if metrics is None:
        return empty_figure("No raw Triplemix A/B data available.")
    plot_data = prepare_triplemix_plot_data(metrics, evidence_filter)
    return triplemix_raw_ma_figure(metrics, search, evidence_filter, plot_data)


def _empty_triplemix_figures(plot_style, label_scale) -> list:
    return [
        apply_plot_display(empty_figure("No Triplemix A/B data available."), plot_style, label_scale)
        for _ in range(1 + 2 * len(triplemix_page.PLOT_ROWS))
    ]


@app.callback(
    Output("triplemix-summary-cards", "children"),
    Output("triplemix-raw-ma-graph", "figure"),
    Output("triplemix-left-identifications-graph", "figure"),
    Output("triplemix-right-identifications-graph", "figure"),
    Output("triplemix-left-completeness-graph", "figure"),
    Output("triplemix-right-completeness-graph", "figure"),
    Output("triplemix-left-ab-scatter-graph", "figure"),
    Output("triplemix-right-ab-scatter-graph", "figure"),
    Output("triplemix-left-ma-graph", "figure"),
    Output("triplemix-right-ma-graph", "figure"),
    Output("triplemix-left-fc-distribution-graph", "figure"),
    Output("triplemix-right-fc-distribution-graph", "figure"),
    Output("triplemix-left-modifications-graph", "figure"),
    Output("triplemix-right-modifications-graph", "figure"),
    Output("triplemix-left-specificity-graph", "figure"),
    Output("triplemix-right-specificity-graph", "figure"),
    Input("dataset-id-store", "data"),
    Input("triplemix-search-dropdown", "value"),
    Input("triplemix-filter-dropdown", "value"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
)
def update_triplemix_page(dataset_id, selected_search, evidence_filter, plot_style, label_scale):
    if not dataset_id:
        return [], *_empty_triplemix_figures(plot_style, label_scale)
    try:
        source_signature = _triplemix_source_signature(dataset_id)
        metrics_by_search = _triplemix_metrics_tables(dataset_id, source_signature)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return [], *_empty_triplemix_figures(plot_style, label_scale)
    if not metrics_by_search:
        return [], *_empty_triplemix_figures(plot_style, label_scale)

    searches = list(metrics_by_search)
    if selected_search not in metrics_by_search:
        selected_search = searches[0]
    selected = metrics_by_search[selected_search]
    filtered = filter_triplemix_metrics(selected, evidence_filter)
    complete = filtered.filter(pl.col("complete_all_replicates") & pl.col("fold_change_normalized").is_finite())
    raw_median_text = "not available"
    if {"raw_complete_all_replicates", "raw_fold_change"}.issubset(filtered.columns):
        raw_complete = filtered.filter(
            pl.col("raw_complete_all_replicates") & pl.col("raw_fold_change").is_finite()
        )
        raw_medians = {
            row["protein_organism"]: row["raw_fold_change"]
            for row in raw_complete.group_by("protein_organism").agg(
                pl.col("raw_fold_change").median()
            ).to_dicts()
        }
        raw_median_text = " · ".join(
            f"{TRIPLEMIX_SHORT_NAMES[organism]} {float(raw_medians[organism]):+.2f}"
            for organism in TRIPLEMIX_ORGANISMS
            if organism in raw_medians and raw_medians[organism] is not None
        ) or raw_median_text
    sample_a = selected.item(0, "samples_A") if selected.height else "n/a"
    sample_b = selected.item(0, "samples_B") if selected.height else "n/a"
    replicates_a = int(selected.item(0, "replicates_A")) if selected.height else 0
    replicates_b = int(selected.item(0, "replicates_B")) if selected.height else 0
    ecoli_centered_offset = selected.item(0, "ecoli_fc_offset") if selected.height else float("nan")
    arabidopsis_reference_center = selected.item(0, "arabidopsis_reference_center") if selected.height else float("nan")
    source_report = selected.item(0, "source_report") if selected.height and "source_report" in selected.columns else "unknown"
    qvalue_threshold = selected.item(0, "source_qvalue_threshold") if selected.height and "source_qvalue_threshold" in selected.columns else float("nan")
    excluded = int(selected.item(0, "excluded_cross_organism_termini")) if selected.height else 0
    cards = [
        (f"Condition A · n={replicates_a}", sample_a),
        (f"Condition B · n={replicates_b}", sample_b),
        (f"Search · {selected_search}", f"{complete.height:,} complete A/B features"),
        ("Normalization A", f"E. coli FC offset {float(ecoli_centered_offset):.3f}"),
        ("Normalization B", f"Arabidopsis reference center {float(arabidopsis_reference_center):.3f}"),
        ("Input audit", f"{source_report} · Q≤{float(qvalue_threshold):g} · non-proteotypic included"),
        ("Ambiguous organism mappings excluded", f"{excluded:,}"),
        ("Raw observed FC medians", raw_median_text),
        ("QMD targets · provisional", "Normalization A/B targets are provisional"),
    ]
    summary = [
        html.Div(
            className="metric-card",
            children=[html.Div(label, className="metric-card-label"), html.Div(value, className="metric-card-value")],
        )
        for label, value in cards
    ]
    filter_key = evidence_filter or "all"
    left_figures = _triplemix_figure_templates(dataset_id, selected_search, filter_key, "ecoli_centered", source_signature)
    right_figures = _triplemix_figure_templates(dataset_id, selected_search, filter_key, "arabidopsis_reference", source_signature)
    raw_figure = copy.deepcopy(
        _triplemix_raw_figure_template(dataset_id, selected_search, filter_key, source_signature)
    )
    figures = [raw_figure, *[
        copy.deepcopy(figure)
        for pair in zip(left_figures, right_figures, strict=True)
        for figure in pair
    ]]
    return summary, *[apply_plot_display(figure, plot_style, label_scale) for figure in figures]


@app.callback(
    Output("ram-step-dropdown", "options"),
    Output("ram-step-dropdown", "value"),
    Input("dataset-id-store", "data"),
)
def update_ram_steps(dataset_id):
    if not dataset_id:
        return [], None
    logs = repository.list_ram_logs(dataset_id)
    steps = sorted({log["step"] for log in logs})
    options = [{"label": "All steps", "value": "__all__"}] + [{"label": step, "value": step} for step in steps]
    return options, "__all__" if logs else None


@app.callback(
    Output("ram-log-dropdown", "options"),
    Output("ram-log-dropdown", "value"),
    Input("ram-step-dropdown", "value"),
    State("dataset-id-store", "data"),
)
def update_ram_logs(step, dataset_id):
    if not dataset_id:
        return [], []
    logs = repository.list_ram_logs(dataset_id)
    if step and step != "__all__":
        logs = [log for log in logs if log["step"] == step]
    options = [
        {"label": log.get("short_label", log["label"]), "value": log["path"]} for log in logs
    ]
    return options, [log["path"] for log in logs]


@app.callback(
    Output("ram-label-fields", "children"),
    Input("ram-log-dropdown", "value"),
    Input("label-mode-store", "data"),
    State("dataset-id-store", "data"),
)
def update_ram_label_fields(selected_paths, label_mode, dataset_id):
    logs = repository.list_ram_logs(dataset_id) if dataset_id else []
    return ram_label_fields(logs, selected_paths, "ram-custom-label", label_mode)


@app.callback(
    Output("ram-usage-graph", "figure"),
    Output("ram-summary-graph", "figure"),
    Output("ram-duration-timeline-graph", "figure"),
    Output("ram-duration-summary-graph", "figure"),
    Output("ram-status", "children"),
    Input("ram-step-dropdown", "value"),
    Input("ram-log-dropdown", "value"),
    Input("ram-mode-dropdown", "value"),
    Input("ram-step-prefix-dropdown", "value"),
    Input({"type": "ram-custom-label", "path": ALL}, "value"),
    Input({"type": "ram-custom-label", "path": ALL}, "id"),
    Input("label-mode-store", "data"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
    State("dataset-id-store", "data"),
)
def update_ram_figures(
    step, selected_logs, mode, step_prefix, custom_values, custom_ids,
    label_mode, plot_style, label_scale, dataset_id,
):
    started = time.perf_counter()
    custom_labels = {
        item["path"]: value
        for item, value in zip(custom_ids or [], custom_values or [])
        if str(value or "").strip()
    }
    if not dataset_id:
        empty = apply_plot_display(ram_usage_figure([], mode or "pipeline", label_mode), plot_style, label_scale)
        return (
            empty,
            apply_plot_display(ram_summary_figure([], label_mode), plot_style, label_scale),
            apply_plot_display(ram_duration_timeline_figure([], label_mode), plot_style, label_scale),
            apply_plot_display(ram_duration_summary_figure([], label_mode), plot_style, label_scale),
            "No dataset selected.",
        )
    logs = repository.list_ram_logs(dataset_id)
    if step and step != "__all__":
        logs = [log for log in logs if log["step"] == step]
    selected = set(selected_logs or [])
    logs = [log for log in logs if log["path"] in selected]
    load_started = time.perf_counter()
    ram_tables = [(log, repository.load_ram_log(log["path"])) for log in logs]
    loaded = time.perf_counter()
    n_points = sum(len(df) for _log, df in ram_tables)
    usage_fig = apply_plot_display(ram_usage_figure(ram_tables, mode or "pipeline", label_mode, custom_labels, step_prefix), plot_style, label_scale)
    summary_fig = apply_plot_display(ram_summary_figure(ram_tables, label_mode, custom_labels, step_prefix), plot_style, label_scale)
    timeline_fig = apply_plot_display(ram_duration_timeline_figure(ram_tables, label_mode, custom_labels, step_prefix), plot_style, label_scale)
    duration_fig = apply_plot_display(ram_duration_summary_figure(ram_tables, label_mode, custom_labels, step_prefix), plot_style, label_scale)
    finished = time.perf_counter()
    return (
        usage_fig,
        summary_fig,
        timeline_fig,
        duration_fig,
        (
            f"{len(ram_tables)} log(s), {n_points:,} time points "
            f"- load {loaded - load_started:.2f}s, plots {finished - loaded:.2f}s, total {finished - started:.2f}s"
        ),
    )


@app.callback(
    Output("entrapment-level-dropdown", "options"),
    Output("entrapment-level-dropdown", "value"),
    Input("entrapment-aggregation-dropdown", "value"),
)
def update_entrapment_level_options(aggregation_mode):
    if aggregation_mode == "diann":
        return entrapment_page.ENTRAPMENT_LEVEL_OPTIONS, ["precursor", "peptidoform", "peptide", "protein_group"]
    return entrapment_page.NTERM_ENTRAPMENT_LEVEL_OPTIONS, ["nterm_peptide", "nterm_site"]


@app.callback(
    Output("entrapment-summary-graph", "figure"),
    Output("entrapment-curve-graph", "figure"),
    Output("entrapment-modification-summary-graph", "figure"),
    Output("entrapment-modification-curve-graph", "figure"),
    Output("entrapment-level-status", "children"),
    Input("entrapment-aggregation-dropdown", "value"),
    Input("entrapment-level-dropdown", "value"),
    Input("entrapment-threshold-dropdown", "value"),
    Input("entrapment-curve-grouping-dropdown", "value"),
    Input("entrapment-run-dropdown", "value"),
    Input({"type": "entrapment-custom-label", "run": ALL}, "value"),
    Input({"type": "entrapment-custom-label", "run": ALL}, "id"),
    Input("label-mode-store", "data"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
    State("dataset-id-store", "data"),
)
def update_entrapment_page(
    aggregation_mode,
    selected_levels,
    qvalue_threshold,
    curve_grouping,
    selected_runs,
    custom_label_values,
    custom_label_ids,
    label_mode,
    plot_style,
    label_scale,
    dataset_id,
):
    selected_runs = list(selected_runs or [])
    custom_labels = {
        pattern_id["run"]: str(value).strip()
        for pattern_id, value in zip(custom_label_ids or [], custom_label_values or [])
        if str(value or "").strip()
    }
    aggregation_mode = aggregation_mode or "nterm"
    allowed_levels = {
        option["value"]
        for option in (
            entrapment_page.ENTRAPMENT_LEVEL_OPTIONS
            if aggregation_mode == "diann"
            else entrapment_page.NTERM_ENTRAPMENT_LEVEL_OPTIONS
        )
    }
    if isinstance(selected_levels, str):
        selected_levels = [selected_levels]
    levels = [level for level in (selected_levels or []) if level in allowed_levels]
    if not levels:
        levels = ["peptide"] if aggregation_mode == "diann" else ["nterm_peptide", "nterm_site"]
    qvalue_threshold = normalize_qvalue_threshold(qvalue_threshold) or 0.01
    split_curve_groups = (curve_grouping or "split") == "split"
    if not dataset_id:
        empty = apply_plot_display(empty_figure("No dataset selected."), plot_style, label_scale)
        return empty, empty, empty, empty, "No dataset selected."
    summary_frames = []
    curve_frames = []
    modification_summary_frames = []
    modification_curve_frames = []

    split_levels = levels
    all_evidence_levels: list[str] = []
    if aggregation_mode == "nterm":
        split_levels = [level for level in levels if level in {"nterm_peptide", "nterm_site"}]
        all_evidence_levels = [level for level in levels if level in {"nterm_peptide", "nterm_site", "nterm_precursor", "nterm_protein"}]

    for specificity_group in ("tryptic", "semitryptic"):
        if split_levels:
            summary = repository.entrapment_summary_by_specificity(
                dataset_id,
                specificity_group,
                levels=tuple(split_levels),
                qvalue_threshold=qvalue_threshold,
            )
            if not summary.is_empty():
                summary_frames.append(summary.with_columns(pl.lit(specificity_group).alias("specificity_group")))
        for level in split_levels:
            curve = repository.entrapment_qvalue_curve_by_specificity(dataset_id, specificity_group, level=level)
            if not curve.is_empty():
                curve_frames.append(curve.with_columns(pl.lit(specificity_group).alias("specificity_group")))

    if all_evidence_levels:
        summary = repository.entrapment_summary_all_evidence(
            dataset_id,
            levels=tuple(all_evidence_levels),
            qvalue_threshold=qvalue_threshold,
        )
        if not summary.is_empty():
            summary_frames.append(summary.with_columns(pl.lit("all evidence").alias("specificity_group")))
        for level in all_evidence_levels:
            curve = repository.entrapment_qvalue_curve_all_evidence(dataset_id, level=level)
            if not curve.is_empty():
                curve_frames.append(curve.with_columns(pl.lit("all evidence").alias("specificity_group")))

    modification_levels = []
    if aggregation_mode == "nterm":
        modification_levels = [level for level in levels if level in {"nterm_peptide", "nterm_site"}]
    for modification_group in ("free", "acetyl", "dimethyl"):
        if modification_levels:
            summary = repository.entrapment_summary_by_modification(
                dataset_id,
                modification_group,
                levels=tuple(modification_levels),
                qvalue_threshold=qvalue_threshold,
            )
            if not summary.is_empty():
                modification_summary_frames.append(summary.with_columns(pl.lit(modification_group).alias("specificity_group")))
        for level in modification_levels:
            curve = repository.entrapment_qvalue_curve_by_modification(dataset_id, modification_group, level=level)
            if not curve.is_empty():
                modification_curve_frames.append(curve.with_columns(pl.lit(modification_group).alias("specificity_group")))

    summary = pl.concat(summary_frames, how="diagonal_relaxed") if summary_frames else pl.DataFrame()
    curve = pl.concat(curve_frames, how="diagonal_relaxed") if curve_frames else pl.DataFrame()
    modification_summary = pl.concat(modification_summary_frames, how="diagonal_relaxed") if modification_summary_frames else pl.DataFrame()
    modification_curve = pl.concat(modification_curve_frames, how="diagonal_relaxed") if modification_curve_frames else pl.DataFrame()
    def only_selected_runs(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty() or "search" not in frame.columns:
            return frame
        return frame.filter(pl.col("search").cast(pl.Utf8).is_in(selected_runs))

    summary = only_selected_runs(summary)
    curve = only_selected_runs(curve)
    modification_summary = only_selected_runs(modification_summary)
    modification_curve = only_selected_runs(modification_curve)
    mode_label = "DIA-NN reported levels" if aggregation_mode == "diann" else "N-terminal evidence levels"
    split_label = "tryptic, semitryptic, and all-evidence curves for peptide/site; precursor/protein shown as all evidence"
    if aggregation_mode == "diann":
        split_label = "tryptic vs semitryptic split is shown where the level can be assigned from peptide evidence"
    modification_note = (
        " N-terminal modification split is shown for nterm_peptide/nterm_site."
        if modification_levels
        else " N-terminal modification split is available only for nterm_peptide/nterm_site."
    )
    return (
        apply_plot_display(
            entrapment_summary_figure(
                summary,
                qvalue_threshold=qvalue_threshold,
                label_mode=label_mode,
                custom_labels=custom_labels,
            ),
            plot_style,
            label_scale,
        ),
        apply_plot_display(
            entrapment_curve_figure(
                curve,
                show_qvalue_reference=True,
                split_groups=split_curve_groups,
                label_mode=label_mode,
                custom_labels=custom_labels,
            ),
            plot_style,
            label_scale,
        ),
        apply_plot_display(
            entrapment_summary_figure(
                modification_summary,
                qvalue_threshold=qvalue_threshold,
                group_label="N-term modification",
                label_mode=label_mode,
                custom_labels=custom_labels,
            ),
            plot_style,
            label_scale,
        ),
        apply_plot_display(
            entrapment_curve_figure(
                modification_curve,
                show_qvalue_reference=True,
                group_label="N-term modification",
                split_groups=split_curve_groups,
                label_mode=label_mode,
                custom_labels=custom_labels,
            ),
            plot_style,
            label_scale,
        ),
        (
            f"{mode_label}; {len(selected_runs)} run(s), {len(levels)} selected level(s); {split_label}; "
            f"curve layout: {'split panels' if split_curve_groups else 'all lines in one plot'}; "
            f"threshold DIA-NN Q.Value <= {qvalue_threshold:g}. "
            "For N-terminal evidence levels, the curve uses row-level DIA-NN Q.Value aggregated as the best evidence per entity."
            f"{modification_note}"
        ),
    )


@app.callback(
    Output("ram-plot-grid", "className"),
    Output("duration-plot-grid", "className"),
    Input("ram-layout-dropdown", "value"),
)
def update_ram_layout(mode):
    mode = mode or "horizontal"
    class_name = f"ram-plot-grid {mode}"
    return class_name, class_name


@app.callback(
    Output("loss-interactive-view", "style"),
    Output("loss-image-view", "style"),
    Input("loss-view-mode", "value"),
)
def update_loss_view(mode):
    if mode == "images":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


@app.callback(
    Output("upset-workspace", "className"),
    Output("single-selection-panel", "style"),
    Output("multi-detail-panel", "className"),
    Input("upset-layout-mode", "value"),
    Input("multi-intersection-mode", "value"),
    Input("multi-detail-toggle", "n_clicks"),
)
def update_upset_layout_mode(mode, multi_mode, toggle_clicks):
    mode = mode or "split"
    if multi_mode == "on":
        collapsed = bool((toggle_clicks or 0) % 2)
        panel_class = "multi-detail-panel collapsed" if collapsed else "multi-detail-panel"
        return f"upset-workspace multi {mode}", {"display": "none"}, panel_class
    return f"upset-workspace {mode}", {}, "multi-detail-panel hidden"


app.clientside_callback(
    """
    function(items, setCols, selected, multiSelected, multiMode, minSize, topN, labelMap, plotStyle, labelScale) {
        minSize = minSize || 1;
        topN = topN || null;

        if (!window.upsetPlot || !window.upsetPlot.makeFigure) {
            return {
                data: [],
                layout: {
                    title: "upsetPlot.js not loaded",
                    annotations: [{
                        text: "dash_plotly_upset/assets/upsetPlot.js was not loaded.",
                        x: 0.5,
                        y: 0.5,
                        xref: "paper",
                        yref: "paper",
                        showarrow: false
                    }]
                }
            };
        }

        return window.upsetPlot.makeFigure({
            items: items || [],
            setCols: setCols || [],
            selected: multiMode === "on" ? null : selected,
            multiSelected: multiMode === "on" ? (multiSelected || []) : [],
            minSize: minSize,
            topN: topN,
            labelMap: labelMap || {},
            plotStyle: plotStyle || "default",
            labelScale: labelScale || 1
        });
    }
    """,
    Output("upset-plot", "figure"),
    Input("item-store", "data"),
    Input("set-columns-store", "data"),
    Input("selected-store", "data"),
    Input("multi-selected-store", "data"),
    Input("multi-intersection-mode", "value"),
    Input("min-size-input", "value"),
    Input("top-n-input", "value"),
    Input("upset-label-map-store", "data"),
    Input("plot-style-store", "data"),
    Input("label-scale-store", "data"),
)


app.clientside_callback(
    """
    function(clickData, clearClicks, multiMode, currentSelection, currentMulti) {
        const triggered = dash_clientside.callback_context.triggered;
        const palette = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626"];

        if (triggered.length > 0) {
            const propId = triggered[0].prop_id;
            if (propId === "clear-selection-button.n_clicks") {
                return [null, []];
            }
        }

        if (!clickData || !clickData.points || clickData.points.length === 0) {
            return [currentSelection || null, currentMulti || []];
        }

        const point = clickData.points[0];
        const cd = point.customdata;

        if (!cd || cd.length < 3) {
            return [currentSelection || null, currentMulti || []];
        }

        const selection = {
            kind: String(cd[0]),
            id: String(cd[1]),
            label: String(cd[2])
        };

        if (multiMode !== "on") {
            return [selection, currentMulti || []];
        }

        let selections = Array.isArray(currentMulti) ? currentMulti.slice() : [];
        const existingIndex = selections.findIndex(function (entry) {
            return entry && entry.kind === selection.kind && entry.id === selection.id;
        });
        if (existingIndex >= 0) {
            selections.splice(existingIndex, 1);
            selections = selections.map(function (entry, index) {
                return Object.assign({}, entry, {slot: index + 1, color: palette[index]});
            });
            return [currentSelection || null, selections];
        }
        if (selections.length >= 5) {
            selections = selections.slice(1);
        }
        selections.push(Object.assign({}, selection, {slot: selections.length + 1, color: palette[selections.length]}));
        selections = selections.map(function (entry, index) {
            return Object.assign({}, entry, {slot: index + 1, color: palette[index]});
        });
        return [currentSelection || null, selections];
    }
    """,
    Output("selected-store", "data", allow_duplicate=True),
    Output("multi-selected-store", "data", allow_duplicate=True),
    Input("upset-plot", "clickData"),
    Input("clear-selection-button", "n_clicks"),
    State("multi-intersection-mode", "value"),
    State("selected-store", "data"),
    State("multi-selected-store", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("multi-selected-store", "data", allow_duplicate=True),
    Input({"type": "remove-multi-selection", "key": ALL}, "n_clicks"),
    State("multi-selected-store", "data"),
    prevent_initial_call=True,
)
def remove_multi_selection(_clicks, current_multi):
    if not any(int(click or 0) > 0 for click in (_clicks or [])):
        return no_update
    triggered_id = ctx.triggered_id
    if not isinstance(triggered_id, dict):
        return no_update
    key = triggered_id.get("key")
    if not key:
        return no_update
    selections = [entry for entry in (current_multi or []) if selection_key(entry) != key]
    palette = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626"]
    return [
        {**entry, "slot": index + 1, "color": palette[index]}
        for index, entry in enumerate(selections[:5])
    ]


@app.callback(
    Output("multi-selection-edit-store", "data"),
    Output("multi-selected-store", "data", allow_duplicate=True),
    Input({"type": "edit-multi-selection", "key": ALL}, "n_clicks"),
    Input({"type": "save-multi-selection", "key": ALL}, "n_clicks"),
    Input({"type": "cancel-multi-selection", "key": ALL}, "n_clicks"),
    State({"type": "rename-multi-selection", "key": ALL}, "value"),
    State({"type": "rename-multi-selection", "key": ALL}, "id"),
    State("multi-selection-edit-store", "data"),
    State("multi-selected-store", "data"),
    prevent_initial_call=True,
)
def edit_multi_selection_name(_edit_clicks, _save_clicks, _cancel_clicks, label_values, label_ids, edit_state, current_multi):
    triggered_id = ctx.triggered_id
    if not isinstance(triggered_id, dict):
        return no_update, no_update
    key = triggered_id.get("key")
    action = triggered_id.get("type")
    if not key or action not in {"edit-multi-selection", "save-multi-selection", "cancel-multi-selection"}:
        return no_update, no_update

    next_edit_state = dict(edit_state or {})
    if action == "edit-multi-selection":
        next_edit_state[key] = True
        return next_edit_state, no_update

    if action == "cancel-multi-selection":
        next_edit_state.pop(key, None)
        return next_edit_state, no_update

    label_by_key = {
        (pattern_id or {}).get("key"): str(value or "").strip()
        for value, pattern_id in zip(label_values or [], label_ids or [])
        if (pattern_id or {}).get("key")
    }
    custom_label = label_by_key.get(key, "")
    renamed = []
    changed = False
    for entry in current_multi or []:
        if selection_key(entry) != key:
            renamed.append(entry)
            continue
        next_entry = dict(entry)
        if custom_label:
            next_entry["custom_label"] = custom_label
        else:
            next_entry.pop("custom_label", None)
        changed = changed or next_entry != entry
        renamed.append(next_entry)
    next_edit_state.pop(key, None)
    return next_edit_state, (renamed if changed else no_update)


@app.callback(
    Output("selected-title", "children"),
    Output("selected-items-table", "columns"),
    Output("selected-items-table", "data"),
    Output("selected-items-plot", "figure"),
    Output("selected-specificity-plot", "figure"),
    Output("selected-modification-plot", "figure"),
    Output("selected-termini-temple-plot", "figure"),
    Output("selected-rt-plot", "figure"),
    Output("selected-rt-modification-plot", "figure"),
    Output("selected-rt-modification-residual-plot", "figure"),
    Output("selected-mass-error-plot", "figure"),
    Output("selected-peptide-properties-plot", "figure"),
    Output("selected-decoy-fraction-plot", "figure"),
    Output("selected-decoy-fraction-panel", "style"),
    Output("single-selection-analysis-grid", "style"),
    Output("ms-property-panel", "style"),
    Output("motif-property-panel", "style"),
    Input("selected-store", "data"),
    Input("label-scale-store", "data"),
    State("dataset-id-store", "data"),
    State("item-store", "data"),
    State("set-columns-store", "data"),
    State("upset-source-dropdown", "value"),
    State("upset-entrapment-dropdown", "value"),
    State("upset-evidence-filter-dropdown", "value"),
    State("label-mode-store", "data"),
    State("plot-style-store", "data"),
)
def update_downstream(selected, label_scale, dataset_id, items, set_columns, item_level, entrapment_mode, evidence_filter, label_mode, plot_style):
    if not dataset_id or not items:
        return empty_selection_outputs()

    item_df = pd.DataFrame(items)
    set_columns = set_columns or []
    if selected:
        kind = selected.get("kind")
        selected_id = selected.get("id")
        label = selected.get("label")
    else:
        kind = None
        selected_id = None
        label = None

    if not selected:
        selected_items = item_df
        title = f"All current UpSet items ({len(selected_items)} items)"
        plot_title = "All current UpSet items"
    elif kind == "intersection":
        selected_items = selected_upset_items(item_df, set_columns, selected)
        title = f"Selected intersection: {label} ({len(selected_items)} items)"
        plot_title = f"Intersection: {label}"
    elif kind == "set" and selected_id in set_columns:
        selected_items = selected_upset_items(item_df, set_columns, selected)
        title = f"Selected set: {selected_id} ({len(selected_items)} items)"
        plot_title = f"Set: {selected_id}"
    else:
        return empty_selection_outputs()

    if "is_decoy" not in selected_items.columns and "is_entrapment" in selected_items.columns:
        selected_items = selected_items.copy()
        selected_items["is_decoy"] = selected_items["is_entrapment"]

    plot_set_columns = [] if selected else set_columns
    table_cols = selected_table_columns_for_level(item_level)
    table_cols = [
        col
        for col in table_cols
        if col in selected_items.columns
    ]
    table_columns = [{"name": readable_selected_column(col), "id": col} for col in table_cols]
    if item_level == "protein":
        hidden_style = {"display": "none"}
        protein_message = empty_figure("Protein-level items aggregate multiple peptide and N-terminal evidence rows.")
        protein_message = apply_plot_display(protein_message, plot_style, label_scale)
        show_decoy = entrapment_mode == "include" and "is_decoy" in selected_items.columns
        return (
            title,
            table_columns,
            selected_items[table_cols].to_dict("records"),
            apply_plot_display(selected_items_plot(selected_items, plot_title), plot_style, label_scale),
            protein_message,
            protein_message,
            protein_message,
            protein_message,
            protein_message,
            protein_message,
            protein_message,
            protein_message,
            apply_plot_display(
                selected_decoy_fraction_plot(selected_items, [], None, label_mode) if show_decoy else selected_decoy_fraction_plot(pd.DataFrame(), []),
                plot_style,
                label_scale,
            ),
            {} if show_decoy else hidden_style,
            hidden_style,
            hidden_style,
            hidden_style,
        )

    return (
        title,
        table_columns,
        selected_items[table_cols].to_dict("records"),
        apply_plot_display(selected_items_plot(selected_items, plot_title), plot_style, label_scale),
        apply_plot_display(selected_specificity_plot(selected_items, plot_set_columns, "full"), plot_style, label_scale),
        apply_plot_display(selected_modification_plot(selected_items, plot_set_columns, evidence_filter, "full"), plot_style, label_scale),
        apply_plot_display(selected_termini_temple_plot(selected_items, plot_set_columns, evidence_filter, "full"), plot_style, label_scale),
        apply_plot_display(selected_rt_plot(selected_items), plot_style, label_scale),
        apply_plot_display(selected_rt_modification_plot(selected_items), plot_style, label_scale),
        apply_plot_display(selected_rt_modification_residual_plot(selected_items), plot_style, label_scale),
        apply_plot_display(selected_mass_error_plot(selected_items), plot_style, label_scale),
        apply_plot_display(selected_peptide_properties_plot(selected_items), plot_style, label_scale),
        apply_plot_display(selected_decoy_fraction_plot(pd.DataFrame(), []), plot_style, label_scale),
        {"display": "none"},
        {},
        {},
        {},
    )


@app.callback(
    Output("multi-selection-strip", "children"),
    Output("multi-selection-results", "children"),
    Output("multi-selection-results", "className"),
    Output("multi-detail-status", "children"),
    Input("multi-selected-store", "data"),
    Input("multi-selection-edit-store", "data"),
    Input("multi-intersection-mode", "value"),
    Input("multi-detail-checklist", "value"),
    Input("label-scale-store", "data"),
    State("item-store", "data"),
    State("set-columns-store", "data"),
    State("upset-source-dropdown", "value"),
    State("upset-evidence-filter-dropdown", "value"),
    State("upset-entrapment-dropdown", "value"),
    State("upset-fdr-dropdown", "value"),
    State("dataset-id-store", "data"),
    State("label-mode-store", "data"),
    State("plot-style-store", "data"),
)
def update_multi_selection_columns(
    multi_selected,
    edit_state,
    multi_mode,
    visible_details,
    label_scale,
    items,
    set_columns,
    item_level,
    evidence_filter,
    entrapment_mode,
    qvalue_threshold,
    dataset_id,
    label_mode,
    plot_style,
):
    if multi_mode != "on":
        return [], [], "multi-selection-results", "Multi mode is off."
    if not dataset_id or not items:
        return [], [], "multi-selection-results", "No current UpSet items available."

    selections = [
        entry
        for entry in (multi_selected or [])
        if entry and entry.get("kind") in {"intersection", "set"}
    ][:5]
    if not selections:
        hint = html.Div(
            "Click up to five intersections or set-size bars. Click a selected item again to remove it.",
            className="multi-selection-hint",
        )
        return hint, [], "multi-selection-results", "No intersections selected."

    visible_details = visible_details or []
    item_df = pd.DataFrame(items)
    set_columns = set_columns or []
    table_cols = [col for col in selected_table_columns_for_level(item_level) if col in item_df.columns]

    chips = [
        html.Div(
            className="multi-selection-chip",
            style={"borderColor": selection.get("color", "#111111")},
            title=selection_full_label(selection),
            children=[
                html.Span(className="multi-color-dot", style={"backgroundColor": selection.get("color", "#111111")}),
                html.Span(selection_short_label(selection, index), className="multi-selection-short-label"),
            ],
        )
        for index, selection in enumerate(selections)
    ]

    comparison_children = []
    column_details = [detail for detail in visible_details if detail not in {"specificity", "modification"}]
    if item_level == "protein":
        column_details = visible_details

    if item_level != "protein" and "specificity" in visible_details:
        specificity_entries = [
            (
                selection_short_label(selection, index),
                selection_full_label(selection),
                selected_upset_items(item_df, set_columns, selection),
            )
            for index, selection in enumerate(selections)
        ]
        comparison_children.append(
            html.Div(
                className="multi-comparison-plot",
                children=followup_graph(
                    figure=selected_category_comparison_plot(
                        specificity_entries,
                        ["specificity"],
                        title="Termini specificity composition by selection",
                        empty_message="No specificity column for these selections.",
                    ),
                    plot_style=plot_style,
                    label_scale=label_scale,
                    filename="multi_selection_specificity_composition",
                ),
            )
        )

    if item_level != "protein" and "modification" in visible_details:
        active_evidence = {evidence_filter} if isinstance(evidence_filter, str) else set(evidence_filter or [])
        modification_order = ["acetyl", "free", "dimethyl"] if "free" in active_evidence else ["acetyl", "dimethyl"]
        modification_entries = [
            (
                selection_short_label(selection, index),
                selection_full_label(selection),
                selected_upset_items(item_df, set_columns, selection),
            )
            for index, selection in enumerate(selections)
        ]
        comparison_children.append(
            html.Div(
                className="multi-comparison-plot",
                children=followup_graph(
                    figure=selected_category_comparison_plot(
                        modification_entries,
                        ["nterm_status", "nterm_modif"],
                        title="N-term modification composition by selection",
                        empty_message="No N-term modification columns for these selections.",
                        category_mapper=nterm_site_modification_class,
                        category_order=modification_order,
                        allowed_categories=set(modification_order),
                    ),
                    plot_style=plot_style,
                    label_scale=label_scale,
                    filename="multi_selection_nterm_modification_composition",
                ),
            )
        )

    columns = []
    counts = []
    for index, selection in enumerate(selections):
        selected_df = selected_upset_items(item_df, set_columns, selection)
        counts.append(len(selected_df))
        columns.append(
            multi_selection_column(
                selection,
                selected_df,
                item_df,
                set_columns,
                table_cols,
                column_details,
                index,
                dataset_id,
                item_level,
                evidence_filter,
                entrapment_mode,
                qvalue_threshold,
                label_mode,
                plot_style,
                label_scale,
            )
        )

    selection_count = min(max(len(selections), 1), 5)
    class_name = f"multi-selection-results count-{selection_count}"
    if len(selections) == 1:
        class_name += " one"
    elif len(selections) == 2:
        class_name += " two"
    else:
        class_name += " many"
    status = " | ".join(
        f"{selection_short_label(selection, index)}: {counts[index]:,} items"
        for index, selection in enumerate(selections)
    )
    column_header_row = html.Section(
        className=f"multi-selection-column-headers count-{selection_count}",
        children=[
            multi_selection_header_cell(selection, index, counts[index], edit_state or {})
            for index, selection in enumerate(selections)
        ],
    )
    if comparison_children:
        columns = [column_header_row, html.Section(className="multi-comparison-row", children=comparison_children), *columns]
    else:
        columns = [column_header_row, *columns]
    return chips, columns, class_name, status


def selection_short_label(selection, index: int) -> str:
    custom_label = str(selection.get("custom_label") or "").strip()
    if custom_label:
        return custom_label
    slot = selection.get("slot", index + 1)
    return f"S{slot}"


def selection_full_label(selection) -> str:
    return str(selection.get("label") or selection.get("id") or "intersection")


def multi_selection_header_cell(selection, index: int, count: int, edit_state: dict):
    color = selection.get("color", "#111111")
    full_label = selection_full_label(selection)
    default_label = f"S{selection.get('slot', index + 1)}"
    current_label = selection_short_label(selection, index)
    key = selection_key(selection)
    kind_label = "Set" if selection.get("kind") == "set" else "Intersection"
    name_control = (
        html.Div(
            className="multi-selection-name-row editing",
            children=[
                dcc.Input(
                    id={"type": "rename-multi-selection", "key": key},
                    type="text",
                    value=str(selection.get("custom_label") or current_label),
                    placeholder=default_label,
                    className="multi-selection-name-input",
                ),
                html.Button("💾", id={"type": "save-multi-selection", "key": key}, n_clicks=0, className="icon-action-button save", title="Save name"),
                html.Button("×", id={"type": "cancel-multi-selection", "key": key}, n_clicks=0, className="icon-action-button cancel", title="Discard name edit"),
            ],
        )
        if edit_state.get(key)
        else html.Div(
            className="multi-selection-name-row",
            children=[
                html.Span(current_label, className="multi-selection-current-name", title=current_label),
                html.Button("✎", id={"type": "edit-multi-selection", "key": key}, n_clicks=0, className="icon-action-button edit", title="Edit name"),
            ],
        )
    )
    return html.Div(
        className="multi-column-header-cell",
        style={"borderLeftColor": color},
        title=full_label,
        children=[
            html.Span(className="multi-color-dot", style={"backgroundColor": color}),
            html.Div(
                className="multi-column-title",
                children=[
                    name_control,
                    html.Div(kind_label, className="muted"),
                    html.Div(full_label, className="multi-column-full-label"),
                    html.Div(f"{count:,} items", className="muted"),
                ],
            ),
            html.Button(
                "Remove",
                id={"type": "remove-multi-selection", "key": selection_key(selection)},
                n_clicks=0,
                className="multi-column-remove",
            ),
        ],
    )


def multi_selection_column(
    selection,
    selected_df,
    item_df,
    set_columns,
    table_cols,
    visible_details,
    index,
    dataset_id,
    item_level,
    evidence_filter,
    entrapment_mode,
    qvalue_threshold,
    label_mode,
    plot_style,
    label_scale,
):
    if item_level == "protein":
        allowed_details = {"overview", "table"}
        if entrapment_mode == "include":
            allowed_details.add("decoy_fraction")
        visible_details = [detail for detail in visible_details if detail in allowed_details]

    if "is_decoy" not in selected_df.columns and "is_entrapment" in selected_df.columns:
        selected_df = selected_df.copy()
        selected_df["is_decoy"] = selected_df["is_entrapment"]

    short_label = selection_short_label(selection, index)
    children = []

    plot_builders = {
        "overview": lambda: selected_items_plot(selected_df, f"Overview - {short_label}"),
        "specificity": lambda: selected_specificity_plot(selected_df, [], label_mode),
        "modification": lambda: selected_modification_plot(selected_df, [], evidence_filter, label_mode),
        "termini_temple": lambda: selected_termini_temple_plot(selected_df, [], evidence_filter, label_mode),
        "decoy_fraction": lambda: selected_decoy_fraction_plot(selected_df, [], normalize_qvalue_threshold(qvalue_threshold), label_mode),
        "decoy_temple": lambda: selected_decoy_temple_plot(selected_df),
        "rt": lambda: selected_rt_plot(selected_df),
        "mass_error": lambda: selected_mass_error_plot(selected_df),
        "peptide_properties": lambda: selected_peptide_properties_plot(selected_df),
    }
    for plot_id in visible_details:
        if plot_id in plot_builders:
            children.append(
                followup_graph(
                    figure=plot_builders[plot_id](),
                    plot_style=plot_style,
                    label_scale=label_scale,
                    filename=f"multi_selection_{plot_id}_{index + 1}",
                    className="multi-column-plot",
                )
            )
            if plot_id == "rt":
                children.append(
                    followup_graph(
                        figure=selected_rt_modification_plot(selected_df),
                        plot_style=plot_style,
                        label_scale=label_scale,
                        filename=f"multi_selection_rt_modifications_{index + 1}",
                        className="multi-column-plot",
                    )
                )
                children.append(
                    followup_graph(
                        figure=selected_rt_modification_residual_plot(selected_df),
                        plot_style=plot_style,
                        label_scale=label_scale,
                        filename=f"multi_selection_rt_modification_residuals_{index + 1}",
                        className="multi-column-plot",
                    )
                )
        elif plot_id == "ms_property_followup":
            children.append(
                build_multi_ms_property_section(
                    selection,
                    selected_df,
                    item_df,
                    set_columns,
                    dataset_id,
                    item_level,
                    evidence_filter,
                    entrapment_mode,
                    qvalue_threshold,
                    plot_style,
                    label_scale,
                )
            )
        elif plot_id == "position_motif":
            children.append(
                build_multi_position_motif_section(
                    selection,
                    selected_df,
                    item_df,
                    set_columns,
                    dataset_id,
                    item_level,
                    evidence_filter,
                    entrapment_mode,
                    qvalue_threshold,
                    plot_style,
                    label_scale,
                )
            )

    if "table" in visible_details:
        columns = [{"name": readable_selected_column(col), "id": col} for col in table_cols]
        children.append(
            dash_table.DataTable(
                columns=columns or [{"name": "item_id", "id": "item_id"}],
                data=selected_df[table_cols].to_dict("records") if table_cols else [],
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "Arial, sans-serif",
                    "fontSize": "12px",
                    "padding": "7px",
                    "textAlign": "left",
                    "border": "1px solid #E5E5E5",
                },
                style_header={"backgroundColor": "#F5F5F5", "fontWeight": "bold", "border": "1px solid #D0D0D0"},
            )
        )

    return html.Section(className="multi-selection-column", children=children)


def build_multi_ms_property_section(
    selection,
    selected_df,
    item_df,
    set_columns,
    dataset_id,
    item_level,
    evidence_filter,
    entrapment_mode,
    qvalue_threshold,
    plot_style=None,
    label_scale=None,
):
    selected_ids = set(selected_df["item_id"].dropna().astype(str)) if "item_id" in selected_df.columns else set()
    if not selected_ids:
        return html.Div("No selected items for MS property follow-up.", className="status-line compact-status")

    include_entrapment = entrapment_mode == "include" and registry.has_entrapment_hits(dataset_id)
    evidence_df = cached_upset_evidence_rows(
        dataset_id,
        item_level,
        evidence_filter,
        include_entrapment=include_entrapment,
        qvalue_threshold=normalize_qvalue_threshold(qvalue_threshold),
    )
    if evidence_df.empty or "item_id" not in evidence_df.columns:
        return html.Div("No supporting peptide evidence rows available.", className="status-line compact-status")

    selected_evidence = evidence_df[evidence_df["item_id"].astype(str).isin(selected_ids)].copy()
    if selected_evidence.empty:
        return html.Div("No supporting peptide evidence rows for this selection.", className="status-line compact-status")
    selected_evidence["group"] = "Selected"
    plot_df = add_ms_properties(selected_evidence)
    plot_ids = [
        "peptide_length",
        "molecular_weight",
        "gravy",
        "pI",
        "c_terminal_residue",
        "missed_cleavages",
        "internal_rk_counts",
    ]
    return html.Details(
        className="multi-followup-section",
        open=False,
        children=[
            html.Summary("MS property follow-up"),
            html.Div(f"Selected n={len(selected_evidence):,} peptide evidence rows.", className="status-line compact-status"),
            html.Div(
                className="property-plot-grid",
                children=[
                    html.Div(
                        className="followup-plot-card",
                        children=followup_graph(
                            ms_property_figure(plot_df, plot_id),
                            f"multi_selection_ms_property_{plot_id}_{selection_key(selection)}",
                            plot_style=plot_style,
                            label_scale=label_scale,
                        ),
                    )
                    for plot_id in plot_ids
                ],
            ),
        ],
    )


def build_multi_position_motif_section(
    selection,
    selected_df,
    item_df,
    set_columns,
    dataset_id,
    item_level,
    evidence_filter,
    entrapment_mode,
    qvalue_threshold,
    plot_style=None,
    label_scale=None,
):
    selected_ids = set(selected_df["item_id"].dropna().astype(str)) if "item_id" in selected_df.columns else set()
    if not selected_ids:
        return html.Div("No selected items for position and motif plots.", className="status-line compact-status")

    include_entrapment = entrapment_mode == "include" and registry.has_entrapment_hits(dataset_id)
    evidence_df = cached_upset_evidence_rows(
        dataset_id,
        item_level,
        evidence_filter,
        include_entrapment=include_entrapment,
        qvalue_threshold=normalize_qvalue_threshold(qvalue_threshold),
    )
    if evidence_df.empty or "item_id" not in evidence_df.columns:
        return html.Div("No supporting N-terminal evidence rows available.", className="status-line compact-status")

    fasta_path = registry.fasta_path_for_dataset(dataset_id)
    sequences = load_fasta_sequences(fasta_path)
    if not sequences:
        return html.Div("No experiment FASTA was found for sequence-based plots.", className="status-line compact-status")

    selected_evidence = evidence_df[evidence_df["item_id"].astype(str).isin(selected_ids)].copy()
    selected_sites = site_table_from_evidence(selected_evidence, sequences)
    if selected_sites.empty:
        return html.Div("No selected N-terminal sites could be mapped to the experiment FASTA.", className="status-line compact-status")
    selected_sites["item_id"] = selected_sites["protein_id"].astype(str) + "|" + selected_sites["start_position"].astype(str)
    selected_sites["group"] = "Selected"
    selected_sites.attrs["logo_mode"] = "selected"
    selected_sites.attrs["background_windows"] = []

    plot_ids = ["relative_nterm_position", "p1_distribution", "p1_prime_distribution", "sequence_logo"]
    valid_windows = int(selected_sites["motif_window"].notna().sum()) if "motif_window" in selected_sites.columns else 0
    return html.Details(
        className="multi-followup-section",
        open=False,
        children=[
            html.Summary("Position & motif plots"),
            html.Div(
                f"Selected {len(selected_sites):,} unique N-terminal site(s), {valid_windows:,} full motif window(s).",
                className="status-line compact-status",
            ),
            html.Div(
                className="property-plot-grid",
                children=[
                    html.Div(
                        className="followup-plot-card",
                        children=followup_graph(
                            position_motif_figure(selected_sites, plot_id),
                            f"multi_selection_position_motif_{plot_id}_{selection_key(selection)}",
                            plot_style=plot_style,
                            label_scale=label_scale,
                        ),
                    )
                    for plot_id in plot_ids
                ],
            ),
        ],
    )


def selection_key(selection) -> str:
    return f"{selection.get('kind', '')}:{selection.get('id', '')}"


def empty_selection_outputs():
    empty = pd.DataFrame()
    return (
        "No selection",
        [{"name": "item_id", "id": "item_id"}],
        [],
        selected_items_plot(empty),
        selected_specificity_plot(empty, []),
        selected_modification_plot(empty, []),
        selected_termini_temple_plot(empty, []),
        selected_rt_plot(empty),
        selected_rt_modification_plot(empty),
        selected_rt_modification_residual_plot(empty),
        selected_mass_error_plot(empty),
        selected_peptide_properties_plot(empty),
        selected_decoy_fraction_plot(empty, []),
        {"display": "none"},
        {},
        {},
        {},
    )


def selected_table_columns_for_level(item_level: str | None) -> list[str]:
    common_tail = [
        "is_decoy",
        "protein_organism",
        "nterm_status",
        "specificity",
        "summed_intensity",
        "evidence_count",
        "peptide_length",
        "nterm_modif",
        "score",
        "RT",
        "Predicted_RT",
        "Abs_RT_Delta",
        "Abs_Best_Fr_Mz_Delta",
        "Abs_Ms1_Apex_Mz_Delta",
        "gravy",
    ]
    if item_level == "precursor":
        return ["item_id", "protein_id", "peptide_start", "stripped_sequence", "charge", "precursor_id", *common_tail]
    if item_level == "nterm_peptide":
        return ["item_id", "protein_id", "peptide_start", "stripped_sequence", "charge_states", *common_tail]
    if item_level == "protein":
        return ["item_id", "protein_id", "is_decoy", "summed_intensity", "evidence_count", "score", "observed_searches"]
    return ["item_id", "protein_id", "peptide_start", "charge_states", *common_tail]


def readable_selected_column(column: str) -> str:
    labels = {
        "item_id": "item_id",
        "protein_id": "protein_id",
        "peptide_start": "start",
        "stripped_sequence": "stripped_sequence",
        "charge": "charge",
        "charge_states": "charge_states",
        "precursor_id": "precursor_id",
        "nterm_status": "nterm_status",
        "summed_intensity": "summed_intensity",
        "evidence_count": "evidence_count",
        "is_decoy": "is_decoy",
        "score": "min DIA-NN Q.Value",
    }
    return labels.get(column, column)


@app.callback(
    Output("property-background-mode", "disabled"),
    Input("property-comparison-mode", "value"),
)
def update_property_background_disabled(mode):
    return mode != "background"


@app.callback(
    Output("motif-background-mode", "disabled"),
    Input("motif-comparison-mode", "value"),
)
def update_motif_background_disabled(mode):
    return mode != "background"


@app.callback(
    Output("property-plot-status", "children"),
    Output("property-plot-grid", "children"),
    Input("selected-store", "data"),
    Input("property-comparison-mode", "value"),
    Input("property-background-mode", "value"),
    Input("property-plot-checklist", "value"),
    Input("label-scale-store", "data"),
    State("dataset-id-store", "data"),
    State("item-store", "data"),
    State("set-columns-store", "data"),
    State("upset-source-dropdown", "value"),
    State("upset-evidence-filter-dropdown", "value"),
    State("upset-entrapment-dropdown", "value"),
    State("upset-fdr-dropdown", "value"),
    State("plot-style-store", "data"),
)
def update_ms_property_plots(
    selected,
    comparison_mode,
    background_mode,
    visible_plots,
    label_scale,
    dataset_id,
    items,
    set_columns,
    item_level,
    evidence_filter,
    entrapment_mode,
    qvalue_threshold,
    plot_style,
):
    if not dataset_id or not items:
        return "No current UpSet items available.", []
    visible_plots = visible_plots or []
    if not visible_plots:
        return "No property plots selected.", []

    item_df = pd.DataFrame(items)
    set_columns = set_columns or []
    if selected:
        selected_df = selected_upset_items(item_df, set_columns, selected)
        selected_label = f"Selected n={len(selected_df)}"
    else:
        selected_df = item_df
        selected_label = f"All current items n={len(selected_df)}"
    selected_ids = set(selected_df["item_id"].dropna().astype(str)) if "item_id" in selected_df.columns else set()
    if not selected_ids:
        return "No selected items for property plots.", []

    background_ids: set[str] = set()
    if comparison_mode == "background":
        background_source = background_items_for_mode(item_df, set_columns, background_mode)
        background_ids = set(background_source["item_id"].dropna().astype(str)) - selected_ids

    include_entrapment = entrapment_mode == "include" and registry.has_entrapment_hits(dataset_id)
    qvalue_threshold = normalize_qvalue_threshold(qvalue_threshold)
    evidence_df = cached_upset_evidence_rows(
        dataset_id,
        item_level,
        evidence_filter,
        include_entrapment=include_entrapment,
        qvalue_threshold=qvalue_threshold,
    )
    if evidence_df.empty or "item_id" not in evidence_df.columns:
        return "No supporting peptide evidence rows available.", []

    selected_evidence = evidence_df[evidence_df["item_id"].astype(str).isin(selected_ids)].copy()
    selected_evidence["group"] = "Selected"
    groups = [selected_evidence]
    if comparison_mode == "background" and background_ids:
        background_evidence = evidence_df[evidence_df["item_id"].astype(str).isin(background_ids)].copy()
        background_evidence["group"] = "Background"
        groups.append(background_evidence)
    plot_df = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    if plot_df.empty:
        return "No supporting peptide evidence rows for the selected items.", []
    plot_df = add_ms_properties(plot_df)

    children = [
        html.Div(
            className="followup-plot-card",
            children=followup_graph(
                ms_property_figure(plot_df, plot_id),
                f"ms_property_{plot_id}",
                plot_style=plot_style,
                label_scale=label_scale,
            ),
        )
        for plot_id in visible_plots
    ]
    if comparison_mode == "background":
        status = f"{selected_label}; Background n={len(groups[1]) if len(groups) > 1 else 0} peptide evidence rows."
    else:
        status = f"{selected_label}; Selected n={len(selected_evidence)} peptide evidence rows."
    return status, children


@app.callback(
    Output("motif-plot-status", "children"),
    Output("motif-plot-grid", "children"),
    Input("selected-store", "data"),
    Input("motif-comparison-mode", "value"),
    Input("motif-background-mode", "value"),
    Input("sequence-logo-mode", "value"),
    Input("motif-plot-checklist", "value"),
    Input("label-scale-store", "data"),
    State("dataset-id-store", "data"),
    State("item-store", "data"),
    State("set-columns-store", "data"),
    State("upset-source-dropdown", "value"),
    State("upset-evidence-filter-dropdown", "value"),
    State("upset-entrapment-dropdown", "value"),
    State("upset-fdr-dropdown", "value"),
    State("plot-style-store", "data"),
)
def update_position_motif_plots(
    selected,
    comparison_mode,
    background_mode,
    logo_mode,
    visible_plots,
    label_scale,
    dataset_id,
    items,
    set_columns,
    item_level,
    evidence_filter,
    entrapment_mode,
    qvalue_threshold,
    plot_style,
):
    if not dataset_id or not items:
        return "No current UpSet items available.", []
    visible_plots = visible_plots or []
    if not visible_plots:
        return "No position or motif plots selected.", []
    if not selected:
        return "Select an UpSet intersection or set to show position and motif plots.", []

    item_df = pd.DataFrame(items)
    set_columns = set_columns or []
    selected_df = selected_upset_items(item_df, set_columns, selected)
    selected_ids = set(selected_df["item_id"].dropna().astype(str)) if "item_id" in selected_df.columns else set()
    if not selected_ids:
        return "No selected items for position and motif plots.", []

    include_entrapment = entrapment_mode == "include" and registry.has_entrapment_hits(dataset_id)
    qvalue_threshold = normalize_qvalue_threshold(qvalue_threshold)
    evidence_df = cached_upset_evidence_rows(
        dataset_id,
        item_level,
        evidence_filter,
        include_entrapment=include_entrapment,
        qvalue_threshold=qvalue_threshold,
    )
    if evidence_df.empty or "item_id" not in evidence_df.columns:
        return "No supporting N-terminal evidence rows available.", []

    fasta_path = registry.fasta_path_for_dataset(dataset_id)
    sequences = load_fasta_sequences(fasta_path)
    if not sequences:
        return "No experiment FASTA was found for sequence-based plots.", []

    selected_evidence = evidence_df[evidence_df["item_id"].astype(str).isin(selected_ids)].copy()
    selected_sites = site_table_from_evidence(selected_evidence, sequences)
    if selected_sites.empty:
        return "No selected N-terminal sites could be mapped to the experiment FASTA.", []
    selected_sites["item_id"] = (
        selected_sites["protein_id"].astype(str) + "|" + selected_sites["start_position"].astype(str)
    )
    selected_sites["group"] = "Selected"

    plot_groups = [selected_sites]
    background_sites = pd.DataFrame()
    if comparison_mode == "background":
        background_evidence = evidence_for_background_mode(evidence_df, item_df, set_columns, background_mode)
        background_sites = site_table_from_evidence(background_evidence, sequences)
        if not background_sites.empty:
            selected_site_keys = set(zip(selected_sites["protein_id"].astype(str), selected_sites["start_position"].astype(int)))
            background_sites = background_sites.loc[
                ~background_sites.apply(lambda row: (str(row["protein_id"]), int(row["start_position"])) in selected_site_keys, axis=1)
            ].copy()
            background_sites["item_id"] = (
                background_sites["protein_id"].astype(str) + "|" + background_sites["start_position"].astype(str)
            )
            background_sites["group"] = "Background"
            plot_groups.append(background_sites)

    plot_df = pd.concat(plot_groups, ignore_index=True) if plot_groups else pd.DataFrame()
    logo_background_windows = []
    if logo_mode and logo_mode != "selected":
        logo_evidence = evidence_for_background_mode(evidence_df, item_df, set_columns, logo_mode)
        logo_background_windows = cached_background_windows(
            cache_root=CONFIG.maplot_root.parent / "maplotsserver" / "cache",
            dataset_id=dataset_id,
            item_level=item_level or "",
            evidence_filter=evidence_filter or "",
            background_mode=logo_mode,
            searches=set_columns,
            evidence_df=logo_evidence,
            sequences=sequences,
        )
    plot_df.attrs["logo_mode"] = logo_mode or "selected"
    plot_df.attrs["background_windows"] = logo_background_windows

    children = [
        html.Div(
            className="followup-plot-card",
            children=followup_graph(
                position_motif_figure(plot_df, plot_id),
                f"position_motif_{plot_id}",
                plot_style=plot_style,
                label_scale=label_scale,
            ),
        )
        for plot_id in visible_plots
    ]
    valid_windows = int(selected_sites["motif_window"].notna().sum()) if "motif_window" in selected_sites.columns else 0
    detail = f"Selected {len(selected_sites):,} unique N-terminal site(s), {valid_windows:,} full motif window(s)."
    if comparison_mode == "background":
        detail += f" Background {len(background_sites):,} unique site(s)."
    return detail, children


def evidence_for_background_mode(evidence_df: pd.DataFrame, item_df: pd.DataFrame, set_columns: list[str], background_mode: str | None) -> pd.DataFrame:
    if evidence_df.empty:
        return evidence_df
    mode = background_mode or "all_identified"
    if mode == "baseline_identified":
        if "search" in evidence_df.columns:
            return evidence_df.loc[evidence_df["search"].astype(str) == "baseline"].copy()
        return evidence_df.iloc[0:0].copy()
    if mode == "core_identified":
        if item_df.empty or not set_columns:
            return evidence_df.iloc[0:0].copy()
        core_ids = set(
            item_df.loc[item_df[set_columns].fillna(False).astype(bool).all(axis=1), "item_id"]
            .dropna()
            .astype(str)
        )
        return evidence_df.loc[evidence_df["item_id"].astype(str).isin(core_ids)].copy()
    return evidence_df.copy()


def background_items_for_mode(item_df: pd.DataFrame, set_columns: list[str], background_mode: str | None) -> pd.DataFrame:
    if item_df.empty:
        return item_df
    if background_mode == "baseline_except_selected" and "baseline" in item_df.columns:
        return item_df.loc[item_df["baseline"].fillna(False).astype(bool)]
    if background_mode == "core_except_selected" and set_columns:
        mask = item_df[set_columns].fillna(False).astype(bool).all(axis=1)
        return item_df.loc[mask]
    return item_df

if __name__ == "__main__":
    app.run(host=CONFIG.host, port=CONFIG.port, debug=CONFIG.debug)
