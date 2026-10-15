#!/usr/bin/env python3
"""Build a portable report artifact for a DIA-NN heavy-priority sweep."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


METRICS = {
    "overall_mae": "Overall",
    "heavy_macro_mae": "Heavy macro",
    "heavy_nterm_mae": "Heavy N-term",
    "heavy_lysine_mae": "Heavy Lys",
    "heavy_both_mae": "Heavy N-term + Lys",
    "nonheavy_mae": "Kein Heavy-Dimethyl",
}

PROFILE_LABELS = {
    "C0": "C0 · kein Oversampling",
    "H24": "H24 · Oversampling",
    "H32": "H32 · Oversampling",
    "H40": "H40 · Oversampling",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def lr_label(value: float) -> str:
    return f"{value:.1e}".replace("e-0", "e-")


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.where(pd.notna(frame), None)
    return json.loads(clean.to_json(orient="records"))


def select(frame: pd.DataFrame, profile: str, lr: float, priority: str) -> pd.DataFrame:
    return frame[
        frame["profile"].eq(profile)
        & frame["lr"].sub(lr).abs().lt(1e-12)
        & frame["checkpoint_selection"].eq(priority)
    ]


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metric_names = list(METRICS)
    grouped = (
        frame.groupby(["profile", "lr", "checkpoint_selection"], as_index=False)[metric_names]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in grouped.columns
    ]
    return grouped


def configuration_row(
    candidates: pd.DataFrame,
    goal: str,
    profile: str,
    lr: float,
    priority: str,
) -> dict:
    selected = select(candidates, profile, lr, priority)
    row = {
        "Zweck": goal,
        "Profil": PROFILE_LABELS[profile],
        "LR": lr_label(lr),
        "Checkpoint": priority.replace("_", "/"),
        "Seeds": int(selected["seed"].nunique()),
    }
    for metric, label in METRICS.items():
        row[label] = float(selected[metric].mean())
    row["Heavy-both SD"] = float(selected["heavy_both_mae"].std())
    return row


def checkpoint_effect(candidates: pd.DataFrame) -> pd.DataFrame:
    keys = ["profile", "lr", "seed"]
    reference = candidates[candidates["checkpoint_selection"].eq("100_0")][
        [*keys, "checkpoint_epoch", "overall_mae", "heavy_macro_mae"]
    ].rename(
        columns={
            "checkpoint_epoch": "reference_checkpoint_epoch",
            "overall_mae": "reference_overall_mae",
            "heavy_macro_mae": "reference_heavy_macro_mae",
        }
    )
    merged = candidates.merge(reference, on=keys, how="inner")
    rows = []
    for priority, part in merged.groupby("checkpoint_selection"):
        for metric, label in (
            ("overall_mae", "Overall"),
            ("heavy_macro_mae", "Heavy macro"),
        ):
            delta = (part[metric] - part[f"reference_{metric}"]) * 60
            rows.append(
                {
                    "Checkpoint": priority.replace("_", "/"),
                    "Metrik": label,
                    "Delta Sekunden": float(delta.mean()),
                    "Runs mit anderer Epoche": int(
                        (part["checkpoint_epoch"] != part["reference_checkpoint_epoch"]).sum()
                    )
                    if priority != "100_0"
                    else 0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates_path = args.analysis_dir / "checkpoint_candidates.csv"
    candidates = pd.read_csv(candidates_path)
    if len(candidates) != 180:
        raise ValueError(f"Expected 180 checkpoint candidates, found {len(candidates)}")
    if candidates["job_id"].nunique() != 36:
        raise ValueError("Expected 36 completed runs")

    config = aggregate(candidates)
    config["Profil"] = config["profile"].map(PROFILE_LABELS)
    config["LR"] = config["lr"].map(lr_label)
    config["Checkpoint"] = config["checkpoint_selection"].str.replace("_", "/", regex=False)
    config["Konfiguration"] = config["Profil"] + " · " + config["LR"] + " · " + config["Checkpoint"]
    tradeoff = config.rename(
        columns={"overall_mae_mean": "Overall MAE", "heavy_macro_mae_mean": "Heavy macro MAE"}
    )[["Konfiguration", "Profil", "LR", "Checkpoint", "Overall MAE", "Heavy macro MAE"]]

    key_specs = [
        ("C0 reference", "C0", 1.5e-4, "50_50"),
        ("H40 comparison at 1e-4", "H40", 1e-4, "100_0"),
        ("H40 comparison at 5e-5", "H40", 5e-5, "50_50"),
    ]
    key_rows = [configuration_row(candidates, *spec) for spec in key_specs]
    key_table = pd.DataFrame(key_rows)

    strata_rows = []
    for goal, profile, lr, priority in key_specs[:2]:
        selected = select(candidates, profile, lr, priority)
        model = f"{profile} · lr {lr_label(lr)} · {priority.replace('_', '/')}"
        for metric, label in METRICS.items():
            strata_rows.append(
                {
                    "Modell": model,
                    "Stratum": label,
                    "MAE Minuten": float(selected[metric].mean()),
                }
            )
    strata = pd.DataFrame(strata_rows)

    # Same-learning-rate, same-priority comparisons isolate the oversampling effect.
    delta_specs = [
        ("H40 vs C0 · lr 5e-5 · 50/50", "H40", 5e-5, "50_50"),
        ("H32 vs C0 · lr 1e-4 · 50/50", "H32", 1e-4, "50_50"),
        ("H40 vs C0 · lr 1e-4 · 50/50", "H40", 1e-4, "50_50"),
    ]
    delta_rows = []
    for label, profile, lr, priority in delta_specs:
        treated = select(candidates, profile, lr, priority).set_index("seed")
        control = select(candidates, "C0", lr, priority).set_index("seed")
        common = treated.index.intersection(control.index)
        for metric, stratum in METRICS.items():
            delta = (treated.loc[common, metric] - control.loc[common, metric]) * 60
            delta_rows.append(
                {
                    "Vergleich": label,
                    "Stratum": stratum,
                    "Delta Sekunden": float(delta.mean()),
                    "Delta SD": float(delta.std()),
                }
            )
    deltas = pd.DataFrame(delta_rows)

    seed_rows = []
    for goal, profile, lr, priority in key_specs[:2]:
        selected = select(candidates, profile, lr, priority)
        model = f"{profile} · lr {lr_label(lr)} · {priority.replace('_', '/')}"
        for row in selected.itertuples():
            seed_rows.append(
                {
                    "Modell": model,
                    "Seed": str(row.seed),
                    "Heavy-both MAE": float(row.heavy_both_mae),
                }
            )
    seed_stability = pd.DataFrame(seed_rows)

    priority_effect = checkpoint_effect(candidates)

    # Derived headline values.
    best_reference = select(candidates, "C0", 1.5e-4, "50_50")
    best_both = select(candidates, "H40", 1e-4, "100_0")
    gain_both_seconds = (
        best_reference["heavy_both_mae"].mean() - best_both["heavy_both_mae"].mean()
    ) * 60
    overall_cost_seconds = (
        best_both["overall_mae"].mean() - best_reference["overall_mae"].mean()
    ) * 60

    # Verify the central all-heavy claim directly over all oversampled configurations.
    c0_best = {
        metric: config.loc[config["profile"].eq("C0"), f"{metric}_mean"].min()
        for metric in ("heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae")
    }
    oversampled = config[~config["profile"].eq("C0")]
    all_heavy_winners = oversampled[
        (oversampled["heavy_nterm_mae_mean"] < c0_best["heavy_nterm_mae"])
        & (oversampled["heavy_lysine_mae_mean"] < c0_best["heavy_lysine_mae"])
        & (oversampled["heavy_both_mae_mean"] < c0_best["heavy_both_mae"])
    ]

    summary = pd.DataFrame(
        [
            {
                "Runs": int(candidates["job_id"].nunique()),
                "Checkpoints": int(len(candidates)),
                "Heavy-both Gewinn (s)": float(gain_both_seconds),
                "Overall Kosten (s)": float(overall_cost_seconds),
                "Oversampling-Konfigurationen besser in allen Heavy-Strata": int(len(all_heavy_winners)),
            }
        ]
    )

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    tradeoff.to_csv(args.analysis_dir / "report_tradeoff.csv", index=False)
    strata.to_csv(args.analysis_dir / "report_key_strata.csv", index=False)
    deltas.to_csv(args.analysis_dir / "report_oversampling_deltas.csv", index=False)
    seed_stability.to_csv(args.analysis_dir / "report_seed_stability.csv", index=False)
    priority_effect.to_csv(args.analysis_dir / "report_checkpoint_effect.csv", index=False)
    key_table.to_csv(args.analysis_dir / "report_key_configurations.csv", index=False)

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    source = {
        "id": "validation-metrics",
        "label": "Fixed validation partition: 36 runs × 5 checkpoint priorities",
        "query": {
            "language": "python",
            "query": "Parse checkpoint_selection.json and validation_metrics/checkpoint_*.json; aggregate source-balanced observed-minute MAE first across sources and then across three seeds.",
            "description": "Validation-only sweep; the test partition was not read. Every displayed configuration mean contains three seeds.",
            "filters": ["target = chronologer_hi", "test_partition = closed"],
            "metric_definitions": [
                "Overall MAE: mean raw absolute RT error in observed minutes, balanced equally across data sources.",
                "Heavy macro MAE: equal mean of Heavy N-term, Heavy Lys, and Heavy N-term + Lys MAE.",
                "Delta: oversampled model minus C0 without oversampling at identical learning rate, checkpoint priority, and seed; negative is better.",
            ],
            "tables_used": ["sweep_manifest.json", "checkpoint_selection.json", "validation_metrics/checkpoint_*.json"],
        },
    }

    charts = [
        {
            "id": "tradeoff",
            "title": "Gesamtfehler versus Heavy-Macro-Fehler",
            "subtitle": "60 Konfigurationen; jeder Punkt ist der Mittelwert aus drei Seeds. Links unten ist besser.",
            "type": "scatter",
            "intent": "relationship",
            "question": "Erzeugt mehr Heavy-Fokus einen besseren Gesamtkompromiss?",
            "rationale": "Die beiden konkurrierenden Ziele werden direkt gegeneinander aufgetragen.",
            "dataset": "tradeoff",
            "sourceId": "validation-metrics",
            "encodings": {
                "x": {"field": "Overall MAE", "type": "quantitative", "label": "Overall MAE (min)"},
                "y": {"field": "Heavy macro MAE", "type": "quantitative", "label": "Heavy macro MAE (min)"},
                "color": {"field": "Profil", "type": "nominal", "label": "Oversampling-Profil"},
                "tooltip": [
                    {"field": "Konfiguration", "type": "text", "label": "Konfiguration"},
                    {"field": "Overall MAE", "type": "quantitative", "label": "Overall MAE"},
                    {"field": "Heavy macro MAE", "type": "quantitative", "label": "Heavy macro MAE"},
                ],
            },
            "layout": "full",
            "surface": {"viewMode": "visualization", "showControls": True},
        },
        {
            "id": "strata",
            "title": "Was gewinnt und was verliert das Heavy-both-Modell?",
            "subtitle": "Predefined C0 and H40 configurations; lower validation errors are better.",
            "type": "bar",
            "intent": "comparison",
            "question": "Verbessern sich alle dimethylierten Peptidtypen gleichzeitig?",
            "rationale": "Die exakten Strata verhindern, dass der Heavy-Macro-Mittelwert gegenläufige Effekte verdeckt.",
            "dataset": "strata",
            "sourceId": "validation-metrics",
            "encodings": {
                "x": {"field": "Stratum", "type": "nominal", "label": "Peptidstratum"},
                "y": {"field": "MAE Minuten", "type": "quantitative", "label": "MAE (min)"},
                "color": {"field": "Modell", "type": "nominal", "label": "Modell"},
            },
            "settings": {"groupMode": "grouped", "categoryLabelPolicy": "wrap"},
            "layout": "full",
        },
        {
            "id": "oversampling-deltas",
            "title": "Isolierter Oversampling-Effekt bei gleicher Lernrate",
            "subtitle": "Seed-gematchte Differenz zu C0 ohne Oversampling; negative Sekunden bedeuten Verbesserung.",
            "type": "bar",
            "intent": "comparison",
            "question": "Welche Heavy-Strata reagieren tatsächlich auf Oversampling?",
            "rationale": "Gleiche Lernrate und Checkpoint-Priorität trennen den Oversampling-Effekt vom Lernraten-Effekt.",
            "dataset": "deltas",
            "sourceId": "validation-metrics",
            "encodings": {
                "x": {"field": "Stratum", "type": "nominal", "label": "Peptidstratum"},
                "y": {"field": "Delta Sekunden", "type": "quantitative", "label": "MAE-Delta (s)"},
                "color": {"field": "Vergleich", "type": "nominal", "label": "Vergleich"},
            },
            "referenceLines": [{"axis": "y", "value": 0, "label": "kein Unterschied", "color": "neutral"}],
            "settings": {"groupMode": "grouped", "categoryLabelPolicy": "wrap"},
            "layout": "full",
        },
        {
            "id": "seed-stability",
            "title": "Heavy-both validation error by seed",
            "subtitle": "C0 and H40 validation errors are shown separately for each seed.",
            "type": "bar",
            "intent": "comparison",
            "question": "Ist der Heavy-both-Gewinn nur ein einzelner glücklicher Seed?",
            "rationale": "Die drei Einzelwerte zeigen die Reproduzierbarkeit ohne Aggregation.",
            "dataset": "seed_stability",
            "sourceId": "validation-metrics",
            "encodings": {
                "x": {"field": "Seed", "type": "ordinal", "label": "Seed"},
                "y": {"field": "Heavy-both MAE", "type": "quantitative", "label": "Heavy N-term + Lys MAE (min)"},
                "color": {"field": "Modell", "type": "nominal", "label": "Modell"},
            },
            "settings": {"groupMode": "grouped"},
            "layout": "half",
        },
        {
            "id": "checkpoint-effect",
            "title": "Checkpoint-selection comparison",
            "subtitle": "Mittlere Abweichung vom 100/0-Checkpoint über 36 identische Runs; negative Sekunden sind besser.",
            "type": "bar",
            "intent": "comparison",
            "question": "Wie stark ändert die 0/100–100/0-Auswahl das Ergebnis?",
            "rationale": "Die fünf gespeicherten Checkpoint-Regeln werden innerhalb desselben Trainingslaufs verglichen.",
            "dataset": "priority_effect",
            "sourceId": "validation-metrics",
            "encodings": {
                "x": {"field": "Checkpoint", "type": "ordinal", "label": "Overall/Heavy-Priorität"},
                "y": {"field": "Delta Sekunden", "type": "quantitative", "label": "MAE-Delta (s)"},
                "color": {"field": "Metrik", "type": "nominal", "label": "Metrik"},
            },
            "referenceLines": [{"axis": "y", "value": 0, "label": "100/0", "color": "neutral"}],
            "settings": {"groupMode": "grouped"},
            "layout": "half",
        },
    ]

    tables = [
        {
            "id": "key-configurations",
            "title": "Konfigurationen für die Entscheidung",
            "subtitle": "Source-balanced Validation-MAE in beobachteten Minuten; jeweils Mittelwert über drei Seeds.",
            "dataset": "key_table",
            "sourceId": "validation-metrics",
            "layout": "full",
            "density": "dense",
            "columns": [
                {"field": "Zweck", "label": "Zweck", "type": "text"},
                {"field": "Profil", "label": "Profil", "type": "text"},
                {"field": "LR", "label": "LR", "type": "text"},
                {"field": "Checkpoint", "label": "Checkpoint", "type": "text"},
                {"field": "Overall", "label": "Overall", "format": "number"},
                {"field": "Heavy macro", "label": "Heavy macro", "format": "number"},
                {"field": "Heavy N-term", "label": "Heavy N-term", "format": "number"},
                {"field": "Heavy Lys", "label": "Heavy Lys", "format": "number"},
                {"field": "Heavy N-term + Lys", "label": "Heavy beide", "format": "number"},
                {"field": "Kein Heavy-Dimethyl", "label": "Kein Heavy", "format": "number"},
                {"field": "Heavy-both SD", "label": "Heavy-beide SD", "format": "number"},
            ],
        }
    ]

    cards = [
        {
            "id": "headline",
            "dataset": "summary",
            "sourceId": "validation-metrics",
            "metrics": [
                {"label": "vollständige Runs", "field": "Runs", "format": "number"},
                {"label": "bewertete Checkpoints", "field": "Checkpoints", "format": "number"},
                {"label": "Heavy-beide Gewinn", "field": "Heavy-both Gewinn (s)", "format": "number"},
                {"label": "Overall-Kosten", "field": "Overall Kosten (s)", "format": "number"},
                {"label": "besser in allen Heavy-Strata", "field": "Oversampling-Konfigurationen besser in allen Heavy-Strata", "format": "number"},
            ],
        }
    ]

    blocks = [
        {
            "id": "summary-text",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Validation comparison\n\nThe accompanying chart or table summarizes the supplied validation metrics.'            ),
        },
        {"id": "headline-block", "type": "metric-strip", "cardIds": ["headline"], "layout": "full"},
        {
            "id": "metric-note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Reading the metrics\n\nErrors are source-balanced MAEs in observed RT minutes, averaged over seeds. Heavy-macro MAE weights the three heavy-dimethyl strata equally. Lower values are better.'           ),
        },
        {
            "id": "tradeoff-note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Overall and heavy-macro errors\n\nEach point compares these two validation metrics for one configuration.'            ),
        },
        {"id": "tradeoff-block", "type": "chart", "chartId": "tradeoff", "layout": "full"},
        {
            "id": "strata-note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Modification-stratum errors\n\nThe chart compares the predefined configurations separately for each modification stratum.'
            ),
        },
        {"id": "strata-block", "type": "chart", "chartId": "strata", "layout": "full"},
        {
            "id": "delta-note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Seed-matched oversampling differences\n\nCompare each profile with the C0 control at the same learning rate, seed and checkpoint rule. Negative differences indicate lower validation errors.'            ),
        },
        {"id": "delta-block", "type": "chart", "chartId": "oversampling-deltas", "layout": "full"},
        {"id": "seed-block", "type": "chart", "chartId": "seed-stability", "layout": "half"},
        {"id": "checkpoint-block", "type": "chart", "chartId": "checkpoint-effect", "layout": "half"},
        {
            "id": "checkpoint-note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Checkpoint-selection rules\n\nThe chart shows differences relative to the 100/0 rule within the same training runs.'          ),
        },
        {"id": "table-block", "type": "table", "tableId": "key-configurations", "layout": "full"},
        {
            "id": "decision",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Interpretation\n\nSelect configurations according to a predefined objective and assess selected models on independent held-out data. The displayed validation comparisons alone are not an independent confirmation.'           ),
        },
        {
            "id": "caveats",
            "type": "markdown",
            "layout": "full",
            "sourceId": "validation-metrics",
            "body": (
                '## Validation comparison\n\nThe accompanying chart or table summarizes the supplied validation metrics.'           ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "DIA-NN Heavy-Dimethyl Priority Sweep",
            "description": "Validation-Auswertung des 4×3×3-Sweeps mit fünf Checkpoint-Prioritäten pro Run.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": records(summary),
                "tradeoff": records(tradeoff),
                "strata": records(strata),
                "deltas": records(deltas),
                "seed_stability": records(seed_stability),
                "priority_effect": records(priority_effect),
                "key_table": records(key_table),
            },
        },
        "sources": [source],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "runs": int(candidates["job_id"].nunique()),
                "checkpoints": int(len(candidates)),
                "heavy_both_gain_seconds": round(float(gain_both_seconds), 3),
                "all_heavy_winners": int(len(all_heavy_winners)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
