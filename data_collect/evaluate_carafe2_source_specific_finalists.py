#!/usr/bin/env python3
"""Evaluate validation-frozen Carafe2 source-specific finalists once on test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_diann_source_specific_sweep import DISPLAY, SOURCE_IDS, SOURCE_ORDER, TARGET_DISPLAY

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "trainingdata" / "finalrt_4source_trainfit15_validation10_v1"
COLORS = {"combined": "#183A5A", "historical_combined": "#4B5563", "celegans_exploris": "#2F6B9A", "triplemix_source_a": "#C47A18", "triplemix_source_b": "#718657", "human_hunter": "#A84E70"}
HEAVY = ("heavy_nterm", "heavy_lysine", "heavy_both")


def parse_args() -> argparse.Namespace:
    default_python = Path("python")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--historical-registry", type=Path, action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--carafe-python", default=os.environ.get("CARAFE2_PYTHON", str(default_python if default_python.is_file() else Path(sys.executable))))
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(payload: dict | None, key: str = "mae") -> float:
    value = (payload or {}).get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else float("nan")


def source_metrics(payload: dict, source_id: str) -> dict[str, float]:
    item = payload["sources"].get(source_id, {})
    result = {"mae": metric(item.get("raw_minutes"))}
    for label in HEAVY:
        result[label] = metric(item.get(f"raw_heavy_{label}"))
    result["heavy_macro"] = float(np.nanmean([result[label] for label in HEAVY]))
    return result


def load_candidates(selection: Path, registries: list[Path], dataset_hash: str) -> tuple[list[dict], dict]:
    payloads = [json.loads(selection.read_text(encoding="utf-8"))]
    payloads += [json.loads(path.read_text(encoding="utf-8")) for path in registries]
    candidates, seen = [], set()
    for payload in payloads:
        if payload.get("dataset_manifest_sha256") != dataset_hash or payload.get("selection_partition") != "validation":
            raise ValueError("Every selected model must prove the canonical split and validation-only selection")
        for candidate in payload["candidates"]:
            key = (candidate["label"], candidate["target"])
            if key in seen:
                raise ValueError(f"Duplicate candidate: {key}")
            seen.add(key)
            candidates.append(candidate)
    return candidates, payloads[0]


def record(candidate: dict, seed: int, metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    values = {source: source_metrics(payload, source_id) for source, source_id in SOURCE_IDS.items()}
    result = {"candidate": candidate["label"], "scope": candidate["scope"], "scope_kind": candidate["scope_kind"], "selection_role": candidate["selection_role"], "strategy": candidate["strategy"], "target": candidate["target"], "seed": seed, "source_balanced_mae": float(np.nanmean([value["mae"] for value in values.values()])), "source_balanced_heavy_macro_mae": float(np.nanmean([value["heavy_macro"] for value in values.values()]))}
    for source, value in values.items():
        result[f"source_{source}_mae"] = value["mae"]
        for label in HEAVY:
            result[f"source_{source}_{label}_mae"] = value[label]
    return result


def save(fig: plt.Figure, base: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plots(summary: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = [target for target in ("source_native", "chronologer_hi", "anchor_irt") if target in set(summary.target)]
    columns = [f"source_{source}_mae" for source in SOURCE_ORDER]
    fig, axes = plt.subplots(1, len(targets), figsize=(5.8 * len(targets), 5.0), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = summary[summary.target.eq(target)]
        for scope, group in subset.groupby("scope", sort=False):
            axis.errorbar(group.source_balanced_mae, group.source_balanced_heavy_macro_mae, xerr=group.source_balanced_mae_ci95, yerr=group.source_balanced_heavy_macro_mae_ci95, fmt="s" if scope == "historical_combined" else "o", color=COLORS[scope], capsize=2.2, label=DISPLAY[scope])
        axis.set(title=TARGET_DISPLAY[target], xlabel="Source-balanced final-test MAE (min)", ylabel="Heavy-Macro final-test MAE (min)")
        axis.grid(alpha=.22)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Final test: overall versus Heavy-Dimethyl performance", y=1.02)
    fig.tight_layout(rect=(0, .09, 1, 1)); save(fig, output / "fig11_final_test_overall_heavy_tradeoff")

    fig, axes = plt.subplots(len(targets), 1, figsize=(9.6, 2.2 * len(targets) + 1.2), squeeze=False)
    for axis, target in zip(axes[:, 0], targets):
        subset = summary[summary.target.eq(target)].copy()
        controls = subset[subset.scope.isin(("combined", "historical_combined"))]
        if controls.empty:
            axis.set_visible(False); continue
        control = controls.sort_values("source_balanced_mae").iloc[0]
        matrix = subset.set_index("candidate")[columns]
        delta = matrix.subtract(control[columns], axis="columns")
        limit = max(.005, float(np.nanmax(np.abs(delta.to_numpy(float)))))
        image = axis.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set(title=f"{TARGET_DISPLAY[target]} vs target-matched combined reference", xticks=np.arange(len(SOURCE_ORDER)), xticklabels=[DISPLAY[source] for source in SOURCE_ORDER], yticks=np.arange(len(delta)), yticklabels=delta.index, xlabel="Evaluation source")
        for row in range(delta.shape[0]):
            for column in range(delta.shape[1]):
                axis.text(column, row, f"{delta.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, label="MAE difference (min; negative favours row)")
    fig.suptitle("Final-test source-specific change versus target-matched combined model", y=1.01)
    fig.tight_layout(); save(fig, output / "fig12_final_test_source_specific_gain")


def main() -> None:
    args = parse_args()
    if not args.confirm_final_test:
        raise SystemExit("Refusing to read final test; review and freeze validation selection first, then pass --confirm-final-test.")
    study, dataset = args.study_root.resolve(), args.dataset_dir.resolve()
    selection = (args.selection or study / "analysis" / "validation_selection.json").resolve()
    dataset_hash = sha256(dataset / "dataset_manifest.json")
    candidates, selection_payload = load_candidates(selection, [path.resolve() for path in args.historical_registry], dataset_hash)
    target_root = Path(selection_payload["prepared_target_root"]).resolve()
    python = str(Path(args.carafe_python).expanduser().resolve())
    output = study / "analysis" / "final_test"
    records = []
    for candidate in candidates:
        target = target_root / candidate["target"]
        for member in candidate["seeds"]:
            run, seed = Path(member["run_dir"]).resolve(), int(member["seed"])
            if not (run / "rt_model.pt").is_file():
                raise FileNotFoundError(f"Missing selected Carafe2 model: {run / 'rt_model.pt'}")
            metrics = output / "evaluations" / candidate["target"] / candidate["scope"] / candidate["selection_role"] / candidate["strategy"] / f"seed_{seed}" / "final_test_metrics.json"
            if not metrics.is_file():
                metrics.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([python, str(ROOT / "ma/data_collect/evaluate_carafe2_trainfit.py"), "--run-dir", str(run), "--partition-dir", str(target / "test"), "--alignment", str(target / "rt_alignment.json"), "--output", str(metrics), "--prediction-dir", str(metrics.parent / "predictions"), "--device", "gpu", "--python", python], check=True)
            records.append(record(candidate, seed, metrics))
    runs = pd.DataFrame(records)
    group = ["candidate", "scope", "scope_kind", "selection_role", "strategy", "target"]
    numeric = [column for column in runs if column.endswith("_mae")]
    summary = runs.groupby(group, as_index=False)[numeric].mean()
    std = runs.groupby(group, as_index=False)[numeric].std().rename(columns={column: f"{column}_sd" for column in numeric})
    counts = runs.groupby(group, as_index=False).size().rename(columns={"size": "seeds"})
    summary = summary.merge(std, on=group).merge(counts, on=group)
    for column in numeric:
        summary[f"{column}_ci95"] = 1.96 * summary[f"{column}_sd"].fillna(0) / np.sqrt(summary.seeds)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "final_test_runs.csv", index=False)
    summary.to_csv(output / "final_test_summary.csv", index=False)
    (output / "README.md").write_text("# Final-test comparison\n\nCandidates were selected on validation only and evaluated once on identical, checksum-recorded test observations. Metrics are observed RT minutes; target coordinates are never pooled.\n", encoding="utf-8")
    plots(summary, study / "analysis" / "plots" / "comparisons")
    print(f"Final-test comparison complete: {output}")


if __name__ == "__main__":
    main()
