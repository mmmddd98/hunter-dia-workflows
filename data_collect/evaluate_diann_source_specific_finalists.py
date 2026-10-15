#!/usr/bin/env python3
"""Evaluate validation-frozen finalists exactly once on the closed final test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "trainingdata" / "finalrt_4source_trainfit15_validation10_v1"
SOURCE_ORDER = ("celegans_exploris", "triplemix_source_a", "triplemix_source_b", "human_hunter")
SOURCE_IDS = {"celegans_exploris": "celegans_exploris_44min_v1", "triplemix_source_a": "triplemix_source_a_entrap_neu_filtered_v1", "triplemix_source_b": "triplemix_pd_hamilton_60min_v1", "human_hunter": "human_source_hunter_human_v1"}
DISPLAY = {"combined": "Combined", "historical_combined": "Historical combined", "celegans_exploris": "C. elegans", "triplemix_source_a": "TripleMix Source A", "triplemix_source_b": "TripleMix Source B", "human_hunter": "Human"}
TARGET_DISPLAY = {"source_native": "Source-gradient normalized RT", "chronologer_hi": "Chronologer HI", "anchor_irt": "Human-anchor RT"}
COLORS = {"combined": "#183A5A", "historical_combined": "#4B5563", "celegans_exploris": "#2F6B9A", "triplemix_source_a": "#C47A18", "triplemix_source_b": "#718657", "human_hunter": "#A84E70"}
HEAVY = ("heavy_nterm", "heavy_lysine", "heavy_both")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""): digest.update(part)
    return digest.hexdigest()


def slug(value: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
def metric(item: dict | None, key: str = "mae") -> float:
    value = (item or {}).get(key); return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else float("nan")


def source_record(payload: dict, source_id: str) -> dict[str, float]:
    item = payload["sources"].get(source_id, {}); out = {"mae": metric(item.get("raw_minutes"))}
    for key in HEAVY: out[key] = metric(item.get(f"raw_heavy_{key}"))
    out["heavy_macro"] = float(np.nanmean([out[key] for key in HEAVY])); return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True); parser.add_argument("--selection", type=Path)
    parser.add_argument("--historical-registry", type=Path, action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET, help="Immutable canonical v1 split; used for comparator identity check")
    parser.add_argument("--target-data-root", type=Path, help="Defaults to prepared_target_root recorded in validation selection")
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable)); parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def load_candidates(selection: Path, registries: list[Path], expected_hash: str) -> tuple[list[dict], dict]:
    selection_payload = json.loads(selection.read_text(encoding="utf-8")); payloads = [selection_payload] + [json.loads(path.read_text(encoding="utf-8")) for path in registries]
    candidates: list[dict] = []; seen = set()
    for payload in payloads:
        if payload.get("dataset_manifest_sha256") != expected_hash: raise ValueError("A comparator uses a different canonical split")
        if payload.get("selection_partition") != "validation": raise ValueError("Every candidate must have been selected from validation only")
        for raw in payload["candidates"]:
            candidate = raw.copy(); candidate.setdefault("target", payload.get("target"))
            if not candidate.get("target"): raise ValueError("Candidate lacks target coordinate")
            key = (candidate["label"], candidate["target"])
            if key in seen: raise ValueError(f"Duplicate candidate {key}")
            seen.add(key); candidates.append(candidate)
    return candidates, selection_payload


def test_hashes(target_root: Path) -> dict[str, str]:
    files = {"aggregate": target_root / "test" / "diann_rt_input" / "rt_test_data.tsv"}
    files.update({name: target_root / "test" / "sources" / f"{source}.tsv" for name, source in SOURCE_IDS.items()})
    if missing := [path for path in files.values() if not path.is_file()]: raise FileNotFoundError(f"Missing prepared final-test files: {missing}")
    return {name: sha256(path) for name, path in files.items()}


def final_record(candidate: dict, seed: int, metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text(encoding="utf-8")); values = {name: source_record(payload, source) for name, source in SOURCE_IDS.items()}
    out = {"candidate": candidate["label"], "scope": candidate["scope"], "scope_kind": candidate["scope_kind"], "selection_role": candidate["selection_role"], "strategy": candidate["strategy"], "target": candidate["target"], "seed": seed,
           "source_balanced_mae": float(np.nanmean([x["mae"] for x in values.values()])), "source_balanced_heavy_macro_mae": float(np.nanmean([x["heavy_macro"] for x in values.values()])), "worst_source_mae": float(np.nanmax([x["mae"] for x in values.values()]))}
    for source, item in values.items():
        out[f"source_{source}_mae"] = item["mae"]
        for key in HEAVY: out[f"source_{source}_{key}_mae"] = item[key]
    return out


def save(fig: plt.Figure, base: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"): fig.savefig(base.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def matched_control(subset: pd.DataFrame) -> pd.Series | None:
    controls = subset[subset.scope.isin(["combined", "historical_combined"])]
    if controls.empty: return None
    preferred = controls[controls.selection_role.isin(["overall", "heavy_guardrail"])]
    return (preferred if not preferred.empty else controls).sort_values("source_balanced_mae").iloc[0]


def plots(summary: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True); targets = [x for x in ("source_native", "chronologer_hi", "anchor_irt") if x in set(summary.target)]
    fig, axes = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), 5.1), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = summary[summary.target.eq(target)]
        for scope, group in subset.groupby("scope", sort=False): axis.errorbar(group.source_balanced_mae, group.source_balanced_heavy_macro_mae, xerr=group.source_balanced_mae_ci95, yerr=group.source_balanced_heavy_macro_mae_ci95, fmt="s" if scope == "historical_combined" else "o", color=COLORS[scope], capsize=2.3, markersize=5.5, label=DISPLAY[scope])
        axis.set(title=TARGET_DISPLAY[target], xlabel="Source-balanced final-test MAE (min)", ylabel="Heavy-Macro final-test MAE (min)"); axis.grid(alpha=.22)
    handles, labels = axes[0][0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False); fig.suptitle("Final test: overall versus Heavy-Dimethyl performance", y=1.02); fig.tight_layout(rect=(0, .09, 1, 1)); save(fig, output / "fig5_final_test_overall_heavy_tradeoff")
    # Absolute per-source final-test errors: coordinate facets retain comparable target and control.
    fig, axes = plt.subplots(len(targets), 1, figsize=(9.6, 2.2 * len(targets) + 1.2), squeeze=False)
    cols = [f"source_{source}_mae" for source in SOURCE_ORDER]
    for axis, target in zip(axes[:, 0], targets):
        subset = summary[summary.target.eq(target)].copy(); matrix = subset.set_index("candidate")[cols]
        image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto"); axis.set(xticks=np.arange(len(SOURCE_ORDER)), xticklabels=[DISPLAY[x] for x in SOURCE_ORDER], yticks=np.arange(len(matrix)), yticklabels=matrix.index, title=TARGET_DISPLAY[target])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]): axis.text(col, row, f"{matrix.iloc[row, col]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, label="Observed-minute MAE")
    fig.suptitle("Final-test MAE by evaluation source and selected model", y=1.01); fig.tight_layout(); save(fig, output / "fig6_final_test_source_matrix")
    # Direct difference only to a target-matched combined reference.
    fig, axes = plt.subplots(len(targets), 1, figsize=(9.6, 2.2 * len(targets) + 1.2), squeeze=False)
    for axis, target in zip(axes[:, 0], targets):
        subset = summary[summary.target.eq(target)].copy(); control = matched_control(subset)
        if control is None: axis.set_visible(False); continue
        matrix = subset.set_index("candidate")[cols]; delta = matrix.subtract(control[cols], axis="columns"); limit = max(.005, float(np.nanmax(np.abs(delta.to_numpy(float)))))
        image = axis.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"); axis.set(xticks=np.arange(len(SOURCE_ORDER)), xticklabels=[DISPLAY[x] for x in SOURCE_ORDER], yticks=np.arange(len(delta)), yticklabels=delta.index, title=f"{TARGET_DISPLAY[target]} vs matched combined reference")
        for row in range(delta.shape[0]):
            for col in range(delta.shape[1]): axis.text(col, row, f"{delta.iloc[row, col]:+.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, label="MAE difference (min; negative favours row)")
    fig.suptitle("Final-test source-specific change versus target-matched combined model", y=1.01); fig.tight_layout(); save(fig, output / "fig7_final_test_source_specific_gain")


def main() -> None:
    args = parse_args()
    if not args.confirm_final_test: raise SystemExit("Refusing to read final test. Re-run only after validation selection is frozen, with --confirm-final-test.")
    study, dataset = args.study_root.resolve(), args.dataset_dir.resolve(); selection = (args.selection or study / "analysis" / "validation_selection.json").resolve(); expected_hash = sha256(dataset / "dataset_manifest.json")
    candidates, selection_payload = load_candidates(selection, [path.resolve() for path in args.historical_registry], expected_hash)
    target_data = (args.target_data_root or Path(selection_payload["prepared_target_root"])).resolve(); output = study / "analysis" / "final_test"; output.mkdir(parents=True, exist_ok=True)
    records = []; provenance = {"canonical_dataset_manifest_sha256": expected_hash, "selection": str(selection), "target_data_root": str(target_data), "candidates": [], "test_hashes": {}}
    for candidate in candidates:
        target_root = target_data / candidate["target"]; alignment, partition = target_root / "rt_alignment.json", target_root / "test"
        if not alignment.is_file(): raise FileNotFoundError(f"Missing target alignment: {alignment}")
        provenance["test_hashes"][candidate["target"]] = test_hashes(target_root); candidate_id = slug(f"{candidate['scope']}__{candidate['selection_role']}__{candidate['strategy']}__{candidate['target']}"); evaluations = []
        for member in candidate["seeds"]:
            run = Path(member["run_dir"]).resolve(); seed = int(member["seed"])
            if not (run / "rt.d0.finetuned.pt").is_file() or not (run / "dict.txt").is_file(): raise FileNotFoundError(f"Incomplete selected model: {run}")
            destination = output / "evaluations" / candidate_id / f"seed_{seed}"; metrics = destination / "final_test_metrics.json"
            if not metrics.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                subprocess.run([args.diann_python, str(ROOT / "ma/data_collect/evaluate_diann_trainfit.py"), "--run-dir", str(run), "--partition-dir", str(partition), "--alignment", str(alignment), "--output", str(metrics), "--prediction-dir", str(destination / "predictions"), "--device", "cuda"], check=True)
            records.append(final_record(candidate, seed, metrics)); evaluations.append({"seed": seed, "run_dir": str(run), "metrics": str(metrics)})
        provenance["candidates"].append({**candidate, "evaluation_id": candidate_id, "evaluations": evaluations})
    runs = pd.DataFrame(records); by = ["candidate", "scope", "scope_kind", "selection_role", "strategy", "target"]; numeric = [x for x in runs if x.endswith("_mae")]
    summary = runs.groupby(by, as_index=False)[numeric].mean(); counts = runs.groupby(by, as_index=False).size().rename(columns={"size": "seeds"}); std = runs.groupby(by, as_index=False)[numeric].std().rename(columns={x: f"{x}_sd" for x in numeric}); summary = summary.merge(counts, on=by, validate="one_to_one").merge(std, on=by, validate="one_to_one")
    for column in numeric: summary[f"{column}_ci95"] = 1.96 * summary[f"{column}_sd"].fillna(0) / np.sqrt(summary.seeds)
    runs.to_csv(output / "final_test_runs.csv", index=False); summary.to_csv(output / "final_test_summary.csv", index=False); (output / "final_test_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8"); plots(summary, study / "analysis" / "plots" / "comparisons")
    (output / "README.md").write_text("# Final-test comparison\n\nCandidates were selected on validation only and evaluated once on checksum-recorded, identical final-test source observations. All metrics are observed RT minutes. Figures are target-faceted; a source-specific difference is only computed against a combined model in the same target coordinate. Three optimizer seeds quantify initialization variation, not biological replication.\n", encoding="utf-8")
    print(f"Final-test comparison complete: {output}")


if __name__ == "__main__": main()
