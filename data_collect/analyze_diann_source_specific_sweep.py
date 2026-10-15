#!/usr/bin/env python3
"""Analyse the closed-test final DIA-NN source-specific validation sweep."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SOURCE_ORDER = ("celegans_exploris", "triplemix_source_a", "triplemix_source_b", "human_hunter")
SOURCE_IDS = {"celegans_exploris": "celegans_exploris_44min_v1", "triplemix_source_a": "triplemix_source_a_entrap_neu_filtered_v1", "triplemix_source_b": "triplemix_pd_hamilton_60min_v1", "human_hunter": "human_source_hunter_human_v1"}
DISPLAY = {"combined": "Combined", "celegans_exploris": "C. elegans", "triplemix_source_a": "TripleMix Source A", "triplemix_source_b": "TripleMix Source B", "human_hunter": "Human"}
SHORT_DISPLAY = {"celegans_exploris": "C. elegans", "triplemix_source_a": "Source A", "triplemix_source_b": "Source B", "human_hunter": "Human"}
TARGET_DISPLAY = {"source_native": "Source-gradient normalized RT", "chronologer_hi": "Chronologer HI", "anchor_irt": "Human-anchor RT"}
COLORS = {"combined": "#183A5A", "celegans_exploris": "#2F6B9A", "triplemix_source_a": "#C47A18", "triplemix_source_b": "#718657", "human_hunter": "#A84E70"}
HEAVY = ("heavy_nterm", "heavy_lysine", "heavy_both")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--study-root", type=Path, required=True); return parser.parse_args()


def value(item: dict | None, key: str = "mae") -> float:
    candidate = (item or {}).get(key); return float(candidate) if isinstance(candidate, (float, int)) and math.isfinite(candidate) else float("nan")


def source_metrics(payload: dict, source_id: str) -> dict[str, float]:
    item = payload["sources"].get(source_id, {}); result = {"mae": value(item.get("raw_minutes"))}
    for label in HEAVY: result[label] = value(item.get(f"raw_heavy_{label}"))
    result["heavy_macro"] = float(np.nanmean([result[label] for label in HEAVY])); return result


def collect(manifest: dict) -> pd.DataFrame:
    rows = []
    for job in manifest["jobs"]:
        metadata = job["metadata"]; run = Path(job["completion_files"][0]).parent; path = run / "validation_metrics.json"
        if not path.is_file(): raise FileNotFoundError(f"Incomplete validation sweep: {path}")
        per_source = {scope: source_metrics(json.loads(path.read_text(encoding="utf-8")), source_id) for scope, source_id in SOURCE_IDS.items()}
        scope = metadata["scope"]; domain = per_source.get(scope)
        record = {"target": metadata["target"], "scope": scope, "scope_label": DISPLAY[scope], "scope_kind": metadata["scope_kind"], "strategy": metadata["strategy"], "strategy_family": metadata["strategy_parameters"]["family"], "seed": int(metadata["seed"]), "run_dir": str(run),
                  "source_balanced_mae": float(np.nanmean([x["mae"] for x in per_source.values()])),
                  "source_balanced_heavy_macro_mae": float(np.nanmean([x["heavy_macro"] for x in per_source.values()])),
                  "worst_source_mae": float(np.nanmax([x["mae"] for x in per_source.values()])),
                  "in_domain_mae": float(domain["mae"]) if domain else float("nan"), "in_domain_heavy_macro_mae": float(domain["heavy_macro"]) if domain else float("nan")}
        for source, metrics in per_source.items():
            record[f"source_{source}_mae"] = metrics["mae"]
            for label in HEAVY: record[f"source_{source}_{label}_mae"] = metrics[label]
        for label in HEAVY: record[f"source_balanced_{label}_mae"] = float(np.nanmean([x[label] for x in per_source.values()]))
        rows.append(record)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    numeric = [x for x in runs if x.endswith("_mae") or x in {"source_balanced_mae", "worst_source_mae"}]
    by = ["target", "scope", "scope_label", "scope_kind", "strategy", "strategy_family"]
    grouped = runs.groupby(by, as_index=False); result = grouped[numeric].mean(); counts = grouped.size().rename(columns={"size": "seeds"})
    std = grouped[numeric].std().rename(columns={x: f"{x}_sd" for x in numeric})
    result = result.merge(counts, on=by, validate="one_to_one").merge(std, on=by, validate="one_to_one")
    for column in numeric: result[f"{column}_ci95"] = 1.96 * result[f"{column}_sd"].fillna(0) / np.sqrt(result.seeds)
    return result


def select(summary: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for (target, scope), group in summary.groupby(["target", "scope"], sort=False):
        score = "source_balanced_mae" if scope == "combined" else "in_domain_mae"; heavy = "source_balanced_heavy_macro_mae" if scope == "combined" else "in_domain_heavy_macro_mae"
        baseline = group.loc[group.strategy.eq("all_mae_baseline")]
        if len(baseline) != 1: raise RuntimeError(f"{target}/{scope}: unweighted MAE baseline missing")
        baseline_row = baseline.iloc[0]; overall = group.sort_values([score, heavy]).iloc[0].copy(); eligible = group[group[score] <= baseline_row[score] * 1.01]
        heavy_row = (eligible if not eligible.empty else group).sort_values([heavy, score]).iloc[0].copy()
        for role, row, guardrail in (("overall", overall, True), ("heavy", heavy_row, not eligible.empty)):
            row["selection_role"] = role; row["selection_metric"] = score if role == "overall" else heavy; row["selection_guardrail_met"] = guardrail
            row["baseline_selection_metric"] = float(baseline_row[row["selection_metric"]]); row["selection_delta_vs_baseline"] = float(row[row["selection_metric"]] - baseline_row[row["selection_metric"]]); chosen.append(row)
    return pd.DataFrame(chosen).reset_index(drop=True)


def save(fig: plt.Figure, base: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"): fig.savefig(base.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_historical_validation(registry: Path, expected_manifest_hash: str) -> pd.DataFrame:
    """Summarise frozen historical controls on their original validation run.

    These are context-only validation comparators, never candidates for this
    sweep's source-specific strategy selection.
    """
    if not registry.is_file():
        return pd.DataFrame()
    payload = json.loads(registry.read_text(encoding="utf-8"))
    if payload.get("dataset_manifest_sha256") != expected_manifest_hash or payload.get("selection_partition") != "validation":
        raise ValueError("Historical registry does not prove the same validation split")
    rows = []
    for candidate in payload["candidates"]:
        for member in candidate["seeds"]:
            path = Path(member["run_dir"]) / "validation_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Historical validation metric missing: {path}")
            metrics = json.loads(path.read_text(encoding="utf-8"))
            by_source = {scope: source_metrics(metrics, source_id) for scope, source_id in SOURCE_IDS.items()}
            row = {"target": candidate["target"], "scope": "historical_combined", "scope_label": "Historical combined", "scope_kind": "historical_combined", "strategy": candidate["strategy"], "strategy_family": "historical", "seed": int(member["seed"]), "run_dir": member["run_dir"], "selection_role": candidate["selection_role"],
                   "source_balanced_mae": float(np.nanmean([x["mae"] for x in by_source.values()])),
                   "source_balanced_heavy_macro_mae": float(np.nanmean([x["heavy_macro"] for x in by_source.values()])),
                   "worst_source_mae": float(np.nanmax([x["mae"] for x in by_source.values()]))}
            for source, values in by_source.items():
                row[f"source_{source}_mae"] = values["mae"]
                for label in HEAVY: row[f"source_{source}_{label}_mae"] = values[label]
            for label in HEAVY: row[f"source_balanced_{label}_mae"] = float(np.nanmean([x[label] for x in by_source.values()]))
            rows.append(row)
    result = summarize(pd.DataFrame(rows))
    result["selection_role"] = [next(x["selection_role"] for x in payload["candidates"] if x["target"] == target) for target in result.target]
    return result


def plots(summary: pd.DataFrame, finalists: pd.DataFrame, runs: pd.DataFrame, historical: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True); targets = [x for x in ("source_native", "chronologer_hi", "anchor_irt") if x in set(summary.target)]
    # Main figure: no shared target-space claim; all axes are observed minutes.
    fig, axes = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), 5.1), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = summary[summary.target.eq(target)]
        for scope, group in subset.groupby("scope", sort=False): axis.scatter(group.source_balanced_mae, group.source_balanced_heavy_macro_mae, s=32, color=COLORS[scope], alpha=.50, edgecolor="white", linewidth=.35)
        for _, row in finalists[finalists.target.eq(target)].iterrows(): axis.scatter(row.source_balanced_mae, row.source_balanced_heavy_macro_mae, s=78, marker="o" if row.selection_role == "overall" else "D", color=COLORS[row.scope], edgecolor="#202020", linewidth=.65, zorder=3)
        axis.set(title=TARGET_DISPLAY[target], xlabel="Source-balanced validation MAE (min)", ylabel="Heavy-Macro MAE (min)"); axis.grid(alpha=.22)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=COLORS[x], label=DISPLAY[x]) for x in COLORS if x in set(summary.scope)] + [Line2D([0], [0], marker="o", linestyle="", color="#202020", markerfacecolor="white", label="Overall finalist"), Line2D([0], [0], marker="D", linestyle="", color="#202020", markerfacecolor="white", label="Heavy finalist")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False); fig.suptitle("Validation trade-off: overall versus Heavy-Dimethyl error", y=1.02); fig.tight_layout(rect=(0, .09, 1, 1)); save(fig, output / "fig1_validation_overall_heavy_pareto")
    # Finalist forest: target-faceted and source-only ranking metric honoured.
    fig, axes = plt.subplots(1, 2, figsize=(13.8, max(5.4, .34 * len(finalists) + 1.6)), sharey=True); ordered = finalists.sort_values(["target", "scope", "selection_role"]).reset_index(drop=True); y = np.arange(len(ordered)); labels = [f"{TARGET_DISPLAY[x.target]} · {DISPLAY[x.scope]} · {x.selection_role}" for _, x in ordered.iterrows()]
    axes[0].errorbar(ordered.source_balanced_mae, y, xerr=ordered.source_balanced_mae_ci95, fmt="o", color="#245A83", ecolor="#7A8793", capsize=2.3); axes[1].errorbar(ordered.source_balanced_heavy_macro_mae, y, xerr=ordered.source_balanced_heavy_macro_mae_ci95, fmt="D", color="#A84E70", ecolor="#7A8793", capsize=2.3)
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Source-balanced MAE (min)", title="Overall validation"); axes[1].set(xlabel="Source-balanced Heavy-Macro MAE (min)", title="Heavy-Dimethyl validation")
    for axis in axes: axis.grid(axis="x", alpha=.22)
    fig.suptitle("Validation-selected candidates; bars are 95% normal-approximation CIs across three seeds", y=1.01); fig.tight_layout(); save(fig, output / "fig2_validation_selected_candidates")
    # Honest transfer diagnostic: source-only candidates, own-domain vs all-domain performance.
    source_only = finalists[finalists.scope_kind.eq("source_specific")].copy(); fig, axes = plt.subplots(1, len(targets), figsize=(5.8 * len(targets), 4.9), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = source_only[source_only.target.eq(target)]
        for _, row in subset.iterrows(): axis.scatter(row.in_domain_mae, row.source_balanced_mae, s=70, marker="o" if row.selection_role == "overall" else "D", color=COLORS[row.scope], edgecolor="#202020", linewidth=.5); axis.annotate(DISPLAY[row.scope], (row.in_domain_mae, row.source_balanced_mae), xytext=(4, 3), textcoords="offset points", fontsize=7)
        values = np.r_[subset.in_domain_mae.to_numpy(), subset.source_balanced_mae.to_numpy()]; lower, upper = np.nanmin(values), np.nanmax(values); pad = max(.004, (upper - lower) * .08); axis.plot([lower-pad, upper+pad], [lower-pad, upper+pad], color="#59636E", lw=.9, ls="--")
        axis.set(title=TARGET_DISPLAY[target], xlabel="In-domain validation MAE (min)", ylabel="Source-balanced MAE across all four sources (min)"); axis.grid(alpha=.22)
    fig.suptitle("In-domain fit versus cross-source transfer; diagonal indicates equal error", y=1.02); fig.tight_layout(); save(fig, output / "fig3_validation_in_domain_transfer")
    # Heavy strata show whether macro optimisation hides a failing stratum.
    columns = [f"source_balanced_{x}_mae" for x in HEAVY]; fig, axes = plt.subplots(1, len(targets), figsize=(5.8 * len(targets), 5.0), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = finalists[finalists.target.eq(target)].reset_index(drop=True); x = np.arange(len(subset)); width = .25
        for offset, (column, label, color) in enumerate(zip(columns, ("N-term", "Lys", "Both"), ("#2F6B9A", "#C47A18", "#A84E70"))): axis.bar(x + (offset-1)*width, subset[column], width, color=color, label=label)
        axis.set(title=TARGET_DISPLAY[target], xticks=x, xticklabels=[f"{SHORT_DISPLAY.get(r.scope, DISPLAY[r.scope])}\n{r.selection_role}" for _, r in subset.iterrows()], ylabel="Source-balanced MAE (min)"); axis.tick_params(axis="x", labelsize=7); axis.grid(axis="y", alpha=.22)
    axes[0][0].legend(title="Heavy stratum", frameon=False); fig.suptitle("Heavy-Dimethyl strata of validation-selected candidates", y=1.02); fig.tight_layout(); save(fig, output / "fig4_validation_heavy_strata")

    # Coordinate robustness, source by source: only like-for-like in-domain measurements.
    overall = source_only[source_only.selection_role.eq("overall")].copy(); fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), sharey=False); target_colors = {"source_native": "#245A83", "chronologer_hi": "#C47A18", "anchor_irt": "#A84E70"}
    for axis, source in zip(axes.flat, SOURCE_ORDER):
        subset = overall[overall.scope.eq(source)]; positions = {target: index for index, target in enumerate(targets)}
        for _, row in subset.iterrows():
            axis.errorbar(positions[row.target], row.in_domain_mae, yerr=row.in_domain_mae_ci95, fmt="o", color=target_colors[row.target], capsize=3, markersize=6)
        if source == "human_hunter": axis.text(positions["anchor_irt"], axis.get_ylim()[0], "not applicable\n(identical target)", ha="center", va="bottom", fontsize=7, color="#59636E")
        axis.set(title=DISPLAY[source], xticks=range(len(targets)), xticklabels=[TARGET_DISPLAY[x] for x in targets], ylabel="In-domain validation MAE (min)"); axis.tick_params(axis="x", rotation=18, labelsize=8); axis.grid(axis="y", alpha=.22)
    fig.suptitle("In-domain accuracy of the overall finalist by RT coordinate", y=1.01); fig.tight_layout(); save(fig, output / "fig5_validation_coordinate_robustness_by_source")

    # The full direction of transfer is hidden in a source-balanced average; expose it directly.
    fig, axes = plt.subplots(1, len(targets), figsize=(5.6 * len(targets), 5.0), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = overall[overall.target.eq(target)].sort_values("scope"); matrix = subset.set_index("scope")[[f"source_{source}_mae" for source in SOURCE_ORDER]]
        image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto")
        axis.set(title=TARGET_DISPLAY[target], xticks=np.arange(len(SOURCE_ORDER)), xticklabels=[SHORT_DISPLAY[x] for x in SOURCE_ORDER], yticks=np.arange(len(matrix)), yticklabels=[DISPLAY[x] for x in matrix.index], xlabel="Evaluation source", ylabel="Training source")
        axis.tick_params(axis="x", rotation=24, labelsize=8)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]): axis.text(column, row, f"{matrix.iloc[row, column]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, label="Observed-minute MAE")
    fig.suptitle("Cross-source transfer of overall source-specific finalists", y=1.02); fig.tight_layout(); save(fig, output / "fig6_validation_transfer_matrix")

    # This is the mechanism screen: every strategy against its own unweighted baseline.
    source_summary = summary[summary.scope_kind.eq("source_specific")].copy(); baseline = source_summary[source_summary.strategy.eq("all_mae_baseline")][["target", "scope", "in_domain_mae", "in_domain_heavy_macro_mae"]].rename(columns={"in_domain_mae": "baseline_overall", "in_domain_heavy_macro_mae": "baseline_heavy"})
    delta = source_summary.merge(baseline, on=["target", "scope"], validate="many_to_one"); delta["delta_overall"] = delta.in_domain_mae - delta.baseline_overall; delta["delta_heavy"] = delta.in_domain_heavy_macro_mae - delta.baseline_heavy
    family_colors = {"core": "#245A83", "weighting": "#C47A18", "oversampling": "#718657", "interaction": "#A84E70", "discriminative": "#7651A6"}; scope_markers = {"celegans_exploris": "o", "triplemix_source_a": "s", "triplemix_source_b": "^", "human_hunter": "D"}
    fig, axes = plt.subplots(1, len(targets), figsize=(5.7 * len(targets), 5.0), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = delta[delta.target.eq(target)]
        for _, row in subset.iterrows(): axis.scatter(row.delta_overall, row.delta_heavy, color=family_colors[row.strategy_family], marker=scope_markers[row.scope], s=30, alpha=.60, edgecolor="white", linewidth=.25)
        axis.axhline(0, color="#5B6570", lw=.8); axis.axvline(0, color="#5B6570", lw=.8); axis.set(title=TARGET_DISPLAY[target], xlabel="Δ overall in-domain MAE vs baseline (min)", ylabel="Δ Heavy-Macro MAE vs baseline (min)"); axis.grid(alpha=.18)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=family) for family, color in family_colors.items()] + [Line2D([0], [0], marker=marker, linestyle="", color="#3C4650", label=DISPLAY[scope]) for scope, marker in scope_markers.items() if scope in set(delta.scope)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False); fig.suptitle("Strategy effects relative to each source's unweighted baseline", y=1.02); fig.tight_layout(rect=(0, .11, 1, 1)); save(fig, output / "fig7_validation_strategy_effects")

    # Seed spread protects against selecting a lucky initialization.
    seed_spread = runs[runs.scope_kind.eq("source_specific")].groupby(["target", "scope", "strategy"], as_index=False).in_domain_mae.std().rename(columns={"in_domain_mae": "seed_sd"})
    fig, axes = plt.subplots(1, len(targets), figsize=(5.6 * len(targets), 4.8), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = seed_spread[seed_spread.target.eq(target)]; scopes = [scope for scope in SOURCE_ORDER if scope in set(subset.scope)]; values = [subset[subset.scope.eq(scope)].seed_sd.dropna().to_numpy() for scope in scopes]
        boxes = axis.boxplot(values, labels=[DISPLAY[x] for x in scopes], patch_artist=True, showfliers=False)
        for patch, scope in zip(boxes["boxes"], scopes): patch.set(facecolor=COLORS[scope], alpha=.72)
        axis.set(title=TARGET_DISPLAY[target], ylabel="Across-seed SD of in-domain MAE (min)"); axis.tick_params(axis="x", rotation=20, labelsize=8); axis.grid(axis="y", alpha=.22)
    fig.suptitle("Optimization stability across all 42 strategies", y=1.02); fig.tight_layout(); save(fig, output / "fig8_validation_seed_stability")

    # Show the realised Heavy objective directly rather than only its three strata.
    fig, axes = plt.subplots(1, len(targets), figsize=(5.7 * len(targets), 4.9), squeeze=False)
    for axis, target in zip(axes[0], targets):
        subset = source_only[source_only.target.eq(target)]
        for scope, group in subset.groupby("scope", sort=False):
            overall_row = group[group.selection_role.eq("overall")].iloc[0]; heavy_row = group[group.selection_role.eq("heavy")].iloc[0]
            axis.plot([overall_row.in_domain_mae, heavy_row.in_domain_mae], [overall_row.in_domain_heavy_macro_mae, heavy_row.in_domain_heavy_macro_mae], color=COLORS[scope], lw=1.1, alpha=.75)
            axis.scatter(overall_row.in_domain_mae, overall_row.in_domain_heavy_macro_mae, color=COLORS[scope], marker="o", s=52, edgecolor="#202020", linewidth=.35); axis.scatter(heavy_row.in_domain_mae, heavy_row.in_domain_heavy_macro_mae, color=COLORS[scope], marker="D", s=52, edgecolor="#202020", linewidth=.35)
        axis.set(title=TARGET_DISPLAY[target], xlabel="In-domain overall MAE (min)", ylabel="In-domain Heavy-Macro MAE (min)"); axis.grid(alpha=.22)
    role_handles = [Line2D([0], [0], marker="o", linestyle="", color="#303840", label="Overall finalist"), Line2D([0], [0], marker="D", linestyle="", color="#303840", label="Heavy finalist")] + [Line2D([0], [0], marker="o", linestyle="", color=COLORS[scope], label=DISPLAY[scope]) for scope in SOURCE_ORDER]
    fig.legend(handles=role_handles, loc="lower center", ncol=3, frameon=False); fig.suptitle("Realised overall–Heavy trade-off in each training source", y=1.02); fig.tight_layout(rect=(0, .10, 1, 1)); save(fig, output / "fig9_validation_within_source_heavy_tradeoff")

    # Valid validation-only contextual comparison to frozen, exact-split pooled controls.
    comparison_rows = []
    for target in targets:
        if target == "source_native":
            reference = finalists[(finalists.target.eq(target)) & (finalists.scope.eq("combined")) & (finalists.selection_role.eq("overall"))]
            candidates = overall[overall.target.eq(target)]
        elif target == "chronologer_hi":
            reference = historical[(historical.target.eq(target)) & (historical.selection_role.eq("overall"))]
            candidates = overall[overall.target.eq(target)]
        else:
            reference = historical[(historical.target.eq(target)) & (historical.selection_role.eq("heavy_guardrail"))]
            candidates = source_only[(source_only.target.eq(target)) & (source_only.selection_role.eq("heavy"))]
        if reference.empty: continue
        ref = reference.iloc[0]
        for _, row in candidates.iterrows():
            item = {"target": target, "scope": row.scope, "reference": "new combined" if target == "source_native" else "frozen historical combined"}
            for source in SOURCE_ORDER: item[f"source_{source}"] = row[f"source_{source}_mae"] - ref[f"source_{source}_mae"]
            comparison_rows.append(item)
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        fig, axes = plt.subplots(1, len(targets), figsize=(5.8 * len(targets), 5.0), squeeze=False)
        for axis, target in zip(axes[0], targets):
            subset = comparison[comparison.target.eq(target)].set_index("scope")[[f"source_{x}" for x in SOURCE_ORDER]]; limit = max(.01, float(np.nanmax(np.abs(subset.to_numpy(float)))))
            image = axis.imshow(subset, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"); axis.set(title=TARGET_DISPLAY[target], xticks=np.arange(len(SOURCE_ORDER)), xticklabels=[SHORT_DISPLAY[x] for x in SOURCE_ORDER], yticks=np.arange(len(subset)), yticklabels=[DISPLAY[x] for x in subset.index], xlabel="Evaluation source", ylabel="Source-specific training")
            axis.tick_params(axis="x", rotation=24, labelsize=8)
            for row in range(subset.shape[0]):
                for column in range(subset.shape[1]): axis.text(column, row, f"{subset.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=7)
            fig.colorbar(image, ax=axis, label="MAE difference (min; negative favours source-specific)")
        fig.suptitle("Validation context: source-specific model versus target-matched combined reference", y=1.02); fig.tight_layout(); save(fig, output / "fig10_validation_source_specific_vs_combined")


def main() -> None:
    args = parse_args(); study = args.study_root.resolve(); manifest = json.loads((study / "study_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("validation_only") or manifest.get("test_partition") != "closed_until_finalist_selection": raise ValueError("Only closed-test manifests are accepted")
    runs = collect(manifest)
    if len(runs) != len(manifest["jobs"]): raise RuntimeError("Validation results do not cover every job")
    grouped = summarize(runs)
    if not (grouped.seeds == 3).all(): raise RuntimeError("All configurations need three seeds before selection")
    finalists = select(grouped); analysis = study / "analysis"; analysis.mkdir(exist_ok=True)
    historical = load_historical_validation(analysis / "historical_combined_registry.json", manifest["dataset_manifest_sha256"])
    runs.to_csv(analysis / "validation_runs.csv", index=False); grouped.to_csv(analysis / "validation_strategy_summary.csv", index=False); finalists.to_csv(analysis / "validation_selected_candidates.csv", index=False)
    payload = {"status": "validation_selection_frozen_candidate_list", "warning": "Selection uses validation only; do not run final test until this candidate list is frozen.", "dataset_manifest_sha256": manifest["dataset_manifest_sha256"], "canonical_dataset_dir": manifest["canonical_dataset_dir"], "prepared_target_root": manifest["prepared_target_root"], "selection_partition": "validation", "candidates": []}
    for _, row in finalists.iterrows():
        members = runs[(runs.target == row.target) & (runs.scope == row.scope) & (runs.strategy == row.strategy)].sort_values("seed")
        payload["candidates"].append({"label": f"{TARGET_DISPLAY[row.target]} · {DISPLAY[row.scope]} · {row.selection_role}", "scope": row.scope, "scope_kind": row.scope_kind, "selection_role": row.selection_role, "strategy": row.strategy, "target": row.target, "selection_metric": row.selection_metric, "selection_value": float(row[row.selection_metric]), "selection_guardrail_met": bool(row.selection_guardrail_met), "seeds": [{"seed": int(x.seed), "run_dir": x.run_dir} for _, x in members.iterrows()]})
    (analysis / "validation_selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not historical.empty: historical.to_csv(analysis / "historical_validation_comparators.csv", index=False)
    plots(grouped, finalists, runs, historical, analysis / "plots")
    shutil.copyfile(Path(__file__).with_name("diann_source_specific_plot_guide.md"), analysis / "plots" / "README.md")
    (analysis / "README.md").write_text("# Validation-only analysis\n\nAll rankings and figures use the frozen validation partition only. Every selected candidate has three optimizer seeds. Overall finalists minimize the scope's intended overall metric. Heavy finalists minimize the corresponding Heavy-Macro metric with a 1% overall-MAE guardrail relative to the unweighted MAE baseline. Coordinates are never mixed inside a plot panel; every plotted value is converted to observed RT minutes. `validation_selection.json` is the only allowed input to the explicitly gated final-test command.\n\n## Figure map\n\n- `fig1` overall-versus-Heavy Pareto screen; `fig2` selected-candidate uncertainty; `fig3` pooled-versus-in-domain transfer diagnostic; `fig4` Heavy strata.\n- `fig5` coordinate robustness separately for each source; `fig6` directed source-transfer matrices; `fig7` all strategy effects relative to each source baseline; `fig8` optimization-seed stability; `fig9` realised overall/Heavy selection trade-off; `fig10` source-specific change against a target-matched new or frozen historical combined comparator.\n\n`fig10` is validation context only, not a final-test claim. The Chronologer and Anchor pooled controls are frozen historical exact-split models; the source-gradient panel uses the new in-sweep combined control.\n", encoding="utf-8")
    print(f"Validation analysis complete: {analysis}")


if __name__ == "__main__": main()
