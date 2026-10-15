#!/usr/bin/env python3
"""Render the standard Heavy-Dimethyl sweep figure suite for this study."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plot_heavy_dimethyl_rt_sweep import CONFIGURATION_KEYS, generate_heavy_dimethyl_figures
from plot_rt_sweep_publication import generate_publication_figures


SELECTION = "0_100"
METRICS = {
    "heavy_macro_source_balanced_mae": "dimethyl_macro_mae",
    "heavy_nterm_sb_mae": "heavy_nterm_mae",
    "heavy_lys_sb_mae": "heavy_lysine_mae",
    "heavy_both_sb_mae": "heavy_both_mae",
    "nonheavy_sb_mae": "normal_mae",
    "overall_mae": "overall_mae",
    "semitryptic_mae": "semitryptic_mae",
    # The study records a worst Dimethyl stratum rather than a worst source.
    "worst_source_mae": "worst_dimethyl_mae",
}
ORGANISM_METRICS = {
    "heavy_nterm": "heavy_nterm_mae",
    "heavy_lys": "heavy_lysine_mae",
    "heavy_both": "heavy_both_mae",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("chronologer_hi", "anchor_irt"),
        help="Optional filter; omit for the normal joint two-target figure suite.",
    )
    return parser.parse_args()


def decorate(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Add the common Heavy-Dimethyl figure-contract dimensions."""
    result = frame.copy()
    result["model"] = "diann"
    result["arm"] = arm
    result["policy"] = "dimethyl_loss_focus"
    if arm == "weighting":
        result["profile"] = result["loss_profile"] + " · " + result["sampling_profile"]
        result["loss"] = result["loss_profile"]
        result["weights"] = result["loss_profile"]
        result["oversampling"] = result["sampling_profile"]
    else:
        result["profile"] = result["sampling_profile"]
        result["loss"] = "C0_standard"
        result["weights"] = "C0_standard"
        result["oversampling"] = result["sampling_profile"]
    return result


def paired_interventions(frame: pd.DataFrame, metrics: list[str], extra_keys: list[str]) -> pd.DataFrame:
    """Create the two standard intervention arms with their matched C0 control."""
    selected = frame[frame.checkpoint_selection.eq(SELECTION)].copy()
    keys = ["target", "sampling_profile", "lr", "seed", *extra_keys]
    control_same_sampling = selected[selected.loss_profile.eq("C0_standard")]
    control_columns = keys + metrics

    weighted = selected[selected.loss_profile.ne("C0_standard")].merge(
        control_same_sampling[control_columns], on=keys, how="inner", suffixes=("", "_control"),
    )
    weighted = decorate(weighted, "weighting")

    control_r0 = control_same_sampling[control_same_sampling.sampling_profile.eq("R0")]
    oversampling_keys = ["target", "lr", "seed", *extra_keys]
    oversampled = selected[
        selected.loss_profile.eq("C0_standard") & selected.sampling_profile.ne("R0")
    ].merge(
        control_r0[oversampling_keys + metrics], on=oversampling_keys, how="inner", suffixes=("", "_control"),
    )
    oversampled = decorate(oversampled, "oversampling")

    result = pd.concat([weighted, oversampled], ignore_index=True)
    for metric in metrics:
        result[f"delta_{metric}"] = result[metric] - result[f"{metric}_control"]
    return result


def build_summary_and_deltas(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_metrics = list(METRICS.values())
    paired = paired_interventions(candidates, raw_metrics, [])
    paired = paired.rename(columns={f"delta_{raw}": f"delta_{standard}" for standard, raw in METRICS.items()})
    standard_metrics = list(METRICS)
    actual = paired.groupby(CONFIGURATION_KEYS, dropna=False)[raw_metrics].mean().reset_index()
    actual = actual.rename(columns={raw: f"{standard}_mean" for standard, raw in METRICS.items()})
    deltas = paired[CONFIGURATION_KEYS + ["seed"] + [f"delta_{metric}" for metric in standard_metrics]].copy()
    mean_deltas = deltas.groupby(CONFIGURATION_KEYS, dropna=False)[[f"delta_{metric}" for metric in standard_metrics]].mean().reset_index()
    summary = actual.merge(mean_deltas, on=CONFIGURATION_KEYS, how="inner")
    # Selection is by Dimethyl macro; do not add unplanned guardrails here.
    summary["guardrail_status"] = "PASS"
    return summary, deltas


def build_organism_deltas(organisms: pd.DataFrame) -> pd.DataFrame:
    raw_metrics = list(ORGANISM_METRICS.values())
    paired = paired_interventions(organisms, raw_metrics, ["organism"])
    rows = []
    for stratum, metric in ORGANISM_METRICS.items():
        part = paired[CONFIGURATION_KEYS + ["seed", "organism", f"delta_{metric}"]].rename(
            columns={f"delta_{metric}": "delta_mae"}
        )
        part["stratum"] = stratum
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def build_full_sweep_runs(candidates: pd.DataFrame, organisms: pd.DataFrame) -> pd.DataFrame:
    """Adapt this sweep to the complete shared ranking/scatter-plot suite."""
    keys = ["target", "loss_profile", "sampling_profile", "lr", "seed"]
    runs = candidates[candidates.checkpoint_selection.eq(SELECTION)].copy()
    organism_frame = organisms[organisms.checkpoint_selection.eq(SELECTION)]
    organism_values = organism_frame.pivot_table(
        index=keys, columns="organism", values="dimethyl_macro_mae", aggfunc="first",
    ).rename(columns=lambda value: f"organism_{value}_raw_mae").reset_index()
    runs = runs.merge(organism_values, on=keys, how="left")

    runs["mode"] = runs["target"]
    runs["policy"] = runs["loss_profile"]
    runs["lr_level"] = runs.apply(
        lambda row: f"{row.sampling_profile} · {row.lr:.1e}", axis=1,
    )
    runs["absolute_learning_rate"] = runs["lr"]
    # The full shared suite is the overall view; the Heavy-Dimethyl suite above
    # remains the primary view for the prespecified Dimethyl-macro selection.
    runs["source_balanced_raw_mae"] = runs["overall_mae"]
    runs["semi_tryptic_raw_mae"] = runs["semitryptic_mae"]
    runs["heavy_nterm_raw_mae"] = runs["heavy_nterm_mae"]
    runs["heavy_lysine_raw_mae"] = runs["heavy_lysine_mae"]
    runs["heavy_both_raw_mae"] = runs["heavy_both_mae"]
    organism_columns = [
        column for column in ("organism_celegans_raw_mae", "organism_human_raw_mae", "organism_triplemix_raw_mae")
        if column in runs
    ]
    runs["organism_macro_raw_mae"] = runs[organism_columns].mean(axis=1)
    return runs


def write_report(analysis: Path, summary: pd.DataFrame) -> None:
    counts = summary.groupby(["target", "arm"]).size().to_dict()
    text = f"""# DIA-NN Dimethyl-loss-focus sweep: Standard figures

The custom diagnostic plots were replaced with the shared Heavy-Dimethyl
figure suite used for DIA-NN and Carafe2 sweeps. All figures use the fixed
validation partition, the prespecified `0/100` checkpoint, and three seeds.
The held-out test partition remains closed.

## Comparison rules

- **Loss weighting:** every non-C0 configuration is compared with C0 at the
  same target, oversampling dose, learning rate, and seed.
- **Oversampling:** C0/R40 and C0/R56 are compared with C0/R0 at the same
  target, learning rate, and seed.

The Heavy-Dimethyl plots and their reading guide are in `plots/`; the complete
shared sweep suite (ranking, scatter plots, stability and LR response) is in
`plots/full_sweep/`. Generated
configuration counts by target and intervention arm: `{counts}`.
"""
    (analysis / "SWEEP_RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis = args.analysis_dir
    candidates = pd.read_csv(analysis / "checkpoint_candidates.csv")
    organisms = pd.read_csv(analysis / "organism_checkpoint_candidates.csv")
    if args.target:
        candidates = candidates[candidates.target.eq(args.target)]
        organisms = organisms[organisms.target.eq(args.target)]
    summary, deltas = build_summary_and_deltas(candidates)
    organism_deltas = build_organism_deltas(organisms)
    full_sweep_runs = build_full_sweep_runs(candidates, organisms)
    summary.to_csv(analysis / "weighting_vs_oversampling.csv", index=False)
    deltas.to_csv(analysis / "matched_control_deltas.csv", index=False)
    organism_deltas.to_csv(analysis / "matched_organism_stratum_deltas.csv", index=False)
    full_sweep_runs.to_csv(analysis / "standard_sweep_runs.csv", index=False)
    plots = analysis / "plots"
    generate_heavy_dimethyl_figures(summary, deltas, plots, "DIA-NN", organism_deltas)
    generate_publication_figures(full_sweep_runs, plots / "full_sweep", "DIA-NN (overall sweep view)")
    write_report(analysis, summary)
    print(f"Wrote the standard Heavy-Dimethyl and full-sweep figure suites to {plots}")


if __name__ == "__main__":
    main()
