#!/usr/bin/env python3
"""Analyse the validation-only Dimethyl-first DIA-NN loss sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SELECTIONS = ("100_0", "75_25", "50_50", "25_75", "0_100")
METRICS = ("overall_mae", "normal_mae", "dimethyl_macro_mae", "worst_dimethyl_mae", "heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae", "dimethyl_range_mae", "semitryptic_mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mae(record: dict | None) -> float | None:
    if not record or int(record.get("rows", 0)) <= 0 or record.get("mae") is None:
        return None
    return float(record["mae"])


def source_balanced(sources: dict, key: str) -> float | None:
    values = [mae(source.get(key)) for source in sources.values()]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def output_dir(job: dict) -> Path:
    command = job["commands"][0]
    return Path(command[command.index("--output-dir") + 1])


def collect_job(job: dict) -> tuple[list[dict], list[dict]]:
    run = output_dir(job)
    rows, organism_rows = [], []
    for selection in SELECTIONS:
        path = run / "validation_metrics" / f"checkpoint_{selection}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        sources = result.get("sources", {})
        nterm = source_balanced(sources, "raw_heavy_heavy_nterm")
        lys = source_balanced(sources, "raw_heavy_heavy_lysine")
        both = source_balanced(sources, "raw_heavy_heavy_both")
        values = [nterm, lys, both]
        row = {
            **job["metadata"], "job_id": job["id"], "run_dir": str(run),
            "checkpoint_selection": selection, "checkpoint_path": result.get("checkpoint_path"),
            "overall_mae": result.get("source_balanced_raw_mae"),
            "normal_mae": source_balanced(sources, "raw_heavy_normal"),
            "heavy_nterm_mae": nterm, "heavy_lysine_mae": lys, "heavy_both_mae": both,
            "semitryptic_mae": source_balanced(sources, "raw_digestion_semi_tryptic"),
            "dimethyl_macro_mae": sum(values) / 3 if all(value is not None for value in values) else None,
            "worst_dimethyl_mae": max(values) if all(value is not None for value in values) else None,
            "dimethyl_range_mae": max(values) - min(values) if all(value is not None for value in values) else None,
        }
        rows.append(row)
        for organism, metrics in result.get("organisms", {}).items():
            org = {
                "heavy_nterm_mae": mae(metrics.get("raw_heavy_heavy_nterm")),
                "heavy_lysine_mae": mae(metrics.get("raw_heavy_heavy_lysine")),
                "heavy_both_mae": mae(metrics.get("raw_heavy_heavy_both")),
            }
            if all(value is not None for value in org.values()):
                organism_rows.append({
                    **row, "organism": organism, **org,
                    "dimethyl_macro_mae": sum(org.values()) / 3,
                    "worst_dimethyl_mae": max(org.values()),
                    "dimethyl_range_mae": max(org.values()) - min(org.values()),
                })
    return rows, organism_rows


def aggregate(frame: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    result = frame.groupby(group, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    result.columns = [col if isinstance(col, str) else "_".join(item for item in col if item) for col in result.columns]
    return result


def pareto(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    frame = frame.reset_index(drop=True)
    for index, row in frame.iterrows():
        other = frame.drop(index)
        dominated = (
            other["dimethyl_macro_mae_mean"].le(row["dimethyl_macro_mae_mean"])
            & other["worst_dimethyl_mae_mean"].le(row["worst_dimethyl_mae_mean"])
            & (other["dimethyl_macro_mae_mean"].lt(row["dimethyl_macro_mae_mean"]) | other["worst_dimethyl_mae_mean"].lt(row["worst_dimethyl_mae_mean"]))
        )
        if not dominated.any():
            keep.append(index)
    return frame.loc[keep].sort_values(["dimethyl_macro_mae_mean", "worst_dimethyl_mae_mean"])


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.sweep_dir / "sweep_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("study") != "diann_dimethyl_loss_focus_fixed_steps_v2_chronologer_anchor" or not manifest.get("validation_only") or manifest.get("test_partition") != "closed":
        raise ValueError("Expected the validation-only, test-closed Dimethyl-first study")
    all_rows, all_organisms = [], []
    for job in manifest["jobs"]:
        rows, organisms = collect_job(job)
        all_rows.extend(rows)
        all_organisms.extend(organisms)
    candidates = pd.DataFrame(all_rows)
    if len(candidates) != 324 * len(SELECTIONS) or candidates["job_id"].nunique() != 324:
        raise RuntimeError(f"Incomplete sweep: {candidates['job_id'].nunique()} runs and {len(candidates)} checkpoint metrics")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / "checkpoint_candidates.csv", index=False)
    organisms = pd.DataFrame(all_organisms)
    organisms.to_csv(output / "organism_checkpoint_candidates.csv", index=False)
    group = ["target", "loss_profile", "sampling_profile", "lr", "checkpoint_selection"]
    summary = aggregate(candidates, group, list(METRICS)).sort_values(["checkpoint_selection", "dimethyl_macro_mae_mean", "worst_dimethyl_mae_mean"])
    summary.to_csv(output / "configuration_summary.csv", index=False)
    organism_summary = aggregate(organisms, [*group, "organism"], ["dimethyl_macro_mae", "worst_dimethyl_mae", "heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae"])
    organism_summary.to_csv(output / "organism_summary.csv", index=False)
    primary = summary[summary["checkpoint_selection"].eq("0_100")].copy()
    primary["rank_dimethyl_macro"] = primary.groupby("target")["dimethyl_macro_mae_mean"].rank(method="min")
    primary["rank_worst_dimethyl"] = primary.groupby("target")["worst_dimethyl_mae_mean"].rank(method="min")
    primary["rank_balanced_dimethyl"] = primary["rank_dimethyl_macro"] + primary["rank_worst_dimethyl"]
    primary = primary.sort_values(["target", "rank_balanced_dimethyl", "dimethyl_macro_mae_mean"])
    primary.to_csv(output / "primary_0_100_rankings.csv", index=False)
    pd.concat([pareto(part) for _, part in primary.groupby("target", sort=True)]).to_csv(output / "primary_0_100_pareto.csv", index=False)
    keys = ["target", "sampling_profile", "lr", "seed", "checkpoint_selection"]
    c0 = candidates[candidates.loss_profile.eq("C0_standard")][keys + list(METRICS)].rename(columns={name: f"c0_{name}" for name in METRICS})
    loss_delta = candidates.merge(c0, on=keys, how="left")
    for name in METRICS:
        loss_delta[f"delta_vs_c0_{name}"] = loss_delta[name] - loss_delta[f"c0_{name}"]
    loss_delta.to_csv(output / "loss_profile_seed_matched_deltas.csv", index=False)
    r0 = candidates[candidates.sampling_profile.eq("R0")][["target", "loss_profile", "lr", "seed", "checkpoint_selection", *METRICS]].rename(columns={name: f"r0_{name}" for name in METRICS})
    dose_delta = candidates.merge(r0, on=["target", "loss_profile", "lr", "seed", "checkpoint_selection"], how="left")
    for name in METRICS:
        dose_delta[f"delta_vs_r0_{name}"] = dose_delta[name] - dose_delta[f"r0_{name}"]
    dose_delta.to_csv(output / "oversampling_seed_matched_deltas.csv", index=False)
    (output / "README.md").write_text("# Dimethyl-first analysis\n\nPrimary ranking: `0_100` checkpoint, source-balanced observed-minute Dimethyl macro MAE. The three exact Dimethyl strata and their worst value are always retained. Test remains closed.\n", encoding="utf-8")
    print(json.dumps({"runs": 324, "checkpoint_candidates": len(candidates), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
