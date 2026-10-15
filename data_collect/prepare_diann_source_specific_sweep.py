#!/usr/bin/env python3
"""Prepare the final closed-test DIA-NN source-specific RT sweep.

The canonical v1 split is never changed. This study makes three target
coordinates *after* that global, peptidoform-disjoint split:
source_native (original Carafe2); frozen Chronologer; frozen Human anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIANN = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune")).expanduser()
DIANN_MODEL = Path(os.environ.get("DIANN_MODEL_DIR", ROOT.parent / "diann_sc" / "DIA-NN-2.5.0" / "diann-2.5.0" / "models")).expanduser()
DATASET_DEFAULT = ROOT / "trainingdata" / "finalrt_4source_trainfit15_validation10_v1"
TARGETS = ("source_native", "chronologer_hi", "anchor_irt")
SEEDS = (2024, 2025, 2026)
KEYS = ("sequence", "mods", "mod_sites")
SOURCES = {
    "celegans_exploris": {"source_id": "celegans_exploris_44min_v1", "organism": "celegans", "label": "C. elegans (Exploris)"},
    "triplemix_source_a": {"source_id": "triplemix_source_a_entrap_neu_filtered_v1", "organism": "triplemix", "label": "TripleMix (Source A)"},
    "triplemix_source_b": {"source_id": "triplemix_pd_hamilton_60min_v1", "organism": "triplemix", "label": "TripleMix (Source B)"},
    "human_hunter": {"source_id": "human_source_hunter_human_v1", "organism": "human", "label": "Human (Hunter)"},
}


def profile_label(values: tuple[float, float, float]) -> str:
    return "_".join(f"{value:g}".replace(".", "p") for value in values)


def build_strategies() -> dict[str, dict]:
    """Balanced 42-configuration screen, identical in every target/scope cell."""
    strategies: dict[str, dict] = {}

    def add(name: str, family: str, **parameters: object) -> None:
        if name in strategies:
            raise AssertionError(f"Duplicate strategy: {name}")
        strategies[name] = {"family": family, **parameters}

    add("all_mae_baseline", "core", loss="mae", trainable="all", lr=3e-4)
    add("all_mae_low_lr", "core", loss="mae", trainable="all", lr=2e-4)
    add("all_mae_high_lr", "core", loss="mae", trainable="all", lr=4.5e-4)
    for delta in (.02, .03, .05):
        add(f"all_huber_d{delta:.2f}".replace(".", "p"), "core", loss="huber", huber_delta=delta, trainable="all", lr=3e-4)
    for lr in (2e-4, 4.5e-4):
        add(f"all_huber_d03_lr{lr:g}".replace(".", "p"), "core", loss="huber", huber_delta=.03, trainable="all", lr=lr)
    for lr in (1.5e-4, 2e-4):
        add(f"all_mse_lr{lr:g}".replace(".", "p"), "core", loss="mse", trainable="all", lr=lr, exploratory=True)
    for weights in ((2., 1.5, 2.), (4., 2., 4.), (6., 2., 1.)):
        for loss, delta in (("mae", None), ("huber", .03)):
            for lr in (2e-4, 4.5e-4):
                params: dict[str, object] = {"loss": loss, "trainable": "all", "lr": lr, "weights": weights}
                if delta is not None:
                    params["huber_delta"] = delta
                add(f"weight_{profile_label(weights)}__{loss}__lr{lr:g}".replace(".", "p"), "weighting", **params)
    for oversampling in ((1.5, 1.25, 2.), (3., 2., 4.), (6., 3., 6.)):
        for lr in (2e-4, 3e-4):
            add(f"oversample_{profile_label(oversampling)}__mae__lr{lr:g}".replace(".", "p"), "oversampling", loss="mae", trainable="all", lr=lr, oversampling=oversampling)
        add(f"oversample_{profile_label(oversampling)}__huber".replace(".", "p"), "oversampling", loss="huber", huber_delta=.03, trainable="all", lr=3e-4, oversampling=oversampling)
    for weights, oversampling in (
        ((3., 2., 3.), (1.5, 1.25, 2.)), ((3., 2., 3.), (3., 2., 4.)),
        ((2., 2., 4.), (1.5, 1.25, 2.)), ((2., 2., 4.), (3., 2., 4.)),
        ((6., 2., 1.), (1.5, 1.25, 2.)), ((6., 2., 1.), (3., 2., 4.)),
    ):
        add(f"interaction_w{profile_label(weights)}__o{profile_label(oversampling)}", "interaction", loss="mae", trainable="all", lr=3e-4, weights=weights, oversampling=oversampling, exploratory=True)
    for lr in (1e-3, 1.5e-3):
        add(f"discriminative_unweighted_lr{lr:g}".replace(".", "p"), "discriminative", loss="mae", trainable="discriminative", lr=lr, backbone_lr=lr / 10)
        add(f"discriminative_w2_2_4_lr{lr:g}".replace(".", "p"), "discriminative", loss="mae", trainable="discriminative", lr=lr, backbone_lr=lr / 10, weights=(2., 2., 4.))
    add("discriminative_w2_2_4_huber", "discriminative", loss="huber", huber_delta=.03, trainable="discriminative", lr=1e-3, backbone_lr=1e-4, weights=(2., 2., 4.))
    if len(strategies) != 42:
        raise AssertionError(f"Expected 42 strategies, got {len(strategies)}")
    return strategies


STRATEGIES = build_strategies()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_and_keys(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path, sep="\t", usecols=lambda name: name in KEYS, dtype=str).fillna("")
    return {"rows": int(len(frame)), "unique_peptidoforms": int(len(frame.drop_duplicates()))}


def copy_exact(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Missing canonical input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Checksum mismatch: {source} -> {destination}")
    return {"canonical_path": str(source), "path": str(destination), "sha256": sha256(destination), **rows_and_keys(destination)}


def write_frame(frame: pd.DataFrame, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, sep="\t", index=False)
    return {"path": str(destination), "sha256": sha256(destination), **rows_and_keys(destination)}


def target_file_name(split: str) -> str:
    return {"train": "rt_train_data.tsv", "validation": "rt_validation_data.tsv", "test": "rt_test_data.tsv"}[split]


def copy_frozen_target(canonical: Path, target: str, destination_root: Path) -> dict:
    target_root, output = canonical / target, destination_root / target
    record: dict[str, dict] = {"kind": "frozen_canonical_copy", "alignment": copy_exact(target_root / "rt_alignment.json", output / "rt_alignment.json"), "splits": {}}
    for split in ("train", "validation", "test"):
        split_record = {
            "aggregate": copy_exact(target_root / split / "diann_rt_input" / target_file_name(split), output / split / "diann_rt_input" / target_file_name(split)),
            "observations": copy_exact(target_root / split / "rt_observations.tsv", output / split / "rt_observations.tsv"), "sources": {},
        }
        for details in SOURCES.values():
            source = details["source_id"]
            split_record["sources"][source] = copy_exact(target_root / split / "sources" / f"{source}.tsv", output / split / "sources" / f"{source}.tsv")
        record["splits"][split] = split_record
    return record


def materialise_source_native(canonical: Path, destination_root: Path, reference_minutes: float) -> dict:
    """Use original Carafe2 RT/gradient targets only after the frozen split."""
    output, source_root = destination_root / "source_native", canonical / "global_60min"
    record: dict[str, object] = {"kind": "derived_after_fixed_split", "derivation": "rt_norm = source_rt_norm = rt / source_rt_max_minutes", "splits": {}}
    slopes: dict[str, float] = {}; ranges: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        frames: list[pd.DataFrame] = []; source_records: dict[str, dict] = {}
        for details in SOURCES.values():
            source = details["source_id"]
            frame = pd.read_csv(source_root / split / "sources" / f"{source}.tsv", sep="\t")
            required = {"rt", "source_rt_norm", "source_rt_max_minutes", *KEYS}
            if missing := required - set(frame):
                raise ValueError(f"{source}/{split} lacks {sorted(missing)}")
            # Derive from the raw RT and documented gradient, not from the
            # rounded source_rt_norm storage column.  The latter is retained
            # for provenance but differs by a few 1e-9 in one source.
            raw = pd.to_numeric(frame.rt, errors="raise").to_numpy(float)
            gradient = pd.to_numeric(frame.source_rt_max_minutes, errors="raise").to_numpy(float)
            stored_native = pd.to_numeric(frame.source_rt_norm, errors="raise").to_numpy(float)
            if not np.allclose(raw / gradient, stored_native, rtol=0, atol=1e-7):
                raise RuntimeError(f"{source}/{split}: provenance source_rt_norm is inconsistent with RT/gradient")
            native = raw / gradient
            frame["rt_norm"] = native
            unique_gradient = np.unique(gradient)
            if len(unique_gradient) != 1:
                raise RuntimeError(f"{source}/{split}: gradient duration is not constant")
            slope = float(unique_gradient[0])
            if source in slopes and not np.isclose(slopes[source], slope, rtol=0, atol=1e-10):
                raise RuntimeError(f"{source}: gradient duration differs across partitions")
            slopes[source] = slope; ranges[source] = {"rt_norm_min": float(native.min()), "rt_norm_max": float(native.max())}
            frames.append(frame); source_records[source] = write_frame(frame, output / split / "sources" / f"{source}.tsv")
        observations = pd.concat(frames, ignore_index=True)
        base = pd.read_csv(source_root / split / "diann_rt_input" / target_file_name(split), sep="\t")
        targets = observations.groupby(list(KEYS), as_index=False, dropna=False)["rt_norm"].median()
        aggregate = base.drop(columns=["rt_norm"]).merge(targets, on=list(KEYS), how="left", validate="one_to_one")
        if len(aggregate) != len(base) or aggregate.rt_norm.isna().any():
            raise RuntimeError(f"{split}: failed to construct source-native combined table")
        record["splits"][split] = {"aggregate": write_frame(aggregate, output / split / "diann_rt_input" / target_file_name(split)),
                                    "observations": write_frame(observations, output / split / "rt_observations.tsv"), "sources": source_records}
    key_sets = []
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(output / split / "diann_rt_input" / target_file_name(split), sep="\t", usecols=list(KEYS), dtype=str).fillna("")
        key_sets.append(set(map(tuple, frame.itertuples(index=False, name=None))))
    if any(left & right for index, left in enumerate(key_sets) for right in key_sets[index + 1:]):
        raise RuntimeError("Source-native construction violated canonical split disjointness")
    alignment = {"mode": "source_native", "reference_minutes": reference_minutes,
                 "fit_scope": "no_fitted_alignment; known_source_gradient_metadata_only",
                 "target_definition": "original Carafe2 per-source gradient normalization: rt_minutes / source_rt_max_minutes",
                 "source_target_ranges": ranges, "target_to_observed_rt": {source: {"slope": slope, "intercept": 0.0} for source, slope in slopes.items()},
                 "alignment": {"mode": "source_native", "method": "none", "leakage_safe": True}}
    alignment_path = output / "rt_alignment.json"; alignment_path.write_text(json.dumps(alignment, indent=2) + "\n", encoding="utf-8")
    record["alignment"] = {"path": str(alignment_path), "sha256": sha256(alignment_path)}
    return record


def train_command(python: str, train: Path, validation: Path, run: Path, strategy: dict, seed: int) -> list[str]:
    command = [python, str(DIANN / "train_carafe_rt.py"), "--train-split", str(train), "--validation-split", str(validation),
               "--model-dir", str(DIANN_MODEL), "--output-dir", str(run), "--device", "cuda", "--epochs", "40", "--warmup-epochs", "10",
               "--batch-size", "512", "--loss", strategy["loss"], "--weight-decay", "0", "--trainable", strategy["trainable"],
               "--learning-rate", f"{strategy['lr']:.12g}", "--steps-per-epoch", "120", "--seed", str(seed), "--no-plots"]
    if "huber_delta" in strategy: command += ["--huber-delta", f"{strategy['huber_delta']:.12g}"]
    if "backbone_lr" in strategy: command += ["--backbone-learning-rate", f"{strategy['backbone_lr']:.12g}"]
    weights = strategy.get("weights", (1., 1., 1.)); oversampling = strategy.get("oversampling", (1., 1., 1.))
    command += ["--heavy-nterm-weight", f"{weights[0]:.12g}", "--heavy-lysine-weight", f"{weights[1]:.12g}", "--heavy-both-weight", f"{weights[2]:.12g}",
                "--heavy-nterm-oversample", f"{oversampling[0]:.12g}", "--heavy-lysine-oversample", f"{oversampling[1]:.12g}", "--heavy-both-oversample", f"{oversampling[2]:.12g}"]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable)); return parser.parse_args()


def main() -> None:
    args = parse_args(); output, dataset = args.output_dir.resolve(), args.dataset_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite existing study: {output}")
    required = [DIANN / "train_carafe_rt.py", DIANN_MODEL / "rt.d0.pt", dataset / "dataset_manifest.json"]
    required += [dataset / target / "rt_alignment.json" for target in ("chronologer_hi", "anchor_irt", "global_60min")]
    if missing := [path for path in required if not path.is_file()]: raise FileNotFoundError(f"Missing prerequisite(s): {missing}")
    output.mkdir(parents=True); canonical_manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8")); prepared_targets = output / "prepared_targets"
    target_files = {"source_native": materialise_source_native(dataset, prepared_targets, float(canonical_manifest["reference_minutes"])),
                    "chronologer_hi": copy_frozen_target(dataset, "chronologer_hi", prepared_targets), "anchor_irt": copy_frozen_target(dataset, "anchor_irt", prepared_targets)}
    scope_inputs: dict[str, dict[str, dict]] = {"source_native": {"combined": {"kind": "combined", "label": "New combined source-native control", "organism": "mixed"}}, "chronologer_hi": {}, "anchor_irt": {}}
    for target in TARGETS:
        for scope, details in SOURCES.items():
            if target == "anchor_irt" and scope == "human_hunter": continue
            scope_inputs[target][scope] = {"kind": "source_specific", **details}
    jobs: list[dict] = []
    for target, scopes in scope_inputs.items():
        target_root = prepared_targets / target; partition, alignment = target_root / "validation", target_root / "rt_alignment.json"
        for scope, details in scopes.items():
            train = target_root / "train" / "diann_rt_input" / target_file_name("train") if scope == "combined" else target_root / "train" / "sources" / f"{details['source_id']}.tsv"
            validation = target_root / "validation" / "diann_rt_input" / target_file_name("validation") if scope == "combined" else target_root / "validation" / "sources" / f"{details['source_id']}.tsv"
            for strategy_name, strategy in STRATEGIES.items():
                for seed in SEEDS:
                    run = output / "runs" / target / scope / strategy_name / f"seed_{seed}"
                    evaluation = [args.diann_python, str(ROOT / "ma/data_collect/evaluate_diann_trainfit.py"), "--run-dir", str(run), "--partition-dir", str(partition), "--alignment", str(alignment), "--output", str(run / "validation_metrics.json"), "--device", "cuda"]
                    jobs.append({"id": f"{target}/{scope}/{strategy_name}/seed_{seed}", "workdir": str(DIANN), "commands": [train_command(args.diann_python, train, validation, run, strategy, seed), evaluation], "completion_files": [str(run / "validation_metrics.json")], "progress_regex": r"epoch=(\\d+)/(\\d+)", "progress_total": 40,
                                 "metadata": {"target": target, "scope": scope, "scope_kind": details["kind"], "strategy": strategy_name, "strategy_parameters": strategy, "seed": seed, "validation_only": True, "test_partition_referenced": False}})
    expected = 12 * len(STRATEGIES) * len(SEEDS)
    if len(jobs) != expected: raise AssertionError(f"Expected {expected} jobs, got {len(jobs)}")
    if any("/test/" in " ".join(job["commands"][0] + job["commands"][1]) for job in jobs): raise AssertionError("Validation sweep must never reference a test input")
    manifest = {"study": "diann_source_specific_3target_vs_combined_v5", "validation_only": True, "test_partition": "closed_until_finalist_selection", "canonical_dataset_dir": str(dataset), "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"), "prepared_target_root": str(prepared_targets), "reference_minutes": canonical_manifest["reference_minutes"], "seeds": list(SEEDS), "strategies": STRATEGIES,
                "targets": {"source_native": {"definition": "original Carafe2 source-gradient normalized RT; no fitted alignment", "scopes": list(scope_inputs["source_native"])}, "chronologer_hi": {"definition": "frozen canonical train-only source-to-Chronologer-HI mapping", "scopes": list(scope_inputs["chronologer_hi"])}, "anchor_irt": {"definition": "frozen canonical train-only Human-reference anchor mapping", "scopes": list(scope_inputs["anchor_irt"]), "omitted_scope": "human_hunter (identical source_native target)"}}, "target_files": target_files, "scopes": scope_inputs, "jobs": jobs}
    (output / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "sweep_manifest.json").write_text(json.dumps({"name": "DIA-NN final source-specific RT sweep", "validation_only": True, "state_root": str(output / ".sweep_state"), "status_file": str(output / "queue_status.json"), "log_root": str(output / "logs"), "jobs": jobs}, indent=2) + "\n", encoding="utf-8")
    readme = f'''# Final DIA-NN source-specific RT sweep

## Question

Does a DIA-NN RT model fine-tuned on one acquisition source improve its own source's observed-minute RT accuracy, especially for Heavy-Dimethyl peptides, over pooled training? This final broad screen has **{len(jobs):,} validation-only runs**: 12 target-by-training-data cells × 42 strategies × 3 fixed seeds.

## Split and leakage contract

The immutable canonical split is `{dataset}`. It was assigned globally by peptidoform (`sequence + mods + mod_sites`) *before* target construction. Hence all observations of a peptidoform stay in exactly one of train, validation or test, even across sources. Source-specific tables inherit that split; they are never separately randomized.

`prepared_targets/` is the study-local data artifact. It contains test only as checksum-audited final-evaluation material; every validation-sweep command reads **only** train and validation. The preparation script rejects any sweep manifest containing a test path. Chronologer and Anchor are byte-identical canonical copies. Source-native is rebuilt after the frozen split and checked for train/validation/test peptidoform disjointness.

## Target coordinates

| Target | DIA-NN target (`rt_norm`) | Fit data | Cells |
|---|---|---|---:|
| `source_native` | Source-gradient normalized RT: observed RT / source-specific known gradient duration | no fitted alignment | Combined + 4 source-only = 5 |
| `chronologer_hi` | RT mapped to external Chronologer HI, then frozen to canonical scale | source training observations only | 4 source-only = 4 |
| `anchor_irt` | RT mapped to Human-Hunter coordinate, then scaled | global training anchors only | C. elegans + two TripleMix = 3 |

Human `anchor_irt` and Human `source_native` are exactly identical because Human is the anchor and its gradient is the reference duration; the redundant 126-run cell is omitted. The two TripleMix acquisitions stay separate: they are distinct sources and generalisation domains.

Native is deliberately **not raw minutes**. The pretrained DIA-NN head expects a normalized coordinate; direct minute targets would be a scale mismatch. Its source-specific inverse mappings return every score to observed minutes.

## Why one new combined arm

Exact-split Chronologer and Anchor combined models already exist as validation-selected historical controls. They are frozen and will be evaluated later on the same test observations. Only source-native lacks a matched pooled comparator, so this sweep adds its combined arm and does not waste compute rerunning the other two pooled arms.

## Sweep: 1,512 runs

All 12 cells use the identical **42-strategy** grid and three seeds: `12 × 42 × 3 = 1,512`. Runs have 40 epochs, 10 warm-up epochs, batch size 512 and exactly 120 optimizer steps/epoch. Fixed updates make oversampling a change in exposure frequency rather than extra optimization.

| Family | Configurations | Purpose |
|---|---:|---|
| Core loss/LR | 10 | MAE, Huber and limited MSE controls |
| Heavy loss weighting | 12 | balanced, strong and N-term-prioritised weighting |
| Pure oversampling | 9 | gentle, medium and strong Heavy exposure |
| Weighting × oversampling | 6 | practical exploratory recipes |
| Discriminative adaptation | 5 | head/backbone learning-rate controls |

The overall finalist minimizes overall MAE (in-domain for source-only; source-balanced for pooled). The Heavy finalist minimizes the matching Heavy-Macro MAE under a 1% overall-MAE guardrail versus that scope's unweighted MAE baseline. Source-balanced metrics weight four sources equally; Heavy-Macro weights N-term, lysine-only and doubly-heavy strata equally.

## Plots and valid comparisons

Validation plots in `analysis/plots/` are exported as PNG, PDF and SVG: target-faceted overall-vs-heavy Pareto views, finalist uncertainty plots, in-domain-versus-transfer diagnostics and Heavy-stratum checks. They do not invent a new pooled Chronologer/Anchor comparison, because those cells are intentionally absent.

After validation selection is frozen, the explicit final-test gate evaluates all finalists and registered historical combined controls on the exact same source-specific test observations in observed minutes. Comparison plots go to `analysis/plots/comparisons/`, remain target-faceted, and never mix validation with test or optimizer seeds with biological replicates.

## Run

```bash
bash {ROOT / 'ma/data_collect/start_diann_source_specific_sweep.sh'}
```

Only after reviewing and freezing `analysis/validation_selection.json`, run the
guarded, one-time final-test comparison:

```bash
python3 {ROOT / 'ma/data_collect/evaluate_diann_source_specific_finalists.py'} \\
  --study-root {output} \\
  --historical-registry {output / 'analysis/historical_combined_registry.json'} \\
  --confirm-final-test
```
'''
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "strategy_count": len(STRATEGIES), "cells": sum(map(len, scope_inputs.values())), "output": str(output)}, indent=2))


if __name__ == "__main__": main()
