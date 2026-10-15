#!/usr/bin/env python3
"""Prepare the final, validation-only Carafe2 source-specific RT sweep.

The immutable global peptidoform split is reused verbatim.  Three RT target
coordinates are materialised only *after* that split, matching the DIA-NN v5
study.  Heavy-Dimethyl oversampling uses Carafe2's hard update budget, so it
changes exposure frequency rather than the number of optimizer updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from prepare_diann_source_specific_sweep import (
    ROOT, SOURCES, TARGETS, SEEDS, copy_frozen_target, materialise_source_native,
)

CARAFE2 = Path(os.environ.get("CARAFE2_ROOT", ROOT / "carafe2-semi")).expanduser()
TRAINER = CARAFE2 / "src/main/resources/py/v2/ai.py"
DATASET_DEFAULT = ROOT / "trainingdata" / "finalrt_4source_trainfit15_validation10_v1"
EPOCHS, WARMUP_EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH = 40, 10, 512, 120


def profile_label(values: tuple[float, float, float]) -> str:
    return "_".join(f"{value:g}".replace(".", "p") for value in values)


def build_strategies() -> dict[str, dict]:
    """42 strategies based on completed Carafe2 LR and Heavy-Dimethyl runs."""
    strategies: dict[str, dict] = {}

    def add(name: str, family: str, **parameters: object) -> None:
        if name in strategies:
            raise AssertionError(f"Duplicate strategy: {name}")
        strategies[name] = {"family": family, **parameters}

    # Carafe2's useful all-parameter region is 3e-4--9e-4.  The completed
    # LR sweep centred on 4.5e-4; the completed Heavy sweep makes 9e-4 a
    # justified sensitivity point for Heavy-weighted training.
    for lr in (3e-4, 4.5e-4, 6e-4, 9e-4):
        name = "all_mae_baseline" if lr == 4.5e-4 else f"all_mae_lr{lr:g}".replace(".", "p")
        add(name, "core", loss="mae", trainable="all", lr=lr)
    for lr in (3e-4, 4.5e-4, 9e-4):
        add(f"all_huber_d03_lr{lr:g}".replace(".", "p"), "core", loss="huber", huber_delta=.03, trainable="all", lr=lr)
    for delta in (.02, .05):
        add(f"all_huber_d{delta:.2f}_lr4p5e-4".replace(".", "p"), "core", loss="huber", huber_delta=delta, trainable="all", lr=4.5e-4)
    add("all_mae_lr7p5e-4", "core", loss="mae", trainable="all", lr=7.5e-4)

    # The previous Carafe2 Heavy sweep established 1/1/6 as the central
    # Heavy-both weighting candidate.  Lower doses and a balanced companion
    # test whether that conclusion changes on an organism-only data set.
    for weights in ((1., 1., 2.), (1., 1., 4.), (1., 1., 6.), (2., 2., 4.)):
        for lr in (3e-4, 4.5e-4, 9e-4):
            add(f"weight_{profile_label(weights)}__mae__lr{lr:g}".replace(".", "p"), "weighting", loss="mae", trainable="all", lr=lr, weights=weights)

    # These are sampled under the fixed 120-step budget, unlike historical
    # Carafe2 oversampling runs whose duplicated rows changed update count.
    for oversampling in ((1., 1., 2.), (1., 1., 4.), (2., 2., 4.)):
        for lr in (3e-4, 4.5e-4, 9e-4):
            add(f"oversample_{profile_label(oversampling)}__mae__lr{lr:g}".replace(".", "p"), "oversampling", loss="mae", trainable="all", lr=lr, oversampling=oversampling)

    for weights, oversampling, lr in (
        ((1., 1., 2.), (1., 1., 2.), 4.5e-4),
        ((1., 1., 4.), (1., 1., 2.), 4.5e-4),
        ((1., 1., 6.), (1., 1., 2.), 4.5e-4),
        ((1., 1., 6.), (1., 1., 4.), 4.5e-4),
        ((2., 2., 4.), (1., 1., 2.), 4.5e-4),
        ((2., 2., 4.), (2., 2., 4.), 9e-4),
    ):
        add(f"interaction_w{profile_label(weights)}__o{profile_label(oversampling)}__lr{lr:g}".replace(".", "p"), "interaction", loss="mae", trainable="all", lr=lr, weights=weights, oversampling=oversampling, exploratory=True)

    # Discriminative fine-tuning is close but less stable in the completed LR
    # study.  Retain it as a focused control, not another broad policy grid.
    for lr in (1.5e-3, 2e-3):
        add(f"discriminative_unweighted_lr{lr:g}".replace(".", "p"), "discriminative", loss="mae", trainable="discriminative", lr=lr, backbone_lr=lr / 10)
        add(f"discriminative_w1_1_6_lr{lr:g}".replace(".", "p"), "discriminative", loss="mae", trainable="discriminative", lr=lr, backbone_lr=lr / 10, weights=(1., 1., 6.))
    add("discriminative_w2_2_4_lr2e-3", "discriminative", loss="mae", trainable="discriminative", lr=2e-3, backbone_lr=2e-4, weights=(2., 2., 4.))
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


def carafe_command(python: str, input_dir: Path, validation: Path, run: Path, strategy: dict, seed: int) -> list[str]:
    weights = strategy.get("weights", (1., 1., 1.))
    oversampling = strategy.get("oversampling", (1., 1., 1.))
    command = [
        python, str(TRAINER), "--in_dir", str(input_dir), "--out_dir", str(run),
        "--device", "gpu", "--tf_type", "rt", "--seed", str(seed), "--verbose", "1", "--use_best_model",
        "--rt_validation_file", str(validation), "--rt_validation_disjoint_by", "peptidoform",
        "--rt_epoch", str(EPOCHS), "--rt_warmup_epoch", str(WARMUP_EPOCHS),
        "--rt_batch_size", str(BATCH_SIZE), "--no_adjust_rt_batch_size",
        "--rt_steps_per_epoch", str(STEPS_PER_EPOCH),
        "--sweep_minimal_artifacts",
        "--rt_loss", "l1" if strategy["loss"] == "mae" else "huber",
        "--rt_huber_delta", f"{strategy.get('huber_delta', .03):.12g}", "--rt_weight_decay", "0",
        "--rt_heavy_nterm_oversample", f"{oversampling[0]:.12g}",
        "--rt_heavy_lysine_oversample", f"{oversampling[1]:.12g}",
        "--rt_heavy_both_oversample", f"{oversampling[2]:.12g}",
        "--rt_heavy_nterm_weight", f"{weights[0]:.12g}",
        "--rt_heavy_lysine_weight", f"{weights[1]:.12g}",
        "--rt_heavy_both_weight", f"{weights[2]:.12g}",
        "--rt_trainable", strategy["trainable"], "--rt_lr", f"{strategy['lr']:.12g}",
    ]
    if "backbone_lr" in strategy:
        command += ["--rt_backbone_lr", f"{strategy['backbone_lr']:.12g}"]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    default_python = Path("python")
    parser.add_argument("--carafe-python", default=os.environ.get("CARAFE2_PYTHON", str(default_python if default_python.is_file() else Path(sys.executable))))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, dataset = args.output_dir.expanduser().resolve(), args.dataset_dir.expanduser().resolve()
    python = Path(args.carafe_python).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing study: {output}")
    required = [TRAINER, python, dataset / "dataset_manifest.json"]
    required += [dataset / target / "rt_alignment.json" for target in ("chronologer_hi", "anchor_irt", "global_60min")]
    if missing := [path for path in required if not path.is_file()]:
        raise FileNotFoundError(f"Missing prerequisite(s): {missing}")

    output.mkdir(parents=True)
    canonical_manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    prepared_targets = output / "prepared_targets"
    target_files = {
        "source_native": materialise_source_native(dataset, prepared_targets, float(canonical_manifest["reference_minutes"])),
        "chronologer_hi": copy_frozen_target(dataset, "chronologer_hi", prepared_targets),
        "anchor_irt": copy_frozen_target(dataset, "anchor_irt", prepared_targets),
    }

    # Preserve the 12 cells of DIA-NN v5.  Human anchor is intentionally
    # omitted because it is exactly the Human source-gradient coordinate.
    scopes: dict[str, dict[str, dict]] = {
        "source_native": {"combined": {"kind": "combined", "label": "New combined source-gradient control", "organism": "mixed"}},
        "chronologer_hi": {}, "anchor_irt": {},
    }
    for target in TARGETS:
        for scope, details in SOURCES.items():
            if target == "anchor_irt" and scope == "human_hunter":
                continue
            scopes[target][scope] = {"kind": "source_specific", **details}

    jobs: list[dict] = []
    input_records: dict[str, dict[str, str]] = {}
    for target, target_scopes in scopes.items():
        target_root = prepared_targets / target
        partition, alignment = target_root / "validation", target_root / "rt_alignment.json"
        for scope, details in target_scopes.items():
            train = target_root / "train" / "diann_rt_input" / "rt_train_data.tsv" if scope == "combined" else target_root / "train" / "sources" / f"{details['source_id']}.tsv"
            validation = target_root / "validation" / "diann_rt_input" / "rt_validation_data.tsv" if scope == "combined" else target_root / "validation" / "sources" / f"{details['source_id']}.tsv"
            input_dir = output / "prepared_inputs" / target / scope
            input_dir.mkdir(parents=True, exist_ok=True)
            input_file = input_dir / "rt_train_data.tsv"
            input_file.symlink_to(train)
            input_records.setdefault(target, {})[scope] = str(input_file)
            for strategy_name, strategy in STRATEGIES.items():
                for seed in SEEDS:
                    run = output / "runs" / target / scope / strategy_name / f"seed_{seed}"
                    train_command = carafe_command(str(python), input_dir, validation, run, strategy, seed)
                    evaluate_command = [str(python), str(ROOT / "ma/data_collect/evaluate_carafe2_trainfit.py"), "--run-dir", str(run), "--partition-dir", str(partition), "--alignment", str(alignment), "--output", str(run / "validation_metrics.json"), "--device", "gpu", "--python", str(python)]
                    jobs.append({
                        "id": f"{target}/{scope}/{strategy_name}/seed_{seed}", "workdir": str(CARAFE2),
                        "commands": [train_command, evaluate_command], "completion_files": [str(run / "validation_metrics.json")],
                        "progress_regex": r"Epoch=(\\d+)", "progress_total": EPOCHS,
                        "metadata": {"model": "carafe2", "target": target, "scope": scope, "scope_kind": details["kind"], "strategy": strategy_name, "strategy_parameters": strategy, "seed": seed, "validation_only": True, "test_partition_referenced": False, "training_budget": {"steps_per_epoch": STEPS_PER_EPOCH, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer_updates": STEPS_PER_EPOCH * EPOCHS, "oversampling_semantics": "fixed-step exposure-frequency intervention"}},
                    })
    expected = 12 * len(STRATEGIES) * len(SEEDS)
    if len(jobs) != expected:
        raise AssertionError(f"Expected {expected} jobs, got {len(jobs)}")
    if any("/test/" in " ".join(part for command in job["commands"] for part in command) for job in jobs):
        raise AssertionError("Validation sweep must never reference a test input")

    study_manifest = {
        "study": "carafe2_source_specific_3target_vs_combined_v1", "model": "carafe2", "validation_only": True,
        "test_partition": "closed_until_finalist_selection", "canonical_dataset_dir": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"), "prepared_target_root": str(prepared_targets),
        "prepared_inputs": input_records, "reference_minutes": canonical_manifest["reference_minutes"], "seeds": list(SEEDS),
        "training_contract": {"epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS, "batch_size": BATCH_SIZE, "steps_per_epoch": STEPS_PER_EPOCH, "optimizer_updates": EPOCHS * STEPS_PER_EPOCH, "oversampling": "fixed-budget with-replacement sampling after Heavy-profile duplication"},
        "strategies": STRATEGIES, "targets": {
            "source_native": {"definition": "source-gradient normalized RT; no fitted alignment", "scopes": list(scopes["source_native"])},
            "chronologer_hi": {"definition": "frozen canonical train-only source-to-Chronologer-HI mapping", "scopes": list(scopes["chronologer_hi"])},
            "anchor_irt": {"definition": "frozen canonical Human-reference anchor mapping", "scopes": list(scopes["anchor_irt"]), "omitted_scope": "human_hunter (identical source-gradient target)"},
        }, "target_files": target_files, "scopes": scopes, "jobs": jobs,
    }
    (output / "study_manifest.json").write_text(json.dumps(study_manifest, indent=2) + "\n", encoding="utf-8")
    (output / "sweep_manifest.json").write_text(json.dumps({"name": "Carafe2 final source-specific RT sweep", "validation_only": True, "state_root": str(output / ".sweep_state"), "status_file": str(output / "queue_status.json"), "log_root": str(output / "logs"), "jobs": jobs}, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(f'''# Final Carafe2 source-specific RT sweep

## Question

Does an organism/source-specific Carafe2 RT model improve observed-minute RT accuracy on its own acquisition source, especially for Heavy-Dimethyl peptidoforms, relative to a pooled model?  This validation-only screen contains **{len(jobs):,} runs**: 12 target-by-training-data cells × 42 strategies × three optimizer seeds.

## Data and leakage contract

The immutable global peptidoform split is `{dataset}`.  It was assigned before every target transformation, globally over `sequence + mods + mod_sites`.  Source-only tables inherit that split; they are never separately split or aligned with validation/test observations.  Chronologer HI and Human-anchor targets are byte-identical frozen copies of their canonical target data.  Source-gradient normalized RT is derived only after the split from raw observed RT and known source-gradient duration.

Every manifest job reads train and validation only.  Test files exist solely as checksum-audited material for a later, explicitly confirmed final-test evaluation.

## Carafe2-specific training contract

All runs use 40 epochs, 10 warm-up epochs, batch size 512 and exactly **120 optimizer updates per epoch**.  This is essential: Heavy-Dimethyl oversampling is sampled under a fixed update budget, so it changes observation exposure rather than silently producing a longer optimization run.  The prior native Carafe2 oversampling studies did not have this causal separation.

To keep this 1,512-run screen efficient, every sweep run stores the model and numerical training/validation metrics but omits redundant per-run PNG diagnostics and self-contained HTML reports.  The central analysis recreates the publication figures from those numeric outputs; detailed per-run artefacts are reserved for validation-selected finalists.

The three coordinates are source-gradient normalized RT (combined plus four source-only cells), Chronologer HI (four source-only cells), and Human-anchor RT (three source-only cells; Human is omitted because its anchor and source-gradient targets are identical).

## 42-strategy grid

| Family | Configurations | Purpose |
|---|---:|---|
| Core loss/LR | 10 | Carafe2 all-parameter MAE/Huber sensitivity in the validated LR range |
| Heavy loss weighting | 12 | `1/1/2`, `1/1/4`, prior-winning `1/1/6`, and balanced `2/2/4` |
| Fixed-budget oversampling | 9 | Same moderate Heavy profiles, now without an update-budget confound |
| Weighting × sampling | 6 | Pre-specified practical combinations |
| Discriminative controls | 5 | Focused high-LR policy comparison |

The overall finalist minimizes in-domain MAE for source-only models (source-balanced MAE for the one pooled cell).  The Heavy finalist minimizes the matching Heavy-Macro MAE subject to a 1% Overall-MAE guardrail relative to `all_mae_baseline` (all-parameter MAE at `4.5e-4`).

Validation figures are generated by `analyze_carafe2_source_specific_sweep.py` with the same ten figure contracts as the DIA-NN v5 sweep.  They are exported in PNG/PDF/SVG, retain target-coordinate facets and use observed RT minutes only.  Final-test commands are deliberately not generated until the validation selection JSON has been frozen.

## Run and final-test gate

Start or resume the validation-only queue with:

```bash
bash {ROOT / 'ma/data_collect/start_carafe2_source_specific_sweep.sh'}
```

After the queue is complete, review `analysis/validation_selection.json` and
the validation plots.  Only then run the one-time held-out comparison, with
the already registered historical controls:

```bash
python3 {ROOT / 'ma/data_collect/evaluate_carafe2_source_specific_finalists.py'} \\
  --study-root {output} \\
  --historical-registry {output / 'analysis/historical_combined_registry.json'} \\
  --confirm-final-test
```

The final-test command never selects or reranks candidates.  It only evaluates
the validation-frozen list on identical source-specific test observations.
''', encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "strategy_count": len(STRATEGIES), "cells": sum(map(len, scopes.values())), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
