#!/usr/bin/env python3
"""Prepare a validation-only, Dimethyl-first DIA-NN RT training sweep.

The study explicitly deprioritizes normal peptides.  It tests whether a
group-balanced loss across Heavy N-term, Heavy Lys and Heavy-both can improve
the Dimethyl macro MAE without allowing one abundant Dimethyl class to mask
the others.  Every run uses the same optimizer-step budget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from heavy_dimethyl_sweep_common import (
    BATCH_SIZE,
    EPOCHS,
    SEEDS,
    WARMUP_EPOCHS,
    lr_label,
    oversampling_budget,
    profile_text,
    training_counts,
)


ROOT = Path(__file__).resolve().parents[2]
DIANN = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune"))
DIANN_MODEL = Path(os.environ.get(
    "DIANN_MODEL_DIR", ROOT.parent / "diann_sc/DIA-NN-2.5.0/diann-2.5.0/models",
))
DATASET_DEFAULT = ROOT / "trainingdata/finalrt_4source_trainfit15_validation10_v1"
TARGETS = ("chronologer_hi", "anchor_irt")
LEARNING_RATES = (5e-5, 1e-4, 1.5e-4)
CHECKPOINT_SELECTIONS = ("100_0", "75_25", "50_50", "25_75", "0_100")

# Every non-control profile reports the desired *group-level loss mass* in the
# order normal, Heavy N-term, Heavy Lys, Heavy-both.  Values are converted to
# per-row weights using the actual post-oversampling group counts, so all
# three Dimethyl strata contribute their requested amount despite different
# sample counts.
LOSS_PROFILES: dict[str, dict] = {
    "C0_standard": {
        "description": "Established unweighted MAE control; normal peptides dominate naturally.",
        "group_masses": None,
    },
    "D99_equal": {
        "description": "1% normal, 99% Dimethyl; equal N-term/Lys/both contribution.",
        "group_masses": (0.01, 0.33, 0.33, 0.33),
    },
    "D95_equal": {
        "description": "5% normal, 95% Dimethyl; equal N-term/Lys/both contribution.",
        "group_masses": (0.05, 0.95 / 3, 0.95 / 3, 0.95 / 3),
    },
    "D90_equal": {
        "description": "10% normal, 90% Dimethyl; equal N-term/Lys/both contribution.",
        "group_masses": (0.10, 0.30, 0.30, 0.30),
    },
    "D95_nterm": {
        "description": "5% normal; N-term gets half of total loss mass to repair the observed conflict.",
        "group_masses": (0.05, 0.475, 0.2375, 0.2375),
    },
    "D95_both": {
        "description": "5% normal; Heavy-both gets half of total loss mass.",
        "group_masses": (0.05, 0.2375, 0.2375, 0.475),
    },
}

# Exact heavy strata reach the stated number of rows after documented trainer
# rounding. R0 is intentionally natural; R40/R56 test the strong-dose range
# where the previous study observed the Heavy-both response.
OVERSAMPLING_TARGET_ROWS = {"R0": None, "R40": 40_000, "R56": 56_000}
FIXED_STEP_PROFILE = "R40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable))
    return parser.parse_args()


def require_inputs(dataset_dir: Path, target: str) -> dict[str, Path]:
    base = dataset_dir / target
    values = {
        "train": base / "train/diann_rt_input/rt_train_data.tsv",
        "validation": base / "validation/diann_rt_input/rt_validation_data.tsv",
        "partition": base / "validation",
        "alignment": base / "rt_alignment.json",
    }
    required = (*values.values(), values["partition"] / "rt_observations.tsv")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required prepared inputs: {missing}")
    if any("test" in str(path).lower() for path in values.values()):
        raise ValueError("This sweep must not reference a test partition")
    return values


def factors_for_target_rows(counts: dict[str, int], target_rows: int | None) -> tuple[float, float, float]:
    if target_rows is None:
        return (1.0, 1.0, 1.0)
    keys = ("heavy_nterm", "heavy_lysine", "heavy_both")
    if any(counts[name] <= 0 for name in keys):
        raise ValueError(f"Cannot balance a missing Heavy stratum: {counts}")
    return tuple(target_rows / counts[name] for name in keys)


def post_oversampling_counts(
    counts: dict[str, int], factors: tuple[float, float, float]
) -> dict[str, int]:
    result = dict(counts)
    for key, factor in zip(("heavy_nterm", "heavy_lysine", "heavy_both"), factors):
        result[key] += int(round((factor - 1.0) * counts[key]))
    return result


def loss_weights(
    profile: str, counts: dict[str, int], factors: tuple[float, float, float]
) -> dict[str, float]:
    """Derive stable per-row weights from requested group-level loss mass.

    Scaling to a mean row weight of one preserves the trainer's loss scale;
    only the relative group contributions change.
    """
    masses = LOSS_PROFILES[profile]["group_masses"]
    if masses is None:
        return {
            "normal_weight": 1.0,
            "heavy_nterm_weight": 1.0,
            "heavy_lysine_weight": 1.0,
            "heavy_both_weight": 1.0,
        }
    effective = post_oversampling_counts(counts, factors)
    keys = ("normal", "heavy_nterm", "heavy_lysine", "heavy_both")
    total_rows = sum(effective.values())
    if not math.isclose(sum(masses), 1.0, abs_tol=1e-12):
        raise AssertionError(f"Loss masses must sum to one: {profile}={masses}")
    weights = {key: mass * total_rows / effective[key] for key, mass in zip(keys, masses)}
    if min(weights.values()) <= 0:
        raise ValueError(f"Invalid non-positive training weight for {profile}: {weights}")
    return {
        "normal_weight": weights["normal"],
        "heavy_nterm_weight": weights["heavy_nterm"],
        "heavy_lysine_weight": weights["heavy_lysine"],
        "heavy_both_weight": weights["heavy_both"],
    }


def evaluation_command(python: str, run: Path, values: dict[str, Path], selection: str) -> list[str]:
    return [
        python, str(ROOT / "ma/data_collect/evaluate_diann_trainfit.py"),
        "--run-dir", str(run), "--partition-dir", str(values["partition"]),
        "--alignment", str(values["alignment"]), "--checkpoint-selection", selection,
        "--output", str(run / "validation_metrics" / f"checkpoint_{selection}.json"),
        "--device", "cuda",
    ]


def write_readme(
    output: Path,
    dataset_dir: Path,
    target_designs: dict[str, dict],
) -> None:
    loss_rows = "\n".join(
        f"| `{name}` | {details['description']} | "
        + ("standard row-wise MAE" if details["group_masses"] is None else "/".join(f"{value:.3f}" for value in details["group_masses"]))
        + " |"
        for name, details in LOSS_PROFILES.items()
    )
    target_rows = "\n".join(
        f"| `{target}` | normal={design['counts']['normal']:,}; N-term={design['counts']['heavy_nterm']:,}; Lys={design['counts']['heavy_lysine']:,}; both={design['counts']['heavy_both']:,} | {design['fixed_steps']:,} |"
        for target, design in target_designs.items()
    )
    text = f"""# DIA-NN Dimethyl-first loss-focus sweep

This is a **validation-only** study. Its explicit goal is to reduce RT error
for Heavy-Dimethyl peptides; normal peptides are a small guardrail or, for the
control, an intentionally retained historical objective. The final test
partition is absent and remains closed.

## Pre-registered design

- Runs: 2 target scales × 6 loss profiles × 3 oversampling doses × 3 learning rates × 3 seeds = **324**
- Targets: {', '.join(f'`{target}`' for target in TARGETS)}; MAE; all parameters trainable
- Epochs / warm-up / batch: `{EPOCHS}` / `{WARMUP_EPOCHS}` / `{BATCH_SIZE}`
- Learning rates: {', '.join(f'`{lr:g}`' for lr in LEARNING_RATES)}
- Seeds: {', '.join(map(str, SEEDS))}

| RT target scale | Training counts before oversampling | Fixed optimizer steps / epoch |
|---|---|---:|
{target_rows}

The fixed-step budget is the `R40` budget. Smaller profiles are sampled with
replacement from their oversampled frame; larger profiles are subsampled.
Thus oversampling changes example composition, not total optimizer updates.

## Loss profiles

For every `D*` profile, loss mass is assigned to the four exact groups in the
order normal / Heavy N-term / Heavy Lys / Heavy both. Per-row weights are
derived after oversampling so the requested **group-level** mass, rather than
raw sample count, determines the objective.

| Profile | Intent | Group loss mass |
|---|---|---:|
{loss_rows}

## Oversampling doses

`R0` is natural sampling; `R40` and `R56` target 40,000 or 56,000 rows in
each exact Heavy stratum after oversampling. Factors are target-specific and
recorded in every job's metadata.

## Checkpoints and selection

Every run saves five validation-only candidates (`100_0`, `75_25`, `50_50`,
`25_75`, `0_100`). For this study, `0_100` is the primary candidate because it
selects the lowest equal-weight Dimethyl macro MAE. Final selection must still
inspect all three exact Dimethyl MAEs; no normal or Overall guardrail is used
for ranking.

## Commands

```bash
python3 {ROOT / 'ma/data_collect/prepare_diann_dimethyl_loss_focus_sweep.py'} \\
  --output-dir {output}

python3 {ROOT / 'ma/data_collect/sweep_template/parallel_gpu_sweep.py'} \\
  --manifest {output / 'sweep_manifest.json'} --gpus 0,1,2 --jobs-per-gpu 1

python3 {ROOT / 'ma/data_collect/analyze_diann_dimethyl_loss_focus_sweep.py'} \\
  --sweep-dir {output} --output-dir {output / 'analysis'}

python3 {ROOT / 'ma/data_collect/plot_diann_dimethyl_loss_focus_sweep.py'} \\
  --analysis-dir {output / 'analysis'}
```

Dataset: `{dataset_dir}`.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output, dataset_dir = args.output_dir.resolve(), args.dataset_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep directory: {output}")
    trainer = DIANN / "train_carafe_rt.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"DIA-NN trainer unavailable: {trainer}")
    if "--normal-weight" not in trainer.read_text(encoding="utf-8"):
        raise RuntimeError("Trainer lacks --normal-weight; update diann-fintune before preparing this sweep")
    output.mkdir(parents=True)
    target_designs: dict[str, dict] = {}
    jobs: list[dict] = []
    for target in TARGETS:
        values = require_inputs(dataset_dir, target)
        counts = training_counts(values["train"])
        oversampling = {name: factors_for_target_rows(counts, target_rows) for name, target_rows in OVERSAMPLING_TARGET_ROWS.items()}
        budgets = {name: oversampling_budget(counts, factors) for name, factors in oversampling.items()}
        fixed_steps = int(budgets[FIXED_STEP_PROFILE]["batches_per_epoch"])
        target_designs[target] = {"counts": counts, "oversampling": oversampling, "budgets": budgets, "fixed_steps": fixed_steps}
        for loss_profile, details in LOSS_PROFILES.items():
            for sampling_profile, factors in oversampling.items():
                row_weights = loss_weights(loss_profile, counts, factors)
                for lr in LEARNING_RATES:
                    for seed in SEEDS:
                        run = output / "runs" / target / loss_profile / sampling_profile / lr_label(lr) / f"seed_{seed}"
                        train = [
                            args.diann_python, str(trainer),
                            "--train-split", str(values["train"]), "--validation-split", str(values["validation"]),
                            "--model-dir", str(DIANN_MODEL), "--output-dir", str(run), "--device", "cuda",
                            "--epochs", str(EPOCHS), "--warmup-epochs", str(WARMUP_EPOCHS),
                            "--batch-size", str(BATCH_SIZE), "--steps-per-epoch", str(fixed_steps),
                            "--loss", "mae", "--weight-decay", "0", "--seed", str(seed),
                            "--trainable", "all", "--learning-rate", f"{lr:.12g}", "--save-heavy-checkpoint-grid", "--no-plots",
                        ]
                        for flag, value in row_weights.items():
                            train.extend([f"--{flag.replace('_', '-')}", f"{value:.12g}"])
                        for flag, value in zip(("heavy-nterm-oversample", "heavy-lysine-oversample", "heavy-both-oversample"), factors):
                            train.extend([f"--{flag}", f"{value:.12g}"])
                        metrics = [run / "validation_metrics" / f"checkpoint_{selection}.json" for selection in CHECKPOINT_SELECTIONS]
                        jobs.append({
                            "id": f"{target}/{loss_profile}/{sampling_profile}/{lr_label(lr)}/seed_{seed}",
                            "workdir": str(DIANN),
                            "commands": [train, *[evaluation_command(args.diann_python, run, values, selection) for selection in CHECKPOINT_SELECTIONS]],
                            "completion_files": [str(path) for path in metrics],
                            "progress_regex": r"epoch=(\d+)/(\d+)", "progress_total": EPOCHS,
                            "metadata": {
                                "model": "diann", "target": target, "arm": "dimethyl_loss_focus_fixed_steps",
                                "loss_profile": loss_profile, "loss_profile_description": details["description"],
                                "loss_group_masses": details["group_masses"], "row_loss_weights": row_weights,
                                "sampling_profile": sampling_profile, "oversampling": profile_text(factors),
                                "oversampling_budget": budgets[sampling_profile], "fixed_steps_per_epoch": fixed_steps,
                                "epochs": EPOCHS, "lr": lr, "seed": seed,
                                "checkpoint_selections": list(CHECKPOINT_SELECTIONS),
                                "validation_only": True, "test_partition_referenced": False,
                            },
                        })
    expected = len(TARGETS) * len(LOSS_PROFILES) * len(OVERSAMPLING_TARGET_ROWS) * len(LEARNING_RATES) * len(SEEDS)
    if len(jobs) != expected or expected != 324:
        raise AssertionError(f"Expected exactly 324 jobs, found {len(jobs)}")
    manifest = {
        "study": "diann_dimethyl_loss_focus_fixed_steps_v2_chronologer_anchor",
        "dataset": str(dataset_dir), "validation_only": True, "test_partition": "closed",
        "primary_selection": "0_100",
        "selection_metric": "equal mean of exact Heavy N-term, Heavy Lys, Heavy-both validation MAE",
        "fixed_steps_per_epoch_by_target": {target: design["fixed_steps"] for target, design in target_designs.items()},
        "loss_profiles": LOSS_PROFILES,
        "target_designs": target_designs,
        "jobs": jobs,
    }
    (output / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "training_budget.json").write_text(
        json.dumps(target_designs, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output, dataset_dir, target_designs)
    print(json.dumps({"jobs": len(jobs), "targets": list(TARGETS), "manifest": str(output / 'sweep_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
