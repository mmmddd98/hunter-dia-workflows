#!/usr/bin/env python3
"""Prepare a validation-only DIA-NN Heavy-Dimethyl checkpoint-grid sweep."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from heavy_dimethyl_sweep_common import (
    BATCH_SIZE, EPOCHS, SEEDS, WARMUP_EPOCHS, lr_label,
    oversampling_budget, profile_text, training_counts,
)

ROOT = Path(__file__).resolve().parents[2]
DIANN = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune"))
DIANN_MODEL = Path(os.environ.get(
    "DIANN_MODEL_DIR", ROOT.parent / "diann_sc/DIA-NN-2.5.0/diann-2.5.0/models",
))
DATASET_DEFAULT = ROOT / "trainingdata/finalrt_4source_trainfit15_validation10_v1"
TARGET = "chronologer_hi"
LEARNING_RATES = (5e-5, 1e-4, 1.5e-4)
CHECKPOINT_SELECTIONS = ("100_0", "75_25", "50_50", "25_75", "0_100")
TARGET_HEAVY_ROWS = {"H24": 24_000, "H32": 32_000, "H40": 40_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable))
    return parser.parse_args()


def require_inputs(dataset_dir: Path) -> dict[str, Path]:
    base = dataset_dir / TARGET
    values = {
        "train": base / "train/diann_rt_input/rt_train_data.tsv",
        "validation": base / "validation/diann_rt_input/rt_validation_data.tsv",
        "partition": base / "validation",
        "alignment": base / "rt_alignment.json",
    }
    required_files = (values["train"], values["validation"], values["alignment"], values["partition"] / "rt_observations.tsv")
    missing = [path for path in required_files if not path.is_file()]
    if not values["partition"].is_dir():
        missing.append(values["partition"])
    if missing:
        raise FileNotFoundError(f"Missing {TARGET} training/validation input: {missing}")
    if any("test" in str(path).lower() for path in values.values()):
        raise ValueError("Priority sweep must not reference a test partition")
    return values


def factors_for_target_rows(counts: dict[str, int], target_rows: int) -> tuple[float, float, float]:
    keys = ("heavy_nterm", "heavy_lysine", "heavy_both")
    if any(counts[key] <= 0 for key in keys):
        raise ValueError(f"Cannot balance missing Heavy stratum: {counts}")
    return tuple(target_rows / counts[key] for key in keys)


def evaluation_command(python: str, run: Path, values: dict[str, Path], selection: str) -> list[str]:
    return [
        python, str(ROOT / "ma/data_collect/evaluate_diann_trainfit.py"),
        "--run-dir", str(run), "--partition-dir", str(values["partition"]),
        "--alignment", str(values["alignment"]), "--checkpoint-selection", selection,
        "--output", str(run / "validation_metrics" / f"checkpoint_{selection}.json"),
        "--device", "cuda",
    ]


def write_readme(output: Path, dataset_dir: Path, profiles: dict[str, tuple[float, float, float]], budgets: dict[str, dict]) -> None:
    profile_rows = "\n".join(
        f"| `{name}` | `{profile_text(factors)}` | {budgets[name]['effective_training_rows']:,} | "
        f"{budgets[name]['batches_per_epoch']:,} | {budgets[name]['total_optimizer_steps']:,} |"
        for name, factors in profiles.items()
    )
    text = f"""# DIA-NN Heavy-Dimethyl priority sweep

This is a **validation-only** DIA-NN sweep. It tests whether strong,
stratum-balanced Heavy-Dimethyl oversampling can improve all three Heavy
groups. The final test partition is intentionally absent from the manifest.

## Fixed design

- Target space: `{TARGET}` only
- Objective / policy: unweighted MAE; all model parameters trainable
- Epochs / warm-up / batch size: `{EPOCHS}` / `{WARMUP_EPOCHS}` / `{BATCH_SIZE}`
- Learning rates: {', '.join(f'`{value:g}`' for value in LEARNING_RATES)}
- Seeds: {', '.join(map(str, SEEDS))}
- Runs: 4 profiles × 3 learning rates × 3 seeds = **36 runs**

Every run writes these validation-selected checkpoint candidates:

| Label | Overall / Heavy-Macro priority |
|---|---:|
| `100_0` | 100% / 0% |
| `75_25` | 75% / 25% |
| `50_50` | 50% / 50% |
| `25_75` | 25% / 75% |
| `0_100` | 0% / 100% |

Heavy-Macro is `(Heavy N-term + Heavy Lys + Heavy both) / 3`; all three
groups therefore count equally. The run-root checkpoint is `100_0`; the other
four live under `checkpoints/`. `checkpoint_selection.json` records best epoch,
score and the individual Validation-MAEs for every selection rule.

## Stratum-balanced oversampling

`H24`, `H32`, and `H40` target this many rows per exact Heavy stratum after
the trainer's documented rounding. Normal peptides remain at 1×. Fixed 40
epochs imply larger profiles also receive more optimizer steps; these practical
training budgets are recorded explicitly.

| Profile | N-term / Lys / both factor | Rows / epoch | Batches / epoch | Steps / 40 epochs |
|---|---:|---:|---:|---:|
{profile_rows}

## Commands

```bash
python3 {ROOT / 'ma/data_collect/prepare_diann_heavy_priority_sweep.py'} \\
  --output-dir {output}

python3 {ROOT / 'ma/data_collect/sweep_template/parallel_gpu_sweep.py'} \\
  --manifest {output / 'sweep_manifest.json'} --gpus 0,1,2 --jobs-per-gpu 1

python3 {ROOT / 'ma/data_collect/analyze_diann_heavy_priority_sweep.py'} \\
  --sweep-dir {output} --output-dir {output / 'analysis'}
```

Use only validation to choose profile, learning rate and checkpoint priority.
Freeze exactly one result before evaluating any new, untouched hold-out.

Dataset: `{dataset_dir}`.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output, dataset_dir = args.output_dir.resolve(), args.dataset_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep directory: {output}")
    if not (DIANN / "train_carafe_rt.py").is_file():
        raise FileNotFoundError(f"DIA-NN trainer unavailable: {DIANN / 'train_carafe_rt.py'}")
    values = require_inputs(dataset_dir)
    counts = training_counts(values["train"])
    profiles = {"C0": (1.0, 1.0, 1.0)}
    profiles.update({name: factors_for_target_rows(counts, target) for name, target in TARGET_HEAVY_ROWS.items()})
    budgets = {name: oversampling_budget(counts, factors) for name, factors in profiles.items()}
    output.mkdir(parents=True)

    jobs: list[dict] = []
    for profile, oversampling in profiles.items():
        for lr in LEARNING_RATES:
            for seed in SEEDS:
                run = output / "runs" / TARGET / profile / lr_label(lr) / f"seed_{seed}"
                train = [
                    args.diann_python, str(DIANN / "train_carafe_rt.py"),
                    "--train-split", str(values["train"]), "--validation-split", str(values["validation"]),
                    "--model-dir", str(DIANN_MODEL), "--output-dir", str(run), "--device", "cuda",
                    "--epochs", str(EPOCHS), "--warmup-epochs", str(WARMUP_EPOCHS),
                    "--batch-size", str(BATCH_SIZE), "--loss", "mae", "--weight-decay", "0",
                    "--heavy-nterm-weight", "1", "--heavy-lysine-weight", "1", "--heavy-both-weight", "1",
                    "--heavy-nterm-oversample", f"{oversampling[0]:.12g}",
                    "--heavy-lysine-oversample", f"{oversampling[1]:.12g}",
                    "--heavy-both-oversample", f"{oversampling[2]:.12g}",
                    "--seed", str(seed), "--trainable", "all", "--learning-rate", f"{lr:.12g}",
                    "--save-heavy-checkpoint-grid", "--no-plots",
                ]
                metrics = [run / "validation_metrics" / f"checkpoint_{selection}.json" for selection in CHECKPOINT_SELECTIONS]
                jobs.append({
                    "id": f"{TARGET}/{profile}/{lr_label(lr)}/seed_{seed}",
                    "workdir": str(DIANN),
                    "commands": [train, *[evaluation_command(args.diann_python, run, values, selection) for selection in CHECKPOINT_SELECTIONS]],
                    "completion_files": [str(path) for path in metrics],
                    "progress_regex": r"epoch=(\\d+)/(\\d+)", "progress_total": EPOCHS,
                    "metadata": {
                        "model": "diann", "target": TARGET, "arm": "heavy_priority_oversampling",
                        "profile": profile, "loss": "mae", "lr": lr, "seed": seed,
                        "weights": "1/1/1", "oversampling": profile_text(oversampling),
                        "training_budget": budgets[profile], "checkpoint_selections": list(CHECKPOINT_SELECTIONS),
                        "validation_only": True, "test_partition_referenced": False,
                    },
                })
    if len(jobs) != 36:
        raise AssertionError(f"Expected 36 jobs, found {len(jobs)}")
    manifest = {
        "study": "diann_heavy_dimethyl_priority_checkpoint_grid_v1",
        "dataset": str(dataset_dir), "validation_only": True, "test_partition": "closed",
        "checkpoint_selection_grid": {
            "metric_scale_during_training": "rt_norm",
            "heavy_macro": "mean(exact Heavy N-term, Heavy Lys, Heavy both MAE)",
            "priorities": list(CHECKPOINT_SELECTIONS),
        },
        "profiles": {name: {"oversampling": profile_text(factors), "training_budget": budgets[name]} for name, factors in profiles.items()},
        "jobs": jobs,
    }
    (output / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "training_budget.json").write_text(json.dumps({"base_counts": counts, "profiles": budgets}, indent=2) + "\n", encoding="utf-8")
    write_readme(output, dataset_dir, profiles, budgets)
    print(json.dumps({"jobs": len(jobs), "manifest": str(output / 'sweep_manifest.json'), "validation_only": True}, indent=2))


if __name__ == "__main__":
    main()
