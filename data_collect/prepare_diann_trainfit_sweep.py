#!/usr/bin/env python3
"""Prepare the broad 420-run DIA-NN RT fine-tuning screening sweep."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIANN = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune"))
MODEL = Path(os.environ.get("DIANN_MODEL_DIR", ROOT.parent / "diann_sc/DIA-NN-2.5.0/diann-2.5.0/models"))

MODES = ("anchor_irt", "chronologer_hi", "linear", "global_60min")
SEEDS = (2024, 2025, 2026)
LR_LABELS = ("very_low", "low", "base", "high", "very_high")
LR_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)

# Each policy has a base LR and, where relevant, a deliberately smaller
# backbone LR. Applying the five factors produces a comparable LR sweep while
# retaining safer absolute rates for full-network adaptation.
POLICIES = {
    "embeddings": {"trainable": "embeddings", "lr": 1e-4, "embedding_lr": 1e-4},
    "head": {"trainable": "head", "lr": 1e-4, "embedding_lr": 1e-4},
    "rt_head": {"trainable": "rt_head", "lr": 1e-4, "backbone_lr": 1e-5, "embedding_lr": 1e-4},
    "cnn_head": {"trainable": "cnn_head", "lr": 1e-4, "backbone_lr": 1e-5, "embedding_lr": 1e-4},
    "gru_head": {"trainable": "gru_head", "lr": 1e-4, "backbone_lr": 1e-5, "embedding_lr": 1e-4},
    "discriminative": {"trainable": "discriminative", "lr": 1e-4, "backbone_lr": 1e-5, "embedding_lr": 1e-4},
    "all": {"trainable": "all", "lr": 3e-5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("trainingdata/finalrt_4source_trainfit15_validation10_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable))
    return parser.parse_args()


def scaled_arguments(policy: dict, factor: float) -> list[str]:
    arguments = ["--trainable", policy["trainable"], "--learning-rate", str(policy["lr"] * factor)]
    if "backbone_lr" in policy:
        arguments += ["--backbone-learning-rate", str(policy["backbone_lr"] * factor)]
    if "embedding_lr" in policy:
        arguments += ["--embedding-learning-rate", str(policy["embedding_lr"] * factor)]
    return arguments


def scientific_or_dash(value: float | None) -> str:
    return "–" if value is None else f"{value:.0e}"


def main() -> None:
    args = parse_args()
    data = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not (DIANN / "train_carafe_rt.py").is_file():
        raise FileNotFoundError(DIANN / "train_carafe_rt.py")
    if not (MODEL / "rt.d0.pt").is_file():
        raise FileNotFoundError(MODEL / "rt.d0.pt")

    jobs: list[dict] = []
    for mode in MODES:
        train = data / mode / "train/diann_rt_input/rt_train_data.tsv"
        validation = data / mode / "validation/diann_rt_input/rt_validation_data.tsv"
        validation_partition = data / mode / "validation"
        alignment = data / mode / "rt_alignment.json"
        for required in (train, validation, alignment):
            if not required.is_file():
                raise FileNotFoundError(required)
        for policy_name, policy in POLICIES.items():
            for lr_label, lr_factor in zip(LR_LABELS, LR_FACTORS):
                for seed in SEEDS:
                    run = output / "runs" / mode / policy_name / lr_label / f"seed_{seed}"
                    training = [
                        args.diann_python, str(DIANN / "train_carafe_rt.py"),
                        "--train-split", str(train), "--validation-split", str(validation),
                        "--model-dir", str(MODEL), "--output-dir", str(run),
                        "--device", "cuda", "--epochs", "40", "--warmup-epochs", "10",
                        "--batch-size", "512", "--loss", "mae", "--weight-decay", "0",
                        "--heavy-nterm-weight", "1", "--heavy-lysine-weight", "1",
                        "--heavy-both-weight", "1", "--seed", str(seed), "--no-plots",
                        *scaled_arguments(policy, lr_factor),
                    ]
                    evaluation = [
                        args.diann_python, str(ROOT / "ma/data_collect/evaluate_diann_trainfit.py"),
                        "--run-dir", str(run), "--partition-dir", str(validation_partition),
                        "--alignment", str(alignment), "--output", str(run / "validation_metrics.json"),
                        "--device", "cuda",
                    ]
                    jobs.append({
                        "id": f"{mode}/{policy_name}/{lr_label}/seed_{seed}",
                        "workdir": str(DIANN), "commands": [training, evaluation],
                        "completion_files": [str(run / "validation_metrics.json")],
                        "progress_regex": r"epoch=(\d+)/(\d+)", "progress_total": 40,
                    })

    expected = len(MODES) * len(POLICIES) * len(LR_FACTORS) * len(SEEDS)
    assert len(jobs) == expected == 420
    output.mkdir(parents=True)
    (output / "sweep_manifest.json").write_text(
        json.dumps({"study": "diann_trainfit_broad_screen_v2", "jobs": jobs}, indent=2) + "\n",
        encoding="utf-8",
    )
    policy_text = "\n".join(
        f"| `{name}` | `{spec['trainable']}` | {spec['lr']:.0e} | "
        f"{scientific_or_dash(spec.get('backbone_lr'))} | "
        f"{scientific_or_dash(spec.get('embedding_lr'))} |"
        for name, spec in POLICIES.items()
    )
    readme = f"""# Broad DIA-NN Train-fit Screening Sweep

This **not-yet-started 420-run** screen tests every planned combination of four RT target spaces, seven DIA-NN layer policies, five LR levels and three seeds. It is validation-only; the final test partition remains closed.

## Fixed data contract

- Dataset: `{data}`
- Per target space: 60,435 train, 6,715 validation and 11,850 held-out test peptidoforms.
- The split unit is the modified peptidoform (`sequence + mods + mod_sites`), so distinct modification forms of one amino-acid sequence may occur in different partitions, but an exact labelled peptidoform never does.
- Alignment was fit from training observations only and frozen before validation/test transformation.
- Training: 40 epochs, 10 warm-up epochs, batch size 512, MAE, and all Heavy-Dimethyl weights/oversampling fixed at 1.0.

## Full factorial grid

| Factor | Values | Count |
|---|---|---:|
| RT target space | `anchor_irt`, `chronologer_hi`, `linear`, `global_60min` | 4 |
| Layer policy | embeddings, output head, RT head, CNN+RT head, GRU+RT head, discriminative, all | 7 |
| LR level | 0.1×, 0.3×, 1×, 3×, 10× policy-specific base LR | 5 |
| Seed | 2024, 2025, 2026 | 3 |
| **Total** | 4 × 7 × 5 × 3 | **420** |

| Policy label | DIA-NN policy | Base head LR | Base backbone LR | Base custom-embedding LR |
|---|---|---:|---:|---:|
{policy_text}

For `all`, the five absolute LRs are 3e-6, 1e-5, 3e-5, 1e-4 and 3e-4. For the other policies the head/custom-embedding LR levels are 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3; when a backbone is active, it remains tenfold lower. This probes a broad range while preserving discriminative learning rates for pretrained layers.

## Selection and safety

The evaluator reports aggregate, per-source and per-organism errors in both model target scale and observed RT minutes, plus fully/semi-tryptic and exact Heavy-Dimethyl strata. The primary validation ranking is mean source-balanced raw-minute MAE over seeds. Worst source, organism macro-average, semi-tryptic and Heavy-Dimethyl errors are guardrails. Heavy-Dimethyl weighting is deliberately not varied here, so it cannot confound the normalization/layer/LR selection; it is a separate second-stage sweep after a winner is selected.

## Start on GPUs 0–2

```bash
python3 {ROOT / 'ma/data_collect/sweep_template/parallel_gpu_sweep.py'} \\
  --manifest {output / 'sweep_manifest.json'} --gpus 0,1,2 --jobs-per-gpu 1
```

The reusable worker resumes any job only when its `validation_metrics.json` exists and matches the manifest fingerprint. It stores logs and status in `{output / '_sweep_logs'}`.

## Analyze after completion

```bash
python3 {ROOT / 'ma/data_collect/analyze_diann_trainfit_sweep.py'} \\
  --sweep-dir {output} --output-dir {output / 'analysis'}
```

This command also writes the reusable publication figure set to `analysis/figures` in 600-dpi PNG, vector PDF and editable SVG formats.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "manifest": str(output / "sweep_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
