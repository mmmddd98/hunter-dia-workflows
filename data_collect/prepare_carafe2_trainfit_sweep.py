#!/usr/bin/env python3
"""Prepare the Carafe2 counterpart of the broad RT train-fit screen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CARAFE2 = Path(os.environ.get("CARAFE2_ROOT", ROOT / "carafe2-semi"))
TRAINER = CARAFE2 / "src/main/resources/py/v2/ai.py"

MODES = ("anchor_irt", "chronologer_hi", "linear", "global_60min")
SEEDS = (2024, 2025, 2026)
LR_LABELS = ("very_low", "low", "base", "high", "very_high")
LR_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)

# These are the five transfer-learning policies implemented by Carafe2. The
# absolute bases mirror the comparable policies in the DIA-NN screen.
POLICIES = {
    "head": {"trainable": "head", "lr": 1e-4},
    "cnn_head": {"trainable": "cnn_head", "lr": 1e-4, "backbone_lr": 1e-5},
    "rnn_head": {"trainable": "rnn_head", "lr": 1e-4, "backbone_lr": 1e-5},
    "discriminative": {"trainable": "discriminative", "lr": 1e-4, "backbone_lr": 1e-5},
    "all": {"trainable": "all", "lr": 3e-5},
}


def parse_args() -> argparse.Namespace:
    default_python = Path("python")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("trainingdata/finalrt_4source_trainfit15_validation10_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--carafe-python",
        default=os.environ.get("CARAFE2_PYTHON", str(default_python if default_python.is_file() else Path(sys.executable))),
    )
    return parser.parse_args()


def scaled_arguments(policy: dict, factor: float) -> list[str]:
    arguments = [
        "--rt_trainable", policy["trainable"],
        "--rt_lr", str(policy["lr"] * factor),
    ]
    if "backbone_lr" in policy:
        arguments += ["--rt_backbone_lr", str(policy["backbone_lr"] * factor)]
    return arguments


def scientific_or_dash(value: float | None) -> str:
    return "–" if value is None else f"{value:.0e}"


def main() -> None:
    args = parse_args()
    data = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    python = Path(args.carafe_python).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if not TRAINER.is_file():
        raise FileNotFoundError(TRAINER)
    if not python.is_file():
        raise FileNotFoundError(python)

    jobs: list[dict] = []
    prepared_inputs = output / "prepared_inputs"
    for mode in MODES:
        train = data / mode / "train/diann_rt_input/rt_train_data.tsv"
        validation = data / mode / "validation/diann_rt_input/rt_validation_data.tsv"
        validation_partition = data / mode / "validation"
        alignment = data / mode / "rt_alignment.json"
        for required in (train, validation, validation_partition / "rt_observations.tsv", alignment):
            if not required.is_file():
                raise FileNotFoundError(required)
        input_dir = prepared_inputs / mode
        for policy_name, policy in POLICIES.items():
            for lr_label, lr_factor in zip(LR_LABELS, LR_FACTORS):
                for seed in SEEDS:
                    run = output / "runs" / mode / policy_name / lr_label / f"seed_{seed}"
                    training = [
                        str(python), str(TRAINER),
                        "--in_dir", str(input_dir), "--out_dir", str(run),
                        "--device", "gpu", "--tf_type", "rt", "--seed", str(seed),
                        "--verbose", "1", "--use_best_model",
                        "--rt_validation_file", str(validation),
                        "--rt_validation_disjoint_by", "peptidoform",
                        "--rt_epoch", "40", "--rt_warmup_epoch", "10",
                        "--rt_batch_size", "512", "--no_adjust_rt_batch_size",
                        "--rt_loss", "l1", "--rt_weight_decay", "0",
                        "--rt_heavy_nterm_oversample", "1",
                        "--rt_heavy_lysine_oversample", "1",
                        "--rt_heavy_both_oversample", "1",
                        "--rt_heavy_nterm_weight", "1",
                        "--rt_heavy_lysine_weight", "1",
                        "--rt_heavy_both_weight", "1",
                        *scaled_arguments(policy, lr_factor),
                    ]
                    evaluation = [
                        str(python), str(ROOT / "ma/data_collect/evaluate_carafe2_trainfit.py"),
                        "--run-dir", str(run), "--partition-dir", str(validation_partition),
                        "--alignment", str(alignment),
                        "--output", str(run / "validation_metrics.json"),
                        "--device", "gpu", "--python", str(python),
                    ]
                    jobs.append({
                        "id": f"{mode}/{policy_name}/{lr_label}/seed_{seed}",
                        "workdir": str(CARAFE2), "commands": [training, evaluation],
                        "completion_files": [str(run / "validation_metrics.json")],
                        "progress_regex": r"Epoch=(\d+)", "progress_total": 40,
                    })

    expected = len(MODES) * len(POLICIES) * len(LR_FACTORS) * len(SEEDS)
    assert len(jobs) == expected == 300
    output.mkdir(parents=True)
    for mode in MODES:
        input_dir = prepared_inputs / mode
        input_dir.mkdir(parents=True)
        source = data / mode / "train/diann_rt_input/rt_train_data.tsv"
        (input_dir / "rt_train_data.tsv").symlink_to(source)
    (output / "sweep_manifest.json").write_text(
        json.dumps({"study": "carafe2_trainfit_broad_screen_v1", "jobs": jobs}, indent=2) + "\n",
        encoding="utf-8",
    )

    policy_text = "\n".join(
        f"| `{name}` | `{spec['trainable']}` | {spec['lr']:.0e} | "
        f"{scientific_or_dash(spec.get('backbone_lr'))} |"
        for name, spec in POLICIES.items()
    )
    worker = ROOT / "ma/data_collect/sweep_template/parallel_gpu_sweep.py"
    analyzer = ROOT / "ma/data_collect/analyze_carafe2_trainfit_sweep.py"
    readme = f"""# Broad Carafe2 Train-fit Screening Sweep

This **not-yet-started 300-run** screen is the Carafe2 counterpart of the DIA-NN broad screen. It uses the same four training-only-fitted RT target spaces, five learning-rate levels, three seeds, fixed validation partition and validation metrics. The held-out test partition remains closed.

Carafe2 implements five relevant RT transfer-learning policies (`head`, `cnn_head`, `rnn_head`, `discriminative`, `all`). DIA-NN's token-embedding-only and separate RT-head policies have no direct Carafe2 CLI equivalent and are therefore not fabricated. Thus the scientifically comparable full Carafe2 grid is **4 × 5 × 5 × 3 = 300**, rather than 420 runs.

## Fixed data and training contract

- Dataset: `{data}`
- Per target space: 60,435 train, 6,715 validation and 11,850 untouched test peptidoforms.
- Alignment/calibration was fitted on the training partition only.
- Training: 40 epochs, 10 warm-up epochs, batch size 512, L1 loss, weight decay 0.
- Heavy-Dimethyl sampling and loss weights remain 1.0.
- Best checkpoint selection uses only the fixed exact-peptidoform-disjoint validation TSV; Carafe2 explicitly verifies non-overlap of `sequence + mods + mod_sites`.
- Validation evaluation reports aggregate, source, organism, digestion and exact Heavy-Dimethyl metrics on normalized and observed-minute scales.

| Factor | Values | Count |
|---|---|---:|
| RT target space | `anchor_irt`, `chronologer_hi`, `linear`, `global_60min` | 4 |
| Trainable policy | output head, CNN+head, RNN+head, discriminative, all | 5 |
| LR level | 0.1×, 0.3×, 1×, 3×, 10× policy-specific base | 5 |
| Seed | 2024, 2025, 2026 | 3 |
| **Total** | 4 × 5 × 5 × 3 | **300** |

| Policy label | Carafe2 policy | Base head LR | Base backbone LR |
|---|---|---:|---:|
{policy_text}

For `all`, absolute LRs are 3e-6, 9e-6, 3e-5, 9e-5 and 3e-4. Other head LRs are 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3; active pretrained backbones remain tenfold lower.

## Start

```bash
python3 {worker} \\
  --manifest {output / 'sweep_manifest.json'} \\
  --gpus 0,1,2 --jobs-per-gpu 2
```

## Analyze and create publication figures

```bash
python3 {analyzer} \\
  --sweep-dir {output} \\
  --output-dir {output / 'analysis'}
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "jobs": len(jobs), "manifest": str(output / "sweep_manifest.json"),
        "carafe_python": str(python),
    }, indent=2))


if __name__ == "__main__":
    main()
