#!/usr/bin/env python3
"""Prepare a focused second-stage RT LR/policy sweep for DIA-NN or Carafe2.

The first screen established that Chronologer HI and Anchor iRT are the only
competitive target coordinates, and that deep adaptation is required.  This
script therefore refines the learning-rate decision without reopening the
already excluded factors or the held-out test partition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIANN = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune"))
DIANN_MODEL = Path(os.environ.get("DIANN_MODEL_DIR", ROOT.parent / "diann_sc/DIA-NN-2.5.0/diann-2.5.0/models"))
CARAFE2 = Path(os.environ.get("CARAFE2_ROOT", ROOT / "carafe2-semi"))
CARAFE2_TRAINER = CARAFE2 / "src/main/resources/py/v2/ai.py"

MODES = ("chronologer_hi", "anchor_irt")
SEEDS = (2024, 2025, 2026)
LR_LEVELS = (
    ("high", 3.0), ("five_x", 5.0), ("very_high", 10.0),
    ("fifteen_x", 15.0), ("twenty_x", 20.0), ("thirty_x", 30.0),
)
POLICIES = {
    "discriminative": {"trainable": "discriminative", "lr": 1e-4, "backbone_lr": 1e-5},
    "all": {"trainable": "all", "lr": 3e-5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("diann", "carafe2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("trainingdata/finalrt_4source_trainfit15_validation10_v1"))
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable))
    default_carafe = Path("python")
    parser.add_argument("--carafe-python", default=os.environ.get("CARAFE2_PYTHON", str(default_carafe if default_carafe.is_file() else Path(sys.executable))))
    return parser.parse_args()


def scaled_diann(policy: dict, factor: float) -> list[str]:
    args = ["--trainable", policy["trainable"], "--learning-rate", str(policy["lr"] * factor)]
    if "backbone_lr" in policy:
        args += ["--backbone-learning-rate", str(policy["backbone_lr"] * factor)]
    return args


def scaled_carafe(policy: dict, factor: float) -> list[str]:
    args = ["--rt_trainable", policy["trainable"], "--rt_lr", str(policy["lr"] * factor)]
    if "backbone_lr" in policy:
        args += ["--rt_backbone_lr", str(policy["backbone_lr"] * factor)]
    return args


def job_for_diann(python: str, train: Path, validation: Path, partition: Path, alignment: Path, run: Path, policy: dict, factor: float, seed: int) -> dict:
    training = [
        python, str(DIANN / "train_carafe_rt.py"), "--train-split", str(train),
        "--validation-split", str(validation), "--model-dir", str(DIANN_MODEL),
        "--output-dir", str(run), "--device", "cuda", "--epochs", "40",
        "--warmup-epochs", "10", "--batch-size", "512", "--loss", "mae",
        "--weight-decay", "0", "--heavy-nterm-weight", "1", "--heavy-lysine-weight", "1",
        "--heavy-both-weight", "1", "--seed", str(seed), "--no-plots",
        *scaled_diann(policy, factor),
    ]
    evaluation = [
        python, str(ROOT / "ma/data_collect/evaluate_diann_trainfit.py"), "--run-dir", str(run),
        "--partition-dir", str(partition), "--alignment", str(alignment),
        "--output", str(run / "validation_metrics.json"), "--device", "cuda",
    ]
    return {"workdir": str(DIANN), "commands": [training, evaluation]}


def job_for_carafe(python: str, input_dir: Path, validation: Path, partition: Path, alignment: Path, run: Path, policy: dict, factor: float, seed: int) -> dict:
    training = [
        python, str(CARAFE2_TRAINER), "--in_dir", str(input_dir), "--out_dir", str(run),
        "--device", "gpu", "--tf_type", "rt", "--seed", str(seed), "--verbose", "1",
        "--use_best_model", "--rt_validation_file", str(validation),
        "--rt_validation_disjoint_by", "peptidoform", "--rt_epoch", "40",
        "--rt_warmup_epoch", "10", "--rt_batch_size", "512", "--no_adjust_rt_batch_size",
        "--rt_loss", "l1", "--rt_weight_decay", "0", "--rt_heavy_nterm_oversample", "1",
        "--rt_heavy_lysine_oversample", "1", "--rt_heavy_both_oversample", "1",
        "--rt_heavy_nterm_weight", "1", "--rt_heavy_lysine_weight", "1",
        "--rt_heavy_both_weight", "1", *scaled_carafe(policy, factor),
    ]
    evaluation = [
        python, str(ROOT / "ma/data_collect/evaluate_carafe2_trainfit.py"), "--run-dir", str(run),
        "--partition-dir", str(partition), "--alignment", str(alignment),
        "--output", str(run / "validation_metrics.json"), "--device", "gpu", "--python", python,
    ]
    return {"workdir": str(CARAFE2), "commands": [training, evaluation]}


def main() -> None:
    args = parse_args()
    output, data = args.output_dir.resolve(), args.dataset_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep: {output}")
    if args.family == "diann":
        if not (DIANN / "train_carafe_rt.py").is_file() or not (DIANN_MODEL / "rt.d0.pt").is_file():
            raise FileNotFoundError("DIA-NN trainer or pretrained rt.d0.pt model is unavailable")
    else:
        if not CARAFE2_TRAINER.is_file() or not Path(args.carafe_python).expanduser().is_file():
            raise FileNotFoundError("Carafe2 RT trainer or selected Python interpreter is unavailable")

    jobs: list[dict] = []
    for mode in MODES:
        train = data / mode / "train/diann_rt_input/rt_train_data.tsv"
        validation = data / mode / "validation/diann_rt_input/rt_validation_data.tsv"
        partition, alignment = data / mode / "validation", data / mode / "rt_alignment.json"
        required = (train, validation, partition / "rt_observations.tsv", alignment)
        if not all(item.is_file() for item in required):
            raise FileNotFoundError(f"Missing refinement input for {mode}: {[str(item) for item in required if not item.is_file()]}")
        input_dir = output / "prepared_inputs" / mode
        for policy_name, policy in POLICIES.items():
            for lr_label, factor in LR_LEVELS:
                for seed in SEEDS:
                    run = output / "runs" / mode / policy_name / lr_label / f"seed_{seed}"
                    if args.family == "diann":
                        job = job_for_diann(args.diann_python, train, validation, partition, alignment, run, policy, factor, seed)
                    else:
                        job = job_for_carafe(str(Path(args.carafe_python).expanduser().resolve()), input_dir, validation, partition, alignment, run, policy, factor, seed)
                    job.update({
                        "id": f"{mode}/{policy_name}/{lr_label}/seed_{seed}",
                        "completion_files": [str(run / "validation_metrics.json")],
                        "progress_regex": r"epoch=(\d+)/(\d+)" if args.family == "diann" else r"Epoch=(\d+)",
                        "progress_total": 40,
                    })
                    jobs.append(job)

    assert len(jobs) == 72
    output.mkdir(parents=True)
    if args.family == "carafe2":
        for mode in MODES:
            input_dir = output / "prepared_inputs" / mode
            input_dir.mkdir(parents=True)
            input_dir.joinpath("rt_train_data.tsv").symlink_to(data / mode / "train/diann_rt_input/rt_train_data.tsv")
    study = f"{args.family}_trainfit_lr_refinement_v1"
    (output / "sweep_manifest.json").write_text(json.dumps({"study": study, "jobs": jobs}, indent=2) + "\n", encoding="utf-8")
    model_label = "DIA-NN" if args.family == "diann" else "Carafe2"
    analyzer = ROOT / f"ma/data_collect/analyze_{args.family}_trainfit_sweep.py"
    (output / "README.md").write_text(f"""# {model_label} RT fine-tuning: focused LR refinement

This pre-registered **72-run validation-only follow-up** converts the broad-screen result into a defensible final choice. It retains only the two competitive target spaces (Chronologer HI and Anchor iRT) and the two policies that can adapt pretrained representations (discriminative and all-parameter fine-tuning). The held-out test set remains untouched.

| Factor | Values | Count |
|---|---|---:|
| RT target | `chronologer_hi`, `anchor_irt` | 2 |
| Trainable policy | `discriminative`, `all` | 2 |
| LR multiplier | 3×, 5×, 10×, 15×, 20×, 30× policy base | 6 |
| Seed | 2024, 2025, 2026 | 3 |
| **Total** | 2 × 2 × 6 × 3 | **72** |

Training is fixed at 40 epochs with a 10-epoch warm-up, batch size 512, MAE/L1 objective, no weight decay, and all Heavy-Dimethyl sampling/weight factors set to 1. The primary selection metric is mean source-balanced validation MAE in raw minutes; seed SD, worst-source, organism macro-MAE, semi-tryptic and Heavy-Dimethyl strata remain guardrails.

The accompanying launcher resumes safely and automatically invokes `{analyzer.name}` after every job has completed successfully. It writes the complete 600-dpi PNG, PDF and SVG publication figure suite and plot-reading guides to `analysis/`.
""", encoding="utf-8")
    print(json.dumps({"study": study, "jobs": len(jobs), "manifest": str(output / "sweep_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
