#!/usr/bin/env python3
"""Run deterministic control or finalist campaigns for the full RT sweep."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DIANN_ROOT = Path(os.environ.get("DIANN_FINETUNE_ROOT", WORKSPACE_ROOT.parent / "diann-fintune")).expanduser()
CARAFE_ROOT = Path(os.environ.get("CARAFE2_ROOT", WORKSPACE_ROOT / "carafe2-semi")).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("diann", "carafe2"), required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration-map", type=Path, required=True)
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--diann-model-dir", type=Path,
        default=Path(os.environ.get(
            "DIANN_MODEL_DIR",
            WORKSPACE_ROOT.parent / "diann_sc" / "DIA-NN-2.5.0" / "diann-2.5.0" / "models",
        )).expanduser(),
    )
    parser.add_argument("--wandb-config", type=Path)
    parser.add_argument("--wandb-group")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds requires unique comma-separated integers")
    return seeds


def add_options(command: list[str], config: dict, names: tuple[str, ...]) -> None:
    for name in names:
        if name in config:
            command.extend([f"--{name}", str(config[name])])


def main() -> None:
    args = parse_args()
    configurations = json.loads(args.configuration_map.read_text(encoding="utf-8"))
    if not configurations:
        raise SystemExit("Configuration map is empty")
    seeds = parse_seeds(args.seeds)
    train_dir = args.prepared_dir / "train"
    validation_dir = args.prepared_dir / "validation"
    required = [
        train_dir / "rt_train_data.tsv", validation_dir / "combined.tsv",
        *[validation_dir / f"{name}.tsv" for name in ("celegans", "triplemix", "human")],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing prepared sweep input(s):\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, original_config in configurations.items():
        for seed in seeds:
            run_dir = args.output_dir / f"config.{name}" / f"seed.{seed}"
            validation_metrics = run_dir / "validation_metrics.json"
            if validation_metrics.is_file():
                print(f"Skipping complete campaign run: {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            config = dict(original_config)
            environment = os.environ.copy()
            if args.model == "diann":
                if "backbone-lr-ratio" in config and "backbone-learning-rate" not in config:
                    config["backbone-learning-rate"] = (
                        float(config["learning-rate"]) * float(config["backbone-lr-ratio"])
                    )
                command = [
                    args.python, "-u", str(DIANN_ROOT / "train_carafe_rt.py"),
                    "--train-split", str(train_dir / "rt_train_data.tsv"),
                    "--validation-split", str(validation_dir / "combined.tsv"),
                    "--model-dir", str(args.diann_model_dir), "--output-dir", str(run_dir),
                    "--device", args.device, "--seed", str(seed),
                ]
                add_options(command, config, (
                    "epochs", "warmup-epochs", "batch-size", "learning-rate",
                    "backbone-learning-rate", "embedding-learning-rate", "weight-decay",
                    "loss", "huber-delta", "trainable", "heavy-nterm-weight",
                    "heavy-lysine-weight", "heavy-both-weight", "heavy-nterm-oversample",
                    "heavy-lysine-oversample", "heavy-both-oversample",
                ))
                evaluator = [
                    args.python, str(DIANN_ROOT / "evaluate_organism_tests.py"),
                    "--run-dir", str(run_dir), "--tests-dir", str(validation_dir),
                    "--device", args.device,
                    "--rt-alignment", str(validation_dir / "rt_alignment.json"),
                ]
                cwd = DIANN_ROOT
            else:
                if "rt_backbone_lr_ratio" in config and "rt_backbone_lr" not in config:
                    config["rt_backbone_lr"] = float(config["rt_lr"]) * float(config["rt_backbone_lr_ratio"])
                command = [
                    args.python, "-u", str(CARAFE_ROOT / "src/main/resources/py/v2/ai.py"),
                    "--in_dir", str(train_dir), "--out_dir", str(run_dir),
                    "--device", args.device, "--tf_type", "rt", "--seed", str(seed),
                    "--verbose", "1", "--use_best_model",
                    "--rt_validation_file", str(validation_dir / "combined.tsv"),
                    "--no_adjust_rt_batch_size",
                ]
                add_options(command, config, (
                    "rt_lr", "rt_batch_size", "rt_epoch", "rt_warmup_epoch",
                    "rt_heavy_nterm_oversample", "rt_heavy_lysine_oversample",
                    "rt_heavy_both_oversample", "rt_heavy_nterm_weight",
                    "rt_heavy_lysine_weight", "rt_heavy_both_weight", "rt_loss",
                    "rt_huber_delta", "rt_trainable", "rt_backbone_lr", "rt_weight_decay",
                ))
                evaluator = [
                    args.python, str(CARAFE_ROOT / "evaluate_organism_tests.py"),
                    "--run-dir", str(run_dir), "--tests-dir", str(validation_dir),
                    "--device", args.device, "--python", args.python,
                    "--rt-alignment", str(validation_dir / "rt_alignment.json"),
                ]
                cwd = CARAFE_ROOT
            if args.wandb_config:
                command.extend([
                    "--wandb-config", str(args.wandb_config),
                    "--wandb-run-name", f"{name}.seed{seed}",
                    "--wandb-group", args.wandb_group or args.output_dir.name,
                ])
            (run_dir / "command.json").write_text(
                json.dumps({"wandb_config": config, "seed": seed, "command": command}, indent=2) + "\n",
                encoding="utf-8",
            )
            print("Running:", " ".join(command), flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, cwd=cwd, env=environment, check=True)
            subprocess.run(evaluator, cwd=cwd, env=environment, check=True)
            shutil.copy2(run_dir / "organism_test_metrics.json", validation_metrics)
    print(f"Campaign complete: {args.output_dir}")


if __name__ == "__main__":
    main()
