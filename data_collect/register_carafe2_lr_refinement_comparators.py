#!/usr/bin/env python3
"""Register validation-selected pooled Carafe2 controls for source comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "trainingdata" / "finalrt_4source_trainfit15_validation10_v1"
DEFAULT_STUDY = Path("./shared_data/carafe2_finetune_out/carafe2_trainfit_lr_refinement_v1")
SEEDS = (2024, 2025, 2026)
CANDIDATES = (
    ("Historical combined overall · Chronologer HI/all/15×", "chronologer_hi", "all", "fifteen_x", "overall"),
    ("Historical combined heavy guardrail · Anchor iRT/all/10×", "anchor_irt", "all", "very_high", "heavy_guardrail"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study, dataset = args.study_root.resolve(), args.dataset_dir.resolve()
    manifest = json.loads((study / "sweep_manifest.json").read_text(encoding="utf-8"))
    commands = {job["id"]: job["commands"][0] for job in manifest["jobs"]}
    records = []
    for label, target, policy, level, role in CANDIDATES:
        members = []
        expected_train = (dataset / target / "train" / "diann_rt_input" / "rt_train_data.tsv").resolve()
        expected_validation = str(dataset / target / "validation" / "diann_rt_input" / "rt_validation_data.tsv")
        for seed in SEEDS:
            job_id = f"{target}/{policy}/{level}/seed_{seed}"
            command = commands.get(job_id)
            if command is None or "--in_dir" not in command or expected_validation not in command:
                raise ValueError(f"Historical run is not proven to use canonical split: {job_id}")
            input_dir = Path(command[command.index("--in_dir") + 1])
            input_train = input_dir / "rt_train_data.tsv"
            if not input_train.is_file() or input_train.resolve() != expected_train:
                raise ValueError(f"Historical run does not resolve to canonical train input: {job_id}")
            run = study / "runs" / target / policy / level / f"seed_{seed}"
            if not (run / "rt_model.pt").is_file() or not (run / "validation_metrics.json").is_file():
                raise FileNotFoundError(f"Historical Carafe2 model or metrics missing: {run}")
            members.append({"seed": seed, "run_dir": str(run)})
        records.append({"label": label, "scope": "historical_combined", "scope_kind": "historical_combined", "selection_role": role, "strategy": f"{target}/{policy}/{level}", "target": target, "seeds": members})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"purpose": "Previously validation-selected combined Carafe2 comparators", "dataset_dir": str(dataset), "dataset_manifest_sha256": hashlib.sha256((dataset / "dataset_manifest.json").read_bytes()).hexdigest(), "selection_partition": "validation", "candidates": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Registered {len(records)} historical Carafe2 comparators: {output}")


if __name__ == "__main__":
    main()
