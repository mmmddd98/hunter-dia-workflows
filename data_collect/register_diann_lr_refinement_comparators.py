#!/usr/bin/env python3
"""Register two validation-selected historical DIA-NN controls for a fair final comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "trainingdata" / "finalrt_4source_trainfit15_validation10_v1"
DEFAULT_STUDY = Path("./shared_data/diann_fintune_out/diann_trainfit_lr_refinement_v1")
SEEDS = (2024, 2025, 2026)
CANDIDATES = (
    ("Historical combined overall · Chronologer HI/all/10×", "chronologer_hi", "all", "very_high"),
    ("Historical combined heavy guardrail · Anchor iRT/all/20×", "anchor_irt", "all", "twenty_x"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


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
    command_by_id = {job["id"]: job["commands"][0] for job in manifest["jobs"]}
    records = []
    for label, target, policy, lr_level in CANDIDATES:
        members = []
        expected_train = str(dataset / target / "train/diann_rt_input/rt_train_data.tsv")
        expected_validation = str(dataset / target / "validation/diann_rt_input/rt_validation_data.tsv")
        for seed in SEEDS:
            job_id = f"{target}/{policy}/{lr_level}/seed_{seed}"
            command = command_by_id.get(job_id)
            if command is None:
                raise ValueError(f"Historical sweep is missing expected job {job_id}")
            if expected_train not in command or expected_validation not in command:
                raise ValueError(f"{job_id} did not train on the canonical split required for comparison")
            run = study / "runs" / target / policy / lr_level / f"seed_{seed}"
            if not (run / "rt.d0.finetuned.pt").is_file() or not (run / "dict.txt").is_file():
                raise FileNotFoundError(f"Historical model is incomplete: {run}")
            members.append({"seed": seed, "run_dir": str(run)})
        records.append({
            "label": label, "scope": "historical_combined", "scope_kind": "historical_combined",
            "selection_role": "overall" if target == "chronologer_hi" else "heavy_guardrail",
            "strategy": f"{target}/{policy}/{lr_level}", "target": target, "seeds": members,
        })
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "purpose": "Previously validation-selected combined DIA-NN comparators",
        "dataset_dir": str(dataset), "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "selection_partition": "validation", "candidates": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Registered {len(records)} historical comparators: {output}")


if __name__ == "__main__":
    main()
