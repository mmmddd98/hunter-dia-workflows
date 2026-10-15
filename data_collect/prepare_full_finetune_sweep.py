#!/usr/bin/env python3
"""Prepare one leak-free full-data split for RT fine-tuning strategy sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from rt_alignment import apply_rt_target, fit_target_alignment, fit_target_inverse, TARGET_MODES


SOURCES = {
    "celegans": "celegans_exploris_44min_rtms2_v1",
    "triplemix": "triplemix_source_a_entrap_neu_filtered_v1",
    "human": "human_source_hunter_human_rtms2_v1",
}
DEFAULT_SOURCE_BASE = Path(
    os.environ.get("CARAFE2_TRAINING_DATA_ROOT", Path(__file__).resolve().parents[2] / "trainingdata")
).expanduser()
RT_REFERENCE_MINUTES = 60.101620738856
HEAVY_NTERM = "Dimethyl:2H(4)13C(2)@N-term"
HEAVY_LYSINE = "Dimethyl:2H(4)13C(2)@K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-base", type=Path, default=DEFAULT_SOURCE_BASE)
    parser.add_argument("--split-seed", type=int, default=5102026)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--rt-target", choices=TARGET_MODES, default="global_60min",
                        help="RT target coordinate written to rt_norm")
    return parser.parse_args()


def sequence_partition(sequence: str, seed: int, validation_fraction: float,
                       test_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{sequence}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def read_source(source_base: Path, organism: str) -> pd.DataFrame:
    path = source_base / SOURCES[organism] / "diann_rt_input" / "rt_train_data.tsv"
    frame = pd.read_csv(path, sep="\t", dtype={"mods": str, "mod_sites": str})
    frame["mods"] = frame["mods"].fillna("").astype(str)
    frame["mod_sites"] = frame["mod_sites"].fillna("").astype(str)
    frame["source_organism"] = organism
    frame["rt"] = pd.to_numeric(frame["rt"], errors="raise")
    frame["source_rt_norm"] = pd.to_numeric(frame["rt_norm"], errors="raise")
    return frame


def frame_summary(frame: pd.DataFrame, path: Path) -> dict:
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains(HEAVY_NTERM, regex=False)
    lysine = mods.str.contains(HEAVY_LYSINE, regex=False)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "unique_sequences": int(frame["sequence"].nunique()),
        "heavy_dimethyl_rows": int((nterm | lysine).sum()),
        "heavy_nterm_dimethyl_rows": int(nterm.sum()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_frame(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    return frame_summary(frame, path)


def main() -> None:
    args = parse_args()
    if args.validation_fraction <= 0 or args.test_fraction <= 0:
        raise SystemExit("Validation and test fractions must be positive")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise SystemExit("Validation plus test fraction must be below 0.5")

    sources = {name: read_source(args.source_base, name) for name in SOURCES}
    all_sequences = sorted({sequence for frame in sources.values() for sequence in frame["sequence"]})
    assignments = {
        sequence: sequence_partition(
            sequence, args.split_seed, args.validation_fraction, args.test_fraction
        )
        for sequence in all_sequences
    }
    partitions: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in SOURCES}
    manifest = {
        "study": "full_multiorganism_finetune_strategy_sweep_v1",
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "rt_reference_minutes": RT_REFERENCE_MINUTES,
        "rt_target": args.rt_target,
        "sequence_disjoint": True,
        "sources": SOURCES,
        "partitions": {},
    }
    for organism, frame in sources.items():
        labels = frame["sequence"].map(assignments)
        manifest["partitions"][organism] = {}
        for partition in ("train", "validation", "test"):
            subset = frame[labels.eq(partition)].copy()
            partitions[organism][partition] = subset

    train_frames = {organism: partitions[organism]["train"] for organism in SOURCES}
    alignment = fit_target_alignment(train_frames, mode=args.rt_target)
    inverse_coefficients = fit_target_inverse(
        train_frames, mode=args.rt_target, reference_minutes=RT_REFERENCE_MINUTES, alignment=alignment
    )
    target_metadata = None
    for partition in ("train", "validation", "test"):
        source_frames = {organism: partitions[organism][partition] for organism in SOURCES}
        transformed, current_metadata = apply_rt_target(
            source_frames, mode=args.rt_target, reference_minutes=RT_REFERENCE_MINUTES,
            alignment=alignment, inverse_coefficients=inverse_coefficients,
        )
        for organism, subset in transformed.items():
            partitions[organism][partition] = subset
            destination = args.output_dir / partition / f"{organism}.tsv"
            manifest["partitions"][organism][partition] = write_frame(subset, destination)
        if partition == "train":
            target_metadata = current_metadata
    manifest["rt_target_metadata"] = target_metadata

    for partition in ("train", "validation", "test"):
        combined = pd.concat(
            [partitions[organism][partition] for organism in SOURCES], ignore_index=True
        )
        name = "rt_train_data.tsv" if partition == "train" else "combined.tsv"
        destination = args.output_dir / partition / name
        manifest[f"combined_{partition}"] = write_frame(combined, destination)

    train_sequences = set(pd.concat([partitions[item]["train"] for item in SOURCES]).sequence)
    validation_sequences = set(pd.concat([partitions[item]["validation"] for item in SOURCES]).sequence)
    test_sequences = set(pd.concat([partitions[item]["test"] for item in SOURCES]).sequence)
    if train_sequences & validation_sequences or train_sequences & test_sequences or validation_sequences & test_sequences:
        raise RuntimeError("Sequence leakage detected between train, validation, and test")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_payload = json.dumps(target_metadata, indent=2) + "\n"
    (args.output_dir / "rt_alignment.json").write_text(alignment_payload, encoding="utf-8")
    # Keep each evaluation partition self-contained. Evaluators deliberately
    # consume only the split directory and this training-fitted inverse map.
    for partition in ("validation", "test"):
        (args.output_dir / partition / "rt_alignment.json").write_text(
            alignment_payload, encoding="utf-8"
        )
    manifest_path = args.output_dir / "sweep_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared full-data fine-tuning sweep inputs: {args.output_dir}")
    print(json.dumps({key: manifest[key] for key in (
        "combined_train", "combined_validation", "combined_test"
    )}, indent=2))


if __name__ == "__main__":
    main()
