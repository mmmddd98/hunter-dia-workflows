#!/usr/bin/env python3
"""Create leak-free, nested RT training sets for the cross-organism study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd

from rt_alignment import apply_rt_target, fit_target_alignment, fit_target_inverse, TARGET_MODES

SOURCES = {
    "celegans": "celegans_exploris_44min_rtms2_v1",
    "triplemix": "triplemix_source_a_entrap_neu_filtered_v1",
    "human": "human_source_hunter_human_rtms2_v1",
}
BASE = Path(
    os.environ.get("CARAFE2_TRAINING_DATA_ROOT", Path(__file__).resolve().parents[2] / "trainingdata")
).expanduser()
RT_REFERENCE_MINUTES = 60.101620738856


def read_source(name: str, source_base: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_base / SOURCES[name] / "diann_rt_input" / "rt_train_data.tsv", sep="\t", dtype={"mods": str, "mod_sites": str})
    frame["mods"] = frame["mods"].fillna(""); frame["mod_sites"] = frame["mod_sites"].fillna("")
    frame["source_organism"] = name
    frame["rt"] = pd.to_numeric(frame["rt"], errors="raise")
    frame["source_rt_norm"] = pd.to_numeric(frame["rt_norm"], errors="raise")
    return frame


def nested(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    sequences = np.array(sorted(frame.sequence.unique()))
    rng = np.random.default_rng(seed)
    chosen = set(rng.permutation(sequences)[:round(len(sequences) * fraction)])
    return frame[frame.sequence.isin(chosen)].copy()


def write(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, sep="\t", index=False)
    return {"path": str(path), "rows": len(frame), "sequences": int(frame.sequence.nunique()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--source-base", type=Path, default=BASE)
    parser.add_argument("--rt-target", choices=TARGET_MODES, default="global_60min")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1: raise SystemExit("--test-fraction must be in (0, 1)")
    sources = {name: read_source(name, args.source_base) for name in SOURCES}
    conditions = [(f"ce_{int(p*100):03d}", {"celegans": p}) for p in (.1,.25,.5,.75,1.)]
    conditions += [(f"ce100_triple_{int(p*100):03d}", {"celegans":1.,"triplemix":p}) for p in (.25,.5,1.)]
    conditions += [(f"ce100_human_{int(p*100):03d}", {"celegans":1.,"human":p}) for p in (.25,.5,1.)]
    conditions += [(f"ce100_all_{int(p*100):03d}", {"celegans":1.,"triplemix":p,"human":p}) for p in (.25,.5,1.)]
    manifest = {
        "rt_reference_minutes": RT_REFERENCE_MINUTES,
        "rt_target": args.rt_target,
        "test_fraction": args.test_fraction,
        "seeds": [],
        "conditions": [{"id": name, "fractions": fractions} for name, fractions in conditions],
    }
    for seed in [int(x) for x in args.seeds.split(",")]:
        tests, forbidden = {}, set()
        for name, frame in sources.items():
            test = nested(frame, args.test_fraction, seed)
            tests[name] = test; forbidden.update(test.sequence)
        pools = {name: frame[~frame.sequence.isin(forbidden)].copy() for name, frame in sources.items()}
        # This fixed, non-test calibration pool is shared by every scaling
        # condition.  It keeps the target coordinate fixed while the amount of
        # data supplied to the model changes.
        alignment = fit_target_alignment(pools, mode=args.rt_target)
        inverse_coefficients = fit_target_inverse(
            pools, mode=args.rt_target, reference_minutes=RT_REFERENCE_MINUTES, alignment=alignment
        )
        transformed_tests, target_metadata = apply_rt_target(
            tests, mode=args.rt_target, reference_minutes=RT_REFERENCE_MINUTES,
            alignment=alignment, inverse_coefficients=inverse_coefficients,
        )
        transformed_pools, _ = apply_rt_target(
            pools, mode=args.rt_target, reference_minutes=RT_REFERENCE_MINUTES,
            alignment=alignment, inverse_coefficients=inverse_coefficients,
        )
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "rt_alignment.json").write_text(
            json.dumps(target_metadata, indent=2) + "\n", encoding="utf-8"
        )
        for organism, test in transformed_tests.items():
            write(test, args.output_dir / f"seed_{seed}" / "tests" / f"{organism}.tsv")
        seed_record = {"seed":seed,"test_sequences":{k:int(v.sequence.nunique()) for k,v in tests.items()},"rt_target_metadata":target_metadata,"runs":[]}
        for name, fractions in conditions:
            train = pd.concat([nested(transformed_pools[k], p, seed) for k,p in fractions.items()], ignore_index=True)
            overlap = set(train.sequence) & forbidden
            if overlap:
                raise RuntimeError(f"{seed}/{name}: {len(overlap)} test sequences leaked into training")
            record = write(train, args.output_dir / f"seed_{seed}" / "trains" / f"{name}.tsv")
            record.update({"condition":name,"fractions":fractions}); seed_record["runs"].append(record)
        manifest["seeds"].append(seed_record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Prepared organism scaling study: {args.output_dir}")

if __name__ == "__main__": main()
