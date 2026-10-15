#!/usr/bin/env python3
"""Evaluate one Carafe2 RT run on a fixed validation or final-test partition.

Normalized predictions are mapped back to each source's observed RT minutes
using calibration parameters that were fitted on the training partition.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CARAFE2 = Path(os.environ.get("CARAFE2_ROOT", ROOT / "carafe2-semi"))
PREDICTOR = CARAFE2 / "src/main/resources/py/v2/ai_pred.py"

from rt_alignment import predictions_to_observed_rt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        help="Optional output directory for prediction tables; defaults to the model run directory.",
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape:
        raise ValueError(f"Target/prediction shape mismatch: {target.shape} != {prediction.shape}")
    residual = prediction - target
    return {
        "rows": int(len(residual)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "bias": float(residual.mean()),
    }


def heavy_strata(frame: pd.DataFrame) -> np.ndarray:
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains("Dimethyl:2H(4)13C(2)@N-term", regex=False).to_numpy()
    lysine = mods.str.contains("Dimethyl:2H(4)13C(2)@K", regex=False).to_numpy()
    labels = np.full(len(frame), "normal", dtype=object)
    labels[nterm & ~lysine] = "heavy_nterm"
    labels[~nterm & lysine] = "heavy_lysine"
    labels[nterm & lysine] = "heavy_both"
    return labels


def stratum_metrics(
    frame: pd.DataFrame, target: np.ndarray, prediction: np.ndarray, *, prefix: str,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    heavy = heavy_strata(frame)
    for label in ("normal", "heavy_nterm", "heavy_lysine", "heavy_both"):
        selected = heavy == label
        result[f"{prefix}heavy_{label}"] = (
            regression_metrics(target[selected], prediction[selected]) if selected.any() else {"rows": 0}
        )
    digestion = frame["digestion_class"].fillna("ambiguous").astype(str).to_numpy()
    for label in ("fully_tryptic", "semi_tryptic", "ambiguous"):
        selected = digestion == label
        result[f"{prefix}digestion_{label}"] = (
            regression_metrics(target[selected], prediction[selected]) if selected.any() else {"rows": 0}
        )
    return result


def record_metrics(
    frame: pd.DataFrame, prediction: np.ndarray, observed_prediction: np.ndarray | None = None,
) -> dict:
    prediction = np.asarray(prediction, dtype=float)
    normalized_target = pd.to_numeric(frame["rt_norm"], errors="raise").to_numpy(float)
    result: dict = {"normalized": regression_metrics(normalized_target, prediction)}
    result.update(stratum_metrics(frame, normalized_target, prediction, prefix="normalized_"))
    if observed_prediction is not None:
        raw_target = pd.to_numeric(frame["rt"], errors="raise").to_numpy(float)
        observed_prediction = np.asarray(observed_prediction, dtype=float)
        result["raw_minutes"] = regression_metrics(raw_target, observed_prediction)
        result.update(stratum_metrics(frame, raw_target, observed_prediction, prefix="raw_"))
    return result


def predict(
    *, python: str, run_dir: Path, input_file: Path, prediction_dir: Path,
    prefix: str, device: str,
) -> pd.DataFrame:
    if not PREDICTOR.is_file():
        raise FileNotFoundError(PREDICTOR)
    subprocess.run([
        python, str(PREDICTOR), "--model_dir", str(run_dir),
        "--in_file", str(input_file), "--out_dir", str(prediction_dir),
        "--out_prefix", prefix, "--device", device, "--tf_type", "rt",
        "--verbose", "0",
    ], check=True)
    output = prediction_dir / f"{prefix}_rt_pred.tsv"
    frame = pd.read_csv(output, sep="\t", dtype={"mods": str, "mod_sites": str})
    required = {"rt", "rt_norm", "rt_pred", "mods", "digestion_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction output {output} lacks columns: {sorted(missing)}")
    return frame


def main() -> None:
    args = parse_args()
    if not (args.run_dir / "rt_model.pt").is_file():
        raise SystemExit(f"Incomplete Carafe2 run: {args.run_dir / 'rt_model.pt'} missing")
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    aggregate_file = next(
        (
            args.partition_dir / "diann_rt_input" / name
            for name in ("rt_validation_data.tsv", "rt_test_data.tsv")
            if (args.partition_dir / "diann_rt_input" / name).is_file()
        ),
        None,
    )
    observations_file = args.partition_dir / "rt_observations.tsv"
    for required in (aggregate_file, observations_file):
        if required is None:
            raise FileNotFoundError(
                f"No aggregate validation/test table in {args.partition_dir / 'diann_rt_input'}"
            )
        if not required.is_file():
            raise FileNotFoundError(required)

    prediction_dir = args.prediction_dir or (args.run_dir / "validation_predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    aggregate = predict(
        python=args.python, run_dir=args.run_dir, input_file=aggregate_file,
        prediction_dir=prediction_dir, prefix="aggregate", device=args.device,
    )
    observations = predict(
        python=args.python, run_dir=args.run_dir, input_file=observations_file,
        prediction_dir=prediction_dir, prefix="source_observations", device=args.device,
    )
    observation_required = {"source_dataset_id", "source_organism"}
    missing = observation_required - set(observations.columns)
    if missing:
        raise ValueError(f"Source-observation predictions lack columns: {sorted(missing)}")

    prediction = pd.to_numeric(observations["rt_pred"], errors="raise").to_numpy(float)
    observed_prediction = predictions_to_observed_rt(
        prediction, observations["source_dataset_id"].astype(str), alignment,
    )
    observations = observations.copy()
    observations["observed_prediction"] = observed_prediction

    sources = {}
    for source, frame in observations.groupby("source_dataset_id", sort=True):
        sources[str(source)] = record_metrics(
            frame, frame["rt_pred"].to_numpy(float), frame["observed_prediction"].to_numpy(float),
        )
    organisms = {}
    for organism, frame in observations.groupby("source_organism", sort=True):
        organisms[str(organism)] = record_metrics(
            frame, frame["rt_pred"].to_numpy(float), frame["observed_prediction"].to_numpy(float),
        )
    result = {
        "aggregate": record_metrics(aggregate, aggregate["rt_pred"].to_numpy(float)),
        "sources": sources,
        "organisms": organisms,
        "all_source_observations": record_metrics(
            observations, prediction, observed_prediction,
        ),
        "source_balanced_raw_mae": float(np.mean([
            record["raw_minutes"]["mae"] for record in sources.values()
        ])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "aggregate_rows": len(aggregate),
        "source_observations": len(observations), "sources": len(sources),
    }, indent=2))


if __name__ == "__main__":
    main()
