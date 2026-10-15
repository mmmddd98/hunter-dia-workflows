#!/usr/bin/env python3
"""Evaluate one DIA-NN RT run on a fixed validation or final-test partition.

The output contains model-scale metrics and observed-minutes metrics per
source and organism.  With ``--prediction-dir`` it also writes the
peptidoform-level prediction tables needed for final-test residual and
calibration plots.  Source tables are deliberately evaluated separately: the
same peptidoform may occur in several sources with different RTs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIANN = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune"))
sys.path.insert(0, str(DIANN))

from diann_finetune.data import load_carafe_rt
from diann_finetune.encoding import TokenEncoder, read_tokens
from diann_finetune.training import (
    HEAVY_CHECKPOINT_SELECTIONS,
    evaluate,
    heavy_dimethyl_row_strata,
    load_torchscript_model,
    prepare_examples,
    resolve_device,
)
from rt_alignment import predictions_to_observed_rt

KEYS = ["sequence", "mods", "mod_sites"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-selection",
        choices=tuple(HEAVY_CHECKPOINT_SELECTIONS),
        help=(
            "Evaluate one validation-selected checkpoint from --save-heavy-checkpoint-grid. "
            "The default is the standard 100/0 checkpoint in the run root."
        ),
    )
    parser.add_argument(
        "--prediction-dir", type=Path,
        help="Optional directory for aggregate and source-observation prediction tables.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {"rows": int(len(residual)), "mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.square(residual).mean())), "bias": float(residual.mean())}


def attach_digestion_class(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Restore the stratum label dropped by the model loader's aggregation."""
    # ``load_carafe_rt`` canonicalizes missing modification strings to "" before
    # grouping. Mirror that representation in the provenance lookup.
    raw = raw.copy()
    for column in ("mods", "mod_sites"):
        raw[column] = raw[column].fillna("").astype(str)
    lookup = raw.loc[:, [*KEYS, "digestion_class"]].drop_duplicates(KEYS)
    result = frame.merge(lookup, on=KEYS, how="left", validate="one_to_one")
    if result["digestion_class"].isna().any():
        raise ValueError("Could not restore digestion_class for every evaluated peptidoform")
    return result.reset_index(drop=True)


def prediction_table(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    observed_prediction: np.ndarray | None = None,
    *,
    source: str | None = None,
    organism: str | None = None,
) -> pd.DataFrame:
    """Return an audit-ready table without modifying the source partition."""
    columns = [column for column in (*KEYS, "mods", "mod_sites", "rt_norm", "rt", "digestion_class") if column in frame]
    result = frame.loc[:, list(dict.fromkeys(columns))].copy()
    result["heavy_dimethyl_stratum"] = heavy_dimethyl_row_strata(frame).to_numpy()
    result["prediction"] = np.asarray(prediction, dtype=float)
    result["normalized_residual"] = result["prediction"] - pd.to_numeric(result["rt_norm"], errors="raise")
    if observed_prediction is not None:
        result["observed_prediction"] = np.asarray(observed_prediction, dtype=float)
        result["raw_minutes_residual"] = result["observed_prediction"] - pd.to_numeric(result["rt"], errors="raise")
    if source is not None:
        result["source_dataset_id"] = source
    if organism is not None:
        result["source_organism"] = organism
    return result


def stratum_metrics(frame: pd.DataFrame, target: np.ndarray, prediction: np.ndarray, *, prefix: str) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    heavy = heavy_dimethyl_row_strata(frame).to_numpy()
    for label in ("normal", "heavy_nterm", "heavy_lysine", "heavy_both"):
        mask = heavy == label
        result[f"{prefix}heavy_{label}"] = regression_metrics(target[mask], prediction[mask]) if mask.any() else {"rows": 0}
    for label in ("fully_tryptic", "semi_tryptic", "ambiguous"):
        mask = frame["digestion_class"].to_numpy() == label
        result[f"{prefix}digestion_{label}"] = regression_metrics(target[mask], prediction[mask]) if mask.any() else {"rows": 0}
    return result


def record_metrics(frame: pd.DataFrame, prediction: np.ndarray, observed_prediction: np.ndarray | None = None) -> dict:
    prediction = np.asarray(prediction, dtype=float)
    result: dict = {"normalized": regression_metrics(frame.rt_norm.to_numpy(float), prediction)}
    result.update(stratum_metrics(frame, frame.rt_norm.to_numpy(float), prediction, prefix="normalized_"))
    if observed_prediction is not None:
        observed_prediction = np.asarray(observed_prediction, dtype=float)
        result["raw_minutes"] = regression_metrics(frame.rt.to_numpy(float), observed_prediction)
        result.update(stratum_metrics(frame, frame.rt.to_numpy(float), observed_prediction, prefix="raw_"))
    return result


def predict(frame: pd.DataFrame, encoder: TokenEncoder, model, batch_size: int, device) -> np.ndarray:
    examples, targets = prepare_examples(frame, encoder, 2)
    _, prediction = evaluate(model, examples, targets, batch_size, device)
    return np.asarray(prediction, dtype=float)


def aggregate_path(partition_dir: Path) -> Path:
    for name in ("rt_validation_data.tsv", "rt_test_data.tsv"):
        path = partition_dir / "diann_rt_input" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No aggregate validation/test table in {partition_dir / 'diann_rt_input'}")


def evaluate_run(
    *, run_dir: Path, partition_dir: Path, alignment_path: Path,
    device_name: str = "auto", batch_size: int = 512,
    checkpoint_selection: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate one model and return metrics, aggregate, and source predictions."""
    device = resolve_device(device_name)
    if checkpoint_selection is None or checkpoint_selection == "100_0":
        checkpoint_path = run_dir / "rt.d0.finetuned.pt"
    else:
        checkpoint_path = run_dir / "checkpoints" / f"selection_{checkpoint_selection}" / "rt.d0.finetuned.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint selection {checkpoint_selection or '100_0'} is unavailable: {checkpoint_path}"
        )
    model = load_torchscript_model(checkpoint_path, device)
    encoder = TokenEncoder(read_tokens(run_dir / "dict.txt"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    aggregate_file = aggregate_path(partition_dir)
    aggregate_raw = pd.read_csv(aggregate_file, sep="\t", dtype={"mods": str, "mod_sites": str})
    aggregate = attach_digestion_class(load_carafe_rt(aggregate_file).frame, aggregate_raw)
    aggregate_prediction = predict(aggregate, encoder, model, batch_size, device)
    aggregate_predictions = prediction_table(aggregate, aggregate_prediction)

    source_records: dict[str, dict] = {}
    organism_frames: dict[str, list[pd.DataFrame]] = {}
    all_source_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for path in sorted((partition_dir / "sources").glob("*.tsv")):
        raw = pd.read_csv(path, sep="\t", dtype={"mods": str, "mod_sites": str})
        source = str(raw["source_dataset_id"].iloc[0])
        organism = str(raw["source_organism"].iloc[0])
        frame = attach_digestion_class(load_carafe_rt(path).frame, raw)
        source_prediction = predict(frame, encoder, model, batch_size, device)
        observed_prediction = predictions_to_observed_rt(source_prediction, np.repeat(source, len(source_prediction)), alignment)
        source_records[source] = record_metrics(frame, source_prediction, observed_prediction)
        metrics_frame = frame.loc[:, ["rt_norm", "rt", "mods", "digestion_class"]].copy()
        metrics_frame["prediction"] = source_prediction
        metrics_frame["observed_prediction"] = observed_prediction
        organism_frames.setdefault(organism, []).append(metrics_frame)
        all_source_frames.append(metrics_frame)
        prediction_frames.append(prediction_table(
            frame, source_prediction, observed_prediction, source=source, organism=organism,
        ))

    organisms: dict[str, dict] = {}
    for organism, pieces in organism_frames.items():
        frame = pd.concat(pieces, ignore_index=True)
        organisms[organism] = record_metrics(frame, frame.prediction.to_numpy(), frame.observed_prediction.to_numpy())
    all_observations = pd.concat(all_source_frames, ignore_index=True)
    result = {
        "checkpoint_selection": checkpoint_selection or "100_0",
        "checkpoint_path": str(checkpoint_path),
        "aggregate": record_metrics(aggregate, aggregate_prediction),
        "sources": source_records,
        "organisms": organisms,
        "all_source_observations": record_metrics(all_observations, all_observations.prediction.to_numpy(), all_observations.observed_prediction.to_numpy()),
        "source_balanced_raw_mae": float(np.mean([value["raw_minutes"]["mae"] for value in source_records.values()])),
    }
    return result, aggregate_predictions, pd.concat(prediction_frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    result, aggregate_predictions, source_predictions = evaluate_run(
        run_dir=args.run_dir, partition_dir=args.partition_dir, alignment_path=args.alignment,
        device_name=args.device, batch_size=args.batch_size,
        checkpoint_selection=args.checkpoint_selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.prediction_dir:
        args.prediction_dir.mkdir(parents=True, exist_ok=True)
        aggregate_predictions.to_csv(args.prediction_dir / "aggregate_predictions.tsv", sep="\t", index=False)
        source_predictions.to_csv(args.prediction_dir / "source_observation_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
