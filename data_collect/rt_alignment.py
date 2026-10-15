"""Retention-time target transformations for multi-source fine-tuning data.

All public functions operate on data frames that retain the usual Carafe/DIA-NN
columns.  The model-facing target remains ``rt_norm``; calibration details are
returned separately for the prepared-data manifest.
"""
from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from rt_alignment_methods import (
    ALIGNMENT_METHODS,
    CalibRTAlignment,
    ChronologerHIAlignment,
    IsotonicAlignment,
    KDEAlignment,
    LinearAlignment,
    LowessAlignment,
    MonotonicSplineAlignment,
    RelativeRTAlignment,
    RetentionTimeAlignment,
    alignment_state,
    create_alignment,
    transform_alignment_state,
)


ALIGNMENT_TARGET_METHODS = {
    "linear": "linear",
    "relative_rt": "relative_rt",
    "lowess": "lowess",
    "isotonic": "isotonic",
    "monotonic_spline": "monotonic_spline",
    "calib_rt": "calib_rt",
    "kde": "kde",
    "chronologer_hi": "chronologer_hi",
}
LEGACY_TARGET_MODES = ("global_60min", "source_native", "anchor_irt")
TARGET_MODES = (*LEGACY_TARGET_MODES, *ALIGNMENT_TARGET_METHODS)
ANCHOR_COLUMNS = ("sequence", "mods", "mod_sites")
DEFAULT_CHRONOLOGER_ROOT = Path(
    os.environ.get(
        "CHRONOLOGER_ROOT",
        Path(__file__).resolve().parents[3] / "chronologer",
    )
).expanduser()
_CHRONOLOGER_MODELS: dict[str, object] = {}
_CHRONOLOGER_HI_CACHE: dict[tuple[str, str], float] = {}


def validate_target_mode(mode: str) -> str:
    if mode not in TARGET_MODES:
        raise ValueError(f"Unknown RT target mode {mode!r}; choose one of {TARGET_MODES}")
    return mode


def _aggregate_anchors(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in (*ANCHOR_COLUMNS, "rt") if column not in frame]
    if missing:
        raise ValueError(f"Missing columns for RT alignment: {missing}")
    values = frame.loc[:, [*ANCHOR_COLUMNS, "rt"]].copy()
    values["rt"] = pd.to_numeric(values["rt"], errors="raise")
    return values.groupby(list(ANCHOR_COLUMNS), as_index=False, dropna=False)["rt"].median()


def target_mode_requires_alignment(mode: str) -> bool:
    validate_target_mode(mode)
    return mode == "anchor_irt" or mode in ALIGNMENT_TARGET_METHODS


def _identity_alignment_state(reference: pd.DataFrame) -> dict:
    source = pd.to_numeric(reference["rt"], errors="raise").to_numpy(float)
    lower, upper = float(source.min()), float(source.max())
    return {
        "method": "identity",
        "kind": "affine",
        "slope": 1.0,
        "intercept": 0.0,
        "metadata": {
            "class": "IdentityAlignment",
            "method": "identity",
            "guarantees_monotonicity": True,
            "n_anchors": int(len(source)),
            "source_range": [lower, upper],
            "reference_range": [lower, upper],
        },
    }


def _alignment_options(method: str, anchor_count: int) -> dict:
    if method == "linear":
        return {"robust": True, "f_scale": 0.5}
    if method == "lowess":
        return {
            "frac": 0.25,
            "iterations": 3,
            "monotonic_postprocess": False,
            "extrapolation": "clip",
        }
    if method == "isotonic":
        return {"extrapolation": "clip"}
    if method == "monotonic_spline":
        return {"n_bins": min(50, anchor_count), "extrapolation": "clip"}
    if method == "calib_rt":
        bins = min(100, max(20, int(round(np.sqrt(anchor_count) * 4))))
        return {"bins": bins, "tol_bins": 10, "manual_frac": -1}
    if method in {"kde", "chronologer_hi"}:
        return {"grid_size": 512, "bandwidth_scale": 1.0, "diagonal_bias": 1.0}
    return {}


def _relative_gradient_end(frame: pd.DataFrame) -> float:
    if "source_rt_norm" not in frame:
        raise ValueError("relative_rt requires source_rt_norm to recover the source gradient scale")
    rt = pd.to_numeric(frame["rt"], errors="raise").to_numpy(float)
    relative = pd.to_numeric(frame["source_rt_norm"], errors="raise").to_numpy(float)
    valid = np.isfinite(rt) & np.isfinite(relative) & (np.abs(relative) > 1e-8)
    if not valid.any():
        raise ValueError("relative_rt could not infer a gradient end from source_rt_norm")
    estimates = rt[valid] / relative[valid]
    gradient_end = float(np.median(estimates))
    if not np.isfinite(gradient_end) or gradient_end <= 0:
        raise ValueError("relative_rt inferred an invalid source gradient end")
    return gradient_end


def fit_reference_alignment(
    train_frames: Mapping[str, pd.DataFrame], *, method: str,
    reference_source: str = "human", min_anchors: int = 12,
) -> dict:
    """Fit independent source/reference maps using training-only anchors."""
    if method not in ALIGNMENT_METHODS or method in {"relative_rt", "chronologer_hi"}:
        raise ValueError(f"{method!r} is not an anchor-based reference alignment method")
    if reference_source not in train_frames:
        raise ValueError(f"Reference source {reference_source!r} is absent")
    aggregates = {name: _aggregate_anchors(frame) for name, frame in train_frames.items()}
    reference = aggregates[reference_source]
    forward_states = {reference_source: _identity_alignment_state(reference)}
    inverse_states = {reference_source: _identity_alignment_state(reference)}
    source_summaries = [{
        "source": reference_source,
        "reference": reference_source,
        "anchors": int(len(reference)),
        "fit_residual_mae_minutes": 0.0,
        "fit_residual_rmse_minutes": 0.0,
    }]

    for source, source_anchors in aggregates.items():
        if source == reference_source:
            continue
        overlap = source_anchors.merge(
            reference,
            on=list(ANCHOR_COLUMNS),
            how="inner",
            suffixes=("_source", "_reference"),
        )
        if len(overlap) < min_anchors:
            raise ValueError(
                f"{source!r} has only {len(overlap)} shared anchors with "
                f"reference {reference_source!r}; at least {min_anchors} are required"
            )
        source_rt = overlap["rt_source"].to_numpy(float)
        reference_rt = overlap["rt_reference"].to_numpy(float)
        options = _alignment_options(method, len(overlap))
        forward = create_alignment(method, **options).fit(source_rt, reference_rt)
        inverse = create_alignment(method, **options).fit(reference_rt, source_rt)
        forward_states[source] = alignment_state(forward)
        inverse_states[source] = alignment_state(inverse)
        residuals = transform_alignment_state(forward_states[source], source_rt) - reference_rt
        source_summaries.append({
            "source": source,
            "reference": reference_source,
            "anchors": int(len(overlap)),
            "fit_residual_mae_minutes": float(np.mean(np.abs(residuals))),
            "fit_residual_rmse_minutes": float(np.sqrt(np.mean(residuals ** 2))),
            "options": options,
        })
    return {
        "mode": method,
        "reference_source": reference_source,
        "method": method,
        "fit_scope": "training_peptidoforms_only",
        "anchor_key": list(ANCHOR_COLUMNS),
        "minimum_direct_reference_anchors": min_anchors,
        "source_fits": source_summaries,
        "models": forward_states,
        "inverse_models": inverse_states,
    }


def _predict_chronologer_hi(
    sequences: list[str], *, chronologer_root: Path = DEFAULT_CHRONOLOGER_ROOT,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict external HI values with the local official Chronologer model."""
    source_root = chronologer_root / "src"
    model_path = chronologer_root / "models" / "Chronologer_20220601193755.pt"
    if not source_root.is_dir() or not model_path.is_file():
        raise FileNotFoundError(
            f"Chronologer source/model not found below {chronologer_root}; "
            "set CHRONOLOGER_ROOT to the official checkout"
        )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        import torch
        from chronologer.chronologer_utils.tensorize import modseqs_to_tensor
        from chronologer.model import initialize_chronologer_model
    except ImportError as error:
        raise ImportError(
            "ChronologerHIAlignment target fitting requires the official local "
            "Chronologer package and PyTorch"
        ) from error

    model_key = str(model_path.resolve())
    missing = [sequence for sequence in sequences if (model_key, sequence) not in _CHRONOLOGER_HI_CACHE]
    if missing:
        model = _CHRONOLOGER_MODELS.get(model_key)
        if model is None:
            model = initialize_chronologer_model(model_key).to("cpu")
            model.eval()
            _CHRONOLOGER_MODELS[model_key] = model
        with torch.inference_mode():
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start + batch_size]
                predictions = model(modseqs_to_tensor(batch)).reshape(-1).cpu().numpy()
                for sequence, prediction in zip(batch, predictions):
                    _CHRONOLOGER_HI_CACHE[(model_key, sequence)] = float(prediction)
    return np.asarray([_CHRONOLOGER_HI_CACHE[(model_key, item)] for item in sequences])


def fit_chronologer_hi_alignment(
    train_frames: Mapping[str, pd.DataFrame], *, min_anchors: int = 100,
    chronologer_root: Path = DEFAULT_CHRONOLOGER_ROOT,
) -> dict:
    """Fit each source RT independently to one external Chronologer HI scale."""
    forward_states, inverse_states, source_summaries = {}, {}, []
    all_hi: list[np.ndarray] = []
    for source, frame in train_frames.items():
        if "mods" not in frame:
            raise ValueError("chronologer_hi requires the mods column for anchor filtering")
        unmodified = frame["mods"].fillna("").astype(str).eq("")
        sequence = frame["sequence"].astype(str)
        compatible = (
            unmodified
            & sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", na=False)
            & sequence.str.len().between(6, 50)
        )
        anchors = frame.loc[compatible, ["sequence", "rt"]].copy()
        anchors["rt"] = pd.to_numeric(anchors["rt"], errors="raise")
        anchors = anchors.groupby("sequence", as_index=False)["rt"].median()
        if len(anchors) < min_anchors:
            raise ValueError(
                f"{source!r} has only {len(anchors)} unmodified Chronologer-compatible "
                f"training anchors; at least {min_anchors} are required"
            )
        sequences = anchors["sequence"].tolist()
        source_rt = anchors["rt"].to_numpy(float)
        reference_hi = _predict_chronologer_hi(
            sequences, chronologer_root=chronologer_root
        )
        options = _alignment_options("chronologer_hi", len(anchors))
        forward = ChronologerHIAlignment(**options).fit(source_rt, reference_hi)
        inverse = ChronologerHIAlignment(**options).fit(reference_hi, source_rt)
        forward_states[source] = alignment_state(forward)
        inverse_states[source] = alignment_state(inverse)
        all_hi.append(reference_hi)
        residuals = transform_alignment_state(forward_states[source], source_rt) - reference_hi
        source_summaries.append({
            "source": source,
            "anchors": int(len(anchors)),
            "fit_residual_mae_hi": float(np.mean(np.abs(residuals))),
            "fit_residual_rmse_hi": float(np.sqrt(np.mean(residuals ** 2))),
            "options": options,
        })
    hi_values = np.concatenate(all_hi)
    target_offset, target_upper = float(hi_values.min()), float(hi_values.max())
    target_scale = target_upper - target_offset
    if target_scale <= 0:
        raise ValueError("Chronologer anchors do not span a usable HI range")
    return {
        "mode": "chronologer_hi",
        "method": "chronologer_hi",
        "reference_source": "external_chronologer_model",
        "fit_scope": "training_unmodified_peptides_only",
        "anchor_filter": "unmodified, canonical amino acids, length 6-50",
        "chronologer_root": str(chronologer_root),
        "target_offset": target_offset,
        "target_scale": target_scale,
        "source_fits": source_summaries,
        "models": forward_states,
        "inverse_models": inverse_states,
    }


def fit_relative_alignment(train_frames: Mapping[str, pd.DataFrame]) -> dict:
    """Record explicit per-source relative-gradient transformations."""
    forward_states, inverse_states, source_summaries = {}, {}, []
    for source, frame in train_frames.items():
        gradient_end = _relative_gradient_end(frame)
        rt = pd.to_numeric(frame["rt"], errors="raise").to_numpy(float)
        reference = pd.to_numeric(frame["source_rt_norm"], errors="raise").to_numpy(float)
        forward = RelativeRTAlignment(gradient_end=gradient_end, t0=0.0).fit(rt, reference)
        forward_states[source] = alignment_state(forward)
        inverse_states[source] = {
            "method": "relative_rt_inverse",
            "kind": "affine",
            "slope": gradient_end,
            "intercept": 0.0,
            "metadata": {"gradient_end": gradient_end, "t0": 0.0},
        }
        residuals = forward.transform(rt) - reference
        source_summaries.append({
            "source": source,
            "rows": int(len(frame)),
            "gradient_end": gradient_end,
            "fit_residual_mae_relative": float(np.mean(np.abs(residuals))),
        })
    return {
        "mode": "relative_rt",
        "method": "relative_rt",
        "fit_scope": "training_rows_only",
        "models": forward_states,
        "inverse_models": inverse_states,
        "source_fits": source_summaries,
    }


def fit_target_alignment(
    train_frames: Mapping[str, pd.DataFrame], *, mode: str, reference_source: str = "human",
) -> dict | None:
    """Fit the alignment object required by a selectable target mode."""
    validate_target_mode(mode)
    if mode == "anchor_irt":
        return fit_anchor_alignment(train_frames, reference_source=reference_source)
    if mode == "relative_rt":
        return fit_relative_alignment(train_frames)
    if mode == "chronologer_hi":
        return fit_chronologer_hi_alignment(train_frames)
    if mode in ALIGNMENT_TARGET_METHODS:
        return fit_reference_alignment(
            train_frames, method=ALIGNMENT_TARGET_METHODS[mode], reference_source=reference_source
        )
    return None


def fit_anchor_alignment(
    train_frames: Mapping[str, pd.DataFrame], *, reference_source: str = "human",
    min_pair_anchors: int = 12,
) -> dict:
    """Fit a robust common RT coordinate from training-only shared peptidoforms.

    The reference source is fixed to identity.  All other source-specific
    affine maps are estimated jointly from every connected pairwise overlap,
    which uses indirect evidence instead of relying only on sparse direct
    C. elegans--Human overlaps.
    """
    names = tuple(train_frames)
    if reference_source not in names:
        raise ValueError(f"Reference source {reference_source!r} is absent")
    aggregates = {name: _aggregate_anchors(frame) for name, frame in train_frames.items()}
    pairs: list[tuple[str, str, pd.DataFrame]] = []
    pair_summary = []
    for left, right in combinations(names, 2):
        overlap = aggregates[left].merge(
            aggregates[right], on=list(ANCHOR_COLUMNS), how="inner", suffixes=("_left", "_right")
        )
        pair_summary.append({"left": left, "right": right, "anchors": int(len(overlap))})
        if len(overlap) >= min_pair_anchors:
            pairs.append((left, right, overlap))
    if not pairs:
        raise ValueError("No source pair has enough shared training peptidoforms for RT alignment")

    unknown = [name for name in names if name != reference_source]
    index = {name: position for position, name in enumerate(unknown)}

    def transform(source: str, values: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        if source == reference_source:
            return values
        position = index[source] * 2
        return parameters[position] * values + parameters[position + 1]

    def residuals(parameters: np.ndarray) -> np.ndarray:
        pieces = []
        for left, right, overlap in pairs:
            left_rt = overlap["rt_left"].to_numpy(float)
            right_rt = overlap["rt_right"].to_numpy(float)
            pieces.append(transform(left, left_rt, parameters) - transform(right, right_rt, parameters))
        return np.concatenate(pieces)

    initial = np.empty(len(unknown) * 2, dtype=float)
    initial[0::2] = 1.0
    initial[1::2] = 0.0
    result = least_squares(residuals, initial, loss="soft_l1", f_scale=0.5, max_nfev=5000)
    if not result.success:
        raise RuntimeError(f"RT alignment fit failed: {result.message}")

    coefficients = {reference_source: {"slope": 1.0, "intercept": 0.0}}
    for source in unknown:
        position = index[source] * 2
        coefficients[source] = {
            "slope": float(result.x[position]),
            "intercept": float(result.x[position + 1]),
        }
    fit_residuals = residuals(result.x)
    return {
        "mode": "anchor_irt",
        "reference_source": reference_source,
        "method": "joint_affine_soft_l1",
        "fit_scope": "training_peptidoforms_only",
        "anchor_key": list(ANCHOR_COLUMNS),
        "minimum_pair_anchors": min_pair_anchors,
        "pairwise_overlaps": pair_summary,
        "coefficients": coefficients,
        "fit_residual_mae_minutes": float(np.mean(np.abs(fit_residuals))),
        "fit_residual_rmse_minutes": float(np.sqrt(np.mean(fit_residuals ** 2))),
    }


def apply_rt_target(
    frames: Mapping[str, pd.DataFrame], *, mode: str, reference_minutes: float,
    alignment: dict | None = None, inverse_coefficients: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return copies with the requested model target in ``rt_norm``."""
    validate_target_mode(mode)
    if reference_minutes <= 0:
        raise ValueError("reference_minutes must be positive")
    if target_mode_requires_alignment(mode) and alignment is None:
        raise ValueError(f"{mode} requires fitted alignment metadata")

    transformed = {}
    source_ranges = {}
    for source, frame in frames.items():
        output = frame.copy()
        rt = pd.to_numeric(output["rt"], errors="raise")
        if mode == "global_60min":
            target = rt / reference_minutes
        elif mode == "source_native":
            if "source_rt_norm" not in output:
                raise ValueError("source_native requires source_rt_norm before preprocessing")
            target = pd.to_numeric(output["source_rt_norm"], errors="raise")
        elif mode == "anchor_irt":
            coefficients = alignment["coefficients"].get(source)
            if coefficients is None:
                raise ValueError(f"No anchor alignment coefficients for {source!r}")
            target = (rt * coefficients["slope"] + coefficients["intercept"]) / reference_minutes
        else:
            state = alignment["models"].get(source)
            if state is None:
                raise ValueError(f"No {mode} alignment model for {source!r}")
            aligned = transform_alignment_state(state, rt.to_numpy(float))
            if mode == "relative_rt":
                target = aligned
            else:
                target_offset = float(alignment.get("target_offset", 0.0))
                target_scale = float(alignment.get("target_scale", reference_minutes))
                if target_scale <= 0:
                    raise ValueError(f"Invalid target scale for {mode}")
                target = (aligned - target_offset) / target_scale
        target_values = np.asarray(target, dtype=float)
        if not np.isfinite(target_values).all():
            raise ValueError(f"Non-finite RT targets produced for {source}")
        output["rt_norm"] = target
        transformed[source] = output
        source_ranges[source] = {
            "rt_norm_min": float(target_values.min()),
            "rt_norm_max": float(target_values.max()),
        }

    if inverse_coefficients is None:
        inverse_coefficients = fit_target_inverse(
            frames, mode=mode, reference_minutes=reference_minutes, alignment=alignment
        )
    metadata = {
        "mode": mode,
        "reference_minutes": reference_minutes,
        "source_target_ranges": source_ranges,
        # Maps a model prediction in rt_norm back to the observed LC RT in
        # minutes.  It is fit only on the calibration/training partition and
        # makes results comparable across target definitions.
        "target_to_observed_rt": inverse_coefficients,
    }
    if alignment is not None:
        metadata["alignment"] = alignment
    return transformed, metadata


def fit_target_inverse(
    frames: Mapping[str, pd.DataFrame], *, mode: str, reference_minutes: float,
    alignment: dict | None = None,
) -> dict:
    """Fit/record target-to-observed-RT maps on calibration data only."""
    validate_target_mode(mode)
    result = {}
    for source, frame in frames.items():
        if mode == "global_60min":
            slope, intercept = reference_minutes, 0.0
        elif mode == "anchor_irt":
            if alignment is None:
                raise ValueError("anchor_irt requires fitted alignment metadata")
            coefficients = alignment["coefficients"].get(source)
            if coefficients is None or coefficients["slope"] == 0:
                raise ValueError(f"No invertible anchor alignment for {source!r}")
            slope = reference_minutes / float(coefficients["slope"])
            intercept = -float(coefficients["intercept"]) / float(coefficients["slope"])
        elif mode == "source_native":
            if "source_rt_norm" not in frame:
                raise ValueError("source_native requires source_rt_norm before preprocessing")
            x = pd.to_numeric(frame["source_rt_norm"], errors="raise").to_numpy(float)
            y = pd.to_numeric(frame["rt"], errors="raise").to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
        else:
            if alignment is None:
                raise ValueError(f"{mode} requires fitted alignment metadata")
            state = alignment["inverse_models"].get(source)
            if state is None:
                raise ValueError(f"No inverse {mode} alignment for {source!r}")
            result[source] = {
                "kind": "alignment_state",
                "input_scale": (
                    1.0 if mode == "relative_rt"
                    else float(alignment.get("target_scale", reference_minutes))
                ),
                "input_offset": (
                    0.0 if mode == "relative_rt"
                    else float(alignment.get("target_offset", 0.0))
                ),
                "state": state,
            }
            continue
        result[source] = {"slope": float(slope), "intercept": float(intercept)}
    return result


def predictions_to_observed_rt(
    predictions: np.ndarray, source_labels: pd.Series | np.ndarray, alignment_metadata: dict,
) -> np.ndarray:
    """Map normalized model predictions back to source-observed RT minutes."""
    values = np.asarray(predictions, dtype=float).reshape(-1)
    labels = np.asarray(source_labels, dtype=str).reshape(-1)
    if values.shape != labels.shape:
        raise ValueError("predictions and source_labels must have equal length")
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    mappings = alignment_metadata["target_to_observed_rt"]
    result = np.empty_like(values)
    for source in np.unique(labels):
        if source not in mappings:
            raise ValueError(f"No target-to-observed RT mapping for {source!r}")
        selected = labels == source
        mapping = mappings[source]
        if "slope" in mapping and "intercept" in mapping:
            result[selected] = values[selected] * float(mapping["slope"]) + float(
                mapping["intercept"]
            )
        elif mapping.get("kind") == "alignment_state":
            scaled = (
                values[selected] * float(mapping.get("input_scale", 1.0))
                + float(mapping.get("input_offset", 0.0))
            )
            result[selected] = transform_alignment_state(mapping["state"], scaled)
        else:
            raise ValueError(f"Unsupported target-to-observed RT mapping for {source!r}")
    return result
