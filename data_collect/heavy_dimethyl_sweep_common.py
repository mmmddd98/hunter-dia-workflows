"""Shared, pre-registered constants and small utilities for the Heavy-Dimethyl RT study.

This module deliberately contains no trainer implementation.  The study uses
the established DIA-NN/Carafe2 wrappers and the established manifest runner.
Keeping the design constants here prevents preparation and analysis drifting.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

MODES = ("chronologer_hi", "anchor_irt")
SEEDS = (2024, 2025, 2026)
EPOCHS, WARMUP_EPOCHS, BATCH_SIZE = 40, 10, 512
WEIGHT_PROFILES = {
    "W0": (1.0, 1.0, 1.0), "W1": (2.0, 1.0, 1.0), "W2": (4.0, 1.0, 1.0),
    "W3": (1.0, 1.5, 1.0), "W4": (1.0, 2.0, 1.0), "W5": (1.0, 1.0, 2.0),
    "W6": (1.0, 1.0, 4.0), "W7": (1.0, 1.0, 6.0), "W8": (3.0, 2.0, 3.0),
    "W9": (2.0, 2.0, 4.0), "W10": (6.0, 2.0, 1.0), "W11": (6.0, 2.0, 2.0),
    # Dedicated heavy-both dose escalation; W8--W11 already encode mixed or
    # other-stratum interventions from the original factorial sweep.
    "W12": (1.0, 1.0, 8.0), "W13": (1.0, 1.0, 10.0),
    # Balanced companion series: N-term and lysine each receive half of the
    # heavy-both factor, bounded below by their unmodified 1x baseline.
    "W14": (2.0, 2.0, 4.0), "W15": (4.0, 4.0, 8.0), "W16": (5.0, 5.0, 10.0),
}
OVERSAMPLE_PROFILES = {
    "O0": (1.0, 1.0, 1.0), "O1": (2.0, 1.0, 1.0), "O2": (1.0, 1.5, 1.0),
    "O3": (1.0, 1.0, 2.0), "O4": (1.0, 1.0, 4.0), "O5": (2.0, 1.5, 3.0),
    "O6": (1.0, 1.0, 6.0), "O7": (1.0, 1.0, 8.0), "O8": (1.0, 1.0, 10.0),
    "O9": (2.0, 2.0, 4.0), "O10": (4.0, 4.0, 8.0), "O11": (5.0, 5.0, 10.0),
}
EMBEDDING_PROFILES = {key: WEIGHT_PROFILES[key] for key in ("W0", "W8", "W9", "W10")}
LEARNING_RATES = {
    "diann": (2e-4, 3e-4, 4.5e-4, 6e-4),
    "carafe2": (3e-4, 4.5e-4, 6e-4, 9e-4),
}
# The factorial DIA-NN sweep already covered 2e-4--6e-4.  This rounded
# follow-up reaches well below that bracket and deliberately retains two
# anchors from it (2e-4, 3e-4), making the new dose-response results directly
# comparable with the original study rather than an isolated low-LR search.
LOW_LR_LEARNING_RATES = (2.5e-5, 5e-5, 7.5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4)
# Complete, matched dose response for the doubly dimethylated stratum.  The
# controls remain separate by intervention arm because they receive separate
# matched-control comparisons downstream.
LOW_LR_DOSE_WEIGHT_PROFILES = ("W0", "W5", "W6", "W7", "W12", "W13")
LOW_LR_DOSE_OVERSAMPLE_PROFILES = ("O0", "O3", "O4", "O6", "O7", "O8")
# Compact final design: logarithmic LR coverage, two anchors from the prior
# DIA-NN grid, and no redundant 1x training in the two intervention arms.
COMPACT_LOW_LR_LEARNING_RATES = (2.5e-5, 5e-5, 1e-4, 2e-4, 3e-4)
COMPACT_DOSE_WEIGHT_PROFILES = ("W5", "W6", "W12", "W13")
COMPACT_DOSE_OVERSAMPLE_PROFILES = ("O3", "O4", "O7", "O8")
COMPACT_BALANCED_WEIGHT_PROFILES = ("W14", "W15", "W16")
COMPACT_BALANCED_OVERSAMPLE_PROFILES = ("O9", "O10", "O11")
# Verified from the preceding embedding-only experiment's established search
# bracket; these values are intentionally independent of full-model LR grids.
EMBEDDING_LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)

# Frozen before looking at this study's results.  A negative delta is an
# improvement.  These are deliberately conservative *cost* limits in minutes.
GUARDRAILS = {
    "max_heavy_other_delta": 0.03,
    "max_nonheavy_delta": 0.02,
    "max_overall_delta": 0.02,
    "max_semitryptic_delta": 0.03,
    "max_worst_source_delta": 0.04,
    "require_heavy_both_nonpositive": True,
}
HUBER_RAW_DELTA_MINUTES = 1.5


def profile_text(values: tuple[float, float, float]) -> str:
    return "/".join(f"{value:g}" for value in values)


def lr_label(value: float) -> str:
    return f"lr_{value:.1e}".replace("+", "")


def stratum_from_row(row: dict[str, str]) -> str:
    value = row.get("heavy_dimethyl_class", "").strip().lower()
    mapping = {
        "heavy_nterm_only": "heavy_nterm", "heavy_lysine_only": "heavy_lysine",
        "heavy_both": "heavy_both", "normal": "normal", "": "normal",
    }
    return mapping.get(value, "normal")


def training_counts(train_file: Path) -> dict[str, int]:
    counts = {name: 0 for name in ("normal", "heavy_nterm", "heavy_lysine", "heavy_both")}
    with train_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            counts[stratum_from_row(row)] += 1
    return counts


def oversampling_budget(base_counts: dict[str, int], factors: tuple[float, float, float]) -> dict[str, int]:
    factor_by_stratum = dict(zip(("heavy_nterm", "heavy_lysine", "heavy_both"), factors))
    added = {name: int(round((factor_by_stratum.get(name, 1.0) - 1.0) * count))
             for name, count in base_counts.items()}
    base_rows = sum(base_counts.values())
    effective_rows = base_rows + sum(added.values())
    steps = math.ceil(effective_rows / BATCH_SIZE)
    return {
        "original_training_rows": base_rows, "effective_training_rows": effective_rows,
        "added_training_rows": sum(added.values()), "batches_per_epoch": steps,
        "optimizer_steps_per_epoch": steps, "total_optimizer_steps": steps * EPOCHS,
        "warmup_steps": steps * WARMUP_EPOCHS, **{f"added_{name}": value for name, value in added.items()},
    }


def target_scale_and_huber_delta(train_file: Path) -> dict[str, float]:
    """Fit the affine raw-minute ~= scale * target + offset conversion.

    Huber is applied to ``rt_norm`` in both trainers.  We therefore use a
    shared 1.5-minute transition and convert it to each target coordinate.
    """
    xs: list[float] = []
    ys: list[float] = []
    with train_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                xs.append(float(row["rt_norm"])); ys.append(float(row["rt"]))
            except (KeyError, TypeError, ValueError):
                continue
    if len(xs) < 2:
        raise ValueError(f"Cannot derive target scale from {train_file}")
    x_mean, y_mean = mean(xs), mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError(f"Degenerate target scale in {train_file}")
    scale = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    if scale <= 0:
        raise ValueError(f"Non-positive target scale in {train_file}: {scale}")
    offset = y_mean - scale * x_mean
    return {"raw_minutes_per_target_unit": scale, "raw_offset_minutes": offset,
            "huber_raw_delta_minutes": HUBER_RAW_DELTA_MINUTES,
            "huber_delta_target_units": HUBER_RAW_DELTA_MINUTES / scale}


def heavy_macro(values: dict[str, float | None]) -> float | None:
    fields = ("heavy_nterm_sb_mae", "heavy_lys_sb_mae", "heavy_both_sb_mae")
    items = [values.get(field) for field in fields]
    return mean(items) if all(item is not None and math.isfinite(float(item)) for item in items) else None
