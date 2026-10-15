from __future__ import annotations

import hashlib
import json
from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = [
    "dataset_row_id",
    "dataset_row_hash",
    "source_file",
    "source_file_id",
    "raw_file_name",
    "raw_file_id",
    "run_id",
    "project_id",
    "sample_id",
    "experiment_type",
    "hunter_status",
    "sequence",
    "modified_sequence",
    "mods",
    "mod_sites",
    "charge",
    "precursor_mz",
    "protein_ids",
    "protein_start_position",
    "is_protein_n_term",
    "is_semi_specific",
    "missed_cleavages",
    "enzyme",
    "observed_rt",
    "rt_unit",
    "gradient_length",
    "q_value",
    "pep",
    "search_score",
    "intensity",
    "is_decoy",
    "scan_number",
    "precursor_id",
    "peptidoform_id",
    "source_metadata",
]

BOOLEAN_COLUMNS = ["is_protein_n_term", "is_semi_specific", "is_decoy"]
NUMERIC_COLUMNS = [
    "charge",
    "precursor_mz",
    "protein_start_position",
    "missed_cleavages",
    "observed_rt",
    "gradient_length",
    "q_value",
    "pep",
    "search_score",
    "intensity",
    "scan_number",
]


def stable_hash(parts: Iterable[object], prefix: str) -> str:
    normalized = "\x1f".join("" if pd.isna(value) else str(value).strip() for value in parts)
    return f"{prefix}_{hashlib.sha256(normalized.encode()).hexdigest()[:20]}"


def assign_stable_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_file_id"] = out["source_file"].map(lambda value: stable_hash([value], "src"))
    out["raw_file_id"] = out["raw_file_name"].map(lambda value: stable_hash([value], "raw"))
    out["peptidoform_id"] = out.apply(
        lambda row: stable_hash([row.get("sequence"), row.get("mods"), row.get("mod_sites")], "pepform"),
        axis=1,
    )
    out["precursor_id"] = out.apply(
        lambda row: stable_hash(
            [row.get("sequence"), row.get("mods"), row.get("mod_sites"), row.get("charge")],
            "prec",
        ),
        axis=1,
    )
    out["dataset_row_hash"] = out.apply(
        lambda row: stable_hash(
            [
                row.get("source_file_id"),
                row.get("raw_file_id"),
                row.get("scan_number"),
                row.get("precursor_id"),
                row.get("observed_rt"),
            ],
            "row",
        ),
        axis=1,
    )
    out = out.sort_values("dataset_row_hash", kind="stable").reset_index(drop=True)
    out["dataset_row_id"] = range(1, len(out) + 1)
    return out


def harmonize_schema(df: pd.DataFrame, preserve_unknown: bool = True) -> pd.DataFrame:
    out = df.copy()
    unknown = [column for column in out.columns if column not in CANONICAL_COLUMNS]
    if preserve_unknown and unknown:
        out["source_metadata"] = out[unknown].apply(
            lambda row: json.dumps({key: value for key, value in row.items() if pd.notna(value)}, default=str, sort_keys=True),
            axis=1,
        )
    out = out.drop(columns=unknown, errors="ignore")
    for column in CANONICAL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in BOOLEAN_COLUMNS:
        out[column] = normalize_boolean(out[column])
    out["hunter_status"] = out["hunter_status"].fillna("UNKNOWN").replace("", "UNKNOWN")
    return out[CANONICAL_COLUMNS]


def normalize_boolean(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}

    def convert(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in truthy:
            return True
        if token in falsy:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")
