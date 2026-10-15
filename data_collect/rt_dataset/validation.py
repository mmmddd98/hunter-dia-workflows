from __future__ import annotations

import re

import pandas as pd


SEQUENCE_PATTERN = re.compile(r"^[A-Z]+$")


def validate_dataset(df: pd.DataFrame, residual_threshold: float | None = None) -> dict:
    checks: dict[str, dict] = {}

    def add(name: str, mask: pd.Series, severity: str = "error") -> None:
        indices = df.index[mask.fillna(False)].tolist()
        checks[name] = {"severity": severity, "count": len(indices), "row_indices": indices[:100]}

    for column in ("dataset_row_hash", "raw_file_id", "precursor_id", "peptidoform_id"):
        if column in df:
            add(f"missing_{column}", df[column].isna() | df[column].astype(str).eq(""))
    if "dataset_row_hash" in df:
        add("duplicate_dataset_row_hash", df["dataset_row_hash"].duplicated(keep=False))
    if "raw_file_name" in df and "raw_file_id" in df:
        add("missing_raw_file_mapping", df["raw_file_id"].isna())
    if "observed_rt" in df:
        rt = pd.to_numeric(df["observed_rt"], errors="coerce")
        add("impossible_rt", rt.isna() | rt.lt(0))
    if "q_value" in df:
        qvalue = pd.to_numeric(df["q_value"], errors="coerce")
        add("invalid_q_value", qvalue.notna() & ~qvalue.between(0, 1))
    if "sequence" in df:
        add("malformed_sequence", ~df["sequence"].fillna("").astype(str).str.match(SEQUENCE_PATTERN))
    if "mod_sites" in df:
        add("malformed_modification_positions", df["mod_sites"].fillna("").astype(str).str.contains(r"[?~]"), "warning")
    if {"raw_file_id", "gradient_length"}.issubset(df):
        gradient_counts = df.groupby("raw_file_id")["gradient_length"].nunique(dropna=True)
        conflicts = set(gradient_counts[gradient_counts.gt(1)].index)
        add("inconsistent_gradient_lengths", df["raw_file_id"].isin(conflicts))
    if {"raw_file_name", "project_id"}.issubset(df):
        mapping_counts = df.groupby("raw_file_name")["project_id"].nunique(dropna=True)
        conflicts = set(mapping_counts[mapping_counts.gt(1)].index)
        add("conflicting_raw_file_metadata", df["raw_file_name"].isin(conflicts))
    if residual_threshold is not None and "rt_conversion_residual" in df:
        residual = pd.to_numeric(df["rt_conversion_residual"], errors="coerce").abs()
        add("large_rt_conversion_residual", residual.gt(residual_threshold), "warning")
    return {
        "rows": len(df),
        "valid": not any(item["count"] and item["severity"] == "error" for item in checks.values()),
        "checks": checks,
    }
