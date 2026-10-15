from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class AggregationConfig:
    level: str = "peptidoform"
    method: str = "median"
    within_raw_file_first: bool = True


def aggregate_identifications(df: pd.DataFrame, config: AggregationConfig) -> pd.DataFrame:
    entity = "precursor_id" if config.level == "precursor" else "peptidoform_id"
    if entity not in df or "observed_rt" not in df:
        raise ValueError(f"Aggregation requires {entity} and observed_rt")
    work = df.copy()
    work["observed_rt"] = pd.to_numeric(work["observed_rt"], errors="coerce")
    work = work.dropna(subset=[entity, "observed_rt"])
    if config.within_raw_file_first and "raw_file_id" in work:
        work = _aggregate_once(work, [entity, "raw_file_id"], config.method)
    return _aggregate_once(work, [entity], config.method).reset_index(drop=True)


def _aggregate_once(df: pd.DataFrame, group_cols: list[str], method: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row["observed_rt"] = aggregate_rt(group, method)
        row["n_psms"] = int(group["n_psms"].sum()) if "n_psms" in group else len(group)
        row["n_precursors"] = group["precursor_id"].nunique() if "precursor_id" in group else 0
        row["n_runs"] = group["raw_file_id"].nunique() if "raw_file_id" in group else 0
        source_ids = group["source_row_ids"] if "source_row_ids" in group else group.get("dataset_row_id", pd.Series(dtype=object))
        row["source_row_ids"] = sorted({str(item) for value in source_ids for item in unpack_ids(value)})
        row["best_q_value"] = pd.to_numeric(group.get("q_value"), errors="coerce").min() if "q_value" in group else np.nan
        row["median_q_value"] = pd.to_numeric(group.get("q_value"), errors="coerce").median() if "q_value" in group else np.nan
        row["q_value"] = row["best_q_value"]
        row["intensity"] = pd.to_numeric(group.get("intensity"), errors="coerce").sum() if "intensity" in group else np.nan
        for column in ("sequence", "modified_sequence", "mods", "mod_sites", "charge", "project_id", "hunter_status"):
            if column in group:
                row[column] = group[column].dropna().iloc[0] if group[column].notna().any() else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_rt(group: pd.DataFrame, method: str) -> float:
    values = pd.to_numeric(group["observed_rt"], errors="coerce")
    if method == "best":
        score = pd.to_numeric(group.get("q_value", pd.Series(index=group.index, dtype=float)), errors="coerce")
        return float(values.loc[score.idxmin()]) if score.notna().any() else float(values.median())
    if method == "mean":
        return float(values.mean())
    if method in {"weighted_median", "intensity_weighted"}:
        weights = pd.to_numeric(group.get("intensity", pd.Series(1.0, index=group.index)), errors="coerce").fillna(0).clip(lower=0)
        if method == "intensity_weighted":
            return float(np.average(values, weights=weights)) if weights.sum() else float(values.median())
        return weighted_median(values.to_numpy(), weights.to_numpy())
    return float(values.median())


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    if weights.sum() <= 0:
        return float(np.median(values))
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")])


def unpack_ids(value) -> list:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    return [] if pd.isna(value) else [value]
