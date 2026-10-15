from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

from .config import FilterConfig


@dataclass(slots=True)
class FilterStep:
    name: str
    rows_before: int
    rows_removed: int
    rows_remaining: int


class RTDatasetFilter:
    def apply(self, df: pd.DataFrame, config: FilterConfig) -> tuple[pd.DataFrame, dict]:
        out = df.copy()
        steps: list[FilterStep] = []

        def apply_mask(name: str, mask: pd.Series) -> None:
            nonlocal out
            before = len(out)
            out = out.loc[mask.fillna(False)].copy()
            steps.append(FilterStep(name, before, before - len(out), len(out)))

        numeric_filters = [
            ("maximum_q_value", "q_value", config.maximum_q_value, lambda s, v: s.le(v)),
            ("maximum_pep", "pep", config.maximum_pep, lambda s, v: s.le(v)),
            ("minimum_rt", "observed_rt", config.minimum_rt, lambda s, v: s.ge(v)),
            ("maximum_rt", "observed_rt", config.maximum_rt, lambda s, v: s.le(v)),
            ("minimum_intensity", "intensity", config.minimum_intensity, lambda s, v: s.ge(v)),
        ]
        for name, column, value, operation in numeric_filters:
            if value is not None and column in out:
                apply_mask(name, operation(pd.to_numeric(out[column], errors="coerce"), value))

        if config.exclude_decoys and "is_decoy" in out:
            apply_mask("exclude_decoys", ~out["is_decoy"].fillna(False).astype(bool))
        if config.require_valid_rt and "observed_rt" in out:
            values = pd.to_numeric(out["observed_rt"], errors="coerce")
            apply_mask("require_valid_rt", values.notna() & np.isfinite(values))

        lengths = out["sequence"].fillna("").astype(str).str.len() if "sequence" in out else pd.Series(index=out.index)
        if config.minimum_peptide_length is not None:
            apply_mask("minimum_peptide_length", lengths.ge(config.minimum_peptide_length))
        if config.maximum_peptide_length is not None:
            apply_mask("maximum_peptide_length", lengths.le(config.maximum_peptide_length))
        if config.allowed_charges is not None and "charge" in out:
            apply_mask("allowed_charges", pd.to_numeric(out["charge"], errors="coerce").isin(config.allowed_charges))

        if config.allowed_modifications is not None and "mods" in out:
            pattern = "|".join(map(str, config.allowed_modifications))
            apply_mask("allowed_modifications", out["mods"].fillna("").astype(str).str.contains(pattern, case=False, regex=True))
        if config.excluded_modifications is not None and "mods" in out:
            pattern = "|".join(map(str, config.excluded_modifications))
            apply_mask("excluded_modifications", ~out["mods"].fillna("").astype(str).str.contains(pattern, case=False, regex=True))
        if config.require_unambiguous_modification_localization and "mod_sites" in out:
            apply_mask("unambiguous_modification_localization", ~out["mod_sites"].fillna("").astype(str).str.contains(r"[?~]"))
        if config.require_n_terminal_modification and "mod_sites" in out:
            apply_mask("require_n_terminal_modification", out["mod_sites"].fillna("").astype(str).str.contains(r"(^|[;,])\s*(N-term|0)\b", case=False))
        for name, column, values in (
            ("hunter_status", "hunter_status", config.hunter_status),
            ("projects", "project_id", config.projects),
            ("runs", "run_id", config.runs),
        ):
            if values is not None and column in out:
                apply_mask(name, out[column].astype(str).isin(map(str, values)))
        if config.specificity is not None and "is_semi_specific" in out:
            allowed = {value.lower() for value in config.specificity}
            semi = out["is_semi_specific"].fillna(False).astype(bool)
            apply_mask("specificity", (semi & ("semi-tryptic" in allowed or "semi_specific" in allowed)) | (~semi & ("tryptic" in allowed)))

        if config.minimum_observations_per_peptidoform is not None and "peptidoform_id" in out:
            counts = out.groupby("peptidoform_id")["peptidoform_id"].transform("size")
            apply_mask("minimum_observations_per_peptidoform", counts.ge(config.minimum_observations_per_peptidoform))
        if config.minimum_runs_per_peptidoform is not None and {"peptidoform_id", "raw_file_id"}.issubset(out):
            counts = out.groupby("peptidoform_id")["raw_file_id"].transform("nunique")
            apply_mask("minimum_runs_per_peptidoform", counts.ge(config.minimum_runs_per_peptidoform))
        if config.remove_gradient_edge_peptides and {"observed_rt", "gradient_length"}.issubset(out):
            rt = pd.to_numeric(out["observed_rt"], errors="coerce")
            gradient = pd.to_numeric(out["gradient_length"], errors="coerce")
            fraction = config.gradient_edge_fraction
            apply_mask("remove_gradient_edge_peptides", rt.ge(gradient * fraction) & rt.le(gradient * (1 - fraction)))
        if config.remove_rt_outliers and {"peptidoform_id", "observed_rt"}.issubset(out):
            apply_mask(
                "remove_rt_outliers",
                robust_group_inlier_mask(out, "peptidoform_id", "observed_rt", config.rt_outlier_mad_multiplier),
            )

        return out.reset_index(drop=True), build_filter_report(out, steps, config)

def robust_group_inlier_mask(df: pd.DataFrame, group_col: str, value_col: str, multiplier: float) -> pd.Series:
    values = pd.to_numeric(df[value_col], errors="coerce")
    median = values.groupby(df[group_col]).transform("median")
    deviation = (values - median).abs()
    mad = deviation.groupby(df[group_col]).transform("median")
    return values.notna() & ((mad.eq(0) & deviation.eq(0)) | deviation.le(multiplier * 1.4826 * mad))


def build_filter_report(df: pd.DataFrame, steps: list[FilterStep], config: FilterConfig) -> dict:
    return {
        "configuration": asdict(config),
        "steps": [asdict(step) for step in steps],
        "summary": {
            "rows": len(df),
            "unique_sequences": df["sequence"].nunique(dropna=True) if "sequence" in df else 0,
            "unique_peptidoforms": df["peptidoform_id"].nunique(dropna=True) if "peptidoform_id" in df else 0,
            "unique_precursors": df["precursor_id"].nunique(dropna=True) if "precursor_id" in df else 0,
            "raw_files": df["raw_file_id"].nunique(dropna=True) if "raw_file_id" in df else 0,
            "projects": df["project_id"].nunique(dropna=True) if "project_id" in df else 0,
            "class_distribution": value_counts(df, "hunter_status"),
            "modification_distribution": value_counts(df, "mods"),
            "rt_distribution": describe(df, "observed_rt"),
        },
    }


def value_counts(df: pd.DataFrame, column: str) -> dict:
    return df[column].fillna("NA").astype(str).value_counts().to_dict() if column in df else {}


def describe(df: pd.DataFrame, column: str) -> dict:
    if column not in df:
        return {}
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return values.describe().to_dict() if not values.empty else {}
