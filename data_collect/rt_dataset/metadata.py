from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


METADATA_COLUMNS = [
    "raw_file_name",
    "project_id",
    "sample_id",
    "experiment_type",
    "hunter_status",
    "gradient_length",
    "instrument",
    "column_id",
    "batch_id",
    "acquisition_date",
    "notes",
]


@dataclass(slots=True)
class MetadataMappingReport:
    unmapped_raw_files: list[str]
    duplicate_mappings: list[str]
    ambiguous_mappings: list[str]


class RawFileMetadataMapper:
    def __init__(self, mapping_table: pd.DataFrame, *, allow_regex: bool = False):
        self.mapping_table = mapping_table.copy()
        self.allow_regex = allow_regex
        if "raw_file_name" not in self.mapping_table.columns:
            raise ValueError("Metadata mapping requires a raw_file_name column")
        duplicates = self.mapping_table["raw_file_name"].astype(str).duplicated(keep=False)
        self.duplicate_mappings = sorted(self.mapping_table.loc[duplicates, "raw_file_name"].astype(str).unique())

    def map_metadata(self, df: pd.DataFrame) -> tuple[pd.DataFrame, MetadataMappingReport]:
        if "raw_file_name" not in df.columns:
            raise ValueError("Input data requires raw_file_name for metadata mapping")
        rows = []
        unmapped: set[str] = set()
        ambiguous: set[str] = set()
        for _, row in df.iterrows():
            raw_name = str(row["raw_file_name"])
            matches = self._matches(raw_name)
            merged = row.to_dict()
            if len(matches) == 1:
                metadata = matches.iloc[0].to_dict()
                for key, value in metadata.items():
                    if key != "raw_file_name" and pd.notna(value):
                        merged[key] = value
            elif len(matches) > 1:
                ambiguous.add(raw_name)
            else:
                unmapped.add(raw_name)
            merged["hunter_status"] = normalize_hunter_status(merged.get("hunter_status"))
            rows.append(merged)
        report = MetadataMappingReport(
            unmapped_raw_files=sorted(unmapped),
            duplicate_mappings=self.duplicate_mappings,
            ambiguous_mappings=sorted(ambiguous),
        )
        return pd.DataFrame(rows), report

    def _matches(self, raw_name: str) -> pd.DataFrame:
        exact = self.mapping_table.loc[self.mapping_table["raw_file_name"].astype(str) == raw_name]
        if not exact.empty or not self.allow_regex:
            return exact
        mask = self.mapping_table["raw_file_name"].astype(str).map(
            lambda pattern: bool(re.fullmatch(pattern, raw_name))
        )
        return self.mapping_table.loc[mask]


def normalize_hunter_status(value) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    token = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    return token if token in {"HUNTER", "PRE_HUNTER"} else "UNKNOWN"
