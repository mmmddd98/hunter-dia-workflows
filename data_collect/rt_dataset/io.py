from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def predataset_path(self) -> Path:
        return self.root / "predataset.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.parquet"

    def save_predataset(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.predataset_path, index=False)

    def load_predataset(self) -> pd.DataFrame:
        return pd.read_parquet(self.predataset_path) if self.predataset_path.exists() else pd.DataFrame()

    def append_predataset(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = self.load_predataset()
        combined = pd.concat([existing, df], ignore_index=True)
        if "dataset_row_hash" in combined.columns:
            combined = combined.drop_duplicates("dataset_row_hash", keep="first")
        combined = combined.sort_values("dataset_row_hash", kind="stable").reset_index(drop=True)
        combined["dataset_row_id"] = range(1, len(combined) + 1)
        self.save_predataset(combined)
        return combined

    def load_manifest(self) -> pd.DataFrame:
        return pd.read_parquet(self.manifest_path) if self.manifest_path.exists() else pd.DataFrame()

    def append_manifest(self, rows: list[dict]) -> pd.DataFrame:
        manifest = pd.concat([self.load_manifest(), pd.DataFrame(rows)], ignore_index=True)
        manifest = manifest.drop_duplicates("file_checksum", keep="first")
        manifest.to_parquet(self.manifest_path, index=False)
        return manifest

    def export_csv(self, df: pd.DataFrame, filename: str = "predataset.csv") -> Path:
        path = self.root / filename
        df.to_csv(path, index=False)
        return path

    def save_mapping_outputs(self, metadata: pd.DataFrame | None, unmapped: list[str]) -> None:
        if metadata is not None:
            metadata.to_parquet(self.root / "raw_file_metadata.parquet", index=False)
        pd.DataFrame({"raw_file_name": unmapped}).to_csv(self.root / "unmapped_raw_files.csv", index=False)

    def save_validation_report(self, report: dict) -> None:
        (self.root / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))
