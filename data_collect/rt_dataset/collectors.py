from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path

import pandas as pd

from . import __version__
from .config import CollectConfig, config_hash
from .io import DatasetStore
from .metadata import RawFileMetadataMapper
from .readers.base import SearchResultReader
from .schemas import assign_stable_ids, harmonize_schema

LOGGER = logging.getLogger(__name__)


class RTDatasetCollector:
    def __init__(
        self,
        reader: SearchResultReader,
        *,
        metadata_mapper: RawFileMetadataMapper | None = None,
        store: DatasetStore | None = None,
        config: CollectConfig | None = None,
    ):
        self.reader = reader
        self.metadata_mapper = metadata_mapper
        self.store = store
        self.config = config or CollectConfig()

    def collect(self, input_paths: list[Path], q_value_threshold: float | None = None) -> pd.DataFrame:
        threshold = self.config.q_value_threshold if q_value_threshold is None else q_value_threshold
        existing_checksums = set()
        if self.store is not None and self.config.incremental:
            manifest = self.store.load_manifest()
            if "file_checksum" in manifest.columns:
                existing_checksums = set(manifest["file_checksum"].astype(str))

        frames = []
        manifest_rows = []
        unmapped: set[str] = set()
        for path in sorted({Path(value).resolve() for value in input_paths}, key=str):
            checksum = file_checksum(path)
            if checksum in existing_checksums:
                LOGGER.info("Skipping previously imported file: %s", path)
                continue
            table = self.reader.read(path)
            rows_read = len(table)
            if "q_value" in table.columns and table["q_value"].notna().any():
                table = table.loc[pd.to_numeric(table["q_value"], errors="coerce").le(threshold)].copy()
            if self.config.exclude_decoys and "is_decoy" in table.columns:
                table = table.loc[~table["is_decoy"].fillna(False).astype(bool)].copy()
            if self.metadata_mapper is not None:
                table, report = self.metadata_mapper.map_metadata(table)
                unmapped.update(report.unmapped_raw_files)
                if report.ambiguous_mappings:
                    raise ValueError(f"Ambiguous metadata mappings: {report.ambiguous_mappings}")
            table = assign_stable_ids(harmonize_schema(table, self.config.reader.preserve_unknown_columns))
            frames.append(table)
            manifest_rows.append(
                {
                    "import_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_path": str(path),
                    "file_checksum": checksum,
                    "rows_read": rows_read,
                    "rows_retained": len(table),
                    "q_value_threshold": threshold,
                    "package_version": __version__,
                    "configuration_hash": config_hash(self.config),
                }
            )

        collected = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not collected.empty:
            collected = collected.drop_duplicates("dataset_row_hash").sort_values("dataset_row_hash").reset_index(drop=True)
            collected["dataset_row_id"] = range(1, len(collected) + 1)
        if self.store is not None:
            if self.config.incremental:
                result = self.store.append_predataset(collected) if not collected.empty else self.store.load_predataset()
            else:
                result = collected
            if not self.config.incremental and not collected.empty:
                self.store.save_predataset(collected)
            if manifest_rows:
                self.store.append_manifest(manifest_rows)
            self.store.save_mapping_outputs(
                self.metadata_mapper.mapping_table if self.metadata_mapper is not None else None,
                sorted(unmapped),
            )
            if self.config.csv_export and not result.empty:
                self.store.export_csv(result)
            return result
        return collected


def file_checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
