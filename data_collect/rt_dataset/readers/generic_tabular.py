from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import ReaderConfig
from ..schemas import harmonize_schema
from .base import SearchResultReader

LOGGER = logging.getLogger(__name__)


class GenericTabularReader(SearchResultReader):
    def __init__(self, config: ReaderConfig):
        self.config = config

    def read(self, path: Path) -> pd.DataFrame:
        table = self._read_table(path)
        reverse_mapping = {source: canonical for canonical, source in self.config.column_mapping.items()}
        duplicate_sources = [name for name in reverse_mapping if list(self.config.column_mapping.values()).count(name) > 1]
        if duplicate_sources:
            raise ValueError(f"Ambiguous source-column mappings: {sorted(set(duplicate_sources))}")
        table = table.rename(columns=reverse_mapping)
        missing = [column for column in self.config.required_columns if column not in table.columns]
        if missing:
            raise ValueError(f"{path}: missing required canonical columns: {missing}")
        table["source_file"] = str(path.resolve())
        LOGGER.info("Read %s rows from %s", len(table), path)
        return harmonize_schema(table, preserve_unknown=self.config.preserve_unknown_columns)

    def _read_table(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep=self.config.delimiter or "\t")
        if suffix == ".csv":
            return pd.read_csv(path, sep=self.config.delimiter or ",")
        raise ValueError(f"Unsupported input format: {path}")
