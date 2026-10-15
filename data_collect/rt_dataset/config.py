from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(slots=True)
class ReaderConfig:
    column_mapping: dict[str, str] = field(default_factory=dict)
    required_columns: list[str] = field(default_factory=lambda: ["sequence", "raw_file_name", "observed_rt"])
    delimiter: str | None = None
    preserve_unknown_columns: bool = True


@dataclass(slots=True)
class CollectConfig:
    input_paths: list[str] = field(default_factory=list)
    output_dir: str = "output"
    metadata_path: str | None = None
    metadata_regex: bool = False
    q_value_threshold: float = 0.01
    exclude_decoys: bool = True
    incremental: bool = True
    csv_export: bool = False
    reader: ReaderConfig = field(default_factory=ReaderConfig)


@dataclass(slots=True)
class FilterConfig:
    maximum_q_value: float | None = None
    maximum_pep: float | None = None
    exclude_decoys: bool = True
    require_valid_rt: bool = True
    minimum_rt: float | None = None
    maximum_rt: float | None = None
    minimum_intensity: float | None = None
    minimum_peptide_length: int | None = None
    maximum_peptide_length: int | None = None
    allowed_charges: list[int] | None = None
    allowed_modifications: list[str] | None = None
    excluded_modifications: list[str] | None = None
    require_unambiguous_modification_localization: bool = False
    require_n_terminal_modification: bool = False
    hunter_status: list[str] | None = None
    projects: list[str] | None = None
    runs: list[str] | None = None
    specificity: list[str] | None = None
    minimum_observations_per_peptidoform: int | None = None
    minimum_runs_per_peptidoform: int | None = None
    remove_gradient_edge_peptides: bool = False
    gradient_edge_fraction: float = 0.02
    remove_rt_outliers: bool = False
    rt_outlier_mad_multiplier: float = 4.5


T = TypeVar("T")


def load_config(path: Path, config_type: type[T]) -> T:
    payload = yaml.safe_load(path.read_text()) or {}
    if config_type is CollectConfig:
        reader = ReaderConfig(**payload.pop("reader", {}))
        return config_type(reader=reader, **payload)
    return config_type(**payload)


def config_hash(config: Any) -> str:
    payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
