"""Tools for building traceable peptide retention-time datasets."""

__version__ = "0.1.0"

from .aggregation import AggregationConfig, aggregate_identifications
from .collectors import RTDatasetCollector
from .config import CollectConfig, FilterConfig, load_config
from .filtering import RTDatasetFilter
from .metadata import RawFileMetadataMapper
from .schemas import CANONICAL_COLUMNS

__all__ = [
    "AggregationConfig",
    "CANONICAL_COLUMNS",
    "CollectConfig",
    "FilterConfig",
    "RTDatasetCollector",
    "RTDatasetFilter",
    "RawFileMetadataMapper",
    "aggregate_identifications",
    "load_config",
]
