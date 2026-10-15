"""Public API for the maplot package.

maplot is a small, extensible toolkit for working with DIA-NN result tables.
It covers the main workflow from loading `final_report` files, through optional
DIA-NN adaptation and termini annotation, to reusable plotting classes for
raw-file-, search-, and multi-search-level analyses.

Key imports:
- `PlotProjectConfig` and `PlotProject` for configuration and orchestration
- `SearchDataHandler`, `ExperimentDataHandler`, and `MultiSearchDataHandler`
  for accessing raw, adapted, and termini-annotated data
- `ResultDataLoader` for reading and writing result tables
- `DiannAdapter` and `TerminiAnnotator` for derived data generation
- `BasePlotter` and its scope-specific subclasses for implementing plots
- `PlotRegistry` and the concrete plotters for execution and visualization
"""

from .config import PlotProjectConfig
from .project import PlotProject
from .data_handler import (
    DataKind,
    ExperimentDataHandler,
    MultiSearchDataHandler,
    PlotDataBundle,
    SearchDataHandler,
)
from .data_loader import ResultDataLoader
from .diann_adapter import DiannAdapter
from .entrapment import (
    ENTRAPMENT_PREFIX,
    add_entrapment_columns,
    entrapment_qvalue_curve,
    entrapment_summary,
    exclude_diann_decoys,
    exclude_entrapment,
)
from .experiment import Experiment, ExperimentRun, QuantGroup
from .experiment import ExperimentSearch
from .termini_annotator import TerminiAnnotator
from .set_analysis import (
    ENTITY_LEVEL_DESCRIPTIONS,
    ENTITY_LEVEL_LABELS,
    EntityLevel,
    IntersectionSelection,
    add_intersection_labels,
    decode_intersection,
    encode_intersection,
    entity_feature_table,
    entity_sets,
    intersection_entities,
    intersection_summary,
)
from .base_plotter import (
    BasePlotter,
    MultiRawFilePlotter,
    MultiRunPlotter,
    MultiSearchPlotter,
    PlotOutputConfig,
    PlotScope,
    PlotStyleConfig,
    SingleRawFilePlotter,
    SingleRunPlotter,
    SingleSearchPlotter,
)
from .plot_registry import PlotRegistry
from .plotters import (
    CategoryCompositionPlotter,
    LabelEfficiencyPlotter,
    MatplotlibRawFilePresenceUpSetPlotter,
    MatplotlibRunPresenceUpSetPlotter,
    MatplotlibSearchPresenceUpSetPlotter,
    MassErrorPlotter,
    ModificationSignatureFractionPlotter,
    MultiSearchCategoryCompositionPlotter,
    PeptideLengthDistributionPlotter,
    PrecursorChargeDistributionPlotter,
    QuantCompletenessPlotter,
    QuantCorrelationPlotter,
    RawFileCountPlotter,
    RtPredictionErrorPlotter,
    RtHistogramPerRawFilePlotter,
    RtHistogramPerRunPlotter,
    RunCountPlotter,
    ScoreDistributionPlotter,
    SearchIntersectionQualityPlotter,
    SearchSummaryPlotter,
    UpSetPlotRawFilePresencePlotter,
    UpSetPlotRunPresencePlotter,
    UpSetPlotSearchPresencePlotter,
)

__all__ = [
    "PlotProjectConfig",
    "PlotProject",
    "DataKind",
    "PlotDataBundle",
    "SearchDataHandler",
    "ExperimentDataHandler",
    "MultiSearchDataHandler",
    "ResultDataLoader",
    "DiannAdapter",
    "ENTRAPMENT_PREFIX",
    "add_entrapment_columns",
    "entrapment_qvalue_curve",
    "entrapment_summary",
    "exclude_diann_decoys",
    "exclude_entrapment",
    "QuantGroup",
    "ExperimentSearch",
    "ExperimentRun",
    "Experiment",
    "TerminiAnnotator",
    "ENTITY_LEVEL_DESCRIPTIONS",
    "ENTITY_LEVEL_LABELS",
    "EntityLevel",
    "IntersectionSelection",
    "add_intersection_labels",
    "decode_intersection",
    "encode_intersection",
    "entity_feature_table",
    "entity_sets",
    "intersection_entities",
    "intersection_summary",
    "BasePlotter",
    "PlotScope",
    "PlotOutputConfig",
    "PlotStyleConfig",
    "SingleRawFilePlotter",
    "SingleRunPlotter",
    "SingleSearchPlotter",
    "MultiSearchPlotter",
    "MultiRawFilePlotter",
    "MultiRunPlotter",
    "PlotRegistry",
    "CategoryCompositionPlotter",
    "LabelEfficiencyPlotter",
    "MassErrorPlotter",
    "ModificationSignatureFractionPlotter",
    "MultiSearchCategoryCompositionPlotter",
    "PeptideLengthDistributionPlotter",
    "PrecursorChargeDistributionPlotter",
    "QuantCompletenessPlotter",
    "QuantCorrelationPlotter",
    "RawFileCountPlotter",
    "RunCountPlotter",
    "RtPredictionErrorPlotter",
    "RtHistogramPerRawFilePlotter",
    "RtHistogramPerRunPlotter",
    "ScoreDistributionPlotter",
    "SearchSummaryPlotter",
    "SearchIntersectionQualityPlotter",
    "MatplotlibRawFilePresenceUpSetPlotter",
    "MatplotlibRunPresenceUpSetPlotter",
    "MatplotlibSearchPresenceUpSetPlotter",
    "UpSetPlotRawFilePresencePlotter",
    "UpSetPlotRunPresencePlotter",
    "UpSetPlotSearchPresencePlotter",
]
