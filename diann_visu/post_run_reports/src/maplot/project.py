from pathlib import Path

from .config import PlotProjectConfig
from .data_handler import SearchDataHandler
from .data_loader import ResultDataLoader
from .plot_registry import PlotRegistry
from .plotters import (
    LabelEfficiencyPlotter,
    MatplotlibRawFilePresenceUpSetPlotter,
    RawFileCountPlotter,
    RtHistogramPerRawFilePlotter,
)


class PlotProject:
    def __init__(self, config: PlotProjectConfig) -> None:
        """
        Build a plot project from validated config.

        Args:
            config: Input/output paths and filtering settings for one search.
        """
        self.config = config
        self.data = SearchDataHandler.from_config(config)
        self.loader = ResultDataLoader(
            input_file=config.input_file,
            qvalue_threshold=config.qvalue_threshold,
        )
        self.registry = PlotRegistry(default_output_dir=config.output_dir)

    def register_default_plotters(self) -> None:
        self.registry.register(RawFileCountPlotter())
        self.registry.register(LabelEfficiencyPlotter())
        self.registry.register(MatplotlibRawFilePresenceUpSetPlotter(entity_col="Stripped.Sequence"))
        self.registry.register(RtHistogramPerRawFilePlotter())

    def run(self) -> list[Path]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        df = self.data.raw()
        return self.registry.run_all(df)
