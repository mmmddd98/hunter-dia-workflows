from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ..base_plotter import PlotOutputConfig, PlotStyleConfig, SingleSearchPlotter


class RawFileCountPlotter(SingleSearchPlotter):
    name = "raw_file_counts"

    def __init__(
        self,
        *,
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Configure raw-file count plotting.

        Args:
            output_config: Save/show behavior shared across plotters.
            style_config: Matplotlib style settings, including plot font family.
            output_dir: Optional default output directory for this plotter instance.
        """
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(
        self,
        df: pl.DataFrame,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        if "Run" not in df.columns:
            return None

        raw_file_counts = df.group_by("Run").len().sort("Run")

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(raw_file_counts["Run"].to_list(), raw_file_counts["len"].to_list())
            ax.set_title("Identifications per raw file")
            ax.set_xlabel("Raw file")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()

        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename="run_counts.png",
            dpi=150,
        )


class RunCountPlotter(RawFileCountPlotter):
    """Backward-compatible name for `RawFileCountPlotter`."""

    name = "run_counts"
