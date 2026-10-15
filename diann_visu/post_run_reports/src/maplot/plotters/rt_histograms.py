from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ..base_plotter import PlotOutputConfig, PlotStyleConfig, SingleRawFilePlotter


class RtHistogramPerRawFilePlotter(SingleRawFilePlotter):
    name = "rt_histogram_per_raw_file"

    def __init__(
        self,
        *,
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Configure per-raw-file RT histogram plotting.

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
        if "RT" not in df.columns:
            return None

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df["RT"].to_numpy(), bins=50, color="steelblue", rwidth=0.9)
            raw_file_name = run_name if run_name is not None else "unknown_raw_file"
            ax.set_title(f"RT histogram: {raw_file_name}")
            ax.set_xlabel("RT")
            ax.set_ylabel("Count")
            fig.tight_layout()

        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename="rt_histogram.png",
            dpi=150,
        )


class RtHistogramPerRunPlotter(RtHistogramPerRawFilePlotter):
    """Backward-compatible name for `RtHistogramPerRawFilePlotter`."""

    name = "rt_histogram_per_run"
