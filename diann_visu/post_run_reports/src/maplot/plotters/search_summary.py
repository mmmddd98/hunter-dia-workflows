from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from ..base_plotter import MultiSearchPlotter, PlotOutputConfig, PlotStyleConfig


class SearchSummaryPlotter(MultiSearchPlotter):
    """Summarize key counts across searches in one experiment."""

    name = "search_summary"

    def __init__(
        self,
        *,
        metrics: tuple[str, ...] = ("rows", "raw_files", "precursors", "peptides", "proteins"),
        output_name: str = "search_summary.png",
        summary_name: str = "search_summary.csv",
        figsize: tuple[float, float] = (10, 5),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.metrics = metrics
        self.output_name = output_name
        self.summary_name = summary_name
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot_many(
        self,
        searches: dict[str, pl.DataFrame],
        output_dir: str | Path | None = None,
    ) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        summary = self.calculate(searches)
        if summary.empty:
            return None

        available_metrics = [metric for metric in self.metrics if metric in summary.columns]
        if not available_metrics:
            return None

        plot_df = summary.set_index("search")[available_metrics]
        dynamic_width = max(self.figsize[0], 4.5 + len(plot_df) * len(available_metrics) * 0.35)

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(dynamic_width, self.figsize[1]))
            plot_df.plot(kind="bar", ax=ax, width=0.78)
            ax.set_xlabel("Search")
            ax.set_ylabel("Count")
            ax.set_title("Search summary")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()

        self.save_csv(summary, output_dir=output_dir, filename=self.summary_name, index=False)
        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename=self.output_name,
            dpi=150,
            bbox_inches="tight",
        )

    def calculate(self, searches: dict[str, pl.DataFrame]) -> pd.DataFrame:
        rows = []
        for search_name, df in searches.items():
            rows.append(
                {
                    "search": search_name,
                    "rows": df.height,
                    "raw_files": self._n_unique(df, ("Run", "File.Name", "sample")),
                    "precursors": self._n_unique(df, ("Precursor.Id", "nterm_modif_peptide")),
                    "peptides": self._n_unique(df, ("Stripped.Sequence", "peptide")),
                    "proteins": self._n_unique(df, ("Protein.Group", "Protein.Ids", "protein")),
                }
            )
        return pd.DataFrame(rows).sort_values("search").reset_index(drop=True)

    @staticmethod
    def _n_unique(df: pl.DataFrame, candidates: tuple[str, ...]) -> int:
        for col in candidates:
            if col in df.columns:
                return df.select(pl.col(col)).drop_nulls().unique().height
        return 0
