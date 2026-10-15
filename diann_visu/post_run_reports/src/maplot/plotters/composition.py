from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from matplotlib.ticker import PercentFormatter

from ..base_plotter import MultiSearchPlotter, PlotOutputConfig, PlotStyleConfig, SingleSearchPlotter


class CategoryCompositionPlotter(SingleSearchPlotter):
    """Plot counts for one categorical column in a single search table."""

    name = "category_composition"

    def __init__(
        self,
        *,
        category_col: str,
        entity_col: str | None = None,
        top_n: int = 20,
        min_count: int = 1,
        output_name: str | None = None,
        summary_name: str | None = None,
        figsize: tuple[float, float] = (8, 4.5),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.category_col = category_col
        self.entity_col = entity_col
        self.top_n = top_n
        self.min_count = min_count
        self.output_name = output_name
        self.summary_name = summary_name
        self.figsize = figsize
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
        summary = self.calculate(df)
        if summary.empty:
            return None

        plot_df = summary.head(self.top_n)
        dynamic_height = max(self.figsize[1], 1.8 + len(plot_df) * 0.34)

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(self.figsize[0], dynamic_height))
            ax.barh(plot_df["category"], plot_df["count"], color="#4f7cac")
            ax.invert_yaxis()
            ax.set_xlabel("Count")
            ax.set_ylabel(self.category_col)
            ax.set_title(f"{self.category_col} composition")
            ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()

        self.save_csv(summary, output_dir=output_dir, filename=self._summary_filename(), index=False)
        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename=self._output_filename(),
            dpi=150,
            bbox_inches="tight",
        )

    def calculate(self, df: pl.DataFrame) -> pd.DataFrame:
        if self.category_col not in df.columns:
            return pd.DataFrame(columns=["category", "count", "fraction"])

        work = df.select(pl.col(self.category_col).cast(pl.Utf8).fill_null("NA").alias("category"))
        if self.entity_col is not None and self.entity_col in df.columns:
            work = df.select(
                pl.col(self.category_col).cast(pl.Utf8).fill_null("NA").alias("category"),
                pl.col(self.entity_col).cast(pl.Utf8).alias("entity"),
            ).drop_nulls("entity").unique()

        counts = (
            work.group_by("category")
            .len()
            .rename({"len": "count"})
            .filter(pl.col("count") >= self.min_count)
            .sort(["count", "category"], descending=[True, False])
        )
        if counts.is_empty():
            return pd.DataFrame(columns=["category", "count", "fraction"])

        total = counts.get_column("count").sum()
        counts = counts.with_columns((pl.col("count") / total).alias("fraction"))
        return counts.to_pandas()

    def _output_filename(self) -> str:
        if self.output_name is not None:
            return self.output_name
        return f"{self._safe_col()}_composition.png"

    def _summary_filename(self) -> str:
        if self.summary_name is not None:
            return self.summary_name
        return f"{self._safe_col()}_composition.csv"

    def _safe_col(self) -> str:
        return self.category_col.replace(".", "_").replace(" ", "_").replace("/", "_")


class MultiSearchCategoryCompositionPlotter(MultiSearchPlotter):
    """Compare category composition across multiple searches."""

    name = "multi_search_category_composition"

    def __init__(
        self,
        *,
        category_col: str,
        entity_col: str | None = None,
        top_n: int = 12,
        output_name: str | None = None,
        summary_name: str | None = None,
        figsize: tuple[float, float] = (10, 5),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.category_col = category_col
        self.entity_col = entity_col
        self.top_n = top_n
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

        top_categories = (
            summary.groupby("category", as_index=False)["count"]
            .sum()
            .sort_values(["count", "category"], ascending=[False, True])
            .head(self.top_n)["category"]
            .tolist()
        )
        plot_df = summary[summary["category"].isin(top_categories)].copy()
        pivot = plot_df.pivot(index="search", columns="category", values="fraction").fillna(0.0)
        pivot = pivot.reindex(columns=top_categories)

        dynamic_width = max(self.figsize[0], 4.8 + len(top_categories) * 0.45)
        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(dynamic_width, self.figsize[1]))
            pivot.plot(kind="bar", stacked=True, ax=ax, width=0.72)
            ax.set_xlabel("Search")
            ax.set_ylabel("Fraction")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.set_title(f"{self.category_col} composition across searches")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, title=self.category_col, bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()

        self.save_csv(summary, output_dir=output_dir, filename=self._summary_filename(), index=False)
        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename=self._output_filename(),
            dpi=150,
            bbox_inches="tight",
        )

    def calculate(self, searches: dict[str, pl.DataFrame]) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for search_name, df in searches.items():
            single = CategoryCompositionPlotter(
                category_col=self.category_col,
                entity_col=self.entity_col,
                output_config=PlotOutputConfig(mode="none"),
            ).calculate(df)
            if single.empty:
                continue
            single.insert(0, "search", search_name)
            rows.append(single)
        if not rows:
            return pd.DataFrame(columns=["search", "category", "count", "fraction"])
        return pd.concat(rows, ignore_index=True)

    def _output_filename(self) -> str:
        if self.output_name is not None:
            return self.output_name
        return f"{self._safe_col()}_composition_by_search.png"

    def _summary_filename(self) -> str:
        if self.summary_name is not None:
            return self.summary_name
        return f"{self._safe_col()}_composition_by_search.csv"

    def _safe_col(self) -> str:
        return self.category_col.replace(".", "_").replace(" ", "_").replace("/", "_")
