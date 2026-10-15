from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from ..base_plotter import MultiSearchPlotter, PlotOutputConfig, PlotStyleConfig, SingleSearchPlotter


class PeptideLengthDistributionPlotter(SingleSearchPlotter):
    """Plot stripped peptide length distribution."""

    name = "peptide_length_distribution"

    def __init__(
        self,
        *,
        sequence_col: str = "Stripped.Sequence",
        entity_col: str | None = "Stripped.Sequence",
        output_name: str = "peptide_length_distribution.png",
        summary_name: str = "peptide_length_distribution.csv",
        figsize: tuple[float, float] = (7, 4),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.sequence_col = sequence_col
        self.entity_col = entity_col
        self.output_name = output_name
        self.summary_name = summary_name
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(self, df: pl.DataFrame, output_dir: str | Path | None = None, run_name: str | None = None) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        summary = self.calculate(df)
        if summary.empty:
            return None

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.bar(summary["length"], summary["count"], color="#5d7895", width=0.85)
            ax.set_xlabel("Peptide length")
            ax.set_ylabel("Count")
            ax.set_title("Peptide length distribution")
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()

        self.save_csv(summary, output_dir=output_dir, filename=self.summary_name, index=False)
        return self.finalize_figure(fig, output_dir=output_dir, filename=self.output_name, dpi=150, bbox_inches="tight")

    def calculate(self, df: pl.DataFrame) -> pd.DataFrame:
        if self.sequence_col not in df.columns:
            return pd.DataFrame(columns=["length", "count", "fraction"])
        cols = [pl.col(self.sequence_col).cast(pl.Utf8).alias("sequence")]
        if self.entity_col is not None and self.entity_col in df.columns:
            cols.append(pl.col(self.entity_col).cast(pl.Utf8).alias("entity"))
            work = df.select(cols).drop_nulls("entity").unique("entity")
        else:
            work = df.select(cols).drop_nulls("sequence")
        counts = (
            work.with_columns(pl.col("sequence").str.len_chars().alias("length"))
            .group_by("length")
            .len()
            .rename({"len": "count"})
            .sort("length")
        )
        if counts.is_empty():
            return pd.DataFrame(columns=["length", "count", "fraction"])
        total = counts.get_column("count").sum()
        return counts.with_columns((pl.col("count") / total).alias("fraction")).to_pandas()


class PrecursorChargeDistributionPlotter(SingleSearchPlotter):
    """Plot precursor charge-state distribution."""

    name = "precursor_charge_distribution"

    def __init__(
        self,
        *,
        charge_col: str = "Precursor.Charge",
        entity_col: str | None = "Precursor.Id",
        output_name: str = "precursor_charge_distribution.png",
        summary_name: str = "precursor_charge_distribution.csv",
        figsize: tuple[float, float] = (6, 4),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.charge_col = charge_col
        self.entity_col = entity_col
        self.output_name = output_name
        self.summary_name = summary_name
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(self, df: pl.DataFrame, output_dir: str | Path | None = None, run_name: str | None = None) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        summary = self.calculate(df)
        if summary.empty:
            return None

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.bar(summary["charge"].astype(str), summary["count"], color="#7a8f5f", width=0.72)
            ax.set_xlabel("Precursor charge")
            ax.set_ylabel("Count")
            ax.set_title("Precursor charge distribution")
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()

        self.save_csv(summary, output_dir=output_dir, filename=self.summary_name, index=False)
        return self.finalize_figure(fig, output_dir=output_dir, filename=self.output_name, dpi=150, bbox_inches="tight")

    def calculate(self, df: pl.DataFrame) -> pd.DataFrame:
        if self.charge_col not in df.columns:
            return pd.DataFrame(columns=["charge", "count", "fraction"])
        cols = [pl.col(self.charge_col).cast(pl.Int64, strict=False).alias("charge")]
        if self.entity_col is not None and self.entity_col in df.columns:
            cols.append(pl.col(self.entity_col).cast(pl.Utf8).alias("entity"))
            work = df.select(cols).drop_nulls("entity").unique("entity")
        else:
            work = df.select(cols)
        counts = work.drop_nulls("charge").group_by("charge").len().rename({"len": "count"}).sort("charge")
        if counts.is_empty():
            return pd.DataFrame(columns=["charge", "count", "fraction"])
        total = counts.get_column("count").sum()
        return counts.with_columns((pl.col("count") / total).alias("fraction")).to_pandas()


class ScoreDistributionPlotter(SingleSearchPlotter):
    """Plot DIA-NN score or error distributions."""

    name = "score_distribution"
    LOWER_IS_BETTER = {"Q.Value", "PEP", "PG.Q.Value", "GG.Q.Value", "Protein.Q.Value", "Lib.Q.Value", "Global.Q.Value"}

    def __init__(
        self,
        *,
        score_col: str = "Q.Value",
        bins: int = 50,
        output_name: str | None = None,
        summary_name: str | None = None,
        figsize: tuple[float, float] = (7, 4),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.score_col = score_col
        self.bins = bins
        self.output_name = output_name
        self.summary_name = summary_name
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(self, df: pl.DataFrame, output_dir: str | Path | None = None, run_name: str | None = None) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        values = self._values(df)
        if values.empty:
            return None

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.hist(values, bins=self.bins, color="#725e9c", rwidth=0.9)
            ax.set_xlabel(self._label())
            ax.set_ylabel("Rows")
            ax.set_title(f"{self.score_col} distribution")
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()

        self.save_csv(self.calculate(df), output_dir=output_dir, filename=self._summary_filename(), index=False)
        return self.finalize_figure(fig, output_dir=output_dir, filename=self._output_filename(), dpi=150, bbox_inches="tight")

    def calculate(self, df: pl.DataFrame) -> pd.DataFrame:
        values = self._values(df)
        if values.empty:
            return pd.DataFrame(columns=["score_col", "n", "median", "mean", "q10", "q90"])
        return pd.DataFrame(
            {
                "score_col": [self.score_col],
                "score_transform": [self._label()],
                "n": [int(values.size)],
                "median": [float(values.median())],
                "mean": [float(values.mean())],
                "q10": [float(values.quantile(0.10))],
                "q90": [float(values.quantile(0.90))],
            }
        )

    def _values(self, df: pl.DataFrame) -> pd.Series:
        if self.score_col not in df.columns:
            return pd.Series([], dtype="float64")
        values = df.select(pl.col(self.score_col).cast(pl.Float64, strict=False).alias("score")).drop_nulls().to_pandas()["score"]
        if self.score_col in self.LOWER_IS_BETTER:
            values = values[values > 0].map(lambda value: -math.log10(value))
        return values.dropna()

    def _label(self) -> str:
        return f"-log10({self.score_col})" if self.score_col in self.LOWER_IS_BETTER else self.score_col

    def _output_filename(self) -> str:
        return self.output_name or f"{self._safe(self.score_col)}_distribution.png"

    def _summary_filename(self) -> str:
        return self.summary_name or f"{self._safe(self.score_col)}_distribution.csv"

    @staticmethod
    def _safe(value: str) -> str:
        return value.replace(".", "_").replace("/", "_").replace(" ", "_")


class RtPredictionErrorPlotter(SingleSearchPlotter):
    """Plot observed vs predicted RT and residuals."""

    name = "rt_prediction_error"

    def __init__(
        self,
        *,
        observed_col: str = "RT",
        predicted_col: str = "Predicted.RT",
        output_name: str = "rt_prediction_error.png",
        summary_name: str = "rt_prediction_error.csv",
        max_points: int = 8000,
        figsize: tuple[float, float] = (9, 4),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.observed_col = observed_col
        self.predicted_col = predicted_col
        self.output_name = output_name
        self.summary_name = summary_name
        self.max_points = max_points
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(self, df: pl.DataFrame, output_dir: str | Path | None = None, run_name: str | None = None) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        work = self._work(df)
        if work.empty:
            return None
        plot_df = work.head(self.max_points)

        with plt.rc_context(self.rc_params()):
            fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=self.figsize)
            ax_scatter.scatter(plot_df["predicted"], plot_df["observed"], s=5, alpha=0.28, color="#355c7d")
            lo = min(plot_df["predicted"].min(), plot_df["observed"].min())
            hi = max(plot_df["predicted"].max(), plot_df["observed"].max())
            ax_scatter.plot([lo, hi], [lo, hi], color="#222222", linewidth=1)
            ax_scatter.set_xlabel(self.predicted_col)
            ax_scatter.set_ylabel(self.observed_col)
            ax_scatter.set_title("Observed vs predicted RT")
            ax_hist.hist(work["residual"], bins=50, color="#c06c84", rwidth=0.9)
            ax_hist.set_xlabel(f"{self.observed_col} - {self.predicted_col}")
            ax_hist.set_ylabel("Rows")
            ax_hist.set_title("RT residuals")
            fig.tight_layout()

        self.save_csv(self.calculate(df), output_dir=output_dir, filename=self.summary_name, index=False)
        return self.finalize_figure(fig, output_dir=output_dir, filename=self.output_name, dpi=150, bbox_inches="tight")

    def calculate(self, df: pl.DataFrame) -> pd.DataFrame:
        work = self._work(df)
        if work.empty:
            return pd.DataFrame(columns=["n", "median_abs_error", "mean_abs_error", "median_error"])
        return pd.DataFrame(
            {
                "n": [int(len(work))],
                "median_abs_error": [float(work["residual"].abs().median())],
                "mean_abs_error": [float(work["residual"].abs().mean())],
                "median_error": [float(work["residual"].median())],
            }
        )

    def _work(self, df: pl.DataFrame) -> pd.DataFrame:
        if self.observed_col not in df.columns or self.predicted_col not in df.columns:
            return pd.DataFrame(columns=["observed", "predicted", "residual"])
        work = (
            df.select(
                pl.col(self.observed_col).cast(pl.Float64, strict=False).alias("observed"),
                pl.col(self.predicted_col).cast(pl.Float64, strict=False).alias("predicted"),
            )
            .drop_nulls()
            .with_columns((pl.col("observed") - pl.col("predicted")).alias("residual"))
        )
        return work.to_pandas()


class MassErrorPlotter(SingleSearchPlotter):
    """Plot mass-error column distribution."""

    name = "mass_error_distribution"

    def __init__(
        self,
        *,
        error_col: str = "Best.Fr.Mz.Delta",
        output_name: str | None = None,
        summary_name: str | None = None,
        bins: int = 60,
        figsize: tuple[float, float] = (7, 4),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.error_col = error_col
        self.output_name = output_name
        self.summary_name = summary_name
        self.bins = bins
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(self, df: pl.DataFrame, output_dir: str | Path | None = None, run_name: str | None = None) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        values = self._values(df)
        if values.empty:
            return None

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.hist(values, bins=self.bins, color="#b56576", rwidth=0.9)
            ax.axvline(0, color="#222222", linewidth=1)
            ax.set_xlabel(self.error_col)
            ax.set_ylabel("Rows")
            ax.set_title(f"{self.error_col} distribution")
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()

        self.save_csv(self.calculate(df), output_dir=output_dir, filename=self._summary_filename(), index=False)
        return self.finalize_figure(fig, output_dir=output_dir, filename=self._output_filename(), dpi=150, bbox_inches="tight")

    def calculate(self, df: pl.DataFrame) -> pd.DataFrame:
        values = self._values(df)
        if values.empty:
            return pd.DataFrame(columns=["error_col", "n", "median", "mean", "mad"])
        return pd.DataFrame(
            {
                "error_col": [self.error_col],
                "n": [int(values.size)],
                "median": [float(values.median())],
                "mean": [float(values.mean())],
                "mad": [float((values - values.median()).abs().median())],
            }
        )

    def _values(self, df: pl.DataFrame) -> pd.Series:
        if self.error_col not in df.columns:
            return pd.Series([], dtype="float64")
        return df.select(pl.col(self.error_col).cast(pl.Float64, strict=False).alias("error")).drop_nulls().to_pandas()["error"].dropna()

    def _output_filename(self) -> str:
        return self.output_name or f"{self._safe(self.error_col)}_distribution.png"

    def _summary_filename(self) -> str:
        return self.summary_name or f"{self._safe(self.error_col)}_distribution.csv"

    @staticmethod
    def _safe(value: str) -> str:
        return value.replace(".", "_").replace("/", "_").replace(" ", "_")


class QuantCompletenessPlotter(MultiSearchPlotter):
    """Plot sample completeness from adapted peptide/protein matrices."""

    name = "quant_completeness"

    def __init__(
        self,
        *,
        id_cols: set[str] | None = None,
        output_name: str = "quant_completeness.png",
        summary_name: str = "quant_completeness.csv",
        figsize: tuple[float, float] = (10, 4.5),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.id_cols = id_cols or {"nterm_modif_peptide", "nterm_modif", "peptide", "protein"}
        self.output_name = output_name
        self.summary_name = summary_name
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot_many(self, searches: dict[str, pl.DataFrame], output_dir: str | Path | None = None) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        summary = self.calculate(searches)
        if summary.empty:
            return None

        pivot = summary.pivot(index="sample", columns="search", values="fraction_present").fillna(0.0)
        dynamic_height = max(self.figsize[1], 2.0 + len(pivot) * 0.28)
        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(self.figsize[0], dynamic_height))
            image = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("Quant completeness")
            ax.set_xlabel("Search")
            ax.set_ylabel("Sample")
            fig.colorbar(image, ax=ax, label="Fraction present")
            fig.tight_layout()

        self.save_csv(summary, output_dir=output_dir, filename=self.summary_name, index=False)
        return self.finalize_figure(fig, output_dir=output_dir, filename=self.output_name, dpi=150, bbox_inches="tight")

    def calculate(self, searches: dict[str, pl.DataFrame]) -> pd.DataFrame:
        rows = []
        for search_name, df in searches.items():
            value_cols = [col for col in df.columns if col not in self.id_cols]
            denominator = df.height
            if denominator == 0:
                continue
            for col in value_cols:
                present = df.select(pl.col(col).is_not_null().sum()).item()
                rows.append({"search": search_name, "sample": col, "present": int(present), "denominator": denominator, "fraction_present": present / denominator})
        return pd.DataFrame(rows)


class QuantCorrelationPlotter(SingleSearchPlotter):
    """Plot sample correlation heatmap from adapted matrices."""

    name = "quant_correlation"

    def __init__(
        self,
        *,
        id_cols: set[str] | None = None,
        output_name: str = "quant_correlation.png",
        summary_name: str = "quant_correlation.csv",
        figsize: tuple[float, float] = (6, 5),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.id_cols = id_cols or {"nterm_modif_peptide", "nterm_modif", "peptide", "protein"}
        self.output_name = output_name
        self.summary_name = summary_name
        self.figsize = figsize
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(self, df: pl.DataFrame, output_dir: str | Path | None = None, run_name: str | None = None) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        corr = self.calculate(df)
        if corr.empty:
            return None
        matrix = corr.pivot(index="sample_a", columns="sample_b", values="correlation")

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=self.figsize)
            image = ax.imshow(matrix.values, aspect="equal", vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels(matrix.index)
            ax.set_title("Sample correlation")
            fig.colorbar(image, ax=ax, label="Pearson r")
            fig.tight_layout()

        self.save_csv(corr, output_dir=output_dir, filename=self.summary_name, index=False)
        return self.finalize_figure(fig, output_dir=output_dir, filename=self.output_name, dpi=150, bbox_inches="tight")

    def calculate(self, df: pl.DataFrame) -> pd.DataFrame:
        value_cols = [col for col in df.columns if col not in self.id_cols]
        if len(value_cols) < 2:
            return pd.DataFrame(columns=["sample_a", "sample_b", "correlation"])
        corr = df.select(value_cols).to_pandas().corr(method="pearson")
        return corr.stack().rename("correlation").reset_index().rename(columns={"level_0": "sample_a", "level_1": "sample_b"})
