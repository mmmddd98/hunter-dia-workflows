from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from ..base_plotter import MultiSearchPlotter, PlotOutputConfig, PlotStyleConfig


class SearchIntersectionQualityPlotter(MultiSearchPlotter):
    """Score search intersections using DIA-NN quality columns.

    The plot uses the same intersection definition as a search-level UpSet:
    each entity belongs to the exact set of searches where it is present.
    Quality is summarized across all evidence rows for entities in that
    intersection. Lower-is-better columns such as Q.Value are converted to
    ``-log10(value)`` so higher bars consistently indicate better IDs.
    """

    name = "search_intersection_quality"

    LOWER_IS_BETTER = {
        "Q.Value",
        "PEP",
        "Global.Q.Value",
        "Lib.Q.Value",
        "Peptidoform.Q.Value",
        "Global.Peptidoform.Q.Value",
        "PG.Q.Value",
        "GG.Q.Value",
        "Protein.Q.Value",
    }

    def __init__(
        self,
        *,
        entity_col: str,
        quality_col: str = "Q.Value",
        presence_cols: list[str] | tuple[str, ...] | str | None = None,
        id_cols: set[str] | None = None,
        max_intersections: int = 30,
        min_entities: int = 1,
        higher_is_better: bool | None = None,
        title: str | None = None,
        output_name: str | None = None,
        summary_name: str | None = None,
        figsize: tuple[float, float] = (12, 5),
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.entity_col = entity_col
        self.quality_col = quality_col
        if presence_cols == "auto":
            self.presence_cols = ("auto",)
        else:
            self.presence_cols = tuple(presence_cols) if presence_cols is not None else None
        self.id_cols = id_cols or {"nterm_modif_peptide", "nterm_modif", "peptide", "protein"}
        self.max_intersections = max_intersections
        self.min_entities = min_entities
        self.higher_is_better = higher_is_better
        self.title = title
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

        plot_df = summary[summary["n_entities"] >= self.min_entities].head(self.max_intersections).copy()
        if plot_df.empty:
            return None

        dynamic_width = max(self.figsize[0], 4.8 + len(plot_df) * 0.35)
        colors = ["#3f6f8f" if size > 1 else "#9a9a9a" for size in plot_df["n_searches"]]

        with plt.rc_context(self.rc_params()):
            fig, ax1 = plt.subplots(figsize=(dynamic_width, self.figsize[1]))
            x = range(len(plot_df))
            ax1.bar(x, plot_df["quality_median_score"], color=colors, width=0.68)
            ax1.set_ylabel(self._score_label())
            ax1.set_xlabel("Search intersection")
            ax1.set_title(self.title or f"ID quality by intersection ({self.entity_col})")
            ax1.set_xticks(list(x))
            ax1.set_xticklabels(plot_df["intersection"], rotation=45, ha="right")
            ax1.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax1.set_axisbelow(True)

            ax2 = ax1.twinx()
            ax2.plot(list(x), plot_df["n_entities"], color="#222222", marker="o", linewidth=1.4)
            ax2.set_ylabel("Entities in intersection")
            ax2.spines["top"].set_visible(False)
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
        membership = self._membership_by_entity(searches)
        if not membership:
            return self._empty()

        rows = []
        for members, entities in self._entities_by_intersection(membership).items():
            evidence_scores = self._quality_scores_for_entities(searches, set(entities))
            if evidence_scores.empty:
                continue
            rows.append(
                {
                    "intersection": " & ".join(members),
                    "n_searches": len(members),
                    "n_entities": len(entities),
                    "n_quality_rows": len(evidence_scores),
                    "quality_col": self.quality_col,
                    "quality_score": self._score_label(),
                    "quality_median_score": float(evidence_scores.median()),
                    "quality_mean_score": float(evidence_scores.mean()),
                    "quality_min_score": float(evidence_scores.min()),
                    "quality_max_score": float(evidence_scores.max()),
                }
            )

        if not rows:
            return self._empty()
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["n_entities", "quality_median_score", "intersection"],
                ascending=[False, not self._higher_is_better(), True],
            )
            .reset_index(drop=True)
        )

    def _membership_by_entity(self, searches: dict[str, pl.DataFrame]) -> dict[str, tuple[str, ...]]:
        entity_to_searches: dict[str, set[str]] = {}
        for search_name, df in searches.items():
            if self.entity_col not in df.columns:
                continue
            work = df
            presence_cols = self._resolve_presence_cols(work)
            if presence_cols:
                present = pl.any_horizontal([pl.col(col).is_not_null() for col in presence_cols])
                work = work.filter(present)

            entities = (
                work.select(pl.col(self.entity_col).cast(pl.Utf8).alias("entity"))
                .drop_nulls()
                .unique()
                .get_column("entity")
                .to_list()
            )
            for entity in entities:
                entity_to_searches.setdefault(str(entity), set()).add(search_name)
        return {entity: tuple(sorted(members)) for entity, members in entity_to_searches.items()}

    @staticmethod
    def _entities_by_intersection(membership: dict[str, tuple[str, ...]]) -> dict[tuple[str, ...], list[str]]:
        out: dict[tuple[str, ...], list[str]] = {}
        for entity, members in membership.items():
            out.setdefault(members, []).append(entity)
        return out

    def _quality_scores_for_entities(self, searches: dict[str, pl.DataFrame], entities: set[str]) -> pd.Series:
        scores = []
        for df in searches.values():
            if self.entity_col not in df.columns or self.quality_col not in df.columns:
                continue
            values = (
                df.filter(pl.col(self.entity_col).cast(pl.Utf8).is_in(entities))
                .select(pl.col(self.quality_col).cast(pl.Float64, strict=False).alias("quality"))
                .drop_nulls()
                .get_column("quality")
                .to_list()
            )
            scores.extend(self._to_score(value) for value in values if value is not None)
        return pd.Series(scores, dtype="float64").dropna()

    def _resolve_presence_cols(self, df: pl.DataFrame) -> tuple[str, ...]:
        if self.presence_cols is None:
            return ()
        if self.presence_cols == ("auto",):
            return tuple(col for col in df.columns if col not in self.id_cols and col != self.entity_col)
        missing = [col for col in self.presence_cols if col not in df.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Presence columns are missing from search table: {missing_str}")
        return self.presence_cols

    def _to_score(self, value: float) -> float | None:
        if self.quality_col in self.LOWER_IS_BETTER:
            if value <= 0:
                return None
            import math

            return -math.log10(value)
        return float(value)

    def _score_label(self) -> str:
        if self.quality_col in self.LOWER_IS_BETTER:
            return f"median -log10({self.quality_col})"
        return f"median {self.quality_col}"

    def _higher_is_better(self) -> bool:
        if self.higher_is_better is not None:
            return self.higher_is_better
        return True

    def _output_filename(self) -> str:
        if self.output_name is not None:
            return self.output_name
        return f"{self._safe(self.entity_col)}_{self._safe(self.quality_col)}_intersection_quality.png"

    def _summary_filename(self) -> str:
        if self.summary_name is not None:
            return self.summary_name
        return f"{self._safe(self.entity_col)}_{self._safe(self.quality_col)}_intersection_quality.csv"

    @staticmethod
    def _safe(value: str) -> str:
        return value.replace(".", "_").replace(" ", "_").replace("/", "_")

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "intersection",
                "n_searches",
                "n_entities",
                "n_quality_rows",
                "quality_col",
                "quality_score",
                "quality_median_score",
                "quality_mean_score",
                "quality_min_score",
                "quality_max_score",
            ]
        )
