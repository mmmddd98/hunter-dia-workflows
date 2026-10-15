from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from matplotlib.ticker import PercentFormatter

from ..base_plotter import MultiSearchPlotter, PlotOutputConfig, PlotStyleConfig
from ..diann_adapter import DEFAULT_UNIMOD_MAPPING


class ModificationSignatureFractionPlotter(MultiSearchPlotter):
    """Plot disjoint modification signatures as fractions of found entities.

    Each entity contributes to exactly one signature per search, so the stacked
    bars sum to 1.0. For raw DIA-NN tables the signature is the set of
    modifications observed across all rows of the entity. For adapted/termini
    tables pass ``category_col="nterm_modif"`` to use existing annotation.
    """

    name = "modification_signature_fraction"

    def __init__(
        self,
        *,
        entity_col: str,
        modified_sequence_col: str = "Modified.Sequence",
        category_col: str | None = None,
        include_unmodified: bool = True,
        top_n: int = 12,
        output_name: str | None = None,
        summary_name: str | None = None,
        title: str | None = None,
        signature_label: str = "Signature",
        figsize: tuple[float, float] = (10, 5),
        unimod_mapping: Mapping[str, str] | None = None,
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.entity_col = entity_col
        self.modified_sequence_col = modified_sequence_col
        self.category_col = category_col
        self.include_unmodified = include_unmodified
        self.top_n = top_n
        self.output_name = output_name
        self.summary_name = summary_name
        self.title = title
        self.signature_label = signature_label
        self.figsize = figsize
        self.unimod_mapping = dict(unimod_mapping or DEFAULT_UNIMOD_MAPPING)
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

        top_signatures = (
            summary.groupby("signature", as_index=False)["count"]
            .sum()
            .sort_values(["count", "signature"], ascending=[False, True])
            .head(self.top_n)["signature"]
            .tolist()
        )
        plot_df = summary.copy()
        plot_df["signature_plot"] = plot_df["signature"].where(
            plot_df["signature"].isin(top_signatures),
            "Other",
        )
        plot_df = (
            plot_df.groupby(["search", "signature_plot"], as_index=False)
            .agg({"count": "sum", "denominator": "first"})
        )
        plot_df["fraction"] = plot_df["count"] / plot_df["denominator"]

        columns = top_signatures + (["Other"] if "Other" in plot_df["signature_plot"].values else [])
        pivot = plot_df.pivot(index="search", columns="signature_plot", values="fraction").fillna(0.0)
        pivot = pivot.reindex(columns=columns)

        dynamic_width = max(self.figsize[0], 4.8 + len(columns) * 0.42)
        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(dynamic_width, self.figsize[1]))
            pivot.plot(kind="bar", stacked=True, ax=ax, width=0.72)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Search")
            ax.set_ylabel("Fraction of found entities")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.set_title(self.title or f"Modification signatures by search ({self.entity_col})")
            ax.legend(title=self.signature_label, bbox_to_anchor=(1.02, 1), loc="upper left")
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
        rows: list[dict[str, object]] = []
        for search_name, df in searches.items():
            per_entity = self._entity_signatures(df)
            if per_entity.empty:
                continue
            denominator = len(per_entity)
            counts = per_entity["signature"].value_counts().rename_axis("signature").reset_index(name="count")
            for record in counts.to_dict("records"):
                rows.append(
                    {
                        "search": search_name,
                        "entity_col": self.entity_col,
                        "signature": record["signature"],
                        "count": int(record["count"]),
                        "denominator": denominator,
                        "fraction": float(record["count"] / denominator) if denominator else 0.0,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["search", "entity_col", "signature", "count", "denominator", "fraction"])
        return pd.DataFrame(rows).sort_values(["search", "count", "signature"], ascending=[True, False, True])

    def _entity_signatures(self, df: pl.DataFrame) -> pd.DataFrame:
        if self.entity_col not in df.columns:
            return pd.DataFrame(columns=["entity", "signature"])

        if self.category_col is not None:
            if self.category_col not in df.columns:
                return pd.DataFrame(columns=["entity", "signature"])
            work = df.select(
                pl.col(self.entity_col).cast(pl.Utf8).alias("entity"),
                pl.col(self.category_col).cast(pl.Utf8).fill_null("n").alias("category"),
            ).drop_nulls("entity").unique()
            pdf = work.to_pandas()
            grouped = pdf.groupby("entity")["category"].apply(
                lambda values: self._signature_from_categories(values)
            )
            return grouped.reset_index(name="signature")

        if self.modified_sequence_col not in df.columns:
            return pd.DataFrame(columns=["entity", "signature"])
        work = df.select(
            pl.col(self.entity_col).cast(pl.Utf8).alias("entity"),
            pl.col(self.modified_sequence_col).cast(pl.Utf8).fill_null("").alias("modified_sequence"),
        ).drop_nulls("entity").unique()
        pdf = work.to_pandas()
        grouped = pdf.groupby("entity")["modified_sequence"].apply(
            lambda values: self._signature_from_sequences(values)
        )
        return grouped.reset_index(name="signature")

    def _signature_from_sequences(self, sequences: pd.Series) -> str:
        mods: set[str] = set()
        for sequence in sequences.dropna().astype(str):
            for token in re.findall(r"\(([^()]*)\)", sequence):
                mapped = self._map_modification(token)
                if mapped is not None:
                    mods.add(mapped)
        if not mods:
            return "Unmodified" if self.include_unmodified else "No named modification"
        return " + ".join(sorted(mods))

    def _signature_from_categories(self, categories: pd.Series) -> str:
        mods = {
            self._normalize_category(category)
            for category in categories.dropna().astype(str)
            if self._normalize_category(category) is not None
        }
        if not mods:
            return "Unmodified" if self.include_unmodified else "No named modification"
        return " + ".join(sorted(mods))

    @staticmethod
    def _normalize_category(category: str) -> str | None:
        token = category.strip()
        if not token or token in {"n", "NA", "None", "nan"}:
            return None
        return token

    def _map_modification(self, token: str) -> str | None:
        stripped = token.strip()
        for key, name in self.unimod_mapping.items():
            if key in stripped:
                return name
        if re.search(r"304\.207[0-9]", stripped) or re.search(r"229\.162[0-9]", stripped):
            return "TMT"
        if re.search(r"42\.010[0-9]", stripped):
            return "Acetyl"
        if re.search(r"28\.031[0-9]", stripped) or re.search(r"34\.063[0-9]", stripped) or re.search(r"36\.075[0-9]", stripped):
            return "Dimethyl"
        if re.search(r"-17\.026[0-9]", stripped) or re.search(r"-18\.010[0-9]", stripped):
            return "Pyro-Glu"
        return stripped if stripped else None

    def _output_filename(self) -> str:
        if self.output_name is not None:
            return self.output_name
        return f"{self._safe(self.entity_col)}_mod_signature_fraction.png"

    def _summary_filename(self) -> str:
        if self.summary_name is not None:
            return self.summary_name
        return f"{self._safe(self.entity_col)}_mod_signature_fraction.csv"

    @staticmethod
    def _safe(value: str) -> str:
        return value.replace(".", "_").replace(" ", "_").replace("/", "_")
