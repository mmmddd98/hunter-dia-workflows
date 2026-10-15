from collections import Counter
from pathlib import Path

import polars as pl

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..base_plotter import MultiSearchPlotter, PlotOutputConfig, PlotStyleConfig, SingleSearchPlotter


RAW_MOD_PATTERNS = {
    "Dimethyl": r"UniMod:36|UniMod:510|28\.031|34\.063|36\.075",
    "Acetyl": r"UniMod:1|42\.010|Acetyl_delta_after_fixed_UniMod510",
    "Oxidation": r"UniMod:35|15\.994",
    "Carbamidomethyl": r"UniMod:4|57\.021",
    "Pyro-Glu": r"UniMod:26|UniMod:27|-17\.026|-18\.010",
}


def raw_mod_label_expr() -> pl.Expr:
    mod = pl.col("Modified.Sequence").cast(pl.Utf8).str.extract_all(r"\(([^()]*)\)")
    joined = mod.list.join(";")
    return (
        pl.when(joined.str.contains(RAW_MOD_PATTERNS["Dimethyl"]))
        .then(pl.lit("Dimethyl"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Acetyl"]))
        .then(pl.lit("Acetyl"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Oxidation"]))
        .then(pl.lit("Oxidation"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Carbamidomethyl"]))
        .then(pl.lit("Carbamidomethyl"))
        .when(joined.str.contains(RAW_MOD_PATTERNS["Pyro-Glu"]))
        .then(pl.lit("Pyro-Glu"))
        .when(mod.list.len() == 0)
        .then(pl.lit("Unmodified"))
        .otherwise(pl.lit("Other named mod"))
        .alias("category")
    )


class MatplotlibRawFilePresenceUpSetPlotter(SingleSearchPlotter):
    name = "raw_file_presence_upset_matplotlib"

    def __init__(
        self,
        entity_col: str = "Stripped.Sequence",
        set_col: str | None = None,
        *,
        min_subset_size: int = 1,
        max_intersections: int = 40,
        figsize: tuple[float, float] = (12, 7),
        output_name: str | None = None,
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Configure UpSet-style overlap plotting across raw files/samples.

        Args:
            entity_col: Entity identifier to intersect (peptide/protein column).
            set_col: Grouping column for sets; auto-detected when omitted.
            min_subset_size: Minimum intersection size to keep.
            max_intersections: Maximum number of intersections shown.
            figsize: Matplotlib figure size in inches.
            output_name: Optional custom output image filename.
            output_config: Save/show behavior shared across plotters.
            style_config: Matplotlib style settings, including plot font family.
            output_dir: Optional default output directory for this plotter instance.
        """
        self.entity_col = entity_col
        self.set_col = set_col
        self.min_subset_size = min_subset_size
        self.max_intersections = max_intersections
        self.figsize = figsize
        self.output_name = output_name
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
        set_col = self._resolve_set_col(df)
        if set_col is None or self.entity_col not in df.columns:
            return None

        set_names, subset_counts, set_sizes = self._build_intersections(df, set_col)
        if not set_names or not subset_counts:
            return None

        subset_items = [
            (members, count)
            for members, count in subset_counts.items()
            if count >= self.min_subset_size
        ]
        subset_items.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        subset_items = subset_items[: self.max_intersections]
        if not subset_items:
            return None

        dynamic_width = max(self.figsize[0], 4.9 + len(subset_items) * 0.28)
        dynamic_height = max(self.figsize[1], 3.1 + len(set_names) * 0.42)

        with plt.rc_context(
            self.rc_params(
                {
                    "axes.edgecolor": "#2b2b2b",
                    "axes.labelcolor": "#111111",
                    "axes.linewidth": 1.15,
                    "figure.facecolor": "white",
                    "savefig.facecolor": "white",
                }
            )
        ):
            fig = plt.figure(figsize=(dynamic_width, dynamic_height), constrained_layout=True)
            grid = fig.add_gridspec(
                nrows=2,
                ncols=2,
                width_ratios=(1.65, max(4.8, len(subset_items) * 0.28)),
                height_ratios=(3.35, max(1.65, len(set_names) * 0.42)),
                hspace=0.015,
                wspace=0.035,
            )
            fig.add_subplot(grid[0, 0]).axis("off")
            ax_bar = fig.add_subplot(grid[0, 1])
            ax_matrix = fig.add_subplot(grid[1, 1], sharex=ax_bar)
            ax_sets = fig.add_subplot(grid[1, 0], sharey=ax_matrix)

            counts = [count for _, count in subset_items]
            x_positions = list(range(len(subset_items)))
            bars = ax_bar.bar(
                x_positions,
                counts,
                width=0.34,
                color="#111111",
                edgecolor="#111111",
                linewidth=0,
            )
            ax_bar.set_ylabel("Intersection Size")
            ax_bar.yaxis.label.set_size(14)
            ax_bar.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda value, _: self._format_count(value)))
            ax_bar.grid(axis="y", color="#e6e6e6", linewidth=0.9)
            ax_bar.set_axisbelow(True)
            ax_bar.tick_params(axis="x", labelbottom=False, length=0)
            ax_bar.tick_params(axis="y", labelsize=11, width=1.0, length=4)
            ax_bar.spines[["top", "right"]].set_visible(False)
            ax_bar.spines["bottom"].set_color("#111111")
            ax_bar.spines["left"].set_color("#111111")

            y_max = max(counts)
            ax_bar.set_ylim(0, y_max * 1.12 if y_max else 1)
            label_threshold = max(1, y_max * 0.035)
            for bar, count in zip(bars, counts):
                if count < label_threshold and len(subset_items) > 28:
                    continue
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.01,
                    self._format_count(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if len(subset_items) > 18 else 0,
                    color="#111111",
                )

            y_positions = list(range(len(set_names)))
            for y in y_positions:
                if y % 2 == 1:
                    ax_matrix.axhspan(y - 0.5, y + 0.5, color="#f1f1f1", zorder=0)
                ax_matrix.hlines(y, -0.5, len(subset_items) - 0.5, color="#eeeeee", linewidth=0.7, zorder=1)

            active_color = "#111111"
            inactive_color = "#dedede"
            for x, (members, _) in zip(x_positions, subset_items):
                active_ys = []
                for y, set_name in enumerate(set_names):
                    is_active = set_name in members
                    ax_matrix.scatter(
                        x,
                        y,
                        s=82 if is_active else 62,
                        color=active_color if is_active else inactive_color,
                        edgecolor=active_color if is_active else inactive_color,
                        linewidth=0,
                        zorder=4 if is_active else 3,
                    )
                    if is_active:
                        active_ys.append(y)
                if len(active_ys) > 1:
                    ax_matrix.vlines(x, min(active_ys), max(active_ys), color=active_color, linewidth=2.4, zorder=2)

            ax_matrix.set_yticks(y_positions)
            ax_matrix.set_yticklabels([self._label(name) for name in set_names])
            ax_matrix.invert_yaxis()
            ax_matrix.set_xlabel("")
            ax_matrix.tick_params(axis="x", labelbottom=False, length=0)
            ax_matrix.tick_params(axis="y", labelsize=13, length=0, pad=14)
            ax_matrix.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax_matrix.set_xlim(-0.5, len(subset_items) - 0.5)

            set_size_values = [set_sizes[name] for name in set_names]
            set_bars = ax_sets.barh(
                y_positions,
                set_size_values,
                color="#111111",
                edgecolor="#111111",
                linewidth=0,
                height=0.34,
            )
            set_max = max(set_size_values) if set_size_values else 1
            ax_sets.set_xlim(set_max * 1.12, 0)
            ax_sets.set_xlabel("Set Size")
            ax_sets.xaxis.label.set_size(13)
            ax_sets.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax_sets.xaxis.set_major_formatter(FuncFormatter(lambda value, _: self._format_count(value)))
            ax_sets.grid(axis="x", color="#e6e6e6", linewidth=0.9)
            ax_sets.set_axisbelow(True)
            ax_sets.tick_params(axis="x", labelsize=10, width=1.0, length=4)
            ax_sets.tick_params(axis="y", labelleft=False, length=0)
            ax_sets.spines[["top", "right", "left"]].set_visible(False)
            ax_sets.spines["bottom"].set_color("#111111")
            for bar, value in zip(set_bars, set_size_values):
                ax_sets.text(
                    value + set_max * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    self._format_count(value),
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="#111111",
                )

        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename=self._output_filename("matplotlib"),
            dpi=150,
            bbox_inches="tight",
        )

    def _resolve_set_col(self, df: pl.DataFrame) -> str | None:
        if self.set_col is not None:
            return self.set_col if self.set_col in df.columns else None
        for candidate in ("Run", "File.Name", "sample"):
            if candidate in df.columns:
                return candidate
        return None

    def _output_filename(self, backend: str) -> str:
        if self.output_name is not None:
            return self.output_name
        safe_entity = self.entity_col.replace(".", "_").replace(" ", "_").replace("/", "_")
        return f"upset_{safe_entity}_{backend}.png"

    @staticmethod
    def _format_count(value: float | int) -> str:
        value = int(round(value))
        return f"{value:,}".replace(",", " ")

    @staticmethod
    def _label(value: str) -> str:
        return value.replace("_", " ").replace(".", " ")

    def _build_intersections(
        self,
        df: pl.DataFrame,
        set_col: str,
    ) -> tuple[list[str], Counter[tuple[str, ...]], dict[str, int]]:
        memberships = (
            df.select([set_col, self.entity_col])
            .drop_nulls()
            .unique()
            .group_by(self.entity_col)
            .agg(pl.col(set_col).cast(pl.Utf8).sort().alias("sets"))
        )
        if memberships.is_empty():
            return [], Counter(), {}

        subset_counts: Counter[tuple[str, ...]] = Counter()
        set_sizes: Counter[str] = Counter()
        for sets in memberships.get_column("sets").to_list():
            members = tuple(str(item) for item in sets if item is not None)
            if not members:
                continue
            subset_counts[members] += 1
            set_sizes.update(members)

        set_names = sorted(set_sizes, key=lambda name: (-set_sizes[name], name))
        return set_names, subset_counts, dict(set_sizes)


class MatplotlibRunPresenceUpSetPlotter(MatplotlibRawFilePresenceUpSetPlotter):
    """Backward-compatible name for `MatplotlibRawFilePresenceUpSetPlotter`."""

    name = "upset_presence_matplotlib"


class MatplotlibSearchPresenceUpSetPlotter(MatplotlibRawFilePresenceUpSetPlotter, MultiSearchPlotter):
    """UpSet plotter for entity overlap across searches in one experiment."""

    name = "search_presence_upset_matplotlib"
    scope = "multi_search"

    def __init__(
        self,
        entity_col: str = "Stripped.Sequence",
        *,
        presence_cols: list[str] | tuple[str, ...] | str | None = None,
        id_cols: set[str] | None = None,
        min_subset_size: int = 1,
        max_intersections: int = 40,
        figsize: tuple[float, float] = (12, 7),
        output_name: str | None = None,
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Configure search-level UpSet plotting.

        Args:
            entity_col: Entity identifier to compare across searches.
            presence_cols: Optional value/sample columns that define whether an
                entity is present in a search. Use `"auto"` to use all non-ID
                columns. When omitted, each row with a non-null entity is
                counted as present.
            id_cols: Optional columns excluded by `auto_presence_cols()`.
            min_subset_size: Minimum intersection size to keep.
            max_intersections: Maximum number of intersections shown.
            figsize: Matplotlib figure size in inches.
            output_name: Optional custom output image filename.
            output_config: Save/show behavior shared across plotters.
            style_config: Matplotlib style settings, including plot font family.
            output_dir: Optional default output directory for this plotter instance.
        """
        super().__init__(
            entity_col=entity_col,
            set_col="search",
            min_subset_size=min_subset_size,
            max_intersections=max_intersections,
            figsize=figsize,
            output_name=output_name,
            output_config=output_config,
            style_config=style_config,
            output_dir=output_dir,
        )
        if presence_cols == "auto":
            self.presence_cols = ("auto",)
        else:
            self.presence_cols = tuple(presence_cols) if presence_cols is not None else None
        self.id_cols = id_cols or {"nterm_modif_peptide", "nterm_modif", "peptide", "protein"}

    def plot_many(
        self,
        searches: dict[str, pl.DataFrame],
        output_dir: str | Path | None = None,
    ) -> Path | None:
        """Plot entity overlap across search names."""
        output_dir = self.resolve_output_dir(output_dir)
        presence_df = self._build_search_presence(searches)
        if presence_df.is_empty():
            return None
        set_names, subset_counts, _set_sizes = self._build_intersections(presence_df, "search")
        subset_items = [
            (members, count)
            for members, count in subset_counts.items()
            if count >= self.min_subset_size
        ]
        subset_items.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        subset_items = subset_items[: self.max_intersections]
        self._write_intersection_artifacts(searches, presence_df, set_names, subset_items, output_dir)
        return super().plot(presence_df, output_dir=output_dir)

    def auto_presence_cols(self, df: pl.DataFrame) -> list[str]:
        """Return non-ID columns usable as sample/value presence columns."""
        return [col for col in df.columns if col not in self.id_cols and col != self.entity_col]

    def _build_search_presence(self, searches: dict[str, pl.DataFrame]) -> pl.DataFrame:
        rows: list[pl.DataFrame] = []
        for search_name, df in searches.items():
            if self.entity_col not in df.columns:
                continue

            work = df
            presence_cols = self._resolve_presence_cols(work)
            if presence_cols:
                present = pl.any_horizontal([pl.col(col).is_not_null() for col in presence_cols])
                work = work.filter(present)

            rows.append(
                work.select(
                    pl.lit(search_name).alias("search"),
                    pl.col(self.entity_col).cast(pl.Utf8).alias(self.entity_col),
                )
                .drop_nulls()
                .unique()
            )

        if not rows:
            return pl.DataFrame(schema={"search": pl.Utf8, self.entity_col: pl.Utf8})
        return pl.concat(rows).unique()

    def _resolve_presence_cols(self, df: pl.DataFrame) -> tuple[str, ...]:
        if self.presence_cols is None:
            return ()

        if self.presence_cols == ("auto",):
            return tuple(self.auto_presence_cols(df))

        missing = [col for col in self.presence_cols if col not in df.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Presence columns are missing from search table: {missing_str}")
        return self.presence_cols

    def _write_intersection_artifacts(
        self,
        searches: dict[str, pl.DataFrame],
        presence_df: pl.DataFrame,
        set_names: list[str],
        subset_items: list[tuple[tuple[str, ...], int]],
        output_dir: Path,
    ) -> None:
        stem = Path(self._output_filename("matplotlib")).stem
        membership = self._membership_table(presence_df, set_names)
        if membership.is_empty():
            return

        membership.write_csv(output_dir / f"{stem}_membership.csv")

        detail_dir = output_dir / f"{stem}_intersections"
        detail_dir.mkdir(parents=True, exist_ok=True)
        top_intersections = [" & ".join(members) for members, _ in subset_items]
        membership.filter(pl.col("intersection").is_in(top_intersections)).write_csv(
            detail_dir / "shown_intersection_entities.csv"
        )

        for artifact in (
            self._category_artifact(searches, membership, kind="modification"),
            self._category_artifact(searches, membership, kind="specificity"),
        ):
            if artifact.is_empty():
                continue
            kind = artifact.get_column("category_kind").item(0)
            artifact.write_csv(detail_dir / f"{kind}_fraction_by_intersection.csv")
            self._plot_stacked_category_artifact(
                artifact,
                top_intersections,
                detail_dir / f"{kind}_fraction_by_intersection.png",
                title=f"{self._label(kind)} composition by UpSet intersection",
            )
            self._plot_single_intersection_category_artifacts(
                artifact,
                membership,
                subset_items,
                detail_dir,
                kind=kind,
            )

    def _membership_table(self, presence_df: pl.DataFrame, set_names: list[str]) -> pl.DataFrame:
        memberships = (
            presence_df.select(["search", self.entity_col])
            .drop_nulls()
            .unique()
            .group_by(self.entity_col)
            .agg(pl.col("search").cast(pl.Utf8).sort().alias("searches"))
            .with_columns(
                pl.col("searches").list.join(" & ").alias("intersection"),
                pl.col("searches").list.len().alias("n_searches"),
            )
        )
        search_flags = [
            pl.col("searches").list.contains(search_name).alias(f"present__{self._safe(search_name)}")
            for search_name in set_names
        ]
        return (
            memberships.with_columns(search_flags)
            .select(
                [
                    pl.col(self.entity_col).alias("entity"),
                    "intersection",
                    "n_searches",
                    *[f"present__{self._safe(search_name)}" for search_name in set_names],
                ]
            )
            .sort(["n_searches", "intersection", "entity"], descending=[True, False, False])
        )

    def _category_artifact(
        self,
        searches: dict[str, pl.DataFrame],
        membership: pl.DataFrame,
        *,
        kind: str,
    ) -> pl.DataFrame:
        evidence = self._category_evidence(searches, kind=kind)
        if evidence.is_empty():
            return pl.DataFrame()

        member_index = membership.select(["entity", "intersection"])
        joined = evidence.join(member_index, on="entity", how="inner").unique(["intersection", "entity", "category"])
        if joined.is_empty():
            return pl.DataFrame()

        counts = joined.group_by(["intersection", "category"]).len(name="n_entities")
        totals = counts.group_by("intersection").agg(pl.col("n_entities").sum().alias("intersection_total"))
        return (
            counts.join(totals, on="intersection")
            .with_columns(
                (pl.col("n_entities") / pl.col("intersection_total") * 100).alias("fraction_percent"),
                pl.lit(kind).alias("category_kind"),
            )
            .select(["category_kind", "intersection", "category", "n_entities", "intersection_total", "fraction_percent"])
            .sort(["intersection_total", "intersection", "category"], descending=[True, False, False])
        )

    def _category_evidence(self, searches: dict[str, pl.DataFrame], *, kind: str) -> pl.DataFrame:
        rows: list[pl.DataFrame] = []
        for search_name, df in searches.items():
            if self.entity_col not in df.columns:
                continue
            category_expr = self._category_expr(df, kind=kind)
            if category_expr is None:
                continue
            rows.append(
                df.select(
                    pl.lit(search_name).alias("search"),
                    pl.col(self.entity_col).cast(pl.Utf8).alias("entity"),
                    category_expr,
                )
                .drop_nulls()
                .unique()
            )
        if not rows:
            return pl.DataFrame(schema={"search": pl.Utf8, "entity": pl.Utf8, "category": pl.Utf8})
        return pl.concat(rows).unique()

    def _category_expr(self, df: pl.DataFrame, *, kind: str) -> pl.Expr | None:
        if kind == "modification":
            if "nterm_modif" in df.columns:
                return (
                    pl.when(pl.col("nterm_modif").cast(pl.Utf8).is_in(["", "n", "NA", "None"]))
                    .then(pl.lit("Unmodified"))
                    .otherwise(pl.col("nterm_modif").cast(pl.Utf8))
                    .alias("category")
                )
            if "mod_label" in df.columns:
                return pl.col("mod_label").cast(pl.Utf8).alias("category")
            if "Modified.Sequence" in df.columns:
                return raw_mod_label_expr()
            return None

        if kind == "specificity" and "specificity" in df.columns:
            value = pl.col("specificity").cast(pl.Utf8)
            return (
                pl.when(value.str.to_lowercase().str.contains("semi"))
                .then(pl.lit("Semi-specific"))
                .when(value.str.to_lowercase().str.contains("specific"))
                .then(pl.lit("Specific"))
                .otherwise(value)
                .alias("category")
            )
        return None

    def _plot_stacked_category_artifact(
        self,
        artifact: pl.DataFrame,
        intersection_order: list[str],
        output_path: Path,
        *,
        title: str,
    ) -> None:
        work = artifact.filter(pl.col("intersection").is_in(intersection_order))
        if work.is_empty():
            return

        pdf = work.to_pandas()
        pivot = (
            pdf.pivot_table(index="intersection", columns="category", values="fraction_percent", fill_value=0)
            .reindex(intersection_order)
            .fillna(0)
        )
        if pivot.empty:
            return

        width = max(8.5, 3.8 + len(pivot) * 0.34)
        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(width, 4.8))
            bottom = [0.0] * len(pivot)
            colors = self._category_colors(list(pivot.columns))
            x_positions = list(range(len(pivot)))
            for category in pivot.columns:
                values = pivot[category].to_list()
                ax.bar(x_positions, values, bottom=bottom, label=category, color=colors[category], width=0.72)
                bottom = [old + value for old, value in zip(bottom, values)]
            ax.set_ylabel("Entities in intersection (%)")
            ax.set_xlabel("UpSet intersection")
            ax.set_title(title)
            ax.set_ylim(0, 100)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(pivot.index, rotation=45, ha="right")
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
            fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_single_intersection_category_artifacts(
        self,
        artifact: pl.DataFrame,
        membership: pl.DataFrame,
        subset_items: list[tuple[tuple[str, ...], int]],
        detail_dir: Path,
        *,
        kind: str,
    ) -> None:
        for idx, (members, _count) in enumerate(subset_items, start=1):
            intersection = " & ".join(members)
            safe_intersection = self._safe("_".join(members))[:120]
            one_dir = detail_dir / f"{idx:02d}_{safe_intersection}"
            one_dir.mkdir(parents=True, exist_ok=True)
            membership.filter(pl.col("intersection") == intersection).write_csv(one_dir / "entities.csv")
            one = artifact.filter(pl.col("intersection") == intersection)
            if one.is_empty():
                continue
            one.write_csv(one_dir / f"{kind}_composition.csv")
            self._plot_single_category_artifact(
                one,
                one_dir / f"{kind}_composition.png",
                title=f"{self._label(kind)}: {intersection}",
            )

    def _plot_single_category_artifact(self, artifact: pl.DataFrame, output_path: Path, *, title: str) -> None:
        pdf = artifact.sort("n_entities", descending=True).to_pandas()
        if pdf.empty:
            return
        colors = self._category_colors(pdf["category"].to_list())
        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=(6.8, max(2.8, 0.34 * len(pdf) + 1.4)))
            ax.barh(pdf["category"], pdf["fraction_percent"], color=[colors[item] for item in pdf["category"]])
            ax.invert_yaxis()
            ax.set_xlabel("Entities in intersection (%)")
            ax.set_title(title)
            ax.set_xlim(0, 100)
            ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            for y, row in enumerate(pdf.itertuples(index=False)):
                ax.text(
                    float(row.fraction_percent) + 1.0,
                    y,
                    f"{row.fraction_percent:.1f}% ({int(row.n_entities)})",
                    va="center",
                    fontsize=8,
                )
            fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _category_colors(categories: list[str]) -> dict[str, str]:
        palette = [
            "#3f6f8f",
            "#c06c84",
            "#6c8f5f",
            "#d29d4f",
            "#7d6aa5",
            "#4f9a9a",
            "#8a8a8a",
            "#b56576",
        ]
        return {category: palette[idx % len(palette)] for idx, category in enumerate(categories)}

    @staticmethod
    def _safe(value: object) -> str:
        return (
            str(value)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(".", "_")
            .replace(":", "_")
            .replace("&", "and")
        )


class UpSetPlotRawFilePresencePlotter(MatplotlibRawFilePresenceUpSetPlotter):
    name = "raw_file_presence_upsetplot"

    def plot(
        self,
        df: pl.DataFrame,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        try:
            from upsetplot import UpSet, from_contents
        except ImportError as exc:
            raise ImportError(
                "UpSetPlotRawFilePresencePlotter requires the 'upsetplot' package. "
                "Install it with: pip install upsetplot"
            ) from exc
        self._patch_upsetplot_matrix_for_pandas3(UpSet)

        set_col = self._resolve_set_col(df)
        if set_col is None or self.entity_col not in df.columns:
            return None

        contents = self._build_contents(df, set_col)
        if not contents:
            return None

        upset_data = from_contents(contents)
        with plt.rc_context(self.rc_params()):
            fig = plt.figure(figsize=self.figsize)
            UpSet(
                upset_data,
                subset_size="count",
                show_counts=False,
                sort_by="cardinality",
                min_subset_size=self.min_subset_size,
            ).plot(fig=fig)
            fig.suptitle(f"{self.entity_col} overlap across {set_col}")
        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename=self._output_filename("upsetplot"),
            dpi=150,
            bbox_inches="tight",
        )

    def _build_contents(self, df: pl.DataFrame, set_col: str) -> dict[str, set[str]]:
        rows = (
            df.select([set_col, self.entity_col])
            .drop_nulls()
            .unique()
            .group_by(set_col)
            .agg(pl.col(self.entity_col).cast(pl.Utf8).alias("entities"))
        )
        contents: dict[str, set[str]] = {}
        for row in rows.iter_rows(named=True):
            contents[str(row[set_col])] = set(row["entities"])
        return contents

    @staticmethod
    def _patch_upsetplot_matrix_for_pandas3(upset_cls: type) -> None:
        if getattr(upset_cls, "_maplot_pandas3_matrix_patch", False):
            return

        def plot_matrix(self, ax):  # type: ignore[no-untyped-def]
            import numpy as np
            import pandas as pd

            ax = self._reorient(ax)
            data = self.intersections
            n_cats = data.index.nlevels
            inclusion = data.index.to_frame().values

            styles = [
                [
                    self.subset_styles[i]
                    if inclusion[i, j]
                    else {"facecolor": self._other_dots_color, "linewidth": 0}
                    for j in range(n_cats)
                ]
                for i in range(len(data))
            ]
            styles = sum(styles, [])
            style_columns = {
                "facecolor": "facecolors",
                "edgecolor": "edgecolors",
                "linewidth": "linewidths",
                "linestyle": "linestyles",
                "hatch": "hatch",
            }
            styles = (
                pd.DataFrame(styles)
                .reindex(columns=style_columns.keys())
                .astype(
                    {
                        "facecolor": "O",
                        "edgecolor": "O",
                        "linewidth": float,
                        "linestyle": "O",
                        "hatch": "O",
                    }
                )
            )
            styles["linewidth"] = styles["linewidth"].fillna(1)
            styles["facecolor"] = styles["facecolor"].fillna(self._facecolor)
            styles["edgecolor"] = styles["edgecolor"].fillna(styles["facecolor"])
            styles["linestyle"] = styles["linestyle"].fillna("solid")
            del styles["hatch"]

            x = np.repeat(np.arange(len(data)), n_cats)
            y = np.tile(np.arange(n_cats), len(data))
            s = (self._element_size * 0.35) ** 2 if self._element_size is not None else 200
            ax.scatter(
                *self._swapaxes(x, y),
                s=s,
                zorder=10,
                **styles.rename(columns=style_columns),
            )

            if self._with_lines:
                idx = np.flatnonzero(inclusion)
                line_data = pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
                colors = pd.Series(
                    [
                        style.get("edgecolor", style.get("facecolor", self._facecolor))
                        for style in self.subset_styles
                    ],
                    name="color",
                )
                line_data = line_data.join(colors)
                ax.vlines(
                    line_data.index.values,
                    line_data["min"],
                    line_data["max"],
                    lw=2,
                    colors=line_data["color"],
                    zorder=5,
                )

            tick_axis = ax.yaxis
            tick_axis.set_ticks(np.arange(n_cats))
            tick_axis.set_ticklabels(data.index.names, rotation=0 if self._horizontal else -90)
            ax.xaxis.set_visible(False)
            ax.tick_params(axis="both", which="both", length=0)
            if not self._horizontal:
                ax.yaxis.set_ticks_position("top")
            ax.set_frame_on(False)
            ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
            ax.grid(False)

        upset_cls.plot_matrix = plot_matrix
        upset_cls._maplot_pandas3_matrix_patch = True


class UpSetPlotRunPresencePlotter(UpSetPlotRawFilePresencePlotter):
    """Backward-compatible name for `UpSetPlotRawFilePresencePlotter`."""

    name = "upset_presence_upsetplot"


class UpSetPlotSearchPresencePlotter(MatplotlibSearchPresenceUpSetPlotter):
    """upsetplot backend for entity overlap across searches in one experiment."""

    name = "search_presence_upsetplot"
    scope = "multi_search"

    def plot_many(
        self,
        searches: dict[str, pl.DataFrame],
        output_dir: str | Path | None = None,
    ) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        try:
            from upsetplot import UpSet, from_contents
        except ImportError as exc:
            raise ImportError(
                "UpSetPlotSearchPresencePlotter requires the 'upsetplot' package. "
                "Install it with: pip install upsetplot"
            ) from exc
        UpSetPlotRawFilePresencePlotter._patch_upsetplot_matrix_for_pandas3(UpSet)

        presence_df = self._build_search_presence(searches)
        if presence_df.is_empty():
            return None

        contents = self._build_contents(presence_df, "search")
        if not contents:
            return None

        upset_data = from_contents(contents)
        with plt.rc_context(self.rc_params()):
            fig = plt.figure(figsize=self.figsize)
            UpSet(
                upset_data,
                subset_size="count",
                show_counts=False,
                sort_by="cardinality",
                min_subset_size=self.min_subset_size,
            ).plot(fig=fig)
            fig.suptitle(f"{self.entity_col} overlap across searches")
        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename=self._output_filename("upsetplot"),
            dpi=150,
            bbox_inches="tight",
        )

    def _build_contents(self, df: pl.DataFrame, set_col: str) -> dict[str, set[str]]:
        rows = (
            df.select([set_col, self.entity_col])
            .drop_nulls()
            .unique()
            .group_by(set_col)
            .agg(pl.col(self.entity_col).cast(pl.Utf8).alias("entities"))
        )
        contents: dict[str, set[str]] = {}
        for row in rows.iter_rows(named=True):
            contents[str(row[set_col])] = set(row["entities"])
        return contents
