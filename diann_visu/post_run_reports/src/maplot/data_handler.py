from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from .config import PlotProjectConfig
from .data_loader import ResultDataLoader
from .diann_adapter import DiannAdapter
from .experiment import Experiment
from .termini_annotator import TerminiAnnotator

DataKind = Literal["raw", "adapted", "termini"]


@dataclass
class PlotDataBundle:
    """Container for optional raw/adapted/termini tables of one search."""

    name: str
    raw: pl.DataFrame | None = None
    adapted: pl.DataFrame | None = None
    termini: pl.DataFrame | None = None

    def get(self, kind: DataKind) -> pl.DataFrame:
        """Return one loaded table from the bundle.

        Args:
            kind: Data stage to retrieve ("raw", "adapted", or "termini").
        """
        df = getattr(self, kind)
        if df is None:
            raise ValueError(f"Data kind is not loaded in bundle: {kind}")
        return df


class SearchDataHandler:
    """
    Load and cache all data tables that belong to one search/quant result.

    The handler keeps raw DIA-NN reports, DIA-NN-adapted peptide matrices, and
    termini-annotated tables separate while exposing one small access point for
    notebooks, plot projects, and later multi-search comparisons.

    Args:
        name:
            Logical search label used for cache/output subfolders.
            Use short stable names like "baseline", "tuned_v2", or search IDs.
        raw_file:
            Path to DIA-NN report (tsv/csv/parquet). Required for `raw()` and
            as source for `adapted()` when no adapted file is provided.
        adapted_file:
            Optional path to a precomputed adapted table. Set this when you
            already exported adapted data and want to skip re-adaptation.
        termini_file:
            Optional path to a precomputed termini-annotated table. Set this
            when annotation was already run and should be reused.
        output_dir:
            Base output folder used by `save_adapted()` / `save_termini()`.
            If omitted, defaults to `<raw_file.parent>/plots_and_data`.
        qvalue_threshold:
            Q-value filter applied when loading raw DIA-NN data. Keep default
            `0.01` for standard 1% FDR unless your analysis needs another cut.
    """

    def __init__(
        self,
        *,
        name: str = "search",
        raw_file: str | Path | None = None,
        fasta_file: str | Path | None = None,
        adapted_file: str | Path | None = None,
        termini_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        qvalue_threshold: float = 0.01,
    ) -> None:
        """
        Initialize one search-level data workflow.

        Args:
            name: Label for cache/output subfolders (for example "baseline").
            raw_file: DIA-NN report path (tsv/csv/parquet); source for `raw()`.
            fasta_file: Path to FASTA file for termini annotation. Required for `termini()`.
            adapted_file: Optional precomputed adapted table to reuse.
            termini_file: Optional precomputed termini table to reuse.
            output_dir: Base output folder; defaults to `<raw_file.parent>/plots_and_data`.
            qvalue_threshold: Filter for `Q.Value` in raw DIA-NN data.
        """
        self.name = name
        self.raw_file = self._optional_existing_path(raw_file)
        self.adapted_file = self._optional_existing_path(adapted_file)
        self.termini_file = self._optional_existing_path(termini_file)
        self.fasta_file = self._optional_existing_path(fasta_file)
        self.output_dir = self._resolve_output_dir(raw_file=self.raw_file, output_dir=output_dir)
        self.qvalue_threshold = qvalue_threshold
        self._cache: dict[DataKind, pl.DataFrame] = {}

    @classmethod
    def from_config(
        cls,
        config: PlotProjectConfig,
        *,
        name: str = "search",
        adapted_file: str | Path | None = None,
        termini_file: str | Path | None = None,
    ) -> "SearchDataHandler":
        """Build a handler from an existing project config.

        Args:
            config: Pre-resolved input/output settings.
            name: Label for cache and output subfolders.
            adapted_file: Optional precomputed adapted table path.
            termini_file: Optional precomputed termini table path.
        """
        return cls(
            name=name,
            raw_file=config.input_file,
            adapted_file=adapted_file,
            termini_file=termini_file,
            output_dir=config.output_dir,
            qvalue_threshold=config.qvalue_threshold,
        )

    @classmethod
    def from_experiment(
        cls,
        *,
        experiment: str,
        output_dir: str | Path | None = None,
        name: str | None = None,
        qvalue_threshold: float = 0.01,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_5_0",
        quant_group: Literal["baseline", "tuned"] = "baseline",
        tuned_name: str | None = None,
        tuned_profile: str | None = None,
        use_fr: bool | None = None,
        lr: str | float | None = None,
    ) -> "SearchDataHandler":
        """Build a handler by resolving DIA-NN files from experiment metadata.

        Args:
            experiment: Experiment folder name under the pipeline output.
            output_dir: Optional base output directory override.
            name: Optional handler label; defaults to tuned name/group.
            qvalue_threshold: Q-value cut for raw loading.
            diann_out_root: Root directory containing DIA-NN pipeline outputs.
            pipeline_version: Pipeline version folder token.
            quant_group: Select baseline or tuned quant outputs.
            tuned_name: Explicit tuned search directory name.
            tuned_profile: Tuned profile token (used when `tuned_name` is not set).
            use_fr: Tuned search FR flag used in path resolution.
            lr: Tuned search learning-rate token used in path resolution.
        """
        config = PlotProjectConfig.from_experiment(
            experiment=experiment,
            output_dir=output_dir,
            qvalue_threshold=qvalue_threshold,
            diann_out_root=diann_out_root,
            pipeline_version=pipeline_version,
            quant_group=quant_group,
            tuned_name=tuned_name,
            tuned_profile=tuned_profile,
            use_fr=use_fr,
            lr=lr,
        )
        experiment_model = Experiment.from_diann_output(
            experiment=experiment,
            diann_out_root=diann_out_root,
            pipeline_version=pipeline_version,
        )
        selected_search = experiment_model.select_search(
            quant_group=quant_group,
            tuned_name=tuned_name,
            tuned_profile=tuned_profile,
            use_fr=use_fr,
            lr=lr,
        )
        handler_name = name or tuned_name or quant_group
        return cls(
            name=handler_name,
            raw_file=selected_search.report_file,
            fasta_file=experiment_model.fasta_file,
            output_dir=config.output_dir,
            qvalue_threshold=config.qvalue_threshold,
        )

    def raw(self, *, force: bool = False) -> pl.DataFrame:
        """Load and cache the raw DIA-NN table.

        Args:
            force: Reload from source even if cached.
        """
        if not force and "raw" in self._cache:
            return self._cache["raw"]
        if self.raw_file is None:
            raise ValueError("raw_file is required to load raw data")

        df = ResultDataLoader(self.raw_file, qvalue_threshold=self.qvalue_threshold).load()
        self._cache["raw"] = df
        return df

    def adapted(
        self,
        *,
        adapter: DiannAdapter | None = None,
        force: bool = False,
        prefer_file: bool = True,
    ) -> pl.DataFrame:
        """Load or build the adapted peptide matrix.

        Args:
            adapter: Adapter instance used when adaptation is computed.
            force: Recompute/reload even if cached.
            prefer_file: Prefer `adapted_file` when available.
        """
        return self._materialize_adapted(
            adapter=adapter,
            force=force,
            prefer_file=prefer_file,
        )

    def termini(
        self,
        *,
        annotator: TerminiAnnotator | None = None,
        fasta_path: str | Path | None = None,
        force: bool = False,
        prefer_file: bool = True,
    ) -> pl.DataFrame:
        """Load or compute termini annotations on adapted data.

        Args:
            annotator: `TerminiAnnotator` used when annotation is computed.
            fasta_path: FASTA path required for on-the-fly annotation.
            force: Recompute/reload even if cached.
            prefer_file: Prefer `termini_file` when available.
        """
        return self._materialize_termini(
            annotator=annotator,
            fasta_path=fasta_path,
            force=force,
            prefer_file=prefer_file,
        )

    def get(self, kind: DataKind, **kwargs) -> pl.DataFrame:
        """Dispatch to `raw()`, `adapted()`, or `termini()` by kind.

        Args:
            kind: Data stage to retrieve.
            **kwargs: Forwarded to the selected loader method.
        """
        if kind == "raw":
            return self.raw(**kwargs)
        if kind == "adapted":
            return self.adapted(**kwargs)
        if kind == "termini":
            return self.termini(**kwargs)
        raise ValueError(f"Unsupported data kind: {kind}")

    def bundle(
        self,
        *,
        include_raw: bool = True,
        include_adapted: bool = False,
        include_termini: bool = False,
        adapter: DiannAdapter | None = None,
        annotator: TerminiAnnotator | None = None,
        fasta_path: str | Path | None = None,
    ) -> PlotDataBundle:
        """Build a bundle with selected data stages.

        Args:
            include_raw: Include raw DIA-NN table.
            include_adapted: Include adapted matrix.
            include_termini: Include termini-annotated table.
            adapter: Optional adapter for computed adapted data.
            annotator: Optional annotator for computed termini data.
            fasta_path: FASTA path for computed termini data.
        """
        return PlotDataBundle(
            name=self.name,
            raw=self.raw() if include_raw else None,
            adapted=self.adapted(adapter=adapter) if include_adapted else None,
            termini=self.termini(annotator=annotator, fasta_path=fasta_path) if include_termini else None,
        )

    def save_adapted(
        self,
        path: str | Path | None = None,
        *,
        adapter: DiannAdapter | None = None,
        force: bool = False,
    ) -> Path:
        """Persist adapted data and update `self.adapted_file`.

        Args:
            path: Optional explicit output path; otherwise default under output_dir.
            adapter: Optional adapter used for recomputation.
            force: Recompute adapted table before saving.
        """
        out_path = Path(path).expanduser().resolve() if path is not None else self._default_output_path("adapted.parquet")
        df = self._materialize_adapted(
            adapter=adapter,
            force=force,
            prefer_file=False,
        )
        saved = ResultDataLoader.write_table(df, out_path)
        self.adapted_file = saved
        return saved

    def save_termini(
        self,
        path: str | Path | None = None,
        *,
        annotator: TerminiAnnotator,
        fasta_path: str | Path | None = None,
        force: bool = False,
    ) -> Path:
        """Persist termini data and update `self.termini_file`.

        Args:
            path: Optional explicit output path; otherwise default under output_dir.
            annotator: Annotator used for termini computation.
            fasta_path: Optional FASTA override used for termini annotation.
            force: Recompute termini table before saving.
        """
        out_path = Path(path).expanduser().resolve() if path is not None else self._default_output_path("termini.parquet")
        df = self._materialize_termini(
            annotator=annotator,
            fasta_path=fasta_path,
            force=force,
            prefer_file=False,
        )
        saved = ResultDataLoader.write_table(df, out_path)
        self.termini_file = saved
        return saved

    def clear_cache(self, kind: DataKind | None = None) -> None:
        """Clear cached data for one kind or for all kinds.

        Args:
            kind: Optional data kind to clear; clear all when omitted.
        """
        if kind is None:
            self._cache.clear()
        else:
            self._cache.pop(kind, None)

    def _materialize_adapted(
        self,
        *,
        adapter: DiannAdapter | None,
        force: bool,
        prefer_file: bool,
    ) -> pl.DataFrame:
        if not force and "adapted" in self._cache:
            return self._cache["adapted"]
        if prefer_file and self.adapted_file is not None and self.adapted_file.exists():
            df = ResultDataLoader.load_table(self.adapted_file)
        else:
            df = (adapter or DiannAdapter()).adapt(self.raw())

        self._cache["adapted"] = df
        return df

    def _materialize_termini(
        self,
        *,
        annotator: TerminiAnnotator | None,
        fasta_path: str | Path | None,
        force: bool,
        prefer_file: bool,
    ) -> pl.DataFrame:
        if not force and "termini" in self._cache:
            return self._cache["termini"]
        if prefer_file and self.termini_file is not None and self.termini_file.exists():
            df = ResultDataLoader.load_table(self.termini_file)
        else:
            if annotator is None:
                raise ValueError("annotator is required when termini_file is not provided")
            resolved_fasta = self._optional_existing_path(fasta_path) if fasta_path is not None else self.fasta_file
            if resolved_fasta is None:
                raise ValueError("fasta_path is required when termini_file is not provided")
            df = annotator.annotate(
                self._materialize_adapted(
                    adapter=None,
                    force=force,
                    prefer_file=True,
                ),
                resolved_fasta,
            )

        self._cache["termini"] = df
        return df

    @staticmethod
    def _optional_existing_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None

        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Provided path does not exist: {resolved}")
        return resolved

    @staticmethod
    def _resolve_output_dir(
        *,
        raw_file: Path | None,
        output_dir: str | Path | None,
    ) -> Path | None:
        if output_dir is not None:
            return Path(output_dir).expanduser().resolve()
        if raw_file is None:
            return None
        return raw_file.parent / "plots_and_data"

    def _default_output_path(self, filename: str) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir is required when no output path is provided")
        return self.output_dir / "data" / self.name / filename


class MultiSearchDataHandler:
    """Manage multiple named searches/quant results with shared access methods.

    In this package, a *search* is one DIA-NN quant result such as the
    experiment baseline or one tuned result directory. This is intentionally
    distinct from the DIA-NN ``Run`` column inside a report, which usually
    names individual samples/injections.
    """

    def __init__(
        self,
        searches: dict[str, SearchDataHandler],
        *,
        experiment: Experiment | None = None,
    ) -> None:
        """
        Group multiple `SearchDataHandler` instances for comparison workflows.

        Args:
            searches: Mapping from search label to configured handler.
            experiment: Optional resolved experiment metadata when the searches
                come from one experiment folder.
        """
        self.searches = searches
        self.experiment = experiment

    @classmethod
    def from_experiment_files(
        cls,
        *,
        experiment: str,
        output_dir: str | Path | None = None,
        qvalue_threshold: float = 0.01,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_3_2",
        quant_group: Literal["baseline", "tuned", "all"] = "all",
    ) -> "MultiSearchDataHandler":
        """Discover all search files for one experiment and build handlers.

        Args:
            experiment: Experiment folder name under the pipeline output.
            output_dir: Optional shared output directory for all handlers.
            qvalue_threshold: Q-value cut for raw loading.
            diann_out_root: Root directory containing DIA-NN pipeline outputs.
            pipeline_version: Pipeline version folder token.
            quant_group: Search selection ("baseline", "tuned", or "all").
        """
        files = PlotProjectConfig.discover_experiment_files(
            diann_out_root=diann_out_root,
            pipeline_version=pipeline_version,
            experiment=experiment,
            quant_group=quant_group,
        )
        experiment_model = Experiment.from_diann_output(
            experiment=experiment,
            diann_out_root=diann_out_root,
            pipeline_version=pipeline_version,
        )
        searches = {
            name: SearchDataHandler(
                name=name,
                raw_file=path,
                fasta_file=experiment_model.fasta_file,
                output_dir=output_dir,
                qvalue_threshold=qvalue_threshold,
            )
            for name, path in files.items()
        }
        return cls(searches, experiment=experiment_model)

    @property
    def search_names(self) -> list[str]:
        """Return search labels such as ``baseline`` and tuned directory names."""
        return sorted(self.searches)

    def get_search(self, name: str) -> SearchDataHandler:
        """Return the handler for one search/quant result."""
        try:
            return self.searches[name]
        except KeyError as exc:
            available = ", ".join(self.search_names)
            raise KeyError(f"Unknown search '{name}'. Available searches: {available}") from exc

    def get(self, kind: DataKind, **kwargs) -> dict[str, pl.DataFrame]:
        """Return one data kind for all searches.

        Args:
            kind: Data stage to retrieve.
            **kwargs: Forwarded to each per-search handler.
        """
        return {name: handler.get(kind, **kwargs) for name, handler in self.searches.items()}

    def raw(self) -> dict[str, pl.DataFrame]:
        """Return raw tables for all searches."""
        return self.get("raw")

    def adapted(self, **kwargs) -> dict[str, pl.DataFrame]:
        """Return adapted tables for all searches."""
        return self.get("adapted", **kwargs)

    def termini(self, **kwargs) -> dict[str, pl.DataFrame]:
        """Return termini tables for all searches."""
        return self.get("termini", **kwargs)


class ExperimentDataHandler(MultiSearchDataHandler):
    """Data handler for a complete experiment with baseline and tuned searches."""

    @classmethod
    def from_experiment(
        cls,
        *,
        experiment: str,
        output_dir: str | Path | None = None,
        qvalue_threshold: float = 0.01,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_5_0",
        quant_group: Literal["baseline", "tuned", "all"] = "all",
    ) -> "ExperimentDataHandler":
        """Load all selected searches for one experiment.

        With the default ``quant_group="all"``, this discovers the baseline
        search plus every tuned search below ``05_quant/tuned``.
        """
        return cls.from_experiment_files(
            experiment=experiment,
            output_dir=output_dir,
            qvalue_threshold=qvalue_threshold,
            diann_out_root=diann_out_root,
            pipeline_version=pipeline_version,
            quant_group=quant_group,
        )
