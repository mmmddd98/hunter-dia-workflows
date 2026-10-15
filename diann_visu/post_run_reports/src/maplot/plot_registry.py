from pathlib import Path

import polars as pl

from .base_plotter import BasePlotter, MultiSearchPlotter


class PlotRegistry:
    def __init__(self, default_output_dir: str | Path | None = None) -> None:
        """Create a registry with an optional default output directory for plots."""
        self._plotters: list[BasePlotter] = []
        self.default_output_dir = Path(default_output_dir).expanduser().resolve() if default_output_dir is not None else None

    def register(self, plotter: BasePlotter) -> None:
        self._plotters.append(plotter)

    def run_all(self, df: pl.DataFrame, output_dir: str | Path | None = None) -> list[Path]:
        resolved_output_dir = self._resolve_output_dir(output_dir)
        # Keep the historical folder name for compatibility; contents are per raw file.
        single_run_dir = resolved_output_dir / "single_run"
        single_search_dir = resolved_output_dir / "single_search"
        single_run_dir.mkdir(parents=True, exist_ok=True)
        single_search_dir.mkdir(parents=True, exist_ok=True)

        output_paths: list[Path] = []

        output_paths.extend(self._run_single_search_plotters(df, single_search_dir))
        output_paths.extend(self._run_single_run_plotters(df, single_run_dir))
        return output_paths

    def run_multi_search(self, searches: dict[str, pl.DataFrame], output_dir: str | Path | None = None) -> list[Path]:
        resolved_output_dir = self._resolve_output_dir(output_dir)
        multi_search_dir = resolved_output_dir / "multi_search"

        return self._run_multi_search_plotters(searches, multi_search_dir)

    def run_experiment(self, searches: dict[str, pl.DataFrame], output_dir: str | Path | None = None) -> list[Path]:
        """Run plotters with experiment-oriented default folders.

        Multi-search plotters are written below ``<output_dir>/<plotter.name>/``.
        Single-search plotters are written directly below
        ``<output_dir>/<search_name>/``. Single-raw-file plotters are skipped in
        experiment mode.
        If no output directory is configured, ``./plots`` is used.
        """
        resolved_output_dir = self._resolve_experiment_output_dir(output_dir)
        output_paths: list[Path] = []

        output_paths.extend(self._run_experiment_multi_search_plotters(searches, resolved_output_dir))
        for search_name, df in searches.items():
            search_dir = resolved_output_dir / self._safe_name(search_name)
            output_paths.extend(self._run_single_search_plotters(df, search_dir))
        return output_paths

    def _run_multi_search_plotters(self, searches: dict[str, pl.DataFrame], output_dir: str | Path) -> list[Path]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        for plotter in self._plotters:
            if plotter.scope != "multi_search":
                continue
            if not isinstance(plotter, MultiSearchPlotter):
                raise TypeError(f"{plotter.name} has multi_search scope but is not a MultiSearchPlotter")
            out = plotter.plot_many(searches, output_dir)
            if out is not None:
                output_paths.append(out)
        return output_paths

    def _run_experiment_multi_search_plotters(
        self,
        searches: dict[str, pl.DataFrame],
        output_dir: str | Path,
    ) -> list[Path]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_paths: list[Path] = []
        for plotter in self._plotters:
            if plotter.scope != "multi_search":
                continue
            if not isinstance(plotter, MultiSearchPlotter):
                raise TypeError(f"{plotter.name} has multi_search scope but is not a MultiSearchPlotter")
            plotter_dir = output_dir / self._safe_name(plotter.name)
            out = plotter.plot_many(searches, plotter_dir)
            if out is not None:
                output_paths.append(out)
        return output_paths

    def _resolve_output_dir(self, output_dir: str | Path | None) -> Path:
        candidate = output_dir if output_dir is not None else self.default_output_dir
        if candidate is None:
            raise ValueError(
                "output_dir is required unless a default output directory was configured on the registry."
            )
        return Path(candidate).expanduser().resolve()

    def _resolve_experiment_output_dir(self, output_dir: str | Path | None) -> Path:
        candidate = output_dir if output_dir is not None else self.default_output_dir
        if candidate is None:
            candidate = "./plots"
        return Path(candidate).expanduser().resolve()

    def _run_single_search_plotters(self, df: pl.DataFrame, output_dir: str | Path) -> list[Path]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_paths: list[Path] = []
        for plotter in self._plotters:
            if plotter.scope != "single_search":
                continue
            out = plotter.plot(df, output_dir)
            if out is not None:
                output_paths.append(out)
        return output_paths

    def _run_single_run_plotters(self, df: pl.DataFrame, output_dir: str | Path) -> list[Path]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_paths: list[Path] = []
        if "Run" not in df.columns:
            return output_paths

        raw_file_names = df.select("Run").unique(maintain_order=True).get_column("Run").to_list()
        for raw_file_name in raw_file_names:
            raw_file_df = df.filter(pl.col("Run") == raw_file_name)
            raw_file_safe = self._safe_name(raw_file_name)
            run_dir = output_dir / raw_file_safe
            run_dir.mkdir(parents=True, exist_ok=True)

            for plotter in self._plotters:
                if plotter.scope != "single_run":
                    continue
                out = plotter.plot(raw_file_df, run_dir, run_name=str(raw_file_name))
                if out is not None:
                    output_paths.append(out)

        return output_paths

    @staticmethod
    def _safe_name(value: object) -> str:
        return str(value).replace("/", "_").replace(" ", "_")
