from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal
import re

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from matplotlib.figure import Figure

PlotScope = Literal["single_run", "single_search", "multi_search"]
PlotMode = Literal["show", "save", "both", "none"]


@dataclass(frozen=True, slots=True)
class PlotOutputConfig:
    """
    Configure whether plots are shown, saved, both, or skipped.

    Attributes:
        mode: One of "show", "save", "both", or "none".
        formats: File formats saved for every figure. The requested filename
            suffix is always saved; additional formats are written next to it.
        publication_names: Prefix saved figure filenames with inferred
            experiment/context/date tokens when an output path follows the
            standard ``<experiment>/ma_plots/...`` layout.
    """
    mode: PlotMode = "save"
    formats: tuple[str, ...] = ("png", "pdf", "svg")
    publication_names: bool = True
    date_stamp: str | None = None

    @property
    def show(self) -> bool:
        return self.mode in {"show", "both"}

    @property
    def save(self) -> bool:
        return self.mode in {"save", "both"}


@dataclass(frozen=True, slots=True)
class PlotStyleConfig:
    """
    Configure shared Matplotlib styling for plotters.

    Args:
        font_family: Ordered sans-serif font fallback stack used for plot text.
        base_font_size: Default Matplotlib font size.
    """

    font_family: tuple[str, ...] = ("Helvetica", "Arial", "sans-serif")
    base_font_size: int = 11

    def rc_params(self) -> dict[str, object]:
        sans_serif_fonts = [font for font in self.font_family if font != "sans-serif"]
        if "DejaVu Sans" not in sans_serif_fonts:
            sans_serif_fonts.append("DejaVu Sans")
        return {
            "axes.axisbelow": True,
            "axes.edgecolor": "#222222",
            "axes.grid": False,
            "axes.labelcolor": "#111111",
            "axes.labelsize": self.base_font_size,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": self.base_font_size + 1,
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": sans_serif_fonts,
            "font.size": self.base_font_size,
            "legend.frameon": False,
            "legend.fontsize": self.base_font_size - 1,
            "legend.title_fontsize": self.base_font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "xtick.color": "#111111",
            "xtick.labelsize": self.base_font_size - 1,
            "ytick.color": "#111111",
            "ytick.labelsize": self.base_font_size - 1,
        }


class BasePlotter(ABC):
    name: str = "base"
    scope: PlotScope = "single_search"
    output_config: PlotOutputConfig = PlotOutputConfig()
    style_config: PlotStyleConfig = PlotStyleConfig()
    output_dir: str | Path | None = None

    @staticmethod
    def ensure_output_dir(output_dir: str | Path) -> Path:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def resolve_output_dir(self, output_dir: str | Path | None = None) -> Path:
        """Resolve an explicit or configured output directory for this plotter."""
        candidate = output_dir if output_dir is not None else self.output_dir
        if candidate is None:
            raise ValueError(
                "output_dir is required unless a default output directory was configured on the plotter "
                "or provided by the registry/project."
            )
        return Path(candidate).expanduser().resolve()

    def rc_params(self, overrides: dict[str, object] | None = None) -> dict[str, object]:
        """Return plotter-level Matplotlib rcParams with optional local overrides."""
        params = self.style_config.rc_params()
        if overrides is not None:
            params.update(overrides)
        return params

    def finalize_figure(
        self,
        fig: Figure,
        *,
        output_dir: str | Path,
        filename: str | None,
        dpi: int = 150,
        bbox_inches: str | None = None,
    ) -> Path | None:
        output_config = self.output_config
        saved_path: Path | None = None

        if output_config.save:
            if filename is None:
                raise ValueError("filename is required when save is enabled")
            output_dir = self.ensure_output_dir(output_dir)
            saved_path = self._publication_path(output_dir, filename)
            save_dpi = max(int(dpi), 300)
            save_bbox = "tight" if bbox_inches is None else bbox_inches
            fig.savefig(saved_path, dpi=save_dpi, bbox_inches=save_bbox)

            requested_format = saved_path.suffix.lstrip(".").lower()
            for fmt in output_config.formats:
                normalized = fmt.lower().lstrip(".")
                if not normalized or normalized == requested_format:
                    continue
                fig.savefig(
                    saved_path.with_suffix(f".{normalized}"),
                    dpi=save_dpi,
                    bbox_inches=save_bbox,
                )

        if output_config.show:
            plt.show()

        plt.close(fig)
        return saved_path

    def _publication_path(self, output_dir: Path, filename: str) -> Path:
        if not self.output_config.publication_names:
            return output_dir / filename

        original = Path(filename)
        suffix = original.suffix or ".png"
        stem = original.stem
        prefix_parts = self._publication_prefix_parts(output_dir)
        if not prefix_parts:
            return output_dir / filename

        prefix = "_".join(prefix_parts)
        safe_stem = self._safe_filename_part(stem)
        if safe_stem.startswith(f"{prefix}_"):
            return output_dir / f"{safe_stem}{suffix}"
        return output_dir / f"{prefix}_{safe_stem}{suffix}"

    def _publication_prefix_parts(self, output_dir: Path) -> list[str]:
        parts = list(output_dir.resolve().parts)
        if "ma_plots" not in parts:
            return []

        ma_idx = parts.index("ma_plots")
        if ma_idx == 0:
            return []

        experiment = self._safe_filename_part(parts[ma_idx - 1])
        date_stamp = self.output_config.date_stamp or date.today().isoformat()
        context = [
            self._safe_filename_part(part)
            for part in parts[ma_idx + 1 :]
            if part and part not in {".", ".."}
        ]
        return [experiment, self._safe_filename_part(date_stamp), *context]

    @staticmethod
    def _safe_filename_part(value: object) -> str:
        text = str(value).strip()
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_.")
        return text or "plot"

    def save_csv(
        self,
        df: pd.DataFrame,
        *,
        output_dir: str | Path,
        filename: str,
        index: bool = False,
    ) -> Path | None:
        if not self.output_config.save:
            return None

        output_dir = self.ensure_output_dir(output_dir)
        out_path = output_dir / filename
        df.to_csv(out_path, index=index)
        return out_path

    @abstractmethod
    def plot(
        self,
        df: pl.DataFrame,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> Path | None:
        raise NotImplementedError


class SingleRunPlotter(BasePlotter, ABC):
    """Plotter for one raw file/sample from the DIA-NN `Run` column."""

    scope: PlotScope = "single_run"


class SingleRawFilePlotter(SingleRunPlotter, ABC):
    """Preferred name for plotters that operate on one raw file/sample."""


class SingleSearchPlotter(BasePlotter, ABC):
    scope: PlotScope = "single_search"


class MultiSearchPlotter(BasePlotter, ABC):
    scope: PlotScope = "multi_search"

    def plot(
        self,
        df: pl.DataFrame,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> Path | None:
        raise NotImplementedError("Multi-search plotters use plot_many().")

    @abstractmethod
    def plot_many(
        self,
        searches: dict[str, pl.DataFrame],
        output_dir: str | Path | None = None,
    ) -> Path | None:
        raise NotImplementedError


class MultiRunPlotter(SingleSearchPlotter, ABC):
    """Backward-compatible name for plotters that compare raw files in one search."""


class MultiRawFilePlotter(SingleSearchPlotter, ABC):
    """Preferred name for plotters that compare raw files in one search."""
