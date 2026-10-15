from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from ..base_plotter import PlotOutputConfig, PlotStyleConfig, SingleSearchPlotter


class LabelEfficiencyPlotter(SingleSearchPlotter):
    name = "label_efficiency"

    def __init__(
        self,
        *,
        sequence_col: str = "Modified.Sequence",
        aa: str = "K",
        mod: str = "UniMod:36",
        mode: Literal["fully", "partially"] = "partially",
        figsize: tuple[float, float] = (6, 4),
        output_name: str | None = None,
        summary_name: str | None = None,
        output_config: PlotOutputConfig | None = None,
        style_config: PlotStyleConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Configure label-efficiency counting and output.

        Args:
            sequence_col: Column with modified peptide sequences.
            aa: Amino acid to evaluate (for example "K").
            mod: Modification token pattern (for example "UniMod:36").
            mode: "fully" for residue-level ratio, "partially" for peptide categories.
            figsize: Matplotlib figure size in inches.
            output_name: Optional custom output image filename.
            summary_name: Optional custom summary CSV filename.
            output_config: Save/show behavior shared across plotters.
            style_config: Matplotlib style settings, including plot font family.
            output_dir: Optional default output directory for this plotter instance.
        """
        self.sequence_col = sequence_col
        self.aa = aa
        self.mod = mod
        self.mode = mode
        self.figsize = figsize
        self.output_name = output_name
        self.summary_name = summary_name
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()
        if output_config is not None:
            self.output_config = output_config
        if style_config is not None:
            self.style_config = style_config

    def plot(
        self,
        df: pl.DataFrame | pd.DataFrame,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> Path | None:
        output_dir = self.resolve_output_dir(output_dir)
        if self.sequence_col not in df.columns:
            return None

        summary_df, plot_df, default_title = self.calculate(df)

        with plt.rc_context(self.rc_params()):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.bar(plot_df["category"], plot_df["count"], color="#5f8f7f")
            ax.set_ylabel("Count")
            ax.set_title(default_title)
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()

        self.save_csv(
            summary_df,
            output_dir=output_dir,
            filename=self._summary_filename(),
            index=False,
        )

        return self.finalize_figure(
            fig,
            output_dir=output_dir,
            filename=self._output_filename(),
            dpi=150,
            bbox_inches="tight",
        )

    def calculate(
        self,
        df: pl.DataFrame | pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        work = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        seq = work[self.sequence_col].fillna("").astype(str)

        count_tot_aa = seq.str.count(self.aa)
        count_mod_aa = seq.str.count(fr"{self.aa}\({self.mod}\)")

        if self.mode == "fully":
            total_aa = int(count_tot_aa.sum())
            total_mod_aa = int(count_mod_aa.sum())
            efficiency = (total_mod_aa / total_aa) if total_aa else 0.0

            summary_df = pd.DataFrame(
                {
                    "metric": [
                        f"Total {self.aa}",
                        f"Total {self.aa}({self.mod})",
                        "Efficiency",
                    ],
                    "value": [total_aa, total_mod_aa, efficiency],
                }
            )
            plot_df = pd.DataFrame(
                {
                    "category": [f"{self.aa} modified", f"{self.aa} unmodified"],
                    "count": [total_mod_aa, max(total_aa - total_mod_aa, 0)],
                }
            )
            title = f"Label efficiency ({self.aa}, {self.mod}) - fully"
            return summary_df, plot_df, title

        if self.mode == "partially":
            has_aa = count_tot_aa > 0
            none_mask = has_aa & (count_mod_aa == 0)
            partial_mask = has_aa & (count_mod_aa > 0) & (count_mod_aa < count_tot_aa)
            full_mask = has_aa & (count_mod_aa == count_tot_aa)

            n_has_aa = int(has_aa.sum())
            n_none = int(none_mask.sum())
            n_partial = int(partial_mask.sum())
            n_full = int(full_mask.sum())

            summary_df = pd.DataFrame(
                {
                    "category": [
                        f"Peptides with {self.aa}",
                        "none modified",
                        "partially modified",
                        "fully modified",
                    ],
                    "count": [n_has_aa, n_none, n_partial, n_full],
                    "fraction_of_peptides_with_aa": [
                        1.0 if n_has_aa else 0.0,
                        (n_none / n_has_aa) if n_has_aa else 0.0,
                        (n_partial / n_has_aa) if n_has_aa else 0.0,
                        (n_full / n_has_aa) if n_has_aa else 0.0,
                    ],
                }
            )
            plot_df = summary_df.iloc[1:][["category", "count"]].copy()
            title = f"Label efficiency ({self.aa}, {self.mod}) - partially"
            return summary_df, plot_df, title

        raise ValueError("mode must be either 'partially' or 'fully'")

    def _output_filename(self) -> str:
        if self.output_name is not None:
            return self.output_name
        return f"label_efficiency_{self.aa}_{self._safe_mod()}_{self.mode}.png"

    def _summary_filename(self) -> str:
        if self.summary_name is not None:
            return self.summary_name
        return f"label_efficiency_{self.aa}_{self._safe_mod()}_{self.mode}.csv"

    def _safe_mod(self) -> str:
        return self.mod.replace(":", "_").replace(" ", "_").replace("/", "_")
