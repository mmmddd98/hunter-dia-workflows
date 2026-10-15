from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


QuantGroup = Literal["baseline", "tuned"]
PREFERRED_REPORT_FILES = ("final_report-first-pass.parquet", "final_report.parquet")


@dataclass(frozen=True)
class ExperimentSearch:
    """One search/quant result that belongs to an experiment."""

    name: str
    report_file: str | Path
    quant_group: QuantGroup

    def __post_init__(self) -> None:
        report_file = Path(self.report_file).expanduser().resolve()
        if not report_file.exists() or not report_file.is_file():
            raise FileNotFoundError(f"Search report file does not exist: {report_file}")
        object.__setattr__(self, "report_file", report_file)

    @property
    def search_dir(self) -> Path:
        """Return the directory that contains the DIA-NN final report."""
        return self.report_file.parent

    @property
    def run_dir(self) -> Path:
        """Backward-compatible alias for `search_dir`."""
        return self.search_dir


ExperimentRun = ExperimentSearch


@dataclass(frozen=True, init=False)
class Experiment:
    """Resolved experiment layout with one or more searches/quant results."""

    name: str
    root_dir: str | Path
    searches: dict[str, ExperimentSearch]
    fasta_file: str | Path | None = None

    def __init__(
        self,
        *,
        name: str,
        root_dir: str | Path,
        searches: dict[str, ExperimentSearch] | None = None,
        fasta_file: str | Path | None = None,
        runs: dict[str, ExperimentSearch] | None = None,
    ) -> None:
        """Create an experiment from searches.

        Args:
            name: Experiment label.
            root_dir: Experiment output directory.
            searches: Mapping from search label to search metadata.
            fasta_file: Optional experiment FASTA path.
            runs: Backward-compatible alias for `searches`.
        """
        if searches is None:
            searches = runs
        elif runs is not None and runs != searches:
            raise ValueError("Provide either searches or runs, not conflicting values for both")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "root_dir", root_dir)
        object.__setattr__(self, "searches", searches)
        object.__setattr__(self, "fasta_file", fasta_file)
        self.__post_init__()

    def __post_init__(self) -> None:
        root_dir = Path(self.root_dir).expanduser().resolve()
        if not root_dir.exists() or not root_dir.is_dir():
            raise FileNotFoundError(f"Experiment root does not exist: {root_dir}")
        if not self.searches:
            raise ValueError("Experiment must contain at least one search")

        normalized_searches: dict[str, ExperimentSearch] = {}
        for search_name, search in self.searches.items():
            if search_name != search.name:
                raise ValueError(
                    f"Search mapping key '{search_name}' does not match search name '{search.name}'"
                )
            normalized_searches[search_name] = search

        fasta_file = None
        if self.fasta_file is not None:
            fasta_file = Path(self.fasta_file).expanduser().resolve()
            if not fasta_file.exists() or not fasta_file.is_file():
                raise FileNotFoundError(f"Experiment FASTA file does not exist: {fasta_file}")

        object.__setattr__(self, "root_dir", root_dir)
        object.__setattr__(self, "searches", normalized_searches)
        object.__setattr__(self, "fasta_file", fasta_file)

    @property
    def runs(self) -> dict[str, ExperimentSearch]:
        """Backward-compatible alias for `searches`."""
        return self.searches

    @classmethod
    def from_root(cls, root_dir: str | Path) -> "Experiment":
        """Build an experiment model from an experiment root directory."""
        resolved_root = Path(root_dir).expanduser().resolve()
        quant_dir = resolved_root / "05_quant"
        if not quant_dir.exists() or not quant_dir.is_dir():
            raise FileNotFoundError(f"Experiment quant directory does not exist: {quant_dir}")

        searches = cls._discover_searches(quant_dir)
        fasta_file = cls._resolve_unique_fasta(resolved_root)
        return cls(name=resolved_root.name, root_dir=resolved_root, searches=searches, fasta_file=fasta_file)

    @classmethod
    def from_diann_output(
        cls,
        *,
        experiment: str,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_3_2",
    ) -> "Experiment":
        """Build an experiment from the standard DIA-NN output layout."""
        root_dir = Path(diann_out_root).expanduser().resolve() / pipeline_version / experiment
        return cls.from_root(root_dir)

    def get_search(self, name: str) -> ExperimentSearch:
        """Return one named search/quant result from the experiment."""
        try:
            return self.searches[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.searches))
            raise KeyError(f"Unknown search '{name}'. Available searches: {available}") from exc

    def get_run(self, name: str) -> ExperimentSearch:
        """Backward-compatible alias for `get_search`."""
        return self.get_search(name)

    def select_search(
        self,
        *,
        quant_group: QuantGroup = "baseline",
        tuned_name: str | None = None,
        tuned_profile: str | None = None,
        use_fr: bool | None = None,
        lr: str | float | None = None,
    ) -> ExperimentSearch:
        """Resolve one search using the same naming rules as the existing config."""
        if quant_group == "baseline":
            return self.get_search("baseline")

        if tuned_name is not None:
            return self.get_search(tuned_name)

        if tuned_profile is None:
            raise ValueError("tuned_profile is required for tuned searches")
        if use_fr is None:
            raise ValueError("use_fr must be provided for tuned searches")
        if lr is None:
            raise ValueError("lr must be provided for tuned searches")

        lr_token = self._normalize_lr_token(lr)
        fr_part = "fr_" if use_fr else ""
        return self.get_search(f"{tuned_profile}_{fr_part}lr_{lr_token}")

    def select_run(
        self,
        *,
        quant_group: QuantGroup = "baseline",
        tuned_name: str | None = None,
        tuned_profile: str | None = None,
        use_fr: bool | None = None,
        lr: str | float | None = None,
    ) -> ExperimentSearch:
        """Backward-compatible alias for `select_search`."""
        return self.select_search(
            quant_group=quant_group,
            tuned_name=tuned_name,
            tuned_profile=tuned_profile,
            use_fr=use_fr,
            lr=lr,
        )

    @property
    def search_names(self) -> list[str]:
        """Return all search labels in deterministic order."""
        return sorted(self.searches)

    @property
    def run_names(self) -> list[str]:
        """Backward-compatible alias for `search_names`."""
        return self.search_names

    @staticmethod
    def _normalize_lr_token(lr: str | float) -> str:
        token = str(lr).strip()
        if not token:
            raise ValueError("lr must not be empty")
        if "p" in token:
            return token
        return token.replace(".", "p")

    @staticmethod
    def _discover_searches(quant_dir: str | Path) -> dict[str, ExperimentSearch]:
        quant_dir = Path(quant_dir).expanduser().resolve()
        searches: dict[str, ExperimentSearch] = {}

        baseline_report = preferred_report_path(quant_dir / "baseline")
        if baseline_report is not None:
            searches["baseline"] = ExperimentSearch(
                name="baseline",
                report_file=baseline_report,
                quant_group="baseline",
            )

        tuned_root = quant_dir / "tuned"
        if tuned_root.exists() and tuned_root.is_dir():
            for run_dir in sorted(tuned_root.iterdir()):
                report_file = preferred_report_path(run_dir)
                if run_dir.is_dir() and report_file is not None:
                    searches[run_dir.name] = ExperimentSearch(
                        name=run_dir.name,
                        report_file=report_file,
                        quant_group="tuned",
                    )

        if not searches:
            raise FileNotFoundError(f"No DIA-NN report parquet files found under {quant_dir}")

        return searches

    _discover_runs = _discover_searches

    @staticmethod
    def _resolve_unique_fasta(root_dir: str | Path) -> Path | None:
        root_dir = Path(root_dir).expanduser().resolve()
        candidates = sorted(root_dir.glob("*.fasta")) + sorted(root_dir.glob("*.fa"))
        if not candidates:
            return None
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(f"Multiple FASTA files found in {root_dir}: {names}")
        return candidates[0].resolve()


def preferred_report_path(search_dir: Path) -> Path | None:
    for filename in PREFERRED_REPORT_FILES:
        candidate = search_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None
