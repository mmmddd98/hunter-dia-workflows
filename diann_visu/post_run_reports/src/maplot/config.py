from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PREFERRED_REPORT_FILES = ("final_report-first-pass.parquet", "final_report.parquet")


@dataclass
class PlotProjectConfig:
    """
    Resolved inputs for one plotting search.

    Attributes:
        input_file: DIA-NN report path (resolved, existing file).
        output_dir: Base directory for generated plots/tables.
        qvalue_threshold: Q-value cut used for raw filtering.
    """
    input_file: str | Path
    output_dir: str | Path
    qvalue_threshold: float = 0.01

    def __post_init__(self) -> None:
        self.input_file = Path(self.input_file).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()

    @staticmethod
    def _resolve_existing_file(path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Input file does not exist: {resolved}")
        return resolved

    @staticmethod
    def _preferred_report_path(search_dir: Path) -> Path | None:
        for filename in PREFERRED_REPORT_FILES:
            candidate = search_dir / filename
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def resolve_output_dir(
        input_file: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path:
        if output_dir is not None:
            return Path(output_dir).expanduser().resolve()
        resolved_input = PlotProjectConfig._resolve_existing_file(input_file)
        return resolved_input.parent / "plots_and_data"

    @staticmethod
    def _normalize_lr_token(lr: str | float) -> str:
        token = str(lr).strip()
        if not token:
            raise ValueError("lr must not be empty")

        if "p" in token:
            return token
        return token.replace(".", "p")

    @classmethod
    def resolve_input_file(
        cls,
        input_file: str | Path | None = None,
        *,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_3_2",
        experiment: str | None = None,
        quant_group: Literal["baseline", "tuned"] = "baseline",
        tuned_name: str | None = None,
        tuned_profile: str | None = None,
        use_fr: bool | None = None,
        lr: str | float | None = None,
    ) -> Path:
        if input_file is not None:
            return cls._resolve_existing_file(input_file)

        if not experiment:
            raise ValueError("experiment is required when input_file is not provided")

        base_quant_dir = (
            Path(diann_out_root).expanduser().resolve()
            / pipeline_version
            / experiment
            / "05_quant"
        )

        if quant_group == "baseline":
            candidate = cls._preferred_report_path(base_quant_dir / "baseline")
            if candidate is None:
                candidate = base_quant_dir / "baseline" / "final_report.parquet"
            return cls._resolve_existing_file(candidate)

        tuned_root = base_quant_dir / "tuned"
        if tuned_name:
            candidate = cls._preferred_report_path(tuned_root / tuned_name)
            if candidate is None:
                candidate = tuned_root / tuned_name / "final_report.parquet"
            return cls._resolve_existing_file(candidate)

        if tuned_profile is None:
            raise ValueError("tuned_profile is required for tuned searches (e.g. fast_rt)")
        if use_fr is None:
            raise ValueError("use_fr must be provided for tuned searches")
        if lr is None:
            raise ValueError("lr must be provided for tuned searches")

        lr_token = cls._normalize_lr_token(lr)
        fr_part = "fr_" if use_fr else ""
        tuned_dir_name = f"{tuned_profile}_{fr_part}lr_{lr_token}"
        candidate = cls._preferred_report_path(tuned_root / tuned_dir_name)
        if candidate is None:
            candidate = tuned_root / tuned_dir_name / "final_report.parquet"

        if candidate.exists() and candidate.is_file():
            return candidate

        available = []
        if tuned_root.exists() and tuned_root.is_dir():
            available = sorted(
                p.name
                for p in tuned_root.iterdir()
                if p.is_dir() and cls._preferred_report_path(p) is not None
            )
        available_str = ", ".join(available) if available else "<none found>"
        raise FileNotFoundError(
            f"Resolved tuned file not found: {candidate}. Available tuned searches: {available_str}"
        )

    @classmethod
    def discover_experiment_files(
        cls,
        *,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_3_2",
        experiment: str,
        quant_group: Literal["baseline", "tuned", "all"] = "all",
    ) -> dict[str, Path]:
        base_quant_dir = (
            Path(diann_out_root).expanduser().resolve()
            / pipeline_version
            / experiment
            / "05_quant"
        )

        discovered: dict[str, Path] = {}

        if quant_group in {"baseline", "all"}:
            baseline_file = cls._preferred_report_path(base_quant_dir / "baseline")
            if baseline_file is not None:
                discovered["baseline"] = baseline_file

        if quant_group in {"tuned", "all"}:
            tuned_root = base_quant_dir / "tuned"
            if tuned_root.exists() and tuned_root.is_dir():
                for run_dir in sorted(tuned_root.iterdir()):
                    candidate = cls._preferred_report_path(run_dir)
                    if run_dir.is_dir() and candidate is not None:
                        discovered[run_dir.name] = candidate

        if not discovered:
            raise FileNotFoundError(
                f"No result files found for experiment={experiment} under {base_quant_dir}"
            )

        return discovered

    @classmethod
    def from_paths(
        cls,
        input_file: str | Path,
        output_dir: str | Path | None = None,
        qvalue_threshold: float = 0.01,
    ) -> "PlotProjectConfig":
        return cls.from_file(
            input_file=input_file,
            output_dir=output_dir,
            qvalue_threshold=qvalue_threshold,
        )

    @classmethod
    def from_file(
        cls,
        *,
        input_file: str | Path,
        output_dir: str | Path | None = None,
        qvalue_threshold: float = 0.01,
    ) -> "PlotProjectConfig":
        resolved_input = cls._resolve_existing_file(input_file)
        return cls(
            input_file=resolved_input,
            output_dir=cls.resolve_output_dir(resolved_input, output_dir),
            qvalue_threshold=qvalue_threshold,
        )

    @classmethod
    def from_experiment(
        cls,
        *,
        experiment: str,
        output_dir: str | Path | None = None,
        qvalue_threshold: float = 0.01,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_3_2",
        quant_group: Literal["baseline", "tuned"] = "baseline",
        tuned_name: str | None = None,
        tuned_profile: str | None = None,
        use_fr: bool | None = None,
        lr: str | float | None = None,
    ) -> "PlotProjectConfig":
        return cls.from_input_or_experiment(
            output_dir=output_dir,
            qvalue_threshold=qvalue_threshold,
            input_file=None,
            diann_out_root=diann_out_root,
            pipeline_version=pipeline_version,
            experiment=experiment,
            quant_group=quant_group,
            tuned_name=tuned_name,
            tuned_profile=tuned_profile,
            use_fr=use_fr,
            lr=lr,
        )

    @classmethod
    def from_input_or_experiment(
        cls,
        *,
        output_dir: str | Path | None = None,
        qvalue_threshold: float = 0.01,
        input_file: str | Path | None = None,
        diann_out_root: str | Path = "./diann_out",
        pipeline_version: str = "2_3_2",
        experiment: str | None = None,
        quant_group: Literal["baseline", "tuned"] = "baseline",
        tuned_name: str | None = None,
        tuned_profile: str | None = None,
        use_fr: bool | None = None,
        lr: str | float | None = None,
    ) -> "PlotProjectConfig":
        resolved_input = cls.resolve_input_file(
            input_file=input_file,
            diann_out_root=diann_out_root,
            pipeline_version=pipeline_version,
            experiment=experiment,
            quant_group=quant_group,
            tuned_name=tuned_name,
            tuned_profile=tuned_profile,
            use_fr=use_fr,
            lr=lr,
        )

        return cls(
            input_file=resolved_input,
            output_dir=cls.resolve_output_dir(resolved_input, output_dir),
            qvalue_threshold=qvalue_threshold,
        )
