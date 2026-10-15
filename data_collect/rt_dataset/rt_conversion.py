from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Self

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import theilslopes


class RTTargetTransformer(ABC):
    @abstractmethod
    def fit(self, calibration_df: pd.DataFrame) -> Self:
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class GradientNormalizer(RTTargetTransformer):
    def fit(self, calibration_df: pd.DataFrame) -> Self:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["rt_norm"] = pd.to_numeric(out["observed_rt"], errors="coerce") / pd.to_numeric(out["gradient_length"], errors="coerce")
        out["rt_conversion_method"] = "gradient_normalization"
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["reconstructed_rt"] = pd.to_numeric(out["rt_norm"], errors="coerce") * pd.to_numeric(out["gradient_length"], errors="coerce")
        return out


class PerRunMinMaxNormalizer(RTTargetTransformer):
    def __init__(self, run_column: str = "raw_file_id"):
        self.run_column = run_column
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, calibration_df: pd.DataFrame) -> Self:
        grouped = calibration_df.assign(observed_rt=pd.to_numeric(calibration_df["observed_rt"], errors="coerce")).groupby(self.run_column)["observed_rt"]
        self.bounds = {str(run): (float(values.min()), float(values.max())) for run, values in grouped}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mins = out[self.run_column].astype(str).map(lambda run: self.bounds.get(run, (np.nan, np.nan))[0])
        maxs = out[self.run_column].astype(str).map(lambda run: self.bounds.get(run, (np.nan, np.nan))[1])
        out["rt_norm_minmax"] = (pd.to_numeric(out["observed_rt"], errors="coerce") - mins) / (maxs - mins)
        out["rt_conversion_method"] = "per_run_minmax"
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mins = out[self.run_column].astype(str).map(lambda run: self.bounds.get(run, (np.nan, np.nan))[0])
        maxs = out[self.run_column].astype(str).map(lambda run: self.bounds.get(run, (np.nan, np.nan))[1])
        out["reconstructed_rt"] = pd.to_numeric(out["rt_norm_minmax"], errors="coerce") * (maxs - mins) + mins
        return out


class IRTTransformer(RTTargetTransformer):
    def __init__(
        self,
        *,
        method: str = "linear",
        run_column: str = "raw_file_id",
        observed_column: str = "observed_rt",
        target_column: str = "reference_irt",
        minimum_anchors: int = 3,
    ):
        self.method = method
        self.run_column = run_column
        self.observed_column = observed_column
        self.target_column = target_column
        self.minimum_anchors = minimum_anchors
        self.models: dict[str, dict] = {}

    def fit(self, calibration_df: pd.DataFrame) -> Self:
        required = {self.run_column, self.observed_column, self.target_column}
        if not required.issubset(calibration_df):
            raise ValueError(f"iRT calibration requires columns: {sorted(required)}")
        self.models = {}
        for run, group in calibration_df.groupby(self.run_column, sort=True):
            x = pd.to_numeric(group[self.observed_column], errors="coerce")
            y = pd.to_numeric(group[self.target_column], errors="coerce")
            valid = x.notna() & y.notna()
            x, y = x[valid].to_numpy(), y[valid].to_numpy()
            if len(x) < self.minimum_anchors:
                raise ValueError(f"Run {run!r} has only {len(x)} calibration anchors")
            self.models[str(run)] = fit_calibration_model(x, y, self.method)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply(df, inverse=False)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply(df, inverse=True)

    def _apply(self, df: pd.DataFrame, *, inverse: bool) -> pd.DataFrame:
        out = df.copy()
        input_column = "observed_irt" if inverse else self.observed_column
        output_column = "reconstructed_rt" if inverse else "observed_irt"
        values = pd.to_numeric(out[input_column], errors="coerce")
        result = pd.Series(np.nan, index=out.index, dtype=float)
        residual = pd.Series(np.nan, index=out.index, dtype=float)
        for run, index in out.groupby(self.run_column).groups.items():
            model = self.models.get(str(run))
            if model is None:
                continue
            result.loc[index] = apply_calibration_model(model, values.loc[index].to_numpy(), inverse=inverse)
            if not inverse and self.target_column in out:
                residual.loc[index] = pd.to_numeric(out.loc[index, self.target_column], errors="coerce") - result.loc[index]
        out[output_column] = result
        if not inverse:
            out["rt_conversion_method"] = f"irt_{self.method}"
            out["rt_conversion_run_id"] = out[self.run_column].astype(str)
            out["rt_conversion_model_version"] = "rt-dataset-0.1"
            out["rt_conversion_residual"] = residual
        return out

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame([{"run_id": run, **model} for run, model in self.models.items()])


class ChronologerHITransformer(RTTargetTransformer):
    def __init__(
        self,
        script_path: Path,
        *,
        python_executable: str = "python",
        id_column: str = "dataset_row_hash",
        output_hi_column: str = "observed_hi",
        extra_args: list[str] | None = None,
        minimum_rows: int = 3,
    ):
        self.script_path = script_path
        self.python_executable = python_executable
        self.id_column = id_column
        self.output_hi_column = output_hi_column
        self.extra_args = extra_args or []
        self.minimum_rows = minimum_rows
        self.version = "unknown"

    def fit(self, calibration_df: pd.DataFrame) -> Self:
        if len(calibration_df) < self.minimum_rows:
            raise ValueError(f"Chronologer conversion requires at least {self.minimum_rows} rows")
        if not self.script_path.exists():
            raise FileNotFoundError(self.script_path)
        self.version = script_checksum(self.script_path)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.id_column not in df:
            raise ValueError(f"Chronologer input requires stable ID column {self.id_column!r}")
        with tempfile.TemporaryDirectory(prefix="rt_dataset_chronologer_") as temp:
            temp_dir = Path(temp)
            input_path = temp_dir / "input.csv"
            output_path = temp_dir / "output.csv"
            df.to_csv(input_path, index=False)
            command = [
                self.python_executable,
                str(self.script_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *self.extra_args,
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"Chronologer failed ({completed.returncode}): {completed.stderr.strip()}")
            if not output_path.exists():
                raise RuntimeError(f"Chronologer produced no output. stdout={completed.stdout.strip()}")
            converted = pd.read_csv(output_path)
        if self.id_column not in converted or self.output_hi_column not in converted:
            raise ValueError(f"Chronologer output requires {self.id_column!r} and {self.output_hi_column!r}")
        if converted[self.id_column].duplicated().any():
            raise ValueError("Chronologer output contains duplicate stable IDs")
        out = df.merge(converted[[self.id_column, self.output_hi_column]], on=self.id_column, how="left", validate="one_to_one")
        if out[self.output_hi_column].isna().all():
            raise ValueError("Chronologer output could not be merged to any input row")
        out["rt_conversion_method"] = "chronologer_hi"
        out["rt_conversion_model_version"] = self.version
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Chronologer inverse conversion must be supplied by the external script")


def fit_calibration_model(x: np.ndarray, y: np.ndarray, method: str) -> dict:
    if method == "linear":
        slope, intercept = np.polyfit(x, y, 1)
        return {"method": method, "slope": float(slope), "intercept": float(intercept)}
    if method == "robust":
        slope, intercept, _, _ = theilslopes(y, x)
        return {"method": method, "slope": float(slope), "intercept": float(intercept)}
    if method == "spline":
        order = np.argsort(x)
        unique_x, unique_index = np.unique(x[order], return_index=True)
        unique_y = y[order][unique_index]
        degree = min(3, len(unique_x) - 1)
        spline = UnivariateSpline(unique_x, unique_y, k=degree, s=0)
        return {"method": method, "x": unique_x.tolist(), "y": spline(unique_x).tolist(), "degree": degree}
    raise ValueError(f"Unsupported iRT method: {method}")


def apply_calibration_model(model: dict, values: np.ndarray, *, inverse: bool) -> np.ndarray:
    if model["method"] in {"linear", "robust"}:
        slope, intercept = model["slope"], model["intercept"]
        return (values - intercept) / slope if inverse else slope * values + intercept
    x = np.asarray(model["x"], dtype=float)
    y = np.asarray(model["y"], dtype=float)
    return np.interp(values, y, x) if inverse else np.interp(values, x, y)


def script_checksum(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def save_calibration_coefficients(transformer: IRTTransformer, path: Path) -> None:
    path.write_text(json.dumps(transformer.models, indent=2))
