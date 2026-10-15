"""Mathematical retention-time alignment models with a shared interface.

Each model learns a one-dimensional mapping from a source LC-MS run to a
reference RT coordinate.  The module deliberately contains no peptide
filtering, train/test splitting, benchmarking, plotting, or model-training
workflow code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module, metadata as importlib_metadata
from typing import ClassVar, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import fftconvolve


ArrayLike = Iterable[float] | np.ndarray
EXTRAPOLATION_MODES = ("clip", "linear")


def _as_1d_finite(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_anchor_pairs(
    source_rt: ArrayLike,
    reference_rt: ArrayLike,
    *,
    minimum_anchors: int,
    minimum_distinct_source: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    source = _as_1d_finite(source_rt, "source_rt")
    reference = _as_1d_finite(reference_rt, "reference_rt")
    if source.shape != reference.shape:
        raise ValueError("source_rt and reference_rt must have equal length")
    if len(source) < minimum_anchors:
        raise ValueError(f"At least {minimum_anchors} anchor pairs are required")
    if np.unique(source).size < minimum_distinct_source:
        raise ValueError(
            f"At least {minimum_distinct_source} distinct source RT values are required"
        )
    return source, reference


def _validate_extrapolation(mode: str) -> str:
    if mode not in EXTRAPOLATION_MODES:
        raise ValueError(f"extrapolation must be one of {EXTRAPOLATION_MODES}")
    return mode


def _aggregate_source_duplicates(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse duplicate/near-duplicate source RTs using coordinate medians."""
    if tolerance < 0:
        raise ValueError("duplicate_tolerance must be non-negative")
    order = np.argsort(source, kind="mergesort")
    source, reference = source[order], reference[order]
    groups: list[tuple[int, int]] = []
    start = 0
    for stop in range(1, len(source)):
        if source[stop] - source[stop - 1] > tolerance:
            groups.append((start, stop))
            start = stop
    groups.append((start, len(source)))
    x = np.asarray([np.median(source[start:stop]) for start, stop in groups], dtype=float)
    y = np.asarray([np.median(reference[start:stop]) for start, stop in groups], dtype=float)
    weights = np.asarray([stop - start for start, stop in groups], dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Anchor aggregation must leave at least two strictly ordered source RTs")
    return x, y, weights


def _pava(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted pool-adjacent-violators algorithm for non-decreasing fits."""
    values = np.asarray(values, dtype=float)
    weights = np.ones(len(values), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("PAVA values and weights must be matching one-dimensional arrays")
    if np.any(weights <= 0):
        raise ValueError("PAVA weights must be positive")

    levels: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []
    for value, weight in zip(values, weights):
        levels.append(float(value))
        block_weights.append(float(weight))
        block_sizes.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_level = (
                levels[-2] * block_weights[-2] + levels[-1] * block_weights[-1]
            ) / merged_weight
            merged_size = block_sizes[-2] + block_sizes[-1]
            levels[-2:] = [merged_level]
            block_weights[-2:] = [merged_weight]
            block_sizes[-2:] = [merged_size]
    return np.concatenate([
        np.full(size, level, dtype=float) for level, size in zip(levels, block_sizes)
    ])


def _piecewise_linear_transform(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    extrapolation: str,
) -> np.ndarray:
    result = np.interp(values, x, y)
    if extrapolation == "clip":
        return result
    left_slope = (y[1] - y[0]) / (x[1] - x[0])
    right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    left = values < x[0]
    right = values > x[-1]
    result[left] = y[0] + left_slope * (values[left] - x[0])
    result[right] = y[-1] + right_slope * (values[right] - x[-1])
    return result


class RetentionTimeAlignment(ABC):
    """Common fit/transform interface for one-dimensional RT mappings."""

    method: ClassVar[str]
    guarantees_monotonicity: ClassVar[bool] = False

    def __init__(self) -> None:
        self._fitted = False
        self.n_anchors_: int | None = None
        self.source_range_: tuple[float, float] | None = None
        self.reference_range_: tuple[float, float] | None = None

    def _record_fit(self, source: np.ndarray, reference: np.ndarray) -> None:
        self._fitted = True
        self.n_anchors_ = int(len(source))
        self.source_range_ = (float(source.min()), float(source.max()))
        self.reference_range_ = (float(reference.min()), float(reference.max()))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")

    @abstractmethod
    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "RetentionTimeAlignment":
        """Fit the source-to-reference mapping and return ``self``."""

    @abstractmethod
    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        """Map arbitrary source RT values into the reference coordinate."""

    def fit_transform(
        self,
        source_rt: ArrayLike,
        reference_rt: ArrayLike,
        rt_to_transform: ArrayLike | None = None,
    ) -> np.ndarray:
        values = source_rt if rt_to_transform is None else rt_to_transform
        return self.fit(source_rt, reference_rt).transform(values)

    def get_params(self) -> dict:
        """Return fitted parameters and method metadata."""
        return self.metadata()

    def metadata(self) -> dict:
        self._check_fitted()
        return {
            "class": type(self).__name__,
            "method": self.method,
            "guarantees_monotonicity": self.guarantees_monotonicity,
            "n_anchors": self.n_anchors_,
            "source_range": list(self.source_range_),
            "reference_range": list(self.reference_range_),
        }


class LinearAlignment(RetentionTimeAlignment):
    """Affine RT alignment with optional robust soft-L1 regression."""

    method = "linear"

    def __init__(self, *, robust: bool = True, f_scale: float = 0.5) -> None:
        super().__init__()
        if f_scale <= 0:
            raise ValueError("f_scale must be positive")
        self.robust = bool(robust)
        self.f_scale = float(f_scale)
        self.intercept_: float | None = None
        self.slope_: float | None = None

    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "LinearAlignment":
        source, reference = _validate_anchor_pairs(source_rt, reference_rt, minimum_anchors=2)
        slope, intercept = np.polyfit(source, reference, 1)
        if self.robust:
            result = least_squares(
                lambda parameters: parameters[0] + parameters[1] * source - reference,
                np.asarray([intercept, slope], dtype=float),
                loss="soft_l1",
                f_scale=self.f_scale,
            )
            if not result.success:
                raise RuntimeError(f"Robust linear RT alignment failed: {result.message}")
            intercept, slope = result.x
        self.intercept_, self.slope_ = float(intercept), float(slope)
        self._record_fit(source, reference)
        return self

    @property
    def guarantees_monotonicity(self) -> bool:  # type: ignore[override]
        return bool(self._fitted and self.slope_ is not None and self.slope_ >= 0)

    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        self._check_fitted()
        values = _as_1d_finite(source_rt, "source_rt")
        return self.intercept_ + self.slope_ * values

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "robust": self.robust,
            "f_scale": self.f_scale,
            "intercept": self.intercept_,
            "slope": self.slope_,
        })
        return result


class RelativeRTAlignment(RetentionTimeAlignment):
    """Convert absolute source RT to its relative gradient position."""

    method = "relative_rt"
    guarantees_monotonicity = True

    def __init__(self, gradient_end: float, *, t0: float = 0.0, clip: bool = False) -> None:
        super().__init__()
        if not np.isfinite([gradient_end, t0]).all() or gradient_end <= t0:
            raise ValueError("gradient_end must be finite and greater than t0")
        self.gradient_end = float(gradient_end)
        self.t0 = float(t0)
        self.clip = bool(clip)

    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "RelativeRTAlignment":
        source, reference = _validate_anchor_pairs(
            source_rt, reference_rt, minimum_anchors=1, minimum_distinct_source=1
        )
        self._record_fit(source, reference)
        return self

    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        self._check_fitted()
        values = (_as_1d_finite(source_rt, "source_rt") - self.t0) / (
            self.gradient_end - self.t0
        )
        return np.clip(values, 0.0, 1.0) if self.clip else values

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({"gradient_end": self.gradient_end, "t0": self.t0, "clip": self.clip})
        return result


class LowessAlignment(RetentionTimeAlignment):
    """Locally weighted nonlinear RT alignment.

    Standard LOWESS does not guarantee a monotonic mapping.  Set
    ``monotonic_postprocess=True`` to apply an explicit isotonic projection to
    the fitted LOWESS knots.
    """

    method = "lowess"

    def __init__(
        self,
        *,
        frac: float = 0.25,
        iterations: int = 3,
        delta: float = 0.0,
        monotonic_postprocess: bool = False,
        extrapolation: str = "clip",
        duplicate_tolerance: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 < frac <= 1:
            raise ValueError("frac must be in (0, 1]")
        if iterations < 0 or delta < 0:
            raise ValueError("iterations and delta must be non-negative")
        self.frac = float(frac)
        self.iterations = int(iterations)
        self.delta = float(delta)
        self.monotonic_postprocess = bool(monotonic_postprocess)
        self.extrapolation = _validate_extrapolation(extrapolation)
        self.duplicate_tolerance = float(duplicate_tolerance)
        self.x_knots_: np.ndarray | None = None
        self.y_knots_: np.ndarray | None = None
        self.fitted_curve_is_monotonic_: bool | None = None

    @property
    def guarantees_monotonicity(self) -> bool:  # type: ignore[override]
        return self.monotonic_postprocess

    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "LowessAlignment":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError as error:
            raise ImportError(
                "LowessAlignment requires statsmodels>=0.14; install the project dependencies."
            ) from error
        source, reference = _validate_anchor_pairs(source_rt, reference_rt, minimum_anchors=4)
        x, y, weights = _aggregate_source_duplicates(
            source, reference, tolerance=self.duplicate_tolerance
        )
        if len(x) < 4:
            raise ValueError("LOWESS requires at least four distinct aggregated source RTs")
        fitted = lowess(
            y,
            x,
            frac=self.frac,
            it=self.iterations,
            delta=self.delta,
            is_sorted=True,
            return_sorted=False,
            missing="raise",
        )
        if self.monotonic_postprocess:
            fitted = _pava(fitted, weights)
        self.x_knots_, self.y_knots_ = x, np.asarray(fitted, dtype=float)
        self.fitted_curve_is_monotonic_ = bool(np.all(np.diff(self.y_knots_) >= -1e-12))
        self._record_fit(source, reference)
        return self

    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        self._check_fitted()
        values = _as_1d_finite(source_rt, "source_rt")
        return _piecewise_linear_transform(
            values, self.x_knots_, self.y_knots_, extrapolation=self.extrapolation
        )

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "frac": self.frac,
            "iterations": self.iterations,
            "delta": self.delta,
            "monotonic_postprocess": self.monotonic_postprocess,
            "fitted_curve_is_monotonic": self.fitted_curve_is_monotonic_,
            "extrapolation": self.extrapolation,
            "duplicate_tolerance": self.duplicate_tolerance,
        })
        return result


class IsotonicAlignment(RetentionTimeAlignment):
    """Non-decreasing isotonic fit with continuous linear interpolation."""

    method = "isotonic"
    guarantees_monotonicity = True

    def __init__(self, *, extrapolation: str = "clip", duplicate_tolerance: float = 0.0) -> None:
        super().__init__()
        self.extrapolation = _validate_extrapolation(extrapolation)
        self.duplicate_tolerance = float(duplicate_tolerance)
        self.x_knots_: np.ndarray | None = None
        self.y_isotonic_: np.ndarray | None = None

    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "IsotonicAlignment":
        source, reference = _validate_anchor_pairs(source_rt, reference_rt, minimum_anchors=2)
        x, y, weights = _aggregate_source_duplicates(
            source, reference, tolerance=self.duplicate_tolerance
        )
        self.x_knots_, self.y_isotonic_ = x, _pava(y, weights)
        self._record_fit(source, reference)
        return self

    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        self._check_fitted()
        values = _as_1d_finite(source_rt, "source_rt")
        return _piecewise_linear_transform(
            values, self.x_knots_, self.y_isotonic_, extrapolation=self.extrapolation
        )

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "continuous_interpolation": "piecewise_linear",
            "isotonic_fit": "piecewise_constant_at_anchor_knots",
            "extrapolation": self.extrapolation,
            "duplicate_tolerance": self.duplicate_tolerance,
        })
        return result


class MonotonicSplineAlignment(RetentionTimeAlignment):
    """Shape-preserving PCHIP over robust, isotonic anchor knots."""

    method = "monotonic_spline"
    guarantees_monotonicity = True

    def __init__(
        self,
        *,
        n_bins: int | None = None,
        extrapolation: str = "clip",
        duplicate_tolerance: float = 0.0,
    ) -> None:
        super().__init__()
        if n_bins is not None and n_bins < 2:
            raise ValueError("n_bins must be at least two")
        self.n_bins = n_bins
        self.extrapolation = _validate_extrapolation(extrapolation)
        self.duplicate_tolerance = float(duplicate_tolerance)
        self.x_knots_: np.ndarray | None = None
        self.y_knots_: np.ndarray | None = None
        self._spline: PchipInterpolator | None = None

    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "MonotonicSplineAlignment":
        source, reference = _validate_anchor_pairs(source_rt, reference_rt, minimum_anchors=2)
        x, y, weights = _aggregate_source_duplicates(
            source, reference, tolerance=self.duplicate_tolerance
        )
        if self.n_bins is not None and len(x) > self.n_bins:
            groups = np.array_split(np.arange(len(x)), self.n_bins)
            x = np.asarray([np.median(x[group]) for group in groups], dtype=float)
            y = np.asarray([np.median(y[group]) for group in groups], dtype=float)
            weights = np.asarray([weights[group].sum() for group in groups], dtype=float)
        y = _pava(y, weights)
        self.x_knots_, self.y_knots_ = x, y
        self._spline = PchipInterpolator(x, y, extrapolate=False)
        self._record_fit(source, reference)
        return self

    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        self._check_fitted()
        values = _as_1d_finite(source_rt, "source_rt")
        clipped = np.clip(values, self.x_knots_[0], self.x_knots_[-1])
        result = np.asarray(self._spline(clipped), dtype=float)
        if self.extrapolation == "linear":
            left_slope = max(0.0, float(self._spline.derivative()(self.x_knots_[0])))
            right_slope = max(0.0, float(self._spline.derivative()(self.x_knots_[-1])))
            left = values < self.x_knots_[0]
            right = values > self.x_knots_[-1]
            result[left] = self.y_knots_[0] + left_slope * (values[left] - self.x_knots_[0])
            result[right] = self.y_knots_[-1] + right_slope * (values[right] - self.x_knots_[-1])
        return result

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "interpolator": "PCHIP",
            "monotonic_enforcement": "weighted_PAVA",
            "n_bins": self.n_bins,
            "n_knots": int(len(self.x_knots_)),
            "extrapolation": self.extrapolation,
            "duplicate_tolerance": self.duplicate_tolerance,
        })
        return result


class KDEAlignment(RetentionTimeAlignment):
    """Chronologer/EncyclopeDIA-style 2D KDE ridge alignment.

    The density construction and monotonic greedy ridge walk are adapted from
    Chronologer's Apache-2.0-licensed ``KDE_align`` implementation.  A weighted
    2D histogram plus FFT convolution replaces its per-anchor stamp loop while
    preserving the same cosine-Gaussian kernel and path geometry.
    """

    method = "kde"
    guarantees_monotonicity = True

    def __init__(
        self,
        *,
        grid_size: int = 512,
        bandwidth_scale: float = 1.0,
        bandwidth_pixels: float | None = None,
        diagonal_bias: float = 1.0,
        extrapolation: str = "clip",
        source_bounds: tuple[float, float] | None = None,
        reference_bounds: tuple[float, float] | None = None,
        duplicate_tolerance: float = 0.0,
        retain_density: bool = False,
    ) -> None:
        super().__init__()
        if grid_size < 32:
            raise ValueError("grid_size must be at least 32")
        if bandwidth_scale <= 0 or (bandwidth_pixels is not None and bandwidth_pixels <= 0):
            raise ValueError("KDE bandwidth values must be positive")
        if diagonal_bias <= 0:
            raise ValueError("diagonal_bias must be positive")
        self.grid_size = int(grid_size)
        self.bandwidth_scale = float(bandwidth_scale)
        self.bandwidth_pixels = None if bandwidth_pixels is None else float(bandwidth_pixels)
        self.diagonal_bias = float(diagonal_bias)
        self.extrapolation = _validate_extrapolation(extrapolation)
        self.source_bounds = source_bounds
        self.reference_bounds = reference_bounds
        self.duplicate_tolerance = float(duplicate_tolerance)
        self.retain_density = bool(retain_density)
        self.x_knots_: np.ndarray | None = None
        self.y_knots_: np.ndarray | None = None
        self.density_map_: np.ndarray | None = None
        self.fitted_bandwidth_pixels_: float | None = None

    @staticmethod
    def _resolve_bounds(
        values: np.ndarray, bounds: tuple[float, float] | None, name: str,
    ) -> tuple[float, float]:
        lower, upper = (float(values.min()), float(values.max())) if bounds is None else bounds
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise ValueError(f"{name} bounds must be finite and strictly increasing")
        tolerance = np.finfo(float).eps * max(1.0, abs(lower), abs(upper)) * 16
        if values.min() < lower - tolerance or values.max() > upper + tolerance:
            raise ValueError(f"{name} anchors lie outside the configured bounds")
        return float(lower), float(upper)

    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "KDEAlignment":
        source, reference = _validate_anchor_pairs(source_rt, reference_rt, minimum_anchors=4)
        x, y, weights = _aggregate_source_duplicates(
            source, reference, tolerance=self.duplicate_tolerance
        )
        if len(x) < 4:
            raise ValueError("KDE alignment requires at least four distinct source RT values")
        min_x, max_x = self._resolve_bounds(x, self.source_bounds, "source")
        min_y, max_y = self._resolve_bounds(y, self.reference_bounds, "reference")
        n = self.grid_size
        x_pixels = np.rint((n - 1) * (x - min_x) / (max_x - min_x)).astype(int)
        y_pixels = np.rint((n - 1) * (y - min_y) / (max_y - min_y)).astype(int)

        histogram = np.zeros((n, n), dtype=np.float32)
        np.add.at(histogram, (x_pixels, y_pixels), weights.astype(np.float32))
        if self.bandwidth_pixels is None:
            bandwidth = len(source) ** (-1 / 6) * (
                np.std(x_pixels) + np.std(y_pixels)
            ) / 2
            bandwidth *= self.bandwidth_scale
        else:
            bandwidth = self.bandwidth_pixels
        bandwidth = max(float(bandwidth), 1.0)
        self.fitted_bandwidth_pixels_ = bandwidth
        kernel_sd = max(bandwidth / (2 * np.sqrt(2 * np.log(2))), 0.5)
        radius = max(1, int(round(2 * kernel_sd)))
        coordinates = np.arange(-radius, radius + 1, dtype=float)
        distance = np.hypot(coordinates[:, None], coordinates[None, :])
        kernel = np.zeros_like(distance, dtype=np.float32)
        support = distance <= 2 * kernel_sd
        kernel[support] = (
            np.pi / (8 * kernel_sd)
            * np.cos(distance[support] * np.pi / (4 * kernel_sd))
        ).astype(np.float32)
        density = fftconvolve(histogram, kernel, mode="same").astype(np.float32)

        peak_x, peak_y = np.unravel_index(int(np.argmax(density)), density.shape)
        steps = ((0, 1), (1, 0), (1, 1))
        fit_points: list[tuple[int, int]] = [(int(peak_x), int(peak_y))]
        trace_x, trace_y = int(peak_x), int(peak_y)
        while trace_x < n - 1 and trace_y < n - 1:
            scores = [density[trace_x + dx, trace_y + dy] for dx, dy in steps]
            scores[2] *= self.diagonal_bias
            choice = int(np.argmax(scores))
            dx, dy = steps[choice]
            trace_x, trace_y = trace_x + dx, trace_y + dy
            if dx:
                fit_points.append((trace_x, trace_y))
        trace_x, trace_y = int(peak_x), int(peak_y)
        while trace_x > 0 and trace_y > 0:
            scores = [density[trace_x - dx, trace_y - dy] for dx, dy in steps]
            scores[2] *= self.diagonal_bias
            choice = int(np.argmax(scores))
            dx, dy = steps[choice]
            trace_x, trace_y = trace_x - dx, trace_y - dy
            if dx:
                fit_points.append((trace_x, trace_y))

        fit_points.sort(key=lambda point: point[0])
        if fit_points[0][0] != 0:
            fit_points.insert(0, (0, fit_points[0][1]))
        if fit_points[-1][0] != n - 1:
            fit_points.append((n - 1, fit_points[-1][1]))
        pixel_x, pixel_y = np.asarray(fit_points, dtype=float).T
        self.x_knots_ = min_x + pixel_x / (n - 1) * (max_x - min_x)
        self.y_knots_ = min_y + pixel_y / (n - 1) * (max_y - min_y)
        if np.any(np.diff(self.x_knots_) <= 0) or np.any(np.diff(self.y_knots_) < -1e-12):
            raise RuntimeError("KDE ridge path is not a monotonic function")
        self.density_map_ = density if self.retain_density else None
        self._record_fit(source, reference)
        return self

    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        self._check_fitted()
        values = _as_1d_finite(source_rt, "source_rt")
        return _piecewise_linear_transform(
            values, self.x_knots_, self.y_knots_, extrapolation=self.extrapolation
        )

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "implementation": "adapted Chronologer KDE ridge walk",
            "upstream_license": "Apache-2.0",
            "grid_size": self.grid_size,
            "bandwidth_scale": self.bandwidth_scale,
            "bandwidth_pixels": self.bandwidth_pixels,
            "fitted_bandwidth_pixels": self.fitted_bandwidth_pixels_,
            "diagonal_bias": self.diagonal_bias,
            "extrapolation": self.extrapolation,
            "duplicate_tolerance": self.duplicate_tolerance,
            "n_knots": int(len(self.x_knots_)),
        })
        return result


class ChronologerHIAlignment(KDEAlignment):
    """Chronologer-style measured RT to external hydrophobic-index mapping."""

    method = "chronologer_hi"

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "target_coordinate": "Chronologer hydrophobic index",
            "reference_is_experimental_run": False,
        })
        return result


class CalibRTAlignment(RetentionTimeAlignment):
    """Adapter for the official ``pycalib_rt`` package.

    Install the optional dependency with ``pip install pycalib_rt==0.1.2``.
    The wrapper intentionally delegates fitting and prediction to
    ``calib_rt.Calib_RT`` instead of reimplementing the published algorithm.
    """

    method = "calib_rt"
    guarantees_monotonicity = True

    def __init__(self, *, bins: int = 100, tol_bins: float = 10, manual_frac: float = -1) -> None:
        super().__init__()
        if bins < 2 or tol_bins < 0:
            raise ValueError("bins must be at least two and tol_bins must be non-negative")
        self.bins = int(bins)
        self.tol_bins = float(tol_bins)
        self.manual_frac = float(manual_frac)
        self._model = None
        self.package_version_: str | None = None

    @staticmethod
    def _load_official_package():
        try:
            return import_module("calib_rt")
        except ImportError as error:
            raise ImportError(
                "CalibRTAlignment requires the official Calib-RT package; "
                "install it with `pip install pycalib_rt==0.1.2`."
            ) from error

    def fit(self, source_rt: ArrayLike, reference_rt: ArrayLike) -> "CalibRTAlignment":
        source, reference = _validate_anchor_pairs(source_rt, reference_rt, minimum_anchors=4)
        package = self._load_official_package()
        self._model = package.Calib_RT(bins=self.bins, tol_bins=self.tol_bins)
        self._model.fit(source, reference, manual_frac=self.manual_frac)
        try:
            self.package_version_ = importlib_metadata.version("pycalib-rt")
        except importlib_metadata.PackageNotFoundError:
            self.package_version_ = "unknown"
        self._record_fit(source, reference)
        return self

    def transform(self, source_rt: ArrayLike) -> np.ndarray:
        self._check_fitted()
        values = _as_1d_finite(source_rt, "source_rt")
        result = np.asarray(self._model.predict(values), dtype=float).reshape(-1)
        if result.shape != values.shape or not np.isfinite(result).all():
            raise RuntimeError("Official Calib-RT returned invalid predictions")
        return result

    def metadata(self) -> dict:
        result = super().metadata()
        result.update({
            "implementation": "official calib_rt.Calib_RT",
            "package": "pycalib-rt",
            "package_version": self.package_version_,
            "bins": self.bins,
            "tol_bins": self.tol_bins,
            "manual_frac": self.manual_frac,
        })
        return result


ALIGNMENT_METHODS = {
    "linear": LinearAlignment,
    "relative_rt": RelativeRTAlignment,
    "lowess": LowessAlignment,
    "isotonic": IsotonicAlignment,
    "monotonic_spline": MonotonicSplineAlignment,
    "calib_rt": CalibRTAlignment,
    "kde": KDEAlignment,
    "chronologer_hi": ChronologerHIAlignment,
}


def alignment_state(alignment: RetentionTimeAlignment, *, calibrt_grid_size: int = 2048) -> dict:
    """Return a JSON-serializable fitted transform for runtime application.

    Calib-RT itself remains the fitting/prediction implementation.  Its fitted
    curve is sampled densely so prepared datasets and later evaluation can
    reproduce the mapping without pickling third-party Python objects.
    """
    alignment._check_fitted()
    common = {"method": alignment.method, "metadata": alignment.metadata()}
    if isinstance(alignment, LinearAlignment):
        return {
            **common,
            "kind": "affine",
            "slope": alignment.slope_,
            "intercept": alignment.intercept_,
        }
    if isinstance(alignment, RelativeRTAlignment):
        return {
            **common,
            "kind": "relative_rt",
            "gradient_end": alignment.gradient_end,
            "t0": alignment.t0,
            "clip": alignment.clip,
        }
    if isinstance(alignment, LowessAlignment):
        return {
            **common,
            "kind": "piecewise_linear",
            "x": alignment.x_knots_.tolist(),
            "y": alignment.y_knots_.tolist(),
            "extrapolation": alignment.extrapolation,
        }
    if isinstance(alignment, IsotonicAlignment):
        return {
            **common,
            "kind": "piecewise_linear",
            "x": alignment.x_knots_.tolist(),
            "y": alignment.y_isotonic_.tolist(),
            "extrapolation": alignment.extrapolation,
        }
    if isinstance(alignment, MonotonicSplineAlignment):
        return {
            **common,
            "kind": "pchip",
            "x": alignment.x_knots_.tolist(),
            "y": alignment.y_knots_.tolist(),
            "extrapolation": alignment.extrapolation,
        }
    if isinstance(alignment, KDEAlignment):
        return {
            **common,
            "kind": "piecewise_linear",
            "x": alignment.x_knots_.tolist(),
            "y": alignment.y_knots_.tolist(),
            "extrapolation": alignment.extrapolation,
        }
    if isinstance(alignment, CalibRTAlignment):
        if calibrt_grid_size < 2:
            raise ValueError("calibrt_grid_size must be at least two")
        lower, upper = alignment.source_range_
        grid = np.linspace(lower, upper, calibrt_grid_size)
        predictions = alignment.transform(grid)
        return {
            **common,
            "kind": "piecewise_linear",
            "x": grid.tolist(),
            "y": predictions.tolist(),
            "extrapolation": "linear",
            "runtime_representation": "dense sampling of official Calib-RT predictor",
        }
    raise TypeError(f"Unsupported alignment class: {type(alignment).__name__}")


def transform_alignment_state(state: dict, source_rt: ArrayLike) -> np.ndarray:
    """Apply a JSON alignment state without requiring its fitting dependency."""
    values = _as_1d_finite(source_rt, "source_rt")
    kind = state.get("kind")
    if kind == "affine":
        return float(state["intercept"]) + float(state["slope"]) * values
    if kind == "relative_rt":
        t0 = float(state["t0"])
        gradient_end = float(state["gradient_end"])
        result = (values - t0) / (gradient_end - t0)
        return np.clip(result, 0.0, 1.0) if state.get("clip", False) else result
    if kind not in {"piecewise_linear", "pchip"}:
        raise ValueError(f"Unknown serialized RT alignment kind: {kind!r}")
    x = _as_1d_finite(state["x"], "state.x")
    y = _as_1d_finite(state["y"], "state.y")
    if x.shape != y.shape or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Serialized RT alignment knots must be matching and strictly ordered")
    extrapolation = _validate_extrapolation(state.get("extrapolation", "clip"))
    if kind == "piecewise_linear":
        return _piecewise_linear_transform(values, x, y, extrapolation=extrapolation)

    spline = PchipInterpolator(x, y, extrapolate=False)
    clipped = np.clip(values, x[0], x[-1])
    result = np.asarray(spline(clipped), dtype=float)
    if extrapolation == "linear":
        left_slope = float(spline.derivative()(x[0]))
        right_slope = float(spline.derivative()(x[-1]))
        if np.all(np.diff(y) >= -1e-12):
            left_slope, right_slope = max(0.0, left_slope), max(0.0, right_slope)
        left, right = values < x[0], values > x[-1]
        result[left] = y[0] + left_slope * (values[left] - x[0])
        result[right] = y[-1] + right_slope * (values[right] - x[-1])
    return result


def create_alignment(method: str, **kwargs) -> RetentionTimeAlignment:
    """Construct one alignment implementation by its canonical method name."""
    try:
        alignment_class = ALIGNMENT_METHODS[method]
    except KeyError as error:
        raise ValueError(
            f"Unknown RT alignment method {method!r}; choose one of {tuple(ALIGNMENT_METHODS)}"
        ) from error
    return alignment_class(**kwargs)


__all__ = [
    "ALIGNMENT_METHODS",
    "CalibRTAlignment",
    "ChronologerHIAlignment",
    "IsotonicAlignment",
    "KDEAlignment",
    "LinearAlignment",
    "LowessAlignment",
    "MonotonicSplineAlignment",
    "RelativeRTAlignment",
    "RetentionTimeAlignment",
    "alignment_state",
    "create_alignment",
    "transform_alignment_state",
]
