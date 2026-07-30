"""Metrics used by the paper protocols without changing core compatibility.

``calibx.metrics`` preserves the historical success-only aggregation used by
the upstream-compatible runner.  Paper tables instead use every *requested*
frame as the denominator for threshold metrics and AUCs; failed PnP frames
therefore count as misses.  Means, medians, and the diagnostic 100 mm AUC are
reported over valid poses only.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np


def _as_finite_array(values: Iterable[float]) -> np.ndarray:
    """Return finite errors as a one-dimensional float64 array."""
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _validate_denominator(denominator: int, valid_count: int) -> None:
    if denominator < 0:
        raise ValueError("denominator must be non-negative")
    if valid_count > denominator:
        raise ValueError(
            "valid pose count cannot exceed the requested-frame denominator"
        )


def strict_accuracy(
    errors: Iterable[float],
    threshold: float,
    denominator: int,
) -> float:
    """Return ``P(error < threshold)`` against an explicit denominator.

    The explicit denominator is important: failures are represented by their
    absence from ``errors`` yet still count as threshold misses.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    values = _as_finite_array(errors)
    _validate_denominator(denominator, len(values))
    if denominator == 0:
        return math.nan
    return float(np.count_nonzero(values < threshold) / denominator)


def integer_threshold_auc(
    errors: Iterable[float],
    maximum_threshold: int,
    denominator: int,
) -> float:
    """Mean strict accuracy at integer thresholds ``0..maximum_threshold-1``.

    This matches the paper protocol for PCK AUC@200 and ADD AUC@400.  The
    result is a fraction in ``[0, 1]`` rather than a percentage.
    """
    if maximum_threshold <= 0:
        raise ValueError("maximum_threshold must be positive")
    values = _as_finite_array(errors)
    _validate_denominator(denominator, len(values))
    if denominator == 0:
        return math.nan
    thresholds = np.arange(maximum_threshold, dtype=np.float64)
    accuracies = (values[:, None] < thresholds[None, :]).sum(axis=0)
    return float((accuracies / denominator).mean())


def continuous_auc_under_threshold(
    errors: Iterable[float],
    maximum_threshold: float,
    step: float,
    denominator: int,
) -> float:
    """d7-compatible continuous AUC using ``error <= threshold``.

    The gate-v1/Table 4 evaluator uses this continuous contract over meters,
    not the integer strict-threshold convention used by official point metrics.
    """
    if maximum_threshold <= 0.0 or step <= 0.0:
        raise ValueError("maximum_threshold and step must be positive")
    values = _as_finite_array(errors)
    _validate_denominator(denominator, len(values))
    if denominator == 0:
        return math.nan
    thresholds = np.arange(0.0, maximum_threshold, step, dtype=np.float64)
    counts = np.asarray([(values <= value).sum() / denominator for value in thresholds])
    return float(np.trapz(counts, dx=step) / maximum_threshold)


def _valid_only_auc(errors: np.ndarray, maximum_threshold: int) -> float:
    return integer_threshold_auc(
        errors,
        maximum_threshold=maximum_threshold,
        denominator=len(errors),
    )


def summarize_paper_pose_errors(
    add_errors_mm: Iterable[float],
    pixel_errors_px: Iterable[float],
    *,
    requested_frames: int,
) -> dict[str, float | int | str]:
    """Summarize pose errors using the paper's denominator contract.

    ``add_errors_mm`` and ``pixel_errors_px`` contain valid-pose errors only.
    Their lengths must match and may be smaller than ``requested_frames`` when
    PnP failed.  Values in the returned AUC and threshold fields are fractions.
    """
    add_values = _as_finite_array(add_errors_mm)
    pixel_values = _as_finite_array(pixel_errors_px)
    if len(add_values) != len(pixel_values):
        raise ValueError("ADD and pixel errors must have the same valid-pose count")
    _validate_denominator(requested_frames, len(add_values))
    valid_count = len(add_values)
    return {
        "n_requested": requested_frames,
        "n_valid_pose": valid_count,
        "n_failed_pose": requested_frames - valid_count,
        "ADD/mean_mm": float(add_values.mean()) if valid_count else math.nan,
        "ADD/median_mm": float(np.median(add_values)) if valid_count else math.nan,
        "pixel_error/mean_px": (
            float(pixel_values.mean()) if valid_count else math.nan
        ),
        "pixel_error/median_px": (
            float(np.median(pixel_values)) if valid_count else math.nan
        ),
        "ADD@100mm": strict_accuracy(add_values, 100.0, requested_frames),
        "ADD/AUC@400mm": integer_threshold_auc(
            add_values,
            maximum_threshold=400,
            denominator=requested_frames,
        ),
        "ADD/AUC@100mm_valid": _valid_only_auc(add_values, 100),
        "PCK@50px": strict_accuracy(pixel_values, 50.0, requested_frames),
        "PCK/AUC@200px": integer_threshold_auc(
            pixel_values,
            maximum_threshold=200,
            denominator=requested_frames,
        ),
        "threshold_comparator": "strictly_less_than",
        "auc_representation": "fraction",
    }


def summarize_refinement_add(
    add_errors_m: Iterable[float],
    *,
    requested_frames: int,
    auc_threshold_m: float = 0.1,
    auc_step_m: float = 1e-4,
) -> dict[str, float | int | str]:
    """Summarize d7 gate-v1 ADD using the Table 4 continuous-AUC contract."""
    values = _as_finite_array(add_errors_m)
    _validate_denominator(requested_frames, len(values))
    valid_count = len(values)
    return {
        "n_requested": requested_frames,
        "n_valid_pose": valid_count,
        "n_failed_pose": requested_frames - valid_count,
        "ADD/mean_m": float(values.mean()) if valid_count else math.nan,
        "ADD/median_m": float(np.median(values)) if valid_count else math.nan,
        "ADD/AUC": continuous_auc_under_threshold(
            values,
            maximum_threshold=auc_threshold_m,
            step=auc_step_m,
            denominator=requested_frames,
        ),
        "ADD/auc_threshold_m": auc_threshold_m,
        "threshold_metric_denominator": "all_requested_frames",
        "threshold_comparator": "less_than_or_equal",
        "auc_representation": "fraction",
    }
