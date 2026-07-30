"""Metric aggregation for Calib-X evaluation results."""

from __future__ import annotations

import numpy as np


def auc_under_threshold(
    values: np.ndarray,
    threshold: float,
    delta: float,
) -> float:
    if len(values) == 0:
        return float("nan")
    thresholds = np.arange(0.0, threshold, delta, dtype=np.float64)
    counts = [(values <= value).mean() for value in thresholds]
    return float(np.trapz(counts, dx=delta) / threshold)


def summarize(
    records: list[dict],
    auc_threshold: float,
    auc_delta: float,
) -> dict:
    successes = [record for record in records if record.get("status") == "success"]
    failures = [record for record in records if record.get("status") != "success"]
    add_values = np.asarray(
        [
            record["keypoint_metrics"]["summary"]["keypoint_add_mean_m"]
            for record in successes
        ],
        dtype=np.float64,
    )
    pixel_values = np.asarray(
        [
            record["keypoint_metrics"]["summary"]["pixel_error_mean"]
            for record in successes
        ],
        dtype=np.float64,
    )
    inlier_values = np.asarray(
        [record["best_pnp"]["inliers"] for record in successes],
        dtype=np.float64,
    )
    reproj_values = np.asarray(
        [record["best_pnp"]["inlier_reprojection_error_mean"] for record in successes],
        dtype=np.float64,
    )
    summary = {
        "n_frames": len(records),
        "n_success": len(successes),
        "n_failed": len(failures),
        "success_rate": (
            float(len(successes) / len(records) * 100.0) if records else float("nan")
        ),
        "ADD/mean": (
            float(add_values.mean())
            if len(add_values)
            else float("nan")
        ),
        "ADD/median": (
            float(np.median(add_values))
            if len(add_values)
            else float("nan")
        ),
        "ADD/AUC": auc_under_threshold(
            add_values,
            auc_threshold,
            auc_delta,
        ),
        "ADD/auc_threshold_m": float(auc_threshold),
        "pixel_error/mean": (
            float(pixel_values.mean())
            if len(pixel_values)
            else float("nan")
        ),
        "pixel_error/median": (
            float(np.median(pixel_values))
            if len(pixel_values)
            else float("nan")
        ),
        "pnp_inliers/mean": (
            float(inlier_values.mean()) if len(inlier_values) else float("nan")
        ),
        "pnp_reprojection_error/mean": (
            float(reproj_values.mean()) if len(reproj_values) else float("nan")
        ),
    }
    for threshold_mm in (10, 20, 40, 60):
        summary[f"ADD<{threshold_mm}mm"] = (
            float((add_values <= threshold_mm * 1e-3).mean() * 100.0)
            if len(add_values)
            else float("nan")
        )
    for threshold_px in (5, 10, 20):
        summary[f"pixel_error<{threshold_px}px"] = (
            float((pixel_values <= threshold_px).mean() * 100.0)
            if len(pixel_values)
            else float("nan")
        )
    return summary
