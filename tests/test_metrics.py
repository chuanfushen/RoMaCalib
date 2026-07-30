from __future__ import annotations

import math

import pytest

from calibx.metrics import summarize


def success(add: float, pixel: float) -> dict:
    return {
        "status": "success",
        "keypoint_metrics": {
            "summary": {
                "keypoint_add_mean_m": add,
                "pixel_error_mean": pixel,
            }
        },
        "best_pnp": {
            "inliers": 20,
            "inlier_reprojection_error_mean": 1.0,
        },
    }


def test_failed_pose_is_excluded_from_success_only_metrics() -> None:
    result = summarize(
        [success(0.005, 2.0), {"status": "failed"}],
        auc_threshold=0.01,
        auc_delta=0.001,
    )
    assert result["n_frames"] == 2
    assert result["n_success"] == 1
    assert result["n_failed"] == 1
    assert result["success_rate"] == 50.0
    assert result["ADD/mean"] == pytest.approx(0.005)
    assert result["ADD/AUC"] == pytest.approx(0.45)
    assert result["ADD<10mm"] == 100.0
    assert result["pixel_error<5px"] == 100.0


def test_empty_summary_is_explicit() -> None:
    result = summarize([], auc_threshold=0.1, auc_delta=0.001)
    assert result["n_frames"] == 0
    assert result["n_failed"] == 0
    assert math.isnan(result["ADD/mean"])
    assert math.isnan(result["ADD/AUC"])
    assert math.isnan(result["ADD<10mm"])
    assert math.isnan(result["pixel_error<5px"])
