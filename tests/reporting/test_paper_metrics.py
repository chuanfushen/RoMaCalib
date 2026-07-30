from __future__ import annotations

import math

import pytest

from calibx.reporting.paper_metrics import (
    continuous_auc_under_threshold,
    integer_threshold_auc,
    strict_accuracy,
    summarize_paper_pose_errors,
    summarize_refinement_add,
)


def test_threshold_metrics_count_missing_poses_as_misses() -> None:
    result = summarize_paper_pose_errors(
        [50.0, 150.0],
        [10.0, 70.0],
        requested_frames=3,
    )
    assert result["n_requested"] == 3
    assert result["n_valid_pose"] == 2
    assert result["n_failed_pose"] == 1
    assert result["ADD/mean_mm"] == pytest.approx(100.0)
    assert result["ADD@100mm"] == pytest.approx(1 / 3)
    assert result["PCK@50px"] == pytest.approx(1 / 3)
    assert result["auc_representation"] == "fraction"


def test_integer_auc_uses_strict_integer_thresholds() -> None:
    assert integer_threshold_auc([0.5, 2.5], 4, 3) == pytest.approx(1 / 3)
    assert strict_accuracy([50.0], 50.0, 2) == 0.0


def test_empty_valid_pose_set_is_explicit() -> None:
    result = summarize_paper_pose_errors([], [], requested_frames=2)
    assert result["n_failed_pose"] == 2
    assert math.isnan(result["ADD/mean_mm"])
    assert result["ADD@100mm"] == 0.0


def test_valid_pose_count_cannot_exceed_requested_frames() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        summarize_paper_pose_errors([1.0], [1.0], requested_frames=0)


def test_refinement_auc_matches_d7_continuous_less_equal_contract() -> None:
    assert continuous_auc_under_threshold([0.0], 0.2, 0.1, 2) == pytest.approx(0.25)
    result = summarize_refinement_add([0.05, 0.15], requested_frames=3)
    assert result["ADD/mean_m"] == pytest.approx(0.1)
    assert result["threshold_metric_denominator"] == "all_requested_frames"
    assert result["threshold_comparator"] == "less_than_or_equal"
