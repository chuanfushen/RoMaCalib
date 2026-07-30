from __future__ import annotations

import pytest

from calibx.reporting.records import refinement_add_errors_m, summarize_refinement_records


def _success(add_error_m: float) -> dict:
    return {
        "status": "success",
        "keypoint_metrics": {"summary": {"keypoint_add_mean_m": add_error_m}},
    }


def test_refinement_summary_uses_all_requested_frames() -> None:
    result = summarize_refinement_records(
        [_success(0.01), {"status": "failed"}],
        requested_frames=2,
    )
    assert result["n_valid_pose"] == 1
    assert result["n_failed_pose"] == 1
    assert result["threshold_metric_denominator"] == "all_requested_frames"


def test_success_without_add_is_rejected() -> None:
    with pytest.raises(ValueError, match="lacks keypoint ADD"):
        refinement_add_errors_m([{"status": "success"}])


def test_requested_count_cannot_discard_loaded_failed_records() -> None:
    with pytest.raises(ValueError, match="cannot be smaller"):
        summarize_refinement_records(
            [_success(0.01), {"status": "failed"}],
            requested_frames=1,
        )
