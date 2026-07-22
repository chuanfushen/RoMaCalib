from types import SimpleNamespace

import numpy as np

from scripts.run_droid_eval import validation_score


def test_validation_score_counts_sam3_no_mask_as_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("scripts.run_droid_eval.frame_indices", lambda *_args: (5,))
    config = {"render": {"width": 1280, "height": 720, "visual_geom_group": 2}}
    session = SimpleNamespace(session_id="episode__ext1")

    result = validation_score(
        config,
        session,
        tmp_path / "output",
        tmp_path / "masks",
        {},
        tmp_path / "validation",
        object(),
        object(),
        np.empty((6, 7)),
        np.empty(6),
    )

    assert result["frame_count"] == 1
    assert result["successful_frames"] == 0
    assert result["failed_frames"] == 1
    assert result["iou_macro"] == 0.0
    assert result["rows"][0]["failure_reason"] == "sam3_no_mask"
