from __future__ import annotations

import numpy as np
import pytest

from calibx.benchmarks.ctrnetx.episode_pnp import (
    EpisodeFrameCorrespondences,
    assert_complete_frame_coverage,
    balanced_episode_shards,
    concatenate_topk_correspondences,
)


def _frame(frame_id: str, scores: list[float]) -> EpisodeFrameCorrespondences:
    count = len(scores)
    return EpisodeFrameCorrespondences(
        frame_id=frame_id,
        episode_id="episode-1",
        image_points=np.arange(count * 2, dtype=np.float64).reshape(count, 2),
        world_points=np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        scores=np.asarray(scores, dtype=np.float64),
        camera_matrix=np.eye(3),
    )


def test_episode_correspondences_use_stable_score_topk() -> None:
    frame = _frame("frame-1", [0.5, 0.9, 0.9, 0.1])
    selected = frame.top_k(2)
    assert selected.scores.tolist() == [0.9, 0.9]
    assert selected.image_points[:, 0].tolist() == [2.0, 4.0]


def test_concatenation_requires_common_episode_and_camera() -> None:
    image_points, world_points, scores, source_ids, camera_matrix = concatenate_topk_correspondences(
        [_frame("frame-1", [0.9, 0.2]), _frame("frame-2", [0.8, 0.1])],
        topk_per_frame=1,
        min_source_frames=2,
    )
    assert image_points.shape == (2, 2)
    assert world_points.shape == (2, 3)
    assert scores.tolist() == [0.9, 0.8]
    assert source_ids == ("frame-1", "frame-2")
    assert np.array_equal(camera_matrix, np.eye(3))


def test_concatenation_returns_camera_from_first_nonempty_source_frame() -> None:
    leading_empty = _frame("frame-empty", [])
    leading_empty = EpisodeFrameCorrespondences(
        **{**leading_empty.__dict__, "camera_matrix": np.diag([2.0, 2.0, 1.0])}
    )
    image_points, _world_points, _scores, source_ids, camera_matrix = (
        concatenate_topk_correspondences(
            [leading_empty, _frame("frame-1", [0.9]), _frame("frame-2", [0.8])],
            topk_per_frame=1,
            min_source_frames=2,
        )
    )
    assert image_points.shape == (2, 2)
    assert source_ids == ("frame-1", "frame-2")
    assert np.array_equal(camera_matrix, np.eye(3))


def test_largest_first_scheduler_is_deterministic() -> None:
    shards = balanced_episode_shards({"a": 10, "b": 8, "c": 5, "d": 5}, 2)
    assert shards == (("a", "d"), ("b", "c"))


def test_coverage_rejects_missing_requested_frame() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        assert_complete_frame_coverage(["a", "b"], ["a"])
