from __future__ import annotations

import numpy as np
import pytest

from calibx.benchmarks.ctrnetx.batch_replay import (
    FrameEvaluationRequest,
    ReplayFrameAttempt,
    Stage1FrameResult,
    run_ctrnetx_closed_loop_batch_replay,
)
from calibx.benchmarks.ctrnetx.episode_pnp import (
    EpisodeFrameCorrespondences,
    EpisodePose,
)


EPISODE_ID = "episode-1"


def _correspondences(frame_id: str) -> EpisodeFrameCorrespondences:
    count = 6
    return EpisodeFrameCorrespondences(
        frame_id=frame_id,
        episode_id=EPISODE_ID,
        image_points=np.arange(count * 2, dtype=np.float64).reshape(count, 2),
        world_points=np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        scores=np.linspace(1.0, 0.5, count),
        camera_matrix=np.eye(3),
    )


def _pose(marker: int, source_frame_ids: tuple[str, ...]) -> EpisodePose:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = marker
    return EpisodePose(
        world_to_camera=transform,
        camera_matrix=np.eye(3),
        inlier_indices=np.arange(6, dtype=np.int64),
        source_frame_ids=source_frame_ids,
        correspondence_count=6,
    )


def test_batch_replay_uses_only_successful_stage1_sources_and_evaluates_all_frames() -> None:
    requested = ("f0", "f1", "f2")
    stage1 = (
        Stage1FrameResult("f0", EPISODE_ID, True, _correspondences("f0")),
        # A correspondence artifact may exist, but failed individual PnP must
        # never enter the i0 shared source set.
        Stage1FrameResult(
            "f1",
            EPISODE_ID,
            False,
            _correspondences("f1"),
            "individual_pnp_failed",
        ),
        Stage1FrameResult("f2", EPISODE_ID, True, _correspondences("f2")),
    )
    solver_sources: list[tuple[str, ...]] = []
    replay_calls: list[tuple[int, str]] = []
    evaluation_calls: list[tuple[int, str, str]] = []

    def solve(frames: tuple[EpisodeFrameCorrespondences, ...], **_kwargs: object) -> EpisodePose:
        source_ids = tuple(frame.frame_id for frame in frames)
        solver_sources.append(source_ids)
        return _pose(len(solver_sources), source_ids)

    def replay(parent: EpisodePose, frame_id: str, iteration: int) -> ReplayFrameAttempt:
        assert parent is not None
        replay_calls.append((iteration, frame_id))
        return ReplayFrameAttempt(
            frame_id=frame_id,
            episode_id=EPISODE_ID,
            correspondences=_correspondences(frame_id),
        )

    def evaluate(request: FrameEvaluationRequest) -> str:
        evaluation_calls.append(
            (request.iteration, request.frame_id, request.status)
        )
        return request.status

    result = run_ctrnetx_closed_loop_batch_replay(
        EPISODE_ID,
        requested,
        stage1,
        replay_frame=replay,
        evaluate_frame=evaluate,
        episode_pnp_solver=solve,
    )

    assert solver_sources == [
        ("f0", "f2"),
        ("f0", "f1", "f2"),
        ("f0", "f1", "f2"),
        ("f0", "f1", "f2"),
    ]
    assert replay_calls == [
        (iteration, frame_id)
        for iteration in (1, 2, 3)
        for frame_id in requested
    ]
    assert evaluation_calls == [
        (iteration, frame_id, "updated")
        for iteration in (0, 1, 2, 3)
        for frame_id in requested
    ]
    assert result.rounds[0].source_frame_ids == ("f0", "f2")
    assert any(
        event.code == "stage1_pnp_unsuccessful_excluded" and event.frame_id == "f1"
        for event in result.recovery_events
    )
    assert any(
        record["code"] == "stage1_pnp_unsuccessful_excluded"
        and record["action"] == "exclude_from_i0_episode_pnp_but_evaluate"
        for record in result.recovery_audit_records()
    )


def test_batch_replay_retains_parent_after_solver_failure_without_skipping_frames() -> None:
    requested = ("f0", "f1")
    stage1 = tuple(
        Stage1FrameResult(frame_id, EPISODE_ID, True, _correspondences(frame_id))
        for frame_id in requested
    )
    solver_sources: list[tuple[str, ...]] = []
    replay_calls: list[tuple[int, str]] = []
    evaluation_calls: list[tuple[int, str, str, float]] = []

    def solve(frames: tuple[EpisodeFrameCorrespondences, ...], **_kwargs: object) -> EpisodePose:
        source_ids = tuple(frame.frame_id for frame in frames)
        solver_sources.append(source_ids)
        if len(solver_sources) == 2:
            raise RuntimeError("/data/private/insufficient episode inliers")
        return _pose(len(solver_sources), source_ids)

    def replay(parent: EpisodePose, frame_id: str, iteration: int) -> ReplayFrameAttempt:
        replay_calls.append((iteration, frame_id))
        if iteration == 1 and frame_id == "f1":
            return ReplayFrameAttempt(
                frame_id=frame_id,
                episode_id=EPISODE_ID,
                correspondences=None,
                failure_reason="no_raw_post_geometry_matches",
            )
        return ReplayFrameAttempt(
            frame_id=frame_id,
            episode_id=EPISODE_ID,
            correspondences=_correspondences(frame_id),
        )

    def evaluate(request: FrameEvaluationRequest) -> str:
        assert request.selected_pose is not None
        evaluation_calls.append(
            (
                request.iteration,
                request.frame_id,
                request.status,
                float(request.selected_pose.world_to_camera[0, 3]),
            )
        )
        return request.status

    result = run_ctrnetx_closed_loop_batch_replay(
        EPISODE_ID,
        requested,
        stage1,
        replay_frame=replay,
        evaluate_frame=evaluate,
        episode_pnp_solver=solve,
    )

    retained = result.rounds[1]
    assert retained.status == "retained_parent"
    assert retained.selected_pose is retained.parent_pose
    assert retained.candidate_pose is None
    assert retained.solver_failure_reason == "episode_pnp_runtime_error"
    assert solver_sources[1] == ("f0",)
    assert replay_calls == [
        (iteration, frame_id)
        for iteration in (1, 2, 3)
        for frame_id in requested
    ]
    assert {(iteration, frame_id) for iteration, frame_id, _status, _marker in evaluation_calls} == {
        (iteration, frame_id)
        for iteration in (0, 1, 2, 3)
        for frame_id in requested
    }
    assert any(
        event.code == "replay_correspondences_unavailable"
        and event.frame_id == "f1"
        and event.reason == "no_raw_post_geometry_matches"
        for event in result.recovery_events
    )
    assert any(
        event.code == "episode_pnp_failed"
        and event.iteration == 1
        and event.action == "retain_parent_for_all_frame_evaluation"
        for event in result.recovery_events
    )


def test_batch_replay_marks_later_rounds_skipped_when_no_parent_pose_exists() -> None:
    requested = ("f0", "f1")
    stage1 = tuple(
        Stage1FrameResult(frame_id, EPISODE_ID, False, failure_reason="stage1_failed")
        for frame_id in requested
    )
    replay_calls: list[tuple[int, str]] = []
    evaluation_calls: list[tuple[int, str, str]] = []

    def solve(_frames: tuple[EpisodeFrameCorrespondences, ...], **_kwargs: object) -> EpisodePose:
        raise RuntimeError("no stage1 sources")

    def replay(_parent: EpisodePose, frame_id: str, iteration: int) -> ReplayFrameAttempt:
        replay_calls.append((iteration, frame_id))
        raise AssertionError("replay must not invent a pose-aligned render without parent")

    def evaluate(request: FrameEvaluationRequest) -> str:
        evaluation_calls.append(
            (request.iteration, request.frame_id, request.status)
        )
        return request.status

    result = run_ctrnetx_closed_loop_batch_replay(
        EPISODE_ID,
        requested,
        stage1,
        replay_frame=replay,
        evaluate_frame=evaluate,
        episode_pnp_solver=solve,
    )

    assert replay_calls == []
    assert [round_.status for round_ in result.rounds] == [
        "solver_failed",
        "skipped_no_parent",
        "skipped_no_parent",
        "skipped_no_parent",
    ]
    assert evaluation_calls == [
        *[(0, frame_id, "solver_failed") for frame_id in requested],
        *[
            (iteration, frame_id, "skipped_no_parent")
            for iteration in (1, 2, 3)
            for frame_id in requested
        ],
    ]
    assert sum(event.code == "parent_pose_unavailable" for event in result.recovery_events) == 6


def test_batch_replay_exports_artifact_recovery_audit_for_cached_render() -> None:
    requested = ("f0", "f1")
    stage1 = tuple(
        Stage1FrameResult(frame_id, EPISODE_ID, True, _correspondences(frame_id))
        for frame_id in requested
    )

    def solve(
        frames: tuple[EpisodeFrameCorrespondences, ...], **_kwargs: object
    ) -> EpisodePose:
        return _pose(1, tuple(frame.frame_id for frame in frames))

    def replay(
        _parent: EpisodePose, frame_id: str, _iteration: int
    ) -> ReplayFrameAttempt:
        if frame_id == "f0":
            return ReplayFrameAttempt(
                frame_id=frame_id,
                episode_id=EPISODE_ID,
                correspondences=_correspondences(frame_id),
                artifact_id="renders/episode-1/f0/projected.png",
                replacement_artifact_id="renders/episode-1/f0/projected_retry.png",
                artifact_recovery_state="recompute_cached_render",
            )
        return ReplayFrameAttempt(
            frame_id=frame_id,
            episode_id=EPISODE_ID,
            correspondences=_correspondences(frame_id),
        )

    result = run_ctrnetx_closed_loop_batch_replay(
        EPISODE_ID,
        requested,
        stage1,
        replay_frame=replay,
        evaluate_frame=lambda request: request.status,
        episode_pnp_solver=solve,
    )

    events = [
        event
        for event in result.recovery_audit_records()
        if event["code"] == "render_artifact_recovery"
    ]
    assert len(events) == 3
    assert all(event["artifact_recovery_state"] == "recompute_cached_render" for event in events)
    assert all(event["artifact_id"] == "renders/episode-1/f0/projected.png" for event in events)


def test_batch_replay_rejects_missing_stage1_frame_before_execution() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        run_ctrnetx_closed_loop_batch_replay(
            EPISODE_ID,
            ("f0", "f1"),
            (Stage1FrameResult("f0", EPISODE_ID, False, failure_reason="failed"),),
            replay_frame=lambda _pose, _frame, _iteration: pytest.fail("must not replay"),
            evaluate_frame=lambda _request: pytest.fail("must not evaluate"),
            episode_pnp_solver=lambda _frames, **_kwargs: pytest.fail("must not solve"),
        )


def test_batch_replay_records_explicit_round_pnp_and_match_seeds() -> None:
    requested = ("f0", "f1")
    stage1 = tuple(
        Stage1FrameResult(frame_id, EPISODE_ID, True, _correspondences(frame_id))
        for frame_id in requested
    )
    solver_seeds: list[int] = []

    def solve(
        frames: tuple[EpisodeFrameCorrespondences, ...], **kwargs: object
    ) -> EpisodePose:
        solver_seeds.append(int(kwargs["seed"]))
        return _pose(len(solver_seeds), tuple(frame.frame_id for frame in frames))

    def replay(
        _parent: EpisodePose, frame_id: str, iteration: int
    ) -> ReplayFrameAttempt:
        return ReplayFrameAttempt(
            frame_id=frame_id,
            episode_id=EPISODE_ID,
            correspondences=_correspondences(frame_id),
            match_seed=10_000 + iteration,
        )

    result = run_ctrnetx_closed_loop_batch_replay(
        EPISODE_ID,
        requested,
        stage1,
        replay_frame=replay,
        evaluate_frame=lambda request: request.status,
        episode_pnp_solver=solve,
        pnp_seed_for_iteration=lambda iteration: 90 + iteration,
    )

    assert solver_seeds == [90, 91, 92, 93]
    assert [round_.pnp_seed for round_ in result.rounds] == [90, 91, 92, 93]
    assert [attempt.match_seed for attempt in result.rounds[1].replay_attempts] == [
        10_001,
        10_001,
    ]
