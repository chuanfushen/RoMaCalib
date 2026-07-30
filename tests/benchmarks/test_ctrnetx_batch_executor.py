from __future__ import annotations

import numpy as np
import pytest

from calibx.benchmarks.ctrnetx.batch_executor import (
    CtrnetxBatchExecutionConfig,
    CtrnetxBatchHooks,
    CtrnetxFrameRuntimeFailure,
    CtrnetxRuntimeEpisode,
    CtrnetxRuntimeFrame,
    CtrnetxRuntimeFrameManifest,
    PoseAlignedRenderArtifact,
    PoseAlignedRenderRequest,
    RawPostGeometryMatchArtifact,
    RawPostGeometryMatchRequest,
    Stage1FrameArtifact,
    Stage1Request,
    ctrnetx_batch_pnp_seed,
    ctrnetx_match_seed,
    raw_post_geometry_from_core_best_view,
    run_ctrnetx_batch_episode,
    stage1_artifact_from_core_best_view,
)
from calibx.benchmarks.ctrnetx.batch_replay import (
    RecoveryEvent,
    ReplayFrameAttempt,
    Stage1FrameResult,
)
from calibx.benchmarks.ctrnetx.episode_pnp import (
    EpisodeFrameCorrespondences,
    EpisodePose,
)


EPISODE_ID = "episode-03"


def _episode() -> CtrnetxRuntimeEpisode:
    return CtrnetxRuntimeEpisode(
        episode_id=EPISODE_ID,
        frames=(
            CtrnetxRuntimeFrame(
                episode_id=EPISODE_ID,
                frame_id="camera-a/000002",
                frame_ordinal=2,
                metadata={"camera": "camera-a", "sample": 2},
            ),
            CtrnetxRuntimeFrame(
                episode_id=EPISODE_ID,
                frame_id="camera-a/000005",
                frame_ordinal=5,
                metadata={"camera": "camera-a", "sample": 5},
            ),
        ),
    )


def _correspondences(frame: CtrnetxRuntimeFrame) -> EpisodeFrameCorrespondences:
    count = 6
    return EpisodeFrameCorrespondences(
        frame_id=frame.frame_id,
        episode_id=frame.episode_id,
        image_points=np.arange(count * 2, dtype=np.float64).reshape(count, 2),
        world_points=np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        scores=np.linspace(1.0, 0.5, count),
        camera_matrix=np.eye(3),
    )


def _pose(marker: int, source_ids: tuple[str, ...]) -> EpisodePose:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = marker
    return EpisodePose(
        world_to_camera=transform,
        camera_matrix=np.eye(3),
        inlier_indices=np.arange(6, dtype=np.int64),
        source_frame_ids=source_ids,
        correspondence_count=6,
    )


def test_batch_executor_records_auditable_seed_schedule_and_all_frame_evaluation() -> None:
    episode = _episode()
    stage1_requests = []
    replay_requests = []
    matcher_requests = []
    solver_seeds = []
    evaluations = []

    def run_stage1(request):
        stage1_requests.append(request)
        return Stage1FrameArtifact(
            result=Stage1FrameResult(
                frame_id=request.frame.frame_id,
                episode_id=request.frame.episode_id,
                pnp_success=True,
                correspondences=_correspondences(request.frame),
            ),
            artifact_id=f"stage1/{request.frame.frame_id}/best.json",
        )

    def render(request):
        replay_requests.append(request)
        return PoseAlignedRenderArtifact(
            episode_id=request.frame.episode_id,
            frame_id=request.frame.frame_id,
            artifact_id=(
                f"replay/{request.frame.frame_id}/iteration_{request.iteration:02d}.png"
            ),
            handle={"test_only": True},
        )

    def match(request):
        matcher_requests.append(request)
        assert request.forbid_frame_level_pnp is True
        assert request.correspondence_space == "raw_post_geometry"
        return RawPostGeometryMatchArtifact(
            correspondences=_correspondences(request.frame),
        )

    def solve(frames, **kwargs):
        solver_seeds.append(kwargs["seed"])
        return _pose(len(solver_seeds), tuple(frame.frame_id for frame in frames))

    def evaluate(request):
        evaluations.append(
            (request.replay_request.iteration, request.frame.frame_id)
        )
        return request.replay_request.status

    execution = run_ctrnetx_batch_episode(
        episode,
        episode_ordinal=3,
        config=CtrnetxBatchExecutionConfig(),
        hooks=CtrnetxBatchHooks(
            run_stage1=run_stage1,
            render_pose_aligned=render,
            match_raw_post_geometry=match,
            evaluate_frame=evaluate,
        ),
        episode_pnp_solver=solve,
    )

    assert [request.match_seed for request in stage1_requests] == [
        ctrnetx_match_seed(90, 2, 0),
        ctrnetx_match_seed(90, 5, 0),
    ]
    assert [(request.iteration, request.match_seed) for request in replay_requests] == [
        (iteration, ctrnetx_match_seed(90, frame.frame_ordinal, iteration))
        for iteration in (1, 2, 3)
        for frame in episode.frames
    ]
    expected_pnp = [ctrnetx_batch_pnp_seed(90, 3, iteration) for iteration in range(4)]
    assert solver_seeds == expected_pnp
    assert list(execution.pnp_seed_by_iteration) == expected_pnp
    assert [round_.pnp_seed for round_ in execution.replay.rounds] == expected_pnp
    assert evaluations == [
        (iteration, frame.frame_id)
        for iteration in range(4)
        for frame in episode.frames
    ]

    audit = execution.audit_record()
    assert audit["rounds"][1]["pnp_seed"] == expected_pnp[1]
    assert audit["stage1"][0]["failure_reason"] is None
    assert [row["match_seed"] for row in audit["rounds"][1]["replay_attempts"]] == [
        ctrnetx_match_seed(90, frame.frame_ordinal, 1)
        for frame in episode.frames
    ]


def test_runtime_frame_manifest_rejects_host_paths_in_metadata() -> None:
    manifest = CtrnetxRuntimeFrameManifest(episodes=(_episode(),))
    assert len(manifest.sha256()) == 64

    unsafe = CtrnetxRuntimeFrame(
        episode_id=EPISODE_ID,
        frame_id="frame-0",
        frame_ordinal=0,
        metadata={"private_path": r"C:\\server\\dataset"},
    )
    with pytest.raises(ValueError, match="host path"):
        unsafe.validate()
    with pytest.raises(ValueError, match="host path"):
        CtrnetxRuntimeFrame(
            episode_id=EPISODE_ID,
            frame_id="frame-1",
            frame_ordinal=1,
            metadata={"private_uri": "file:///opt/data/private/frame.png"},
        ).validate()
    with pytest.raises(ValueError, match="unsafe metadata key"):
        CtrnetxRuntimeFrame(
            episode_id=EPISODE_ID,
            frame_id="frame-2",
            frame_ordinal=2,
            metadata={"/opt/data/private": "opaque"},
        ).validate()
    with pytest.raises(ValueError, match="portable relative"):
        Stage1FrameArtifact(
            result=Stage1FrameResult(
                frame_id="frame-3",
                episode_id=EPISODE_ID,
                pnp_success=False,
                failure_reason="stage1_unavailable",
            ),
            artifact_id="ssh://root@192.168.21.57/opt/data/private/stage1.npz",
        ).validate()


def test_batch_executor_turns_a_recoverable_stage1_failure_into_all_frame_coverage() -> None:
    episode = _episode()
    evaluations: list[tuple[int, str, str]] = []

    def run_stage1(request):
        if request.frame.frame_id == episode.frames[0].frame_id:
            raise CtrnetxFrameRuntimeFailure(
                "stage1_render_unavailable",
                artifact_id="stage1/camera-a/000002",
            )
        return Stage1FrameArtifact(
            result=Stage1FrameResult(
                frame_id=request.frame.frame_id,
                episode_id=request.frame.episode_id,
                pnp_success=True,
                correspondences=_correspondences(request.frame),
            )
        )

    execution = run_ctrnetx_batch_episode(
        episode,
        episode_ordinal=0,
        config=CtrnetxBatchExecutionConfig(),
        hooks=CtrnetxBatchHooks(
            run_stage1=run_stage1,
            render_pose_aligned=lambda _request: pytest.fail("no parent should replay"),
            match_raw_post_geometry=lambda _request: pytest.fail("no parent should match"),
            evaluate_frame=lambda request: evaluations.append(
                (
                    request.replay_request.iteration,
                    request.frame.frame_id,
                    request.replay_request.status,
                )
            ),
        ),
    )

    assert execution.stage1_artifacts[0].result.pnp_success is False
    assert execution.stage1_artifacts[0].result.failure_reason == "stage1_render_unavailable"
    assert [round_.status for round_ in execution.replay.rounds] == [
        "solver_failed",
        "skipped_no_parent",
        "skipped_no_parent",
        "skipped_no_parent",
    ]
    assert evaluations == [
        (iteration, frame.frame_id, status)
        for iteration, status in (
            (0, "solver_failed"),
            (1, "skipped_no_parent"),
            (2, "skipped_no_parent"),
            (3, "skipped_no_parent"),
        )
        for frame in episode.frames
    ]
    assert any(
        event.reason == "stage1_render_unavailable"
        for event in execution.replay.recovery_events
    )


def test_batch_public_failure_fields_reject_local_paths() -> None:
    with pytest.raises(ValueError, match="failure code"):
        CtrnetxFrameRuntimeFailure("render failed at /data/private/frame.png")
    with pytest.raises(ValueError, match="failure_reason"):
        Stage1FrameResult(
            "frame-0",
            EPISODE_ID,
            False,
            failure_reason=r"C:\\private\\frame.png",
        ).validate()
    with pytest.raises(ValueError, match="failure_reason"):
        ReplayFrameAttempt(
            "frame-0",
            EPISODE_ID,
            None,
            failure_reason="no raw pairs at /data/private",
        ).validate()
    with pytest.raises(ValueError, match="failure_reason"):
        RawPostGeometryMatchArtifact(
            correspondences=None,
            failure_reason="no pairs at /data/private",
        ).validate(frame=_episode().frames[0])
    with pytest.raises(ValueError, match="recovery reason"):
        RecoveryEvent(
            iteration=0,
            scope="episode",
            code="episode_pnp_failed",
            action="record_failure",
            reason="RuntimeError: C:/private/output",
        ).validate()


def test_core_record_adapters_preserve_stage1_vs_replay_correspondence_sources() -> None:
    frame = _episode().frames[0]
    stage1_request = Stage1Request(
        frame=frame,
        match_seed=ctrnetx_match_seed(90, 2, 0),
    )
    best_stage1 = {
        "pnp": {"status": "success"},
        "_pose": {
            "image_points": np.full((2, 2), 4.0),
            "world_points": np.full((2, 3), 5.0),
            "scores": np.full(2, 6.0),
        },
        "_image_points": np.full((5, 2), 1.0),
        "_world_points": np.full((5, 3), 2.0),
        "_scores": np.full(5, 3.0),
    }
    stage1 = stage1_artifact_from_core_best_view(
        stage1_request,
        best_stage1,
        np.eye(3),
    )
    assert stage1.result.pnp_success is True
    assert len(stage1.result.correspondences.scores) == 2
    np.testing.assert_array_equal(
        stage1.result.correspondences.image_points,
        best_stage1["_pose"]["image_points"],
    )

    parent = _pose(1, (frame.frame_id,))
    render_request = PoseAlignedRenderRequest(
        frame=frame,
        parent_pose=parent,
        iteration=1,
        match_seed=ctrnetx_match_seed(90, frame.frame_ordinal, 1),
    )
    render = PoseAlignedRenderArtifact(
        episode_id=frame.episode_id,
        frame_id=frame.frame_id,
        artifact_id="replay/frame.png",
    )
    replay_request = RawPostGeometryMatchRequest(
        frame=frame,
        parent_pose=parent,
        iteration=1,
        match_seed=render_request.match_seed,
        render=render,
    )
    best_replay = {
        "pnp": {"status": "too_few_correspondences"},
        "_image_points": np.full((5, 2), 1.0),
        "_world_points": np.full((5, 3), 2.0),
        "_scores": np.full(5, 3.0),
    }
    replay = raw_post_geometry_from_core_best_view(
        replay_request,
        best_replay,
        np.eye(3),
    )
    assert replay.correspondences is not None
    assert len(replay.correspondences.scores) == 5

    with pytest.raises(ValueError, match="frame-level PnP success"):
        raw_post_geometry_from_core_best_view(
            replay_request,
            {**best_replay, "pnp": {"status": "success"}},
            np.eye(3),
        )
