from __future__ import annotations

import numpy as np
import pytest

from calibx.benchmarks.ctrnetx.single_executor import (
    CtrnetxSingleExecutionConfig,
    CtrnetxSingleHooks,
    CtrnetxSingleRuntimeFrame,
    CtrnetxSingleRuntimeManifest,
    CtrnetxSingleStage0Artifact,
    run_ctrnetx_single_frame,
)
from calibx.benchmarks.dream_real.legacy_runner import (
    LegacyEstimate,
    LegacyPoseArtifact,
    LegacyRefinementAttempt,
    LegacyStage1Artifact,
    legacy_match_seed,
)


def _pose(marker: int) -> LegacyPoseArtifact:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[0, 3] = -float(marker)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[0, 3] = float(marker)
    count = 6
    return LegacyPoseArtifact(
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        image_points=np.arange(count * 2, dtype=np.float64).reshape(count, 2),
        world_points=np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        scores=np.linspace(1.0, 0.5, count),
        inlier_indices=np.arange(count, dtype=np.int64),
        reprojection_errors=np.zeros(count, dtype=np.float64),
    )


def _estimate(marker: int) -> LegacyEstimate:
    pose = _pose(marker)
    return LegacyEstimate(
        pose=pose,
        pnp={
            "status": "success",
            "world_to_camera": pose.world_to_camera,
            "camera_to_robot_base": pose.camera_to_world,
        },
        render_path=f"runtime/render-{marker}.png",
        camera_npz=f"runtime/render-{marker}_camera.npz",
        keypoint_metrics={"summary": {"marker": marker}},
        pose_npz=f"runtime/pose-{marker}.npz",
    )


def _stage0(tmp_path, *, successful: bool, with_mask: bool = False):
    stage_dir = tmp_path / "stage0"
    stage_dir.mkdir()
    if with_mask:
        (stage_dir / "input_mask.png").write_bytes(b"test-mask")
    summary = {
        "status": "success" if successful else "failed",
        "views": [{"view": index} for index in range(6)],
    }
    legacy = LegacyStage1Artifact(
        summary=summary,
        frame_dir=stage_dir,
        estimate=_estimate(0) if successful else None,
        pose_path=stage_dir / "best_pose.npz" if successful else None,
    )
    return CtrnetxSingleStage0Artifact.from_legacy_stage1(
        legacy,
        artifact_id="stage0/frame-000017/frame_summary.json",
        input_mask_artifact_id=(
            "stage0/frame-000017/input_mask.png" if with_mask else None
        ),
    )


def _frame() -> CtrnetxSingleRuntimeFrame:
    return CtrnetxSingleRuntimeFrame(
        split="robot_in_view_full_body",
        frame_id="episode-00/frame-000017",
        frame_index=17,
        metadata={"camera": "azure", "ordinal": 17},
    )


def test_single_executor_replaces_r1_then_fails_r2_without_parent_retention(tmp_path) -> None:
    frame = _frame()
    stage0 = _stage0(tmp_path, successful=True)
    stage0_requests = []
    refinement_requests = []

    def run_stage0(request):
        stage0_requests.append(request)
        return stage0

    def refine_one(request):
        refinement_requests.append(request)
        if request.iteration == 1:
            return LegacyRefinementAttempt(
                render_path="runtime/r1/projected.png",
                camera_npz="runtime/r1/projected_camera.npz",
                match={"pnp": {"status": "success"}},
                candidate=_estimate(1),
            )
        if request.iteration == 2:
            return LegacyRefinementAttempt(
                render_path="runtime/r2/projected.png",
                camera_npz="runtime/r2/projected_camera.npz",
                match={"pnp": {"status": "too_few_correspondences"}},
                candidate=None,
            )
        raise AssertionError("R3 must be skipped after the R2 PnP failure")

    execution = run_ctrnetx_single_frame(
        frame,
        config=CtrnetxSingleExecutionConfig(),
        hooks=CtrnetxSingleHooks(run_stage0=run_stage0, refine_one=refine_one),
    )

    assert stage0_requests[0].views == 6
    assert stage0_requests[0].match_batch_size == 6
    assert stage0_requests[0].visual_geom_group == 2
    assert stage0_requests[0].match_seed == legacy_match_seed(90, 17, 0)
    assert [request.iteration for request in refinement_requests] == [1, 2]
    assert [request.match_seed for request in refinement_requests] == [
        legacy_match_seed(90, 17, 1),
        legacy_match_seed(90, 17, 2),
    ]
    assert execution.trajectory.frame_status == "failed"
    assert execution.trajectory.final_estimate is None
    assert execution.audit_record()["executed_refinement_match_seeds"] == [
        {"iteration": 1, "match_seed": legacy_match_seed(90, 17, 1)},
        {"iteration": 2, "match_seed": legacy_match_seed(90, 17, 2)},
    ]
    assert [record["status"] for record in execution.trajectory.iterations] == [
        "success",
        "success",
        "failed",
        "skipped",
    ]
    assert execution.audit_record()["iterations"][3]["status"] == "skipped"


def test_single_executor_stage0_failure_skips_every_refinement(tmp_path) -> None:
    stage0 = _stage0(tmp_path, successful=False)
    calls = []

    execution = run_ctrnetx_single_frame(
        _frame(),
        config=CtrnetxSingleExecutionConfig(),
        hooks=CtrnetxSingleHooks(
            run_stage0=lambda request: stage0,
            refine_one=lambda request: calls.append(request),
        ),
    )

    assert calls == []
    assert execution.trajectory.frame_status == "failed"
    assert [record["status"] for record in execution.trajectory.iterations] == [
        "failed",
        "skipped",
        "skipped",
        "skipped",
    ]


def test_single_executor_replays_existing_stage0_mask_without_sam(tmp_path) -> None:
    frame = _frame()
    stage0 = _stage0(tmp_path, successful=True, with_mask=True)
    requests = []

    def refine_one(request):
        requests.append(request)
        assert request.stage0.input_mask_source == "stage0_input_mask"
        assert request.runtime_input_mask_path == tmp_path / "stage0" / "input_mask.png"
        return LegacyRefinementAttempt(
            render_path=f"runtime/r{request.iteration}/projected.png",
            camera_npz=f"runtime/r{request.iteration}/projected_camera.npz",
            match={"pnp": {"status": "success"}},
            candidate=_estimate(request.iteration),
        )

    execution = run_ctrnetx_single_frame(
        frame,
        config=CtrnetxSingleExecutionConfig(),
        hooks=CtrnetxSingleHooks(
            run_stage0=lambda request: stage0,
            refine_one=refine_one,
        ),
    )

    assert [request.iteration for request in requests] == [1, 2, 3]
    audit = execution.audit_record()
    assert audit["stage0"]["input_mask"] == {
        "source": "stage0_input_mask",
        "artifact_id": "stage0/frame-000017/input_mask.png",
        "sam_rerun": False,
    }
    assert audit["frame_status"] == "success"


def test_single_executor_reports_runtime_exception_at_outer_frame_boundary(tmp_path) -> None:
    execution = run_ctrnetx_single_frame(
        _frame(),
        config=CtrnetxSingleExecutionConfig(),
        hooks=CtrnetxSingleHooks(
            run_stage0=lambda request: _stage0(tmp_path, successful=True),
            refine_one=lambda request: (_ for _ in ()).throw(RuntimeError("renderer")),
        ),
    )

    assert execution.trajectory is None
    audit = execution.audit_record()
    assert audit["frame_status"] == "error"
    assert audit["outer_error_type"] == "RuntimeError"
    assert audit["iterations"] == []


def test_single_executor_keeps_stage0_runtime_error_per_frame(tmp_path) -> None:
    refinement_calls = []

    execution = run_ctrnetx_single_frame(
        _frame(),
        config=CtrnetxSingleExecutionConfig(),
        hooks=CtrnetxSingleHooks(
            run_stage0=lambda request: (_ for _ in ()).throw(RuntimeError("stage0")),
            refine_one=lambda request: refinement_calls.append(request),
        ),
    )

    assert refinement_calls == []
    audit = execution.audit_record()
    assert audit["frame_status"] == "error"
    assert audit["stage0"]["pnp_success"] is None
    assert audit["executed_refinement_match_seeds"] == []


def test_single_runtime_manifest_rejects_host_paths_and_is_not_full_scope() -> None:
    manifest = CtrnetxSingleRuntimeManifest(frames=(_frame(),))
    assert len(manifest.sha256()) == 64
    with pytest.raises(ValueError, match="archived split counts"):
        manifest.validate_full_scope()

    unsafe = CtrnetxSingleRuntimeFrame(
        split="robot_in_view_full_body",
        frame_id="frame-0",
        frame_index=0,
        metadata={"private_data_root": r"C:\\server\\ctrnet-x"},
    )
    with pytest.raises(ValueError, match="host path"):
        unsafe.validate()
    with pytest.raises(ValueError, match="host path"):
        CtrnetxSingleRuntimeFrame(
            split="robot_in_view_full_body",
            frame_id="frame-1",
            frame_index=1,
            metadata={"private_uri": "file:///opt/data/private/frame.png"},
        ).validate()
