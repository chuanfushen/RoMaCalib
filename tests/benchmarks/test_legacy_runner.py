from __future__ import annotations

import json

import numpy as np

from calibx.benchmarks.dream_real.legacy_runner import (
    LEGACY_REFINEMENT_SEMANTIC_ID,
    LegacyEstimate,
    LegacyPoseArtifact,
    LegacyRefinementAttempt,
    LegacyStage1Artifact,
    legacy_match_seed,
    load_legacy_stage1_artifact,
    run_legacy_unconditional_trajectory,
    save_legacy_pose_npz,
)


def _pose(translation_x: float) -> LegacyPoseArtifact:
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[0, 3] = translation_x
    world_to_camera = np.linalg.inv(camera_to_world)
    return LegacyPoseArtifact(
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        image_points=np.array([[1.0, 2.0], [3.0, 4.0]]),
        world_points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        scores=np.array([0.9, 0.8]),
        inlier_indices=np.array([0, 1], dtype=np.int64),
        reprojection_errors=np.array([0.1, 0.2]),
    )


def _estimate(translation_x: float, *, label: str) -> LegacyEstimate:
    pose = _pose(translation_x)
    return LegacyEstimate(
        pose=pose,
        pnp={
            "status": "success",
            "world_to_camera": pose.world_to_camera.tolist(),
            "camera_to_robot_base": pose.camera_to_world.tolist(),
            "inliers": 2,
        },
        render_path=f"renders/{label}.png",
        camera_npz=f"renders/{label}_camera.npz",
        keypoint_metrics={
            "summary": {"pixel_error_mean": 1.0, "keypoint_add_mean_m": 0.01}
        },
        pose_npz=f"poses/{label}.npz",
        keypoint_visualization=f"visuals/{label}.jpg",
    )


def _stage1(estimate: LegacyEstimate | None) -> LegacyStage1Artifact:
    if estimate is None:
        return LegacyStage1Artifact(
            summary={"status": "failed", "reason": "No R0 PnP."},
            frame_dir="stage1/000007",
            estimate=None,
            pose_path=None,
        )
    return LegacyStage1Artifact(
        summary={"status": "success", "views": [{"view": 0}]},
        frame_dir="stage1/000007",
        estimate=estimate,
        pose_path="stage1/000007/best_pose.npz",
    )


def test_legacy_runner_replaces_each_successful_parent_without_a_gate() -> None:
    first = _estimate(0.0, label="r0")
    candidates = [_estimate(1.0, label="r1"), _estimate(2.0, label="r2")]
    parents: list[float] = []

    def attempt(parent: LegacyEstimate, iteration: int, seed: int | None) -> LegacyRefinementAttempt:
        parents.append(float(parent.pose.camera_to_world[0, 3]))
        return LegacyRefinementAttempt(
            render_path=f"projected/r{iteration}.png",
            match={"pnp": {"status": "success"}, "_pose": "private"},
            candidate=candidates[iteration - 1],
        )

    trajectory = run_legacy_unconditional_trajectory(
        _stage1(first),
        frame_index=7,
        refinement_iterations=2,
        match_seed=90,
        refinement_attempt=attempt,
    )

    assert trajectory.semantic_id == LEGACY_REFINEMENT_SEMANTIC_ID
    assert trajectory.frame_status == "success"
    assert parents == [0.0, 1.0]
    assert trajectory.final_estimate is candidates[-1]
    assert [record["status"] for record in trajectory.iterations] == [
        "success",
        "success",
        "success",
    ]
    assert trajectory.iterations[0]["best_camera_npz"] == "renders/r0_camera.npz"
    assert "camera_npz" not in trajectory.iterations[0]
    assert trajectory.iterations[1]["match_seed"] == legacy_match_seed(90, 7, 1)
    assert "_pose" not in trajectory.iterations[1]["match"]
    assert "accepted" not in trajectory.iterations[1]


def test_legacy_runner_marks_the_frame_failed_after_one_failed_refinement() -> None:
    first = _estimate(0.0, label="r0")
    second = _estimate(1.0, label="r1")
    calls: list[int] = []

    def attempt(parent: LegacyEstimate, iteration: int, seed: int | None) -> LegacyRefinementAttempt:
        del parent, seed
        calls.append(iteration)
        if iteration == 1:
            return LegacyRefinementAttempt(
                render_path="projected/r1.png",
                match={"pnp": {"status": "success"}},
                candidate=second,
            )
        return LegacyRefinementAttempt(
            render_path="projected/r2.png",
            match={"pnp": {"status": "failed", "error": "no PnP"}},
        )

    trajectory = run_legacy_unconditional_trajectory(
        _stage1(first),
        frame_index=7,
        refinement_iterations=3,
        match_seed=90,
        refinement_attempt=attempt,
    )

    assert calls == [1, 2]
    assert trajectory.frame_status == "failed"
    assert trajectory.final_estimate is None
    assert [record["status"] for record in trajectory.iterations] == [
        "success",
        "success",
        "failed",
        "skipped",
    ]
    assert "match_seed" not in trajectory.iterations[2]
    assert trajectory.iterations[3]["reason"] == (
        "The previous iteration did not produce a valid PnP pose."
    )


def test_stage1_failure_skips_all_refinement_callbacks() -> None:
    def should_not_run(*_args: object) -> LegacyRefinementAttempt:
        raise AssertionError("a failed Stage1 must not enter refinement")

    trajectory = run_legacy_unconditional_trajectory(
        _stage1(None),
        frame_index=7,
        refinement_iterations=3,
        match_seed=90,
        refinement_attempt=should_not_run,
    )

    assert trajectory.frame_status == "failed"
    assert [record["status"] for record in trajectory.iterations] == [
        "failed",
        "skipped",
        "skipped",
        "skipped",
    ]


def test_legacy_runner_records_an_unexpected_callback_error_and_stops() -> None:
    def explode(*_args: object) -> LegacyRefinementAttempt:
        raise RuntimeError("render backend unavailable")

    trajectory = run_legacy_unconditional_trajectory(
        _stage1(_estimate(0.0, label="r0")),
        frame_index=7,
        refinement_iterations=3,
        match_seed=90,
        refinement_attempt=explode,
    )

    assert trajectory.frame_status == "error"
    assert trajectory.final_estimate is None
    assert [record["status"] for record in trajectory.iterations] == [
        "success",
        "error",
        "skipped",
        "skipped",
    ]
    assert trajectory.iterations[1]["error"] == "RuntimeError('render backend unavailable')"
    assert "RuntimeError: render backend unavailable" in trajectory.iterations[1]["traceback"]


def test_stage1_loader_round_trips_the_historical_npz_schema(tmp_path) -> None:
    frame_dir = tmp_path / "stage1" / "000007"
    estimate = _estimate(0.0, label="r0")
    pose_path = save_legacy_pose_npz(
        frame_dir,
        estimate.pose,
        np.eye(3),
        estimate.keypoint_metrics,
        filename="archived_pose.npz",
    )
    (frame_dir / "frame_summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "pose_npz": str(pose_path),
                "best_render_path": estimate.render_path,
                "best_camera_npz": estimate.camera_npz,
                "best_pnp": estimate.pnp,
                "keypoint_metrics": estimate.keypoint_metrics,
                "views": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_legacy_stage1_artifact(tmp_path / "stage1", 7)

    assert loaded.pose_path == pose_path
    assert loaded.estimate is not None
    assert np.array_equal(loaded.estimate.pose.image_points, estimate.pose.image_points)
    assert loaded.estimate.pnp["status"] == "success"
