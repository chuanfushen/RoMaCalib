from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from romav2.benchmarks.utils.evaluator import summarize
from romav2.benchmarks.utils.geometry import deduplicate_pnp_correspondences
from romav2.benchmarks.utils.pose import (
    mujoco_camera_to_opencv_pose,
    opencv_pose_to_mujoco_camera,
)
from romav2.benchmarks.utils.refinement import (
    CameraModel,
    is_proper_se3,
    is_small_update,
    is_two_cycle,
    refinement_evidence,
    resize_camera_matrix_half_pixel,
    select_refinement_candidate,
    verify_frozen_specs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_gate_v1_hashes() -> None:
    verified = verify_frozen_specs(PROJECT_ROOT)
    assert set(verified) == {
        "gate_config",
        "gate_spec",
        "evaluation_spec",
        "freeze_index",
    }


def test_half_pixel_camera_resize() -> None:
    raw = np.array(
        [[500.0, 0.0, 319.5], [0.0, 510.0, 239.5], [0.0, 0.0, 1.0]]
    )
    resized = resize_camera_matrix_half_pixel(raw, (480, 640), (240, 320))
    np.testing.assert_allclose(
        resized,
        [[250.0, 0.0, 159.5], [0.0, 255.0, 119.5], [0.0, 0.0, 1.0]],
    )


def test_camera_model_rectified_round_trip() -> None:
    camera = CameraModel.create(
        raw_size=(480, 640),
        process_size=(240, 320),
        K_raw=np.array(
            [[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]]
        ),
        distortion_raw=None,
        distortion_state="rectified",
    )
    raw_points = np.array([[0.0, 0.0], [319.5, 239.5], [639.0, 479.0]])
    process_points = camera.raw_to_process_points(raw_points)
    recovered = camera.process_to_raw_points(process_points)
    np.testing.assert_allclose(recovered, raw_points, atol=1e-9)


def test_camera_model_distorted_round_trip() -> None:
    camera = CameraModel.create(
        raw_size=(480, 640),
        process_size=(480, 640),
        K_raw=np.array(
            [[520.0, 0.0, 319.5], [0.0, 515.0, 239.5], [0.0, 0.0, 1.0]]
        ),
        distortion_raw=np.array([-0.1, 0.03, 0.001, -0.002, 0.0]),
        distortion_state="distorted",
    )
    raw_points = np.array([[100.0, 120.0], [319.5, 239.5], [540.0, 360.0]])
    process_points = camera.raw_to_process_points(raw_points)
    recovered = camera.process_to_raw_points(process_points)
    np.testing.assert_allclose(recovered, raw_points, atol=1e-5)


def test_opencv_mujoco_pose_round_trip() -> None:
    rng = np.random.default_rng(20260717)
    for _ in range(20):
        rvec = rng.normal(size=3)
        rvec *= rng.uniform(0.0, np.pi) / max(np.linalg.norm(rvec), 1e-12)
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = rng.normal(size=3)
        position, quaternion = opencv_pose_to_mujoco_camera(transform)
        recovered = mujoco_camera_to_opencv_pose(position, quaternion)
        np.testing.assert_allclose(recovered, transform, atol=1e-10)


def test_proper_se3_checks() -> None:
    valid = np.eye(4)
    assert is_proper_se3(valid) == (True, [])
    invalid = valid.copy()
    invalid[0, 0] = 2.0
    accepted, reasons = is_proper_se3(invalid)
    assert not accepted
    assert "se3_rotation_orthogonality" in reasons
    assert "se3_rotation_determinant" in reasons


def _passing_pose() -> dict:
    x, y = np.meshgrid(np.linspace(-0.4, 0.4, 4), np.linspace(-0.3, 0.3, 3))
    world = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    image = np.column_stack(
        [
            np.linspace(120.0, 520.0, len(world)),
            np.tile([100.0, 240.0, 380.0], 4),
        ]
    )
    transform = np.eye(4)
    transform[2, 3] = 2.0
    return {
        "world_to_camera": transform,
        "world_points": world,
        "image_points": image,
        "inlier_indices": np.arange(len(world)),
        "reprojection_errors": np.ones(len(world)),
    }


def test_gate_v1_accepts_well_spread_candidate() -> None:
    pose = _passing_pose()
    observed_mask = np.zeros((480, 640), dtype=bool)
    observed_mask[100:380, 120:520] = True
    evidence = refinement_evidence(
        pose["world_to_camera"],
        pose,
        (480, 640),
        1.0,
        observed_mask,
        observed_mask,
        observed_mask,
    )
    assert evidence.V
    assert evidence.O
    assert evidence.L
    assert evidence.accepted
    assert is_small_update(evidence)


def test_gate_v1_rejects_mask_iou_drop() -> None:
    pose = _passing_pose()
    observed_mask = np.zeros((480, 640), dtype=bool)
    observed_mask[100:300, 100:300] = True
    candidate_mask = np.zeros_like(observed_mask)
    candidate_mask[320:470, 450:630] = True
    evidence = refinement_evidence(
        pose["world_to_camera"],
        pose,
        (480, 640),
        1.0,
        observed_mask,
        observed_mask,
        candidate_mask,
    )
    assert evidence.V
    assert evidence.O
    assert not evidence.L
    assert not evidence.accepted
    assert "mask_iou_drop_above_0.02" in evidence.reasons
    parent = object()
    candidate = object()
    selected, accepted, reason = select_refinement_candidate(
        parent,
        candidate,
        evidence,
        two_cycle=False,
    )
    assert selected is parent
    assert not accepted
    assert reason == "gate_v1_reject"


def test_two_cycle_uses_pose_two_steps_before_parent() -> None:
    transform = np.eye(4)
    assert is_two_cycle(transform, transform, 1.0)
    shifted = transform.copy()
    shifted[0, 3] = 0.01
    assert not is_two_cycle(shifted, transform, 1.0)


def test_dedup_preserves_deterministic_source_order() -> None:
    image = np.array([[10.0, 10.0], [10.5, 10.5], [30.0, 30.0]])
    world = np.column_stack([image, np.zeros(3)])
    scores = np.array([0.8, 0.8, 0.7])
    _, _, _, kept = deduplicate_pnp_correspondences(
        image,
        world,
        scores,
        radius=3.0,
        return_indices=True,
    )
    np.testing.assert_array_equal(kept, [0, 2])


def test_all_frame_auc_counts_failure_as_miss() -> None:
    success = {
        "status": "success",
        "keypoint_metrics": {
            "summary": {"keypoint_add_mean_m": 0.01, "pixel_error_mean": 5.0}
        },
        "best_pnp": {"inliers": 20, "inlier_reprojection_error_mean": 1.0},
    }
    failure = {"status": "failed"}
    summary = summarize(
        [success, failure],
        0.1,
        1e-4,
        all_frame_threshold_denominator=True,
    )
    assert summary["Baxter/evaluation_denominator"] == 2
    assert summary["ADD<20mm"] == 50.0
    assert summary["pixel_error<10px"] == 50.0
