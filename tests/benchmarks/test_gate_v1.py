from __future__ import annotations

import numpy as np

from calibx.benchmarks.gate_v1 import (
    CameraModel,
    RefinementEvidence,
    is_proper_se3,
    mask_iou,
    pose_delta,
    select_refinement_candidate,
)


def _accepted_evidence() -> RefinementEvidence:
    return RefinementEvidence(
        finite_proper_se3=True,
        n_correspondences=20,
        n_inliers=20,
        inlier_ratio=1.0,
        positive_depth_ratio=1.0,
        reprojection_mean_px=1.0,
        reprojection_mean_diagonal_ratio=0.001,
        image_hull_area_ratio=0.1,
        world_second_spread_ratio=0.1,
        rotation_delta_deg=1.0,
        camera_center_delta_ratio=0.01,
        mask_iou_parent=None,
        mask_iou_candidate=None,
        V=True,
        O=True,
        L=True,
        accepted=True,
        reasons=(),
    )


def test_camera_model_uses_half_pixel_intrinsic_resizing() -> None:
    model = CameraModel.create(
        raw_size=(480, 640),
        process_size=(240, 320),
        K_raw=np.array([[400.0, 0.0, 319.5], [0.0, 400.0, 239.5], [0.0, 0.0, 1.0]]),
        distortion_raw=None,
        distortion_state="rectified",
    )
    assert model.K_process[0, 0] == 200.0
    assert model.K_process[0, 2] == 159.5


def test_gate_selects_candidate_only_when_evidence_accepts() -> None:
    assert select_refinement_candidate("parent", "candidate", _accepted_evidence(), two_cycle=False) == (
        "candidate",
        True,
        "gate_v1_accept",
    )
    assert select_refinement_candidate("parent", "candidate", _accepted_evidence(), two_cycle=True) == (
        "parent",
        False,
        "two_cycle",
    )


def test_pose_and_mask_primitives_are_explicit() -> None:
    identity = np.eye(4)
    shifted = np.eye(4)
    shifted[0, 3] = 0.1
    rotation, center_ratio = pose_delta(identity, shifted, robot_radius_m=1.0)
    assert rotation == 0.0
    assert center_ratio == 0.1
    assert is_proper_se3(identity)[0]
    assert mask_iou(np.array([[1, 0]]), np.array([[1, 1]])) == 0.5
