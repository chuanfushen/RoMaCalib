from __future__ import annotations

import numpy as np

from calibx.benchmarks.gate_trajectory import CandidateAttempt, run_gate_v1_trajectory
from calibx.benchmarks.gate_v1 import RefinementEvidence


def _evidence(*, accepted: bool = True, small: bool = False) -> RefinementEvidence:
    rotation = 0.05 if small else 1.0
    center = 0.0005 if small else 0.01
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
        rotation_delta_deg=rotation,
        camera_center_delta_ratio=center,
        mask_iou_parent=None,
        mask_iou_candidate=None,
        V=accepted,
        O=accepted,
        L=accepted,
        accepted=accepted,
        reasons=() if accepted else ("gate_reject",),
    )


def _pose(value: float) -> dict:
    transform = np.eye(4)
    transform[0, 3] = value
    return {"world_to_camera": transform}


def test_rejection_keeps_parent_and_fills_remaining_iterations() -> None:
    trajectory = run_gate_v1_trajectory(
        _pose(0.0),
        iterations=3,
        candidate_attempt=lambda parent, iteration: CandidateAttempt(_pose(0.1), _evidence(accepted=False)),
        pose_of=lambda pose: pose["world_to_camera"],
        robot_radius_m=1.0,
    )
    assert np.array_equal(
        trajectory.final["world_to_camera"],
        trajectory.initial["world_to_camera"],
    )
    assert [step.status for step in trajectory.steps] == ["rejected", "filled", "filled"]


def test_small_update_is_accepted_then_fills_remaining_iterations() -> None:
    trajectory = run_gate_v1_trajectory(
        _pose(0.0),
        iterations=3,
        candidate_attempt=lambda parent, iteration: CandidateAttempt(_pose(0.01), _evidence(small=True)),
        pose_of=lambda pose: pose["world_to_camera"],
        robot_radius_m=1.0,
    )
    assert trajectory.final["world_to_camera"][0, 3] == 0.01
    assert [step.status for step in trajectory.steps] == ["accepted", "filled", "filled"]
    assert trajectory.stop_reason == "small_update"
