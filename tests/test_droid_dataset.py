from __future__ import annotations

import numpy as np

from romav2.benchmarks.droid import (
    binary_mask_metrics,
    camera_matrix_from_annotation,
    camera_to_base_vector_to_world_to_camera,
    make_frame_split,
    select_best_iteration,
    solve_shared_pose,
)


def test_camera_matrix_annotation_order_is_fx_cx_fy_cy() -> None:
    matrix = camera_matrix_from_annotation({"cameraMatrix": [500, 641, 501, 352]})
    np.testing.assert_allclose(matrix, [[500, 0, 641], [0, 501, 352], [0, 0, 1]])


def test_camera_to_base_vector_is_inverted() -> None:
    world_to_camera = camera_to_base_vector_to_world_to_camera([1, 2, 3, 0, 0, 0])
    np.testing.assert_allclose(world_to_camera[:3, 3], [-1, -2, -3])


def test_split_is_disjoint_complete_and_reserves_mod_five() -> None:
    split = make_frame_split(119, fit_count=16, validation_count=8)
    assert len(split.fit) == 16
    assert len(split.validation) == 8
    assert len(split.heldout) == 119 - 16 - 8
    assert all(index in split.heldout for index in range(0, 119, 5))
    assert not (set(split.fit) & set(split.validation))
    assert not (set(split.fit) & set(split.heldout))
    assert not (set(split.validation) & set(split.heldout))
    assert set((*split.fit, *split.validation, *split.heldout)) == set(range(119))


def test_split_honors_configurable_heldout_reservation() -> None:
    split = make_frame_split(30, fit_count=4, validation_count=2, heldout_modulus=3, heldout_remainder=1)
    assert all(index in split.heldout for index in range(1, 30, 3))


def test_select_best_iteration_uses_validation_and_earlier_tie_break() -> None:
    records = [
        {"iteration": 0, "validation_iou": 0.4, "pose_delta": {"translation_m": 0.0}},
        {"iteration": 1, "validation_iou": 0.6, "pose_delta": {"translation_m": 0.02}},
        {"iteration": 2, "validation_iou": 0.6, "pose_delta": {"translation_m": 0.01}},
        {"iteration": 3, "validation_iou": 0.5, "pose_delta": {"translation_m": 0.01}},
    ]
    assert select_best_iteration(records)["iteration"] == 2


def test_binary_mask_metrics() -> None:
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    metrics = binary_mask_metrics(prediction, target)
    assert metrics["intersection_px"] == 1
    assert metrics["union_px"] == 3
    assert metrics["iou"] == 1 / 3


def test_solve_shared_pose_recovers_synthetic_camera() -> None:
    import cv2

    rng = np.random.default_rng(4)
    world = rng.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.6], size=(40, 3))
    camera_matrix = np.array([[520, 0, 320], [0, 520, 180], [0, 0, 1]], dtype=np.float64)
    rvec = np.array([0.1, -0.05, 0.03], dtype=np.float64)
    tvec = np.array([0.02, -0.01, 1.5], dtype=np.float64)
    image, _ = cv2.projectPoints(world, rvec, tvec, camera_matrix, None)
    image = image.reshape(-1, 2)
    records = [
        {"frame_index": 0, "image_points": image[:20], "world_points": world[:20], "scores": np.ones(20)},
        {"frame_index": 1, "image_points": image[20:], "world_points": world[20:], "scores": np.ones(20)},
    ]
    pose, details = solve_shared_pose(records, camera_matrix, seed=7)
    expected_rotation, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(pose["world_to_camera"][:3, :3], expected_rotation, atol=1e-5)
    np.testing.assert_allclose(pose["world_to_camera"][:3, 3], tvec, atol=1e-5)
    assert details["pnp_inlier_count"] == 40
