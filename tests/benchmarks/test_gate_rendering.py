from __future__ import annotations

import numpy as np
import pytest

from calibx.benchmarks.gate_rendering import (
    PoseAlignedRenderSettings,
    mujoco_components_to_opencv_pose,
    opencv_pose_to_mujoco_components,
)


def test_pose_aligned_camera_components_round_trip_without_mujoco() -> None:
    angle = np.deg2rad(25.0)
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    world_to_camera[:3, 3] = [0.3, -0.2, 1.4]

    position, rotation = opencv_pose_to_mujoco_components(world_to_camera)

    assert np.allclose(
        mujoco_components_to_opencv_pose(position, rotation),
        world_to_camera,
    )


def test_pose_aligned_renderer_rejects_invalid_matrix_and_settings() -> None:
    with pytest.raises(ValueError, match="4x4"):
        opencv_pose_to_mujoco_components(np.eye(3))
    with pytest.raises(ValueError, match="positive"):
        PoseAlignedRenderSettings(width=0, height=480).validate()
    with pytest.raises(ValueError, match="visual_geom_group=2"):
        PoseAlignedRenderSettings(width=640, height=480, visual_geom_group=1).validate()
