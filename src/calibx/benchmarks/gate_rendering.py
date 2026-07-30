"""Pose-aligned MuJoCo rendering for the d7 ``gate-v1`` protocol.

The historical full DREAM and CTRNet-X runners used an orbit renderer.  The
Table 4 SAM ablation instead re-renders the robot from the current PnP pose at
each refinement step.  Keeping that bridge here makes the protocol boundary
explicit: it is not a replacement for the core orbit-view renderer.

Only the actual render calls import MuJoCo, so this module remains importable
for offline configuration and record tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CAMERA_NAME = "zed_render_camera"
DEFAULT_VISUAL_GEOM_GROUP = 2


@dataclass(frozen=True)
class PoseAlignedRenderSettings:
    """The rendering subset fixed by the pose-aligned refinement protocol."""

    width: int
    height: int
    visual_geom_group: int = DEFAULT_VISUAL_GEOM_GROUP
    camera_name: str = DEFAULT_CAMERA_NAME

    def validate(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("pose-aligned render width and height must be positive")
        if self.visual_geom_group != DEFAULT_VISUAL_GEOM_GROUP:
            raise ValueError(
                "pose-aligned refinement requires visual_geom_group=2 so rendering "
                "and mesh picking share the same visual geometry"
            )
        if not self.camera_name:
            raise ValueError("camera_name must not be empty")


def pose_aligned_settings_from_args(args: object) -> PoseAlignedRenderSettings:
    """Adapt an evaluator namespace without exposing it as the public API.

    d7 stores these fields on its evaluator namespace.  The adapter keeps that
    source compatibility local while new callers pass the typed settings above.
    """
    try:
        settings = PoseAlignedRenderSettings(
            width=int(getattr(args, "width")),
            height=int(getattr(args, "height")),
            visual_geom_group=int(
                getattr(args, "visual_geom_group", DEFAULT_VISUAL_GEOM_GROUP)
            ),
        )
    except AttributeError as error:
        raise ValueError("pose-aligned rendering requires width and height") from error
    settings.validate()
    return settings


def _world_to_camera_matrix(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_to_camera must be 4x4")
    if not np.isfinite(transform).all():
        raise ValueError("world_to_camera must be finite")
    return transform


def opencv_pose_to_mujoco_components(
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return MuJoCo camera position and rotation matrix before quaternioning.

    This is the exact d7 coordinate conversion for OpenCV
    ``T_camera<-robot_base``.  It is intentionally pure NumPy so its matrix
    convention can be tested without a MuJoCo/EGL runtime.
    """
    transform = _world_to_camera_matrix(world_to_camera)
    camera_to_world_rotation = transform[:3, :3].T
    camera_position = -camera_to_world_rotation @ transform[:3, 3]
    right = camera_to_world_rotation[:, 0]
    up = -camera_to_world_rotation[:, 1]
    backward = -camera_to_world_rotation[:, 2]
    mujoco_rotation = np.column_stack([right, up, backward])
    return camera_position, mujoco_rotation


def mujoco_components_to_opencv_pose(
    camera_position: np.ndarray,
    mujoco_rotation: np.ndarray,
) -> np.ndarray:
    """Pure inverse of :func:`opencv_pose_to_mujoco_components`."""
    position = np.asarray(camera_position, dtype=np.float64).reshape(-1)
    rotation = np.asarray(mujoco_rotation, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("camera_position must contain three values")
    if rotation.shape != (3, 3):
        raise ValueError("mujoco_rotation must be 3x3")
    camera_to_world_rotation = np.column_stack(
        [rotation[:, 0], -rotation[:, 1], -rotation[:, 2]]
    )
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = camera_to_world_rotation.T
    world_to_camera[:3, 3] = -camera_to_world_rotation.T @ position
    return world_to_camera


def opencv_pose_to_mujoco_camera(
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV ``T_camera<-base`` to MuJoCo camera position/quaternion."""
    import mujoco

    camera_position, mujoco_rotation = opencv_pose_to_mujoco_components(world_to_camera)
    camera_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(camera_quaternion, mujoco_rotation.reshape(-1))
    return camera_position, camera_quaternion


def mujoco_camera_to_opencv_pose(
    camera_position: np.ndarray,
    camera_quaternion: np.ndarray,
) -> np.ndarray:
    """Convert a MuJoCo camera pose back to OpenCV ``T_camera<-base``."""
    import mujoco

    mujoco_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(
        mujoco_rotation,
        np.asarray(camera_quaternion, dtype=np.float64),
    )
    return mujoco_components_to_opencv_pose(camera_position, mujoco_rotation.reshape(3, 3))


def visual_scene_option(visual_geom_group: int):
    """Create the d7 visual-only scene option (MuJoCo geom group 2 by default)."""
    import mujoco

    if not 0 <= visual_geom_group < len(mujoco.MjvOption().geomgroup):
        raise ValueError(f"visual geom group out of range: {visual_geom_group}")
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[visual_geom_group] = 1
    return scene_option


def _render_mujoco_camera_artifacts(
    model,
    data,
    settings: PoseAlignedRenderSettings,
    camera_matrix: np.ndarray,
    output_dir: Path,
    camera_position: np.ndarray,
    camera_quaternion: np.ndarray,
    *,
    stem: str,
    save_rgb: bool,
) -> tuple[Path | None, Path, Path, np.ndarray]:
    """Render RGB/segmentation artifacts from an explicitly positioned camera."""
    import mujoco

    settings.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        settings.camera_name,
    )
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera {settings.camera_name!r} was not found")
    model.cam_pos[camera_id] = np.asarray(camera_position, dtype=np.float64)
    model.cam_quat[camera_id] = np.asarray(camera_quaternion, dtype=np.float64)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=settings.height, width=settings.width)
    try:
        renderer.update_scene(
            data,
            camera=camera_id,
            scene_option=visual_scene_option(settings.visual_geom_group),
        )
        rgb = renderer.render().copy() if save_rgb else None
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()

    render_mask = segmentation[..., 0] >= 0
    image_path = output_dir / f"{stem}.png" if save_rgb else None
    mask_path = output_dir / f"{stem}_mask.png"
    camera_path = output_dir / f"{stem}_camera.npz"
    if image_path is not None:
        Image.fromarray(rgb).save(image_path)
    Image.fromarray(render_mask.astype(np.uint8) * 255).save(mask_path)
    np.savez_compressed(
        camera_path,
        camera_position=data.cam_xpos[camera_id].copy(),
        camera_rotation=data.cam_xmat[camera_id].reshape(3, 3).copy(),
        camera_forward=-data.cam_xmat[camera_id].reshape(3, 3)[:, 2],
        camera_up=data.cam_xmat[camera_id].reshape(3, 3)[:, 1],
        camera_quaternion=np.asarray(camera_quaternion, dtype=np.float64),
        image_height=settings.height,
        image_width=settings.width,
        zed_camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        camera_source="pose_aligned_opencv_T_camera_from_robot_base",
    )
    return image_path, mask_path, camera_path, render_mask


def render_pose_aligned_artifacts(
    model,
    data,
    settings: PoseAlignedRenderSettings,
    camera_matrix: np.ndarray,
    world_to_camera: np.ndarray,
    output_dir: Path,
    *,
    stem: str = "render",
) -> tuple[Path, Path, Path, np.ndarray]:
    """Render the current pose estimate plus its visual-geometry mask."""
    camera_position, camera_quaternion = opencv_pose_to_mujoco_camera(world_to_camera)
    image_path, mask_path, camera_path, render_mask = _render_mujoco_camera_artifacts(
        model,
        data,
        settings,
        camera_matrix,
        output_dir,
        camera_position,
        camera_quaternion,
        stem=stem,
        save_rgb=True,
    )
    assert image_path is not None
    return image_path, mask_path, camera_path, render_mask


def render_pose_mask(
    model,
    data,
    settings: PoseAlignedRenderSettings,
    camera_matrix: np.ndarray,
    world_to_camera: np.ndarray,
    output_dir: Path,
    *,
    stem: str = "candidate_render",
) -> tuple[Path, Path, np.ndarray]:
    """Render only the candidate visual mask used by the gate's IoU evidence."""
    camera_position, camera_quaternion = opencv_pose_to_mujoco_camera(world_to_camera)
    _, mask_path, camera_path, render_mask = _render_mujoco_camera_artifacts(
        model,
        data,
        settings,
        camera_matrix,
        output_dir,
        camera_position,
        camera_quaternion,
        stem=stem,
        save_rgb=False,
    )
    return mask_path, camera_path, render_mask


__all__ = [
    "DEFAULT_CAMERA_NAME",
    "DEFAULT_VISUAL_GEOM_GROUP",
    "PoseAlignedRenderSettings",
    "mujoco_camera_to_opencv_pose",
    "mujoco_components_to_opencv_pose",
    "opencv_pose_to_mujoco_camera",
    "opencv_pose_to_mujoco_components",
    "pose_aligned_settings_from_args",
    "render_pose_aligned_artifacts",
    "render_pose_mask",
    "visual_scene_option",
]
