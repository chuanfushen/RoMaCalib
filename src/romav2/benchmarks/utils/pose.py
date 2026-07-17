#!/usr/bin/env python3
"""Evaluate MuJoCo render-to-real matching on one DREAM-style Panda frame.

The pipeline is:
1. Load a DREAM JSON file with joint positions and annotated keypoints.
2. Render N MuJoCo orbit views for the same qpos.
3. Match the input RGB against each render with this repository's native RoMaV2.
4. Apply a Homography RANSAC filter used by the existing
   DROID mesh matching script.
5. Back-project matched render pixels to MuJoCo mesh surface points and solve
   camera-to-robot-base with PnP.
6. Keep the best render view and report keypoint reprojection/3D metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch
from PIL import Image

torch.set_float32_matmul_precision("highest")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from .geometry import (  # noqa: E402
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ImcuiMatcher,
    add_zed_camera,
    draw_matches,
    imcui_keypoint_ransac_mask,
    restrict_visual_geoms_to_bodies,
    select_rendered_mesh_points,
    solve_camera_pose,
    visual_bounds,
    visual_scene_option,
)
from .render import apply_json_qpos, load_joint_positions  # noqa: E402

DEFAULT_FRAME_JSON = Path(
    "/LargeModelDev/users/chuanfu.shen/workspace/paper/calib/dream-data/real/"
    "panda-3cam_azure/panda-3cam_azure/000000.json"
)
DEFAULT_FRAME_IMAGE = DEFAULT_FRAME_JSON.with_suffix(".rgb.jpg")
DEFAULT_MJCF = PROJECT_ROOT / "assets/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/dream_mujoco_match_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_FRAME_IMAGE)
    parser.add_argument("--json", type=Path, default=DEFAULT_FRAME_JSON)
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--visual-geom-group", type=int, default=2, help="MuJoCo geom group containing renderable visual meshes.")
    parser.add_argument(
        "--visual-body-names",
        nargs="*",
        default=None,
        help="Optional exact body-name allowlist for rendering, bounds, and mesh picking.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--views", "-x", type=int, default=6)
    parser.add_argument("--width", type=int, default=None, help="Render width. Defaults to input image width.")
    parser.add_argument("--height", type=int, default=None, help="Render height. Defaults to input image height.")
    parser.add_argument("--distance-scale", type=float, default=2.8)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--azimuth-offset", type=float, default=0.0)
    parser.add_argument("--matcher", default="RoMaV2", choices=("RoMaV2",))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--detect-threshold", type=float, default=0.005)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--score-filter", type=float, default=0.0)
    parser.add_argument("--ransac-method", default=DEFAULT_RANSAC_METHOD)
    parser.add_argument("--ransac-threshold", type=float, default=DEFAULT_RANSAC_REPROJ_THRESHOLD)
    parser.add_argument("--ransac-confidence", type=float, default=DEFAULT_RANSAC_CONFIDENCE)
    parser.add_argument("--ransac-max-iter", type=int, default=DEFAULT_RANSAC_MAX_ITER)
    parser.add_argument("--pnp-threshold", type=float, default=5.0)
    parser.add_argument("--min-pnp-correspondences", type=int, default=6)
    parser.add_argument("--max-pnp-correspondences", type=int, default=512)
    parser.add_argument(
        "--mask-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask the observed input image with SAM3 before matching.",
    )
    parser.add_argument("--mask-prompt", default="robotic arm", help="Text prompt passed to SAM3 when --mask-input is enabled.")
    parser.add_argument("--camera-settings", type=Path, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--fallback-focal", type=float, default=400.0)
    parser.add_argument("--save-all-matches", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_dream_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def make_sam3_extractor(enabled: bool, checkpoint_path: Path | None = None):
    if not enabled:
        return None
    torch.set_float32_matmul_precision("highest")
    from .mask import Sam3Extractor

    return Sam3Extractor(bpe_path=None, ckpt_path=checkpoint_path)


def sam3_mask_to_numpy(mask, image_shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().float().cpu().numpy()
    else:
        mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask > 0
    height, width = image_shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def apply_input_mask(
    image: np.ndarray,
    enabled: bool,
    extractor=None,
    prompt: str = "robotic arm",
) -> tuple[np.ndarray, np.ndarray | None]:
    if not enabled:
        return image, None
    if extractor is None:
        extractor = make_sam3_extractor(True)
    mask_raw = extractor.extract_masks(Image.fromarray(image), prompt=prompt)
    if mask_raw is None:
        return image, None
    mask = sam3_mask_to_numpy(mask_raw, image.shape)
    masked = image.copy()
    masked[~mask] = 0
    return masked, mask


def load_camera_matrix(args: argparse.Namespace, image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    if all(value is not None for value in (args.fx, args.fy, args.cx, args.cy)):
        return np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    settings_path = args.camera_settings
    if settings_path is None:
        candidate = args.json.with_name("_camera_settings.json")
        if candidate.is_file():
            settings_path = candidate

    if settings_path is not None and settings_path.is_file():
        payload = json.loads(settings_path.read_text())
        intrinsic = payload["camera_settings"][0]["intrinsic_settings"]
        return np.array(
            [
                [float(intrinsic["fx"]), float(intrinsic.get("s", 0.0)), float(intrinsic["cx"])],
                [0.0, float(intrinsic["fy"]), float(intrinsic["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    return np.array(
        [[args.fallback_focal, 0.0, width / 2.0], [0.0, args.fallback_focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def dream_keypoints(payload: dict) -> list[dict]:
    objects = payload.get("objects", [])
    if not objects:
        return []
    return objects[0].get("keypoints", [])


def keypoint_name_to_body(name: str) -> str:
    if name.startswith("panda_"):
        name = name.removeprefix("panda_")
    if name == "panda":
        return "link0"
    return name


CTRNET_BAXTER_LEFT_JOINTS = (
    "left_s0",
    "left_s1",
    "left_e0",
    "left_e1",
    "left_w0",
    "left_w1",
    "left_w2",
)


def _ctrnet_baxter_dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0, a],
            [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -d * sin_alpha],
            [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, d * cos_alpha],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def ctrnet_baxter_ee_point(model, data) -> np.ndarray:
    """Return the official CtRNet Baxter evaluation endpoint in MuJoCo base coordinates."""
    import mujoco

    joint_positions = []
    for joint_name in CTRNET_BAXTER_LEFT_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"CtRNet Baxter joint {joint_name!r} was not found")
        joint_positions.append(float(data.qpos[int(model.jnt_qposadr[joint_id])]))
    theta = np.asarray(joint_positions, dtype=np.float64)

    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = 0.27035
    transforms = (
        _ctrnet_baxter_dh_transform(0.0, 0.0, 0.0, theta[0]),
        _ctrnet_baxter_dh_transform(-np.pi / 2.0, 0.069, 0.0, theta[1] + np.pi / 2.0),
        _ctrnet_baxter_dh_transform(np.pi / 2.0, 0.0, 0.36435, theta[2]),
        _ctrnet_baxter_dh_transform(-np.pi / 2.0, 0.069, 0.0, theta[3]),
        _ctrnet_baxter_dh_transform(np.pi / 2.0, 0.0, 0.37429, theta[4]),
        _ctrnet_baxter_dh_transform(-np.pi / 2.0, 0.010, 0.0, theta[5]),
        _ctrnet_baxter_dh_transform(np.pi / 2.0, 0.0, 0.0, theta[6]),
    )
    for joint_transform in transforms:
        transform = transform @ joint_transform
    endpoint_in_arm_base = (transform @ np.array([0.0, 0.0, 0.3683, 1.0]))[:3]

    mount_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_arm_mount")
    if mount_id < 0:
        raise ValueError("CtRNet Baxter reference body 'left_arm_mount' was not found")
    mount_rotation = np.asarray(data.xmat[mount_id], dtype=np.float64).reshape(3, 3)
    mount_translation = np.asarray(data.xpos[mount_id], dtype=np.float64)
    return mount_rotation @ endpoint_in_arm_base + mount_translation


def fk_keypoints(model, data, keypoints: list[dict]) -> dict[str, np.ndarray]:
    import mujoco

    result = {}
    for keypoint in keypoints:
        if keypoint["name"] == "ctrnet_baxter_ee":
            result[keypoint["name"]] = ctrnet_baxter_ee_point(model, data)
            continue
        body_name = keypoint_name_to_body(keypoint["name"])
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        result[keypoint["name"]] = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    return result


def project_points(points_robot: np.ndarray, world_to_camera: np.ndarray, camera_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.concatenate([points_robot, np.ones((len(points_robot), 1), dtype=np.float64)], axis=1)
    points_camera = (world_to_camera @ hom.T).T[:, :3]
    projected = points_camera @ camera_matrix.T
    pixels = projected[:, :2] / projected[:, 2:3]
    return pixels, points_camera


def dream_camera_location_scale(payload: dict) -> float:
    locations = []
    for keypoint in dream_keypoints(payload):
        if "location" not in keypoint:
            continue
        location = np.asarray(keypoint["location"], dtype=np.float64)
        if np.isfinite(location).all():
            locations.append(location)
    if not locations:
        return 1.0
    norms = np.linalg.norm(np.stack(locations, axis=0), axis=1)
    return 0.01 if float(np.median(norms)) > 10.0 else 1.0


def keypoint_metrics(payload: dict, fk_points: dict[str, np.ndarray], pose: dict, camera_matrix: np.ndarray) -> dict:
    rows = []
    gt_camera_scale = dream_camera_location_scale(payload)
    for keypoint in dream_keypoints(payload):
        name = keypoint["name"]
        if name not in fk_points or "projected_location" not in keypoint:
            continue
        pred_px, pred_cam = project_points(fk_points[name][None], pose["world_to_camera"], camera_matrix)
        gt_px = np.asarray(keypoint["projected_location"], dtype=np.float64)
        gt_cam = np.asarray(keypoint.get("location", [np.nan, np.nan, np.nan]), dtype=np.float64) * gt_camera_scale
        pixel_error = float(np.linalg.norm(pred_px[0] - gt_px))
        camera_error = float(np.linalg.norm(pred_cam[0] - gt_cam)) if np.isfinite(gt_cam).all() else float("nan")
        rows.append(
            {
                "name": name,
                "predicted_pixel": pred_px[0].tolist(),
                "gt_pixel": gt_px.tolist(),
                "pixel_error": pixel_error,
                "predicted_camera_xyz": pred_cam[0].tolist(),
                "gt_camera_xyz": gt_cam.tolist(),
                "camera_3d_error_m": camera_error,
            }
        )

    pixel_errors = np.asarray([row["pixel_error"] for row in rows], dtype=np.float64)
    camera_errors = np.asarray([row["camera_3d_error_m"] for row in rows], dtype=np.float64)
    valid_camera_errors = camera_errors[np.isfinite(camera_errors)]
    summary = {
        "keypoint_count": len(rows),
        "gt_camera_location_scale": float(gt_camera_scale),
        "pixel_error_mean": float(pixel_errors.mean()) if len(pixel_errors) else float("nan"),
        "pixel_error_median": float(np.median(pixel_errors)) if len(pixel_errors) else float("nan"),
        "pixel_error_max": float(pixel_errors.max()) if len(pixel_errors) else float("nan"),
        "pck_5px": float((pixel_errors <= 5.0).mean() * 100.0) if len(pixel_errors) else float("nan"),
        "pck_10px": float((pixel_errors <= 10.0).mean() * 100.0) if len(pixel_errors) else float("nan"),
        "pck_20px": float((pixel_errors <= 20.0).mean() * 100.0) if len(pixel_errors) else float("nan"),
        "keypoint_add_mean_m": float(valid_camera_errors.mean()) if len(valid_camera_errors) else float("nan"),
    }
    for threshold_mm in (10, 20, 40, 60):
        summary[f"keypoint_ADD<{threshold_mm}mm"] = (
            float((valid_camera_errors <= threshold_mm * 1e-3).mean() * 100.0) if len(valid_camera_errors) else float("nan")
        )
    return {"summary": summary, "per_keypoint": rows}


def draw_keypoint_eval(image: np.ndarray, metrics: dict) -> np.ndarray:
    canvas = image.copy()
    for item in metrics["per_keypoint"]:
        pred = tuple(np.rint(item["predicted_pixel"]).astype(int))
        gt = tuple(np.rint(item["gt_pixel"]).astype(int))
        cv2.circle(canvas, gt, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, pred, 5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, gt, pred, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, item["name"].replace("panda_", ""), (gt[0] + 6, gt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def make_model_and_data(
    mjcf: Path,
    width: int,
    height: int,
    camera_matrix: np.ndarray,
    payload: dict,
    visual_body_names: list[str] | tuple[str, ...] | None = None,
    visual_geom_group: int = 2,
):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)
    add_zed_camera(spec, "zed_render_camera", width, height, camera_matrix)
    model = spec.compile()
    restrict_visual_geoms_to_bodies(model, visual_body_names, visual_geom_group)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    apply_json_qpos(model, data, load_joint_positions_from_payload(payload))
    mujoco.mj_forward(model, data)
    return model, data


def load_joint_positions_from_payload(payload: dict) -> dict[str, float]:
    return {joint["name"]: float(joint["position"]) for joint in payload["sim_state"]["joints"]}


def render_orbit_views(model, data, args: argparse.Namespace, camera_matrix: np.ndarray, output_dir: Path) -> list[Path]:
    import mujoco

    minimum, maximum = visual_bounds(model, data, args.visual_geom_group)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * np.linalg.norm(maximum - minimum)
    distance = max(args.min_distance, args.distance_scale * radius)

    scene_option = visual_scene_option(args.visual_geom_group)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_render_camera")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    render_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for view_index in range(args.views):
            azimuth = args.azimuth_offset + view_index * 360.0 / args.views
            free_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, free_camera)
            free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            free_camera.lookat[:] = center
            free_camera.distance = float(distance)
            free_camera.azimuth = float(azimuth)
            free_camera.elevation = float(args.elevation)

            renderer.update_scene(data, camera=free_camera, scene_option=scene_option)
            scene_cameras = renderer.scene.camera
            camera_position = np.mean([np.asarray(scene_camera.pos) for scene_camera in scene_cameras], axis=0)
            camera_forward = np.mean([np.asarray(scene_camera.forward) for scene_camera in scene_cameras], axis=0)
            camera_forward /= np.linalg.norm(camera_forward)
            camera_up = np.mean([np.asarray(scene_camera.up) for scene_camera in scene_cameras], axis=0)
            camera_up /= np.linalg.norm(camera_up)
            camera_rotation = np.column_stack([np.cross(camera_forward, camera_up), camera_up, -camera_forward])
            camera_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(camera_quaternion, camera_rotation.reshape(-1))

            model.cam_pos[camera_id] = camera_position
            model.cam_quat[camera_id] = camera_quaternion
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
            rgb = renderer.render().copy()

            image_path = output_dir / f"view_{view_index:02d}.png"
            Image.fromarray(rgb).save(image_path)
            np.savez_compressed(
                output_dir / f"view_{view_index:02d}_camera.npz",
                camera_position=data.cam_xpos[camera_id].copy(),
                camera_rotation=data.cam_xmat[camera_id].reshape(3, 3).copy(),
                camera_forward=-data.cam_xmat[camera_id].reshape(3, 3)[:, 2],
                camera_up=data.cam_xmat[camera_id].reshape(3, 3)[:, 1],
                camera_quaternion=camera_quaternion,
                image_height=args.height,
                image_width=args.width,
                zed_camera_matrix=camera_matrix,
                camera_lookat=center,
                camera_azimuth=azimuth,
                camera_elevation=args.elevation,
                camera_distance=distance,
            )
            render_paths.append(image_path)
    finally:
        renderer.close()
    return render_paths


def match_one_render(
    observed_rgb: np.ndarray,
    render_path: Path,
    matcher_api: ImcuiMatcher,
    model,
    data,
    camera_matrix: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    prediction: dict | None = None,
    render_rgb: np.ndarray | None = None,
) -> dict:
    if render_rgb is None:
        render_rgb = np.asarray(Image.open(render_path).convert("RGB"))
    if prediction is None:
        torch.set_float32_matmul_precision("highest")
        prediction = matcher_api(observed_rgb, render_rgb)
    points0 = np.asarray(prediction.get("mkeypoints0_orig", np.empty((0, 2))), dtype=np.float32)
    points1 = np.asarray(prediction.get("mkeypoints1_orig", np.empty((0, 2))), dtype=np.float32)
    scores = np.asarray(prediction.get("mconf", np.ones(len(points0))), dtype=np.float32).reshape(-1)
    if len(points0) != len(points1):
        raise RuntimeError("Matcher returned inconsistent correspondence counts")

    geometric, ransac_status, geom_info = imcui_keypoint_ransac_mask(
        points0,
        points1,
        observed_rgb.shape,
        args.ransac_method,
        args.ransac_threshold,
        args.ransac_confidence,
        args.ransac_max_iter,
    )
    selected = geometric & (scores >= args.score_filter)
    selected_indices = np.flatnonzero(selected)
    camera_path = render_path.with_name(f"{render_path.stem}_camera.npz")
    world_points, valid_surface, geom_ids, body_ids = select_rendered_mesh_points(
        points1[selected], camera_path, model, data, args.visual_geom_group
    )
    mesh_hit_indices = selected_indices[valid_surface]
    image_points = points0[mesh_hit_indices]
    mesh_points = world_points[valid_surface]
    mesh_scores = scores[mesh_hit_indices]
    max_pnp_correspondences = getattr(args, "max_pnp_correspondences", None)
    if max_pnp_correspondences is not None and len(image_points) > max_pnp_correspondences:
        keep = np.argsort(mesh_scores)[-int(max_pnp_correspondences) :]
        image_points = image_points[keep]
        mesh_points = mesh_points[keep]
        mesh_scores = mesh_scores[keep]

    view_record = {
        "render_path": str(render_path),
        "camera_npz": str(camera_path),
        "raw_matches": int(len(points0)),
        "ransac_status": ransac_status,
        "ransac_inliers": int(geometric.sum()),
        "selected_matches": int(selected.sum()),
        "mesh_surface_points": int(valid_surface.sum()),
        "geom_info": geom_info,
        "pnp": {"status": "not_run"},
    }

    if args.save_all_matches:
        stem = render_path.stem
        match_vis = draw_matches(
            observed_rgb,
            render_rgb,
            image_points,
            points1[mesh_hit_indices],
            mesh_scores,
            f"{stem} raw={len(points0)} ransac={int(geometric.sum())} mesh={len(mesh_points)}",
        )
        Image.fromarray(match_vis).save(output_dir / f"{stem}_matches.jpg", quality=92)
        np.savez_compressed(
            output_dir / f"{stem}_matches.npz",
            points0=points0,
            points1=points1,
            scores=scores,
            ransac_inliers=geometric,
            selected=selected,
            image_points=image_points,
            world_points=mesh_points,
            mesh_scores=mesh_scores,
            geom_ids=geom_ids,
            body_ids=body_ids,
        )

    if len(image_points) < args.min_pnp_correspondences:
        view_record["pnp"] = {"status": "too_few_correspondences", "correspondences": int(len(image_points))}
        return {**view_record, "_image_points": image_points, "_world_points": mesh_points, "_scores": mesh_scores}

    try:
        pose = solve_camera_pose(image_points, mesh_points, mesh_scores, camera_matrix, args.pnp_threshold)
    except (RuntimeError, cv2.error) as error:
        view_record["pnp"] = {"status": "failed", "error": str(error), "correspondences": int(len(image_points))}
        return {**view_record, "_image_points": image_points, "_world_points": mesh_points, "_scores": mesh_scores}

    inlier_errors = pose["reprojection_errors"][pose["inlier_indices"]]
    view_record["pnp"] = {
        "status": "success",
        "correspondences": int(len(pose["image_points"])),
        "inliers": int(len(pose["inlier_indices"])),
        "inlier_reprojection_error_mean": float(inlier_errors.mean()),
        "inlier_reprojection_error_median": float(np.median(inlier_errors)),
        "inlier_reprojection_error_max": float(inlier_errors.max()),
        "world_to_camera": pose["world_to_camera"].tolist(),
        "camera_to_robot_base": pose["camera_to_world"].tolist(),
    }
    return {**view_record, "_pose": pose, "_image_points": image_points, "_world_points": mesh_points, "_scores": mesh_scores}


def best_view_key(record: dict) -> tuple:
    pnp = record["pnp"]
    if pnp["status"] != "success":
        return (-1, -1, -float("inf"), -record["ransac_inliers"])
    return (
        1,
        int(pnp["inliers"]),
        -float(pnp["inlier_reprojection_error_mean"]),
        int(record["ransac_inliers"]),
    )


def save_pose_npz(output_dir: Path, best: dict, camera_matrix: np.ndarray, metrics: dict) -> Path:
    pose = best["_pose"]
    path = output_dir / "best_pose.npz"
    np.savez_compressed(
        path,
        world_to_camera=pose["world_to_camera"],
        camera_to_robot_base=pose["camera_to_world"],
        image_points=pose["image_points"],
        world_points=pose["world_points"],
        scores=pose["scores"],
        inlier_indices=pose["inlier_indices"],
        reprojection_errors=pose["reprojection_errors"],
        camera_matrix=camera_matrix,
        keypoint_pixel_error_mean=metrics["summary"]["pixel_error_mean"],
        keypoint_add_mean_m=metrics["summary"]["keypoint_add_mean_m"],
    )
    return path


def remove_run_artifacts(run_dir: Path) -> None:
    for filename in ("input.png", "input_mask.png", "best_keypoint_eval.jpg", "best_pose.npz", "summary.json"):
        path = run_dir / filename
        if path.is_file():
            path.unlink()
    match_dir = run_dir / "matches"
    if match_dir.exists():
        for path in match_dir.glob("view_*_matches.*"):
            if path.is_file():
                path.unlink()


def process(args: argparse.Namespace) -> Path:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    if args.views < 1:
        raise ValueError("--views must be >= 1")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.json.is_file():
        raise FileNotFoundError(args.json)
    if not args.mujoco_xml.is_file():
        raise FileNotFoundError(args.mujoco_xml)

    payload = load_dream_payload(args.json)
    observed_rgb = np.asarray(Image.open(args.image).convert("RGB"))
    mask_extractor = make_sam3_extractor(args.mask_input)
    observed_masked, input_mask = apply_input_mask(observed_rgb, args.mask_input, mask_extractor, args.mask_prompt)
    args.height = observed_masked.shape[0] if args.height is None else args.height
    args.width = observed_masked.shape[1] if args.width is None else args.width
    if (args.height, args.width) != observed_masked.shape[:2]:
        observed_for_match = np.asarray(Image.fromarray(observed_masked).resize((args.width, args.height), Image.BILINEAR))
    else:
        observed_for_match = observed_masked

    camera_matrix = load_camera_matrix(args, observed_for_match.shape)
    run_dir = args.output_dir / args.json.stem
    render_dir = run_dir / "renders"
    match_dir = run_dir / "matches"
    run_dir.mkdir(parents=True, exist_ok=True)
    remove_run_artifacts(run_dir)
    match_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observed_for_match).save(run_dir / "input.png")
    if input_mask is not None:
        Image.fromarray(input_mask.astype(np.uint8) * 255).save(run_dir / "input_mask.png")

    model, data = make_model_and_data(
        args.mujoco_xml,
        args.width,
        args.height,
        camera_matrix,
        payload,
        args.visual_body_names,
        args.visual_geom_group,
    )
    fk_points = fk_keypoints(model, data, dream_keypoints(payload))
    render_paths = render_orbit_views(model, data, args, camera_matrix, render_dir)
    matcher_api = ImcuiMatcher(args)

    view_records = []
    for render_path in render_paths:
        print(f"Matching {args.image.name} <-> {render_path.name}")
        view_records.append(match_one_render(observed_for_match, render_path, matcher_api, model, data, camera_matrix, args, match_dir))

    best = max(view_records, key=best_view_key)
    if best["pnp"]["status"] != "success":
        summary = {
            "status": "failed",
            "reason": "No render view produced a valid PnP pose.",
            "image": str(args.image),
            "json": str(args.json),
            "input_mask": {
                "enabled": bool(args.mask_input),
                "prompt": args.mask_prompt,
                "source": "SAM3 via romav2.benchmarks.utils.mask.Sam3Extractor.",
                "mask_path": str(run_dir / "input_mask.png") if input_mask is not None else None,
            },
            "views": [{key: value for key, value in record.items() if not key.startswith("_")} for record in view_records],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return run_dir

    metrics = keypoint_metrics(payload, fk_points, best["_pose"], camera_matrix)
    keypoint_vis = draw_keypoint_eval(observed_for_match, metrics)
    keypoint_vis_path = run_dir / "best_keypoint_eval.jpg"
    Image.fromarray(keypoint_vis).save(keypoint_vis_path, quality=95)
    pose_npz = save_pose_npz(run_dir, best, camera_matrix, metrics)

    clean_views = []
    for record in view_records:
        clean = {key: value for key, value in record.items() if not key.startswith("_")}
        clean["selected_best_view"] = record is best
        clean_views.append(clean)

    summary = {
        "status": "success",
        "image": str(args.image),
        "json": str(args.json),
        "mujoco_xml": str(args.mujoco_xml),
        "matcher": args.matcher,
        "input_mask": {
            "enabled": bool(args.mask_input),
            "prompt": args.mask_prompt,
            "source": "SAM3 via romav2.benchmarks.utils.mask.Sam3Extractor.",
            "mask_path": str(run_dir / "input_mask.png") if input_mask is not None else None,
        },
        "camera_matrix": camera_matrix.tolist(),
        "best_render_path": best["render_path"],
        "best_camera_npz": best["camera_npz"],
        "pose_npz": str(pose_npz),
        "keypoint_visualization": str(keypoint_vis_path),
        "camera_to_robot_base": best["pnp"]["camera_to_robot_base"],
        "world_to_camera": best["pnp"]["world_to_camera"],
        "keypoint_metrics": metrics,
        "views": clean_views,
        "robopose_reference": {
            "source": "https://github.com/yannlabb/robopose",
            "note": "RoboPose DREAM meter reports 3D keypoint ADD mean/AUC; this script reports keypoint ADD-style mean and threshold percentages plus 2D reprojection errors.",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir


def main() -> None:
    output_dir = process(parse_args())
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
