#!/usr/bin/env python3
"""Geometry, RANSAC, PnP, visualization, and native RoMaV2 helpers for DREAM evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_RANSAC_METHOD = "CV2_USAC_MAGSAC"
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8.0
DEFAULT_RANSAC_CONFIDENCE = 0.9999
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_MIN_NUM_MATCHES = 4


def restrict_visual_geoms_to_bodies(
    model,
    body_names: list[str] | tuple[str, ...] | None,
    visual_geom_group: int = 2,
) -> list[str]:
    """Hide visual geoms outside an exact body-name allowlist.

    The model is modified in place so rendering, visual bounds, and
    ``mjv_select`` all see the same set of robot surfaces.
    """
    if not body_names:
        return []

    import mujoco

    names = list(dict.fromkeys(str(name) for name in body_names))
    body_ids = {}
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Visual body {name!r} was not found in the MuJoCo model")
        body_ids[name] = int(body_id)

    allowed_ids = set(body_ids.values())
    hidden_group = (int(visual_geom_group) + 1) % 6
    kept = 0
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != int(visual_geom_group):
            continue
        if int(model.geom_bodyid[geom_id]) in allowed_ids:
            kept += 1
        else:
            model.geom_group[geom_id] = hidden_group
    if kept == 0:
        raise RuntimeError(
            f"None of the requested bodies has a geom in visual group {visual_geom_group}: {names}"
        )
    return names


def visual_bounds(model, data, visual_geom_group: int = 2) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    chunks = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom_id] != visual_geom_group:
            continue
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start : start + count]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        chunks.append(vertices @ rotation.T + data.geom_xpos[geom_id])
    if not chunks:
        raise RuntimeError(f"The MuJoCo model has no group-{visual_geom_group} visual meshes")
    vertices = np.concatenate(chunks)
    return vertices.min(axis=0), vertices.max(axis=0)


def subtree_visual_bounds(model, data, root_body_name: str, visual_geom_group: int = 2) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_body_id < 0:
        raise ValueError(f"Body {root_body_name!r} was not found in the MJCF")

    def belongs_to_subtree(body_id: int) -> bool:
        while body_id > 0:
            if body_id == root_body_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return body_id == root_body_id

    chunks = []
    for geom_id in range(model.ngeom):
        if (
            model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH
            or model.geom_group[geom_id] != visual_geom_group
            or not belongs_to_subtree(int(model.geom_bodyid[geom_id]))
        ):
            continue
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start : start + count]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        chunks.append(vertices @ rotation.T + data.geom_xpos[geom_id])
    if not chunks:
        raise RuntimeError(f"Body subtree {root_body_name!r} has no group-{visual_geom_group} visual meshes")
    vertices = np.concatenate(chunks)
    return vertices.min(axis=0), vertices.max(axis=0)


def visual_scene_option(visual_geom_group: int = 2):
    import mujoco

    option = mujoco.MjvOption()
    if not 0 <= visual_geom_group < len(option.geomgroup):
        raise ValueError(f"visual_geom_group must be in [0, {len(option.geomgroup) - 1}], got {visual_geom_group}")
    option.geomgroup[:] = 0
    option.geomgroup[visual_geom_group] = 1
    return option


def add_zed_camera(spec, name: str, width: int, height: int, camera_matrix: np.ndarray):
    zed_camera = spec.worldbody.add_camera(name=name)
    zed_camera.resolution[:] = [width, height]
    zed_camera.focal_pixel[:] = [camera_matrix[0, 0], camera_matrix[1, 1]]
    zed_camera.principal_pixel[:] = [camera_matrix[0, 2] - width / 2.0, camera_matrix[1, 2] - height / 2.0]
    zed_camera.sensor_size[:] = [1.0, 1.0]
    return zed_camera


def render_frame_center_views(qpos: np.ndarray, source_index: int, output_dir: Path, args, camera_matrix: np.ndarray, center_name: str) -> list[Path]:
    import mujoco

    spec = mujoco.MjSpec.from_file(str(args.mujoco_xml))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.mesh_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.mesh_height)
    add_zed_camera(spec, "zed_render_camera", args.mesh_width, args.mesh_height, camera_matrix)
    model = spec.compile()
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_render_camera")
    data = mujoco.MjData(model)
    joint_count = min(qpos.reshape(-1).shape[0], model.nq)
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    data.qpos[:joint_count] = qpos.reshape(-1)[:joint_count]
    mujoco.mj_forward(model, data)

    if center_name == "full_arm":
        minimum, maximum = visual_bounds(model, data, getattr(args, "visual_geom_group", 2))
        distance_floor = 1.25
        distance_scale = args.mesh_distance_scale
    elif center_name == "end_effector":
        minimum, maximum = subtree_visual_bounds(model, data, "link6", getattr(args, "visual_geom_group", 2))
        distance_floor = 0.60
        distance_scale = args.end_effector_distance_scale
    else:
        raise ValueError(f"Unknown render center: {center_name}")

    center = 0.5 * (minimum + maximum)
    radius = 0.5 * np.linalg.norm(maximum - minimum)
    distance = max(distance_floor, distance_scale * radius)
    azimuth_min, azimuth_max = parse_range(args.mesh_azimuth_range)
    elevation_min, elevation_max = parse_range(args.mesh_elevation_range)
    rng = np.random.default_rng(args.mesh_seed)
    if args.mesh_views == 1:
        azimuths = np.array([(azimuth_min + azimuth_max) * 0.5])
    else:
        azimuth_step = (azimuth_max - azimuth_min) / args.mesh_views
        azimuths = azimuth_min + rng.uniform(0.0, azimuth_step) + np.arange(args.mesh_views) * azimuth_step
    elevations = rng.uniform(elevation_min, elevation_max, size=args.mesh_views)

    scene_option = visual_scene_option(getattr(args, "visual_geom_group", 2))
    renderer = mujoco.Renderer(model, height=args.mesh_height, width=args.mesh_width)
    render_paths = []
    cameras = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for view_index, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = center
            camera.azimuth = float(azimuth)
            camera.elevation = float(elevation)
            camera.distance = float(distance)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)

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

            view_dir = output_dir / f"view_{view_index:02d}"
            view_dir.mkdir(parents=True, exist_ok=True)
            image_path = view_dir / f"frame_{source_index:06d}.png"
            Image.fromarray(rgb).save(image_path)
            np.savez_compressed(
                view_dir / f"frame_{source_index:06d}_camera.npz",
                camera_position=data.cam_xpos[camera_id].copy(),
                camera_rotation=data.cam_xmat[camera_id].reshape(3, 3).copy(),
                camera_forward=-data.cam_xmat[camera_id].reshape(3, 3)[:, 2],
                camera_up=data.cam_xmat[camera_id].reshape(3, 3)[:, 1],
                camera_quaternion=camera_quaternion,
                image_height=args.mesh_height,
                image_width=args.mesh_width,
                zed_camera_matrix=camera_matrix,
                fixed_camera_id=camera_id,
                camera_lookat=center,
                camera_azimuth=azimuth,
                camera_elevation=elevation,
                camera_distance=distance,
                center_name=center_name,
            )
            render_paths.append(image_path)
            cameras.append(
                {
                    "view_index": view_index,
                    "center_name": center_name,
                    "azimuth_degrees": float(azimuth),
                    "elevation_degrees": float(elevation),
                    "distance": float(distance),
                }
            )
    finally:
        renderer.close()

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_frame_index": source_index,
                "center_name": center_name,
                "center": center.tolist(),
                "radius": float(radius),
                "cameras": cameras,
            },
            indent=2,
        )
        + "\n"
    )
    return render_paths


def render_frame_views(qpos: np.ndarray, source_index: int, output_dir: Path, args, camera_matrix: np.ndarray) -> list[Path]:
    render_paths = []
    for center_name in args.render_centers:
        render_paths.extend(render_frame_center_views(qpos, source_index, output_dir / center_name, args, camera_matrix, center_name))
    return render_paths


def select_rendered_mesh_points(
    points: np.ndarray,
    camera_path: Path,
    model,
    data,
    visual_geom_group: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import mujoco

    camera = np.load(camera_path)
    height = int(camera["image_height"])
    width = int(camera["image_width"])
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_render_camera")
    model.cam_pos[camera_id] = camera["camera_position"]
    model.cam_quat[camera_id] = camera["camera_quaternion"]
    mujoco.mj_forward(model, data)
    scene_option = visual_scene_option(visual_geom_group)
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_id, scene_option=scene_option)

    world = np.full((len(points), 3), np.nan, dtype=np.float32)
    geom_ids = np.full(len(points), -1, dtype=np.int32)
    body_ids = np.full(len(points), -1, dtype=np.int32)
    valid = np.zeros(len(points), dtype=bool)
    try:
        for index, point in enumerate(points):
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                continue
            selected_point = np.zeros(3, dtype=np.float64)
            geom_id = np.array([-1], dtype=np.int32)
            flex_id = np.array([-1], dtype=np.int32)
            skin_id = np.array([-1], dtype=np.int32)
            body_id = mujoco.mjv_select(
                model,
                data,
                scene_option,
                width / height,
                (float(point[0]) + 0.5) / width,
                1.0 - (float(point[1]) + 0.5) / height,
                renderer.scene,
                selected_point,
                geom_id,
                flex_id,
                skin_id,
            )
            if body_id < 0 or geom_id[0] < 0:
                continue
            valid[index] = True
            body_ids[index] = body_id
            geom_ids[index] = geom_id[0]
            world[index] = selected_point
    finally:
        renderer.close()
    return world, valid, geom_ids, body_ids


RANSAC_ZOO = {
    "POSELIB": "LO-RANSAC",
    "CV2_RANSAC": cv2.RANSAC,
    "CV2_USAC_MAGSAC": cv2.USAC_MAGSAC,
    "CV2_USAC_DEFAULT": cv2.USAC_DEFAULT,
    "CV2_USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "CV2_USAC_PROSAC": cv2.USAC_PROSAC,
    "CV2_USAC_FAST": cv2.USAC_FAST,
    "CV2_USAC_ACCURATE": cv2.USAC_ACCURATE,
    "CV2_USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def _filter_matches_opencv(
    points0: np.ndarray,
    points1: np.ndarray,
    method: int,
    reproj_threshold: float,
    confidence: float,
    max_iter: int,
    geometry_type: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        if geometry_type == "Homography":
            matrix, mask = cv2.findHomography(
                points0,
                points1,
                method=method,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iter,
            )
        elif geometry_type == "Fundamental":
            matrix, mask = cv2.findFundamentalMat(
                points0,
                points1,
                method=method,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iter,
            )
        else:
            raise NotImplementedError(geometry_type)
    except cv2.error:
        return None, None
    if mask is None:
        return None, None
    return matrix, np.array(mask.ravel().astype("bool"), dtype="bool")


def _filter_matches_poselib(
    points0: np.ndarray,
    points1: np.ndarray,
    reproj_threshold: float,
    confidence: float,
    max_iter: int,
    geometry_type: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        import poselib
    except ImportError:
        return None, None
    ransac_options = {
        "max_iterations": max_iter,
        "success_prob": confidence,
        "max_reproj_error": reproj_threshold,
    }
    try:
        if geometry_type == "Homography":
            matrix, info = poselib.estimate_homography(points0, points1, ransac_options)
        elif geometry_type == "Fundamental":
            matrix, info = poselib.estimate_fundamental(points0, points1, ransac_options)
        else:
            raise NotImplementedError(geometry_type)
    except Exception:
        return None, None
    return matrix, np.array(info["inliers"], dtype=bool)


def proc_imcui_ransac_matches(
    points0: np.ndarray,
    points1: np.ndarray,
    ransac_method: str,
    reproj_threshold: float,
    confidence: float,
    max_iter: int,
    geometry_type: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if ransac_method not in RANSAC_ZOO:
        ransac_method = DEFAULT_RANSAC_METHOD
    if ransac_method.startswith("CV2"):
        return _filter_matches_opencv(
            points0,
            points1,
            RANSAC_ZOO[ransac_method],
            reproj_threshold,
            confidence,
            max_iter,
            geometry_type,
        )
    if ransac_method.startswith("POSELIB"):
        return _filter_matches_poselib(
            points0,
            points1,
            reproj_threshold,
            confidence,
            max_iter,
            geometry_type,
        )
    raise NotImplementedError(ransac_method)


def imcui_keypoint_ransac_mask(
    points0: np.ndarray,
    points1: np.ndarray,
    image0_shape: tuple[int, int, int],
    ransac_method: str,
    reproj_threshold: float,
    confidence: float,
    max_iter: int,
) -> tuple[np.ndarray, str, dict]:
    """Local equivalent of imcui.ui.utils.filter_matches for keypoints.

    imcui computes both Fundamental and Homography geometry, then uses the
    Homography inlier mask (`mask_h`) to populate mmkeypoints*_orig.
    """
    if ransac_method not in RANSAC_ZOO:
        ransac_method = DEFAULT_RANSAC_METHOD
    if len(points0) < DEFAULT_MIN_NUM_MATCHES:
        return np.zeros(len(points0), dtype=bool), "too_few_matches", {}
    if len(points0) < 2 * DEFAULT_MIN_NUM_MATCHES:
        return np.zeros(len(points0), dtype=bool), "too_few_for_compute_geometry", {}

    geom_info = {}
    fundamental, mask_f = proc_imcui_ransac_matches(
        points0,
        points1,
        ransac_method,
        reproj_threshold,
        confidence,
        max_iter,
        "Fundamental",
    )
    if fundamental is not None:
        geom_info["Fundamental"] = np.asarray(fundamental).tolist()
        geom_info["mask_f_count"] = int(mask_f.sum()) if mask_f is not None else 0

    homography, mask_h = proc_imcui_ransac_matches(
        points0,
        points1,
        ransac_method,
        reproj_threshold,
        confidence,
        max_iter,
        "Homography",
    )
    if homography is None or mask_h is None:
        return np.zeros(len(points0), dtype=bool), "homography_failed", geom_info

    geom_info["Homography"] = np.asarray(homography).tolist()
    geom_info["mask_h_count"] = int(mask_h.sum())
    if fundamental is not None:
        h0, w0 = image0_shape[:2]
        try:
            _, h1, h2 = cv2.stereoRectifyUncalibrated(
                points0.reshape(-1, 2),
                points1.reshape(-1, 2),
                np.asarray(fundamental),
                imgSize=(w0, h0),
            )
            geom_info["H1"] = h1.tolist()
            geom_info["H2"] = h2.tolist()
        except cv2.error:
            pass
    return mask_h, "success", geom_info


def draw_matches(image0: np.ndarray, image1: np.ndarray, points0: np.ndarray, points1: np.ndarray, scores: np.ndarray, title: str) -> np.ndarray:
    height = max(image0.shape[0], image1.shape[0])
    width0 = image0.shape[1]
    canvas = np.zeros((height, width0 + image1.shape[1], 3), dtype=np.uint8)
    canvas[: image0.shape[0], :width0] = image0
    canvas[: image1.shape[0], width0:] = image1
    for point0, point1, score in zip(points0, points1, scores):
        clipped = float(np.clip(score, 0.0, 1.0))
        color = (int(255 * (1.0 - clipped)), int(255 * clipped), 40)
        start = tuple(np.rint(point0).astype(int))
        end = tuple(np.rint(point1 + [width0, 0]).astype(int))
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 3, color, -1, cv2.LINE_AA)
    canvas = cv2.copyMakeBorder(canvas, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(canvas, title[:220], (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def render_mesh_highlights(qpos: np.ndarray, camera_path: Path, scores: np.ndarray, world_points: np.ndarray, args, title: str) -> np.ndarray:
    import mujoco

    camera_data = np.load(camera_path)
    spec = mujoco.MjSpec.from_file(str(args.mujoco_xml))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.mesh_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.mesh_height)
    add_zed_camera(spec, "zed_render_camera", args.mesh_width, args.mesh_height, camera_data["zed_camera_matrix"])
    model = spec.compile()
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_render_camera")
    model.cam_pos[camera_id] = camera_data["camera_position"]
    model.cam_quat[camera_id] = camera_data["camera_quaternion"]
    data = mujoco.MjData(model)
    joint_count = min(qpos.reshape(-1).shape[0], model.nq)
    data.qpos[:joint_count] = qpos.reshape(-1)[:joint_count]
    mujoco.mj_forward(model, data)
    scene_option = visual_scene_option(getattr(args, "visual_geom_group", 2))
    renderer = mujoco.Renderer(model, height=args.mesh_height, width=args.mesh_width)
    try:
        renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
        for point, score in zip(world_points, scores):
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                break
            confidence = float(np.clip(score, 0.0, 1.0))
            rgba = np.array([1.0, 0.0, 1.0 - 0.55 * confidence, 1.0], dtype=np.float32)
            mujoco.mjv_initGeom(
                renderer.scene.geoms[renderer.scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.012, dtype=np.float64),
                point.astype(np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                rgba,
            )
            renderer.scene.ngeom += 1
        image = renderer.render().copy()
    finally:
        renderer.close()
    image = cv2.copyMakeBorder(image, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(image, title[:180], (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
    return image


def concatenate_visualizations(matching: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    target_height = max(matching.shape[0], mesh.shape[0])

    def pad(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        return cv2.copyMakeBorder(image, 0, target_height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)

    separator = np.full((target_height, 8, 3), 255, dtype=np.uint8)
    return np.concatenate([pad(matching), separator, pad(mesh)], axis=1)


def deduplicate_pnp_correspondences(image_points: np.ndarray, world_points: np.ndarray, scores: np.ndarray, radius: float = 3.0):
    order = np.argsort(scores)[::-1]
    kept = []
    radius_squared = radius * radius
    for index in order:
        if any(np.sum((image_points[index] - image_points[other]) ** 2) < radius_squared for other in kept):
            continue
        kept.append(int(index))
    kept = np.asarray(kept, dtype=np.int64)
    return image_points[kept], world_points[kept], scores[kept]


def solve_camera_pose(image_points: np.ndarray, world_points: np.ndarray, scores: np.ndarray, camera_matrix: np.ndarray, reprojection_threshold: float) -> dict:
    image_points, world_points, scores = deduplicate_pnp_correspondences(image_points, world_points, scores)
    if len(image_points) < 6:
        raise RuntimeError(f"PnP needs at least 6 correspondences, found {len(image_points)}")
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        world_points.astype(np.float64),
        image_points.astype(np.float64),
        camera_matrix,
        None,
        iterationsCount=10_000,
        reprojectionError=float(reprojection_threshold),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        count = 0 if inliers is None else len(inliers)
        raise RuntimeError(f"PnP RANSAC failed with {count} inliers")
    inlier_indices = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(
        world_points[inlier_indices].astype(np.float64),
        image_points[inlier_indices].astype(np.float64),
        camera_matrix,
        None,
        rvec,
        tvec,
    )
    projected, _ = cv2.projectPoints(world_points.astype(np.float64), rvec, tvec, camera_matrix, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = tvec.ravel()
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = (-rotation.T @ tvec).ravel()
    return {
        "rvec": rvec,
        "tvec": tvec,
        "world_to_camera": world_to_camera,
        "camera_to_world": camera_to_world,
        "image_points": image_points,
        "world_points": world_points,
        "scores": scores,
        "inlier_indices": inlier_indices,
        "reprojection_errors": errors,
    }


def render_pose_overlay(rgb: np.ndarray, qpos: np.ndarray, pose: dict, raw_camera_matrix: np.ndarray, distortion: np.ndarray, undistorted_camera_matrix: np.ndarray, args) -> np.ndarray:
    import mujoco

    height, width = rgb.shape[:2]
    undistorted = cv2.undistort(rgb, raw_camera_matrix, distortion, None, undistorted_camera_matrix)
    spec = mujoco.MjSpec.from_file(str(args.mujoco_xml))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)
    add_zed_camera(spec, "zed_overlay_camera", width, height, undistorted_camera_matrix)
    model = spec.compile()
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_overlay_camera")
    data = mujoco.MjData(model)
    joint_count = min(qpos.reshape(-1).shape[0], model.nq)
    data.qpos[:joint_count] = qpos.reshape(-1)[:joint_count]
    mujoco.mj_forward(model, data)
    scene_option = visual_scene_option(getattr(args, "visual_geom_group", 2))

    camera_to_world_rotation = pose["world_to_camera"][:3, :3].T
    camera_position = pose["camera_to_world"][:3, 3]
    forward = camera_to_world_rotation @ np.array([0.0, 0.0, 1.0])
    up = camera_to_world_rotation @ np.array([0.0, -1.0, 0.0])
    camera_rotation = np.column_stack([np.cross(forward, up), up, -forward])
    camera_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(camera_quaternion, camera_rotation.reshape(-1))
    model.cam_pos[camera_id] = camera_position
    model.cam_quat[camera_id] = camera_quaternion
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
        rendered = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()

    robot_mask = segmentation[..., 0] >= 0
    overlay = undistorted.copy()
    alpha = float(args.overlay_alpha)
    overlay[robot_mask] = ((1.0 - alpha) * overlay[robot_mask] + alpha * rendered[robot_mask]).astype(np.uint8)
    for point in pose["image_points"][pose["inlier_indices"]]:
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 4, (0, 255, 0), -1, cv2.LINE_AA)
    return overlay



class ImcuiMatcher:
    """Compatibility name backed exclusively by this repository's native RoMaV2 API."""

    def __init__(self, args, matcher=None):
        self.device = args.device
        self.max_keypoints = args.max_keypoints
        if matcher is None:
            from romav2 import RoMaV2

            matcher = RoMaV2()
        self.matcher = matcher.eval().to(self.device)
        if str(self.device).startswith("cpu"):
            self.matcher.float()

    @torch.inference_mode()
    def __call__(self, image0: np.ndarray, image1: np.ndarray) -> dict:
        preds = self.matcher.match(image0, image1)
        matches, confidence, _, _ = self.matcher.sample(preds, self.max_keypoints)
        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]
        points0, points1 = self.matcher.to_pixel_coordinates(matches, h0, w0, h1, w1)
        return {
            "mkeypoints0_orig": points0.detach().cpu().numpy(),
            "mkeypoints1_orig": points1.detach().cpu().numpy(),
            "mconf": confidence.detach().cpu().numpy(),
        }
