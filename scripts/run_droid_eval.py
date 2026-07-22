#!/usr/bin/env python3
"""Run staged DROID session-level calibration experiments.

Stages intentionally separate data audit, SAM3 pseudo-label generation and
pose evaluation so held-out masks cannot be read during configuration tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import h5py
import mujoco
import numpy as np
from PIL import Image

from romav2.benchmarks.droid import (
    DroidSession,
    binary_mask_metrics,
    build_sessions,
    make_frame_split,
    select_best_iteration,
    sha256_file,
    solve_shared_pose,
)
from romav2.benchmarks.utils.geometry import (
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    ImcuiMatcher,
)
from romav2.benchmarks.utils.pose import (
    best_view_key,
    make_model_and_data,
    make_sam3_extractor,
    match_one_render,
    render_orbit_views,
    render_pose_aligned_artifacts,
    render_pose_mask,
)
from romav2.droid_config import load_config, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stage", choices=("audit", "masks", "raw-eval", "roma", "caliball"), required=True)
    parser.add_argument("--scope", choices=("tuning", "full"), default="tuning")
    parser.add_argument("--split", choices=("fit", "validation", "heldout", "all-calibration"), default="validation")
    parser.add_argument("--session", default=None, help="Optional exact session id.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--caliball-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_record() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": command("status", "--short").splitlines(),
    }


def video_record(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        return {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_metadata": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def select_sessions(config: dict, sessions: list[DroidSession], args: argparse.Namespace) -> list[DroidSession]:
    selected = sessions
    if args.scope == "tuning":
        episode_count = int(config["data"]["tuning_episode_count"])
        episode_ids = {episode.uuid for episode in sorted({s.episode for s in sessions}, key=lambda item: item.rank)[:episode_count]}
        selected = [session for session in sessions if session.episode.uuid in episode_ids]
    if args.session is not None:
        selected = [session for session in selected if session.session_id == args.session]
        if not selected:
            raise KeyError(f"Unknown or out-of-scope session: {args.session}")
    return selected


def frame_indices(config: dict, session: DroidSession, split_name: str) -> tuple[int, ...]:
    video_frames = video_record(session.video_path)["frame_count"]
    split = make_frame_split(
        video_frames,
        fit_count=int(config["data"]["fit_frames"]),
        validation_count=int(config["data"]["validation_frames"]),
    )
    if split_name == "all-calibration":
        return tuple(sorted((*split.fit, *split.validation)))
    return getattr(split, split_name)


def read_video_frames(path: Path, indices: tuple[int, ...]) -> dict[int, np.ndarray]:
    requested = set(indices)
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        for index in range(max(indices, default=-1) + 1):
            ok, bgr = capture.read()
            if not ok:
                break
            if index in requested:
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    missing = sorted(requested - frames.keys())
    if missing:
        raise RuntimeError(f"Missing video frames {missing} in {path}")
    return frames


def stage_audit(config: dict, config_path: Path, sessions: list[DroidSession], output_root: Path) -> None:
    records = []
    for session in sessions:
        video = video_record(session.video_path)
        with h5py.File(session.episode.trajectory_h5, "r") as handle:
            state_count = int(handle["observation/robot_state/joint_positions"].shape[0])
            gripper_count = int(handle["observation/robot_state/gripper_position"].shape[0])
            extrinsic_key = f"observation/camera_extrinsics/{session.camera_serial}_left"
            extrinsic_count = int(handle[extrinsic_key].shape[0])
        split = make_frame_split(
            video["frame_count"],
            fit_count=int(config["data"]["fit_frames"]),
            validation_count=int(config["data"]["validation_frames"]),
        )
        records.append(
            {
                "session_id": session.session_id,
                "episode_rank": session.episode.rank,
                "episode_uuid": session.episode.uuid,
                "camera_name": session.camera_name,
                "camera_serial": session.camera_serial,
                "video": {"path": str(session.video_path), **video},
                "state_count": state_count,
                "gripper_count": gripper_count,
                "extrinsic_count": extrinsic_count,
                "video_minus_state": video["frame_count"] - state_count,
                "camera_matrix": session.camera_matrix.tolist(),
                "raw_world_to_camera": session.raw_world_to_camera.tolist(),
                "fit_indices": list(split.fit),
                "validation_indices": list(split.validation),
                "heldout_count": len(split.heldout),
                "split_overlap": 0,
            }
        )
    write_json(
        output_root / "audit" / "droid_manifest.json",
        {
            "stage": "audit",
            "created_at_unix": time.time(),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "git": git_record(),
            "session_count": len(records),
            "sessions": records,
        },
    )


def frame_dir(output_root: Path, session: DroidSession, index: int) -> Path:
    return output_root / "sessions" / session.session_id / "frames" / f"{index:06d}"


def stage_masks(
    config: dict,
    sessions: list[DroidSession],
    output_root: Path,
    split_name: str,
    max_frames: int | None,
) -> None:
    checkpoint = project_path(config["paths"]["sam3_checkpoint"])
    extractor = make_sam3_extractor(True, checkpoint)
    prompt = str(config["matching"]["mask_prompt"])
    records = []
    for session in sessions:
        indices = frame_indices(config, session, split_name)
        if max_frames is not None:
            indices = indices[:max_frames]
        images = read_video_frames(session.video_path, indices)
        for index in indices:
            destination = frame_dir(output_root, session, index)
            destination.mkdir(parents=True, exist_ok=True)
            image = images[index]
            image_path = destination / "real.png"
            mask_path = destination / "sam3_mask.png"
            Image.fromarray(image).save(image_path)
            mask_raw = extractor.extract_masks(Image.fromarray(image), prompt=prompt)
            if mask_raw is None:
                mask = None
                status = "no_mask"
            else:
                from romav2.benchmarks.utils.pose import sam3_mask_to_numpy

                mask = sam3_mask_to_numpy(mask_raw, image.shape)
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                status = "success"
            record = {
                "session_id": session.session_id,
                "frame_index": index,
                "split": split_name,
                "status": status,
                "prompt": prompt,
                "image": str(image_path),
                "image_sha256": sha256_file(image_path),
                "mask": str(mask_path) if mask is not None else None,
                "mask_sha256": sha256_file(mask_path) if mask is not None else None,
                "mask_area_ratio": float(mask.mean()) if mask is not None else None,
            }
            write_json(destination / "mask_summary.json", record)
            records.append(record)
    write_json(
        output_root / f"mask_manifest_{split_name}.json",
        {"split": split_name, "record_count": len(records), "records": records},
    )


def set_joint(model, data, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"Joint {name!r} is absent from model")
    data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def set_droid_state(model, data, joints: np.ndarray, gripper_closedness: float) -> None:
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    for index, value in enumerate(np.asarray(joints).reshape(7), start=1):
        set_joint(model, data, f"joint{index}", float(value))
    driver = 0.725 * float(np.clip(gripper_closedness, 0.0, 1.0))
    mimic = {
        "finger_joint": 1.0,
        "left_inner_knuckle_joint": 1.0,
        "left_inner_finger_joint": -1.0,
        "right_inner_knuckle_joint": -1.0,
        "right_inner_finger_joint": 1.0,
        "right_outer_knuckle_joint": -1.0,
    }
    for name, multiplier in mimic.items():
        set_joint(model, data, name, multiplier * driver)
    mujoco.mj_forward(model, data)


def make_droid_model(config: dict, session: DroidSession):
    width = int(config["render"]["width"])
    height = int(config["render"]["height"])
    payload = {
        "sim_state": {
            "joints": [{"name": f"panda_joint{index}", "position": 0.0} for index in range(1, 8)]
        }
    }
    return make_model_and_data(
        project_path(config["paths"]["mujoco_xml"]),
        width,
        height,
        session.camera_matrix,
        payload,
        visual_geom_group=int(config["render"]["visual_geom_group"]),
    )


def matcher_args(config: dict) -> SimpleNamespace:
    matching = config["matching"]
    render = config["render"]
    return SimpleNamespace(
        device=str(matching["device"]),
        max_keypoints=int(matching["max_keypoints"]),
        width=int(render["width"]),
        height=int(render["height"]),
        views=int(render["views"]),
        distance_scale=float(render["distance_scale"]),
        min_distance=float(render["min_distance"]),
        elevation=float(render["elevation"]),
        azimuth_offset=float(render["azimuth_offset"]),
        visual_geom_group=int(render["visual_geom_group"]),
        ransac_method=DEFAULT_RANSAC_METHOD,
        ransac_threshold=float(matching["ransac_threshold_px"]),
        ransac_confidence=DEFAULT_RANSAC_CONFIDENCE,
        ransac_max_iter=DEFAULT_RANSAC_MAX_ITER,
        score_filter=float(matching["score_filter"]),
        min_pnp_correspondences=6,
        max_pnp_correspondences=None,
        pnp_threshold=float(matching["pnp_threshold_px"]),
        pnp_dedup_radius=3.0,
        save_all_matches=True,
        opencv_rng_seed=int(matching["seed"]),
    )


def save_pose(path: Path, pose: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pose)


def load_pose(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def validation_score(
    config: dict,
    session: DroidSession,
    output_root: Path,
    pose: dict[str, np.ndarray],
    destination: Path,
    model,
    data,
    joints: np.ndarray,
    gripper: np.ndarray,
) -> dict[str, Any]:
    args = SimpleNamespace(
        width=int(config["render"]["width"]),
        height=int(config["render"]["height"]),
        visual_geom_group=int(config["render"]["visual_geom_group"]),
    )
    indices = frame_indices(config, session, "validation")
    rows = []
    for index in indices:
        target_path = frame_dir(output_root, session, index) / "sam3_mask.png"
        if not target_path.is_file():
            raise FileNotFoundError(f"Generate validation masks first: {target_path}")
        target = np.asarray(Image.open(target_path).convert("L")) > 0
        set_droid_state(model, data, joints[index], gripper[index])
        frame_output = destination / "validation" / f"{index:06d}"
        mask_path, camera_path, rendered = render_pose_mask(
            model,
            data,
            args,
            session.camera_matrix,
            pose["world_to_camera"],
            frame_output,
            stem="render",
        )
        metrics = binary_mask_metrics(rendered, target)
        rows.append(
            {
                "frame_index": index,
                "metrics": metrics,
                "render_mask": str(mask_path),
                "camera_npz": str(camera_path),
            }
        )
    return {
        "frame_count": len(rows),
        "iou_macro": float(np.mean([row["metrics"]["iou"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }


def masked_observed(output_root: Path, session: DroidSession, index: int) -> np.ndarray:
    destination = frame_dir(output_root, session, index)
    image = np.asarray(Image.open(destination / "real.png").convert("RGB"))
    mask = np.asarray(Image.open(destination / "sam3_mask.png").convert("L")) > 0
    observed = image.copy()
    observed[~mask] = 0
    return observed


def roma_correspondences_for_frame(
    config: dict,
    session: DroidSession,
    output_root: Path,
    iteration: int,
    index: int,
    current_pose: dict[str, np.ndarray] | None,
    matcher: ImcuiMatcher,
    model,
    data,
    args: SimpleNamespace,
) -> dict[str, Any]:
    observed = masked_observed(output_root, session, index)
    destination = output_root / "sessions" / session.session_id / "poses" / f"iteration_{iteration:02d}" / "matches" / f"{index:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    if current_pose is None:
        render_paths = render_orbit_views(model, data, args, session.camera_matrix, destination / "renders")
    else:
        render_path, _, _, _ = render_pose_aligned_artifacts(
            model,
            data,
            args,
            session.camera_matrix,
            current_pose["world_to_camera"],
            destination / "renders",
            stem="projected",
        )
        render_paths = [render_path]
    candidates = [
        match_one_render(
            observed,
            render_path,
            matcher,
            model,
            data,
            session.camera_matrix,
            args,
            destination,
            artifact_stem=f"view_{ordinal:02d}",
        )
        for ordinal, render_path in enumerate(render_paths)
    ]
    best = max(candidates, key=best_view_key)
    record = {
        "frame_index": index,
        "image_points": np.asarray(best["_image_points"], dtype=np.float32),
        "world_points": np.asarray(best["_world_points"], dtype=np.float32),
        "scores": np.asarray(best["_scores"], dtype=np.float32),
    }
    np.savez_compressed(destination / "correspondences.npz", **record)
    write_json(
        destination / "summary.json",
        {
            "frame_index": index,
            "iteration": iteration,
            "candidate_count": len(candidates),
            "selected": {key: value for key, value in best.items() if not key.startswith("_")},
            "correspondence_count": len(record["scores"]),
        },
    )
    return record


def stage_roma(
    config: dict,
    sessions: list[DroidSession],
    output_root: Path,
    iteration: int,
    max_frames: int | None,
) -> None:
    if not 0 <= iteration <= int(config["refinement"]["iterations"]):
        raise ValueError("iteration is outside configured range")
    args = matcher_args(config)
    matcher = ImcuiMatcher(args)
    for session_ordinal, session in enumerate(sessions):
        indices = frame_indices(config, session, "fit")
        if max_frames is not None:
            indices = indices[:max_frames]
        iteration_dir = output_root / "sessions" / session.session_id / "poses" / f"iteration_{iteration:02d}"
        current_pose = None
        if iteration > 0:
            parent_path = output_root / "sessions" / session.session_id / "poses" / f"iteration_{iteration - 1:02d}" / "selected_pose.npz"
            if not parent_path.is_file():
                parent_path = parent_path.with_name("roma_pose.npz")
            current_pose = load_pose(parent_path)
        model, data = make_droid_model(config, session)
        correspondences = []
        with h5py.File(session.episode.trajectory_h5, "r") as handle:
            joints = np.asarray(handle["observation/robot_state/joint_positions"])
            gripper = np.asarray(handle["observation/robot_state/gripper_position"])
            for index in indices:
                set_droid_state(model, data, joints[index], gripper[index])
                record = roma_correspondences_for_frame(
                    config,
                    session,
                    output_root,
                    iteration,
                    index,
                    current_pose,
                    matcher,
                    model,
                    data,
                    args,
                )
                if len(record["scores"]):
                    correspondences.append(record)
            pose, pnp = solve_shared_pose(
                correspondences,
                session.camera_matrix,
                topk_per_frame=int(config["matching"]["topk_per_frame"]),
                reprojection_error_px=float(config["matching"]["pnp_threshold_px"]),
                iterations=int(config["matching"]["pnp_iterations"]),
                confidence=float(config["matching"]["pnp_confidence"]),
                seed=int(config["matching"]["seed"]) + session_ordinal * 1009 + iteration,
            )
            score = validation_score(
                config,
                session,
                output_root,
                pose,
                iteration_dir / "roma_candidate",
                model,
                data,
                joints,
                gripper,
            )
        save_pose(iteration_dir / "roma_pose.npz", pose)
        if iteration == 0:
            save_pose(iteration_dir / "selected_pose.npz", pose)
        write_json(
            iteration_dir / "roma_summary.json",
            {
                "session_id": session.session_id,
                "iteration": iteration,
                "fit_frame_indices": list(indices),
                "pnp": pnp,
                "validation": score,
                "world_to_camera": pose["world_to_camera"].tolist(),
            },
        )


def local_geom_mesh(model, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise NotImplementedError(
            f"DROID CalibAll currently requires mesh visual geoms, got {mujoco.mjtGeom(int(model.geom_type[geom_id])).name}"
        )
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    return (
        np.asarray(model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=np.float32).copy(),
        np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int32).copy(),
    )


def combined_world_mesh(model, data, geom_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    vertex_chunks = []
    face_chunks = []
    offset = 0
    for geom_id in geom_ids:
        local_vertices, local_faces = local_geom_mesh(model, geom_id)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        vertex_chunks.append((local_vertices.astype(np.float64) @ rotation.T + translation).astype(np.float32))
        face_chunks.append(local_faces + offset)
        offset += len(local_vertices)
    return np.concatenate(vertex_chunks), np.concatenate(face_chunks).astype(np.int32)


def pose_delta(initial: np.ndarray, refined: np.ndarray) -> dict[str, float]:
    initial_camera = np.linalg.inv(initial)
    refined_camera = np.linalg.inv(refined)
    translation = float(np.linalg.norm(refined_camera[:3, 3] - initial_camera[:3, 3]))
    relative = initial_camera[:3, :3].T @ refined_camera[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return {"translation_m": translation, "rotation_deg": float(np.degrees(np.arccos(cosine)))}


def prepare_caliball_bundle(
    config: dict,
    session: DroidSession,
    output_root: Path,
    iteration: int,
    model,
    data,
    joints: np.ndarray,
    gripper: np.ndarray,
) -> Path:
    caliball = config["caliball"]
    width = int(caliball["fit_width"])
    height = int(caliball["fit_height"])
    source_width = int(config["render"]["width"])
    source_height = int(config["render"]["height"])
    scale_x, scale_y = width / source_width, height / source_height
    camera_matrix = session.camera_matrix.copy()
    camera_matrix[0, :] *= scale_x
    camera_matrix[1, :] *= scale_y
    geom_group = int(config["render"]["visual_geom_group"])
    geom_ids = [index for index in range(model.ngeom) if int(model.geom_group[index]) == geom_group]
    if not geom_ids:
        raise RuntimeError("No DROID visual geoms for CalibAll")
    indices = frame_indices(config, session, "fit")
    vertices = []
    targets = []
    shared_faces = None
    for index in indices:
        set_droid_state(model, data, joints[index], gripper[index])
        frame_vertices, faces = combined_world_mesh(model, data, geom_ids)
        if shared_faces is None:
            shared_faces = faces
        elif not np.array_equal(shared_faces, faces):
            raise RuntimeError("DROID articulated meshes changed topology")
        vertices.append(frame_vertices)
        mask_path = frame_dir(output_root, session, index) / "sam3_mask.png"
        target = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        targets.append(np.asarray(target) > 0)
    iteration_dir = output_root / "sessions" / session.session_id / "poses" / f"iteration_{iteration:02d}"
    initial_pose = load_pose(iteration_dir / "roma_pose.npz")
    bundle_path = iteration_dir / "caliball" / "bundle.npz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_path,
        vertices=np.stack(vertices).astype(np.float32),
        faces=np.asarray(shared_faces, dtype=np.int32),
        camera_matrix=camera_matrix.astype(np.float32),
        target_masks=np.stack(targets).astype(np.uint8),
        initial_world_to_camera=np.asarray(initial_pose["world_to_camera"], dtype=np.float64),
        frame_indices=np.asarray(indices, dtype=np.int64),
    )
    return bundle_path


def stage_caliball(
    config: dict,
    sessions: list[DroidSession],
    output_root: Path,
    iteration: int,
    steps: int | None,
) -> None:
    if not 0 <= iteration <= int(config["refinement"]["iterations"]):
        raise ValueError("iteration is outside configured range")
    steps = int(steps or config["caliball"]["candidate_steps"][0])
    if steps not in set(map(int, config["caliball"]["candidate_steps"])):
        raise ValueError(f"CalibAll steps {steps} are outside the frozen candidate grid")
    for session in sessions:
        iteration_dir = output_root / "sessions" / session.session_id / "poses" / f"iteration_{iteration:02d}"
        roma_path = iteration_dir / "roma_pose.npz"
        if not roma_path.is_file():
            raise FileNotFoundError(f"Run RoMa iteration first: {roma_path}")
        model, data = make_droid_model(config, session)
        with h5py.File(session.episode.trajectory_h5, "r") as handle:
            joints = np.asarray(handle["observation/robot_state/joint_positions"])
            gripper = np.asarray(handle["observation/robot_state/gripper_position"])
            bundle = prepare_caliball_bundle(
                config, session, output_root, iteration, model, data, joints, gripper
            )
            destination = iteration_dir / "caliball" / f"steps_{steps}"
            configured_command = config["paths"].get("caliball_command")
            if configured_command is None:
                configured_command = [str(config["paths"].get("caliball_python", "python"))]
            command = [
                *map(str, configured_command),
                str(Path(__file__).with_name("run_droid_caliball.py")),
                "--bundle",
                str(bundle),
                "--output-dir",
                str(destination),
                "--max-steps",
                str(steps),
                "--device",
                str(config["matching"]["device"]),
                "--learning-rate",
                str(config["caliball"]["learning_rate"]),
                "--weight-decay",
                str(config["caliball"]["weight_decay"]),
            ]
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
            fit = load_pose(destination / "fit_result.npz")
            refined_pose = {
                "world_to_camera": np.asarray(fit["refined_world_to_camera"], dtype=np.float64),
                "camera_to_world": np.linalg.inv(np.asarray(fit["refined_world_to_camera"], dtype=np.float64)),
                "camera_matrix": session.camera_matrix,
            }
            roma_pose = load_pose(roma_path)
            refined_delta = pose_delta(roma_pose["world_to_camera"], refined_pose["world_to_camera"])
            refined_validation = validation_score(
                config,
                session,
                output_root,
                refined_pose,
                destination / "refined_candidate",
                model,
                data,
                joints,
                gripper,
            )
            roma_summary = json.loads((iteration_dir / "roma_summary.json").read_text(encoding="utf-8"))
            candidates = [
                {
                    "name": "roma",
                    "iteration": iteration,
                    "validation_iou": float(roma_summary["validation"]["iou_macro"]),
                    "pose_delta": {"translation_m": 0.0, "rotation_deg": 0.0},
                    "pose": roma_pose,
                },
                {
                    "name": "caliball",
                    "iteration": iteration,
                    "validation_iou": float(refined_validation["iou_macro"]),
                    "pose_delta": refined_delta,
                    "pose": refined_pose,
                },
            ]
            if iteration > 0:
                parent_path = output_root / "sessions" / session.session_id / "poses" / f"iteration_{iteration - 1:02d}" / "selected_pose.npz"
                parent_pose = load_pose(parent_path)
                parent_validation = validation_score(
                    config,
                    session,
                    output_root,
                    parent_pose,
                    destination / "parent_candidate",
                    model,
                    data,
                    joints,
                    gripper,
                )
                candidates.append(
                    {
                        "name": "parent",
                        "iteration": iteration - 1,
                        "validation_iou": float(parent_validation["iou_macro"]),
                        "pose_delta": {"translation_m": 0.0, "rotation_deg": 0.0},
                        "pose": parent_pose,
                    }
                )
        numerically_valid = bool(np.isfinite(refined_pose["world_to_camera"]).all())
        within_trust_region = bool(
            refined_delta["translation_m"] <= float(config["caliball"]["translation_max_m"])
            and refined_delta["rotation_deg"] <= float(config["caliball"]["rotation_max_deg"])
        )
        if not numerically_valid or not within_trust_region:
            candidates = [candidate for candidate in candidates if candidate["name"] != "caliball"]
        selected = select_best_iteration(candidates)
        save_pose(iteration_dir / "selected_pose.npz", selected.pop("pose"))
        serializable_candidates = []
        for candidate in candidates:
            candidate = dict(candidate)
            candidate.pop("pose", None)
            serializable_candidates.append(candidate)
        write_json(
            destination / "selection_summary.json",
            {
                "session_id": session.session_id,
                "iteration": iteration,
                "steps": steps,
                "numerically_valid": numerically_valid,
                "within_trust_region": within_trust_region,
                "candidates": serializable_candidates,
                "selected": selected,
            },
        )


def stage_raw_eval(
    config: dict,
    sessions: list[DroidSession],
    output_root: Path,
    split_name: str,
    max_frames: int | None,
) -> None:
    args = SimpleNamespace(
        width=int(config["render"]["width"]),
        height=int(config["render"]["height"]),
        visual_geom_group=int(config["render"]["visual_geom_group"]),
    )
    all_records = []
    for session in sessions:
        indices = frame_indices(config, session, split_name)
        if max_frames is not None:
            indices = indices[:max_frames]
        model, data = make_droid_model(config, session)
        with h5py.File(session.episode.trajectory_h5, "r") as handle:
            joints = np.asarray(handle["observation/robot_state/joint_positions"])
            gripper = np.asarray(handle["observation/robot_state/gripper_position"])
            for index in indices:
                destination = frame_dir(output_root, session, index)
                mask_path = destination / "sam3_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Generate {split_name} masks first: {mask_path}")
                target = np.asarray(Image.open(mask_path).convert("L")) > 0
                set_droid_state(model, data, joints[index], gripper[index])
                rendered_path, camera_path, rendered = render_pose_mask(
                    model,
                    data,
                    args,
                    session.camera_matrix,
                    session.raw_world_to_camera,
                    destination,
                    stem="droid_raw_calibration",
                )
                metrics = binary_mask_metrics(rendered, target)
                record = {
                    "session_id": session.session_id,
                    "frame_index": index,
                    "split": split_name,
                    "status": "success",
                    "system": "DROID-raw-calib",
                    "metrics": metrics,
                    "render_mask": str(rendered_path),
                    "camera_npz": str(camera_path),
                }
                write_json(destination / "raw_eval.json", record)
                all_records.append(record)
    write_json(
        output_root / f"raw_eval_{split_name}.json",
        {"split": split_name, "record_count": len(all_records), "records": all_records},
    )


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_root = project_path(config["paths"]["data_root"])
    output_root = project_path(config["paths"]["output_root"])
    sessions = select_sessions(config, build_sessions(data_root), args)
    print(f"config={config_path}")
    print(f"stage={args.stage} scope={args.scope} split={args.split} sessions={len(sessions)}")
    if args.dry_run:
        for session in sessions:
            indices = frame_indices(config, session, args.split)
            if args.max_frames is not None:
                indices = indices[: args.max_frames]
            print(f"{session.session_id}: {len(indices)} frames {indices}")
        return
    if args.stage == "audit":
        stage_audit(config, config_path, sessions, output_root)
    elif args.stage == "masks":
        stage_masks(config, sessions, output_root, args.split, args.max_frames)
    elif args.stage == "raw-eval":
        stage_raw_eval(config, sessions, output_root, args.split, args.max_frames)
    elif args.stage == "roma":
        stage_roma(config, sessions, output_root, args.iteration, args.max_frames)
    else:
        stage_caliball(config, sessions, output_root, args.iteration, args.caliball_steps)


if __name__ == "__main__":
    main()
