"""DROID raw-video discovery, split and evaluation helpers.

The unit of calibration is one ``episode x fixed external camera`` session.
All poses are OpenCV ``T_camera<-robot_base`` transforms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DroidEpisode:
    rank: int
    uuid: str
    lab: str
    episode_dir: Path
    trajectory_h5: Path
    ext1_serial: str
    ext2_serial: str
    frame_count: int


@dataclass(frozen=True)
class DroidSession:
    episode: DroidEpisode
    camera_name: str
    camera_serial: str
    video_path: Path
    camera_matrix: np.ndarray
    distortion: np.ndarray
    raw_world_to_camera: np.ndarray

    @property
    def session_id(self) -> str:
        return f"{self.episode.uuid}__{self.camera_name}__{self.camera_serial}"


@dataclass(frozen=True)
class FrameSplit:
    fit: tuple[int, ...]
    validation: tuple[int, ...]
    heldout: tuple[int, ...]

    def validate(self) -> None:
        fit, validation, heldout = map(set, (self.fit, self.validation, self.heldout))
        if fit & validation or fit & heldout or validation & heldout:
            raise ValueError("DROID fit/validation/heldout splits overlap")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(episode_dir: Path) -> Path:
    paths = sorted(episode_dir.glob("metadata_*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one metadata JSON in {episode_dir}, got {len(paths)}")
    return paths[0]


def discover_episodes(data_root: Path) -> list[DroidEpisode]:
    selection = read_json(data_root / "selection_manifest.json")
    episodes: list[DroidEpisode] = []
    for selected in selection["accepted"]:
        episode_dir = data_root / "raw_episodes" / selected["relative_episode_path"]
        metadata = read_json(_metadata_path(episode_dir))
        trajectory = episode_dir / "trajectory.h5"
        if not trajectory.is_file():
            raise FileNotFoundError(trajectory)
        episodes.append(
            DroidEpisode(
                rank=int(selected["rank"]),
                uuid=str(metadata["uuid"]),
                lab=str(metadata["lab"]),
                episode_dir=episode_dir,
                trajectory_h5=trajectory,
                ext1_serial=str(metadata["ext1_cam_serial"]),
                ext2_serial=str(metadata["ext2_cam_serial"]),
                frame_count=int(metadata["trajectory_length"]),
            )
        )
    if len(episodes) != 10:
        raise RuntimeError(f"Expected 10 selected DROID episodes, got {len(episodes)}")
    return episodes


def camera_matrix_from_annotation(annotation: dict[str, Any]) -> np.ndarray:
    fx, cx, fy, cy = map(float, annotation["cameraMatrix"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def euler_xyz_matrix(angles: Iterable[float]) -> np.ndarray:
    x, y, z = map(float, angles)
    sx, cx = np.sin(x), np.cos(x)
    sy, cy = np.sin(y), np.cos(y)
    sz, cz = np.sin(z), np.cos(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def camera_to_base_vector_to_world_to_camera(vector: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(vector), dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Expected six camera-to-base values, got {values.shape}")
    camera_to_base = np.eye(4, dtype=np.float64)
    camera_to_base[:3, :3] = euler_xyz_matrix(values[3:])
    camera_to_base[:3, 3] = values[:3]
    return np.linalg.inv(camera_to_base)


def build_sessions(data_root: Path, episodes: list[DroidEpisode] | None = None) -> list[DroidSession]:
    import h5py

    episodes = discover_episodes(data_root) if episodes is None else episodes
    intrinsics = read_json(data_root / "official_annotations" / "intrinsics.json")
    sessions: list[DroidSession] = []
    for episode in episodes:
        episode_intrinsics = intrinsics.get(episode.uuid)
        if not episode_intrinsics:
            raise KeyError(f"No official intrinsics for {episode.uuid}")
        with h5py.File(episode.trajectory_h5, "r") as handle:
            for camera_name, serial in (("ext1", episode.ext1_serial), ("ext2", episode.ext2_serial)):
                key = f"observation/camera_extrinsics/{serial}_left"
                if key not in handle:
                    raise KeyError(f"Missing {key} in {episode.trajectory_h5}")
                camera_to_base = np.asarray(handle[key][0], dtype=np.float64)
                annotation = episode_intrinsics[serial]
                video_path = episode.episode_dir / "recordings" / "MP4" / f"{serial}.mp4"
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                sessions.append(
                    DroidSession(
                        episode=episode,
                        camera_name=camera_name,
                        camera_serial=serial,
                        video_path=video_path,
                        camera_matrix=camera_matrix_from_annotation(annotation),
                        distortion=np.asarray(annotation.get("distCoeffs", []), dtype=np.float64),
                        raw_world_to_camera=camera_to_base_vector_to_world_to_camera(camera_to_base),
                    )
                )
    if len(sessions) != 20:
        raise RuntimeError(f"Expected 20 fixed external-camera sessions, got {len(sessions)}")
    return sessions


def evenly_spaced(indices: Iterable[int], count: int) -> tuple[int, ...]:
    values = tuple(sorted(set(map(int, indices))))
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0 or not values:
        return ()
    if count >= len(values):
        return values
    positions = np.linspace(0, len(values) - 1, count)
    selected = tuple(values[int(round(position))] for position in positions)
    if len(set(selected)) != count:
        raise RuntimeError("Even sampling produced duplicate indices")
    return selected


def make_frame_split(video_frame_count: int, fit_count: int = 16, validation_count: int = 8) -> FrameSplit:
    all_indices = tuple(range(int(video_frame_count)))
    heldout = tuple(index for index in all_indices if index % 5 == 0)
    calibration = tuple(index for index in all_indices if index % 5 != 0)
    validation = evenly_spaced(calibration, min(validation_count, len(calibration)))
    fit_candidates = tuple(index for index in calibration if index not in set(validation))
    fit = evenly_spaced(fit_candidates, min(fit_count, len(fit_candidates)))
    split = FrameSplit(fit=fit, validation=validation, heldout=heldout)
    split.validate()
    return split


def binary_mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    intersection = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    return {
        "iou": float(intersection / union) if union else 0.0,
        "intersection_px": intersection,
        "union_px": union,
        "prediction_px": int(pred.sum()),
        "target_px": int(truth.sum()),
    }


def select_best_iteration(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select by validation IoU, then smaller pose jump, then earlier iteration."""
    candidates = list(records)
    if not candidates:
        raise ValueError("No iteration candidates")
    for record in candidates:
        if "validation_iou" not in record or "iteration" not in record:
            raise KeyError("Iteration record needs validation_iou and iteration")
    return max(
        candidates,
        key=lambda record: (
            float(record["validation_iou"]),
            -float(record.get("pose_delta", {}).get("translation_m", 0.0)),
            -float(record.get("pose_delta", {}).get("rotation_deg", 0.0)),
            -int(record["iteration"]),
        ),
    )


def solve_shared_pose(
    correspondence_sets: Iterable[dict[str, Any]],
    camera_matrix: np.ndarray,
    *,
    topk_per_frame: int = 64,
    min_source_frames: int = 2,
    min_correspondences: int = 6,
    reprojection_error_px: float = 5.0,
    iterations: int = 10_000,
    confidence: float = 0.999,
    seed: int = 20260717,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pool frame-balanced correspondences and solve one session pose."""
    import cv2

    image_chunks = []
    world_chunks = []
    score_chunks = []
    source_frames = []
    counts_before = []
    counts_after = []
    for item in correspondence_sets:
        image_points = np.asarray(item["image_points"], dtype=np.float64).reshape(-1, 2)
        world_points = np.asarray(item["world_points"], dtype=np.float64).reshape(-1, 3)
        scores = np.asarray(item["scores"], dtype=np.float64).reshape(-1)
        if not (len(image_points) == len(world_points) == len(scores)):
            raise ValueError("Correspondence arrays have different lengths")
        if not len(scores):
            continue
        if not np.isfinite(image_points).all() or not np.isfinite(world_points).all() or not np.isfinite(scores).all():
            raise ValueError("Non-finite correspondence")
        order = np.argsort(-scores, kind="stable")[: min(topk_per_frame, len(scores))]
        image_chunks.append(image_points[order])
        world_chunks.append(world_points[order])
        score_chunks.append(scores[order])
        source_frames.append(int(item["frame_index"]))
        counts_before.append(len(scores))
        counts_after.append(len(order))
    if len(source_frames) < min_source_frames:
        raise RuntimeError(f"Only {len(source_frames)} source frames; need {min_source_frames}")
    image_points = np.concatenate(image_chunks, axis=0)
    world_points = np.concatenate(world_chunks, axis=0)
    scores = np.concatenate(score_chunks, axis=0)
    if len(image_points) < min_correspondences:
        raise RuntimeError(f"Only {len(image_points)} correspondences; need {min_correspondences}")
    cv2.setRNGSeed(int(seed) % (2**31 - 1))
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=world_points,
        imagePoints=image_points,
        cameraMatrix=np.asarray(camera_matrix, dtype=np.float64),
        distCoeffs=None,
        iterationsCount=int(iterations),
        reprojectionError=float(reprojection_error_px),
        confidence=float(confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_correspondences:
        count = 0 if inliers is None else len(inliers)
        raise RuntimeError(f"PnP RANSAC failed with {count} inliers")
    inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        world_points[inlier_indices],
        image_points[inlier_indices],
        np.asarray(camera_matrix, dtype=np.float64),
        None,
        rvec,
        tvec,
    )
    projected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = tvec.reshape(3)
    pose = {
        "world_to_camera": world_to_camera,
        "camera_to_world": np.linalg.inv(world_to_camera),
        "camera_matrix": np.asarray(camera_matrix, dtype=np.float64),
        "rvec": np.asarray(rvec, dtype=np.float64),
        "tvec": np.asarray(tvec, dtype=np.float64),
        "inlier_indices": inlier_indices,
        "reprojection_errors": errors,
        "image_points": image_points,
        "world_points": world_points,
        "scores": scores,
    }
    details = {
        "status": "success",
        "seed": int(seed),
        "source_frame_indices": source_frames,
        "correspondence_count_before_topk": int(sum(counts_before)),
        "correspondence_count_after_topk": int(sum(counts_after)),
        "pnp_inlier_count": int(len(inlier_indices)),
        "pnp_inlier_ratio": float(len(inlier_indices) / len(image_points)),
        "pnp_inlier_reprojection_error_mean_px": float(errors[inlier_indices].mean()),
        "pnp_inlier_reprojection_error_median_px": float(np.median(errors[inlier_indices])),
    }
    return pose, details
