"""Shared episode-level PnP primitives for CTRNet-X closed-loop replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeFrameCorrespondences:
    """Post-geometry correspondences extracted from one source frame."""

    frame_id: str
    episode_id: str
    image_points: np.ndarray
    world_points: np.ndarray
    scores: np.ndarray
    camera_matrix: np.ndarray

    def validate(self) -> None:
        image_points = np.asarray(self.image_points)
        world_points = np.asarray(self.world_points)
        scores = np.asarray(self.scores)
        camera_matrix = np.asarray(self.camera_matrix)
        if image_points.ndim != 2 or image_points.shape[1] != 2:
            raise ValueError("image_points must have shape [N, 2]")
        if world_points.ndim != 2 or world_points.shape[1] != 3:
            raise ValueError("world_points must have shape [N, 3]")
        if scores.ndim != 1 or len(scores) != len(image_points):
            raise ValueError("scores must have shape [N] matching image_points")
        if len(world_points) != len(image_points):
            raise ValueError("world_points and image_points must have the same length")
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape [3, 3]")
        if not (
            np.isfinite(image_points).all()
            and np.isfinite(world_points).all()
            and np.isfinite(scores).all()
            and np.isfinite(camera_matrix).all()
        ):
            raise ValueError("correspondence values must be finite")

    def top_k(self, count: int) -> "EpisodeFrameCorrespondences":
        """Keep score-descending candidates with stable original-index ties."""
        self.validate()
        if count < 1:
            raise ValueError("count must be at least one")
        indices = np.argsort(-np.asarray(self.scores), kind="stable")[:count]
        return EpisodeFrameCorrespondences(
            frame_id=self.frame_id,
            episode_id=self.episode_id,
            image_points=np.asarray(self.image_points)[indices],
            world_points=np.asarray(self.world_points)[indices],
            scores=np.asarray(self.scores)[indices],
            camera_matrix=np.asarray(self.camera_matrix),
        )


@dataclass(frozen=True)
class EpisodePose:
    """An episode-wide camera pose estimated from selected source frames."""

    world_to_camera: np.ndarray
    camera_matrix: np.ndarray
    inlier_indices: np.ndarray
    source_frame_ids: tuple[str, ...]
    correspondence_count: int


def concatenate_topk_correspondences(
    frames: Iterable[EpisodeFrameCorrespondences],
    *,
    topk_per_frame: int,
    min_source_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Concatenate stable top-K pairs and verify one camera model per episode."""
    materialized = list(frames)
    for frame in materialized:
        frame.validate()
    selected = [
        frame.top_k(topk_per_frame)
        for frame in materialized
        if len(frame.scores)
    ]
    if len(selected) < min_source_frames:
        raise ValueError(
            f"need at least {min_source_frames} source frames, found {len(selected)}"
        )
    episode_ids = {frame.episode_id for frame in selected}
    if len(episode_ids) != 1:
        raise ValueError("episode PnP source frames must share an episode_id")
    camera_matrix = selected[0].camera_matrix
    if not all(np.allclose(frame.camera_matrix, camera_matrix) for frame in selected[1:]):
        raise ValueError("episode PnP source frames must share one camera matrix")
    image_points = np.concatenate([frame.image_points for frame in selected], axis=0)
    world_points = np.concatenate([frame.world_points for frame in selected], axis=0)
    scores = np.concatenate([frame.scores for frame in selected], axis=0)
    return (
        image_points,
        world_points,
        scores,
        tuple(frame.frame_id for frame in selected),
        np.asarray(camera_matrix, dtype=np.float64),
    )


def solve_episode_pnp(
    frames: Iterable[EpisodeFrameCorrespondences],
    *,
    topk_per_frame: int = 64,
    min_source_frames: int = 2,
    min_pnp_correspondences: int = 6,
    reprojection_threshold: float = 5.0,
    iterations: int = 10_000,
    confidence: float = 0.999,
    seed: int = 90,
) -> EpisodePose:
    """Solve one robust, shared episode pose using all selected frame pairs."""
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("episode PnP requires at least one source frame")
    (
        image_points,
        world_points,
        _scores,
        source_frame_ids,
        camera_matrix,
    ) = concatenate_topk_correspondences(
        frame_list,
        topk_per_frame=topk_per_frame,
        min_source_frames=min_source_frames,
    )
    if len(image_points) < min_pnp_correspondences:
        raise ValueError(
            "not enough aggregated correspondences for episode-level PnP"
        )
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("episode PnP requires the eval extra (opencv-python)") from error

    cv2.setRNGSeed(seed)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(world_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        camera_matrix,
        None,
        iterationsCount=iterations,
        reprojectionError=reprojection_threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_pnp_correspondences:
        raise RuntimeError("episode solvePnPRansac did not return enough inliers")
    inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
    refined = cv2.solvePnPRefineLM(
        np.asarray(world_points, dtype=np.float64)[inlier_indices],
        np.asarray(image_points, dtype=np.float64)[inlier_indices],
        camera_matrix,
        None,
        rvec,
        tvec,
    )
    if isinstance(refined, tuple):
        rvec, tvec = refined
    rotation, _ = cv2.Rodrigues(rvec)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return EpisodePose(
        world_to_camera=world_to_camera,
        camera_matrix=camera_matrix,
        inlier_indices=inlier_indices,
        source_frame_ids=source_frame_ids,
        correspondence_count=len(image_points),
    )


def balanced_episode_shards(
    episode_frame_counts: dict[str, int],
    shard_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Use the historical largest-first/min-load deterministic scheduler."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least one")
    if any(count < 1 for count in episode_frame_counts.values()):
        raise ValueError("every episode must contain at least one frame")
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0 for _ in range(shard_count)]
    for episode_id, count in sorted(
        episode_frame_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[target].append(episode_id)
        loads[target] += count
    return tuple(tuple(shard) for shard in shards)


def assert_complete_frame_coverage(
    requested_frame_ids: Iterable[str],
    evaluated_frame_ids: Iterable[str],
) -> None:
    """Ensure recovery/retry logic did not silently skip requested frames."""
    requested = tuple(requested_frame_ids)
    evaluated = tuple(evaluated_frame_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("requested frame IDs must be unique")
    if set(requested) != set(evaluated) or len(evaluated) != len(requested):
        missing = sorted(set(requested) - set(evaluated))
        unexpected = sorted(set(evaluated) - set(requested))
        raise ValueError(
            f"episode coverage mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}"
        )


__all__ = [
    "EpisodeFrameCorrespondences",
    "EpisodePose",
    "assert_complete_frame_coverage",
    "balanced_episode_shards",
    "concatenate_topk_correspondences",
    "solve_episode_pnp",
]
