"""Pure compatibility layer for the historical DREAM R0--R3 runner.

The archived CF evaluator used a deliberately simple refinement policy:

* a valid orbit-search (R0) PnP pose was required before refinement;
* each later pose-aligned render/match/PnP result replaced its parent directly;
* one failed refinement made the *whole frame* fail and later rounds skipped.

This module records that legacy behavior without importing the current core
pipeline, MuJoCo, RoMa, SAM, OpenCV, or Torch.  It is therefore suitable for
replaying saved Stage1 artifacts and for an execution adapter that supplies the
render/match/PnP work separately.  In particular, it contains no gate,
candidate rejection, rollback, or parent-pose fallback logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .legacy import LEGACY_REFINEMENT_SEMANTIC_ID

LEGACY_ORBIT_STAGE = "orbit_search"
LEGACY_REFINEMENT_STAGE = "pose_projection_refinement"
LEGACY_SKIPPED_REASON = "The previous iteration did not produce a valid PnP pose."
LEGACY_REFINEMENT_FAILURE_REASON = (
    "The projected render did not produce a valid PnP pose."
)


def clean_legacy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Match the historical summary writer by hiding underscore-prefixed data."""
    return {key: value for key, value in record.items() if not key.startswith("_")}


def legacy_match_seed(
    match_seed: int | None,
    frame_index: int,
    iteration: int,
) -> int | None:
    """Return the exact old per-frame/per-iteration RNG seed.

    The archived evaluator applied this value to Torch, NumPy, and OpenCV.
    Dependency-specific seeding is intentionally left to a runtime adapter;
    keeping this function pure makes the formula testable without GPU packages.
    """
    if match_seed is None:
        return None
    return (int(match_seed) + int(frame_index) * 1009 + int(iteration)) % (
        2**31 - 1
    )


def configure_legacy_match_rng(seed: int | None) -> None:
    """Apply the archived seed to Torch, NumPy, and OpenCV.

    The pure state machine receives the derived seed as a callback argument so
    it stays importable without the GPU evaluation stack.  A concrete
    render/match adapter calls this function immediately before its matcher
    invocation to preserve the historical three-library seeding behavior.
    """
    if seed is None:
        return
    try:
        import cv2
        import torch
    except ImportError as error:  # pragma: no cover - optional runtime deps
        raise RuntimeError(
            "legacy match seeding requires the eval PyTorch and OpenCV runtime"
        ) from error
    seed_value = int(seed)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    cv2.setRNGSeed(seed_value)


def _copied_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.array(value, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _matrix(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = _copied_array(value, name)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


def _points(value: Any, name: str, width: int) -> np.ndarray:
    array = _copied_array(value, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")
    return array


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    array = _copied_array(value, name)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape [{length}]")
    return array


@dataclass(frozen=True)
class LegacyPoseArtifact:
    """The seven pose arrays saved by the historical ``best_pose.npz`` file."""

    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    image_points: np.ndarray
    world_points: np.ndarray
    scores: np.ndarray
    inlier_indices: np.ndarray
    reprojection_errors: np.ndarray

    def __post_init__(self) -> None:
        world_to_camera = _matrix(
            self.world_to_camera, "world_to_camera", (4, 4)
        )
        camera_to_world = _matrix(
            self.camera_to_world, "camera_to_world", (4, 4)
        )
        image_points = _points(self.image_points, "image_points", 2)
        world_points = _points(self.world_points, "world_points", 3)
        if len(image_points) != len(world_points):
            raise ValueError("image_points and world_points must have the same length")
        scores = _vector(self.scores, "scores", len(image_points))
        reprojection_errors = _vector(
            self.reprojection_errors, "reprojection_errors", len(image_points)
        )
        inlier_indices = _copied_array(self.inlier_indices, "inlier_indices")
        if inlier_indices.ndim != 1:
            raise ValueError("inlier_indices must have shape [N]")
        if inlier_indices.dtype.kind not in "iu":
            raise ValueError("inlier_indices must use an integer dtype")
        if np.any(inlier_indices < 0) or np.any(inlier_indices >= len(image_points)):
            raise ValueError("inlier_indices are out of range")
        object.__setattr__(self, "world_to_camera", world_to_camera)
        object.__setattr__(self, "camera_to_world", camera_to_world)
        object.__setattr__(self, "image_points", image_points)
        object.__setattr__(self, "world_points", world_points)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "inlier_indices", inlier_indices)
        object.__setattr__(self, "reprojection_errors", reprojection_errors)

    @property
    def camera_to_robot_base(self) -> np.ndarray:
        """Use the historical NPZ field name without changing internal semantics."""
        return self.camera_to_world

    def npz_payload(
        self,
        camera_matrix: np.ndarray,
        keypoint_metrics: Mapping[str, Any],
    ) -> dict[str, np.ndarray | float]:
        """Return exactly the fields emitted by old ``save_pose_npz``."""
        matrix = _matrix(camera_matrix, "camera_matrix", (3, 3))
        try:
            metric_summary = keypoint_metrics["summary"]
            pixel_error = float(metric_summary["pixel_error_mean"])
            add_error = float(metric_summary["keypoint_add_mean_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "keypoint_metrics must contain summary.pixel_error_mean and "
                "summary.keypoint_add_mean_m"
            ) from error
        if not np.isfinite((pixel_error, add_error)).all():
            raise ValueError("legacy keypoint summary values must be finite")
        return {
            "world_to_camera": self.world_to_camera,
            "camera_to_robot_base": self.camera_to_world,
            "image_points": self.image_points,
            "world_points": self.world_points,
            "scores": self.scores,
            "inlier_indices": self.inlier_indices,
            "reprojection_errors": self.reprojection_errors,
            "camera_matrix": matrix,
            "keypoint_pixel_error_mean": pixel_error,
            "keypoint_add_mean_m": add_error,
        }


def save_legacy_pose_npz(
    output_dir: Path,
    pose: LegacyPoseArtifact,
    camera_matrix: np.ndarray,
    keypoint_metrics: Mapping[str, Any],
    *,
    filename: str = "best_pose.npz",
) -> Path:
    """Write a historical pose NPZ using the old field names and compression."""
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a single non-empty file name")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    np.savez_compressed(path, **pose.npz_payload(camera_matrix, keypoint_metrics))
    return path


def load_legacy_pose_npz(path: Path) -> LegacyPoseArtifact:
    """Load the exact pose-array subset required by the historical refinement."""
    required = (
        "world_to_camera",
        "camera_to_robot_base",
        "image_points",
        "world_points",
        "scores",
        "inlier_indices",
        "reprojection_errors",
    )
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as saved:
            missing = [name for name in required if name not in saved]
            if missing:
                raise ValueError(
                    f"legacy pose artifact is missing required arrays: {missing}"
                )
            return LegacyPoseArtifact(
                world_to_camera=saved["world_to_camera"],
                camera_to_world=saved["camera_to_robot_base"],
                image_points=saved["image_points"],
                world_points=saved["world_points"],
                scores=saved["scores"],
                inlier_indices=saved["inlier_indices"],
                reprojection_errors=saved["reprojection_errors"],
            )
    except OSError as error:
        raise ValueError(f"cannot load legacy pose artifact: {source}") from error


def resolve_legacy_stage1_pose_path(
    frame_dir: Path,
    summary: Mapping[str, Any],
) -> Path:
    """Use the historical default-first/fallback-second pose lookup order."""
    default_path = Path(frame_dir) / "best_pose.npz"
    if default_path.is_file():
        return default_path
    configured = summary.get("pose_npz")
    if configured:
        configured_path = Path(str(configured))
        if configured_path.is_file():
            return configured_path
    raise FileNotFoundError(f"Missing initial pose: {default_path}")


@dataclass(frozen=True)
class LegacyEstimate:
    """A successful old-evaluator pose plus serializable summary fields."""

    pose: LegacyPoseArtifact
    pnp: Mapping[str, Any]
    render_path: str | None
    camera_npz: str | None
    keypoint_metrics: Mapping[str, Any]
    pose_npz: str | None = None
    keypoint_visualization: str | None = None

    def __post_init__(self) -> None:
        pnp = dict(self.pnp)
        if pnp.get("status") != "success":
            raise ValueError("a LegacyEstimate requires pnp.status == 'success'")
        if not isinstance(self.keypoint_metrics, Mapping):
            raise ValueError("keypoint_metrics must be a mapping")
        object.__setattr__(self, "pnp", pnp)
        object.__setattr__(self, "keypoint_metrics", dict(self.keypoint_metrics))

    def summary_pose_fields(self) -> dict[str, Any]:
        """Fields historically copied into successful R0/R1/R2/R3 records."""
        return {
            "camera_npz": self.camera_npz,
            "pose_npz": self.pose_npz,
            "keypoint_visualization": self.keypoint_visualization,
            "camera_to_robot_base": self.pnp.get("camera_to_robot_base"),
            "world_to_camera": self.pnp.get("world_to_camera"),
            "keypoint_metrics": dict(self.keypoint_metrics),
            "best_pnp": dict(self.pnp),
        }


@dataclass(frozen=True)
class LegacyStage1Artifact:
    """Loaded Stage1 output used as R0 by the historical R1--R3 evaluator."""

    summary: Mapping[str, Any]
    frame_dir: Path
    estimate: LegacyEstimate | None
    pose_path: Path | None

    def __post_init__(self) -> None:
        summary = dict(self.summary)
        status = summary.get("status")
        if status not in {"success", "failed", "error"}:
            raise ValueError("Stage1 summary status must be success, failed, or error")
        if status == "success" and self.estimate is None:
            raise ValueError("a successful Stage1 summary requires a pose estimate")
        if status != "success" and self.estimate is not None:
            raise ValueError("a failed Stage1 summary cannot carry a pose estimate")
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "frame_dir", Path(self.frame_dir))
        if self.pose_path is not None:
            object.__setattr__(self, "pose_path", Path(self.pose_path))

    @property
    def is_successful(self) -> bool:
        return self.estimate is not None

    @property
    def views(self) -> list[Any]:
        views = self.summary.get("views", [])
        if not isinstance(views, list):
            raise ValueError("Stage1 summary views must be a list")
        return list(views)

    def initial_failure_pnp(self) -> dict[str, Any]:
        """Reproduce the old fallback record for a non-successful Stage1 summary."""
        return {"status": "failed", "reason": self.summary.get("reason")}

    @property
    def input_mask_path(self) -> Path | None:
        """Return the replayable Stage1 SAM mask path when the old artifact has it."""
        path = self.frame_dir / "input_mask.png"
        return path if path.is_file() else None


def load_legacy_stage1_artifact(
    initial_results_dir: Path,
    frame_index: int,
) -> LegacyStage1Artifact:
    """Load one archived Stage1 frame exactly as the old evaluator did.

    ``best_pose.npz`` takes priority.  When it is absent, the historical
    ``summary['pose_npz']`` fallback is respected, including an absolute path
    in a legacy artifact.
    """
    frame_dir = Path(initial_results_dir) / f"{int(frame_index):06d}"
    summary_path = frame_dir / "frame_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing initial frame summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid initial frame summary: {summary_path}") from error
    if not isinstance(summary, Mapping):
        raise ValueError(f"Initial frame summary must be an object: {summary_path}")
    if summary.get("status") != "success":
        return LegacyStage1Artifact(summary, frame_dir, None, None)

    pose_path = resolve_legacy_stage1_pose_path(frame_dir, summary)
    pose = load_legacy_pose_npz(pose_path)
    try:
        estimate = LegacyEstimate(
            pose=pose,
            pnp=summary["best_pnp"],
            render_path=summary["best_render_path"],
            camera_npz=summary["best_camera_npz"],
            keypoint_metrics=summary["keypoint_metrics"],
            pose_npz=str(pose_path),
            keypoint_visualization=summary.get("keypoint_visualization"),
        )
    except KeyError as error:
        raise ValueError(
            f"Successful Stage1 summary is missing {error.args[0]!r}: {summary_path}"
        ) from error
    return LegacyStage1Artifact(summary, frame_dir, estimate, pose_path)


def legacy_pose_delta(
    previous: LegacyPoseArtifact,
    current: LegacyPoseArtifact,
) -> dict[str, float]:
    """Match the old camera-translation/rotation diagnostics exactly."""
    previous_camera = previous.camera_to_world
    current_camera = current.camera_to_world
    translation = np.linalg.norm(current_camera[:3, 3] - previous_camera[:3, 3])
    relative_rotation = previous_camera[:3, :3].T @ current_camera[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    return {
        "camera_translation_m": float(translation),
        "camera_rotation_deg": float(np.degrees(np.arccos(cosine))),
    }


@dataclass(frozen=True)
class LegacyRefinementAttempt:
    """One externally executed projected-render/match/PnP attempt.

    ``candidate is None`` represents any non-successful PnP result.  A runtime
    adapter owns matching/PnP and supplies its clean match record here.
    """

    render_path: str
    match: Mapping[str, Any]
    candidate: LegacyEstimate | None = None
    camera_npz: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.match, Mapping):
            raise ValueError("match must be a mapping")
        match = dict(self.match)
        pnp = match.get("pnp")
        if not isinstance(pnp, Mapping) or not isinstance(pnp.get("status"), str):
            raise ValueError("match.pnp.status must be present")
        if self.candidate is None and pnp["status"] == "success":
            raise ValueError("a successful match PnP requires a candidate estimate")
        if self.candidate is not None and pnp["status"] != "success":
            raise ValueError("a candidate estimate requires match.pnp.status == 'success'")
        object.__setattr__(self, "match", match)


LegacyRefinementCallback = Callable[
    [LegacyEstimate, int, int | None], LegacyRefinementAttempt
]


@dataclass(frozen=True)
class LegacyRefinementTrajectory:
    """The legacy R0--R3 decision result and the historical iteration records."""

    semantic_id: str
    frame_status: str
    reason: str | None
    iterations: tuple[dict[str, Any], ...]
    final_estimate: LegacyEstimate | None


def _stage0_record(stage1: LegacyStage1Artifact) -> dict[str, Any]:
    if stage1.estimate is None:
        return {
            "iteration": 0,
            "stage": LEGACY_ORBIT_STAGE,
            "status": "failed",
            "best_render_path": stage1.summary.get("best_render_path"),
            "best_pnp": stage1.initial_failure_pnp(),
            "views": stage1.views,
        }
    fields = stage1.estimate.summary_pose_fields()
    fields.pop("camera_npz")
    return {
        "iteration": 0,
        "stage": LEGACY_ORBIT_STAGE,
        "status": "success",
        "best_render_path": stage1.estimate.render_path,
        "best_camera_npz": stage1.estimate.camera_npz,
        **fields,
        "views": stage1.views,
    }


def _skipped_iteration_record(iteration: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "stage": LEGACY_REFINEMENT_STAGE,
        "status": "skipped",
        "reason": LEGACY_SKIPPED_REASON,
    }


def run_legacy_unconditional_trajectory(
    stage1: LegacyStage1Artifact,
    *,
    frame_index: int,
    refinement_iterations: int,
    match_seed: int | None,
    refinement_attempt: LegacyRefinementCallback,
) -> LegacyRefinementTrajectory:
    """Run historical R0--R3 state transitions without changing their semantics.

    Every successful candidate replaces ``current`` unconditionally.  A failed
    candidate returns a failed frame even if R0/R1/R2 was valid; no previous
    pose is reported as the final result and no later callback is invoked.
    """
    if not 0 <= int(refinement_iterations) <= 3:
        raise ValueError("legacy DREAM refinement_iterations must be in [0, 3]")
    requested = int(refinement_iterations)
    iterations = [_stage0_record(stage1)]
    if stage1.estimate is None:
        iterations.extend(
            _skipped_iteration_record(iteration)
            for iteration in range(1, requested + 1)
        )
        return LegacyRefinementTrajectory(
            semantic_id=LEGACY_REFINEMENT_SEMANTIC_ID,
            frame_status="failed",
            reason="No render view produced a valid PnP pose.",
            iterations=tuple(iterations),
            final_estimate=None,
        )

    current = stage1.estimate
    for iteration in range(1, requested + 1):
        seed = legacy_match_seed(match_seed, frame_index, iteration)
        # The archived evaluator did not turn an unexpected render/match
        # exception into a synthetic refinement state.  It propagated to the
        # frame-level handler, which wrote one ``status=error`` summary with a
        # traceback.  Keep that boundary here: only an ordinary non-successful
        # PnP result has the historical failed-and-skipped trajectory below.
        attempt = refinement_attempt(current, iteration, seed)
        if not isinstance(attempt, LegacyRefinementAttempt):
            raise TypeError("refinement_attempt must return LegacyRefinementAttempt")
        clean_match = clean_legacy_record(attempt.match)
        if attempt.candidate is None:
            iterations.append(
                {
                    "iteration": iteration,
                    "stage": LEGACY_REFINEMENT_STAGE,
                    "status": "failed",
                    "render_source_world_to_camera": current.pnp.get("world_to_camera"),
                    "render_path": str(attempt.render_path),
                    "camera_npz": attempt.camera_npz,
                    "match": clean_match,
                    "reason": LEGACY_REFINEMENT_FAILURE_REASON,
                }
            )
            iterations.extend(
                _skipped_iteration_record(skipped)
                for skipped in range(iteration + 1, requested + 1)
            )
            return LegacyRefinementTrajectory(
                semantic_id=LEGACY_REFINEMENT_SEMANTIC_ID,
                frame_status="failed",
                reason=(
                    f"Refinement iteration {iteration} did not produce a valid "
                    "PnP pose."
                ),
                iterations=tuple(iterations),
                final_estimate=None,
            )

        candidate = attempt.candidate
        iterations.append(
            {
                "iteration": iteration,
                "stage": LEGACY_REFINEMENT_STAGE,
                "status": "success",
                "match_seed": seed,
                "render_source_world_to_camera": current.pnp.get("world_to_camera"),
                "render_path": str(attempt.render_path),
                **candidate.summary_pose_fields(),
                "pose_delta_from_previous": legacy_pose_delta(current.pose, candidate.pose),
                "match": clean_match,
            }
        )
        # This direct assignment is the key historical behavior: no gate or rollback.
        current = candidate

    return LegacyRefinementTrajectory(
        semantic_id=LEGACY_REFINEMENT_SEMANTIC_ID,
        frame_status="success",
        reason=None,
        iterations=tuple(iterations),
        final_estimate=current,
    )


__all__ = [
    "LEGACY_ORBIT_STAGE",
    "LEGACY_REFINEMENT_FAILURE_REASON",
    "LEGACY_REFINEMENT_SEMANTIC_ID",
    "LEGACY_REFINEMENT_STAGE",
    "LEGACY_SKIPPED_REASON",
    "LegacyEstimate",
    "LegacyPoseArtifact",
    "LegacyRefinementAttempt",
    "LegacyRefinementCallback",
    "LegacyRefinementTrajectory",
    "LegacyStage1Artifact",
    "clean_legacy_record",
    "configure_legacy_match_rng",
    "legacy_match_seed",
    "legacy_pose_delta",
    "load_legacy_pose_npz",
    "load_legacy_stage1_artifact",
    "resolve_legacy_stage1_pose_path",
    "run_legacy_unconditional_trajectory",
    "save_legacy_pose_npz",
]
