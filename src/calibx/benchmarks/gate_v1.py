"""Gate-v1 evidence primitives for pose-aligned refinement.

This is the d7-style gated protocol used by the verified SAM ablation path.
It is intentionally separate from ``dream_real.legacy``: historical full
experiments used unconditional PnP replacement and are not semantic aliases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


GATE_ID = "gate-v1"
DETERMINISTIC_SEED = 20260717


def _opencv():
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("gate-v1 image operations require the eval extra") from error
    return cv2


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize evidence inputs deterministically for provenance fingerprints."""
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes({"dtype": str(array.dtype), "shape": array.shape})
    return sha256_bytes(header + array.tobytes())


def pose_sha256(world_to_camera: np.ndarray) -> str:
    """Fingerprint an OpenCV ``T_camera<-robot_base`` in d7's dtype contract."""
    return array_sha256(np.asarray(world_to_camera, dtype=np.float64))


def resize_camera_matrix_half_pixel(
    camera_matrix: np.ndarray,
    raw_size: tuple[int, int],
    process_size: tuple[int, int],
) -> np.ndarray:
    """Resize intrinsics under the OpenCV integer-pixel-center convention."""
    raw_height, raw_width = raw_size
    process_height, process_width = process_size
    scale_x = process_width / raw_width
    scale_y = process_height / raw_height
    result = np.asarray(camera_matrix, dtype=np.float64).copy()
    result[0, 0] *= scale_x
    result[1, 1] *= scale_y
    result[0, 1] *= scale_x
    result[0, 2] = (result[0, 2] + 0.5) * scale_x - 0.5
    result[1, 2] = (result[1, 2] + 0.5) * scale_y - 0.5
    return result


@dataclass(frozen=True)
class CameraModel:
    """Raw official grid and rectified pinhole processing grid."""

    raw_size: tuple[int, int]
    process_size: tuple[int, int]
    K_raw: np.ndarray
    distortion_raw: np.ndarray | None
    K_process: np.ndarray
    distortion_state: str
    raw_to_process_fingerprint: str
    evaluation_projection_fingerprint: str
    fingerprint: str

    @classmethod
    def create(
        cls,
        raw_size: tuple[int, int],
        process_size: tuple[int, int],
        K_raw: np.ndarray,
        distortion_raw: np.ndarray | None,
        distortion_state: str,
    ) -> "CameraModel":
        state = str(distortion_state).lower()
        if state not in {"rectified", "distorted", "unknown"}:
            raise ValueError(f"unsupported distortion state: {distortion_state!r}")
        if state == "unknown":
            raise ValueError("gate-v1 requires rectified or distorted camera input")
        matrix = np.asarray(K_raw, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("K_raw must be a finite 3x3 matrix")
        if abs(float(matrix[0, 1])) > 1e-12:
            raise ValueError("gate-v1 requires explicit skew rectification")
        distortion = None
        if distortion_raw is not None:
            distortion = np.asarray(distortion_raw, dtype=np.float64).reshape(-1)
            if not np.isfinite(distortion).all():
                raise ValueError("raw distortion coefficients must be finite")
        if state == "rectified":
            if distortion is not None and np.any(np.abs(distortion) > 1e-12):
                raise ValueError("a rectified camera cannot have non-zero distortion")
            distortion = None
        elif distortion is None:
            raise ValueError("a distorted camera requires distortion coefficients")
        processed = resize_camera_matrix_half_pixel(matrix, raw_size, process_size)
        mapping_payload = {
            "raw_size": raw_size,
            "process_size": process_size,
            "K_raw": matrix,
            "distortion_raw": distortion,
            "K_process": processed,
            "rgb_resampling": "bilinear",
            "mask_resampling": "nearest",
            "crop_policy": "forbidden",
            "pixel_center": "opencv_integer_center",
        }
        evaluation_payload = {
            "raw_size": raw_size,
            "K_raw": matrix,
            "distortion_raw": distortion,
            "distortion_state": state,
        }
        raw_to_process_fingerprint = sha256_bytes(canonical_json_bytes(mapping_payload))
        evaluation_projection_fingerprint = sha256_bytes(
            canonical_json_bytes(evaluation_payload)
        )
        fingerprint = sha256_bytes(
            canonical_json_bytes(
                {
                    "raw_to_process": raw_to_process_fingerprint,
                    "evaluation_projection": evaluation_projection_fingerprint,
                }
            )
        )
        return cls(
            raw_size=tuple(int(value) for value in raw_size),
            process_size=tuple(int(value) for value in process_size),
            K_raw=matrix,
            distortion_raw=distortion,
            K_process=processed,
            distortion_state=state,
            raw_to_process_fingerprint=raw_to_process_fingerprint,
            evaluation_projection_fingerprint=evaluation_projection_fingerprint,
            fingerprint=fingerprint,
        )

    def to_record(self) -> dict[str, Any]:
        """Emit the d7 camera-grid provenance fields used by frame records."""
        return {
            "raw_size": list(self.raw_size),
            "process_size": list(self.process_size),
            "K_raw": self.K_raw.tolist(),
            "distortion_raw": (
                None if self.distortion_raw is None else self.distortion_raw.tolist()
            ),
            "K_process": self.K_process.tolist(),
            "distortion_state": self.distortion_state,
            "raw_to_process_fingerprint": self.raw_to_process_fingerprint,
            "evaluation_projection_fingerprint": self.evaluation_projection_fingerprint,
            "fingerprint": self.fingerprint,
        }

    def process_rgb(self, image: np.ndarray) -> np.ndarray:
        return self._remap(image, interpolation="linear")

    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        return self._remap(mask.astype(np.uint8), interpolation="nearest").astype(bool)

    def _remap(self, image: np.ndarray, *, interpolation: str) -> np.ndarray:
        if image.shape[:2] != self.raw_size:
            raise ValueError(
                f"raw image shape {image.shape[:2]} does not match camera raw_size "
                f"{self.raw_size}"
            )
        cv2 = _opencv()
        method = cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_NEAREST
        height, width = self.process_size
        if self.distortion_raw is None:
            if image.shape[:2] == self.process_size:
                return image.copy()
            return cv2.resize(image, (width, height), interpolation=method)
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.distortion_raw,
            None,
            self.K_process,
            (width, height),
            cv2.CV_32FC1,
        )
        return cv2.remap(image, map_x, map_y, method, borderMode=cv2.BORDER_CONSTANT)

    def raw_to_process_points(self, points: np.ndarray) -> np.ndarray:
        """Map raw-image observations onto the processing grid for PnP/matching."""
        cv2 = _opencv()
        source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        converted = cv2.undistortPoints(
            source,
            self.K_raw,
            self.distortion_raw,
            P=self.K_process,
        )
        return converted.reshape(-1, 2)

    def process_to_raw_points(self, points: np.ndarray) -> np.ndarray:
        """Map processing-grid points back to the official raw evaluation grid."""
        cv2 = _opencv()
        process_points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        normalized = np.column_stack(
            [
                (process_points[:, 0] - self.K_process[0, 2]) / self.K_process[0, 0],
                (process_points[:, 1] - self.K_process[1, 2]) / self.K_process[1, 1],
                np.ones(len(process_points), dtype=np.float64),
            ]
        )
        projected, _ = cv2.projectPoints(
            normalized,
            np.zeros(3),
            np.zeros(3),
            self.K_raw,
            self.distortion_raw,
        )
        return projected.reshape(-1, 2)

    def project_raw(
        self,
        points_robot: np.ndarray,
        world_to_camera: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project robot points onto the official raw grid and retain depth."""
        cv2 = _opencv()
        robot_points = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
        transform = np.asarray(world_to_camera, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("world_to_camera must be 4x4")
        camera_points = (
            transform[:3, :3] @ robot_points.T + transform[:3, 3:4]
        ).T
        rvec, _ = cv2.Rodrigues(transform[:3, :3])
        projected, _ = cv2.projectPoints(
            robot_points,
            rvec,
            transform[:3, 3],
            self.K_raw,
            self.distortion_raw,
        )
        return projected.reshape(-1, 2), camera_points


@dataclass(frozen=True)
class GateV1Thresholds:
    min_inliers: int = 12
    min_inlier_ratio: float = 0.10
    min_positive_depth_ratio: float = 0.95
    max_reprojection_diagonal_ratio: float = 0.005
    min_image_hull_area_ratio: float = 0.005
    min_world_second_spread_ratio: float = 0.02
    max_rotation_delta_deg: float = 20.0
    max_camera_center_delta_ratio: float = 0.25
    max_mask_iou_drop: float = 0.02
    small_rotation_delta_deg: float = 0.10
    small_camera_center_delta_ratio: float = 0.001


DEFAULT_GATE_V1_THRESHOLDS = GateV1Thresholds()


def is_proper_se3(transform: np.ndarray) -> tuple[bool, list[str]]:
    """Validate the rigid transform constraints used before gate acceptance."""
    matrix = np.asarray(transform, dtype=np.float64)
    reasons: list[str] = []
    if matrix.shape != (4, 4):
        return False, ["se3_shape"]
    if not np.isfinite(matrix).all():
        reasons.append("se3_nonfinite")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
        reasons.append("se3_last_row")
    rotation = matrix[:3, :3]
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(orthogonality) or orthogonality > 1e-4:
        reasons.append("se3_rotation_orthogonality")
    if not np.isfinite(determinant) or abs(determinant - 1.0) > 1e-4:
        reasons.append("se3_rotation_determinant")
    return not reasons, reasons


def camera_center(world_to_camera: np.ndarray) -> np.ndarray:
    matrix = np.asarray(world_to_camera, dtype=np.float64)
    return -matrix[:3, :3].T @ matrix[:3, 3]


def pose_delta(
    parent_world_to_camera: np.ndarray,
    candidate_world_to_camera: np.ndarray,
    robot_radius_m: float,
) -> tuple[float, float]:
    if robot_radius_m <= 1e-6:
        raise ValueError("robot_radius_m must be greater than 1e-6")
    parent = np.asarray(parent_world_to_camera, dtype=np.float64)
    candidate = np.asarray(candidate_world_to_camera, dtype=np.float64)
    relative_rotation = candidate[:3, :3] @ parent[:3, :3].T
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_degrees = float(np.degrees(np.arccos(cosine)))
    center_ratio = float(
        np.linalg.norm(camera_center(candidate) - camera_center(parent)) / robot_radius_m
    )
    return rotation_degrees, center_ratio


def mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    first_array = np.asarray(first, dtype=bool)
    second_array = np.asarray(second, dtype=bool)
    if first_array.shape != second_array.shape:
        raise ValueError("mask shapes differ")
    union = np.logical_or(first_array, second_array).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_array, second_array).sum() / union)


@dataclass(frozen=True)
class RefinementEvidence:
    """The V/O/L gate record for a successful candidate PnP pose."""

    finite_proper_se3: bool
    n_correspondences: int
    n_inliers: int
    inlier_ratio: float
    positive_depth_ratio: float
    reprojection_mean_px: float
    reprojection_mean_diagonal_ratio: float
    image_hull_area_ratio: float
    world_second_spread_ratio: float
    rotation_delta_deg: float
    camera_center_delta_ratio: float
    mask_iou_parent: float | None
    mask_iou_candidate: float | None
    V: bool
    O: bool
    L: bool
    accepted: bool
    reasons: tuple[str, ...]
    pnp_success: bool = True

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def refinement_evidence(
    parent_world_to_camera: np.ndarray,
    candidate_pose: dict[str, Any],
    process_size: tuple[int, int],
    robot_radius_m: float,
    observed_mask: np.ndarray | None,
    parent_render_mask: np.ndarray | None,
    candidate_render_mask: np.ndarray | None,
    *,
    thresholds: GateV1Thresholds = DEFAULT_GATE_V1_THRESHOLDS,
) -> RefinementEvidence:
    """Compute the fixed gate-v1 V/O/L evidence for a candidate pose."""
    cv2 = _opencv()
    candidate_transform = np.asarray(candidate_pose["world_to_camera"], dtype=np.float64)
    proper, proper_reasons = is_proper_se3(candidate_transform)
    inliers = np.asarray(candidate_pose["inlier_indices"], dtype=np.int64).reshape(-1)
    image_points = np.asarray(candidate_pose["image_points"], dtype=np.float64)
    world_points = np.asarray(candidate_pose["world_points"], dtype=np.float64)
    reprojection_errors = np.asarray(candidate_pose["reprojection_errors"], dtype=np.float64)
    if len(image_points) != len(world_points) or len(image_points) != len(reprojection_errors):
        raise ValueError("candidate correspondence arrays must share a length")
    if np.any(inliers < 0) or np.any(inliers >= len(image_points)):
        raise ValueError("candidate inlier indices are out of bounds")
    inlier_image = image_points[inliers]
    inlier_world = world_points[inliers]
    n_correspondences = len(image_points)
    n_inliers = len(inliers)
    inlier_ratio = n_inliers / n_correspondences if n_correspondences else 0.0
    homogeneous = np.column_stack([inlier_world, np.ones(n_inliers, dtype=np.float64)])
    inlier_camera = (candidate_transform @ homogeneous.T).T[:, :3]
    positive_depth_ratio = float((inlier_camera[:, 2] > 0.0).mean()) if n_inliers else 0.0
    reprojection_mean = float(reprojection_errors[inliers].mean()) if n_inliers else float("inf")
    height, width = process_size
    image_diagonal = float(np.hypot(width, height))
    reprojection_ratio = reprojection_mean / image_diagonal
    if n_inliers >= 3:
        hull = cv2.convexHull(inlier_image.astype(np.float32))
        hull_ratio = float(cv2.contourArea(hull) / (width * height))
    else:
        hull_ratio = 0.0
    if n_inliers >= 2:
        normalized = (inlier_world - inlier_world.mean(axis=0)) / robot_radius_m
        covariance = normalized.T @ normalized / n_inliers
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        second_spread = float(np.sqrt(max(float(eigenvalues[1]), 0.0)))
    else:
        second_spread = 0.0
    rotation_delta, center_delta = pose_delta(
        parent_world_to_camera,
        candidate_transform,
        robot_radius_m,
    )
    parent_iou = mask_iou(observed_mask, parent_render_mask)
    candidate_iou = mask_iou(observed_mask, candidate_render_mask)
    reasons = list(proper_reasons)
    validity_checks = {
        "pnp_inliers_below_12": n_inliers >= thresholds.min_inliers,
        "pnp_inlier_ratio_below_0.10": inlier_ratio >= thresholds.min_inlier_ratio,
        "positive_depth_ratio_below_0.95": (
            positive_depth_ratio >= thresholds.min_positive_depth_ratio
        ),
        "reprojection_ratio_above_0.005": (
            reprojection_ratio <= thresholds.max_reprojection_diagonal_ratio
        ),
    }
    observability_checks = {
        "image_hull_ratio_below_0.005": (
            hull_ratio >= thresholds.min_image_hull_area_ratio
        ),
        "world_second_spread_below_0.02": (
            second_spread >= thresholds.min_world_second_spread_ratio
        ),
    }
    local_checks = {
        "rotation_delta_above_20deg": rotation_delta <= thresholds.max_rotation_delta_deg,
        "camera_center_delta_ratio_above_0.25": (
            center_delta <= thresholds.max_camera_center_delta_ratio
        ),
    }
    if observed_mask is not None:
        local_checks["candidate_mask_render_unavailable"] = candidate_iou is not None
        local_checks["mask_iou_drop_above_0.02"] = (
            parent_iou is not None
            and candidate_iou is not None
            and candidate_iou - parent_iou >= -thresholds.max_mask_iou_drop
        )
    for checks in (validity_checks, observability_checks, local_checks):
        reasons.extend(reason for reason, passed in checks.items() if not passed)
    V = proper and all(validity_checks.values())
    O = all(observability_checks.values())
    L = all(local_checks.values())
    return RefinementEvidence(
        finite_proper_se3=proper,
        n_correspondences=n_correspondences,
        n_inliers=n_inliers,
        inlier_ratio=float(inlier_ratio),
        positive_depth_ratio=positive_depth_ratio,
        reprojection_mean_px=reprojection_mean,
        reprojection_mean_diagonal_ratio=reprojection_ratio,
        image_hull_area_ratio=hull_ratio,
        world_second_spread_ratio=second_spread,
        rotation_delta_deg=rotation_delta,
        camera_center_delta_ratio=center_delta,
        mask_iou_parent=parent_iou,
        mask_iou_candidate=candidate_iou,
        V=V,
        O=O,
        L=L,
        accepted=V and O and L,
        reasons=tuple(reasons),
    )


def is_small_update(
    evidence: RefinementEvidence,
    *,
    thresholds: GateV1Thresholds = DEFAULT_GATE_V1_THRESHOLDS,
) -> bool:
    return (
        evidence.rotation_delta_deg <= thresholds.small_rotation_delta_deg
        and evidence.camera_center_delta_ratio
        <= thresholds.small_camera_center_delta_ratio
    )


def is_two_cycle(
    candidate_world_to_camera: np.ndarray,
    pose_two_steps_before_parent: np.ndarray | None,
    robot_radius_m: float,
    *,
    thresholds: GateV1Thresholds = DEFAULT_GATE_V1_THRESHOLDS,
) -> bool:
    if pose_two_steps_before_parent is None:
        return False
    rotation_delta, center_delta = pose_delta(
        pose_two_steps_before_parent,
        candidate_world_to_camera,
        robot_radius_m,
    )
    return (
        rotation_delta <= thresholds.small_rotation_delta_deg
        and center_delta <= thresholds.small_camera_center_delta_ratio
    )


def select_refinement_candidate(
    parent: Any,
    candidate: Any,
    evidence: RefinementEvidence,
    *,
    two_cycle: bool,
) -> tuple[Any, bool, str]:
    """Select a gate-v1 candidate without silently discarding its evidence."""
    if two_cycle:
        return parent, False, "two_cycle"
    if not evidence.accepted:
        return parent, False, "gate_v1_reject"
    return candidate, True, "gate_v1_accept"


def configure_determinism(seed: int = DETERMINISTIC_SEED) -> dict[str, Any]:
    """Configure the documented deterministic settings when optional deps exist."""
    cv2 = _opencv()
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("gate-v1 determinism requires PyTorch") from error
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "opencv_num_threads": cv2.getNumThreads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": True,
    }


def frozen_spec_record() -> dict[str, str]:
    """Record the fixed threshold family without claiming legacy equivalence."""
    return {
        "gate_id": GATE_ID,
        "evidence_schema_version": "1.0.0",
        "trajectory_protocol_version": "1.0.0",
        "threshold_source": "analytic_preregistration",
    }


__all__ = [
    "CameraModel",
    "DEFAULT_GATE_V1_THRESHOLDS",
    "DETERMINISTIC_SEED",
    "GATE_ID",
    "GateV1Thresholds",
    "RefinementEvidence",
    "canonical_json_bytes",
    "configure_determinism",
    "frozen_spec_record",
    "is_proper_se3",
    "is_small_update",
    "is_two_cycle",
    "mask_iou",
    "pose_delta",
    "pose_sha256",
    "refinement_evidence",
    "resize_camera_matrix_half_pixel",
    "select_refinement_candidate",
]
