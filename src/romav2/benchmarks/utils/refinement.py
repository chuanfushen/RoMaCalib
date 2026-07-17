"""Frozen gate-v1 correctness primitives for pose-aligned refinement."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

GATE_ID = "gate-v1"
GATE_SPEC_SHA256 = "75622abaa45010857a0aa3a6908739188deee1d3fbb37c3bf8027dbdd37fb4e5"
GATE_CONFIG_SHA256 = "d70db4e55b7b1378a6e7b9e0d53220289c9610e3c1a7aeab44fb866e28d578fa"
EVALUATION_SPEC_SHA256 = "d351f3bcdb17be70ac40205e1bfbeef384c96970f8a676893ab6436b5c03a7d34"
DETERMINISTIC_SEED = 20260717
SAM3_EXPECTED_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
ROMAV2_EXPECTED_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
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
    return array_sha256(np.asarray(world_to_camera, dtype=np.float64))


def resize_camera_matrix_half_pixel(
    camera_matrix: np.ndarray,
    raw_size: tuple[int, int],
    process_size: tuple[int, int],
) -> np.ndarray:
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
    """Raw official grid plus the rectified pinhole processing grid."""

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
            raise ValueError(f"Unsupported distortion state: {distortion_state!r}")
        if state == "unknown":
            raise ValueError("Camera distortion state is unknown; gate-v1 requires rectified or distorted")

        K_raw = np.asarray(K_raw, dtype=np.float64)
        if K_raw.shape != (3, 3) or not np.isfinite(K_raw).all():
            raise ValueError("K_raw must be a finite 3x3 matrix")
        if abs(float(K_raw[0, 1])) > 1e-12:
            raise ValueError("gate-v1 requires explicit skew rectification before processing")
        distortion = None
        if distortion_raw is not None:
            distortion = np.asarray(distortion_raw, dtype=np.float64).reshape(-1)
            if not np.isfinite(distortion).all():
                raise ValueError("Raw distortion coefficients must be finite")
        if state == "rectified":
            if distortion is not None and np.any(np.abs(distortion) > 1e-12):
                raise ValueError("Rectified camera cannot carry non-zero distortion coefficients")
            distortion = None
        elif distortion is None:
            raise ValueError("Distorted camera requires raw distortion coefficients")

        K_process = resize_camera_matrix_half_pixel(K_raw, raw_size, process_size)
        mapping_payload = {
            "raw_size": raw_size,
            "process_size": process_size,
            "K_raw": K_raw,
            "distortion_raw": distortion,
            "K_process": K_process,
            "rgb_resampling": "bilinear",
            "mask_resampling": "nearest",
            "crop_policy": "forbidden",
            "pixel_center": "opencv_integer_center",
        }
        evaluation_payload = {
            "raw_size": raw_size,
            "K_raw": K_raw,
            "distortion_raw": distortion,
            "distortion_state": state,
        }
        raw_to_process_fingerprint = sha256_bytes(canonical_json_bytes(mapping_payload))
        evaluation_projection_fingerprint = sha256_bytes(canonical_json_bytes(evaluation_payload))
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
            K_raw=K_raw,
            distortion_raw=distortion,
            K_process=K_process,
            distortion_state=state,
            raw_to_process_fingerprint=raw_to_process_fingerprint,
            evaluation_projection_fingerprint=evaluation_projection_fingerprint,
            fingerprint=fingerprint,
        )

    def to_record(self) -> dict:
        return {
            "raw_size": list(self.raw_size),
            "process_size": list(self.process_size),
            "K_raw": self.K_raw.tolist(),
            "distortion_raw": None if self.distortion_raw is None else self.distortion_raw.tolist(),
            "K_process": self.K_process.tolist(),
            "distortion_state": self.distortion_state,
            "raw_to_process_fingerprint": self.raw_to_process_fingerprint,
            "evaluation_projection_fingerprint": self.evaluation_projection_fingerprint,
            "fingerprint": self.fingerprint,
        }

    def _remap(self, image: np.ndarray, interpolation: int) -> np.ndarray:
        process_height, process_width = self.process_size
        if self.distortion_raw is None:
            if image.shape[:2] == self.process_size:
                return image.copy()
            return cv2.resize(image, (process_width, process_height), interpolation=interpolation)
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.distortion_raw,
            None,
            self.K_process,
            (process_width, process_height),
            cv2.CV_32FC1,
        )
        return cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT)

    def process_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != self.raw_size:
            raise ValueError(f"Raw RGB shape {image.shape[:2]} does not match camera raw_size {self.raw_size}")
        return self._remap(image, cv2.INTER_LINEAR)

    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.shape[:2] != self.raw_size:
            raise ValueError(f"Raw mask shape {mask.shape[:2]} does not match camera raw_size {self.raw_size}")
        return self._remap(mask.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)

    def raw_to_process_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        converted = cv2.undistortPoints(
            points,
            self.K_raw,
            self.distortion_raw,
            P=self.K_process,
        )
        return converted.reshape(-1, 2)

    def process_to_raw_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        normalized = np.column_stack(
            [
                (points[:, 0] - self.K_process[0, 2]) / self.K_process[0, 0],
                (points[:, 1] - self.K_process[1, 2]) / self.K_process[1, 1],
                np.ones(len(points), dtype=np.float64),
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

    def project_raw(self, points_robot: np.ndarray, world_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_robot = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
        transform = np.asarray(world_to_camera, dtype=np.float64)
        points_camera = (
            transform[:3, :3] @ points_robot.T + transform[:3, 3:4]
        ).T
        rvec, _ = cv2.Rodrigues(transform[:3, :3])
        projected, _ = cv2.projectPoints(
            points_robot,
            rvec,
            transform[:3, 3],
            self.K_raw,
            self.distortion_raw,
        )
        return projected.reshape(-1, 2), points_camera


@dataclass(frozen=True)
class GeometryProfile:
    model_sha256: str
    visual_geom_group: int
    visual_body_names: tuple[str, ...]
    q_fingerprint: str
    posed_bounds: np.ndarray
    robot_radius_m: float
    fingerprint: str

    def to_record(self) -> dict:
        return {
            "model_sha256": self.model_sha256,
            "visual_geom_group": self.visual_geom_group,
            "visual_body_names": list(self.visual_body_names),
            "q_fingerprint": self.q_fingerprint,
            "posed_bounds": self.posed_bounds.tolist(),
            "robot_radius_m": self.robot_radius_m,
            "fingerprint": self.fingerprint,
        }


def make_geometry_profile(
    model,
    data,
    model_path: Path,
    visual_geom_group: int,
    visual_body_names: list[str] | tuple[str, ...] | None,
) -> GeometryProfile:
    import mujoco

    from .geometry import visual_bounds

    minimum, maximum = visual_bounds(model, data, visual_geom_group)
    posed_bounds = np.stack([minimum, maximum]).astype(np.float64)
    robot_radius = 0.5 * float(np.linalg.norm(maximum - minimum))
    if not np.isfinite(posed_bounds).all() or not np.isfinite(robot_radius) or robot_radius <= 1e-6:
        raise ValueError("GeometryProfile requires finite visual bounds and robot radius > 1e-6 m")

    q_values = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        start = int(model.jnt_qposadr[joint_id])
        end = int(model.jnt_qposadr[joint_id + 1]) if joint_id + 1 < model.njnt else int(model.nq)
        q_values.append({"name": name, "qpos": np.asarray(data.qpos[start:end], dtype=np.float64).tolist()})
    q_fingerprint = sha256_bytes(canonical_json_bytes(q_values))
    model_hash = sha256_file(Path(model_path))
    names = tuple(str(name) for name in (visual_body_names or ()))
    payload = {
        "model_sha256": model_hash,
        "visual_geom_group": int(visual_geom_group),
        "visual_body_names": names,
        "q_fingerprint": q_fingerprint,
        "posed_bounds": posed_bounds,
        "robot_radius_m": robot_radius,
    }
    return GeometryProfile(
        model_sha256=model_hash,
        visual_geom_group=int(visual_geom_group),
        visual_body_names=names,
        q_fingerprint=q_fingerprint,
        posed_bounds=posed_bounds,
        robot_radius_m=robot_radius,
        fingerprint=sha256_bytes(canonical_json_bytes(payload)),
    )


def is_proper_se3(transform: np.ndarray) -> tuple[bool, list[str]]:
    reasons = []
    transform = np.asarray(transform)
    if transform.shape != (4, 4):
        return False, ["se3_shape"]
    if not np.isfinite(transform).all():
        reasons.append("se3_nonfinite")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
        reasons.append("se3_last_row")
    rotation = transform[:3, :3]
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(orthogonality) or orthogonality > 1e-4:
        reasons.append("se3_rotation_orthogonality")
    if not np.isfinite(determinant) or abs(determinant - 1.0) > 1e-4:
        reasons.append("se3_rotation_determinant")
    return not reasons, reasons


def camera_center(world_to_camera: np.ndarray) -> np.ndarray:
    transform = np.asarray(world_to_camera, dtype=np.float64)
    return -transform[:3, :3].T @ transform[:3, 3]


def pose_delta(
    parent_world_to_camera: np.ndarray,
    candidate_world_to_camera: np.ndarray,
    robot_radius_m: float,
) -> tuple[float, float]:
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
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"Mask shapes differ: {first.shape} vs {second.shape}")
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


@dataclass
class RefinementEvidence:
    pnp_success: bool
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
    reasons: list[str]

    def to_record(self) -> dict:
        return asdict(self)


def refinement_evidence(
    parent_world_to_camera: np.ndarray,
    candidate_pose: dict,
    process_size: tuple[int, int],
    robot_radius_m: float,
    observed_mask: np.ndarray | None,
    parent_render_mask: np.ndarray | None,
    candidate_render_mask: np.ndarray | None,
) -> RefinementEvidence:
    candidate_transform = np.asarray(candidate_pose["world_to_camera"], dtype=np.float64)
    proper, proper_reasons = is_proper_se3(candidate_transform)
    inliers = np.asarray(candidate_pose["inlier_indices"], dtype=np.int64)
    image_points = np.asarray(candidate_pose["image_points"], dtype=np.float64)
    world_points = np.asarray(candidate_pose["world_points"], dtype=np.float64)
    reprojection_errors = np.asarray(candidate_pose["reprojection_errors"], dtype=np.float64)
    inlier_image = image_points[inliers]
    inlier_world = world_points[inliers]
    n_correspondences = int(len(image_points))
    n_inliers = int(len(inliers))
    inlier_ratio = float(n_inliers / n_correspondences) if n_correspondences else 0.0

    homogeneous = np.column_stack([inlier_world, np.ones(n_inliers, dtype=np.float64)])
    inlier_camera = (candidate_transform @ homogeneous.T).T[:, :3]
    positive_depth_ratio = float((inlier_camera[:, 2] > 0.0).mean()) if n_inliers else 0.0
    reprojection_mean = (
        float(reprojection_errors[inliers].mean()) if n_inliers else float("inf")
    )
    process_height, process_width = process_size
    image_diagonal = float(np.hypot(process_width, process_height))
    reprojection_ratio = reprojection_mean / image_diagonal

    if n_inliers >= 3:
        hull = cv2.convexHull(inlier_image.astype(np.float32))
        hull_ratio = float(cv2.contourArea(hull) / (process_width * process_height))
    else:
        hull_ratio = 0.0
    if n_inliers >= 2:
        normalized = (inlier_world - inlier_world.mean(axis=0)) / robot_radius_m
        covariance = normalized.T @ normalized / n_inliers
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        second_spread = float(np.sqrt(max(float(eigenvalues[1]), 0.0)))
    else:
        second_spread = 0.0

    rotation_delta, center_delta_ratio = pose_delta(
        parent_world_to_camera,
        candidate_transform,
        robot_radius_m,
    )
    parent_iou = mask_iou(observed_mask, parent_render_mask)
    candidate_iou = mask_iou(observed_mask, candidate_render_mask)

    reasons = list(proper_reasons)
    validity_checks = {
        "pnp_inliers_below_12": n_inliers >= 12,
        "pnp_inlier_ratio_below_0.10": inlier_ratio >= 0.10,
        "positive_depth_ratio_below_0.95": positive_depth_ratio >= 0.95,
        "reprojection_ratio_above_0.005": reprojection_ratio <= 0.005,
    }
    for reason, passed in validity_checks.items():
        if not passed:
            reasons.append(reason)
    V = proper and all(validity_checks.values())

    observability_checks = {
        "image_hull_ratio_below_0.005": hull_ratio >= 0.005,
        "world_second_spread_below_0.02": second_spread >= 0.02,
    }
    for reason, passed in observability_checks.items():
        if not passed:
            reasons.append(reason)
    O = all(observability_checks.values())

    local_checks = {
        "rotation_delta_above_20deg": rotation_delta <= 20.0,
        "camera_center_delta_ratio_above_0.25": center_delta_ratio <= 0.25,
    }
    if observed_mask is not None:
        local_checks["candidate_mask_render_unavailable"] = candidate_iou is not None
        local_checks["mask_iou_drop_above_0.02"] = (
            parent_iou is not None
            and candidate_iou is not None
            and candidate_iou - parent_iou >= -0.02
        )
    for reason, passed in local_checks.items():
        if not passed:
            reasons.append(reason)
    L = all(local_checks.values())
    accepted = V and O and L
    return RefinementEvidence(
        pnp_success=True,
        finite_proper_se3=proper,
        n_correspondences=n_correspondences,
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio,
        positive_depth_ratio=positive_depth_ratio,
        reprojection_mean_px=reprojection_mean,
        reprojection_mean_diagonal_ratio=reprojection_ratio,
        image_hull_area_ratio=hull_ratio,
        world_second_spread_ratio=second_spread,
        rotation_delta_deg=rotation_delta,
        camera_center_delta_ratio=center_delta_ratio,
        mask_iou_parent=parent_iou,
        mask_iou_candidate=candidate_iou,
        V=V,
        O=O,
        L=L,
        accepted=accepted,
        reasons=reasons,
    )


def is_small_update(evidence: RefinementEvidence) -> bool:
    return (
        evidence.rotation_delta_deg <= 0.10
        and evidence.camera_center_delta_ratio <= 0.001
    )


def is_two_cycle(
    candidate_world_to_camera: np.ndarray,
    pose_two_steps_before_parent: np.ndarray | None,
    robot_radius_m: float,
) -> bool:
    if pose_two_steps_before_parent is None:
        return False
    rotation_delta, center_delta = pose_delta(
        pose_two_steps_before_parent,
        candidate_world_to_camera,
        robot_radius_m,
    )
    return rotation_delta <= 0.10 and center_delta <= 0.001


def select_refinement_candidate(
    parent: Any,
    candidate: Any,
    evidence: RefinementEvidence,
    *,
    two_cycle: bool,
) -> tuple[Any, bool, str]:
    if two_cycle:
        return parent, False, "two_cycle"
    if not evidence.accepted:
        return parent, False, "gate_v1_reject"
    return candidate, True, "gate_v1_accept"


def configure_determinism(seed: int = DETERMINISTIC_SEED) -> dict:
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
        "python_random_seed": True,
        "numpy_seed": True,
        "torch_cpu_and_cuda_seed": True,
        "opencv_rng_seed": True,
        "opencv_num_threads": cv2.getNumThreads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": True,
    }


def frozen_spec_record() -> dict:
    return {
        "gate_id": GATE_ID,
        "gate_config_sha256": GATE_CONFIG_SHA256,
        "gate_spec_sha256": GATE_SPEC_SHA256,
        "evaluation_spec_sha256": EVALUATION_SPEC_SHA256,
        "evidence_schema_version": "1.0.0",
        "trajectory_protocol_version": "1.0.0",
    }


def verify_frozen_specs(project_root: Path) -> dict:
    expected = {
        "gate_config": (
            Path("config/refinement/gate-v1-design.toml"),
            GATE_CONFIG_SHA256,
        ),
        "gate_spec": (
            Path("config/refinement/gate-v1-spec.json"),
            GATE_SPEC_SHA256,
        ),
        "evaluation_spec": (
            Path("config/refinement/evaluation-v1-spec.json"),
            EVALUATION_SPEC_SHA256,
        ),
    }
    verified = {}
    for name, (relative_path, expected_hash) in expected.items():
        path = project_root / relative_path
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen {name} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        verified[name] = {"path": str(relative_path), "sha256": actual_hash}
    freeze_path = project_root / "config/refinement/gate-v1-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if (
        freeze.get("gate_id") != GATE_ID
        or freeze.get("config_sha256") != GATE_CONFIG_SHA256
        or freeze.get("method_spec_sha256") != GATE_SPEC_SHA256
        or freeze.get("evaluation_spec_sha256") != EVALUATION_SPEC_SHA256
    ):
        raise RuntimeError("gate-v1 freeze index does not match compiled expected hashes")
    verified["freeze_index"] = {
        "path": str(freeze_path.relative_to(project_root)),
        "sha256": sha256_file(freeze_path),
    }
    return verified
