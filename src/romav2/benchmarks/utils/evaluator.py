#!/usr/bin/env python3
"""Batch evaluation for DREAM-style Panda frames with MuJoCo render matching."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from .geometry import (  # noqa: E402
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ImcuiMatcher,
)
from .pose import (  # noqa: E402
    DEFAULT_MJCF,
    best_view_key,
    apply_input_mask,
    draw_keypoint_eval,
    dream_keypoints,
    fk_keypoints,
    keypoint_metrics,
    load_camera_matrix,
    load_camera_model,
    load_dream_payload,
    make_sam3_extractor,
    make_model_and_data,
    match_one_render,
    render_orbit_views,
    render_pose_aligned_artifacts,
    render_pose_mask,
    save_pose_npz,
)
from .refinement import (  # noqa: E402
    CameraModel,
    DETERMINISTIC_SEED,
    ROMAV2_EXPECTED_SHA256,
    SAM3_EXPECTED_SHA256,
    canonical_json_bytes,
    configure_determinism,
    frozen_spec_record,
    is_small_update,
    is_two_cycle,
    make_geometry_profile,
    mask_iou,
    pose_sha256,
    refinement_evidence,
    select_refinement_candidate,
    sha256_bytes,
    sha256_file,
    verify_frozen_specs,
)

DEFAULT_DATASET_DIR = Path(
    "/LargeModelDev/users/chuanfu.shen/workspace/paper/calib/dream-data/real/"
    "panda-3cam_azure/panda-3cam_azure"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/dream_mujoco_match_eval_batch"


def count_dream_frames(dataset_dir: Path) -> int:
    count = 0
    for json_path in dataset_dir.glob("*.json"):
        if json_path.name == "_camera_settings.json" or not json_path.stem.isdigit():
            continue
        if json_path.with_suffix(".rgb.jpg").is_file():
            count += 1
    return count


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    """Resolve nested DREAM exports to the directory containing frame files."""
    if count_dream_frames(dataset_dir) > 0:
        return dataset_dir

    candidates: dict[Path, int] = {}
    for camera_settings in dataset_dir.rglob("_camera_settings.json"):
        parent = camera_settings.parent
        count = count_dream_frames(parent)
        if count > 0:
            candidates[parent] = count

    if not candidates:
        frame_dirs = {path.parent for path in dataset_dir.rglob("*.json") if path.stem.isdigit()}
        for frame_dir in frame_dirs:
            count = count_dream_frames(frame_dir)
            if count > 0:
                candidates[frame_dir] = count

    if not candidates:
        return dataset_dir

    return sorted(candidates.items(), key=lambda item: (-item[1], len(item[0].parts), str(item[0])))[0][0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--visual-geom-group", type=int, default=2, help="MuJoCo geom group containing renderable visual meshes.")
    parser.add_argument(
        "--source-visual-geom-group",
        type=int,
        default=None,
        help="Optional imported visual group remapped in memory to --visual-geom-group.",
    )
    parser.add_argument(
        "--visual-body-names",
        nargs="*",
        default=None,
        help="Optional exact body-name allowlist for rendering, bounds, and mesh picking.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-source", type=Path, default=None)
    parser.add_argument("--prerender-dir", type=Path, default=None, help="Use pre-rendered views from prerender_dream_mujoco_views.py.")
    parser.add_argument("--views", "-x", type=int, default=6)
    parser.add_argument("--match-batch-size", type=int, default=None, help="RoMaV2 pairs per forward. Defaults to --views.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
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
    parser.add_argument("--sam3-checkpoint", type=Path, default=None)
    parser.add_argument("--camera-settings", type=Path, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--fallback-focal", type=float, default=400.0)
    parser.add_argument(
        "--distortion-state",
        choices=("rectified", "distorted", "unknown"),
        default="unknown",
        help="Raw camera distortion contract required by the refinement protocol.",
    )
    parser.add_argument("--distortion-coefficients", type=float, nargs="*", default=None)
    parser.add_argument("--frame-indices", type=int, nargs="*", default=None)
    parser.add_argument("--sample-count", type=int, default=None, help="Randomly sample this many frames after range/stride filtering.")
    parser.add_argument("--sample-seed", type=int, default=90)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-renders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-all-matches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auc-threshold", type=float, default=0.1, help="ADD AUC upper threshold in meters.")
    parser.add_argument("--auc-delta", type=float, default=1e-4, help="ADD AUC integration step in meters.")
    parser.add_argument(
        "--refinement-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the frozen gate-v1 protocol; iterations=0 is the S0 parity control.",
    )
    parser.add_argument("--refine-iterations", type=int, default=0, choices=range(4))
    parser.add_argument(
        "--save-refinement-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def frame_index_from_json(path: Path) -> int:
    return int(path.stem)


def discover_frames(args: argparse.Namespace) -> list[tuple[int, Path, Path]]:
    json_paths = sorted(
        path for path in args.dataset_dir.glob("*.json") if path.name != "_camera_settings.json" and path.stem.isdigit()
    )
    frames = []
    requested = None if args.frame_indices is None else {int(index) for index in args.frame_indices}
    for json_path in json_paths:
        index = frame_index_from_json(json_path)
        if requested is not None and index not in requested:
            continue
        if args.start_index is not None and index < args.start_index:
            continue
        if args.end_index is not None and index > args.end_index:
            continue
        if args.stride > 1 and index % args.stride != 0:
            continue
        image_path = json_path.with_suffix(".rgb.jpg")
        if not image_path.is_file():
            continue
        frames.append((index, json_path, image_path))
    if args.sample_count is not None:
        if args.sample_count < 1:
            raise ValueError("--sample-count must be >= 1")
        if args.sample_count > len(frames):
            raise ValueError(f"--sample-count={args.sample_count} exceeds available frames {len(frames)}")
        rng = np.random.default_rng(args.sample_seed)
        sampled = rng.choice(len(frames), size=args.sample_count, replace=False)
        frames = [frames[int(index)] for index in sorted(sampled)]
    if args.limit is not None:
        frames = frames[: args.limit]
    return frames


def _namespace_record(args: argparse.Namespace) -> dict:
    ignored = {"run_provenance", "run_fingerprint", "frame_manifest_sha256"}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in ignored
    }


def _git_record() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "status": None, "error": repr(error)}
    return {"commit": commit, "status": status, "dirty": bool(status)}


def _runtime_provenance(args: argparse.Namespace, frames: list[tuple[int, Path, Path]]) -> dict:
    import mujoco

    visible_gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            visible_gpus.append(
                {
                    "logical_index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        nvidia_smi = {"error": repr(error)}
    try:
        uv_version = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        uv_version = repr(error)

    manifest = [
        {
            "frame_index": index,
            "json": str(json_path),
            "image": str(image_path),
            "json_size": json_path.stat().st_size,
            "image_size": image_path.stat().st_size,
            "json_sha256": sha256_file(json_path),
            "image_sha256": sha256_file(image_path),
        }
        for index, json_path, image_path in frames
    ]
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
    config_hash = sha256_bytes(canonical_json_bytes(_namespace_record(args)))
    record = {
        "git": _git_record(),
        "started_at": datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "launch_command": [sys.executable, *sys.argv],
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "resolved_runtime_config_sha256": config_hash,
        "frame_manifest_sha256": manifest_hash,
        "frame_manifest": manifest,
        "mujoco_model": {
            "path": str(args.mujoco_xml),
            "sha256": sha256_file(args.mujoco_xml),
        },
        "config_source": (
            None
            if args.config_source is None
            else {
                "path": str(args.config_source),
                "sha256": sha256_file(args.config_source),
            }
        ),
        "frozen_spec": frozen_spec_record(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mujoco": mujoco.__version__,
            "uv": uv_version,
        },
        "visible_gpus": visible_gpus,
        "nvidia_smi": nvidia_smi,
    }
    return record


def _verified_weight_record(path: Path, expected_sha256: str, name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{name} weight was not found at {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return {"path": str(path), "sha256": actual}


def _weight_provenance(args: argparse.Namespace) -> dict:
    roma_path = Path(torch.hub.get_dir()) / "checkpoints" / "romav2.0.1.pt"
    weights = {
        "romav2": _verified_weight_record(
            roma_path,
            ROMAV2_EXPECTED_SHA256,
            "RoMaV2 2.0.1",
        )
    }
    if args.mask_input:
        if args.sam3_checkpoint is None:
            raise ValueError("gate-v1 with input masking requires --sam3-checkpoint")
        weights["sam3"] = _verified_weight_record(
            args.sam3_checkpoint,
            SAM3_EXPECTED_SHA256,
            "SAM3",
        )
    else:
        weights["sam3"] = {"used": False}
    return weights


def clean_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def remove_render_artifacts(render_dir: Path) -> None:
    if not render_dir.exists():
        return
    for path in render_dir.glob("view_*"):
        if path.is_file():
            path.unlink()


def remove_frame_artifacts(frame_dir: Path) -> None:
    for filename in ("input_mask.png", "best_keypoint_eval.jpg", "best_pose.npz", "frame_summary.json"):
        path = frame_dir / filename
        if path.is_file():
            path.unlink()
    match_dir = frame_dir / "matches"
    if match_dir.exists():
        for path in match_dir.glob("view_*_matches.*"):
            if path.is_file():
                path.unlink()


@torch.inference_mode()
def match_romav2_images_batched(
    matcher_api: ImcuiMatcher,
    observed_rgb: np.ndarray,
    render_paths: list[Path],
    match_batch_size: int,
) -> list[tuple[dict, np.ndarray]]:
    render_rgbs = [np.asarray(Image.open(path).convert("RGB")) for path in render_paths]
    observed_tensor = torch.from_numpy(observed_rgb.copy()).permute(2, 0, 1)
    device = matcher_api.device
    net = matcher_api.matcher
    outputs = []

    for start in range(0, len(render_rgbs), match_batch_size):
        chunk_rgbs = render_rgbs[start : start + match_batch_size]
        render_tensors = []
        for render_rgb in chunk_rgbs:
            render_tensors.append(torch.from_numpy(render_rgb.copy()).permute(2, 0, 1))

        image0 = observed_tensor.to(device)[None].repeat(len(render_tensors), 1, 1, 1)
        image1 = torch.stack(render_tensors, dim=0).to(device)
        torch.set_float32_matmul_precision("highest")
        preds = net.match(image0, image1)
        h0, w0 = image0.shape[-2:]
        h1, w1 = image1.shape[-2:]

        for batch_index, render_rgb in enumerate(chunk_rgbs):
            sliced = {key: value[batch_index : batch_index + 1] if isinstance(value, torch.Tensor) else value for key, value in preds.items()}
            matches, confidence, _, _ = net.sample(sliced, matcher_api.max_keypoints)
            mkpts0, mkpts1 = net.to_pixel_coordinates(matches, h0, w0, h1, w1)
            outputs.append(
                (
                    {
                        "mkeypoints0_orig": mkpts0.detach().cpu().numpy(),
                        "mkeypoints1_orig": mkpts1.detach().cpu().numpy(),
                        "mconf": confidence.detach().cpu().numpy(),
                    },
                    render_rgb,
                )
            )
    return outputs


def match_render_paths(
    observed_rgb: np.ndarray,
    render_paths: list[Path],
    matcher_api: ImcuiMatcher,
    model,
    data,
    camera_matrix: np.ndarray,
    args: argparse.Namespace,
    match_dir: Path,
) -> list[dict]:
    if args.matcher == "RoMaV2":
        match_batch_size = args.match_batch_size or args.views
        predictions = match_romav2_images_batched(matcher_api, observed_rgb, render_paths, match_batch_size)
        return [
            match_one_render(
                observed_rgb,
                render_path,
                matcher_api,
                model,
                data,
                camera_matrix,
                args,
                match_dir,
                prediction=prediction,
                render_rgb=render_rgb,
            )
            for render_path, (prediction, render_rgb) in zip(render_paths, predictions)
        ]
    return [
        match_one_render(observed_rgb, render_path, matcher_api, model, data, camera_matrix, args, match_dir)
        for render_path in render_paths
    ]


def _mask_fingerprint(mask: np.ndarray | None) -> str | None:
    if mask is None:
        return None
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    return sha256_bytes(canonical_json_bytes({"shape": mask.shape}) + mask.tobytes())


def _save_iteration_pose(
    iteration_dir: Path,
    selected: dict,
    camera_matrix: np.ndarray,
    metrics: dict,
    *,
    filename: str = "pose.npz",
) -> Path:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    pose_path = save_pose_npz(
        iteration_dir,
        selected,
        camera_matrix,
        metrics,
        filename=filename,
    )
    pose = selected["_pose"]
    pose_record = {
        "world_to_camera": pose["world_to_camera"].tolist(),
        "camera_to_robot_base": pose["camera_to_world"].tolist(),
        "pose_sha256": pose_sha256(pose["world_to_camera"]),
        "npz": str(pose_path),
    }
    (iteration_dir / f"{Path(filename).stem}.json").write_text(
        json.dumps(pose_record, indent=2) + "\n"
    )
    return pose_path


def _write_iteration_summary(iteration_dir: Path, record: dict, enabled: bool) -> None:
    if not enabled:
        return
    iteration_dir.mkdir(parents=True, exist_ok=True)
    (iteration_dir / "frame_summary.json").write_text(json.dumps(record, indent=2) + "\n")


def _copy_stage1_matches(best: dict, match_dir: Path, iteration_dir: Path) -> dict:
    artifacts = {}
    stem = Path(best["render_path"]).stem
    for suffix, target_name in (("_matches.jpg", "matches.jpg"), ("_matches.npz", "matches.npz")):
        source = match_dir / f"{stem}{suffix}"
        if source.is_file():
            target = iteration_dir / target_name
            shutil.copy2(source, target)
            artifacts[target_name] = str(target)
    return artifacts


def _filled_iteration_record(
    iteration_index: int,
    reason: str,
    selected: dict,
    selected_metrics: dict,
    selected_source: int,
    camera_matrix: np.ndarray,
    frame_dir: Path,
    save_artifacts: bool,
) -> dict:
    iteration_dir = frame_dir / f"iteration_{iteration_index:02d}"
    artifacts = {}
    if save_artifacts:
        pose_path = _save_iteration_pose(
            iteration_dir,
            selected,
            camera_matrix,
            selected_metrics,
        )
        artifacts["pose_npz"] = str(pose_path)
        artifacts["pose_json"] = str(iteration_dir / "pose.json")
    record = {
        "iteration": iteration_index,
        "stage": "pose_aligned_refine",
        "status": "filled",
        "attempted": False,
        "accepted": False,
        "reason": reason,
        "selected_pose_source_iteration": selected_source,
        "selected_pose_sha256": pose_sha256(selected["_pose"]["world_to_camera"]),
        "selected_pnp": selected["pnp"],
        "selected_metrics": selected_metrics,
        "candidate": None,
        "evidence": None,
        "artifacts": artifacts,
    }
    _write_iteration_summary(iteration_dir, record, save_artifacts)
    return record


def _validate_refinement_resume(existing: dict, expected_fingerprint: str) -> None:
    protocol = existing.get("refinement_protocol") or {}
    if protocol.get("resume_fingerprint") != expected_fingerprint:
        raise RuntimeError("refinement fingerprint mismatch")
    if existing.get("status") != "success":
        return
    required = {
        "T0_sha256": protocol.get("T0_sha256"),
        "camera_model_fingerprint": (protocol.get("camera_model") or {}).get(
            "fingerprint"
        ),
        "geometry_profile_fingerprint": (
            protocol.get("geometry_profile") or {}
        ).get("fingerprint"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"resume evidence is missing required fingerprints: {missing}")
    iterations = protocol.get("iterations") or []
    if not iterations or iterations[0].get("selected_pose_sha256") != required["T0_sha256"]:
        raise RuntimeError("resume T0 fingerprint does not match iteration_00")
    pose_json_path = (iterations[0].get("artifacts") or {}).get("pose_json")
    if pose_json_path is None or not Path(pose_json_path).is_file():
        raise RuntimeError("resume iteration_00 pose.json is missing")
    pose_payload = json.loads(Path(pose_json_path).read_text())
    if pose_payload.get("pose_sha256") != required["T0_sha256"]:
        raise RuntimeError("resume iteration_00 pose artifact hash mismatch")

    expected_mask_hash = protocol.get("observed_mask_sha256")
    mask_path = existing.get("input_mask", {}).get("mask_path")
    if expected_mask_hash is not None:
        if mask_path is None or not Path(mask_path).is_file():
            raise RuntimeError("resume processing-grid input mask is missing")
        stored_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if _mask_fingerprint(stored_mask) != expected_mask_hash:
            raise RuntimeError("resume input mask artifact hash mismatch")


def run_refinement_trajectory(
    args: argparse.Namespace,
    matcher_api: ImcuiMatcher,
    observed_for_match: np.ndarray,
    observed_mask: np.ndarray | None,
    model,
    data,
    camera_model: CameraModel,
    geometry_profile,
    payload: dict,
    fk_points: dict[str, np.ndarray],
    stage1_best: dict,
    stage1_metrics: dict,
    frame_dir: Path,
    match_dir: Path,
) -> tuple[dict, dict, list[dict]]:
    camera_matrix = camera_model.K_process
    save_artifacts = bool(args.save_refinement_artifacts)
    iterations = []
    stage0_dir = frame_dir / "iteration_00"
    stage0_artifacts = {}
    stage0_render_mask = None
    stage0_render_error = None
    if save_artifacts:
        try:
            render_path, mask_path, camera_path, stage0_render_mask = render_pose_aligned_artifacts(
                model,
                data,
                args,
                camera_matrix,
                stage1_best["_pose"]["world_to_camera"],
                stage0_dir,
            )
            stage0_artifacts.update(
                {
                    "render": str(render_path),
                    "render_mask": str(mask_path),
                    "camera_npz": str(camera_path),
                }
            )
        except Exception as error:
            stage0_render_error = repr(error)
        pose_path = _save_iteration_pose(
            stage0_dir,
            stage1_best,
            camera_matrix,
            stage1_metrics,
        )
        stage0_artifacts.update(
            {
                "pose_npz": str(pose_path),
                "pose_json": str(stage0_dir / "pose.json"),
                **_copy_stage1_matches(stage1_best, match_dir, stage0_dir),
            }
        )
    stage0_record = {
        "iteration": 0,
        "stage": "stage1",
        "status": "success",
        "attempted": True,
        "accepted": True,
        "reason": "stage1_selected_best_view",
        "selected_pose_source_iteration": 0,
        "selected_pose_sha256": pose_sha256(stage1_best["_pose"]["world_to_camera"]),
        "selected_pnp": stage1_best["pnp"],
        "selected_metrics": stage1_metrics,
        "candidate": clean_record(stage1_best),
        "evidence": None,
        "observed_mask_iou": mask_iou(observed_mask, stage0_render_mask),
        "artifact_error": stage0_render_error,
        "artifacts": stage0_artifacts,
    }
    iterations.append(stage0_record)
    _write_iteration_summary(stage0_dir, stage0_record, save_artifacts)

    parent = stage1_best
    parent_metrics = stage1_metrics
    parent_source = 0
    accepted_history = [np.asarray(parent["_pose"]["world_to_camera"], dtype=np.float64)]
    stop_reason = "iteration_budget_exhausted"
    accepted_updates = 0
    attempted_updates = 0

    for iteration_index in range(1, args.refine_iterations + 1):
        attempted_updates += 1
        iteration_dir = frame_dir / f"iteration_{iteration_index:02d}"
        artifacts = {}
        try:
            render_path, render_mask_path, camera_path, parent_render_mask = (
                render_pose_aligned_artifacts(
                    model,
                    data,
                    args,
                    camera_matrix,
                    parent["_pose"]["world_to_camera"],
                    iteration_dir,
                )
            )
            artifacts.update(
                {
                    "render": str(render_path),
                    "render_mask": str(render_mask_path),
                    "camera_npz": str(camera_path),
                }
            )
        except Exception as error:
            stop_reason = "aligned_render_failure"
            record = {
                "iteration": iteration_index,
                "stage": "pose_aligned_refine",
                "status": "rejected",
                "attempted": True,
                "accepted": False,
                "reason": stop_reason,
                "error": repr(error),
                "selected_pose_source_iteration": parent_source,
                "selected_pose_sha256": pose_sha256(parent["_pose"]["world_to_camera"]),
                "selected_pnp": parent["pnp"],
                "selected_metrics": parent_metrics,
                "candidate": None,
                "evidence": None,
                "artifacts": artifacts,
            }
            if save_artifacts:
                pose_path = _save_iteration_pose(
                    iteration_dir,
                    parent,
                    camera_matrix,
                    parent_metrics,
                )
                record["artifacts"]["pose_npz"] = str(pose_path)
                record["artifacts"]["pose_json"] = str(iteration_dir / "pose.json")
            iterations.append(record)
            _write_iteration_summary(iteration_dir, record, save_artifacts)
            break

        prediction, render_rgb = match_romav2_images_batched(
            matcher_api,
            observed_for_match,
            [render_path],
            1,
        )[0]
        candidate = match_one_render(
            observed_for_match,
            render_path,
            matcher_api,
            model,
            data,
            camera_matrix,
            args,
            iteration_dir,
            prediction=prediction,
            render_rgb=render_rgb,
            artifact_stem="matches",
        )
        artifacts.update(
            {
                "matches_jpg": str(iteration_dir / "matches.jpg"),
                "matches_npz": str(iteration_dir / "matches.npz"),
            }
        )
        if candidate["pnp"]["status"] != "success":
            stop_reason = f"pnp_{candidate['pnp']['status']}"
            record = {
                "iteration": iteration_index,
                "stage": "pose_aligned_refine",
                "status": "rejected",
                "attempted": True,
                "accepted": False,
                "reason": stop_reason,
                "selected_pose_source_iteration": parent_source,
                "selected_pose_sha256": pose_sha256(parent["_pose"]["world_to_camera"]),
                "selected_pnp": parent["pnp"],
                "selected_metrics": parent_metrics,
                "candidate": clean_record(candidate),
                "evidence": None,
                "artifacts": artifacts,
            }
            if save_artifacts:
                pose_path = _save_iteration_pose(
                    iteration_dir,
                    parent,
                    camera_matrix,
                    parent_metrics,
                )
                record["artifacts"]["pose_npz"] = str(pose_path)
                record["artifacts"]["pose_json"] = str(iteration_dir / "pose.json")
            iterations.append(record)
            _write_iteration_summary(iteration_dir, record, save_artifacts)
            break

        candidate_metrics = keypoint_metrics(
            payload,
            fk_points,
            candidate["_pose"],
            camera_matrix,
            camera_model,
        )
        candidate_render_mask = None
        candidate_mask_error = None
        try:
            candidate_mask_path, candidate_camera_path, candidate_render_mask = render_pose_mask(
                model,
                data,
                args,
                camera_matrix,
                candidate["_pose"]["world_to_camera"],
                iteration_dir,
            )
            artifacts["candidate_render_mask"] = str(candidate_mask_path)
            artifacts["candidate_camera_npz"] = str(candidate_camera_path)
        except Exception as error:
            candidate_mask_error = repr(error)

        evidence = refinement_evidence(
            parent["_pose"]["world_to_camera"],
            candidate["_pose"],
            camera_model.process_size,
            geometry_profile.robot_radius_m,
            observed_mask,
            parent_render_mask,
            candidate_render_mask,
        )
        cycle_reference = accepted_history[-2] if len(accepted_history) >= 2 else None
        two_cycle = is_two_cycle(
            candidate["_pose"]["world_to_camera"],
            cycle_reference,
            geometry_profile.robot_radius_m,
        )
        selected, accepted, reason = select_refinement_candidate(
            parent,
            candidate,
            evidence,
            two_cycle=two_cycle,
        )
        if two_cycle:
            evidence.reasons.append("two_cycle")
        if candidate_mask_error is not None and observed_mask is None:
            evidence.reasons.append("candidate_mask_diagnostic_unavailable")

        if save_artifacts:
            candidate_pose_path = _save_iteration_pose(
                iteration_dir,
                candidate,
                camera_matrix,
                candidate_metrics,
                filename="candidate_pose.npz",
            )
            artifacts["candidate_pose_npz"] = str(candidate_pose_path)
            artifacts["candidate_pose_json"] = str(iteration_dir / "candidate_pose.json")

        if accepted:
            parent = selected
            parent_metrics = candidate_metrics
            parent_source = iteration_index
            accepted_history.append(
                np.asarray(parent["_pose"]["world_to_camera"], dtype=np.float64)
            )
            accepted_updates += 1
            status = "accepted"
        else:
            status = "rejected"
            stop_reason = reason

        if save_artifacts:
            pose_path = _save_iteration_pose(
                iteration_dir,
                parent,
                camera_matrix,
                parent_metrics,
            )
            artifacts["pose_npz"] = str(pose_path)
            artifacts["pose_json"] = str(iteration_dir / "pose.json")

        record = {
            "iteration": iteration_index,
            "stage": "pose_aligned_refine",
            "status": status,
            "attempted": True,
            "accepted": accepted,
            "reason": reason,
            "parent_pose_sha256": pose_sha256(
                iterations[-1]["selected_pnp"]["world_to_camera"]
            ),
            "candidate_pose_sha256": pose_sha256(candidate["_pose"]["world_to_camera"]),
            "selected_pose_source_iteration": parent_source,
            "selected_pose_sha256": pose_sha256(parent["_pose"]["world_to_camera"]),
            "selected_pnp": parent["pnp"],
            "selected_metrics": parent_metrics,
            "candidate": clean_record(candidate),
            "candidate_metrics": candidate_metrics,
            "candidate_mask_error": candidate_mask_error,
            "evidence": evidence.to_record(),
            "pose_update": {
                "rotation_delta_deg": evidence.rotation_delta_deg,
                "camera_center_delta_m": (
                    evidence.camera_center_delta_ratio
                    * geometry_profile.robot_radius_m
                ),
                "camera_center_delta_over_robot_radius": (
                    evidence.camera_center_delta_ratio
                ),
            },
            "two_cycle": two_cycle,
            "artifacts": artifacts,
        }
        iterations.append(record)
        _write_iteration_summary(iteration_dir, record, save_artifacts)
        if not accepted:
            break
        if is_small_update(evidence):
            stop_reason = "small_update"
            break

    while len(iterations) <= args.refine_iterations:
        iteration_index = len(iterations)
        filled = _filled_iteration_record(
            iteration_index,
            f"filled_after_{stop_reason}",
            parent,
            parent_metrics,
            parent_source,
            camera_matrix,
            frame_dir,
            save_artifacts,
        )
        iterations.append(filled)

    trajectory = {
        "enabled": True,
        "gate_id": "gate-v1",
        "requested_iterations": args.refine_iterations,
        "attempted_updates": attempted_updates,
        "accepted_updates": accepted_updates,
        "stopped_reason": stop_reason,
        "T0_sha256": pose_sha256(stage1_best["_pose"]["world_to_camera"]),
        "observed_mask_sha256": _mask_fingerprint(observed_mask),
        "processing_thresholds": {
            "homography_ransac_px": float(args.ransac_threshold),
            "pnp_ransac_px": float(args.pnp_threshold),
            "pnp_dedup_radius_px": float(args.pnp_dedup_radius),
            "image_diagonal_px": float(np.hypot(args.width, args.height)),
        },
        "camera_model": camera_model.to_record(),
        "geometry_profile": geometry_profile.to_record(),
        "iterations": iterations,
    }
    return parent, parent_metrics, trajectory


def process_frame(
    args: argparse.Namespace,
    matcher_api: ImcuiMatcher,
    index: int,
    json_path: Path,
    image_path: Path,
    mask_extractor=None,
) -> dict:
    frame_args = copy.copy(args)
    frame_args.json = json_path
    frame_args.image = image_path
    frame_args.save_all_matches = args.save_all_matches

    frame_dir = args.output_dir / f"{index:06d}"
    summary_path = frame_dir / "frame_summary.json"
    if args.resume and summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        if args.refinement_enabled:
            try:
                _validate_refinement_resume(
                    existing,
                    args.run_provenance["resume_fingerprint"],
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"Refusing resume for frame {index:06d}: {error}"
                ) from error
        return existing

    frame_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        remove_frame_artifacts(frame_dir)
    render_dir = frame_dir / "renders"
    match_dir = frame_dir / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)

    payload = load_dream_payload(json_path)
    observed_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    observed_masked, input_mask_raw = apply_input_mask(
        observed_rgb,
        args.mask_input,
        mask_extractor,
        args.mask_prompt,
    )
    input_mask_path = None
    frame_args.height = observed_masked.shape[0] if args.height is None else args.height
    frame_args.width = observed_masked.shape[1] if args.width is None else args.width
    camera_model = None
    input_mask = input_mask_raw
    if args.refinement_enabled:
        camera_model = load_camera_model(
            frame_args,
            observed_rgb.shape,
            (frame_args.height, frame_args.width),
        )
        observed_for_match = camera_model.process_rgb(observed_rgb)
        input_mask = (
            None if input_mask_raw is None else camera_model.process_mask(input_mask_raw)
        )
        if input_mask is not None:
            observed_for_match = observed_for_match.copy()
            observed_for_match[~input_mask] = 0
        image_diagonal = float(np.hypot(frame_args.width, frame_args.height))
        frame_args.ransac_threshold = 0.01000 * image_diagonal
        frame_args.pnp_threshold = 0.00625 * image_diagonal
        frame_args.pnp_dedup_radius = 0.00375 * image_diagonal
        frame_args.opencv_rng_seed = DETERMINISTIC_SEED
    elif (frame_args.height, frame_args.width) != observed_masked.shape[:2]:
        observed_for_match = np.asarray(
            Image.fromarray(observed_masked).resize(
                (frame_args.width, frame_args.height),
                Image.BILINEAR,
            )
        )
    else:
        observed_for_match = observed_masked

    if input_mask is not None:
        input_mask_path = frame_dir / "input_mask.png"
        Image.fromarray(input_mask.astype(np.uint8) * 255).save(input_mask_path)
    camera_matrix = (
        camera_model.K_process
        if camera_model is not None
        else load_camera_matrix(frame_args, observed_for_match.shape)
    )

    try:
        model, data = make_model_and_data(
            args.mujoco_xml,
            frame_args.width,
            frame_args.height,
            camera_matrix,
            payload,
            args.visual_body_names,
            args.visual_geom_group,
            args.source_visual_geom_group,
        )
        fk_points = fk_keypoints(model, data, dream_keypoints(payload))
        geometry_profile = (
            make_geometry_profile(
                model,
                data,
                args.mujoco_xml,
                args.visual_geom_group,
                args.visual_body_names,
            )
            if args.refinement_enabled
            else None
        )
        if args.prerender_dir is not None:
            prerender_frame_dir = args.prerender_dir / f"{index:06d}"
            render_paths = [prerender_frame_dir / f"view_{view_index:02d}.png" for view_index in range(args.views)]
            missing = [path for path in render_paths if not path.is_file() or not path.with_name(f"{path.stem}_camera.npz").is_file()]
            if missing:
                raise FileNotFoundError(f"Missing pre-rendered view artifacts: {missing[:3]}")
            if camera_model is not None:
                for render_path in render_paths:
                    with np.load(
                        render_path.with_name(f"{render_path.stem}_camera.npz")
                    ) as camera_data:
                        if (
                            int(camera_data["image_height"]) != frame_args.height
                            or int(camera_data["image_width"]) != frame_args.width
                            or not np.allclose(
                                camera_data["zed_camera_matrix"],
                                camera_matrix,
                                atol=1e-9,
                                rtol=0.0,
                            )
                        ):
                            raise ValueError(
                                f"Pre-render camera contract mismatch: {render_path}"
                            )
        else:
            render_paths = render_orbit_views(model, data, frame_args, camera_matrix, render_dir)
        view_records = match_render_paths(observed_for_match, render_paths, matcher_api, model, data, camera_matrix, frame_args, match_dir)
        best = max(view_records, key=best_view_key)
        clean_views = []
        for record in view_records:
            clean = clean_record(record)
            clean["selected_best_view"] = record is best
            clean_views.append(clean)

        if best["pnp"]["status"] != "success":
            refinement_protocol = None
            if args.refinement_enabled:
                unavailable_iterations = []
                for iteration_index in range(args.refine_iterations + 1):
                    unavailable_iterations.append(
                        {
                            "iteration": iteration_index,
                            "stage": "stage1"
                            if iteration_index == 0
                            else "pose_aligned_refine",
                            "status": "unavailable",
                            "attempted": iteration_index == 0,
                            "accepted": False,
                            "reason": "stage1_missing",
                            "selected_metrics": None,
                            "selected_pnp": None,
                        }
                    )
                refinement_protocol = {
                    "enabled": True,
                    "gate_id": "gate-v1",
                    "requested_iterations": args.refine_iterations,
                    "resume_fingerprint": args.run_provenance["resume_fingerprint"],
                    "camera_model": camera_model.to_record(),
                    "geometry_profile": geometry_profile.to_record(),
                    "iterations": unavailable_iterations,
                }
            frame_summary = {
                "status": "failed",
                "frame_index": index,
                "json": str(json_path),
                "image": str(image_path),
                "input_mask": {
                    "enabled": bool(args.mask_input),
                    "prompt": args.mask_prompt,
                    "source": "SAM3 via romav2.benchmarks.utils.mask.Sam3Extractor.",
                    "mask_path": None if input_mask_path is None else str(input_mask_path),
                    "status": "fallback_unmasked"
                    if args.mask_input and input_mask is None
                    else ("detected" if input_mask is not None else "disabled"),
                },
                "reason": "No render view produced a valid PnP pose.",
                "views": clean_views,
                "refinement_protocol": refinement_protocol,
            }
        else:
            stage1_metrics = keypoint_metrics(
                payload,
                fk_points,
                best["_pose"],
                camera_matrix,
                camera_model,
            )
            refinement_protocol = None
            final_best = best
            metrics = stage1_metrics
            if args.refinement_enabled:
                final_best, metrics, refinement_protocol = run_refinement_trajectory(
                    frame_args,
                    matcher_api,
                    observed_for_match,
                    input_mask,
                    model,
                    data,
                    camera_model,
                    geometry_profile,
                    payload,
                    fk_points,
                    best,
                    stage1_metrics,
                    frame_dir,
                    match_dir,
                )
                refinement_protocol["resume_fingerprint"] = args.run_provenance[
                    "resume_fingerprint"
                ]
            pose_npz = save_pose_npz(
                frame_dir,
                final_best,
                camera_matrix,
                metrics,
            )
            keypoint_vis_path = None
            if args.save_visualizations:
                keypoint_vis_path = frame_dir / "best_keypoint_eval.jpg"
                visualization_rgb = (
                    observed_rgb if camera_model is not None else observed_for_match
                )
                Image.fromarray(draw_keypoint_eval(visualization_rgb, metrics)).save(
                    keypoint_vis_path,
                    quality=95,
                )
            frame_summary = {
                "status": "success",
                "frame_index": index,
                "json": str(json_path),
                "image": str(image_path),
                "input_mask": {
                    "enabled": bool(args.mask_input),
                    "prompt": args.mask_prompt,
                    "source": "SAM3 via romav2.benchmarks.utils.mask.Sam3Extractor.",
                    "mask_path": None if input_mask_path is None else str(input_mask_path),
                    "status": "fallback_unmasked"
                    if args.mask_input and input_mask is None
                    else ("detected" if input_mask is not None else "disabled"),
                },
                "best_render_path": best["render_path"],
                "best_camera_npz": best["camera_npz"],
                "stage1_best_pnp": best["pnp"],
                "stage1_keypoint_metrics": stage1_metrics,
                "pose_npz": str(pose_npz),
                "keypoint_visualization": None if keypoint_vis_path is None else str(keypoint_vis_path),
                "camera_to_robot_base": final_best["pnp"]["camera_to_robot_base"],
                "world_to_camera": final_best["pnp"]["world_to_camera"],
                "keypoint_metrics": metrics,
                "best_pnp": final_best["pnp"],
                "views": clean_views,
                "refinement_protocol": refinement_protocol,
            }
    except Exception as error:
        frame_summary = {
            "status": "error",
            "frame_index": index,
            "json": str(json_path),
            "image": str(image_path),
            "input_mask": {
                "enabled": bool(args.mask_input),
                "prompt": args.mask_prompt,
                "source": "SAM3 via romav2.benchmarks.utils.mask.Sam3Extractor.",
                "mask_path": None if input_mask_path is None else str(input_mask_path),
                "status": "fallback_unmasked"
                if args.mask_input and input_mask is None
                else ("detected" if input_mask is not None else "disabled"),
            },
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "refinement_protocol": {
                "enabled": True,
                "resume_fingerprint": args.run_provenance["resume_fingerprint"],
                "reason": "frame_exception",
            }
            if args.refinement_enabled
            else None,
        }

    summary_path.write_text(json.dumps(frame_summary, indent=2) + "\n")
    if not args.keep_renders:
        remove_render_artifacts(render_dir)
    return frame_summary


def auc_under_threshold(
    values: np.ndarray,
    threshold: float,
    delta: float,
    denominator: int | None = None,
) -> float:
    denominator = len(values) if denominator is None else int(denominator)
    if denominator < 1:
        return float("nan")
    thresholds = np.arange(0.0, threshold, delta, dtype=np.float64)
    counts = [(values <= value).sum() / denominator for value in thresholds]
    return float(np.trapz(counts, dx=delta) / threshold)


def discrete_accuracy_auc(
    values: np.ndarray,
    thresholds: np.ndarray,
    denominator: int | None = None,
) -> float:
    """Match the threshold-loop AUC used by the official CtRNet Baxter notebook."""
    denominator = len(values) if denominator is None else int(denominator)
    if denominator < 1:
        return float("nan")
    return float(np.mean([(values < threshold).sum() / denominator for threshold in thresholds]))


def summarize(
    records: list[dict],
    auc_threshold: float,
    auc_delta: float,
    *,
    all_frame_threshold_denominator: bool = False,
) -> dict:
    successes = [record for record in records if record.get("status") == "success"]
    failures = [record for record in records if record.get("status") != "success"]
    add_values = np.asarray(
        [record["keypoint_metrics"]["summary"]["keypoint_add_mean_m"] for record in successes],
        dtype=np.float64,
    )
    pixel_values = np.asarray(
        [record["keypoint_metrics"]["summary"]["pixel_error_mean"] for record in successes],
        dtype=np.float64,
    )
    inlier_values = np.asarray([record["best_pnp"]["inliers"] for record in successes], dtype=np.float64)
    reproj_values = np.asarray(
        [record["best_pnp"]["inlier_reprojection_error_mean"] for record in successes],
        dtype=np.float64,
    )
    mask_statuses = [
        record.get("input_mask", {}).get("status", "unknown") for record in records
    ]

    summary = {
        "n_frames": len(records),
        "n_success": len(successes),
        "n_failed": len(failures),
        "success_rate": float(len(successes) / len(records) * 100.0) if records else float("nan"),
        "ADD/mean": float(add_values.mean()) if len(add_values) else float("nan"),
        "ADD/median": float(np.median(add_values)) if len(add_values) else float("nan"),
        "ADD/AUC": auc_under_threshold(
            add_values,
            auc_threshold,
            auc_delta,
            denominator=len(records) if all_frame_threshold_denominator else None,
        ),
        "ADD/auc_threshold_m": float(auc_threshold),
        "pixel_error/mean": float(pixel_values.mean()) if len(pixel_values) else float("nan"),
        "pixel_error/median": float(np.median(pixel_values)) if len(pixel_values) else float("nan"),
        "pnp_inliers/mean": float(inlier_values.mean()) if len(inlier_values) else float("nan"),
        "pnp_reprojection_error/mean": float(reproj_values.mean()) if len(reproj_values) else float("nan"),
        "input_mask/detected_frames": mask_statuses.count("detected"),
        "input_mask/fallback_unmasked_frames": mask_statuses.count(
            "fallback_unmasked"
        ),
        "input_mask/disabled_frames": mask_statuses.count("disabled"),
        "Baxter/evaluation_denominator": len(records),
        "Baxter/PCK@50px": (
            float((pixel_values < 50.0).sum() / len(records) * 100.0) if records else float("nan")
        ),
        "Baxter/PCK_AUC@200px": discrete_accuracy_auc(
            pixel_values, np.arange(200, dtype=np.float64), denominator=len(records)
        ),
        "Baxter/ADD@100mm": (
            float((add_values < 0.1).sum() / len(records) * 100.0) if records else float("nan")
        ),
        "Baxter/ADD_AUC@400mm": discrete_accuracy_auc(
            add_values * 1000.0, np.arange(400, dtype=np.float64), denominator=len(records)
        ),
        "Baxter/Mean_2D_error_success_only_px": (
            float(pixel_values.mean()) if len(pixel_values) else float("nan")
        ),
        "Baxter/Mean_3D_error_success_only_mm": (
            float(add_values.mean() * 1000.0) if len(add_values) else float("nan")
        ),
    }
    threshold_denominator = (
        len(records) if all_frame_threshold_denominator else len(add_values)
    )
    for threshold_mm in (10, 20, 40, 60):
        summary[f"ADD<{threshold_mm}mm"] = (
            float(
                (add_values <= threshold_mm * 1e-3).sum()
                / threshold_denominator
                * 100.0
            )
            if threshold_denominator
            else float("nan")
        )
    pixel_denominator = (
        len(records) if all_frame_threshold_denominator else len(pixel_values)
    )
    for threshold_px in (5, 10, 20):
        summary[f"pixel_error<{threshold_px}px"] = (
            float((pixel_values <= threshold_px).sum() / pixel_denominator * 100.0)
            if pixel_denominator
            else float("nan")
        )
    summary["threshold_metric_denominator"] = (
        "all_requested_frames"
        if all_frame_threshold_denominator
        else "successful_frames"
    )
    return summary


def summarize_refinement_iterations(
    records: list[dict],
    requested_iterations: int,
    auc_threshold: float,
    auc_delta: float,
) -> dict:
    result = {}
    for iteration_index in range(requested_iterations + 1):
        derived = []
        statuses = {}
        accepted = 0
        attempted = 0
        catastrophic_absolute = 0
        catastrophic_regression = 0
        for frame in records:
            protocol = frame.get("refinement_protocol") or {}
            iterations = protocol.get("iterations", [])
            iteration = (
                iterations[iteration_index]
                if iteration_index < len(iterations)
                else {
                    "status": "unavailable",
                    "selected_metrics": None,
                    "selected_pnp": None,
                }
            )
            status = str(iteration.get("status", "unavailable"))
            statuses[status] = statuses.get(status, 0) + 1
            attempted += int(bool(iteration.get("attempted")))
            if iteration_index > 0:
                accepted += int(bool(iteration.get("accepted")))
            metrics = iteration.get("selected_metrics")
            pnp = iteration.get("selected_pnp")
            stage0_iterations = protocol.get("iterations", [])
            stage0_metrics = (
                stage0_iterations[0].get("selected_metrics")
                if stage0_iterations
                else None
            )
            stage0_error = (
                stage0_metrics["summary"]["keypoint_add_mean_m"]
                if stage0_metrics is not None
                else float("inf")
            )
            if metrics is None or pnp is None or pnp.get("status") != "success":
                derived.append({"status": "failed"})
                catastrophic_absolute += 1
                if stage0_error < 0.1:
                    catastrophic_regression += 1
            else:
                selected_error = metrics["summary"]["keypoint_add_mean_m"]
                derived.append(
                    {
                        "status": "success",
                        "keypoint_metrics": metrics,
                        "best_pnp": pnp,
                    }
                )
                if selected_error >= 0.1:
                    catastrophic_absolute += 1
                    if stage0_error < 0.1:
                        catastrophic_regression += 1
        summary = summarize(
            derived,
            auc_threshold,
            auc_delta,
            all_frame_threshold_denominator=True,
        )
        summary.update(
            {
                "iteration": iteration_index,
                "attempted_frames": attempted,
                "accepted_updates": accepted,
                "gate_coverage": (
                    float(accepted / attempted) if attempted else float("nan")
                ),
                "catastrophic_absolute_count_ADD>=100mm_or_failure": (
                    catastrophic_absolute
                ),
                "catastrophic_absolute_rate": (
                    float(catastrophic_absolute / len(records))
                    if records
                    else float("nan")
                ),
                "catastrophic_regression_from_S0_count": catastrophic_regression,
                "status_counts": statuses,
            }
        )
        result[f"iteration_{iteration_index:02d}"] = summary
    return result


def process(args: argparse.Namespace, matcher_api: ImcuiMatcher | None = None) -> Path:
    input_dataset_dir = args.dataset_dir
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    if args.views < 1:
        raise ValueError("--views must be >= 1")
    if args.match_batch_size is not None and args.match_batch_size < 1:
        raise ValueError("--match-batch-size must be >= 1")
    if args.refine_iterations and not args.refinement_enabled:
        raise ValueError("--refine-iterations requires --refinement-enabled")
    if args.refinement_enabled and args.distortion_state == "unknown":
        raise ValueError(
            "--refinement-enabled requires --distortion-state rectified or distorted"
        )
    if args.refinement_enabled:
        frozen_values = {
            "max_keypoints": (args.max_keypoints, 2048),
            "score_filter": (args.score_filter, 0.0),
            "ransac_method": (args.ransac_method, "CV2_USAC_MAGSAC"),
            "ransac_confidence": (args.ransac_confidence, 0.9999),
            "ransac_max_iter": (args.ransac_max_iter, 10_000),
            "min_pnp_correspondences": (args.min_pnp_correspondences, 6),
            "max_pnp_correspondences": (args.max_pnp_correspondences, 512),
        }
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in frozen_values.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"gate-v1 frozen solver configuration mismatch: {mismatches}")
        if not args.save_all_matches or not args.save_refinement_artifacts:
            raise ValueError(
                "gate-v1 requires match and per-iteration evidence artifacts"
            )
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    if not args.mujoco_xml.is_file():
        raise FileNotFoundError(args.mujoco_xml)

    frames = discover_frames(args)
    if not frames:
        raise RuntimeError(f"No frames found under {args.dataset_dir}")

    if (
        args.refinement_enabled
        and not args.resume
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(
            f"Refinement output directory already contains artifacts: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.refinement_enabled and args.config_source is not None:
        shutil.copy2(args.config_source, args.output_dir / "runtime_config.toml")
    determinism = (
        configure_determinism() if args.refinement_enabled else {"enabled": False}
    )
    args.run_provenance = _runtime_provenance(args, frames)
    args.run_provenance["determinism"] = determinism
    args.run_provenance["verified_frozen_specs"] = (
        verify_frozen_specs(PROJECT_ROOT) if args.refinement_enabled else None
    )
    args.run_provenance["verified_weights"] = (
        _weight_provenance(args) if args.refinement_enabled else None
    )
    resume_payload = {
        key: args.run_provenance[key]
        for key in (
            "git",
            "resolved_runtime_config_sha256",
            "frame_manifest_sha256",
            "mujoco_model",
            "config_source",
            "frozen_spec",
            "determinism",
            "verified_frozen_specs",
            "verified_weights",
        )
    }
    args.run_provenance["resume_fingerprint"] = sha256_bytes(
        canonical_json_bytes(resume_payload)
    )
    provenance_path = args.output_dir / "run_provenance.json"
    provenance_path.write_text(json.dumps(args.run_provenance, indent=2) + "\n")
    matcher_api = matcher_api or ImcuiMatcher(args)
    mask_extractor = make_sam3_extractor(args.mask_input, args.sam3_checkpoint)
    records = []
    for ordinal, (index, json_path, image_path) in enumerate(frames, start=1):
        print(f"[{ordinal}/{len(frames)}] frame {index:06d}")
        record = process_frame(args, matcher_api, index, json_path, image_path, mask_extractor)
        records.append(record)
        running = summarize(
            records,
            args.auc_threshold,
            args.auc_delta,
            all_frame_threshold_denominator=args.refinement_enabled,
        )
        (args.output_dir / "summary_running.json").write_text(json.dumps(running, indent=2) + "\n")

    summary = summarize(
        records,
        args.auc_threshold,
        args.auc_delta,
        all_frame_threshold_denominator=args.refinement_enabled,
    )
    refinement_summary = (
        summarize_refinement_iterations(
            records,
            args.refine_iterations,
            args.auc_threshold,
            args.auc_delta,
        )
        if args.refinement_enabled
        else None
    )
    output = {
        "dataset_dir_input": str(input_dataset_dir),
        "dataset_dir": str(args.dataset_dir),
        "mujoco_xml": str(args.mujoco_xml),
        "visual_geom_group": args.visual_geom_group,
        "source_visual_geom_group": args.source_visual_geom_group,
        "visual_body_names": args.visual_body_names,
        "matcher": args.matcher,
        "views": args.views,
        "input_mask": {
            "enabled": bool(args.mask_input),
            "prompt": args.mask_prompt,
            "source": "SAM3 via romav2.benchmarks.utils.mask.Sam3Extractor.",
        },
        "frame_count": len(frames),
        "summary": summary,
        "refinement": {
            "enabled": bool(args.refinement_enabled),
            "gate_id": "gate-v1" if args.refinement_enabled else None,
            "requested_iterations": args.refine_iterations,
            "iteration_summary": refinement_summary,
        },
        "run_provenance": args.run_provenance,
        "frames": records,
        "metric_note": {
            "ADD/mean": "Mean of per-frame mean 3D FK keypoint error in camera coordinates.",
            "ADD/AUC": "RoboPose-style area under accuracy-threshold curve over per-frame ADD values.",
            "pixel_error/mean": "Mean of per-frame mean 2D FK keypoint reprojection error.",
            "Baxter/evaluation_denominator": "All requested frames; failed PnP frames count as misses at every official threshold.",
            "Baxter/PCK@50px": "Official CtRNet endpoint accuracy below 50 pixels, using all requested frames as denominator.",
            "Baxter/PCK_AUC@200px": "Official CtRNet discrete PCK AUC over integer thresholds 0..199 pixels; failed PnP frames are misses.",
            "Baxter/ADD@100mm": "Official CtRNet endpoint 3D accuracy below 100 mm, using all requested frames as denominator.",
            "Baxter/ADD_AUC@400mm": "Official CtRNet discrete endpoint ADD AUC over integer thresholds 0..399 mm; failed PnP frames are misses.",
            "Baxter/Mean_2D_error_success_only_px": "Mean endpoint reprojection error over frames with a valid PnP pose; not directly comparable to the official 100-frame mean when failures exist.",
            "Baxter/Mean_3D_error_success_only_mm": "Mean endpoint 3D error over frames with a valid PnP pose; not directly comparable to the official 100-frame mean when failures exist.",
        },
    }
    args.run_provenance["ended_at"] = datetime.now().astimezone().isoformat()
    args.run_provenance["exit_code"] = 0
    provenance_path.write_text(json.dumps(args.run_provenance, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    return args.output_dir


def main() -> None:
    args = parse_args()
    try:
        output_dir = process(args)
    except Exception as error:
        provenance_path = args.output_dir / "run_provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            provenance["ended_at"] = datetime.now().astimezone().isoformat()
            provenance["exit_code"] = 1
            provenance["error"] = repr(error)
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        raise
    print(f"Saved batch evaluation to {output_dir}")


if __name__ == "__main__":
    main()
