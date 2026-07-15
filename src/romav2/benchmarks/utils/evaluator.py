#!/usr/bin/env python3
"""Batch evaluation for DREAM-style Panda frames with MuJoCo render matching."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
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
    load_dream_payload,
    make_sam3_extractor,
    make_model_and_data,
    match_one_render,
    render_orbit_views,
    save_pose_npz,
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        return json.loads(summary_path.read_text())

    frame_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        remove_frame_artifacts(frame_dir)
    render_dir = frame_dir / "renders"
    match_dir = frame_dir / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)

    payload = load_dream_payload(json_path)
    observed_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    observed_masked, input_mask = apply_input_mask(observed_rgb, args.mask_input, mask_extractor, args.mask_prompt)
    input_mask_path = None
    if input_mask is not None:
        input_mask_path = frame_dir / "input_mask.png"
        Image.fromarray(input_mask.astype(np.uint8) * 255).save(input_mask_path)
    frame_args.height = observed_masked.shape[0] if args.height is None else args.height
    frame_args.width = observed_masked.shape[1] if args.width is None else args.width
    if (frame_args.height, frame_args.width) != observed_masked.shape[:2]:
        observed_for_match = np.asarray(Image.fromarray(observed_masked).resize((frame_args.width, frame_args.height), Image.BILINEAR))
    else:
        observed_for_match = observed_masked

    camera_matrix = load_camera_matrix(frame_args, observed_for_match.shape)

    try:
        model, data = make_model_and_data(args.mujoco_xml, frame_args.width, frame_args.height, camera_matrix, payload)
        fk_points = fk_keypoints(model, data, dream_keypoints(payload))
        if args.prerender_dir is not None:
            prerender_frame_dir = args.prerender_dir / f"{index:06d}"
            render_paths = [prerender_frame_dir / f"view_{view_index:02d}.png" for view_index in range(args.views)]
            missing = [path for path in render_paths if not path.is_file() or not path.with_name(f"{path.stem}_camera.npz").is_file()]
            if missing:
                raise FileNotFoundError(f"Missing pre-rendered view artifacts: {missing[:3]}")
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
                },
                "reason": "No render view produced a valid PnP pose.",
                "views": clean_views,
            }
        else:
            metrics = keypoint_metrics(payload, fk_points, best["_pose"], camera_matrix)
            pose_npz = save_pose_npz(frame_dir, best, camera_matrix, metrics)
            keypoint_vis_path = None
            if args.save_visualizations:
                keypoint_vis_path = frame_dir / "best_keypoint_eval.jpg"
                Image.fromarray(draw_keypoint_eval(observed_for_match, metrics)).save(keypoint_vis_path, quality=95)
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
                },
                "best_render_path": best["render_path"],
                "best_camera_npz": best["camera_npz"],
                "pose_npz": str(pose_npz),
                "keypoint_visualization": None if keypoint_vis_path is None else str(keypoint_vis_path),
                "camera_to_robot_base": best["pnp"]["camera_to_robot_base"],
                "world_to_camera": best["pnp"]["world_to_camera"],
                "keypoint_metrics": metrics,
                "best_pnp": best["pnp"],
                "views": clean_views,
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
            },
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }

    summary_path.write_text(json.dumps(frame_summary, indent=2) + "\n")
    if not args.keep_renders:
        remove_render_artifacts(render_dir)
    return frame_summary


def auc_under_threshold(values: np.ndarray, threshold: float, delta: float) -> float:
    if len(values) == 0:
        return float("nan")
    thresholds = np.arange(0.0, threshold, delta, dtype=np.float64)
    counts = [(values <= value).mean() for value in thresholds]
    return float(np.trapz(counts, dx=delta) / threshold)


def summarize(records: list[dict], auc_threshold: float, auc_delta: float) -> dict:
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

    summary = {
        "n_frames": len(records),
        "n_success": len(successes),
        "n_failed": len(failures),
        "success_rate": float(len(successes) / len(records) * 100.0) if records else float("nan"),
        "ADD/mean": float(add_values.mean()) if len(add_values) else float("nan"),
        "ADD/median": float(np.median(add_values)) if len(add_values) else float("nan"),
        "ADD/AUC": auc_under_threshold(add_values, auc_threshold, auc_delta),
        "ADD/auc_threshold_m": float(auc_threshold),
        "pixel_error/mean": float(pixel_values.mean()) if len(pixel_values) else float("nan"),
        "pixel_error/median": float(np.median(pixel_values)) if len(pixel_values) else float("nan"),
        "pnp_inliers/mean": float(inlier_values.mean()) if len(inlier_values) else float("nan"),
        "pnp_reprojection_error/mean": float(reproj_values.mean()) if len(reproj_values) else float("nan"),
    }
    for threshold_mm in (10, 20, 40, 60):
        summary[f"ADD<{threshold_mm}mm"] = (
            float((add_values <= threshold_mm * 1e-3).mean() * 100.0) if len(add_values) else float("nan")
        )
    for threshold_px in (5, 10, 20):
        summary[f"pixel_error<{threshold_px}px"] = (
            float((pixel_values <= threshold_px).mean() * 100.0) if len(pixel_values) else float("nan")
        )
    return summary


def process(args: argparse.Namespace, matcher_api: ImcuiMatcher | None = None) -> Path:
    input_dataset_dir = args.dataset_dir
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    if args.views < 1:
        raise ValueError("--views must be >= 1")
    if args.match_batch_size is not None and args.match_batch_size < 1:
        raise ValueError("--match-batch-size must be >= 1")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matcher_api = matcher_api or ImcuiMatcher(args)
    mask_extractor = make_sam3_extractor(args.mask_input, args.sam3_checkpoint)
    records = []
    for ordinal, (index, json_path, image_path) in enumerate(frames, start=1):
        print(f"[{ordinal}/{len(frames)}] frame {index:06d}")
        record = process_frame(args, matcher_api, index, json_path, image_path, mask_extractor)
        records.append(record)
        running = summarize(records, args.auc_threshold, args.auc_delta)
        (args.output_dir / "summary_running.json").write_text(json.dumps(running, indent=2) + "\n")

    summary = summarize(records, args.auc_threshold, args.auc_delta)
    output = {
        "dataset_dir_input": str(input_dataset_dir),
        "dataset_dir": str(args.dataset_dir),
        "mujoco_xml": str(args.mujoco_xml),
        "matcher": args.matcher,
        "views": args.views,
        "input_mask": {
            "enabled": bool(args.mask_input),
            "prompt": args.mask_prompt,
            "source": "SAM3 via romav2.benchmarks.utils.mask.Sam3Extractor.",
        },
        "frame_count": len(frames),
        "summary": summary,
        "frames": records,
        "metric_note": {
            "ADD/mean": "Mean of per-frame mean 3D FK keypoint error in camera coordinates.",
            "ADD/AUC": "RoboPose-style area under accuracy-threshold curve over per-frame ADD values.",
            "pixel_error/mean": "Mean of per-frame mean 2D FK keypoint reprojection error.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    return args.output_dir


def main() -> None:
    output_dir = process(parse_args())
    print(f"Saved batch evaluation to {output_dir}")


if __name__ == "__main__":
    main()
