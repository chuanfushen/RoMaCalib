"""Single-frame Calib-X pose-estimation pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .matching import ImcuiMatcher
from .pose import (
    apply_input_mask,
    best_view_key,
    draw_keypoint_eval,
    dream_keypoints,
    fk_keypoints,
    keypoint_metrics,
    load_camera_matrix,
    load_dream_payload,
    make_model_and_data,
    match_one_render,
    render_orbit_views,
    save_pose_npz,
)


def _clean_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _remove_render_artifacts(render_dir: Path) -> None:
    if not render_dir.exists():
        return
    for path in render_dir.glob("view_*"):
        if path.is_file():
            path.unlink()


def _remove_frame_artifacts(frame_dir: Path) -> None:
    for filename in (
        "input_mask.png",
        "best_keypoint_eval.jpg",
        "best_pose.npz",
        "frame_summary.json",
    ):
        path = frame_dir / filename
        if path.is_file():
            path.unlink()
    match_dir = frame_dir / "matches"
    if match_dir.exists():
        for path in match_dir.glob("view_*_matches.*"):
            if path.is_file():
                path.unlink()


@torch.inference_mode()
def _match_romav2_images_batched(
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
        render_tensors = [
            torch.from_numpy(render_rgb.copy()).permute(2, 0, 1)
            for render_rgb in chunk_rgbs
        ]
        image0 = observed_tensor.to(device)[None].repeat(
            len(render_tensors),
            1,
            1,
            1,
        )
        image1 = torch.stack(render_tensors, dim=0).to(device)
        torch.set_float32_matmul_precision("highest")
        predictions = net.match(image0, image1)
        h0, w0 = image0.shape[-2:]
        h1, w1 = image1.shape[-2:]

        for batch_index, render_rgb in enumerate(chunk_rgbs):
            sliced = {
                key: (
                    value[batch_index : batch_index + 1]
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in predictions.items()
            }
            matches, confidence, _, _ = net.sample(
                sliced,
                matcher_api.max_keypoints,
            )
            mkpts0, mkpts1 = net.to_pixel_coordinates(
                matches,
                h0,
                w0,
                h1,
                w1,
            )
            outputs.append(
                (
                    {
                        "mkeypoints0_orig": (mkpts0.detach().cpu().numpy()),
                        "mkeypoints1_orig": (mkpts1.detach().cpu().numpy()),
                        "mconf": confidence.detach().cpu().numpy(),
                    },
                    render_rgb,
                )
            )
    return outputs


def _match_render_paths(
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
        predictions = _match_romav2_images_batched(
            matcher_api,
            observed_rgb,
            render_paths,
            match_batch_size,
        )
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
            for render_path, (prediction, render_rgb) in zip(
                render_paths,
                predictions,
            )
        ]
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
        )
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
    """Estimate and evaluate the camera pose for one DREAM-style frame."""
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
        _remove_frame_artifacts(frame_dir)
    render_dir = frame_dir / "renders"
    match_dir = frame_dir / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)

    payload = load_dream_payload(json_path)
    observed_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    observed_masked, input_mask = apply_input_mask(
        observed_rgb,
        args.mask_input,
        mask_extractor,
        args.mask_prompt,
    )
    input_mask_path = None
    if input_mask is not None:
        input_mask_path = frame_dir / "input_mask.png"
        Image.fromarray(input_mask.astype(np.uint8) * 255).save(input_mask_path)
    frame_args.height = observed_masked.shape[0] if args.height is None else args.height
    frame_args.width = observed_masked.shape[1] if args.width is None else args.width
    if (frame_args.height, frame_args.width) != observed_masked.shape[:2]:
        observed_for_match = np.asarray(
            Image.fromarray(observed_masked).resize(
                (frame_args.width, frame_args.height),
                Image.BILINEAR,
            )
        )
    else:
        observed_for_match = observed_masked

    camera_matrix = load_camera_matrix(
        frame_args,
        observed_for_match.shape,
    )

    try:
        model, data = make_model_and_data(
            args.mujoco_xml,
            frame_args.width,
            frame_args.height,
            camera_matrix,
            payload,
        )
        fk_points = fk_keypoints(model, data, dream_keypoints(payload))
        if args.prerender_dir is not None:
            prerender_frame_dir = args.prerender_dir / f"{index:06d}"
            render_paths = [
                prerender_frame_dir / f"view_{view_index:02d}.png"
                for view_index in range(args.views)
            ]
            missing = [
                path
                for path in render_paths
                if not path.is_file()
                or not path.with_name(f"{path.stem}_camera.npz").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing pre-rendered view artifacts: {missing[:3]}"
                )
        else:
            render_paths = render_orbit_views(
                model,
                data,
                frame_args,
                camera_matrix,
                render_dir,
            )
        view_records = _match_render_paths(
            observed_for_match,
            render_paths,
            matcher_api,
            model,
            data,
            camera_matrix,
            frame_args,
            match_dir,
        )
        best = max(view_records, key=best_view_key)
        clean_views = []
        for record in view_records:
            clean = _clean_record(record)
            clean["selected_best_view"] = record is best
            clean_views.append(clean)

        input_mask_record = {
            "enabled": bool(args.mask_input),
            "prompt": args.mask_prompt,
            "source": "SAM3 via calibx.masking.Sam3Extractor.",
            "mask_path": (None if input_mask_path is None else str(input_mask_path)),
        }
        if best["pnp"]["status"] != "success":
            frame_summary = {
                "status": "failed",
                "frame_index": index,
                "json": str(json_path),
                "image": str(image_path),
                "input_mask": input_mask_record,
                "reason": "No render view produced a valid PnP pose.",
                "views": clean_views,
            }
        else:
            keypoint_result = keypoint_metrics(
                payload,
                fk_points,
                best["_pose"],
                camera_matrix,
            )
            pose_npz = save_pose_npz(
                frame_dir,
                best,
                camera_matrix,
                keypoint_result,
            )
            keypoint_vis_path = None
            if args.save_visualizations:
                keypoint_vis_path = frame_dir / "best_keypoint_eval.jpg"
                Image.fromarray(
                    draw_keypoint_eval(
                        observed_for_match,
                        keypoint_result,
                    )
                ).save(keypoint_vis_path, quality=95)
            frame_summary = {
                "status": "success",
                "frame_index": index,
                "json": str(json_path),
                "image": str(image_path),
                "input_mask": input_mask_record,
                "best_render_path": best["render_path"],
                "best_camera_npz": best["camera_npz"],
                "pose_npz": str(pose_npz),
                "keypoint_visualization": (
                    None if keypoint_vis_path is None else str(keypoint_vis_path)
                ),
                "camera_to_robot_base": (best["pnp"]["camera_to_robot_base"]),
                "world_to_camera": best["pnp"]["world_to_camera"],
                "keypoint_metrics": keypoint_result,
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
                "source": "SAM3 via calibx.masking.Sam3Extractor.",
                "mask_path": (
                    None if input_mask_path is None else str(input_mask_path)
                ),
            },
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }

    summary_path.write_text(json.dumps(frame_summary, indent=2) + "\n")
    if not args.keep_renders:
        _remove_render_artifacts(render_dir)
    return frame_summary
