#!/usr/bin/env python3
"""Render a Baxter pose from an evaluator summary and alpha-blend it on the real RGB."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from romav2.benchmarks.utils.geometry import add_zed_camera, visual_scene_option
from romav2.benchmarks.utils.pose import load_camera_matrix
from romav2.benchmarks.utils.render import apply_json_qpos, load_joint_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--mujoco-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-geom-group", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


def find_frame(summary: dict, frame_index: int) -> dict:
    for frame in summary["frames"]:
        if int(frame["frame_index"]) == frame_index:
            return frame
    raise KeyError(f"frame {frame_index} was not found")


def camera_pose(model, camera_id: int, world_to_camera: np.ndarray) -> None:
    rotation_camera_to_world = world_to_camera[:3, :3].T
    camera_position = -rotation_camera_to_world @ world_to_camera[:3, 3]
    forward = rotation_camera_to_world @ np.array([0.0, 0.0, 1.0])
    up = rotation_camera_to_world @ np.array([0.0, -1.0, 0.0])
    rotation = np.column_stack([np.cross(forward, up), up, -forward])
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    model.cam_pos[camera_id] = camera_position
    model.cam_quat[camera_id] = quaternion


def labeled(image: np.ndarray, text: str) -> Image.Image:
    pil = Image.fromarray(image)
    result = Image.new("RGB", (pil.width, pil.height + 40), "white")
    result.paste(pil, (0, 40))
    ImageDraw.Draw(result).text((12, 12), text, fill="black", font=ImageFont.load_default())
    return result


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    frame = find_frame(summary, args.frame_index)
    if frame["status"] != "success":
        raise RuntimeError(f"frame status is {frame['status']!r}, not success")

    json_path = Path(frame["json"])
    image_path = Path(frame["image"])
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = rgb.shape[:2]
    matrix_args = SimpleNamespace(
        json=json_path,
        camera_settings=None,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        fallback_focal=400.0,
    )
    camera_matrix = load_camera_matrix(matrix_args, rgb.shape)

    spec = mujoco.MjSpec.from_file(str(args.mujoco_xml))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)
    add_zed_camera(spec, "estimated_pose_camera", width, height, camera_matrix)
    model = spec.compile()
    data = mujoco.MjData(model)
    apply_json_qpos(model, data, load_joint_positions(json_path))
    world_to_camera = np.asarray(frame["world_to_camera"], dtype=np.float64)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "estimated_pose_camera")
    camera_pose(model, camera_id, world_to_camera)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(
            data,
            camera=camera_id,
            scene_option=visual_scene_option(args.visual_geom_group),
        )
        rendered = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()

    mask = segmentation[..., 0] >= 0
    overlay = rgb.copy()
    overlay[mask] = (
        (1.0 - args.alpha) * overlay[mask] + args.alpha * rendered[mask]
    ).astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{args.frame_index:06d}"
    Image.fromarray(rendered).save(args.output_dir / f"{stem}_estimated_pose_render.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(
        args.output_dir / f"{stem}_estimated_pose_mask.png"
    )
    Image.fromarray(overlay).save(
        args.output_dir / f"{stem}_estimated_pose_overlay.jpg",
        quality=95,
        subsampling=0,
    )

    panels = [
        labeled(rgb, "real"),
        labeled(rendered, "full Baxter at estimated pose"),
        labeled(overlay, f"overlay alpha={args.alpha:.2f}"),
    ]
    preview_width = 683
    resized = [
        panel.resize(
            (preview_width, round(panel.height * preview_width / panel.width)),
            Image.Resampling.LANCZOS,
        )
        for panel in panels
    ]
    sheet = Image.new(
        "RGB",
        (sum(panel.width for panel in resized), max(panel.height for panel in resized)),
        "white",
    )
    offset = 0
    for panel in resized:
        sheet.paste(panel, (offset, 0))
        offset += panel.width
    sheet.save(
        args.output_dir / f"{stem}_real_render_overlay.jpg",
        quality=94,
        subsampling=0,
    )
    print(
        json.dumps(
            {
                "frame_index": args.frame_index,
                "image_size": [width, height],
                "rendered_mask_coverage_percent": float(mask.mean() * 100.0),
                "world_to_camera_source": str(args.summary),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
