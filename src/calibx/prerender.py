#!/usr/bin/env python3
"""Pre-render MuJoCo orbit views for a DREAM-style Panda dataset.

This script compiles the MJCF once, creates one MuJoCo renderer, then iterates
over all JSON frames by only updating qpos and calling mj_forward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from .geometry import add_zed_camera, visual_bounds  # noqa: E402
from .pose import load_camera_matrix  # noqa: E402
from .rendering import apply_json_qpos, load_joint_positions  # noqa: E402

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dream"
DEFAULT_MJCF = (
    PROJECT_ROOT / "assets/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/dream_mujoco_prerendered_views"


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
        frame_dirs = {
            path.parent for path in dataset_dir.rglob("*.json") if path.stem.isdigit()
        }
        for frame_dir in frame_dirs:
            count = count_dream_frames(frame_dir)
            if count > 0:
                candidates[frame_dir] = count

    if not candidates:
        return dataset_dir

    return sorted(
        candidates.items(),
        key=lambda item: (-item[1], len(item[0].parts), str(item[0])),
    )[0][0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--views", "-x", type=int, default=6)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--distance-scale", type=float, default=2.8)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--azimuth-offset", type=float, default=0.0)
    parser.add_argument("--camera-settings", type=Path, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--fallback-focal", type=float, default=400.0)
    parser.add_argument("--frame-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Randomly sample this many frames after range/stride filtering.",
    )
    parser.add_argument("--sample-seed", type=int, default=90)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-npz", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-png", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def discover_frames(args: argparse.Namespace) -> list[tuple[int, Path]]:
    requested = (
        None
        if args.frame_indices is None
        else {int(index) for index in args.frame_indices}
    )
    frames = []
    for json_path in sorted(args.dataset_dir.glob("*.json")):
        if json_path.name == "_camera_settings.json" or not json_path.stem.isdigit():
            continue
        index = int(json_path.stem)
        if requested is not None and index not in requested:
            continue
        if args.start_index is not None and index < args.start_index:
            continue
        if args.end_index is not None and index > args.end_index:
            continue
        if args.stride > 1 and index % args.stride != 0:
            continue
        if not json_path.with_suffix(".rgb.jpg").is_file():
            continue
        frames.append((index, json_path))
    if args.sample_count is not None:
        if args.sample_count < 1:
            raise ValueError("--sample-count must be >= 1")
        if args.sample_count > len(frames):
            raise ValueError(
                f"--sample-count={args.sample_count} exceeds available frames {len(frames)}"
            )
        rng = np.random.default_rng(args.sample_seed)
        sampled = rng.choice(len(frames), size=args.sample_count, replace=False)
        frames = [frames[int(index)] for index in sorted(sampled)]
    if args.limit is not None:
        frames = frames[: args.limit]
    frames = frames[args.shard_index :: args.num_shards]
    return frames


def compile_model(args: argparse.Namespace, camera_matrix: np.ndarray):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(args.mujoco_xml))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.height)
    add_zed_camera(spec, "zed_render_camera", args.width, args.height, camera_matrix)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def compute_orbit_camera(
    model,
    data,
    renderer,
    scene_option,
    center: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
):
    import mujoco

    free_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, free_camera)
    free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    free_camera.lookat[:] = center
    free_camera.distance = float(distance)
    free_camera.azimuth = float(azimuth)
    free_camera.elevation = float(elevation)

    renderer.update_scene(data, camera=free_camera, scene_option=scene_option)
    scene_cameras = renderer.scene.camera
    camera_position = np.mean(
        [np.asarray(scene_camera.pos) for scene_camera in scene_cameras], axis=0
    )
    camera_forward = np.mean(
        [np.asarray(scene_camera.forward) for scene_camera in scene_cameras], axis=0
    )
    camera_forward /= np.linalg.norm(camera_forward)
    camera_up = np.mean(
        [np.asarray(scene_camera.up) for scene_camera in scene_cameras], axis=0
    )
    camera_up /= np.linalg.norm(camera_up)
    camera_rotation = np.column_stack(
        [np.cross(camera_forward, camera_up), camera_up, -camera_forward]
    )
    camera_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(camera_quaternion, camera_rotation.reshape(-1))
    return camera_position, camera_quaternion


def render_frame(
    args: argparse.Namespace,
    model,
    data,
    renderer,
    scene_option,
    camera_id: int,
    camera_matrix: np.ndarray,
    index: int,
    json_path: Path,
) -> dict:
    import mujoco

    frame_dir = args.output_dir / f"{index:06d}"
    manifest_path = frame_dir / "metadata.json"
    if args.resume and manifest_path.is_file():
        return json.loads(manifest_path.read_text())

    frame_dir.mkdir(parents=True, exist_ok=True)
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    applied = apply_json_qpos(model, data, load_joint_positions(json_path))
    mujoco.mj_forward(model, data)

    minimum, maximum = visual_bounds(model, data)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * np.linalg.norm(maximum - minimum)
    distance = max(args.min_distance, args.distance_scale * radius)

    views = []
    for view_index in range(args.views):
        azimuth = args.azimuth_offset + view_index * 360.0 / args.views
        camera_position, camera_quaternion = compute_orbit_camera(
            model,
            data,
            renderer,
            scene_option,
            center,
            distance,
            azimuth,
            args.elevation,
        )
        model.cam_pos[camera_id] = camera_position
        model.cam_quat[camera_id] = camera_quaternion
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera_id, scene_option=scene_option)

        image_path = frame_dir / f"view_{view_index:02d}.png"
        camera_path = frame_dir / f"view_{view_index:02d}_camera.npz"
        if args.save_png:
            Image.fromarray(renderer.render().copy()).save(image_path)
        if args.save_npz:
            np.savez_compressed(
                camera_path,
                camera_position=data.cam_xpos[camera_id].copy(),
                camera_rotation=data.cam_xmat[camera_id].reshape(3, 3).copy(),
                camera_forward=-data.cam_xmat[camera_id].reshape(3, 3)[:, 2],
                camera_up=data.cam_xmat[camera_id].reshape(3, 3)[:, 1],
                camera_quaternion=camera_quaternion,
                image_height=args.height,
                image_width=args.width,
                zed_camera_matrix=camera_matrix,
                camera_lookat=center,
                camera_azimuth=azimuth,
                camera_elevation=args.elevation,
                camera_distance=distance,
            )
        views.append(
            {
                "view_index": view_index,
                "image": str(image_path) if args.save_png else None,
                "camera_npz": str(camera_path) if args.save_npz else None,
                "azimuth_degrees": float(azimuth % 360.0),
                "elevation_degrees": float(args.elevation),
                "distance": float(distance),
            }
        )

    metadata = {
        "frame_index": index,
        "json": str(json_path),
        "rgb": str(json_path.with_suffix(".rgb.jpg")),
        "center": center.tolist(),
        "radius": float(radius),
        "applied_joints": applied,
        "views": views,
    }
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def process(args: argparse.Namespace) -> Path:
    import mujoco

    input_dataset_dir = args.dataset_dir
    if args.views < 1:
        raise ValueError("--views must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    if not args.mujoco_xml.is_file():
        raise FileNotFoundError(args.mujoco_xml)
    if not args.save_png and not args.save_npz:
        raise ValueError("At least one of --save-png or --save-npz must be enabled")

    frames = discover_frames(args)
    if not frames:
        raise RuntimeError(f"No frames found under {args.dataset_dir}")

    dummy_image_shape = (args.height, args.width, 3)
    camera_args = argparse.Namespace(
        json=frames[0][1],
        camera_settings=args.camera_settings,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        fallback_focal=args.fallback_focal,
    )
    camera_matrix = load_camera_matrix(camera_args, dummy_image_shape)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, data = compile_model(args, camera_matrix)
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_render_camera"
    )
    if camera_id < 0:
        raise RuntimeError("zed_render_camera was not added to the MuJoCo model")

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[2] = 1
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    records = []
    try:
        for ordinal, (index, json_path) in enumerate(frames, start=1):
            print(f"[{ordinal}/{len(frames)}] render frame {index:06d}")
            records.append(
                render_frame(
                    args,
                    model,
                    data,
                    renderer,
                    scene_option,
                    camera_id,
                    camera_matrix,
                    index,
                    json_path,
                )
            )
    finally:
        renderer.close()

    manifest = {
        "dataset_dir_input": str(input_dataset_dir),
        "dataset_dir": str(args.dataset_dir),
        "mujoco_xml": str(args.mujoco_xml),
        "output_dir": str(args.output_dir),
        "frame_count": len(records),
        "views": args.views,
        "image_width": args.width,
        "image_height": args.height,
        "camera_matrix": camera_matrix.tolist(),
        "frames": [
            {
                "frame_index": record["frame_index"],
                "metadata": str(
                    args.output_dir / f"{record['frame_index']:06d}" / "metadata.json"
                ),
            }
            for record in records
        ],
    }
    manifest_name = (
        "manifest.json"
        if args.num_shards == 1
        else f"manifest_shard_{args.shard_index:02d}.json"
    )
    (args.output_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    return args.output_dir


def main() -> None:
    output_dir = process(parse_args())
    print(f"Saved pre-rendered views to {output_dir}")


if __name__ == "__main__":
    main()
