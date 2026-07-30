from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MJCF = (
    PROJECT_ROOT / "assets/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
)
DEFAULT_QPOS_JSON = PROJECT_ROOT / "data" / "dream" / "000000.json"
DEFAULT_OUTPUT_DIR = Path("outputs/panda_mujoco_views")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render multiple MuJoCo views around a Franka Panda pose loaded from a DREAM-style JSON file."
    )
    parser.add_argument(
        "--mjcf", type=Path, default=DEFAULT_MJCF, help="Path to the Panda MJCF XML."
    )
    parser.add_argument(
        "--qpos-json",
        type=Path,
        default=DEFAULT_QPOS_JSON,
        help="JSON file containing sim_state.joints entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered PNGs.",
    )
    parser.add_argument(
        "-x", "--views", type=int, default=6, help="Number of orbit views to render."
    )
    parser.add_argument("--width", type=int, default=640, help="Rendered image width.")
    parser.add_argument(
        "--height", type=int, default=480, help="Rendered image height."
    )
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=2.8,
        help="Camera distance as a multiple of arm radius.",
    )
    parser.add_argument(
        "--min-distance", type=float, default=1.2, help="Minimum camera distance."
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-20.0,
        help="Free-camera elevation in degrees.",
    )
    parser.add_argument(
        "--azimuth-offset", type=float, default=0.0, help="Azimuth offset in degrees."
    )
    parser.add_argument(
        "--hide-collision",
        action="store_true",
        default=True,
        help="Render visual geoms only.",
    )
    parser.add_argument(
        "--show-collision",
        dest="hide_collision",
        action="store_false",
        help="Render MuJoCo default visible geom groups, including collision geoms if enabled by the model.",
    )
    return parser.parse_args()


def load_joint_positions(json_path: Path) -> dict[str, float]:
    payload = json.loads(json_path.read_text())
    try:
        joints = payload["sim_state"]["joints"]
    except KeyError as exc:
        raise KeyError(f"{json_path} does not contain sim_state.joints") from exc
    return {joint["name"]: float(joint["position"]) for joint in joints}


def json_joint_name_to_mjcf(name: str) -> str:
    name = name.rsplit("/", 1)[-1]
    if name.startswith("panda_finger_joint"):
        return name.removeprefix("panda_")
    if name.startswith("panda_joint"):
        return name.removeprefix("panda_")
    return name


def apply_json_qpos(model, data, json_positions: dict[str, float]) -> dict[str, float]:
    import mujoco

    applied = {}
    for json_name, position in json_positions.items():
        joint_name = json_joint_name_to_mjcf(json_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_address = model.jnt_qposadr[joint_id]
        qpos_width = (
            model.jnt_qposadr[joint_id + 1] - qpos_address
            if joint_id + 1 < model.njnt
            else model.nq - qpos_address
        )
        if qpos_width != 1:
            raise ValueError(
                f"Joint {joint_name!r} has qpos width {qpos_width}; this script expects scalar joints."
            )
        data.qpos[qpos_address] = position
        applied[joint_name] = position

    if not applied:
        raise ValueError("No JSON joints matched MJCF joints.")
    return applied


def visual_bounds(model, data) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    points = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[geom_id] != 2:
            continue
        points.append(np.asarray(data.geom_xpos[geom_id], dtype=np.float64))

    if not points:
        points = [
            np.asarray(data.xpos[body_id], dtype=np.float64)
            for body_id in range(1, model.nbody)
        ]

    stacked = np.stack(points, axis=0)
    padding = np.array([0.25, 0.25, 0.25], dtype=np.float64)
    return stacked.min(axis=0) - padding, stacked.max(axis=0) + padding


def compile_model(mjcf_path: Path, width: int, height: int):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)
    return spec.compile()


def render_views(args: argparse.Namespace) -> list[Path]:
    import mujoco

    if args.views < 1:
        raise ValueError("--views must be >= 1")
    if not args.mjcf.is_file():
        raise FileNotFoundError(args.mjcf)
    if not args.qpos_json.is_file():
        raise FileNotFoundError(args.qpos_json)

    model = compile_model(args.mjcf, args.width, args.height)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    applied_joints = apply_json_qpos(model, data, load_joint_positions(args.qpos_json))
    mujoco.mj_forward(model, data)

    minimum, maximum = visual_bounds(model, data)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * np.linalg.norm(maximum - minimum)
    distance = max(args.min_distance, args.distance_scale * radius)

    scene_option = mujoco.MjvOption()
    if args.hide_collision:
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[2] = 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    image_paths = []
    cameras = []
    try:
        for view_index in range(args.views):
            azimuth = args.azimuth_offset + view_index * 360.0 / args.views
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = center
            camera.distance = float(distance)
            camera.azimuth = float(azimuth)
            camera.elevation = float(args.elevation)

            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            rgb = renderer.render().copy()

            image_path = args.output_dir / f"view_{view_index:02d}.png"
            Image.fromarray(rgb).save(image_path)
            image_paths.append(image_path)
            cameras.append(
                {
                    "view_index": view_index,
                    "azimuth_degrees": float(math.fmod(azimuth, 360.0)),
                    "elevation_degrees": float(args.elevation),
                    "distance": float(distance),
                }
            )
    finally:
        renderer.close()

    metadata = {
        "mjcf": str(args.mjcf),
        "qpos_json": str(args.qpos_json),
        "output_dir": str(args.output_dir),
        "image_width": args.width,
        "image_height": args.height,
        "views": args.views,
        "center": center.tolist(),
        "radius": float(radius),
        "applied_joints": applied_joints,
        "cameras": cameras,
        "images": [str(path) for path in image_paths],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return image_paths


def main() -> None:
    image_paths = render_views(parse_args())
    print(f"Rendered {len(image_paths)} views:")
    for image_path in image_paths:
        print(image_path)


if __name__ == "__main__":
    main()
