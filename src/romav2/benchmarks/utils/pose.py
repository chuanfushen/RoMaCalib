"""Compatibility exports for the legacy single-frame DREAM entry point."""

from __future__ import annotations

import argparse
from copy import copy as _copy
from pathlib import Path

import torch

from calibx.pose import *  # noqa: F401,F403
from calibx.pose import process as _calibx_process
from calibx.rendering import load_joint_positions as load_joint_positions

DEFAULT_FRAME_JSON = PROJECT_ROOT / "data" / "dream" / "000000.json"
DEFAULT_FRAME_IMAGE = DEFAULT_FRAME_JSON.with_suffix(".rgb.jpg")
DEFAULT_MJCF = (
    PROJECT_ROOT
    / "assets/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/dream_mujoco_match_eval"


def parse_args() -> argparse.Namespace:
    """Parse the historical ``romav2`` single-frame command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_FRAME_IMAGE)
    parser.add_argument("--json", type=Path, default=DEFAULT_FRAME_JSON)
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--views", "-x", type=int, default=6)
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Render width. Defaults to input image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Render height. Defaults to input image height.",
    )
    parser.add_argument("--distance-scale", type=float, default=2.8)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--azimuth-offset", type=float, default=0.0)
    parser.add_argument("--matcher", default="RoMaV2", choices=("RoMaV2",))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--detect-threshold", type=float, default=0.005)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--score-filter", type=float, default=0.0)
    parser.add_argument("--ransac-method", default=DEFAULT_RANSAC_METHOD)
    parser.add_argument(
        "--ransac-threshold", type=float, default=DEFAULT_RANSAC_REPROJ_THRESHOLD
    )
    parser.add_argument(
        "--ransac-confidence", type=float, default=DEFAULT_RANSAC_CONFIDENCE
    )
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
    parser.add_argument(
        "--mask-prompt",
        default="robotic arm",
        help="Text prompt passed to SAM3 when --mask-input is enabled.",
    )
    parser.add_argument("--camera-settings", type=Path, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--fallback-focal", type=float, default=400.0)
    parser.add_argument(
        "--save-all-matches", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def process(args: argparse.Namespace) -> Path:
    """Run the Calib-X implementation with a legacy argument namespace."""
    if not hasattr(args, "sam3_checkpoint"):
        args = _copy(args)
        args.sam3_checkpoint = None
    return _calibx_process(args)


def main() -> None:
    output_dir = process(parse_args())
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
