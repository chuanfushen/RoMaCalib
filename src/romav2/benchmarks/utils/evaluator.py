"""Compatibility exports for the legacy DREAM batch-evaluation entry point."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from calibx.matching import (
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ImcuiMatcher,
)
from calibx.metrics import auc_under_threshold, summarize
from calibx.pipeline import (
    _clean_record as clean_record,
    _match_render_paths as match_render_paths,
    _match_romav2_images_batched as match_romav2_images_batched,
    _remove_frame_artifacts as remove_frame_artifacts,
    _remove_render_artifacts as remove_render_artifacts,
    process_frame,
)
from calibx.pose import (
    DEFAULT_MJCF,
    apply_input_mask,
    best_view_key,
    draw_keypoint_eval,
    dream_keypoints,
    fk_keypoints,
    keypoint_metrics,
    load_camera_matrix,
    load_dream_payload,
    make_model_and_data,
    make_sam3_extractor,
    match_one_render,
    render_orbit_views,
    save_pose_npz,
)
from calibx.runner import (
    _frame_index_from_json as frame_index_from_json,
    count_dream_frames,
    discover_frames,
    process,
    resolve_dataset_dir,
)

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data/dream/real/panda-3cam_azure/panda-3cam_azure"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/dream_mujoco_match_eval_batch"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the historical ``romav2`` evaluator command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prerender-dir",
        type=Path,
        default=None,
        help="Use pre-rendered views from prerender_dream_mujoco_views.py.",
    )
    parser.add_argument("--views", "-x", type=int, default=6)
    parser.add_argument(
        "--match-batch-size",
        type=int,
        default=None,
        help="RoMaV2 pairs per forward. Defaults to --views.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
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
    parser.add_argument("--sam3-checkpoint", type=Path, default=None)
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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--keep-renders", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-visualizations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-all-matches", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=0.1,
        help="ADD AUC upper threshold in meters.",
    )
    parser.add_argument(
        "--auc-delta",
        type=float,
        default=1e-4,
        help="ADD AUC integration step in meters.",
    )
    return parser.parse_args(argv)


def main() -> None:
    output_dir = process(parse_args())
    print(f"Saved batch evaluation to {output_dir}")


if __name__ == "__main__":
    main()
