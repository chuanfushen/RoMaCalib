"""Batch runner for DREAM-style Calib-X evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .matching import (
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ImcuiMatcher,
)
from .metrics import summarize
from .pipeline import process_frame
from .pose import DEFAULT_MJCF, make_sam3_extractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data/dream/real/panda-3cam_azure/panda-3cam_azure"
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
    """Resolve nested DREAM exports to the directory containing frames."""
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

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1],
            len(item[0].parts),
            str(item[0]),
        ),
    )
    return ranked[0][0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MJCF)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--prerender-dir",
        type=Path,
        default=None,
        help="Use pre-rendered views instead of rendering each frame.",
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
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--detect-threshold", type=float, default=0.005)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--score-filter", type=float, default=0.0)
    parser.add_argument("--ransac-method", default=DEFAULT_RANSAC_METHOD)
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=DEFAULT_RANSAC_REPROJ_THRESHOLD,
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=DEFAULT_RANSAC_CONFIDENCE,
    )
    parser.add_argument(
        "--ransac-max-iter",
        type=int,
        default=DEFAULT_RANSAC_MAX_ITER,
    )
    parser.add_argument("--pnp-threshold", type=float, default=5.0)
    parser.add_argument(
        "--min-pnp-correspondences",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--max-pnp-correspondences",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--mask-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask the observed input image with SAM3 before matching.",
    )
    parser.add_argument("--mask-prompt", default="robotic arm")
    parser.add_argument("--sam3-checkpoint", type=Path, default=None)
    parser.add_argument("--camera-settings", type=Path, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--fallback-focal", type=float, default=400.0)
    parser.add_argument(
        "--frame-indices",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=90)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-renders",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-all-matches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--auc-threshold", type=float, default=0.1)
    parser.add_argument("--auc-delta", type=float, default=1e-4)
    return parser.parse_args(argv)


def _frame_index_from_json(path: Path) -> int:
    return int(path.stem)


def discover_frames(
    args: argparse.Namespace,
) -> list[tuple[int, Path, Path]]:
    json_paths = sorted(
        path
        for path in args.dataset_dir.glob("*.json")
        if path.name != "_camera_settings.json" and path.stem.isdigit()
    )
    frames = []
    requested = (
        None
        if args.frame_indices is None
        else {int(index) for index in args.frame_indices}
    )
    for json_path in json_paths:
        index = _frame_index_from_json(json_path)
        if requested is not None and index not in requested:
            continue
        if args.start_index is not None and index < args.start_index:
            continue
        if args.end_index is not None and index > args.end_index:
            continue
        if args.stride > 1 and index % args.stride != 0:
            continue
        image_path = json_path.with_suffix(".rgb.jpg")
        if image_path.is_file():
            frames.append((index, json_path, image_path))

    if args.sample_count is not None:
        if args.sample_count < 1:
            raise ValueError("--sample-count must be >= 1")
        if args.sample_count > len(frames):
            raise ValueError(
                f"--sample-count={args.sample_count} exceeds "
                f"available frames {len(frames)}"
            )
        rng = np.random.default_rng(args.sample_seed)
        sampled = rng.choice(
            len(frames),
            size=args.sample_count,
            replace=False,
        )
        frames = [frames[int(index)] for index in sorted(sampled)]
    if args.limit is not None:
        frames = frames[: args.limit]
    return frames


def process(
    args: argparse.Namespace,
    matcher_api: ImcuiMatcher | None = None,
) -> Path:
    """Run a deterministic batch and persist frame and aggregate records."""
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
    mask_extractor = make_sam3_extractor(
        args.mask_input,
        args.sam3_checkpoint,
    )
    records = []
    for ordinal, (
        index,
        json_path,
        image_path,
    ) in enumerate(frames, start=1):
        print(f"[{ordinal}/{len(frames)}] frame {index:06d}")
        record = process_frame(
            args,
            matcher_api,
            index,
            json_path,
            image_path,
            mask_extractor,
        )
        records.append(record)
        running = summarize(
            records,
            args.auc_threshold,
            args.auc_delta,
        )
        (args.output_dir / "summary_running.json").write_text(
            json.dumps(running, indent=2) + "\n"
        )

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
            "source": "SAM3 via calibx.masking.Sam3Extractor.",
        },
        "frame_count": len(frames),
        "summary": summary,
        "frames": records,
        "metric_note": {
            "ADD/mean": (
                "Mean of per-frame mean 3D FK keypoint error in camera coordinates."
            ),
            "ADD/AUC": (
                "Area under the accuracy-threshold curve over per-frame ADD values."
            ),
            "pixel_error/mean": (
                "Mean of per-frame mean 2D FK keypoint reprojection error."
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    return args.output_dir


def main() -> None:
    output_dir = process(parse_args())
    print(f"Saved batch evaluation to {output_dir}")


if __name__ == "__main__":
    main()
