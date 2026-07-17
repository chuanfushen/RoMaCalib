#!/usr/bin/env python3
"""Analyze Baxter real calibration failures against existing Franka results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


FRANKA_SUMMARIES = {
    "panda_azure": Path("outputs/dream_mujoco_match_eval_batch/panda-3cam_azure_sample300/summary.json"),
    "panda_kinect360": Path("outputs/dream_mujoco_match_eval_batch/panda-3cam_kinect360_sample300/summary.json"),
    "panda_realsense": Path("outputs/dream_mujoco_match_eval_batch/panda-3cam_realsense_sample300/summary.json"),
    "panda_synth_dr": Path("outputs/dream_mujoco_match_eval_batch/panda_synth_test_dr_sample300/summary.json"),
}


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = rankdata(np.asarray(x)), rankdata(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def group_bin(records: list[dict], field: str, edges: list[float]) -> list[dict]:
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        subset = [
            record
            for record in records
            if lower <= float(record[field]) < upper
        ]
        if not subset:
            continue
        errors = [record["add_mm"] for record in subset]
        rows.append(
            {
                "range": [lower, upper],
                "count": len(subset),
                "add_median_mm": float(np.median(errors)),
                "add_mean_mm": float(np.mean(errors)),
                "add_at_100mm_percent": 100.0 * sum(error < 100 for error in errors) / len(errors),
                "catastrophic_over_400mm_percent": 100.0
                * sum(error > 400 for error in errors)
                / len(errors),
            }
        )
    return rows


def pairwise_mask_iou(masks: list[np.ndarray]) -> float | None:
    values = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            union = np.logical_or(masks[first], masks[second]).sum()
            if union:
                values.append(float(np.logical_and(masks[first], masks[second]).sum() / union))
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "outputs/dream_baxter_real_leftarm_nogripper_2048_full100_corrected_ee_match_eval/"
            "baxter_real_dream_sample100/summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/baxter_real_full100_failure_analysis.json"),
    )
    args = parser.parse_args()

    batch = json.loads(args.summary.read_text())
    records: list[dict] = []
    failures: list[dict] = []
    poses: dict[int, list[dict]] = defaultdict(list)

    for frame in batch["frames"]:
        index = int(frame["frame_index"])
        annotation = json.loads(Path(frame["json"]).read_text())
        pose_index = int(annotation["source_metadata"]["pose_index"])
        mask_info = frame.get("input_mask", {})
        mask_path = Path(mask_info["mask_path"]) if mask_info.get("mask_path") else None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path and mask_path.exists() else None
        mask_bool = mask > 0 if mask is not None else None
        mask_coverage = float(mask_bool.mean()) if mask_bool is not None else 0.0

        common = {
            "frame": index,
            "pose": pose_index,
            "status": frame["status"],
            "mask_coverage": mask_coverage,
            "_mask": mask_bool,
        }
        if frame["status"] != "success":
            view_rows = frame.get("views", [])
            failures.append(
                {
                    **{key: value for key, value in common.items() if key != "_mask"},
                    "reason": frame.get("reason"),
                    "max_ransac_inliers": max(
                        (int(view.get("ransac_inliers", 0)) for view in view_rows), default=0
                    ),
                    "max_pnp_correspondences": max(
                        (int(view.get("pnp", {}).get("correspondences", 0)) for view in view_rows),
                        default=0,
                    ),
                }
            )
            poses[pose_index].append(common)
            continue

        metrics = frame["keypoint_metrics"]["summary"]
        pnp = frame["best_pnp"]
        correspondences = int(pnp["correspondences"])
        record = {
            **common,
            "add_mm": 1000.0 * float(metrics["keypoint_add_mean_m"]),
            "pixel_error_px": float(metrics["pixel_error_mean"]),
            "best_view": int(Path(frame["best_render_path"]).stem.split("_")[-1]),
            "correspondences": correspondences,
            "inliers": int(pnp["inliers"]),
            "inlier_ratio": float(pnp["inliers"]) / correspondences,
            "reprojection_px": float(pnp["inlier_reprojection_error_mean"]),
        }
        records.append(record)
        poses[pose_index].append(record)

    add = [record["add_mm"] for record in records]
    pixel = [record["pixel_error_px"] for record in records]
    view_counts = Counter(record["best_view"] for record in records)

    pose_rows = []
    for pose_index in sorted(poses):
        group = poses[pose_index]
        success = [record for record in group if record["status"] == "success"]
        pose_masks = [record["_mask"] for record in group if record["_mask"] is not None]
        errors = [record["add_mm"] for record in success]
        pose_rows.append(
            {
                "pose": pose_index,
                "frames": [record["frame"] for record in group],
                "success_count": len(success),
                "failure_count": len(group) - len(success),
                "add_median_mm": float(np.median(errors)) if errors else None,
                "add_mean_mm": float(np.mean(errors)) if errors else None,
                "add_min_mm": float(np.min(errors)) if errors else None,
                "add_max_mm": float(np.max(errors)) if errors else None,
                "add_range_mm": float(np.max(errors) - np.min(errors)) if errors else None,
                "add_at_100mm_count": sum(error < 100 for error in errors),
                "mask_coverage_mean": float(np.mean([record["mask_coverage"] for record in group])),
                "mask_pairwise_iou_mean": pairwise_mask_iou(pose_masks),
                "best_view_counts": dict(Counter(record["best_view"] for record in success)),
            }
        )

    correlation_fields = [
        "mask_coverage",
        "correspondences",
        "inliers",
        "inlier_ratio",
        "reprojection_px",
    ]
    correlations = {
        field: {
            "spearman_with_add": spearman(
                [record[field] for record in records], add
            ),
            "spearman_with_pixel": spearman(
                [record[field] for record in records], pixel
            ),
        }
        for field in correlation_fields
    }

    franka = {}
    for name, path in FRANKA_SUMMARIES.items():
        if not path.exists():
            continue
        summary = json.loads(path.read_text())["summary"]
        franka[name] = {
            key: summary[key]
            for key in [
                "n_frames",
                "n_success",
                "success_rate",
                "ADD/mean",
                "ADD/median",
                "pixel_error/mean",
                "pixel_error/median",
                "pnp_inliers/mean",
                "pnp_reprojection_error/mean",
            ]
            if key in summary
        }

    result = {
        "source_summary": args.summary.as_posix(),
        "metric_note": (
            "Baxter ADD is the official CtRNet Baxter end-effector single-point 3D error; "
            "Franka DREAM ADD aggregates multiple annotated robot keypoints."
        ),
        "baxter": {
            "frame_count": len(batch["frames"]),
            "success_count": len(records),
            "failure_count": len(failures),
            "add_success_only_mm": quantiles(add),
            "pixel_success_only_px": quantiles(pixel),
            "official_denominator_100": {
                "PCK@50px_percent": 100.0
                * sum(record["pixel_error_px"] < 50 for record in records)
                / len(batch["frames"]),
                "ADD@100mm_percent": 100.0
                * sum(record["add_mm"] < 100 for record in records)
                / len(batch["frames"]),
                "failure_percent": 100.0 * len(failures) / len(batch["frames"]),
                "over_200mm_percent": 100.0
                * sum(record["add_mm"] > 200 for record in records)
                / len(batch["frames"]),
                "over_400mm_percent": 100.0
                * sum(record["add_mm"] > 400 for record in records)
                / len(batch["frames"]),
            },
            "best_view_counts_success_only": dict(sorted(view_counts.items())),
            "correlations_success_only": correlations,
            "by_inliers": group_bin(records, "inliers", [0, 100, 150, 200, 250, 300, 10000]),
            "by_inlier_ratio": group_bin(records, "inlier_ratio", [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]),
            "by_reprojection_px": group_bin(records, "reprojection_px", [0, 1.5, 2.0, 2.5, 3.0, 10]),
            "by_mask_coverage": group_bin(records, "mask_coverage", [0, 0.02, 0.04, 0.06, 0.08, 0.12, 1.01]),
            "failures": failures,
            "poses": pose_rows,
            "worst_successes": sorted(
                [{key: value for key, value in record.items() if key != "_mask"} for record in records],
                key=lambda record: record["add_mm"],
                reverse=True,
            )[:15],
        },
        "franka_existing_results": franka,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
