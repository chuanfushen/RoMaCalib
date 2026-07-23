#!/usr/bin/env python3
"""Summarize DROID IoU and create a CtRNet-X Figure-4-style comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


RAW_COLOR = np.asarray([238, 133, 54], dtype=np.float32)
OURS_COLOR = np.asarray([67, 126, 191], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--final-summary", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-count", type=int, default=5)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_path(mask_root: Path, record: dict[str, Any], name: str) -> Path:
    return (
        mask_root
        / "sessions"
        / str(record["session_id"])
        / "frames"
        / f"{int(record['frame_index']):06d}"
        / name
    )


def native_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float(intersection / union) if union else 0.0


def score_records(summary: dict[str, Any], mask_root: Path) -> list[dict[str, Any]]:
    scored = []
    for record in summary["records"]:
        row = dict(record)
        target_path = reference_path(mask_root, row, "sam3_mask.png")
        render_path = Path(str(row.get("render_mask", "")))
        if row.get("status") != "success" or not target_path.is_file() or not render_path.is_file():
            row["native_iou_1280x720"] = None
            scored.append(row)
            continue
        row["native_iou_1280x720"] = mask_iou(
            native_mask(render_path),
            native_mask(target_path),
        )
        scored.append(row)
    return scored


def summarize(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in records if row["native_iou_1280x720"] is not None]
    native = [
        float(row["metrics"]["iou"])
        for row in records
        if row.get("status") == "success" and "iou" in row.get("metrics", {})
    ]
    sessions = sorted({str(row["session_id"]) for row in evaluable})
    episodes = sorted({str(row["episode_uuid"]) for row in evaluable})
    session_means = [
        float(
            np.mean(
                [
                    row["native_iou_1280x720"]
                    for row in evaluable
                    if str(row["session_id"]) == session_id
                ]
            )
        )
        for session_id in sessions
    ]
    episode_means = [
        float(
            np.mean(
                [
                    row["native_iou_1280x720"]
                    for row in evaluable
                    if str(row["episode_uuid"]) == episode_uuid
                ]
            )
        )
        for episode_uuid in episodes
    ]
    return {
        "method": name,
        "requested_frames": len(records),
        "evaluable_frames": len(evaluable),
        "missing_or_failed_frames": len(records) - len(evaluable),
        "native_frame_iou_macro": float(np.mean(native)) if native else None,
        "native_1280x720_frame_iou_macro": float(
            np.mean([row["native_iou_1280x720"] for row in evaluable])
        ),
        "native_1280x720_session_iou_macro": float(np.mean(session_means)),
        "native_1280x720_episode_iou_macro": float(np.mean(episode_means)),
    }


def tint(image: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = 0.45 * result[mask] + 0.55 * color
    return np.rint(result).clip(0, 255).astype(np.uint8)


def choose_visuals(
    raw_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    count: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    raw_by_frame = {
        (str(row["session_id"]), int(row["frame_index"])): row
        for row in raw_records
        if row["native_iou_1280x720"] is not None
    }
    pairs = [
        (raw_by_frame[(str(row["session_id"]), int(row["frame_index"]))], row)
        for row in final_records
        if row["native_iou_1280x720"] is not None
        and (str(row["session_id"]), int(row["frame_index"])) in raw_by_frame
    ]
    episode_best = []
    for episode_uuid in sorted({str(final["episode_uuid"]) for _, final in pairs}):
        candidates = [pair for pair in pairs if str(pair[1]["episode_uuid"]) == episode_uuid]
        episode_best.append(
            max(
                candidates,
                key=lambda pair: (
                    float(pair[1]["native_iou_1280x720"]),
                    float(pair[1]["native_iou_1280x720"])
                    - float(pair[0]["native_iou_1280x720"]),
                ),
            )
        )
    return sorted(
        episode_best,
        key=lambda pair: float(pair[1]["native_iou_1280x720"]),
        reverse=True,
    )[:count]


def make_figure(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    mask_root: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    panel_size = (384, 216)
    left_margin, top_margin, row_gap = 176, 38, 8
    width = left_margin + panel_size[0] * len(pairs)
    height = top_margin + panel_size[1] * 3 + row_gap * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    labels = ("Real RGB", "DROID extrinsics", "RoMaCalib")
    manifest = []
    for column, (raw, final) in enumerate(pairs):
        real_path = reference_path(mask_root, final, "real.png")
        real = np.asarray(Image.open(real_path).convert("RGB"))
        raw_mask = np.asarray(Image.open(raw["render_mask"]).convert("L")) > 0
        final_mask = np.asarray(Image.open(final["render_mask"]).convert("L")) > 0
        images = (real, tint(real, raw_mask, RAW_COLOR), tint(real, final_mask, OURS_COLOR))
        x = left_margin + column * panel_size[0]
        draw.text((x + 8, 7), f"Episode {int(final['episode_rank']) + 1}", fill="black", font=font)
        for row, image in enumerate(images):
            y = top_margin + row * (panel_size[1] + row_gap)
            panel = ImageOps.fit(
                Image.fromarray(image),
                panel_size,
                method=Image.Resampling.LANCZOS,
            )
            canvas.paste(panel, (x, y))
        manifest.append(
            {
                "episode_rank": int(final["episode_rank"]),
                "episode_uuid": final["episode_uuid"],
                "session_id": final["session_id"],
                "frame_index": int(final["frame_index"]),
                "raw_iou_1280x720": float(raw["native_iou_1280x720"]),
                "final_iou_1280x720": float(final["native_iou_1280x720"]),
            }
        )
    for row, label in enumerate(labels):
        y = top_margin + row * (panel_size[1] + row_gap) + panel_size[1] // 2 - 10
        draw.text((12, y), label, fill="black", font=font)
    canvas.save(destination, dpi=(300, 300))
    canvas.save(destination.with_suffix(".pdf"), resolution=300)
    return manifest


def write_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = list(rows[0])
    with (output_dir / "droid_iou_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    latex_rows = "\n".join(
        f"{row['method']} & {row['native_1280x720_frame_iou_macro']:.4f} \\\\"
        for row in rows
    )
    (output_dir / "droid_iou_table.tex").write_text(
        "\\begin{tabular}{lc}\n\\toprule\nMethod & IoU $\\uparrow$ \\\\\n"
        "\\midrule\n"
        f"{latex_rows}\n"
        "\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = score_records(read_json(args.raw_summary), args.mask_root)
    final = score_records(read_json(args.final_summary), args.mask_root)
    table = [summarize("DROID provided extrinsics", raw), summarize("RoMaCalib", final)]
    write_table(table, args.output_dir)
    selected = choose_visuals(raw, final, args.figure_count)
    manifest = make_figure(selected, args.mask_root, args.output_dir / "droid_fig4_style.png")
    (args.output_dir / "droid_final_summary.json").write_text(
        json.dumps(
            {
                "metric": "frame-wise mask IoU at the native 1280x720 resolution",
                "table": table,
                "visual_selection": "best final-IoU frame per episode, then top episodes",
                "visuals": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
