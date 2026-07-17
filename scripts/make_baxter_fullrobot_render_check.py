#!/usr/bin/env python3
"""Build deterministic real/orbit-view contact sheets for Baxter render checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prerender-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=512)
    return parser.parse_args()


def labeled_tile(image_path: Path, label: str, tile_width: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    tile_height = round(image.height * tile_width / image.width)
    image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    label_height = 36
    tile = Image.new("RGB", (tile_width, tile_height + label_height), "white")
    tile.paste(image, (0, label_height))
    draw = ImageDraw.Draw(tile)
    draw.text((10, 9), label, fill="black", font=ImageFont.load_default())
    return tile


def make_sheet(args: argparse.Namespace, frame_index: int) -> Image.Image:
    stem = f"{frame_index:06d}"
    tiles = [
        labeled_tile(args.dataset_dir / f"{stem}.rgb.jpg", f"real frame {stem}", args.tile_width)
    ]
    for view_index in range(args.views):
        azimuth = view_index * 360.0 / args.views
        tiles.append(
            labeled_tile(
                args.prerender_dir / stem / f"view_{view_index:02d}.png",
                f"sim view_{view_index:02d}, az={azimuth:.0f} deg",
                args.tile_width,
            )
        )

    columns = 4
    rows = 2
    cell_width = max(tile.width for tile in tiles)
    cell_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (235, 235, 235))
    for index, tile in enumerate(tiles):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(tile, (x, y))
    return sheet


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for frame_index in args.frame_indices:
        sheet = make_sheet(args, frame_index)
        output_path = args.output_dir / f"frame_{frame_index:06d}_real_vs_6views.jpg"
        sheet.save(output_path, quality=94, subsampling=0)
        sheets.append(sheet)

    overview_width = max(sheet.width for sheet in sheets)
    overview_height = sum(sheet.height for sheet in sheets)
    overview = Image.new("RGB", (overview_width, overview_height), "white")
    offset = 0
    for sheet in sheets:
        overview.paste(sheet, (0, offset))
        offset += sheet.height
    overview.save(
        args.output_dir / "smoke5_real_vs_6views_overview.jpg",
        quality=92,
        subsampling=0,
    )
    print(f"Saved {len(sheets)} frame sheets and one overview to {args.output_dir}")


if __name__ == "__main__":
    main()
