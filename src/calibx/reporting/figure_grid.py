"""Path-independent Pillow utilities for paper figure layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image


@dataclass(frozen=True)
class GridLayout:
    """A fixed cell layout for qualitative paper figures."""

    rows: int
    columns: int
    cell_width: int
    cell_height: int
    gutter: int = 4

    def validate(self) -> None:
        if self.rows < 1 or self.columns < 1:
            raise ValueError("rows and columns must be positive")
        if self.cell_width < 1 or self.cell_height < 1:
            raise ValueError("cell dimensions must be positive")
        if self.gutter < 0:
            raise ValueError("gutter must be non-negative")

    @property
    def size(self) -> tuple[int, int]:
        self.validate()
        return (
            self.columns * self.cell_width + (self.columns - 1) * self.gutter,
            self.rows * self.cell_height + (self.rows - 1) * self.gutter,
        )


def center_crop_resize(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    """Center-crop to the requested aspect ratio, then resize with LANCZOS."""
    if width < 1 or height < 1:
        raise ValueError("target dimensions must be positive")
    source_width, source_height = image.size
    if source_width < 1 or source_height < 1:
        raise ValueError("source image dimensions must be positive")
    source_ratio = source_width / source_height
    target_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = int(round(source_height * target_ratio))
        left = (source_width - crop_width) // 2
        box = (left, 0, left + crop_width, source_height)
    else:
        crop_height = int(round(source_width / target_ratio))
        top = (source_height - crop_height) // 2
        box = (0, top, source_width, top + crop_height)
    return image.crop(box).resize((width, height), Image.Resampling.LANCZOS)


def build_grid(
    rows: list[list[Image.Image]],
    layout: GridLayout,
    *,
    resize_mode: Literal["center_crop", "stretch"] = "center_crop",
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Compose already-prepared assets without accessing source directories."""
    layout.validate()
    if len(rows) != layout.rows:
        raise ValueError(f"expected {layout.rows} rows, got {len(rows)}")
    if any(len(row) != layout.columns for row in rows):
        raise ValueError(f"every grid row must contain {layout.columns} images")
    canvas = Image.new("RGB", layout.size, background)
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            source = image.convert("RGB")
            if resize_mode == "center_crop":
                cell = center_crop_resize(source, layout.cell_width, layout.cell_height)
            elif resize_mode == "stretch":
                cell = source.resize(
                    (layout.cell_width, layout.cell_height),
                    Image.Resampling.LANCZOS,
                )
            else:
                raise ValueError(f"unknown resize mode: {resize_mode}")
            x = column_index * (layout.cell_width + layout.gutter)
            y = row_index * (layout.cell_height + layout.gutter)
            canvas.paste(cell, (x, y))
    return canvas
