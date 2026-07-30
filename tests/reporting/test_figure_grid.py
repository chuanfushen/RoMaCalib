from __future__ import annotations

from PIL import Image
import pytest

from calibx.reporting.figure_grid import GridLayout, build_grid, center_crop_resize


def test_center_crop_resize_preserves_target_aspect() -> None:
    source = Image.new("RGB", (12, 4), "red")
    result = center_crop_resize(source, 4, 4)
    assert result.size == (4, 4)


def test_grid_uses_expected_size_and_gutter() -> None:
    layout = GridLayout(rows=2, columns=3, cell_width=10, cell_height=8, gutter=2)
    rows = [[Image.new("RGB", (4, 4), "blue") for _ in range(3)] for _ in range(2)]
    result = build_grid(rows, layout)
    assert result.size == (34, 18)
    assert result.getpixel((11, 0)) == (255, 255, 255)


def test_grid_rejects_wrong_matrix_shape() -> None:
    with pytest.raises(ValueError, match="expected 2 rows"):
        build_grid([[Image.new("RGB", (1, 1), "black")]], GridLayout(2, 1, 1, 1))
