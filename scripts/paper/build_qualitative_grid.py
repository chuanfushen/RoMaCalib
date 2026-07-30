"""Build a qualitative figure grid from a path-independent layout JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from calibx.reporting.figure_grid import GridLayout, build_grid


def _resolve_asset(asset_root: Path, key: str) -> Path:
    candidate = (asset_root / key).resolve()
    root = asset_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"asset key escapes --asset-root: {key}")
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.layout.read_text(encoding="utf-8"))
    layout = GridLayout(**payload["layout"])
    rows = []
    for asset_row in payload["rows"]:
        rows.append(
            [
                Image.open(_resolve_asset(args.asset_root, str(asset_key))).convert("RGB")
                for asset_key in asset_row
            ]
        )
    image = build_grid(
        rows,
        layout,
        resize_mode=payload.get("resize_mode", "center_crop"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Wrote {layout.rows}x{layout.columns} qualitative grid to {args.output}")


if __name__ == "__main__":
    main()
