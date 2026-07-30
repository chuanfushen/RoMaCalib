"""Create the public Figure 3 selection manifest from normalized candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibx.reporting.selection import DroidFigure3Candidate, make_droid_figure3_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    records = payload["candidates"] if isinstance(payload, dict) else payload
    candidates = [DroidFigure3Candidate(**record) for record in records]
    manifest = make_droid_figure3_manifest(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write(args.output)
    print(f"Wrote {len(manifest.items)} qualitative selections to {args.output}")


if __name__ == "__main__":
    main()
