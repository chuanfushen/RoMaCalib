"""Render public compact table JSON as LaTeX and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibx.reporting.tables import TableColumn, render_latex_table, render_markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    columns = [TableColumn(**column) for column in payload["columns"]]
    rows = payload["rows"]
    row_label_key = payload.get("row_label_key", "method")
    row_label = payload.get("row_label", "Method")
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.write_text(
        render_latex_table(rows, columns, row_label_key=row_label_key, row_label=row_label),
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        render_markdown_table(rows, columns, row_label_key=row_label_key, row_label=row_label),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
