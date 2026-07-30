"""Small dependency-free renderers for compact public paper table inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class TableColumn:
    key: str
    label: str
    precision: int = 3
    percentage: bool = False

    def format(self, value: Any) -> str:
        if value is None:
            return "--"
        if isinstance(value, (float, int)):
            numeric = float(value)
            if math.isnan(numeric):
                return "--"
            if self.percentage:
                numeric *= 100.0
            suffix = "\\%" if self.percentage else ""
            return f"{numeric:.{self.precision}f}{suffix}"
        return str(value)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def render_latex_table(
    rows: list[dict[str, Any]],
    columns: list[TableColumn],
    *,
    row_label_key: str = "method",
    row_label: str = "Method",
) -> str:
    """Render a standalone ``tabular`` body from compact metric dictionaries."""
    if not columns:
        raise ValueError("at least one metric column is required")
    headers = [row_label, *(column.label for column in columns)]
    alignment = "l" + "r" * len(columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(_latex_escape(header) for header in headers) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        if row_label_key not in row:
            raise ValueError(f"table row is missing {row_label_key!r}")
        cells = [_latex_escape(str(row[row_label_key]))]
        cells.extend(column.format(row.get(column.key)) for column in columns)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def render_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[TableColumn],
    *,
    row_label_key: str = "method",
    row_label: str = "Method",
) -> str:
    """Render the same compact source as a reviewable Markdown table."""
    if not columns:
        raise ValueError("at least one metric column is required")
    header = [row_label, *(column.label for column in columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        if row_label_key not in row:
            raise ValueError(f"table row is missing {row_label_key!r}")
        cells = [str(row[row_label_key])]
        cells.extend(
            column.format(row.get(column.key)).replace("\\%", "%")
            for column in columns
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["TableColumn", "render_latex_table", "render_markdown_table"]
