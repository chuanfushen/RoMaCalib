from __future__ import annotations

from calibx.reporting.tables import TableColumn, render_latex_table, render_markdown_table


def test_compact_table_formats_fractions_as_percentages() -> None:
    columns = [TableColumn("auc", "AUC", precision=2, percentage=True)]
    rows = [{"method": "Calib-X", "auc": 0.85559}]
    assert "85.56\\%" in render_latex_table(rows, columns)
    assert "85.56%" in render_markdown_table(rows, columns)


def test_latex_renderer_escapes_method_names() -> None:
    result = render_latex_table(
        [{"method": "RoMa_v2", "auc": 0.1}],
        [TableColumn("auc", "AUC")],
    )
    assert "RoMa\\_v2" in result
