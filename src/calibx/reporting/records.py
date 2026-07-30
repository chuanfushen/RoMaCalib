"""Extract paper metric inputs from saved frame-level result records."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from .paper_metrics import summarize_refinement_add


def load_top_level_frame_records(run_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    """Load direct child frame summaries, avoiding nested iteration artifacts."""
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*/frame_summary.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise RuntimeError("no top-level frame_summary.json files were found")
    return records


def refinement_add_errors_m(records: Iterable[dict[str, Any]]) -> list[float]:
    """Return ADD errors for successful top-level records only."""
    values: list[float] = []
    for record in records:
        if record.get("status") != "success":
            continue
        try:
            value = record["keypoint_metrics"]["summary"]["keypoint_add_mean_m"]
        except KeyError as error:
            raise ValueError("successful frame record lacks keypoint ADD summary") from error
        values.append(float(value))
    return values


def summarize_refinement_records(
    records: Iterable[dict[str, Any]],
    *,
    requested_frames: int | None = None,
) -> dict[str, float | int | str]:
    """Aggregate frame records with the d7/Table 4 all-frame AUC contract."""
    materialized = list(records)
    if requested_frames is not None and requested_frames < len(materialized):
        raise ValueError(
            "requested_frames cannot be smaller than the supplied frame records"
        )
    denominator = len(materialized) if requested_frames is None else requested_frames
    return summarize_refinement_add(
        refinement_add_errors_m(materialized),
        requested_frames=denominator,
    )


__all__ = [
    "load_top_level_frame_records",
    "refinement_add_errors_m",
    "summarize_refinement_records",
]
