"""Public result contracts, provenance, and paper-facing aggregation."""

from .paper_metrics import (
    continuous_auc_under_threshold,
    summarize_paper_pose_errors,
    summarize_refinement_add,
)
from .provenance import ExperimentProvenance
from .figure_grid import GridLayout, build_grid, center_crop_resize
from .selection import (
    DroidFigure3Candidate,
    CtrnetxQualitativeCase,
    QualitativeFigureManifest,
    QualitativeItem,
    RefinementRound,
    make_droid_figure3_manifest,
    select_best_successful_round,
    select_ctrnetx_cases,
)
from .tables import TableColumn, render_latex_table, render_markdown_table
from .records import (
    load_top_level_frame_records,
    refinement_add_errors_m,
    summarize_refinement_records,
)

__all__ = [
    "CtrnetxQualitativeCase",
    "DroidFigure3Candidate",
    "ExperimentProvenance",
    "GridLayout",
    "QualitativeFigureManifest",
    "QualitativeItem",
    "RefinementRound",
    "TableColumn",
    "build_grid",
    "center_crop_resize",
    "continuous_auc_under_threshold",
    "load_top_level_frame_records",
    "make_droid_figure3_manifest",
    "select_best_successful_round",
    "select_ctrnetx_cases",
    "summarize_paper_pose_errors",
    "summarize_refinement_add",
    "render_latex_table",
    "render_markdown_table",
    "refinement_add_errors_m",
    "summarize_refinement_records",
]
