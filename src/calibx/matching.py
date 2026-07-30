"""Dense matching and geometric-consistency interfaces.

The initial main-branch implementation remains in :mod:`calibx.geometry`
during compatibility migration. Callers use this module so the implementation
can be replaced without coupling the pipeline to geometry internals.
"""

from .geometry import (
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ImcuiMatcher,
    imcui_keypoint_ransac_mask,
)

__all__ = [
    "DEFAULT_RANSAC_CONFIDENCE",
    "DEFAULT_RANSAC_MAX_ITER",
    "DEFAULT_RANSAC_METHOD",
    "DEFAULT_RANSAC_REPROJ_THRESHOLD",
    "ImcuiMatcher",
    "imcui_keypoint_ransac_mask",
]
