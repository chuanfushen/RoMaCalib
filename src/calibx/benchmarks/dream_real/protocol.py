"""Protocol definition for full DREAM-real reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DREAM_REAL_CAMERAS = ("azure", "kinect360", "orb", "realsense")
DREAM_REAL_FULL_FRAME_COUNTS = {
    "azure": 6_394,
    "kinect360": 4_966,
    "orb": 32_315,
    "realsense": 5_944,
}


def assert_full_dream_real_scope(camera_frame_counts: Mapping[str, int]) -> None:
    """Validate the four-camera CF full-run scope, without asserting results."""
    normalized = {str(camera): int(count) for camera, count in camera_frame_counts.items()}
    if normalized != DREAM_REAL_FULL_FRAME_COUNTS:
        raise ValueError(
            "DREAM-real full scope must use the archived per-camera requested "
            f"frame counts: {DREAM_REAL_FULL_FRAME_COUNTS}"
        )


def is_full_dream_real_scope(camera_frame_counts: Mapping[str, int]) -> bool:
    """Return whether declared counts exactly match the four-camera paper scope."""
    try:
        assert_full_dream_real_scope(camera_frame_counts)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class DreamRealFullProtocol:
    """Frozen CF full-run settings shared by all four DREAM-real cameras."""

    camera: str
    views: int = 6
    refinement_iterations: int = 3
    mask_input: bool = True
    match_batch_size: int = 6

    def validate(self) -> None:
        if self.camera not in DREAM_REAL_CAMERAS:
            raise ValueError(f"unknown DREAM-real camera: {self.camera}")
        if self.views != 6:
            raise ValueError("DREAM-real full protocol is frozen at six views")
        if self.refinement_iterations != 3:
            raise ValueError("DREAM-real full protocol is frozen at R3")
        if not self.mask_input:
            raise ValueError("DREAM-real full protocol requires SAM input masking")
        if self.match_batch_size != 6:
            raise ValueError("DREAM-real full protocol is frozen at match_batch_size=6")


__all__ = [
    "DREAM_REAL_CAMERAS",
    "DREAM_REAL_FULL_FRAME_COUNTS",
    "DreamRealFullProtocol",
    "assert_full_dream_real_scope",
    "is_full_dream_real_scope",
]
