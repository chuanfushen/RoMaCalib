"""Paired SAM foreground-mask ablation contract used by Table 4."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from ..common import FrameManifest, assert_paired_manifests
from ..gate_v1 import DETERMINISTIC_SEED


SAM_TABLE4_SPLITS = ("robot_in_view", "robot_in_and_out")
SAM_TABLE4_FRAMES_PER_SPLIT = 900


@dataclass(frozen=True)
class SamAblationCondition:
    """One side of a paired foreground-mask comparison."""

    split: Literal["robot_in_view", "robot_in_and_out"]
    manifest: FrameManifest
    mask_input: bool
    mask_prompt: str = "robotic arm"
    views: int = 6
    refinement_iterations: int = 3
    stage1_match_batch_size: int = 6
    refinement_match_batch_size: int = 1
    determinism_seed: int = DETERMINISTIC_SEED

    def validate(self) -> None:
        _validate_condition(self)


@lru_cache(maxsize=128)
def _validate_condition(condition: SamAblationCondition) -> None:
    """Validate a frozen condition once even when 900 runtime hooks inspect it.

    The public contract is immutable/hashable, so caching does not weaken a
    validation guarantee.  It avoids repeatedly scanning the same 900 opaque
    frame IDs on every Stage-0/R1--R3 request.
    """

    if condition.split not in SAM_TABLE4_SPLITS:
        raise ValueError(f"unknown SAM ablation split: {condition.split}")
    condition.manifest.validate()
    if not condition.mask_prompt:
        raise ValueError("SAM ablation mask_prompt must not be empty")
    if len(condition.manifest.frame_ids) != SAM_TABLE4_FRAMES_PER_SPLIT:
        raise ValueError(
            f"{condition.split} requires {SAM_TABLE4_FRAMES_PER_SPLIT} requested frames"
        )
    if condition.views != 6:
        raise ValueError("Table 4 uses exactly six candidate views")
    if condition.refinement_iterations != 3:
        raise ValueError("Table 4 uses exactly three refinement iterations")
    if condition.stage1_match_batch_size != 6:
        raise ValueError("Table 4 stage 1 uses match_batch_size=6")
    if condition.refinement_match_batch_size != 1:
        raise ValueError(
            "Table 4 pose-aligned refinement uses one-render match batches"
        )
    if condition.determinism_seed != DETERMINISTIC_SEED:
        raise ValueError(
            "Table 4 gate-v1 execution uses the frozen determinism seed "
            f"{DETERMINISTIC_SEED}"
        )


@dataclass(frozen=True)
class SamAblationPair:
    """A Table 4 comparison in which masking is the only allowed variable."""

    without_sam: SamAblationCondition
    with_sam: SamAblationCondition

    def validate(self) -> None:
        self.without_sam.validate()
        self.with_sam.validate()
        if self.without_sam.mask_input:
            raise ValueError("without_sam.mask_input must be false")
        if not self.with_sam.mask_input:
            raise ValueError("with_sam.mask_input must be true")
        if self.without_sam.split != self.with_sam.split:
            raise ValueError("SAM pair must compare the same dataset split")
        if (
            self.without_sam.views != self.with_sam.views
            or self.without_sam.mask_prompt != self.with_sam.mask_prompt
            or self.without_sam.refinement_iterations
            != self.with_sam.refinement_iterations
            or self.without_sam.stage1_match_batch_size
            != self.with_sam.stage1_match_batch_size
            or self.without_sam.refinement_match_batch_size
            != self.with_sam.refinement_match_batch_size
            or self.without_sam.determinism_seed != self.with_sam.determinism_seed
        ):
            raise ValueError("masking must be the only differing protocol setting")
        assert_paired_manifests(self.without_sam.manifest, self.with_sam.manifest)
        without_sha256 = self.without_sam.manifest.sha256()
        with_sha256 = self.with_sam.manifest.sha256()
        if without_sha256 != with_sha256:
            raise ValueError(
                "paired SAM ablations require identical complete manifest "
                "SHA-256 provenance"
            )


__all__ = [
    "SAM_TABLE4_FRAMES_PER_SPLIT",
    "SamAblationCondition",
    "SamAblationPair",
]
