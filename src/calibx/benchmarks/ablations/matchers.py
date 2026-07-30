"""Optional feature-matcher ablation contract used for Table 5 recovery.

This module records the execution shape audited from the historical CF
launcher.  It deliberately does not embed its private dependencies, weights,
paths, or numerical outputs.  In particular, a run satisfying this contract
is a candidate reproduction, not evidence that the paper values have been
recovered.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ..common import FrameManifest, assert_paired_manifests


TABLE5_MATCHERS = ("romav2", "romav1", "lightglue", "mast3r")
TABLE5_REQUESTED_FRAMES = 300
TABLE5_GATE_ID = "gate-v1"
TABLE5_SHARD_COUNT = 2
TABLE5_FRAMES_PER_SHARD = 150
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class MatcherAblationCondition:
    """One optional matcher run on the fixed Panda-ORB frame set."""

    matcher: str
    manifest: FrameManifest
    views: int = 6
    refinement_iterations: int = 3
    gate_id: str = TABLE5_GATE_ID
    mask_input: bool = True
    shard_count: int = TABLE5_SHARD_COUNT
    frames_per_shard: int = TABLE5_FRAMES_PER_SHARD
    dependency_requirement: str = "external-installation-required"

    def validate(self) -> None:
        if self.matcher not in TABLE5_MATCHERS:
            raise ValueError(f"unsupported Table 5 matcher: {self.matcher}")
        self.manifest.validate()
        if len(self.manifest.frame_ids) != TABLE5_REQUESTED_FRAMES:
            raise ValueError(f"Table 5 requires {TABLE5_REQUESTED_FRAMES} frames")
        if self.views != 6 or self.refinement_iterations != 3:
            raise ValueError("Table 5 fixes six views and three refinement iterations")
        if self.gate_id != TABLE5_GATE_ID:
            raise ValueError(f"Table 5 requires {TABLE5_GATE_ID}")
        if not self.mask_input:
            raise ValueError("Table 5 requires the audited SAM input mask")
        if (
            self.shard_count != TABLE5_SHARD_COUNT
            or self.frames_per_shard != TABLE5_FRAMES_PER_SHARD
        ):
            raise ValueError("Table 5 requires two 150-frame shards")
        if self.shard_count * self.frames_per_shard != TABLE5_REQUESTED_FRAMES:
            raise ValueError("Table 5 shard scope must cover all requested frames")
        if not self.dependency_requirement:
            raise ValueError("dependency_requirement must be declared")


@dataclass(frozen=True)
class MatcherAblationProvenance:
    """Path-neutral identity record for one completed matcher comparison.

    ``parent_manifest_sha256`` identifies the full 300-frame selection.  The
    two child hashes identify the submitted 150-frame shards.  A consumer must
    compare these hashes across *all* matcher outputs before aggregating them.
    The record does not make a numerical paper-value claim.
    """

    parent_manifest_sha256: str
    shard_manifest_sha256: tuple[str, ...]
    aggregation: str = "merge_per_frame_records_before_metrics"

    def validate(self) -> None:
        if not _SHA256_RE.fullmatch(self.parent_manifest_sha256):
            raise ValueError("parent_manifest_sha256 must be a lowercase SHA-256")
        if len(self.shard_manifest_sha256) != TABLE5_SHARD_COUNT:
            raise ValueError("Table 5 provenance requires two shard hashes")
        if any(not _SHA256_RE.fullmatch(value) for value in self.shard_manifest_sha256):
            raise ValueError("shard_manifest_sha256 must contain lowercase SHA-256 values")
        if self.aggregation != "merge_per_frame_records_before_metrics":
            raise ValueError("Table 5 must merge per-frame records before metrics")


def matcher_ablation_provenance(manifest: FrameManifest) -> MatcherAblationProvenance:
    """Produce the generic shared-manifest evidence required by Table 5.

    The returned values are derived only from a caller-provided public frame
    manifest; no historical CF hash is embedded in the source tree.
    """

    condition = MatcherAblationCondition("romav2", manifest)
    condition.validate()
    shards = tuple(
        manifest.shard(shard_index=index, shard_count=TABLE5_SHARD_COUNT)
        for index in range(TABLE5_SHARD_COUNT)
    )
    if any(len(shard.frame_ids) != TABLE5_FRAMES_PER_SHARD for shard in shards):
        raise AssertionError("validated Table 5 sharding must produce 150-frame shards")
    provenance = MatcherAblationProvenance(
        parent_manifest_sha256=manifest.sha256(),
        shard_manifest_sha256=tuple(shard.sha256() for shard in shards),
    )
    provenance.validate()
    return provenance


def assert_shared_matcher_provenance(
    baseline: MatcherAblationProvenance,
    candidate: MatcherAblationProvenance,
) -> None:
    """Require completed matcher outputs to identify the same two shards."""

    baseline.validate()
    candidate.validate()
    if baseline.parent_manifest_sha256 != candidate.parent_manifest_sha256:
        raise ValueError("matcher outputs must share the parent manifest SHA-256")
    if baseline.shard_manifest_sha256 != candidate.shard_manifest_sha256:
        raise ValueError("matcher outputs must share the two shard manifest SHA-256 values")


def assert_matcher_comparable(
    baseline: MatcherAblationCondition,
    candidate: MatcherAblationCondition,
) -> None:
    """Require an exact frame set before comparing optional matcher outputs."""
    baseline.validate()
    candidate.validate()
    if baseline.views != candidate.views or (
        baseline.refinement_iterations != candidate.refinement_iterations
    ):
        raise ValueError("matcher runs must share views and refinement iterations")
    if (
        baseline.gate_id != candidate.gate_id
        or baseline.mask_input != candidate.mask_input
        or baseline.shard_count != candidate.shard_count
        or baseline.frames_per_shard != candidate.frames_per_shard
    ):
        raise ValueError("matcher runs must share gate, mask, and shard protocol")
    assert_paired_manifests(baseline.manifest, candidate.manifest)


__all__ = [
    "MatcherAblationCondition",
    "MatcherAblationProvenance",
    "TABLE5_FRAMES_PER_SHARD",
    "TABLE5_GATE_ID",
    "TABLE5_MATCHERS",
    "TABLE5_REQUESTED_FRAMES",
    "TABLE5_SHARD_COUNT",
    "assert_matcher_comparable",
    "assert_shared_matcher_provenance",
    "matcher_ablation_provenance",
]
