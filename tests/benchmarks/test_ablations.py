from __future__ import annotations

import pytest

from calibx.benchmarks.ablations.matchers import (
    MatcherAblationCondition,
    assert_matcher_comparable,
    assert_shared_matcher_provenance,
    matcher_ablation_provenance,
)
from calibx.benchmarks.ablations.sam import SamAblationCondition, SamAblationPair
from calibx.benchmarks.common import FrameManifest


def _manifest(protocol: str, count: int) -> FrameManifest:
    return FrameManifest(protocol, "ctrnetx", tuple(f"{index:04d}" for index in range(count)))


def test_sam_pair_allows_only_mask_setting_to_differ() -> None:
    manifest = _manifest("sam_pair", 900)
    without_sam = SamAblationCondition("robot_in_view", manifest, False)
    with_sam = SamAblationCondition("robot_in_view", manifest, True)
    SamAblationPair(without_sam, with_sam).validate()


def test_sam_pair_rejects_different_requested_frames() -> None:
    without_sam = SamAblationCondition("robot_in_view", _manifest("no_sam", 900), False)
    changed = FrameManifest("sam", "ctrnetx", tuple(f"x{index:04d}" for index in range(900)))
    with_sam = SamAblationCondition("robot_in_view", changed, True)
    with pytest.raises(ValueError, match="exactly the same"):
        SamAblationPair(without_sam, with_sam).validate()


def test_sam_pair_freezes_the_stage1_and_replay_match_batches() -> None:
    manifest = _manifest("sam", 900)
    with pytest.raises(ValueError, match="stage 1 uses match_batch_size=6"):
        SamAblationCondition(
            "robot_in_view",
            manifest,
            True,
            stage1_match_batch_size=1,
        ).validate()
    with pytest.raises(ValueError, match="one-render match batches"):
        SamAblationCondition(
            "robot_in_view",
            manifest,
            True,
            refinement_match_batch_size=6,
        ).validate()


def test_matcher_comparison_requires_same_manifest() -> None:
    romav2 = MatcherAblationCondition("romav2", _manifest("matcher", 300))
    mast3r = MatcherAblationCondition("mast3r", _manifest("matcher", 300))
    assert_matcher_comparable(romav2, mast3r)


def test_matcher_comparison_freezes_gate_mask_and_two_shards() -> None:
    manifest = _manifest("matcher", 300)
    with pytest.raises(ValueError, match="requires gate-v1"):
        MatcherAblationCondition("romav2", manifest, gate_id="legacy").validate()
    with pytest.raises(ValueError, match="requires the audited SAM input mask"):
        MatcherAblationCondition("romav2", manifest, mask_input=False).validate()
    with pytest.raises(ValueError, match="requires two 150-frame shards"):
        MatcherAblationCondition("romav2", manifest, shard_count=3).validate()


def test_matcher_provenance_derives_two_public_shard_hashes() -> None:
    manifest = _manifest("matcher", 300)
    provenance = matcher_ablation_provenance(manifest)
    provenance.validate()
    assert provenance.parent_manifest_sha256 == manifest.sha256()
    assert len(provenance.shard_manifest_sha256) == 2
    assert provenance.shard_manifest_sha256[0] != provenance.shard_manifest_sha256[1]
    assert_shared_matcher_provenance(provenance, provenance)

    other = matcher_ablation_provenance(
        FrameManifest(
            "matcher",
            "ctrnetx",
            tuple(f"other{index:04d}" for index in range(300)),
        )
    )
    with pytest.raises(ValueError, match="parent manifest SHA-256"):
        assert_shared_matcher_provenance(provenance, other)
