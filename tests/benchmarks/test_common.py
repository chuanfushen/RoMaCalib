from __future__ import annotations

import pytest

from calibx.benchmarks.common import FrameManifest, assert_paired_manifests


def test_sampling_matches_sorted_numpy_choice_positions() -> None:
    manifest = FrameManifest.deterministic_sample(
        protocol="matcher_ablation",
        dataset_id="panda-orb",
        available_frame_ids=[f"{index:06d}" for index in range(10)],
        count=4,
        seed=90,
    )
    assert manifest.frame_ids == ("000000", "000002", "000004", "000006")
    assert manifest.selection_method == "numpy_default_rng_choice_sorted_positions"


def test_contiguous_shards_cover_the_parent_once() -> None:
    manifest = FrameManifest(
        protocol="sam_ablation",
        dataset_id="ctrnetx-video30",
        frame_ids=tuple(str(index) for index in range(5)),
    )
    first = manifest.shard(shard_index=0, shard_count=2)
    second = manifest.shard(shard_index=1, shard_count=2)
    assert first.frame_ids == ("0", "1")
    assert second.frame_ids == ("2", "3", "4")
    assert first.frame_ids + second.frame_ids == manifest.frame_ids


def test_paired_ablations_reject_different_requested_frames() -> None:
    first = FrameManifest("sam", "panda", ("000001", "000002"))
    second = FrameManifest("no_sam", "panda", ("000001", "000003"))
    with pytest.raises(ValueError, match="exactly the same"):
        assert_paired_manifests(first, second)


@pytest.mark.parametrize("frame_id", ("C:private-frame", r"\private-frame"))
def test_manifest_rejects_windows_rooted_or_drive_relative_identifier(
    frame_id: str,
) -> None:
    manifest = FrameManifest("sam", "panda", (frame_id,))
    with pytest.raises(ValueError, match="public relative"):
        manifest.validate()


def test_manifest_rejects_uri_identifiers() -> None:
    for frame_id in (
        "file:///opt/data/private/frame.png",
        "ssh://root@192.168.21.57/opt/data/private",
    ):
        with pytest.raises(ValueError, match="public relative"):
            FrameManifest("sam", "panda", (frame_id,)).validate()
