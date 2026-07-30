from __future__ import annotations

import pytest

from calibx.benchmarks.ctrnetx.protocol import (
    CTRNETX_SINGLE_FRAME_COUNTS,
    CTRNETX_SINGLE_TOTAL_FRAMES,
    CtrnetxBatchManifest,
    CtrnetxEpisodeScope,
    CtrnetxProtocol,
    assert_full_ctrnetx_single_scope,
)
from calibx.benchmarks.dream_real.protocol import (
    DREAM_REAL_FULL_FRAME_COUNTS,
    DreamRealFullProtocol,
    assert_full_dream_real_scope,
    is_full_dream_real_scope,
)


def test_dream_real_protocol_accepts_known_camera() -> None:
    protocol = DreamRealFullProtocol(camera="orb")
    protocol.validate()


def test_dream_real_protocol_rejects_unknown_camera() -> None:
    with pytest.raises(ValueError, match="unknown DREAM-real"):
        DreamRealFullProtocol(camera="unsupported").validate()


def test_dream_real_full_protocol_rejects_nonpaper_settings() -> None:
    with pytest.raises(ValueError, match="six views"):
        DreamRealFullProtocol(camera="orb", views=1).validate()
    with pytest.raises(ValueError, match="match_batch_size=6"):
        DreamRealFullProtocol(camera="orb", match_batch_size=1).validate()


def test_ctrnetx_protocol_keeps_batch_and_single_distinct() -> None:
    CtrnetxProtocol(mode="single").validate()
    CtrnetxProtocol(mode="batch", replay_views=1).validate()
    with pytest.raises(ValueError, match="one pose-aligned replay render"):
        CtrnetxProtocol(mode="batch", replay_views=6).validate()
    with pytest.raises(ValueError, match="R3"):
        CtrnetxProtocol(mode="single", refinement_iterations=0).validate()
    with pytest.raises(ValueError, match="match_batch_size=6"):
        CtrnetxProtocol(mode="single", stage1_match_batch_size=1).validate()


def test_archived_full_run_scopes_are_explicit() -> None:
    assert sum(DREAM_REAL_FULL_FRAME_COUNTS.values()) == 49_619
    assert CTRNETX_SINGLE_TOTAL_FRAMES == 17_306
    assert_full_dream_real_scope(DREAM_REAL_FULL_FRAME_COUNTS)
    assert is_full_dream_real_scope(DREAM_REAL_FULL_FRAME_COUNTS)
    assert_full_ctrnetx_single_scope(CTRNETX_SINGLE_FRAME_COUNTS)


def test_archived_full_run_scope_rejects_a_changed_count() -> None:
    changed = dict(CTRNETX_SINGLE_FRAME_COUNTS)
    changed["robot_in_view_partial"] -= 1
    with pytest.raises(ValueError, match="archived split counts"):
        assert_full_ctrnetx_single_scope(changed)
    assert not is_full_dream_real_scope({"orb": 32_315})


def test_batch_manifest_requires_the_archived_episode_and_frame_scope() -> None:
    manifest = CtrnetxBatchManifest(
        episodes=tuple(
            CtrnetxEpisodeScope(f"episode-{index:02d}", 314 if index == 0 else 288)
            for index in range(60)
        )
    )
    manifest.validate()
    assert len(manifest.sha256()) == 64


def test_batch_manifest_rejects_duplicate_episode_id() -> None:
    episodes = [
        CtrnetxEpisodeScope(f"episode-{index:02d}", 314 if index == 0 else 288)
        for index in range(60)
    ]
    episodes[-1] = CtrnetxEpisodeScope("episode-00", 288)
    with pytest.raises(ValueError, match="must be unique"):
        CtrnetxBatchManifest(tuple(episodes)).validate()
