from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.run_droid_eval import (
    batch_frame_set,
    ctrnetx_batch_pnp_seed,
    ctrnetx_frame_correspondences,
    frame_indices,
    masked_observed,
)


def test_ctrnetx_batch_protocol_uses_every_video_frame(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.run_droid_eval.video_record",
        lambda _path: {"frame_count": 7},
    )
    config = {"data": {"batch_frame_set": "all"}}
    session = SimpleNamespace(video_path=Path("unused.mp4"))

    assert batch_frame_set(config) == "all"
    assert frame_indices(config, session, "all") == tuple(range(7))


def test_batch_matching_falls_back_to_unmasked_frame(tmp_path) -> None:
    session = SimpleNamespace(session_id="episode__ext1")
    destination = tmp_path / "sessions" / session.session_id / "frames" / "000003"
    destination.mkdir(parents=True)
    image = np.full((4, 5, 3), 127, dtype=np.uint8)
    Image.fromarray(image).save(destination / "real.png")

    np.testing.assert_array_equal(masked_observed(tmp_path, session, 3), image)


def test_ctrnetx_t0_replays_single_frame_pnp_input_correspondences() -> None:
    best = {
        "pnp": {"status": "success"},
        "_image_points": np.full((5, 2), 1.0),
        "_world_points": np.full((5, 3), 2.0),
        "_scores": np.full(5, 3.0),
        "_pose": {
            "image_points": np.full((2, 2), 4.0),
            "world_points": np.full((2, 3), 5.0),
            "scores": np.full(2, 6.0),
        },
    }

    image_points, world_points, scores, eligible = ctrnetx_frame_correspondences(
        best,
        refinement=False,
    )

    assert eligible is True
    assert len(scores) == 2
    np.testing.assert_array_equal(image_points, best["_pose"]["image_points"])
    np.testing.assert_array_equal(world_points, best["_pose"]["world_points"])


def test_ctrnetx_refinement_keeps_raw_post_geometry_correspondences() -> None:
    best = {
        "pnp": {"status": "failed"},
        "_image_points": np.full((5, 2), 1.0),
        "_world_points": np.full((5, 3), 2.0),
        "_scores": np.full(5, 3.0),
        "_pose": None,
    }

    image_points, world_points, scores, eligible = ctrnetx_frame_correspondences(
        best,
        refinement=True,
    )

    assert eligible is True
    assert len(scores) == 5
    np.testing.assert_array_equal(image_points, best["_image_points"])
    np.testing.assert_array_equal(world_points, best["_world_points"])


def test_ctrnetx_batch_pnp_seed_matches_replay_and_closed_loop_ordinals() -> None:
    assert ctrnetx_batch_pnp_seed(90, 0, 0) == 90
    assert ctrnetx_batch_pnp_seed(90, 3, 0) == 90 + 3 * 1009
    assert ctrnetx_batch_pnp_seed(90, 3, 1) == 90 + 31 * 1009
    assert ctrnetx_batch_pnp_seed(90, 3, 3) == 90 + 33 * 1009
