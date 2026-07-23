from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.run_droid_eval import batch_frame_set, frame_indices, masked_observed


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
