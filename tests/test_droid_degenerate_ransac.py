import numpy as np

from romav2.benchmarks.utils import geometry


def test_imcui_ransac_tolerates_failed_stereo_rectification(monkeypatch) -> None:
    points0 = np.arange(40, dtype=np.float32).reshape(20, 2)
    points1 = points0 + 1

    def fake_ransac(*args):
        geometry_type = args[-1]
        matrix = np.eye(3, dtype=np.float64)
        mask = np.ones(20, dtype=bool)
        assert geometry_type in {"Fundamental", "Homography"}
        return matrix, mask

    monkeypatch.setattr(geometry, "proc_imcui_ransac_matches", fake_ransac)
    monkeypatch.setattr(
        geometry.cv2,
        "stereoRectifyUncalibrated",
        lambda *_args, **_kwargs: (False, None, None),
    )

    mask, status, info = geometry.imcui_keypoint_ransac_mask(
        points0,
        points1,
        (720, 1280, 3),
        geometry.DEFAULT_RANSAC_METHOD,
        4.0,
        0.999,
        10_000,
    )

    assert status == "success"
    assert mask.all()
    assert info["stereo_rectified"] is False
    assert "H1" not in info
    assert "H2" not in info
