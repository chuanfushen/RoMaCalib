from __future__ import annotations

import numpy as np
import pytest

from calibx.benchmarks.ablations.external_matchers import (
    ExternalMatcherSettings,
    empty_prediction,
    make_prediction,
    matcher_cli_name,
    restore_mast3r_coordinates,
)


def test_external_matcher_contract_imports_without_optional_projects() -> None:
    prediction = empty_prediction()
    assert prediction["mkeypoints0_orig"].shape == (0, 2)
    assert prediction["mkeypoints1_orig"].shape == (0, 2)
    assert prediction["mconf"].shape == (0,)
    assert matcher_cli_name("mast3r") == "MASt3R"


def test_prediction_normalization_requires_paired_correspondences() -> None:
    prediction = make_prediction(
        np.array([[1.0, 2.0]], dtype=np.float64),
        np.array([[3.0, 4.0]], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
    )
    assert prediction["mkeypoints0_orig"].dtype == np.float32
    with pytest.raises(ValueError, match="same length"):
        make_prediction(
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        )


def test_mast3r_coordinate_restore_inverts_crop_and_resize() -> None:
    restored = restore_mast3r_coordinates(
        np.array([[0.0, 0.0], [512.0, 384.0]], dtype=np.float32),
        scale_x=0.8,
        scale_y=0.8,
        crop_left=0.0,
        crop_top=0.0,
    )
    np.testing.assert_allclose(restored, [[0.0, 0.0], [640.0, 480.0]])


def test_lightglue_requires_explicit_provider_opt_in_before_import() -> None:
    with pytest.raises(ValueError, match="explicit opt-in"):
        ExternalMatcherSettings("lightglue").validate()


def test_external_asset_requirements_are_explicit() -> None:
    with pytest.raises(ValueError, match="RoMa v1 and DINOv2"):
        ExternalMatcherSettings("romav1").validate()
    with pytest.raises(ValueError, match="MASt3R checkpoint"):
        ExternalMatcherSettings("mast3r").validate()
