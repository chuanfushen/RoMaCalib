from __future__ import annotations

from pathlib import Path

import pytest

from calibx.configuration import load_config, validate_config


def minimal_config() -> dict:
    return {
        "paths": {
            "data_root": "data",
            "mujoco_xml": "robot.xml",
            "prerender_root": "renders",
            "output_root": "outputs",
        },
        "datasets": {
            "example": {
                "name": "example",
                "path": "example",
            }
        },
        "evaluation": {
            "datasets": ["example"],
            "views": 6,
            "match_batch_size": 6,
        },
        "prerender": {"workers": 1},
    }


def test_validate_config_accepts_minimal_configuration() -> None:
    validate_config(minimal_config())


def test_validate_config_rejects_unknown_dataset() -> None:
    config = minimal_config()
    config["evaluation"]["datasets"] = ["missing"]
    with pytest.raises(ValueError, match="unknown datasets"):
        validate_config(config)


def test_validate_config_keeps_legacy_masking_config_valid() -> None:
    config = minimal_config()
    config["evaluation"]["mask_input"] = True
    validate_config(config)


def test_repository_default_config_is_valid() -> None:
    config, path = load_config()
    assert path.name == "dream_eval.toml"
    assert config["evaluation"]["views"] >= 1
    data_root = Path(config["paths"]["data_root"])
    dataset_path = Path(config["datasets"]["dr"]["path"])
    assert data_root == Path("data/dream")
    assert dataset_path == data_root / "synthetic/panda_synth_test_dr"
