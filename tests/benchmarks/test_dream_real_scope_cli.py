from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]


def _load_script(relative_path: str):
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _duplicate_camera_config() -> dict:
    return {
        "protocol": {"semantic_id": "legacy-unconditional-pnp-replace-v1"},
        "evaluation": {"datasets": ["azure", "azure"]},
        "datasets": {"azure": {}},
    }


def _full_scope_config(*, frame_indices: tuple[int, ...] = ()) -> dict:
    return {
        "evaluation": {
            "datasets": ["azure", "kinect360", "orb", "realsense"],
            "frame_indices": list(frame_indices),
        },
        "datasets": {
            "azure": {"sample_count": 6_394},
            "kinect360": {"sample_count": 4_966},
            "orb": {"sample_count": 32_315},
            "realsense": {"sample_count": 5_944},
        },
    }


def test_full_dream_cli_rejects_duplicate_camera_selection() -> None:
    run_script = _load_script("scripts/reproduce/dream_real/run_legacy_full.py")
    plan_script = _load_script("scripts/reproduce/dream_real/plan_legacy_full.py")

    with pytest.raises(ValueError, match="duplicate"):
        run_script._selected_datasets(_duplicate_camera_config(), None)
    with pytest.raises(ValueError, match="duplicate"):
        plan_script._planned_datasets(_duplicate_camera_config(), None)


def test_configured_frame_indices_mark_a_dream_run_as_subset() -> None:
    run_script = _load_script("scripts/reproduce/dream_real/run_legacy_full.py")
    config = _full_scope_config(frame_indices=(12, 34, 56, 78, 90))

    effective = run_script._effective_frame_indices(config, ())
    scope = run_script._scope_payload(
        config,
        tuple(config["evaluation"]["datasets"]),
        explicit_dataset_selection=False,
        frame_indices=effective,
        limit=None,
    )

    assert effective == (12, 34, 56, 78, 90)
    assert scope["kind"] == "explicit_subset_or_smoke"
    assert scope["paper_full_eligible"] is False
    assert scope["restriction"]["frame_indices"] == [12, 34, 56, 78, 90]


def test_cli_frame_indices_override_the_toml_selection() -> None:
    run_script = _load_script("scripts/reproduce/dream_real/run_legacy_full.py")
    config = _full_scope_config(frame_indices=(12, 34, 56, 78, 90))

    assert run_script._effective_frame_indices(config, (101, 202)) == (101, 202)
