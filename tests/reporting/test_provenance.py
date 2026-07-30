from __future__ import annotations

import pytest

from calibx.reporting.provenance import ExperimentProvenance


def test_clean_reproduction_record_is_public_safe() -> None:
    record = ExperimentProvenance(
        experiment_id="dream-real-full",
        protocol="dream_real",
        source_commit="d7e5e890da95c0a64fb4b60032b9b3d04595baa3",
        source_dirty=False,
        status="reproducible",
        requested_frames=8367,
        config="config/reproduction/dream_real/full.toml",
        manifest="paper/manifests/dream-real-full.json",
    )
    assert record.to_dict()["status"] == "reproducible"


def test_dirty_historical_record_cannot_be_called_reproducible() -> None:
    record = ExperimentProvenance(
        experiment_id="legacy",
        protocol="ctrnetx",
        source_commit="75b605d",
        source_dirty=True,
        status="reproducible",
        requested_frames=1,
        config="config/reproduction/ctrnetx/single.toml",
    )
    with pytest.raises(ValueError, match="dirty"):
        record.validate()


def test_absolute_paths_are_not_public_provenance() -> None:
    record = ExperimentProvenance(
        experiment_id="legacy",
        protocol="ctrnetx",
        source_commit="75b605d",
        source_dirty=True,
        status="historical",
        requested_frames=1,
        config="C:\\server\\secret.toml",
    )
    with pytest.raises(ValueError, match="repository-relative"):
        record.validate()


@pytest.mark.parametrize("config_path", ("C:private.toml", r"\private.toml"))
def test_windows_rooted_or_drive_relative_paths_are_not_public_provenance(
    config_path: str,
) -> None:
    record = ExperimentProvenance(
        experiment_id="legacy",
        protocol="ctrnetx",
        source_commit="75b605d",
        source_dirty=True,
        status="historical",
        requested_frames=1,
        config=config_path,
    )
    with pytest.raises(ValueError, match="repository-relative"):
        record.validate()
