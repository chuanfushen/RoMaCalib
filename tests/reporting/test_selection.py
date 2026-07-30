from __future__ import annotations

import pytest

from calibx.reporting.selection import (
    CtrnetxQualitativeCase,
    DroidFigure3Candidate,
    QualitativeFigureManifest,
    QualitativeItem,
    RefinementRound,
    select_best_successful_round,
    select_ctrnetx_cases,
    select_droid_figure3_candidates,
)


def test_qualitative_manifest_keeps_selection_rule_and_reused_pose() -> None:
    manifest = QualitativeFigureManifest(
        figure_id="fig4-robochallenge",
        source_protocol="robochallenge_qualitative",
        selection_rule="pose-diverse qualitative cases; reused poses disclosed",
        items=(
            QualitativeItem(
                result_id="arx5-episode-01",
                display_frame_id="arx5/episode-01/frame-040",
                source_result_id="arx5/episode-01/frame-038",
                pose_reused=True,
            ),
        ),
    )
    payload = manifest.to_dict()
    assert payload["figure_kind"] == "qualitative"
    assert payload["items"][0]["pose_reused"] is True


def test_reused_pose_requires_its_source_identifier() -> None:
    manifest = QualitativeFigureManifest(
        figure_id="fig4-robochallenge",
        source_protocol="robochallenge_qualitative",
        selection_rule="qualitative",
        items=(
            QualitativeItem(
                result_id="case-1",
                display_frame_id="episode/frame",
                pose_reused=True,
            ),
        ),
    )
    with pytest.raises(ValueError, match="source_result_id"):
        manifest.validate()


def test_absolute_asset_paths_are_rejected() -> None:
    manifest = QualitativeFigureManifest(
        figure_id="fig3-droid",
        source_protocol="droid_qualitative",
        selection_rule="best cases",
        items=(
            QualitativeItem(
                result_id="case-1",
                display_frame_id="/opt/private/frame.png",
            ),
        ),
    )
    with pytest.raises(ValueError, match="public relative"):
        manifest.validate()


@pytest.mark.parametrize("result_id", ("C:private", r"\private"))
def test_windows_rooted_or_drive_relative_identifiers_are_rejected(
    result_id: str,
) -> None:
    manifest = QualitativeFigureManifest(
        figure_id="fig3-droid",
        source_protocol="droid_qualitative",
        selection_rule="best cases",
        items=(QualitativeItem(result_id, "episode/frame"),),
    )
    with pytest.raises(ValueError, match="public relative"):
        manifest.validate()


def test_droid_figure3_selector_uses_gain_and_one_case_per_episode() -> None:
    selected = select_droid_figure3_candidates(
        [
            DroidFigure3Candidate("a-best", "episode-a", "a/10", 0.90, 0.20),
            DroidFigure3Candidate("a-second", "episode-a", "a/11", 0.99, 0.40),
            DroidFigure3Candidate("b", "episode-b", "b/10", 0.82, 0.30),
            DroidFigure3Candidate("c", "episode-c", "c/10", 0.85, 0.20),
            DroidFigure3Candidate("d", "episode-d", "d/10", 0.84, 0.21),
            DroidFigure3Candidate("e", "episode-e", "e/10", 0.83, 0.22),
            DroidFigure3Candidate("f", "episode-f", "f/10", 0.81, 0.23),
            DroidFigure3Candidate("too-low", "episode-g", "g/10", 0.79, 0.00),
        ]
    )
    assert [candidate.result_id for candidate in selected] == [
        "a-best",
        "c",
        "d",
        "e",
        "f",
        "b",
    ]


def test_figure5_uses_best_successful_round_only() -> None:
    best = select_best_successful_round(
        [
            RefinementRound(0, "success", 0.60),
            RefinementRound(1, "failed", None),
            RefinementRound(2, "success", 0.82),
        ]
    )
    assert best.round_index == 2


def test_ctrnetx_panels_are_best_and_visible_high_error_cases() -> None:
    cases = [
        CtrnetxQualitativeCase(f"frame-{index}", "success", float(index), True)
        for index in range(10)
    ]
    best, high_error = select_ctrnetx_cases(cases, top_k=2)
    assert [case.frame_id for case in best] == ["frame-0", "frame-1"]
    assert [case.frame_id for case in high_error] == ["frame-9", "frame-8"]
