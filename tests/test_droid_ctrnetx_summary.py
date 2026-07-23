import pytest

from scripts.summarize_droid_ctrnetx import choose_visuals, summarize


def record(episode: str, frame: int, iou: float) -> dict:
    return {
        "session_id": f"{episode}__ext1",
        "episode_uuid": episode,
        "episode_rank": int(episode[-1]),
        "frame_index": frame,
        "status": "success",
        "metrics": {"iou": iou},
        "native_iou_1280x720": iou,
    }


def test_summary_reports_frame_session_and_episode_macros() -> None:
    records = [record("episode0", 0, 0.5), record("episode0", 1, 0.7), record("episode1", 0, 0.9)]
    records.append(
        {
            **record("episode1", 1, 0.0),
            "status": "failure",
            "native_iou_1280x720": None,
        }
    )

    result = summarize("method", records)

    assert result["requested_frames"] == 4
    assert result["evaluable_frames"] == 3
    assert result["missing_or_failed_frames"] == 1
    assert result["native_1280x720_frame_iou_macro"] == pytest.approx(0.7)
    assert result["native_1280x720_session_iou_macro"] == pytest.approx(0.75)
    assert result["native_1280x720_episode_iou_macro"] == pytest.approx(0.75)


def test_visual_selection_uses_one_best_frame_per_episode() -> None:
    raw = [record("episode0", 0, 0.1), record("episode0", 1, 0.2), record("episode1", 0, 0.3)]
    final = [record("episode0", 0, 0.8), record("episode0", 1, 0.9), record("episode1", 0, 0.7)]

    selected = choose_visuals(raw, final, count=2)

    assert [(row[1]["episode_uuid"], row[1]["frame_index"]) for row in selected] == [
        ("episode0", 1),
        ("episode1", 0),
    ]
