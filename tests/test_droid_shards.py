import json
from types import SimpleNamespace

import pytest

from scripts.run_droid_eval import shard_or_global_path, stage_aggregate_eval, stage_aggregate_masks


def _record(session_id: str, episode_uuid: str, frame_index: int, iou: float) -> dict:
    return {
        "session_id": session_id,
        "episode_rank": 0,
        "episode_uuid": episode_uuid,
        "camera_name": session_id.rsplit("__", 1)[-1],
        "frame_index": frame_index,
        "split": "heldout",
        "status": "success",
        "metrics": {"iou": iou},
    }


def test_shard_paths_do_not_replace_legacy_global_path(tmp_path) -> None:
    assert shard_or_global_path(tmp_path, "raw_eval_heldout", None) == tmp_path / "raw_eval_heldout.json"
    assert shard_or_global_path(tmp_path, "raw_eval_heldout", "episode__ext1") == (
        tmp_path / "manifests" / "raw_eval_heldout" / "episode__ext1.json"
    )


def test_aggregate_eval_combines_session_shards(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scripts.run_droid_eval.git_record",
        lambda: {"commit": "test", "branch": "test", "status_short": []},
    )
    sessions = [
        SimpleNamespace(session_id="episode__ext1"),
        SimpleNamespace(session_id="episode__ext2"),
    ]
    shard_root = tmp_path / "manifests" / "pose_eval_heldout_iteration_03"
    shard_root.mkdir(parents=True)
    for session, iou in zip(sessions, (0.4, 0.8), strict=True):
        record = _record(session.session_id, "episode", 1, iou)
        (shard_root / f"{session.session_id}.json").write_text(
            json.dumps({"records": [record]}),
            encoding="utf-8",
        )

    stage_aggregate_eval(sessions, tmp_path, "heldout", 3, "pose")

    result = json.loads((tmp_path / "pose_eval_heldout_iteration_03.json").read_text(encoding="utf-8"))
    assert result["requested_frames"] == 2
    assert result["successful_frames"] == 2
    assert result["frame_iou_macro"] == pytest.approx(0.6)
    assert result["session_iou_macro"] == pytest.approx(0.6)
    assert result["episode_iou_macro"] == pytest.approx(0.6)
    assert result["aggregation"]["expected_sessions"] == 2


def test_aggregate_masks_requires_and_combines_all_session_shards(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scripts.run_droid_eval.git_record",
        lambda: {"commit": "test", "branch": "test", "status_short": []},
    )
    sessions = [SimpleNamespace(session_id="episode__ext1"), SimpleNamespace(session_id="episode__ext2")]
    shard_root = tmp_path / "manifests" / "mask_manifest_heldout"
    shard_root.mkdir(parents=True)
    for session in sessions:
        record = {
            "session_id": session.session_id,
            "frame_index": 2,
            "status": "success",
        }
        (shard_root / f"{session.session_id}.json").write_text(
            json.dumps({"records": [record]}),
            encoding="utf-8",
        )

    stage_aggregate_masks(sessions, tmp_path, "heldout")

    result = json.loads((tmp_path / "mask_manifest_heldout.json").read_text(encoding="utf-8"))
    assert result["record_count"] == 2
    assert result["successful_records"] == 2
    assert result["failed_records"] == 0
    assert len(result["shards"]) == 2
