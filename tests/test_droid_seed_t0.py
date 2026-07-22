import json
from types import SimpleNamespace

import pytest

from scripts.run_droid_eval import stage_seed_t0


def test_seed_t0_copies_exact_artifacts_and_refuses_overwrite(tmp_path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    session = SimpleNamespace(session_id="episode__ext1__serial")
    source_dir = source_root / "sessions" / session.session_id / "poses" / "iteration_00"
    source_dir.mkdir(parents=True)
    payloads = {
        "roma_pose.npz": b"roma-pose",
        "selected_pose.npz": b"selected-pose",
        "roma_summary.json": b'{"validation":{"iou_macro":0.5}}\n',
    }
    for name, payload in payloads.items():
        (source_dir / name).write_bytes(payload)

    stage_seed_t0([session], output_root, source_root)

    destination_dir = output_root / "sessions" / session.session_id / "poses" / "iteration_00"
    for name, payload in payloads.items():
        assert (destination_dir / name).read_bytes() == payload
    manifest = json.loads((output_root / "seed_t0_manifest.json").read_text(encoding="utf-8"))
    assert manifest["session_count"] == 1
    assert len(manifest["records"][0]["files"]) == 3

    with pytest.raises(FileExistsError):
        stage_seed_t0([session], output_root, source_root)
