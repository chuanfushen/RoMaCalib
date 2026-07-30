from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from calibx.benchmarks.dream_real.legacy_executor import (
    LegacyExecutionOptions,
    LegacyFrameInput,
    LegacyPreparedFrame,
    discover_legacy_frames,
    execute_legacy_batch,
    execute_legacy_frame,
)
from calibx.benchmarks.dream_real.legacy_runner import (
    LegacyEstimate,
    LegacyPoseArtifact,
    LegacyRefinementAttempt,
    LegacyStage1Artifact,
    save_legacy_pose_npz,
)


def _pose(offset: float = 0.0) -> LegacyPoseArtifact:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[0, 3] = offset
    camera_to_world = np.linalg.inv(world_to_camera)
    return LegacyPoseArtifact(
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        image_points=np.array([[4.0, 5.0]], dtype=np.float64),
        world_points=np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        scores=np.array([0.9], dtype=np.float64),
        inlier_indices=np.array([0], dtype=np.int64),
        reprojection_errors=np.array([0.5], dtype=np.float64),
    )


def _metrics() -> dict:
    return {
        "summary": {
            "pixel_error_mean": 1.0,
            "keypoint_add_mean_m": 0.01,
        },
        "per_keypoint": [],
    }


def _estimate(offset: float = 0.0, *, suffix: str = "") -> LegacyEstimate:
    pose = _pose(offset)
    return LegacyEstimate(
        pose=pose,
        pnp={
            "status": "success",
            "inliers": 1,
            "inlier_reprojection_error_mean": 0.5,
            "world_to_camera": pose.world_to_camera.tolist(),
            "camera_to_robot_base": pose.camera_to_world.tolist(),
        },
        render_path=f"render{suffix}.png",
        camera_npz=f"render{suffix}_camera.npz",
        keypoint_metrics=_metrics(),
        pose_npz=f"pose{suffix}.npz",
    )


def _write_stage1(root: Path, index: int, *, with_mask: bool = True) -> None:
    frame_dir = root / f"{index:06d}"
    frame_dir.mkdir(parents=True)
    pose = _pose()
    pose_path = save_legacy_pose_npz(frame_dir, pose, np.eye(3), _metrics())
    summary = {
        "status": "success",
        "frame_index": index,
        "best_render_path": "stage0.png",
        "best_camera_npz": "stage0_camera.npz",
        "pose_npz": str(pose_path),
        "keypoint_metrics": _metrics(),
        "best_pnp": _estimate().pnp,
        "views": [{"view_index": 0, "pnp": {"status": "success"}}],
    }
    (frame_dir / "frame_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if with_mask:
        Image.fromarray(np.array([[255, 0], [255, 255]], dtype=np.uint8)).save(
            frame_dir / "input_mask.png"
        )


def _frame(dataset_dir: Path, index: int) -> LegacyFrameInput:
    json_path = dataset_dir / f"{index:06d}.json"
    json_path.write_text("{}", encoding="utf-8")
    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(
        json_path.with_suffix(".rgb.jpg")
    )
    return LegacyFrameInput(index, json_path, json_path.with_suffix(".rgb.jpg"))


def _options(tmp_path: Path, *, refinement_iterations: int = 3) -> LegacyExecutionOptions:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _frame(dataset, 0)
    initial = tmp_path / "initial"
    _write_stage1(initial, 0)
    xml = tmp_path / "panda.xml"
    xml.write_text("<mujoco/>", encoding="utf-8")
    runtime_identity = tmp_path / "runtime_identity.json"
    runtime_identity.write_text(
        '{"code_commit":"test","weight_sha256":"test"}\n',
        encoding="utf-8",
    )
    return LegacyExecutionOptions(
        dataset_dir=dataset,
        mujoco_xml=xml,
        initial_results_dir=initial,
        output_dir=tmp_path / "output",
        sample_count=1,
        sample_seed=90,
        refinement_iterations=refinement_iterations,
        runtime_identity_file=runtime_identity,
    )


class _FakeBackend:
    def __init__(self, *, raise_iteration: int | None = None) -> None:
        self.prepared = 0
        self.calls: list[tuple[int, int | None]] = []
        self.raise_iteration = raise_iteration
        self.last_prepared: LegacyPreparedFrame | None = None

    def prepare_frame(self, options, frame, stage1):
        self.prepared += 1
        rgb = np.asarray(Image.open(frame.image_path).convert("RGB"))
        mask_path = stage1.input_mask_path if options.mask_input else None
        if mask_path is None:
            source = "original_rgb_fallback"
            matched = rgb
        else:
            source = "stage1_input_mask"
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            matched = rgb.copy()
            matched[~mask] = 0
            mask_path = options.output_dir / f"{frame.index:06d}" / "input_mask.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        prepared = LegacyPreparedFrame(
            frame=frame,
            stage1=stage1,
            output_dir=options.output_dir / f"{frame.index:06d}",
            observed_rgb=rgb,
            observed_for_match=matched,
            input_mask_path=mask_path,
            input_mask_source=source,
            runtime=None,
        )
        self.last_prepared = prepared
        return prepared

    def refinement_attempt(self, _options, _prepared, _parent, iteration, match_seed):
        self.calls.append((iteration, match_seed))
        if self.raise_iteration == iteration:
            raise RuntimeError("synthetic runtime failure")
        if iteration == 1:
            candidate = _estimate(0.1, suffix="_r1")
            return LegacyRefinementAttempt(
                render_path="projected_r1.png",
                camera_npz="projected_r1_camera.npz",
                match={"pnp": candidate.pnp},
                candidate=candidate,
            )
        return LegacyRefinementAttempt(
            render_path=f"projected_r{iteration}.png",
            camera_npz=f"projected_r{iteration}_camera.npz",
            match={"pnp": {"status": "too_few_correspondences"}},
            candidate=None,
        )


def test_legacy_executor_replays_stage1_mask_and_stops_after_failed_pnp(tmp_path: Path) -> None:
    options = _options(tmp_path)
    backend = _FakeBackend()
    execution = execute_legacy_batch(options, backend)

    assert backend.prepared == 1
    assert backend.calls == [(1, 91), (2, 92)]
    record = execution.records[0]
    assert record["status"] == "failed"
    assert record["input_mask"]["source"] == "stage1_input_mask"
    assert [item["status"] for item in record["iterations"]] == [
        "success",
        "success",
        "failed",
        "skipped",
    ]
    assert (options.output_dir / "000000" / "input_mask.png").is_file()
    assert record["input_mask"]["mask_path"] == str(
        options.output_dir / "000000" / "input_mask.png"
    )
    assert (
        options.output_dir / "000000" / "iterations" / "iteration_00" / "pose.npz"
    ).is_file()
    assert (options.output_dir / "summary.json").is_file()


def test_legacy_executor_records_callback_traceback_as_outer_frame_error(tmp_path: Path) -> None:
    options = _options(tmp_path)
    backend = _FakeBackend(raise_iteration=1)
    frame = discover_legacy_frames(options)[0]
    record = execute_legacy_frame(options, frame, backend)

    assert record["status"] == "error"
    assert backend.calls == [(1, 91)]
    assert "iterations" not in record
    assert "synthetic runtime failure" in record["traceback"]
    assert record["input_mask"]["mask_path"] == str(
        options.output_dir / "000000" / "input_mask.png"
    )


def test_legacy_executor_does_not_initialize_runtime_for_failed_stage1(tmp_path: Path) -> None:
    options = _options(tmp_path)
    initial_frame = options.initial_results_dir / "000000"
    (initial_frame / "frame_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "frame_index": 0,
                "reason": "no_stage1_pnp",
                "views": [],
            }
        ),
        encoding="utf-8",
    )

    class RuntimeMustNotStart:
        def prepare_frame(self, *_args):
            raise AssertionError("failed Stage 1 must not initialize a runtime backend")

        def refinement_attempt(self, *_args):
            raise AssertionError("failed Stage 1 must not enter refinement")

    record = execute_legacy_frame(
        options,
        discover_legacy_frames(options)[0],
        RuntimeMustNotStart(),
    )
    assert record["status"] == "failed"
    assert [item["status"] for item in record["iterations"]] == [
        "failed",
        "skipped",
        "skipped",
        "skipped",
    ]


def test_discovery_keeps_numeric_sorted_pairs_and_explicit_frame_indices(tmp_path: Path) -> None:
    options = _options(tmp_path, refinement_iterations=0)
    _frame(options.dataset_dir, 2)
    _write_stage1(options.initial_results_dir, 2, with_mask=False)
    selected = discover_legacy_frames(
        replace(options, frame_indices=(2, 0), sample_count=2)
    )
    assert [frame.index for frame in selected] == [0, 2]


def test_legacy_executor_does_not_replay_mask_when_disabled(tmp_path: Path) -> None:
    options = replace(_options(tmp_path, refinement_iterations=0), mask_input=False)
    backend = _FakeBackend()

    record = execute_legacy_frame(options, discover_legacy_frames(options)[0], backend)

    assert record["status"] == "success"
    assert backend.last_prepared is not None
    assert backend.last_prepared.input_mask_source == "original_rgb_fallback"
    assert backend.last_prepared.input_mask_path is None
    assert np.array_equal(
        backend.last_prepared.observed_rgb,
        backend.last_prepared.observed_for_match,
    )
    assert not (options.output_dir / "000000" / "input_mask.png").exists()


def test_discovery_resolves_a_nested_dream_export(tmp_path: Path) -> None:
    options = _options(tmp_path, refinement_iterations=0)
    outer = tmp_path / "nested_export"
    nested = outer / "panda-3cam_azure"
    nested.mkdir(parents=True)
    _frame(nested, 4)
    _write_stage1(options.initial_results_dir, 4, with_mask=False)

    selected = discover_legacy_frames(
        replace(options, dataset_dir=outer, frame_indices=(4,), sample_count=1)
    )

    assert len(selected) == 1
    assert selected[0].index == 4
    assert selected[0].json_path.parent == nested


def test_legacy_executor_writes_historical_stage0_and_final_pose_paths(tmp_path: Path) -> None:
    options = _options(tmp_path, refinement_iterations=0)
    record = execute_legacy_frame(options, discover_legacy_frames(options)[0], _FakeBackend())

    stage0_path = (
        options.output_dir / "000000" / "iterations" / "iteration_00" / "pose.npz"
    )
    final_path = options.output_dir / "000000" / "best_pose.npz"
    assert stage0_path.is_file()
    assert final_path.is_file()
    assert record["iterations"][0]["pose_npz"] == str(stage0_path)
    assert record["pose_npz"] == str(final_path)
    with np.load(stage0_path, allow_pickle=False) as stage0, np.load(
        final_path, allow_pickle=False
    ) as final:
        assert np.array_equal(stage0["world_to_camera"], final["world_to_camera"])


def test_legacy_batch_resume_requires_an_exact_local_run_manifest(tmp_path: Path) -> None:
    options = _options(tmp_path, refinement_iterations=0)
    first_backend = _FakeBackend()
    first = execute_legacy_batch(options, first_backend)

    manifest_path = options.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = manifest["run_fingerprint_sha256"]
    assert manifest["publication_policy"] == "machine_local_only_do_not_publish"
    assert first.summary["run_fingerprint_sha256"] == fingerprint
    assert first.records[0]["run_fingerprint_sha256"] == fingerprint

    resumed_backend = _FakeBackend()
    resumed = execute_legacy_batch(replace(options, resume=True), resumed_backend)
    assert resumed.records == first.records
    assert resumed_backend.prepared == 0
    assert resumed_backend.calls == []


def test_legacy_batch_rejects_resume_with_a_changed_run_fingerprint(tmp_path: Path) -> None:
    options = _options(tmp_path, refinement_iterations=0)
    execute_legacy_batch(options, _FakeBackend())

    with pytest.raises(ValueError, match="fingerprint differs"):
        execute_legacy_batch(
            replace(options, resume=True, match_seed=91),
            _FakeBackend(),
        )


def test_legacy_batch_rejects_resume_after_a_hashed_input_changes(tmp_path: Path) -> None:
    options = _options(tmp_path, refinement_iterations=0)
    execute_legacy_batch(options, _FakeBackend())

    Image.fromarray(np.full((2, 2, 3), 64, dtype=np.uint8)).save(
        options.dataset_dir / "000000.rgb.jpg"
    )
    with pytest.raises(ValueError, match="fingerprint differs"):
        execute_legacy_batch(replace(options, resume=True), _FakeBackend())
