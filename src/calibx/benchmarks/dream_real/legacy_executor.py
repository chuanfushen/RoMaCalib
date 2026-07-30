"""Runtime boundary for the archived DREAM-real legacy R0--R3 protocol.

The state machine itself lives in :mod:`legacy_runner`; this module deliberately
keeps data discovery, artifact persistence, and optional core-runtime hooks out
of the generic Calib-X runner.  In particular, it is *not* an alias for
``calibx.runner``: R0 is loaded from a previously completed six-view stage-1
run, and R1--R3 each consume exactly one pose-aligned render.

Only the concrete backend imports Torch, OpenCV, MuJoCo, or the RoMa matcher,
and it does so while an ``--execute`` invocation is running.  The public
planning/record interfaces remain usable in a lightweight environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import traceback
from typing import Any, Literal, Protocol

import numpy as np
from PIL import Image

from calibx.metrics import summarize

from .legacy import LEGACY_REFINEMENT_SEMANTIC_ID, LegacyEvaluatorRequest
from .legacy_runner import (
    LegacyEstimate,
    LegacyPoseArtifact,
    LegacyRefinementAttempt,
    LegacyStage1Artifact,
    configure_legacy_match_rng,
    load_legacy_stage1_artifact,
    run_legacy_unconditional_trajectory,
    save_legacy_pose_npz,
)


LEGACY_EXECUTION_SEMANTIC_ID = "dream-real-legacy-r0-artifact-replay-v1"
LEGACY_RUN_MANIFEST_FILENAME = "run_manifest.json"
LEGACY_RUN_MANIFEST_SCHEMA_VERSION = 2
InputMaskSource = Literal["stage1_input_mask", "original_rgb_fallback"]


def _json_default(value: object) -> object:
    """Serialize runtime summaries without changing their numerical contents."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Encode a local execution identity deterministically for resume checks."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


@dataclass(frozen=True)
class LegacyFrameInput:
    """One numeric DREAM frame and its paired observed RGB image."""

    index: int
    json_path: Path
    image_path: Path

    def validate(self) -> None:
        if self.index < 0:
            raise ValueError("legacy frame index must be non-negative")
        if self.json_path.name != f"{self.index:06d}.json":
            raise ValueError("legacy JSON file name must match its numeric frame index")
        if self.image_path != self.json_path.with_suffix(".rgb.jpg"):
            raise ValueError("legacy image must use the paired .rgb.jpg name")


@dataclass(frozen=True)
class LegacyExecutionOptions:
    """Host-local inputs for the archived full DREAM-real refinement stage.

    ``initial_results_dir`` is intentionally mandatory for execution.  This
    evaluator consumes the prior stage-1 artifacts and must never silently
    rerun the orbit search or SAM merely because an artifact is absent.
    """

    dataset_dir: Path
    mujoco_xml: Path
    initial_results_dir: Path
    output_dir: Path
    sample_count: int
    sample_seed: int
    views: int = 6
    match_batch_size: int = 6
    refinement_iterations: int = 3
    device: str = "cuda"
    mask_input: bool = True
    mask_prompt: str = "robotic arm"
    match_seed: int | None = 90
    frame_indices: tuple[int, ...] = ()
    limit: int | None = None
    resume: bool = False
    save_visualizations: bool = True
    save_all_matches: bool = True
    visual_geom_group: int = 2
    visual_body_names: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None
    camera_settings: Path | None = None
    runtime_identity_file: Path | None = None
    distance_scale: float = 2.8
    min_distance: float = 1.2
    elevation: float = -20.0
    azimuth_offset: float = 0.0
    semantic_id: str = LEGACY_REFINEMENT_SEMANTIC_ID

    @classmethod
    def from_request(
        cls,
        request: LegacyEvaluatorRequest,
        *,
        width: int | None = None,
        height: int | None = None,
        camera_settings: Path | None = None,
        runtime_identity_file: Path | None = None,
        distance_scale: float = 2.8,
        min_distance: float = 1.2,
        elevation: float = -20.0,
        azimuth_offset: float = 0.0,
        limit: int | None = None,
    ) -> "LegacyExecutionOptions":
        """Convert the existing plan-only adapter into an executable request."""
        request.validate()
        if request.initial_results_dir is None:
            raise ValueError(
                "legacy execution requires paths.initial_results_root and a "
                "per-dataset initial_results_name"
            )
        return cls(
            dataset_dir=Path(request.dataset_dir),
            mujoco_xml=Path(request.mujoco_xml),
            initial_results_dir=Path(request.initial_results_dir),
            output_dir=Path(request.output_dir),
            sample_count=request.sample_count,
            sample_seed=request.sample_seed,
            views=request.views,
            match_batch_size=request.match_batch_size,
            refinement_iterations=request.refinement_iterations,
            device=request.device,
            mask_input=request.mask_input,
            mask_prompt=request.mask_prompt,
            match_seed=request.match_seed,
            frame_indices=request.frame_indices,
            limit=limit,
            resume=request.resume,
            save_visualizations=request.save_visualizations,
            save_all_matches=request.save_all_matches,
            visual_geom_group=request.visual_geom_group,
            visual_body_names=request.visual_body_names,
            width=width,
            height=height,
            camera_settings=camera_settings,
            runtime_identity_file=runtime_identity_file,
            distance_scale=distance_scale,
            min_distance=min_distance,
            elevation=elevation,
            azimuth_offset=azimuth_offset,
        )

    def validate(self) -> None:
        if self.semantic_id != LEGACY_REFINEMENT_SEMANTIC_ID:
            raise ValueError("unexpected legacy refinement semantic ID")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least one")
        if self.views != 6 or self.match_batch_size != 6:
            raise ValueError(
                "the archived DREAM-real stage-1 protocol fixes views=6 and "
                "match_batch_size=6"
            )
        if not 0 <= self.refinement_iterations <= 3:
            raise ValueError("legacy refinement_iterations must be in [0, 3]")
        if self.visual_geom_group != 2:
            raise ValueError("legacy pose-aligned rendering requires visual_geom_group=2")
        if any(not name.strip() for name in self.visual_body_names):
            raise ValueError("visual_body_names must not contain empty names")
        if not self.device:
            raise ValueError("device must not be empty")
        if not self.mask_prompt:
            raise ValueError("mask_prompt must not be empty")
        if self.match_seed is not None and (
            isinstance(self.match_seed, bool) or int(self.match_seed) < 0
        ):
            raise ValueError("match_seed must be non-negative when supplied")
        if any(index < 0 for index in self.frame_indices):
            raise ValueError("frame_indices must be non-negative")
        if len(set(self.frame_indices)) != len(self.frame_indices):
            raise ValueError("frame_indices must not contain duplicates")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive when supplied")
        if self.width is not None and self.width < 1:
            raise ValueError("width must be positive when supplied")
        if self.height is not None and self.height < 1:
            raise ValueError("height must be positive when supplied")
        if self.resume and self.runtime_identity_file is None:
            raise ValueError(
                "legacy resume requires a machine-local runtime_identity_file that "
                "pins the weights and environment used by the original invocation"
            )
        if self.output_dir.resolve() == self.initial_results_dir.resolve():
            raise ValueError(
                "legacy output_dir must differ from initial_results_dir; "
                "the archived Stage-1 artifacts are read-only"
            )

    def to_plan(self) -> dict[str, object]:
        """Return an inspectable plan without checking host-local assets."""
        self.validate()
        return {
            "execution_semantic_id": LEGACY_EXECUTION_SEMANTIC_ID,
            "semantic_id": self.semantic_id,
            "dataset_dir": str(self.dataset_dir),
            "initial_results_dir": str(self.initial_results_dir),
            "output_dir": str(self.output_dir),
            "sample_count": self.sample_count,
            "sample_seed": self.sample_seed,
            "frame_indices": list(self.frame_indices),
            "limit": self.limit,
            "runtime_identity_file": (
                None
                if self.runtime_identity_file is None
                else str(self.runtime_identity_file)
            ),
            "stage1": {"views": self.views, "match_batch_size": self.match_batch_size},
            "robot": {
                "visual_geom_group": self.visual_geom_group,
                "visual_body_names": list(self.visual_body_names),
            },
            "refinement_iterations": self.refinement_iterations,
            "mask": {
                "configured": self.mask_input,
                "replay_rule": (
                    "when enabled, use stage1 input_mask.png if present; "
                    "otherwise use original RGB"
                ),
                "sam_rerun": False,
            },
            "replay": {
                "views_per_iteration": 1,
                "candidate_policy": "unconditional_replace_on_pnp_success",
                "pnp_failure_policy": "frame_failed_then_later_iterations_skipped",
            },
        }


@dataclass(frozen=True)
class LegacyPreparedFrame:
    """One prepared runtime frame handed from a backend to the state machine."""

    frame: LegacyFrameInput
    stage1: LegacyStage1Artifact
    output_dir: Path
    observed_rgb: np.ndarray
    observed_for_match: np.ndarray
    input_mask_path: Path | None
    input_mask_source: InputMaskSource
    runtime: object

    def validate(self) -> None:
        self.frame.validate()
        if self.output_dir.name != f"{self.frame.index:06d}":
            raise ValueError("prepared frame output_dir must use the numeric frame ID")
        for image, label in (
            (self.observed_rgb, "observed_rgb"),
            (self.observed_for_match, "observed_for_match"),
        ):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"{label} must have shape [H, W, 3]")
        if self.input_mask_source == "stage1_input_mask" and self.input_mask_path is None:
            raise ValueError("stage1_input_mask source requires input_mask_path")
        if self.input_mask_source == "original_rgb_fallback" and self.input_mask_path:
            raise ValueError("original_rgb_fallback must not name an input mask")


class LegacyIterationBackend(Protocol):
    """Runtime hooks for artifact replay, not a replacement pipeline API."""

    def prepare_frame(
        self,
        options: LegacyExecutionOptions,
        frame: LegacyFrameInput,
        stage1: LegacyStage1Artifact,
    ) -> LegacyPreparedFrame:
        """Load local data and replay the existing stage-1 mask if present."""

    def refinement_attempt(
        self,
        options: LegacyExecutionOptions,
        prepared: LegacyPreparedFrame,
        parent: LegacyEstimate,
        iteration: int,
        match_seed: int | None,
    ) -> LegacyRefinementAttempt:
        """Render one parent-pose view, match it, and return its PnP attempt."""


@dataclass(frozen=True)
class LegacyBatchExecution:
    """Result paths and records for one full historical-refinement invocation."""

    options: LegacyExecutionOptions
    frames: tuple[LegacyFrameInput, ...]
    records: tuple[Mapping[str, object], ...]
    summary: Mapping[str, object]

    def validate(self) -> None:
        self.options.validate()
        if len(self.frames) != len(self.records):
            raise ValueError("legacy batch frames and records must have equal length")
        for frame in self.frames:
            frame.validate()
        expected = tuple(frame.index for frame in self.frames)
        observed = tuple(int(record.get("frame_index", -1)) for record in self.records)
        if expected != observed:
            raise ValueError("legacy batch records must preserve discovered frame order")


def count_legacy_frame_pairs(dataset_dir: Path) -> int:
    """Count numeric DREAM JSON/RGB pairs in one candidate export directory."""
    return sum(
        1
        for json_path in dataset_dir.glob("*.json")
        if json_path.stem.isdigit() and json_path.with_suffix(".rgb.jpg").is_file()
    )


def resolve_legacy_dataset_dir(dataset_dir: Path) -> Path:
    """Resolve a nested DREAM export exactly like the historical evaluator.

    Public configs name camera-level roots such as ``real/panda-3cam_azure``.
    The original export may contain one additional directory layer holding the
    numbered frame files.  Prefer the directory with the most valid pairs and
    use the same deterministic tie-break order as the archived evaluator.
    """
    source = Path(dataset_dir)
    if count_legacy_frame_pairs(source) > 0:
        return source

    candidates: dict[Path, int] = {}
    for camera_settings in source.rglob("_camera_settings.json"):
        parent = camera_settings.parent
        count = count_legacy_frame_pairs(parent)
        if count > 0:
            candidates[parent] = count
    if not candidates:
        frame_dirs = {
            path.parent
            for path in source.rglob("*.json")
            if path.stem.isdigit()
        }
        for frame_dir in frame_dirs:
            count = count_legacy_frame_pairs(frame_dir)
            if count > 0:
                candidates[frame_dir] = count
    if not candidates:
        return source
    return sorted(
        candidates.items(),
        key=lambda item: (-item[1], len(item[0].parts), str(item[0])),
    )[0][0]


def discover_legacy_frames(options: LegacyExecutionOptions) -> tuple[LegacyFrameInput, ...]:
    """Discover the d57 numeric ``.json``/``.rgb.jpg`` pairs deterministically.

    The historical selection samples *positions in sorted eligible frames* using
    ``default_rng(sample_seed).choice(..., replace=False)`` and then restores
    ascending file order.  Supplying ``frame_indices`` replaces that sampling
    step; it is useful for a one-frame remote smoke test while retaining the
    original numeric filename convention.
    """
    options.validate()
    requested_dataset_dir = Path(options.dataset_dir)
    if not requested_dataset_dir.is_dir():
        raise FileNotFoundError(requested_dataset_dir)
    dataset_dir = resolve_legacy_dataset_dir(requested_dataset_dir)
    eligible: list[LegacyFrameInput] = []
    for json_path in sorted(dataset_dir.glob("*.json")):
        if not json_path.stem.isdigit():
            continue
        index = int(json_path.stem)
        image_path = json_path.with_suffix(".rgb.jpg")
        if image_path.is_file():
            frame = LegacyFrameInput(index, json_path, image_path)
            frame.validate()
            eligible.append(frame)
    if not eligible:
        raise RuntimeError(f"no numeric DREAM json/rgb frame pairs under {dataset_dir}")

    if options.frame_indices:
        requested = set(options.frame_indices)
        selected = [frame for frame in eligible if frame.index in requested]
        missing = sorted(requested - {frame.index for frame in selected})
        if missing:
            raise FileNotFoundError(
                "requested legacy frame indices lack a json/rgb pair: "
                f"{missing[:5]}"
            )
    else:
        if options.sample_count > len(eligible):
            raise ValueError(
                f"sample_count={options.sample_count} exceeds {len(eligible)} eligible frames"
            )
        selected_positions = np.random.default_rng(options.sample_seed).choice(
            len(eligible),
            size=options.sample_count,
            replace=False,
        )
        selected = [eligible[int(position)] for position in sorted(selected_positions)]

    if options.limit is not None:
        selected = selected[: options.limit]
    if not selected:
        raise RuntimeError("legacy frame selection is empty")
    return tuple(selected)


def _replay_stage1_input_mask(
    options: LegacyExecutionOptions,
    stage1: LegacyStage1Artifact,
    observed_rgb: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, Path | None, InputMaskSource]:
    """Apply and persist the d57 Stage-1 mask replay rule.

    The old refiner only reused ``initial/input_mask.png`` when masking was
    enabled.  It then saved the resulting binary mask again under this run's
    frame directory, making the new result independently resumable.  It never
    reran SAM merely because the archived mask was absent.
    """
    source_mask_path = stage1.input_mask_path if options.mask_input else None
    if source_mask_path is None:
        return observed_rgb, None, "original_rgb_fallback"
    mask = np.asarray(Image.open(source_mask_path).convert("L")) > 0
    if mask.shape != observed_rgb.shape[:2]:
        raise ValueError(
            "archived input_mask.png shape does not match the observed RGB; "
            "legacy replay refuses to resize it"
        )
    observed_masked = observed_rgb.copy()
    observed_masked[~mask] = 0
    output_mask_path = Path(output_dir) / "input_mask.png"
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_mask_path)
    return observed_masked, output_mask_path, "stage1_input_mask"


def _copy_legacy_pose_artifact(source: Path | None, target: Path) -> Path | None:
    """Copy an existing historical pose NPZ without changing its payload."""
    if source is None or not Path(source).is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _copy_existing_artifact(source: str | Path | None, target: Path) -> Path | None:
    """Copy an optional result artifact into this run's portable frame tree."""
    if source is None or not Path(source).is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _frame_input_mask_record(prepared: LegacyPreparedFrame, options: LegacyExecutionOptions) -> dict[str, object]:
    return {
        "enabled": bool(options.mask_input),
        "prompt": options.mask_prompt,
        "source": prepared.input_mask_source,
        "mask_path": (
            None if prepared.input_mask_path is None else str(prepared.input_mask_path)
        ),
        "sam_rerun": False,
    }


def _trajectory_record(
    prepared: LegacyPreparedFrame,
    trajectory,
    options: LegacyExecutionOptions,
    *,
    stage0_pose_path: Path | None = None,
    final_pose_path: Path | None = None,
    final_visualization_path: Path | None = None,
) -> dict[str, object]:
    """Materialize the historical state machine without generic-runner fields."""
    iterations = [dict(item) for item in trajectory.iterations]
    if stage0_pose_path is not None and iterations:
        iterations[0]["pose_npz"] = str(stage0_pose_path)
    record: dict[str, object] = {
        "status": trajectory.frame_status,
        "frame_index": prepared.frame.index,
        "json": str(prepared.frame.json_path),
        "image": str(prepared.frame.image_path),
        "semantic_id": trajectory.semantic_id,
        "execution_semantic_id": LEGACY_EXECUTION_SEMANTIC_ID,
        "initial_results_frame_dir": str(prepared.stage1.frame_dir),
        "input_mask": _frame_input_mask_record(prepared, options),
        "views": prepared.stage1.views,
        "iterations": iterations,
    }
    if trajectory.reason is not None:
        record["reason"] = trajectory.reason
    if trajectory.final_estimate is not None:
        record.update(trajectory.final_estimate.summary_pose_fields())
        if final_pose_path is not None:
            record["pose_npz"] = str(final_pose_path)
        if final_visualization_path is not None:
            record["keypoint_visualization"] = str(final_visualization_path)
        record["best_render_path"] = trajectory.final_estimate.render_path
        record["best_camera_npz"] = trajectory.final_estimate.camera_npz
    return record


def _error_record(
    frame: LegacyFrameInput,
    options: LegacyExecutionOptions,
    error: Exception,
    prepared: LegacyPreparedFrame | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "status": "error",
        "frame_index": frame.index,
        "json": str(frame.json_path),
        "image": str(frame.image_path),
        "semantic_id": options.semantic_id,
        "execution_semantic_id": LEGACY_EXECUTION_SEMANTIC_ID,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    if prepared is not None:
        record["input_mask"] = _frame_input_mask_record(prepared, options)
    return record


def _assert_output_destination(options: LegacyExecutionOptions) -> None:
    """Avoid overwriting a distinct historical result when resume is disabled."""
    output_dir = Path(options.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.is_dir() and not options.resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty legacy output_dir without resume: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _file_content_identity(path: Path) -> dict[str, object]:
    """Return one local file's immutable content identity for resume safety."""
    source = Path(path)
    resolved = source.resolve()
    if not source.is_file():
        return {"path": str(resolved), "state": "missing"}
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while calculating its resume identity: {source}")
    return {
        "path": str(resolved),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _stage1_content_identity(
    options: LegacyExecutionOptions,
    frame: LegacyFrameInput,
) -> dict[str, object]:
    """Hash the exact archived files that the legacy reader can consume."""
    frame_dir = Path(options.initial_results_dir) / f"{frame.index:06d}"
    summary_path = frame_dir / "frame_summary.json"
    identity: dict[str, object] = {
        "summary": _file_content_identity(summary_path),
        "best_pose": _file_content_identity(frame_dir / "best_pose.npz"),
    }
    if options.mask_input:
        identity["input_mask"] = _file_content_identity(frame_dir / "input_mask.png")
    if summary_path.is_file() and not (frame_dir / "best_pose.npz").is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = None
        if isinstance(summary, Mapping) and summary.get("pose_npz"):
            identity["summary_pose_fallback"] = _file_content_identity(
                Path(str(summary["pose_npz"]))
            )
    return identity


def _execution_code_content_identity() -> dict[str, object]:
    """Pin the public executor and core bridge sources, including dirty edits."""
    executor_path = Path(__file__).resolve()
    calibx_root = executor_path.parents[2]
    source_root = calibx_root.parent
    project_root = source_root.parent
    paths = (
        executor_path,
        executor_path.with_name("legacy_runner.py"),
        calibx_root / "matching.py",
        calibx_root / "pose.py",
        calibx_root / "benchmarks" / "gate_rendering.py",
        source_root / "romav2" / "benchmarks" / "utils" / "evaluator.py",
        project_root / "pyproject.toml",
        project_root / "uv.lock",
    )
    return {
        str(path.relative_to(project_root)).replace("\\", "/"): _file_content_identity(path)
        for path in paths
    }


def _legacy_run_fingerprint_inputs(
    options: LegacyExecutionOptions,
    frames: tuple[LegacyFrameInput, ...],
) -> dict[str, object]:
    """Return the machine-local inputs that make a resumed run comparable.

    This record intentionally stays next to the private result directory.  It
    contains resolved local paths and content digests solely to reject a resume
    against changed images, Stage-1 inputs, XML, public code, or renderer
    configuration; it is not a publication artifact.
    """
    return {
        "execution_semantic_id": LEGACY_EXECUTION_SEMANTIC_ID,
        "semantic_id": options.semantic_id,
        "source": {
            "dataset_dir": str(Path(options.dataset_dir).resolve()),
            "initial_results_dir": str(Path(options.initial_results_dir).resolve()),
            "mujoco_xml": str(Path(options.mujoco_xml).resolve()),
        },
        "selection": {
            "sample_count": options.sample_count,
            "sample_seed": options.sample_seed,
            "frame_indices": list(options.frame_indices),
            "limit": options.limit,
            "selected_frames": [
                {
                    "frame_index": frame.index,
                    "json": _file_content_identity(frame.json_path),
                    "image": _file_content_identity(frame.image_path),
                    "stage1": _stage1_content_identity(options, frame),
                }
                for frame in frames
            ],
        },
        "stage1": {
            "views": options.views,
            "match_batch_size": options.match_batch_size,
        },
        "refinement": {
            "iterations": options.refinement_iterations,
            "match_seed": options.match_seed,
        },
        "matching": {
            "mask_input": options.mask_input,
            "mask_prompt": options.mask_prompt,
        },
        "render": {
            "visual_geom_group": options.visual_geom_group,
            "visual_body_names": list(options.visual_body_names),
            "width": options.width,
            "height": options.height,
            "camera_settings": (
                {"state": "not_configured"}
                if options.camera_settings is None
                else _file_content_identity(Path(options.camera_settings))
            ),
            "distance_scale": options.distance_scale,
            "min_distance": options.min_distance,
            "elevation": options.elevation,
            "azimuth_offset": options.azimuth_offset,
        },
        "device": options.device,
        "mujoco_xml": _file_content_identity(options.mujoco_xml),
        "runtime_identity": (
            {"state": "not_supplied_resume_disabled"}
            if options.runtime_identity_file is None
            else _file_content_identity(options.runtime_identity_file)
        ),
        "public_execution_code": _execution_code_content_identity(),
    }


def _legacy_run_manifest(
    options: LegacyExecutionOptions,
    frames: tuple[LegacyFrameInput, ...],
) -> dict[str, object]:
    fingerprint_inputs = _legacy_run_fingerprint_inputs(options, frames)
    fingerprint = hashlib.sha256(_canonical_json_bytes(fingerprint_inputs)).hexdigest()
    return {
        "schema_version": LEGACY_RUN_MANIFEST_SCHEMA_VERSION,
        "execution_semantic_id": LEGACY_EXECUTION_SEMANTIC_ID,
        "resume_policy": "exact_content_and_runtime_identity_fingerprint_required",
        "publication_policy": "machine_local_only_do_not_publish",
        "run_fingerprint_sha256": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
    }


def _initialize_or_validate_run_manifest(
    options: LegacyExecutionOptions,
    frames: tuple[LegacyFrameInput, ...],
) -> str:
    """Create a fresh private manifest or prove an existing resume is exact."""
    manifest_path = Path(options.output_dir) / LEGACY_RUN_MANIFEST_FILENAME
    expected = _legacy_run_manifest(options, frames)
    expected_fingerprint = str(expected["run_fingerprint_sha256"])
    if not options.resume:
        _write_json(manifest_path, expected)
        return expected_fingerprint
    if not manifest_path.is_file():
        raise ValueError(
            "legacy resume requires run_manifest.json from the original invocation; "
            "start a new output directory instead"
        )
    try:
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid legacy run manifest: {manifest_path}") from error
    if not isinstance(observed, Mapping):
        raise ValueError(f"legacy run manifest must be an object: {manifest_path}")
    if observed.get("schema_version") != LEGACY_RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("legacy run manifest schema is incompatible with this executor")
    if observed.get("execution_semantic_id") != LEGACY_EXECUTION_SEMANTIC_ID:
        raise ValueError("legacy run manifest has an unexpected execution semantic ID")
    if observed.get("run_fingerprint_sha256") != expected_fingerprint:
        raise ValueError(
            "legacy resume fingerprint differs from the existing run; "
            "do not mix a different dataset, Stage-1 source, selection, or runtime config"
        )
    return expected_fingerprint


def _load_resumed_record(
    options: LegacyExecutionOptions,
    frame: LegacyFrameInput,
    *,
    run_fingerprint: str | None,
) -> Mapping[str, object] | None:
    if not options.resume:
        return None
    if run_fingerprint is None:
        raise ValueError(
            "resuming one legacy frame requires a batch-validated run manifest"
        )
    summary_path = Path(options.output_dir) / f"{frame.index:06d}" / "frame_summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid resumed frame summary: {summary_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"resumed frame summary must be an object: {summary_path}")
    if int(payload.get("frame_index", -1)) != frame.index:
        raise ValueError(f"resumed frame summary has the wrong index: {summary_path}")
    if payload.get("run_fingerprint_sha256") != run_fingerprint:
        raise ValueError(
            f"resumed frame summary belongs to a different legacy run: {summary_path}"
        )
    return dict(payload)


def _prepare_stage1_failure(
    options: LegacyExecutionOptions,
    frame: LegacyFrameInput,
    stage1: LegacyStage1Artifact,
) -> LegacyPreparedFrame:
    """Produce a recordable R0-failure frame without loading GPU/MuJoCo state.

    A failed archived Stage 1 never reaches R1.  Historically it therefore did
    not need a model, renderer, matcher, or fresh SAM invocation.  Keeping this
    lightweight branch outside the concrete backend also prevents an unrelated
    runtime dependency failure from relabelling an archived R0 failure as an
    executor error.
    """
    observed_rgb = np.asarray(Image.open(frame.image_path).convert("RGB"))
    observed_for_match, input_mask_path, mask_source = _replay_stage1_input_mask(
        options,
        stage1,
        observed_rgb,
        Path(options.output_dir) / f"{frame.index:06d}",
    )
    prepared = LegacyPreparedFrame(
        frame=frame,
        stage1=stage1,
        output_dir=Path(options.output_dir) / f"{frame.index:06d}",
        observed_rgb=observed_rgb,
        observed_for_match=observed_for_match,
        input_mask_path=input_mask_path,
        input_mask_source=mask_source,
        runtime=None,
    )
    prepared.validate()
    return prepared


def execute_legacy_frame(
    options: LegacyExecutionOptions,
    frame: LegacyFrameInput,
    backend: LegacyIterationBackend,
    *,
    run_fingerprint: str | None = None,
) -> Mapping[str, object]:
    """Replay one archived R0 result through the old unconditional R1--R3 flow."""
    options.validate()
    frame.validate()
    resumed = _load_resumed_record(
        options,
        frame,
        run_fingerprint=run_fingerprint,
    )
    if resumed is not None:
        return resumed

    frame_dir = Path(options.output_dir) / f"{frame.index:06d}"
    prepared: LegacyPreparedFrame | None = None
    try:
        stage1 = load_legacy_stage1_artifact(options.initial_results_dir, frame.index)
        stage0_pose_path = _copy_legacy_pose_artifact(
            stage1.pose_path,
            frame_dir / "iterations" / "iteration_00" / "pose.npz",
        )
        prepared = (
            _prepare_stage1_failure(options, frame, stage1)
            if stage1.estimate is None
            else backend.prepare_frame(options, frame, stage1)
        )
        if not isinstance(prepared, LegacyPreparedFrame):
            raise TypeError("prepare_frame must return LegacyPreparedFrame")
        prepared.validate()
        if prepared.frame != frame:
            raise ValueError("prepared frame does not match requested frame")
        # ``LegacyStage1Artifact`` contains NumPy arrays, so structural dataclass
        # equality would be ambiguous.  The backend receives this exact loaded
        # artifact and must preserve that identity rather than reconstruct it.
        if prepared.stage1 is not stage1:
            raise ValueError("prepared Stage-1 artifact does not match the loaded artifact")
        if Path(prepared.output_dir) != frame_dir:
            raise ValueError("prepared output_dir does not match the requested frame output")
        if not options.mask_input and prepared.input_mask_path is not None:
            raise ValueError("mask_input=false must not replay a Stage-1 input mask")
        if prepared.input_mask_source == "stage1_input_mask":
            expected_mask_path = frame_dir / "input_mask.png"
            if (
                prepared.input_mask_path is None
                or Path(prepared.input_mask_path).resolve()
                != expected_mask_path.resolve()
                or not expected_mask_path.is_file()
            ):
                raise ValueError(
                    "replayed Stage-1 input mask must be persisted as this run's "
                    "frame_dir/input_mask.png"
                )

        def attempt(
            parent: LegacyEstimate,
            iteration: int,
            match_seed: int | None,
        ) -> LegacyRefinementAttempt:
            result = backend.refinement_attempt(
                options,
                prepared,
                parent,
                iteration,
                match_seed,
            )
            if not isinstance(result, LegacyRefinementAttempt):
                raise TypeError("refinement_attempt must return LegacyRefinementAttempt")
            return result

        trajectory = run_legacy_unconditional_trajectory(
            stage1,
            frame_index=frame.index,
            refinement_iterations=options.refinement_iterations,
            match_seed=options.match_seed,
            refinement_attempt=attempt,
        )
        final_pose_path = _copy_legacy_pose_artifact(
            (
                None
                if trajectory.final_estimate is None
                or trajectory.final_estimate.pose_npz is None
                else Path(trajectory.final_estimate.pose_npz)
            ),
            frame_dir / "best_pose.npz",
        )
        final_visualization_path = _copy_existing_artifact(
            (
                None
                if trajectory.final_estimate is None
                else trajectory.final_estimate.keypoint_visualization
            ),
            frame_dir / "best_keypoint_eval.jpg",
        )
        record = _trajectory_record(
            prepared,
            trajectory,
            options,
            stage0_pose_path=stage0_pose_path,
            final_pose_path=final_pose_path,
            final_visualization_path=final_visualization_path,
        )
    except Exception as error:
        record = _error_record(frame, options, error, prepared)
    if run_fingerprint is not None:
        record["run_fingerprint_sha256"] = run_fingerprint
    _write_json(frame_dir / "frame_summary.json", record)
    return record


def _legacy_summary(records: list[Mapping[str, object]]) -> dict[str, object]:
    """Use the existing success-only aggregation without relabelling its meaning."""
    aggregate = summarize(records, auc_threshold=0.1, auc_delta=1e-4)
    return {
        "aggregation": "upstream_success_only_summary",
        "summary": aggregate,
        "n_frames": len(records),
        "n_success": sum(record.get("status") == "success" for record in records),
        "n_failed_or_error": sum(record.get("status") != "success" for record in records),
    }


def execute_legacy_batch(
    options: LegacyExecutionOptions,
    backend: LegacyIterationBackend,
) -> LegacyBatchExecution:
    """Run all selected frames, retaining per-frame errors as historical records."""
    options.validate()
    _assert_output_destination(options)
    if not Path(options.initial_results_dir).is_dir():
        raise FileNotFoundError(options.initial_results_dir)
    if not Path(options.mujoco_xml).is_file():
        raise FileNotFoundError(options.mujoco_xml)
    if (
        options.runtime_identity_file is not None
        and not Path(options.runtime_identity_file).is_file()
    ):
        raise FileNotFoundError(options.runtime_identity_file)
    frames = discover_legacy_frames(options)
    run_fingerprint = _initialize_or_validate_run_manifest(options, frames)
    records: list[Mapping[str, object]] = []
    for ordinal, frame in enumerate(frames, start=1):
        print(f"[{ordinal}/{len(frames)}] legacy DREAM frame {frame.index:06d}")
        record = execute_legacy_frame(
            options,
            frame,
            backend,
            run_fingerprint=run_fingerprint,
        )
        records.append(record)
        _write_json(
            Path(options.output_dir) / "summary_running.json",
            {
                "run_fingerprint_sha256": run_fingerprint,
                **_legacy_summary(records),
            },
        )
    summary = _legacy_summary(records)
    final_payload: dict[str, object] = {
        "execution_semantic_id": LEGACY_EXECUTION_SEMANTIC_ID,
        "semantic_id": options.semantic_id,
        "run_fingerprint_sha256": run_fingerprint,
        "run_manifest": LEGACY_RUN_MANIFEST_FILENAME,
        "publication_policy": "machine_local_only_do_not_publish",
        "dataset_dir": str(options.dataset_dir),
        "initial_results_dir": str(options.initial_results_dir),
        "mujoco_xml": str(options.mujoco_xml),
        "stage1": {"views": options.views, "match_batch_size": options.match_batch_size},
        "refinement_iterations": options.refinement_iterations,
        "mask": {
            "configured": options.mask_input,
            "sam_rerun": False,
            "fallback": "original_rgb_when_stage1_input_mask_is_missing",
        },
        "frames": list(records),
        **summary,
    }
    _write_json(Path(options.output_dir) / "summary.json", final_payload)
    execution = LegacyBatchExecution(
        options=options,
        frames=frames,
        records=tuple(records),
        summary=final_payload,
    )
    execution.validate()
    return execution


# -- Optional bridge to the existing Calib-X/MuJoCo implementation ---------


@dataclass(frozen=True)
class _CoreLegacyRuntime:
    """Machine-local state intentionally absent from public manifests/records."""

    payload: Mapping[str, Any]
    camera_matrix: np.ndarray
    model: object
    data: object
    fk_points: Mapping[str, np.ndarray]
    matcher_args: object


class CalibxLegacyIterationBackend:
    """Concrete one-render adapter around the existing RoMaV2/PnP primitives.

    It is deliberately constructed only by a real ``--execute`` command.  No
    SAM extractor appears here: the old d57 refinement either reuses the
    archived ``input_mask.png`` verbatim or falls back to unmasked original
    RGB.  This prevents an unavailable or changed SAM model from silently
    altering a historical replay.
    """

    def __init__(self, options: LegacyExecutionOptions) -> None:
        options.validate()
        self._options = options
        self._matcher: object | None = None

    @staticmethod
    def _runtime_args(
        options: LegacyExecutionOptions,
        frame: LegacyFrameInput,
        *,
        width: int,
        height: int,
    ) -> object:
        """Construct just the existing-core namespace needed by one render."""
        import argparse

        from calibx.matching import (
            DEFAULT_RANSAC_CONFIDENCE,
            DEFAULT_RANSAC_MAX_ITER,
            DEFAULT_RANSAC_METHOD,
            DEFAULT_RANSAC_REPROJ_THRESHOLD,
        )

        return argparse.Namespace(
            json=frame.json_path,
            image=frame.image_path,
            device=options.device,
            matcher="RoMaV2",
            width=width,
            height=height,
            distance_scale=options.distance_scale,
            min_distance=options.min_distance,
            elevation=options.elevation,
            azimuth_offset=options.azimuth_offset,
            max_keypoints=2048,
            detect_threshold=0.005,
            match_threshold=0.2,
            score_filter=0.0,
            ransac_method=DEFAULT_RANSAC_METHOD,
            ransac_threshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
            ransac_confidence=DEFAULT_RANSAC_CONFIDENCE,
            ransac_max_iter=DEFAULT_RANSAC_MAX_ITER,
            pnp_threshold=5.0,
            min_pnp_correspondences=6,
            max_pnp_correspondences=512,
            save_all_matches=options.save_all_matches,
            camera_settings=options.camera_settings,
            fx=None,
            fy=None,
            cx=None,
            cy=None,
            fallback_focal=400.0,
            visual_geom_group=options.visual_geom_group,
        )

    def _matcher_for(self, matcher_args: object) -> object:
        if self._matcher is None:
            from calibx.matching import ImcuiMatcher

            self._matcher = ImcuiMatcher(matcher_args)
        return self._matcher

    def prepare_frame(
        self,
        options: LegacyExecutionOptions,
        frame: LegacyFrameInput,
        stage1: LegacyStage1Artifact,
    ) -> LegacyPreparedFrame:
        if options != self._options:
            raise ValueError("CalibxLegacyIterationBackend was created for different options")
        frame.validate()
        observed_rgb = np.asarray(Image.open(frame.image_path).convert("RGB"))
        observed_masked, input_mask_path, mask_source = _replay_stage1_input_mask(
            options,
            stage1,
            observed_rgb,
            Path(options.output_dir) / f"{frame.index:06d}",
        )

        height = observed_masked.shape[0] if options.height is None else options.height
        width = observed_masked.shape[1] if options.width is None else options.width
        if (height, width) == observed_masked.shape[:2]:
            observed_for_match = observed_masked
        else:
            observed_for_match = np.asarray(
                Image.fromarray(observed_masked).resize((width, height), Image.BILINEAR)
            )

        # All of these imports are intentionally inside a real frame-preparation
        # call.  A dry plan and a fake-backend test need none of them.
        from calibx.pose import (
            dream_keypoints,
            fk_keypoints,
            load_camera_matrix,
            load_dream_payload,
            make_model_and_data,
        )

        matcher_args = self._runtime_args(
            options,
            frame,
            width=width,
            height=height,
        )
        payload = load_dream_payload(frame.json_path)
        camera_matrix = load_camera_matrix(matcher_args, observed_for_match.shape)
        model, data = make_model_and_data(
            options.mujoco_xml,
            width,
            height,
            camera_matrix,
            payload,
            options.visual_body_names,
            options.visual_geom_group,
        )
        runtime = _CoreLegacyRuntime(
            payload=payload,
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            model=model,
            data=data,
            fk_points=fk_keypoints(model, data, dream_keypoints(payload)),
            matcher_args=matcher_args,
        )
        prepared = LegacyPreparedFrame(
            frame=frame,
            stage1=stage1,
            output_dir=Path(options.output_dir) / f"{frame.index:06d}",
            observed_rgb=observed_rgb,
            observed_for_match=observed_for_match,
            input_mask_path=input_mask_path,
            input_mask_source=mask_source,
            runtime=runtime,
        )
        prepared.validate()
        return prepared

    @staticmethod
    def _annotate_render_camera(
        camera_path: Path,
        parent: LegacyEstimate,
    ) -> None:
        """Keep the parent pose next to the projected render for audit only."""
        with np.load(camera_path, allow_pickle=False) as saved:
            payload = {name: saved[name] for name in saved.files}
        payload["source_world_to_camera"] = parent.pose.world_to_camera
        payload["source_camera_to_robot_base"] = parent.pose.camera_to_world
        np.savez_compressed(camera_path, **payload)

    def refinement_attempt(
        self,
        options: LegacyExecutionOptions,
        prepared: LegacyPreparedFrame,
        parent: LegacyEstimate,
        iteration: int,
        match_seed: int | None,
    ) -> LegacyRefinementAttempt:
        if options != self._options:
            raise ValueError("CalibxLegacyIterationBackend was created for different options")
        prepared.validate()
        if iteration not in {1, 2, 3}:
            raise ValueError("legacy runtime refinement iteration must be in [1, 3]")
        runtime = prepared.runtime
        if not isinstance(runtime, _CoreLegacyRuntime):
            raise TypeError("CalibxLegacyIterationBackend received foreign prepared runtime")

        from calibx.benchmarks.gate_rendering import (
            PoseAlignedRenderSettings,
            render_pose_aligned_artifacts,
        )
        from calibx.pose import draw_keypoint_eval, keypoint_metrics, match_one_render

        iteration_dir = prepared.output_dir / "iterations" / f"iteration_{iteration:02d}"
        render_path, _mask_path, camera_path, _render_mask = render_pose_aligned_artifacts(
            runtime.model,
            runtime.data,
            PoseAlignedRenderSettings(
                width=int(np.asarray(prepared.observed_for_match).shape[1]),
                height=int(np.asarray(prepared.observed_for_match).shape[0]),
                visual_geom_group=options.visual_geom_group,
            ),
            runtime.camera_matrix,
            parent.pose.world_to_camera,
            iteration_dir,
            stem="projected",
        )
        self._annotate_render_camera(camera_path, parent)
        configure_legacy_match_rng(match_seed)
        best = match_one_render(
            prepared.observed_for_match,
            render_path,
            self._matcher_for(runtime.matcher_args),
            runtime.model,
            runtime.data,
            runtime.camera_matrix,
            runtime.matcher_args,
            iteration_dir,
        )
        if best["pnp"].get("status") != "success":
            return LegacyRefinementAttempt(
                render_path=str(render_path),
                camera_npz=str(camera_path),
                match=best,
                candidate=None,
            )

        pose_data = best.get("_pose")
        if not isinstance(pose_data, Mapping):
            raise ValueError("successful one-render PnP is missing its pose arrays")
        metrics = keypoint_metrics(
            dict(runtime.payload),
            dict(runtime.fk_points),
            dict(pose_data),
            runtime.camera_matrix,
        )
        pose = LegacyPoseArtifact(
            world_to_camera=pose_data["world_to_camera"],
            camera_to_world=pose_data["camera_to_world"],
            image_points=pose_data["image_points"],
            world_points=pose_data["world_points"],
            scores=pose_data["scores"],
            inlier_indices=pose_data["inlier_indices"],
            reprojection_errors=pose_data["reprojection_errors"],
        )
        pose_path = save_legacy_pose_npz(
            iteration_dir,
            pose,
            runtime.camera_matrix,
            metrics,
            filename="pose.npz",
        )
        visualization_path: Path | None = None
        if options.save_visualizations:
            visualization_path = iteration_dir / "keypoint_eval.jpg"
            Image.fromarray(
                draw_keypoint_eval(prepared.observed_for_match, metrics)
            ).save(visualization_path, quality=95)
        candidate = LegacyEstimate(
            pose=pose,
            pnp=dict(best["pnp"]),
            render_path=str(render_path),
            camera_npz=str(camera_path),
            keypoint_metrics=metrics,
            pose_npz=str(pose_path),
            keypoint_visualization=(
                None if visualization_path is None else str(visualization_path)
            ),
        )
        return LegacyRefinementAttempt(
            render_path=str(render_path),
            camera_npz=str(camera_path),
            match=best,
            candidate=candidate,
        )


__all__ = [
    "CalibxLegacyIterationBackend",
    "InputMaskSource",
    "LEGACY_EXECUTION_SEMANTIC_ID",
    "LegacyBatchExecution",
    "LegacyExecutionOptions",
    "LegacyFrameInput",
    "LegacyIterationBackend",
    "LegacyPreparedFrame",
    "count_legacy_frame_pairs",
    "discover_legacy_frames",
    "execute_legacy_batch",
    "execute_legacy_frame",
    "resolve_legacy_dataset_dir",
]
