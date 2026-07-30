"""Runtime boundary for the historical CTRNet-X Panda single-frame protocol.

The CF single-frame runs and the later CTRNet-X closed-loop batch runs have
different state transitions.  This module deliberately implements only the
former:

* R0 performs a SAM-masked, six-view orbit search with ``match_batch_size=6``;
* R1--R3 each render *one* image from the current per-frame pose;
* every successful R1--R3 PnP candidate replaces its parent immediately;
* one non-successful refinement PnP fails that frame and skips its later
  rounds.

There is no episode aggregation, retained-parent recovery, gate decision, or
rollback in this module.  The public runtime manifest contains portable frame
identifiers only.  A caller supplies local dataset/model/output handling via
typed hooks, so no private CTRNet-X layout or server path is encoded here.

The pure historical R0--R3 state machine is shared with DREAM-real because it
is the same pre-gate evaluator contract.  The typed CTRNet-X boundary below is
separate so a future dataset adapter cannot accidentally route single-frame
evaluation through the closed-loop batch protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from calibx.benchmarks.common import (
    is_public_metadata_key,
    is_public_relative_identifier as _shared_public_relative_identifier,
)
from calibx.benchmarks.dream_real.legacy import LEGACY_REFINEMENT_SEMANTIC_ID
from calibx.benchmarks.dream_real.legacy_runner import (
    LEGACY_ORBIT_STAGE,
    LEGACY_REFINEMENT_FAILURE_REASON,
    LEGACY_REFINEMENT_STAGE,
    LEGACY_SKIPPED_REASON,
    LegacyEstimate,
    LegacyRefinementAttempt,
    LegacyRefinementTrajectory,
    LegacyStage1Artifact,
    clean_legacy_record,
    configure_legacy_match_rng,
    legacy_match_seed,
    legacy_pose_delta,
)

from .protocol import (
    CTRNETX_SPLITS,
    CtrnetxProtocol,
    assert_full_ctrnetx_single_scope,
)


JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
CtrnetxSingleInputMaskSource = Literal[
    "stage0_input_mask",
    "original_rgb_fallback",
]

CTRNETX_SINGLE_EXECUTION_SEMANTIC_ID = (
    "ctrnetx-single-frame-legacy-unconditional-r0-r3-v1"
)
CTRNETX_SINGLE_RUNTIME_MANIFEST_PROTOCOL = "ctrnetx_single_frame_runtime_frames"


def _is_public_relative_identifier(value: str) -> bool:
    """Keep compatibility with existing local call sites for the shared guard."""
    return _shared_public_relative_identifier(value)


def _public_json_copy(value: object, *, label: str) -> JsonValue:
    """Copy finite JSON metadata while rejecting host-dependent strings."""
    try:
        copied: JsonValue = json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only finite JSON values") from error

    def visit(item: JsonValue, location: str) -> None:
        if isinstance(item, str):
            if not _is_public_relative_identifier(item):
                raise ValueError(
                    f"{location} must not contain a host path, URI, or unsafe identifier"
                )
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not is_public_metadata_key(key):
                    raise ValueError(f"{location} contains an unsafe metadata key")
                visit(child, f"{location}.{key}")

    visit(copied, label)
    return copied


def _safe_artifact_id(value: str | None, *, label: str) -> None:
    if value is not None and not _is_public_relative_identifier(value):
        raise ValueError(f"{label} must be a portable relative identifier")


@dataclass(frozen=True)
class CtrnetxSingleRuntimeFrame:
    """One independent CTRNet-X frame addressed without a local data path.

    ``frame_index`` is explicit because the historical matcher seed was based
    on the evaluator's numeric frame index, not the order in a newly generated
    manifest.  It is deliberately not inferred from ``frame_id``.
    """

    split: str
    frame_id: str
    frame_index: int
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def validate(self) -> None:
        if self.split not in CTRNETX_SPLITS:
            raise ValueError(f"unknown CTRNet-X single split: {self.split!r}")
        if not _is_public_relative_identifier(self.frame_id):
            raise ValueError("frame_id must be a public relative identifier")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        _public_json_copy(dict(self.metadata), label="metadata")

    def to_dict(self) -> dict[str, JsonValue]:
        self.validate()
        return {
            "split": self.split,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "metadata": _public_json_copy(dict(self.metadata), label="metadata"),
        }


@dataclass(frozen=True)
class CtrnetxSingleRuntimeManifest:
    """Portable per-frame layer for the archived three-split single protocol.

    A complete paper manifest has exactly 1,864 full-body, 6,264 partial, and
    9,178 in-and-out records.  Small test/smoke manifests remain valid as
    runtime input but must not be represented as a full paper scope.
    """

    frames: tuple[CtrnetxSingleRuntimeFrame, ...]
    protocol: str = CTRNETX_SINGLE_RUNTIME_MANIFEST_PROTOCOL
    version: int = 1

    def validate(self) -> None:
        if self.protocol != CTRNETX_SINGLE_RUNTIME_MANIFEST_PROTOCOL:
            raise ValueError("unexpected CTRNet-X single runtime manifest protocol")
        if self.version != 1:
            raise ValueError(
                f"unsupported CTRNet-X single runtime manifest version: {self.version}"
            )
        if not self.frames:
            raise ValueError("CTRNet-X single runtime manifest must contain frames")
        for frame in self.frames:
            frame.validate()
        identities = tuple((frame.split, frame.frame_id) for frame in self.frames)
        if len(set(identities)) != len(identities):
            raise ValueError("frame_id values must be unique within each split")
        indices = tuple((frame.split, frame.frame_index) for frame in self.frames)
        if len(set(indices)) != len(indices):
            raise ValueError("frame_index values must be unique within each split")

    def split_frame_counts(self) -> dict[str, int]:
        self.validate()
        return {
            split: sum(frame.split == split for frame in self.frames)
            for split in CTRNETX_SPLITS
            if any(frame.split == split for frame in self.frames)
        }

    def validate_full_scope(self) -> None:
        """Require exactly the archived 17,306-frame single-frame scope."""
        self.validate()
        assert_full_ctrnetx_single_scope(self.split_frame_counts())

    def select(
        self,
        *,
        splits: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[CtrnetxSingleRuntimeFrame, ...]:
        """Select ordered frames without changing their historic frame indices."""
        self.validate()
        selected_splits = tuple(CTRNETX_SPLITS if splits is None else splits)
        if not selected_splits:
            raise ValueError("at least one CTRNet-X split must be selected")
        unknown = [split for split in selected_splits if split not in CTRNETX_SPLITS]
        if unknown:
            raise ValueError(f"unknown CTRNet-X single splits: {unknown}")
        if len(set(selected_splits)) != len(selected_splits):
            raise ValueError("selected CTRNet-X split names must be unique")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when supplied")
        selected = tuple(frame for frame in self.frames if frame.split in selected_splits)
        if not selected:
            raise ValueError("selected CTRNet-X single scope is empty")
        return selected if limit is None else selected[:limit]

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "protocol": self.protocol,
            "version": self.version,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def write(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> "CtrnetxSingleRuntimeManifest":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid CTRNet-X single runtime manifest: {source}") from error
        try:
            frames = tuple(
                CtrnetxSingleRuntimeFrame(
                    split=str(item["split"]),
                    frame_id=str(item["frame_id"]),
                    frame_index=int(item["frame_index"]),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload["frames"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid CTRNet-X single runtime manifest schema: {source}"
            ) from error
        manifest = cls(
            protocol=str(payload.get("protocol", CTRNETX_SINGLE_RUNTIME_MANIFEST_PROTOCOL)),
            version=int(payload.get("version", 1)),
            frames=frames,
        )
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class CtrnetxSingleExecutionConfig:
    """Frozen CF settings for one independent Panda frame at a time."""

    stage0_views: int = 6
    stage0_match_batch_size: int = 6
    refinement_iterations: int = 3
    mask_input: bool = True
    mask_prompt: str = "robotic arm"
    match_seed: int = 90
    visual_geom_group: int = 2

    def validate(self) -> None:
        CtrnetxProtocol(
            mode="single",
            stage1_views=self.stage0_views,
            stage1_match_batch_size=self.stage0_match_batch_size,
            refinement_iterations=self.refinement_iterations,
            mask_input=self.mask_input,
        ).validate()
        if self.match_seed != 90:
            raise ValueError("CTRNet-X paper single-frame runs are frozen at match_seed=90")
        if not self.mask_prompt:
            raise ValueError("mask_prompt must not be empty")
        if self.visual_geom_group != 2:
            raise ValueError("CTRNet-X projected rendering requires visual_geom_group=2")

    def seed_for(self, frame: CtrnetxSingleRuntimeFrame, iteration: int) -> int:
        """Return the exact old per-frame matcher seed for R0 through R3."""
        self.validate()
        frame.validate()
        if iteration not in {0, 1, 2, 3}:
            raise ValueError("CTRNet-X single iteration must be in [0, 3]")
        seed = legacy_match_seed(self.match_seed, frame.frame_index, iteration)
        if seed is None:  # Defensive: the frozen public protocol always has 90.
            raise AssertionError("CTRNet-X single match seed unexpectedly absent")
        return seed


@dataclass(frozen=True)
class CtrnetxSingleStage0Request:
    """One archived six-view R0 request for a local runtime adapter."""

    frame: CtrnetxSingleRuntimeFrame
    match_seed: int
    views: int = 6
    match_batch_size: int = 6
    mask_input: bool = True
    mask_prompt: str = "robotic arm"
    visual_geom_group: int = 2
    iteration: int = 0

    def validate(self) -> None:
        self.frame.validate()
        if self.iteration != 0:
            raise ValueError("CTRNet-X single Stage 0 must have iteration=0")
        if self.views != 6 or self.match_batch_size != 6:
            raise ValueError("CTRNet-X single Stage 0 requires six views and batch size six")
        if not self.mask_input:
            raise ValueError("CTRNet-X single Stage 0 requires SAM input masking")
        if not self.mask_prompt:
            raise ValueError("mask_prompt must not be empty")
        if self.visual_geom_group != 2:
            raise ValueError("CTRNet-X Stage 0 rendering requires visual_geom_group=2")
        if isinstance(self.match_seed, bool) or int(self.match_seed) < 0:
            raise ValueError("Stage 0 match_seed must be a non-negative integer")


@dataclass(frozen=True)
class CtrnetxSingleStage0Artifact:
    """A six-view R0 result plus portable artifact/mask audit fields.

    ``legacy_stage1`` is runtime-only and may point to a local historical
    result directory.  The public record uses ``artifact_id`` and
    ``input_mask_artifact_id`` instead, so the manifest never contains a
    server path.  When an existing Stage-0 ``input_mask.png`` is available it
    must be replayed for R1--R3; a fresh SAM pass is forbidden here.
    """

    legacy_stage1: LegacyStage1Artifact
    artifact_id: str | None = None
    input_mask_source: CtrnetxSingleInputMaskSource = "original_rgb_fallback"
    input_mask_artifact_id: str | None = None
    render_count: int = 6
    match_batch_size: int = 6
    match_seed: int | None = None

    @classmethod
    def from_legacy_stage1(
        cls,
        legacy_stage1: LegacyStage1Artifact,
        *,
        artifact_id: str | None = None,
        input_mask_artifact_id: str | None = None,
        match_seed: int | None = None,
    ) -> "CtrnetxSingleStage0Artifact":
        """Wrap an existing R0 artifact without exposing its local paths."""
        source: CtrnetxSingleInputMaskSource = (
            "stage0_input_mask"
            if legacy_stage1.input_mask_path is not None
            else "original_rgb_fallback"
        )
        return cls(
            legacy_stage1=legacy_stage1,
            artifact_id=artifact_id,
            input_mask_source=source,
            input_mask_artifact_id=input_mask_artifact_id,
            match_seed=match_seed,
        )

    def validate(self) -> None:
        # ``LegacyStage1Artifact`` validates its input in ``__post_init__``.
        if len(self.legacy_stage1.views) != 6:
            raise ValueError("CTRNet-X single Stage 0 artifact must record six views")
        if self.render_count != 6 or self.match_batch_size != 6:
            raise ValueError("CTRNet-X single Stage 0 artifact must be six-view/batch-six")
        if self.match_seed is not None and (
            isinstance(self.match_seed, bool) or int(self.match_seed) < 0
        ):
            raise ValueError("Stage 0 artifact match_seed must be non-negative")
        _safe_artifact_id(self.artifact_id, label="Stage 0 artifact_id")
        _safe_artifact_id(
            self.input_mask_artifact_id,
            label="Stage 0 input_mask_artifact_id",
        )
        actual_mask = self.legacy_stage1.input_mask_path
        if self.input_mask_source == "stage0_input_mask":
            if actual_mask is None:
                raise ValueError(
                    "stage0_input_mask requires an existing Stage-0 input_mask.png"
                )
            if self.input_mask_artifact_id is None:
                raise ValueError(
                    "stage0_input_mask requires a portable input_mask_artifact_id"
                )
        elif self.input_mask_source == "original_rgb_fallback":
            if actual_mask is not None:
                raise ValueError(
                    "an existing Stage-0 input_mask.png must be replayed, not ignored"
                )
            if self.input_mask_artifact_id is not None:
                raise ValueError(
                    "original_rgb_fallback must not name an input mask artifact"
                )
        else:
            raise ValueError("unsupported Stage 0 input mask source")

    @property
    def runtime_input_mask_path(self) -> Path | None:
        """Return the local R0 mask for a runtime hook, without serializing it."""
        self.validate()
        return self.legacy_stage1.input_mask_path


@dataclass(frozen=True)
class CtrnetxSingleRefinementRequest:
    """One one-render R1--R3 request handed to a local runtime adapter."""

    frame: CtrnetxSingleRuntimeFrame
    stage0: CtrnetxSingleStage0Artifact
    parent: LegacyEstimate
    iteration: int
    match_seed: int
    visual_geom_group: int = 2
    render_count: int = 1

    def validate(self) -> None:
        self.frame.validate()
        self.stage0.validate()
        if self.iteration not in {1, 2, 3}:
            raise ValueError("CTRNet-X single refinement iteration must be in [1, 3]")
        if self.render_count != 1:
            raise ValueError("CTRNet-X single refinement requires exactly one render")
        if self.visual_geom_group != 2:
            raise ValueError("CTRNet-X projected rendering requires visual_geom_group=2")
        if isinstance(self.match_seed, bool) or int(self.match_seed) < 0:
            raise ValueError("refinement match_seed must be a non-negative integer")

    @property
    def runtime_input_mask_path(self) -> Path | None:
        """Expose only the already-created R0 mask to the local adapter."""
        self.validate()
        return self.stage0.runtime_input_mask_path


Stage0Executor = Callable[[CtrnetxSingleStage0Request], CtrnetxSingleStage0Artifact]
SingleFrameRefiner = Callable[
    [CtrnetxSingleRefinementRequest],
    LegacyRefinementAttempt,
]


@dataclass(frozen=True)
class CtrnetxSingleHooks:
    """Dataset/runtime operations kept outside the public source manifest."""

    run_stage0: Stage0Executor
    refine_one: SingleFrameRefiner

    def validate(self) -> None:
        for name, callback in (
            ("run_stage0", self.run_stage0),
            ("refine_one", self.refine_one),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")


@dataclass(frozen=True)
class CtrnetxSingleFrameExecution:
    """One completed R0--R3 independent-frame transition record."""

    semantic_id: str
    frame: CtrnetxSingleRuntimeFrame
    stage0: CtrnetxSingleStage0Artifact | None
    trajectory: LegacyRefinementTrajectory | None
    stage0_match_seed: int
    executed_refinement_match_seeds: tuple[int, ...] = ()
    outer_error_type: str | None = None

    def validate(self) -> None:
        if self.semantic_id != CTRNETX_SINGLE_EXECUTION_SEMANTIC_ID:
            raise ValueError("unexpected CTRNet-X single execution semantic ID")
        self.frame.validate()
        if self.stage0 is not None:
            self.stage0.validate()
        if self.trajectory is None:
            if not self.outer_error_type:
                raise ValueError("an outer frame error requires outer_error_type")
            if self.stage0 is None and self.executed_refinement_match_seeds:
                raise ValueError("a Stage-0 runtime error cannot attempt refinement seeds")
        else:
            if self.stage0 is None:
                raise ValueError("a completed trajectory requires a Stage-0 artifact")
            if self.outer_error_type is not None:
                raise ValueError("a completed trajectory cannot carry an outer error")
            if self.trajectory.semantic_id != LEGACY_REFINEMENT_SEMANTIC_ID:
                raise ValueError(
                    "CTRNet-X single must use the legacy unconditional state machine"
                )
            if len(self.trajectory.iterations) != 4:
                raise ValueError("CTRNet-X single paper protocol must record R0 through R3")
        if isinstance(self.stage0_match_seed, bool) or int(self.stage0_match_seed) < 0:
            raise ValueError("stage0_match_seed must be a non-negative integer")
        for seed in self.executed_refinement_match_seeds:
            if isinstance(seed, bool) or int(seed) < 0:
                raise ValueError("refinement match seeds must be non-negative integers")
        if len(self.executed_refinement_match_seeds) > 3:
            raise ValueError("CTRNet-X single cannot attempt more than three refinements")

    @staticmethod
    def _iteration_audit(iteration: Mapping[str, Any]) -> dict[str, object]:
        """Keep a portable state audit without serializing local output paths."""
        row: dict[str, object] = {
            "iteration": int(iteration["iteration"]),
            "stage": str(iteration["stage"]),
            "status": str(iteration["status"]),
        }
        if "match_seed" in iteration:
            row["match_seed"] = int(iteration["match_seed"])
        if "reason" in iteration:
            row["reason"] = str(iteration["reason"])
        pnp = iteration.get("best_pnp")
        if not isinstance(pnp, Mapping):
            match = iteration.get("match")
            pnp = match.get("pnp") if isinstance(match, Mapping) else None
        if isinstance(pnp, Mapping) and isinstance(pnp.get("status"), str):
            row["pnp_status"] = pnp["status"]
        return row

    def audit_record(self) -> dict[str, object]:
        """Return serializable protocol evidence with no runtime path values."""
        self.validate()
        trajectory = self.trajectory
        stage0 = self.stage0
        stage0_record: dict[str, object] = {
            "views": 6 if stage0 is None else stage0.render_count,
            "match_batch_size": 6 if stage0 is None else stage0.match_batch_size,
            "match_seed": self.stage0_match_seed,
            "visual_geom_group": 2,
            "artifact_id": None if stage0 is None else stage0.artifact_id,
            "pnp_success": (
                None if stage0 is None else stage0.legacy_stage1.is_successful
            ),
            "input_mask": {
                "source": (
                    "not_available_due_outer_error"
                    if stage0 is None
                    else stage0.input_mask_source
                ),
                "artifact_id": (
                    None if stage0 is None else stage0.input_mask_artifact_id
                ),
                "sam_rerun": False,
            },
        }
        return {
            "semantic_id": self.semantic_id,
            "legacy_semantic_id": LEGACY_REFINEMENT_SEMANTIC_ID,
            "split": self.frame.split,
            "frame_id": self.frame.frame_id,
            "frame_index": self.frame.frame_index,
            "stage0": stage0_record,
            "frame_status": "error" if trajectory is None else trajectory.frame_status,
            "reason": (
                "outer_frame_runtime_error"
                if trajectory is None
                else trajectory.reason
            ),
            "outer_error_type": self.outer_error_type,
            "executed_refinement_match_seeds": [
                {"iteration": iteration, "match_seed": seed}
                for iteration, seed in enumerate(
                    self.executed_refinement_match_seeds,
                    start=1,
                )
            ],
            "iterations": (
                []
                if trajectory is None
                else [
                    self._iteration_audit(iteration)
                    for iteration in trajectory.iterations
                ]
            ),
        }


def configure_ctrnetx_single_match_rng(seed: int) -> None:
    """Apply the historic Torch/NumPy/OpenCV seed immediately before matching.

    A real Stage-0 or one-render refinement hook calls this helper immediately
    before its matcher.  It remains lazy through ``legacy_runner`` so manifest
    validation and fake-backend tests require no GPU evaluation packages.
    """
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("CTRNet-X single matcher seed must be non-negative")
    configure_legacy_match_rng(seed)


def _bound_stage0_artifact(
    artifact: CtrnetxSingleStage0Artifact,
    request: CtrnetxSingleStage0Request,
) -> CtrnetxSingleStage0Artifact:
    if not isinstance(artifact, CtrnetxSingleStage0Artifact):
        raise TypeError("run_stage0 hook must return CtrnetxSingleStage0Artifact")
    request.validate()
    artifact.validate()
    if artifact.match_seed is not None and artifact.match_seed != request.match_seed:
        raise ValueError("Stage 0 artifact match_seed does not match its request")
    return replace(artifact, match_seed=request.match_seed)


def _stage0_record(stage0: LegacyStage1Artifact) -> dict[str, Any]:
    """Materialize the d57 R0 record without the later callback-error extension."""
    if stage0.estimate is None:
        return {
            "iteration": 0,
            "stage": LEGACY_ORBIT_STAGE,
            "status": "failed",
            "best_render_path": stage0.summary.get("best_render_path"),
            "best_pnp": stage0.initial_failure_pnp(),
            "views": stage0.views,
        }
    fields = stage0.estimate.summary_pose_fields()
    fields.pop("camera_npz")
    return {
        "iteration": 0,
        "stage": LEGACY_ORBIT_STAGE,
        "status": "success",
        "best_render_path": stage0.estimate.render_path,
        "best_camera_npz": stage0.estimate.camera_npz,
        **fields,
        "views": stage0.views,
    }


def _skipped_iteration_record(iteration: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "stage": LEGACY_REFINEMENT_STAGE,
        "status": "skipped",
        "reason": LEGACY_SKIPPED_REASON,
    }


def _run_ctrnetx_single_legacy_trajectory(
    stage0: LegacyStage1Artifact,
    *,
    frame_index: int,
    refinement_iterations: int,
    match_seed: int,
    refinement_attempt: Callable[[LegacyEstimate, int, int], LegacyRefinementAttempt],
) -> LegacyRefinementTrajectory:
    """Run d57 PnP transitions and let runtime exceptions escape to the frame.

    The generic DREAM compatibility runner has a versioned callback-error audit
    extension.  Historical d57 single-frame execution instead let an unexpected
    callback exception reach its outer per-frame handler.  Keeping that
    difference here prevents an ``error`` + later ``skipped`` trace from being
    misrepresented as a historical CTRNet-X single result.
    """
    if refinement_iterations != 3:
        raise ValueError("CTRNet-X single historical protocol is frozen at R3")
    iterations: list[dict[str, Any]] = [_stage0_record(stage0)]
    if stage0.estimate is None:
        iterations.extend(_skipped_iteration_record(iteration) for iteration in range(1, 4))
        return LegacyRefinementTrajectory(
            semantic_id=LEGACY_REFINEMENT_SEMANTIC_ID,
            frame_status="failed",
            reason="No render view produced a valid PnP pose.",
            iterations=tuple(iterations),
            final_estimate=None,
        )

    current = stage0.estimate
    for iteration in range(1, 4):
        seed = legacy_match_seed(match_seed, frame_index, iteration)
        if seed is None:  # The frozen protocol always supplies base seed 90.
            raise AssertionError("CTRNet-X single refinement match seed unexpectedly absent")
        attempt = refinement_attempt(current, iteration, seed)
        if not isinstance(attempt, LegacyRefinementAttempt):
            raise TypeError("refine_one hook must return LegacyRefinementAttempt")
        clean_match = clean_legacy_record(attempt.match)
        if attempt.candidate is None:
            iterations.append(
                {
                    "iteration": iteration,
                    "stage": LEGACY_REFINEMENT_STAGE,
                    "status": "failed",
                    "render_source_world_to_camera": current.pnp.get(
                        "world_to_camera"
                    ),
                    "render_path": str(attempt.render_path),
                    "camera_npz": attempt.camera_npz,
                    "match": clean_match,
                    "reason": LEGACY_REFINEMENT_FAILURE_REASON,
                }
            )
            iterations.extend(
                _skipped_iteration_record(skipped)
                for skipped in range(iteration + 1, 4)
            )
            return LegacyRefinementTrajectory(
                semantic_id=LEGACY_REFINEMENT_SEMANTIC_ID,
                frame_status="failed",
                reason=(
                    f"Refinement iteration {iteration} did not produce a valid "
                    "PnP pose."
                ),
                iterations=tuple(iterations),
                final_estimate=None,
            )

        candidate = attempt.candidate
        iterations.append(
            {
                "iteration": iteration,
                "stage": LEGACY_REFINEMENT_STAGE,
                "status": "success",
                "match_seed": seed,
                "render_source_world_to_camera": current.pnp.get("world_to_camera"),
                "render_path": str(attempt.render_path),
                **candidate.summary_pose_fields(),
                "pose_delta_from_previous": legacy_pose_delta(current.pose, candidate.pose),
                "match": clean_match,
            }
        )
        # d57 single-frame behavior: direct replacement, never gate/rollback.
        current = candidate

    return LegacyRefinementTrajectory(
        semantic_id=LEGACY_REFINEMENT_SEMANTIC_ID,
        frame_status="success",
        reason=None,
        iterations=tuple(iterations),
        final_estimate=current,
    )


def run_ctrnetx_single_frame(
    frame: CtrnetxSingleRuntimeFrame,
    *,
    config: CtrnetxSingleExecutionConfig,
    hooks: CtrnetxSingleHooks,
) -> CtrnetxSingleFrameExecution:
    """Run exact independent-frame R0--R3 legacy transitions for one frame.

    The only state update is the d57 unconditional transition implemented
    above: an R1--R3 PnP success replaces the parent, and a PnP failure fails
    the frame.  In particular, this call deliberately cannot retain an older
    pose or invoke a closed-loop batch solver.
    """
    frame.validate()
    config.validate()
    hooks.validate()
    stage0_seed = config.seed_for(frame, 0)
    stage0_request = CtrnetxSingleStage0Request(
        frame=frame,
        match_seed=stage0_seed,
        views=config.stage0_views,
        match_batch_size=config.stage0_match_batch_size,
        mask_input=config.mask_input,
        mask_prompt=config.mask_prompt,
        visual_geom_group=config.visual_geom_group,
    )
    stage0_request.validate()
    try:
        stage0 = _bound_stage0_artifact(hooks.run_stage0(stage0_request), stage0_request)
    except Exception as error:
        execution = CtrnetxSingleFrameExecution(
            semantic_id=CTRNETX_SINGLE_EXECUTION_SEMANTIC_ID,
            frame=frame,
            stage0=None,
            trajectory=None,
            stage0_match_seed=stage0_seed,
            outer_error_type=type(error).__name__,
        )
        execution.validate()
        return execution
    executed_refinement_match_seeds: list[int] = []

    def refinement_attempt(
        parent: LegacyEstimate,
        iteration: int,
        match_seed: int | None,
    ) -> LegacyRefinementAttempt:
        if match_seed is None:  # Defensive: frozen config always supplies 90.
            raise AssertionError("CTRNet-X single refinement match seed unexpectedly absent")
        request = CtrnetxSingleRefinementRequest(
            frame=frame,
            stage0=stage0,
            parent=parent,
            iteration=iteration,
            match_seed=match_seed,
            visual_geom_group=config.visual_geom_group,
        )
        request.validate()
        # Keep a compact boundary audit even when historical per-iteration
        # records omit the seed on a failed PnP result.
        executed_refinement_match_seeds.append(match_seed)
        attempt = hooks.refine_one(request)
        if not isinstance(attempt, LegacyRefinementAttempt):
            raise TypeError("refine_one hook must return LegacyRefinementAttempt")
        return attempt

    try:
        trajectory = _run_ctrnetx_single_legacy_trajectory(
            stage0.legacy_stage1,
            frame_index=frame.frame_index,
            refinement_iterations=config.refinement_iterations,
            match_seed=config.match_seed,
            refinement_attempt=refinement_attempt,
        )
        outer_error_type = None
    except Exception as error:
        # Historical d57 catches an unexpected runtime failure at the outer
        # per-frame boundary.  Do not manufacture an inline ``error`` round or
        # mark later R rounds as historically skipped.
        trajectory = None
        outer_error_type = type(error).__name__
    execution = CtrnetxSingleFrameExecution(
        semantic_id=CTRNETX_SINGLE_EXECUTION_SEMANTIC_ID,
        frame=frame,
        stage0=stage0,
        trajectory=trajectory,
        stage0_match_seed=stage0_seed,
        executed_refinement_match_seeds=tuple(executed_refinement_match_seeds),
        outer_error_type=outer_error_type,
    )
    execution.validate()
    return execution


def run_ctrnetx_single_frames(
    frames: Sequence[CtrnetxSingleRuntimeFrame],
    *,
    config: CtrnetxSingleExecutionConfig,
    hooks: CtrnetxSingleHooks,
) -> tuple[CtrnetxSingleFrameExecution, ...]:
    """Run an explicit (possibly smoke-test) ordered frame selection."""
    if not frames:
        raise ValueError("CTRNet-X single frame selection must not be empty")
    return tuple(
        run_ctrnetx_single_frame(frame, config=config, hooks=hooks) for frame in frames
    )


def run_ctrnetx_single_formal_scope(
    manifest: CtrnetxSingleRuntimeManifest,
    *,
    config: CtrnetxSingleExecutionConfig,
    hooks: CtrnetxSingleHooks,
    splits: Sequence[str] | None = None,
) -> tuple[CtrnetxSingleFrameExecution, ...]:
    """Run one or more complete archived splits from a full 17,306-frame manifest.

    Splits are independent in the historical single-frame protocol.  Selecting
    one is therefore permitted only after the caller has supplied and validated
    the complete public manifest; no implicit cross-split aggregation occurs.
    """
    manifest.validate_full_scope()
    selected = manifest.select(splits=splits)
    return run_ctrnetx_single_frames(selected, config=config, hooks=hooks)


__all__ = [
    "CTRNETX_SINGLE_EXECUTION_SEMANTIC_ID",
    "CTRNETX_SINGLE_RUNTIME_MANIFEST_PROTOCOL",
    "CtrnetxSingleExecutionConfig",
    "CtrnetxSingleFrameExecution",
    "CtrnetxSingleHooks",
    "CtrnetxSingleInputMaskSource",
    "CtrnetxSingleRefinementRequest",
    "CtrnetxSingleRuntimeFrame",
    "CtrnetxSingleRuntimeManifest",
    "CtrnetxSingleStage0Artifact",
    "CtrnetxSingleStage0Request",
    "SingleFrameRefiner",
    "Stage0Executor",
    "configure_ctrnetx_single_match_rng",
    "run_ctrnetx_single_formal_scope",
    "run_ctrnetx_single_frame",
    "run_ctrnetx_single_frames",
]
