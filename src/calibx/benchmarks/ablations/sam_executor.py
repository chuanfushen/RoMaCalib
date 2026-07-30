"""Executable boundary for the paired Table 4 SAM foreground ablation.

The recovered CF d7 implementation has two deliberately different matching
stages.  This module makes that distinction executable without pretending that
the public repository knows the private Panda/DROID data layout:

* Stage 0 renders six orbit candidates and sends all six pairs through one
  ``match_batch_size=6`` call.
* R1--R3 each render the *current parent pose* once and call the RoMa matcher
  with that one pair (``match_batch_size=1``).  Gate-v1, not a generic
  best-score policy, decides whether the candidate replaces its parent.

Dataset loading, SAM invocation, MuJoCo state restoration, RoMaV2 matching,
and metric calculation are supplied by typed runtime hooks.  Public manifests
therefore contain only opaque relative frame IDs; no server paths, raw frames,
checkpoints, or historical table values are encoded here.  Heavy libraries are
not imported by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Generic, Literal, Protocol, TypeVar

import numpy as np

from ..gate_trajectory import CandidateAttempt, GateTrajectory, run_gate_v1_trajectory
from ..gate_v1 import (
    DETERMINISTIC_SEED,
    GATE_ID,
    RefinementEvidence,
    is_proper_se3,
    pose_sha256,
)
from .sam import SamAblationCondition, SamAblationPair


T = TypeVar("T")
InputMaskStatus = Literal["disabled", "detected", "fallback_unmasked"]
Stage0Status = Literal["success", "unavailable"]
IterationStatus = Literal["success", "accepted", "rejected", "unavailable", "filled"]

SAM_TABLE4_EXECUTION_SEMANTIC_ID = "sam-foreground-ablation-gate-v1-v1"


def _is_public_relative_identifier(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        bool(value)
        and not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and not windows.root
        and ".." not in posix.parts
        and ".." not in windows.parts
    )


def _validate_artifact_id(value: str | None, *, label: str) -> None:
    if value is not None and not _is_public_relative_identifier(value):
        raise ValueError(f"{label} must be a portable relative identifier")


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def configure_sam_table4_determinism(
    seed: int = DETERMINISTIC_SEED,
) -> dict[str, object]:
    """Apply the frozen d7 gate-v1 RNG settings at real-execution time.

    The import is intentionally local: plans, manifest validation, test fakes,
    and ``--help`` stay independent of Torch/OpenCV.  A concrete backend may
    call this once before its first Stage-0 match, or use the identical seed
    carried by every typed request if its runtime owns global initialization.
    """

    if seed != DETERMINISTIC_SEED:
        raise ValueError(
            "SAM Table 4 uses the frozen gate-v1 determinism seed "
            f"{DETERMINISTIC_SEED}"
        )
    from ..gate_v1 import configure_determinism

    return configure_determinism(seed)


@dataclass(frozen=True)
class SamPose:
    """One finite proper OpenCV ``T_camera<-robot_base`` estimate.

    The optional artifact ID is an audit handle only.  It is intentionally
    relative so writing a record never discloses a machine-local output root.
    """

    world_to_camera: np.ndarray
    artifact_id: str | None = None

    def validate(self) -> None:
        transform = np.asarray(self.world_to_camera, dtype=np.float64)
        proper, reasons = is_proper_se3(transform)
        if not proper:
            raise ValueError(
                f"SAM Table 4 pose is not a proper finite SE(3): {reasons}"
            )
        _validate_artifact_id(self.artifact_id, label="pose artifact_id")

    def fingerprint(self) -> str:
        self.validate()
        return pose_sha256(np.asarray(self.world_to_camera, dtype=np.float64))


@dataclass(frozen=True)
class SamPreparedFrame:
    """Dataset-adapter state prepared once for a single opaque frame ID.

    ``runtime`` is deliberately opaque and never included in an audit record.
    It can hold local image arrays, model/data handles, camera intrinsics, or
    adapter-specific metadata without leaking them into a public manifest.
    """

    frame_id: str
    robot_radius_m: float
    input_mask_status: InputMaskStatus
    runtime: object = field(default=None, repr=False, compare=False)
    input_artifact_id: str | None = None

    def validate(self, *, mask_input: bool) -> None:
        if not _is_public_relative_identifier(self.frame_id):
            raise ValueError("prepared frame_id must be a public relative identifier")
        if not np.isfinite(self.robot_radius_m) or self.robot_radius_m <= 1e-6:
            raise ValueError("prepared robot_radius_m must be finite and positive")
        if mask_input:
            if self.input_mask_status not in {"detected", "fallback_unmasked"}:
                raise ValueError(
                    "a SAM-enabled condition must record detected or "
                    "fallback_unmasked input-mask status"
                )
        elif self.input_mask_status != "disabled":
            raise ValueError(
                "a no-SAM condition must record input_mask_status=disabled"
            )
        _validate_artifact_id(self.input_artifact_id, label="input_artifact_id")


@dataclass(frozen=True)
class SamFramePreparationRequest:
    """One preparation request passed to a local dataset adapter."""

    condition: SamAblationCondition
    frame_id: str

    def validate(self) -> None:
        self.condition.validate()
        if self.frame_id not in self.condition.manifest.frame_ids:
            raise ValueError("preparation frame_id is outside the condition manifest")


@dataclass(frozen=True)
class SamStage0Request:
    """The frozen six-view orbital Stage-0 request."""

    condition: SamAblationCondition
    prepared: SamPreparedFrame
    orbit_render_count: int = 6
    match_batch_size: int = 6
    matcher_pair_count: int = 6
    matcher_call_count: int = 1
    determinism_seed: int = DETERMINISTIC_SEED
    iteration: int = 0

    def validate(self) -> None:
        self.condition.validate()
        self.prepared.validate(mask_input=self.condition.mask_input)
        if self.prepared.frame_id not in self.condition.manifest.frame_ids:
            raise ValueError("Stage 0 frame is outside the condition manifest")
        if self.iteration != 0:
            raise ValueError("SAM Table 4 Stage 0 must be iteration 0")
        if self.determinism_seed != self.condition.determinism_seed:
            raise ValueError("Stage 0 must receive the condition determinism seed")
        if (
            self.orbit_render_count != 6
            or self.matcher_pair_count != 6
            or self.match_batch_size != 6
            or self.matcher_call_count != 1
        ):
            raise ValueError(
                "SAM Table 4 Stage 0 requires six orbit renders in one "
                "six-pair matcher call (match_batch_size=6)"
            )


@dataclass(frozen=True)
class SamStage0Artifact:
    """The selected individual-frame PnP result from six Stage-0 views."""

    frame_id: str
    pose: SamPose | None
    failure_reason: str | None = None
    orbit_render_count: int = 6
    matcher_pair_count: int = 6
    match_batch_size: int = 6
    matcher_call_count: int = 1
    determinism_seed: int = DETERMINISTIC_SEED
    artifact_id: str | None = None

    @property
    def status(self) -> Stage0Status:
        return "success" if self.pose is not None else "unavailable"

    def validate(self, request: SamStage0Request) -> None:
        request.validate()
        if self.frame_id != request.prepared.frame_id:
            raise ValueError("Stage 0 artifact frame_id does not match its request")
        if (
            self.orbit_render_count != request.orbit_render_count
            or self.matcher_pair_count != request.matcher_pair_count
            or self.match_batch_size != request.match_batch_size
            or self.matcher_call_count != request.matcher_call_count
            or self.determinism_seed != request.determinism_seed
        ):
            raise ValueError(
                "Stage 0 artifact changed the frozen six-view matcher contract"
            )
        if self.pose is None:
            if not self.failure_reason:
                raise ValueError("an unavailable Stage 0 result needs a failure_reason")
        else:
            self.pose.validate()
            if self.failure_reason is not None:
                raise ValueError(
                    "a successful Stage 0 pose cannot have a failure_reason"
                )
        _validate_artifact_id(self.artifact_id, label="stage0 artifact_id")


@dataclass(frozen=True)
class SamRefinementRequest:
    """One d7 pose-aligned R1--R3 request from the selected parent pose."""

    condition: SamAblationCondition
    prepared: SamPreparedFrame
    parent_pose: SamPose
    iteration: int
    match_render_count: int = 1
    matcher_pair_count: int = 1
    match_batch_size: int = 1
    matcher_call_count: int = 1
    determinism_seed: int = DETERMINISTIC_SEED
    candidate_policy: str = GATE_ID

    def validate(self) -> None:
        self.condition.validate()
        self.prepared.validate(mask_input=self.condition.mask_input)
        self.parent_pose.validate()
        if self.prepared.frame_id not in self.condition.manifest.frame_ids:
            raise ValueError("refinement frame is outside the condition manifest")
        if not 1 <= self.iteration <= 3:
            raise ValueError("SAM Table 4 only executes R1 through R3")
        if (
            self.match_render_count != 1
            or self.matcher_pair_count != 1
            or self.match_batch_size != 1
            or self.matcher_call_count != 1
        ):
            raise ValueError(
                "each SAM Table 4 refinement uses one pose-aligned matching "
                "render and one matcher pair (match_batch_size=1)"
            )
        if self.candidate_policy != GATE_ID:
            raise ValueError("SAM Table 4 refinement must use gate-v1 candidate policy")
        if self.determinism_seed != self.condition.determinism_seed:
            raise ValueError("refinement must receive the condition determinism seed")


@dataclass(frozen=True)
class SamRefinementArtifact:
    """One R1--R3 matching/PnP candidate before gate-v1 selection.

    ``match_render_count`` counts only the render sent to the matcher.  A
    backend may separately render a candidate mask for gate evidence; that
    diagnostic render must not be relabelled as another match pair.
    """

    frame_id: str
    iteration: int
    candidate_pose: SamPose | None
    evidence: RefinementEvidence | None = None
    failure_reason: str | None = None
    match_render_count: int = 1
    matcher_pair_count: int = 1
    match_batch_size: int = 1
    matcher_call_count: int = 1
    determinism_seed: int = DETERMINISTIC_SEED
    match_artifact_id: str | None = None
    candidate_mask_artifact_id: str | None = None

    def validate(self, request: SamRefinementRequest) -> None:
        request.validate()
        if (
            self.frame_id != request.prepared.frame_id
            or self.iteration != request.iteration
        ):
            raise ValueError("refinement artifact does not match its request")
        if (
            self.match_render_count != request.match_render_count
            or self.matcher_pair_count != request.matcher_pair_count
            or self.match_batch_size != request.match_batch_size
            or self.matcher_call_count != request.matcher_call_count
            or self.determinism_seed != request.determinism_seed
        ):
            raise ValueError(
                "refinement artifact changed the one-pair matcher contract"
            )
        if self.candidate_pose is None:
            if self.evidence is not None:
                raise ValueError(
                    "an unavailable refinement candidate cannot have gate evidence"
                )
            if not self.failure_reason:
                raise ValueError(
                    "an unavailable refinement candidate needs a failure_reason"
                )
        else:
            self.candidate_pose.validate()
            if self.evidence is None:
                raise ValueError("a refinement candidate requires gate-v1 evidence")
            if not self.evidence.pnp_success:
                raise ValueError(
                    "a refinement candidate requires a successful PnP evidence record"
                )
            if self.evidence.accepted != (
                self.evidence.V and self.evidence.O and self.evidence.L
            ):
                raise ValueError(
                    "gate-v1 evidence accepted must equal its V/O/L conjunction"
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "a refinement candidate cannot also have a failure_reason"
                )
        _validate_artifact_id(self.match_artifact_id, label="match_artifact_id")
        _validate_artifact_id(
            self.candidate_mask_artifact_id,
            label="candidate_mask_artifact_id",
        )

    def candidate_attempt(self) -> CandidateAttempt[SamPose]:
        if self.candidate_pose is None:
            return CandidateAttempt(
                candidate=None,
                evidence=None,
                failure_reason=self.failure_reason,
            )
        return CandidateAttempt(candidate=self.candidate_pose, evidence=self.evidence)


@dataclass(frozen=True)
class SamIterationRecord:
    """Portable control/evidence record for T0 or one requested refinement."""

    iteration: int
    stage: Literal["stage1", "pose_aligned_refine"]
    status: IterationStatus
    attempted: bool
    accepted: bool
    reason: str | None
    selected_pose: SamPose | None
    candidate_pose: SamPose | None = None
    evidence: RefinementEvidence | None = None
    refinement: SamRefinementArtifact | None = None

    def validate(self) -> None:
        if not 0 <= self.iteration <= 3:
            raise ValueError("SAM Table 4 records only T0 through R3")
        if self.iteration == 0:
            if self.stage != "stage1" or self.refinement is not None:
                raise ValueError("iteration 0 must be a Stage-1 record")
            if self.status not in {"success", "unavailable"}:
                raise ValueError("Stage 1 status must be success or unavailable")
        elif self.stage != "pose_aligned_refine":
            raise ValueError("R1--R3 must be pose_aligned_refine records")
        if self.selected_pose is not None:
            self.selected_pose.validate()
        if self.candidate_pose is not None:
            self.candidate_pose.validate()
        if self.candidate_pose is None and self.evidence is not None:
            raise ValueError("a missing candidate cannot contain gate evidence")
        if self.refinement is not None:
            if self.refinement.iteration != self.iteration:
                raise ValueError("refinement record iteration does not match artifact")
            if self.refinement.candidate_pose is not self.candidate_pose:
                raise ValueError(
                    "iteration candidate pose must preserve artifact identity"
                )
            if self.refinement.evidence is not self.evidence:
                raise ValueError("iteration evidence must preserve artifact identity")

    def to_audit_record(self) -> dict[str, object]:
        self.validate()
        record: dict[str, object] = {
            "iteration": self.iteration,
            "stage": self.stage,
            "status": self.status,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "reason": self.reason,
            "selected_pose_sha256": (
                None if self.selected_pose is None else self.selected_pose.fingerprint()
            ),
            "candidate_pose_sha256": (
                (
                    None
                    if self.candidate_pose is None
                    else self.candidate_pose.fingerprint()
                )
            ),
            "evidence": None if self.evidence is None else self.evidence.to_record(),
        }
        if self.refinement is not None:
            record["matcher"] = {
                "match_render_count": self.refinement.match_render_count,
                "matcher_pair_count": self.refinement.matcher_pair_count,
                "match_batch_size": self.refinement.match_batch_size,
                "matcher_call_count": self.refinement.matcher_call_count,
                "determinism_seed": self.refinement.determinism_seed,
                "match_artifact_id": self.refinement.match_artifact_id,
                "candidate_mask_artifact_id": (
                    self.refinement.candidate_mask_artifact_id
                ),
            }
        return record


@dataclass(frozen=True)
class SamEvaluationRequest:
    """One selected pose handed to a dataset-specific metric hook."""

    condition: SamAblationCondition
    prepared: SamPreparedFrame
    iteration: SamIterationRecord

    def validate(self) -> None:
        self.condition.validate()
        self.prepared.validate(mask_input=self.condition.mask_input)
        self.iteration.validate()
        if self.prepared.frame_id not in self.condition.manifest.frame_ids:
            raise ValueError("evaluation frame is outside the condition manifest")
        if self.iteration.selected_pose is None:
            raise ValueError("a metric hook cannot evaluate an unavailable pose")


@dataclass(frozen=True)
class SamEvaluationRecord(Generic[T]):
    """An evaluation value, or an explicit unavailable placeholder."""

    iteration: int
    evaluated: bool
    value: T | None

    def validate(self) -> None:
        if not 0 <= self.iteration <= 3:
            raise ValueError("evaluation iteration must be in [0, 3]")
        if not self.evaluated and self.value is not None:
            raise ValueError("an unavailable evaluation must not carry a value")


@dataclass(frozen=True)
class SamFrameExecution(Generic[T]):
    """All four requested Table-4 iterations for one frame."""

    condition: SamAblationCondition
    prepared: SamPreparedFrame
    stage0: SamStage0Artifact
    iterations: tuple[SamIterationRecord, ...]
    evaluations: tuple[SamEvaluationRecord[T], ...]

    def validate(self) -> None:
        self.condition.validate()
        self.prepared.validate(mask_input=self.condition.mask_input)
        if self.prepared.frame_id not in self.condition.manifest.frame_ids:
            raise ValueError("frame execution is outside the condition manifest")
        request = SamStage0Request(condition=self.condition, prepared=self.prepared)
        self.stage0.validate(request)
        if tuple(record.iteration for record in self.iterations) != (0, 1, 2, 3):
            raise ValueError("every Table 4 frame must record T0 through R3")
        if tuple(record.iteration for record in self.evaluations) != (0, 1, 2, 3):
            raise ValueError("every Table 4 frame must have four evaluation entries")
        for record, evaluation in zip(self.iterations, self.evaluations, strict=True):
            record.validate()
            evaluation.validate()
            if record.selected_pose is None and evaluation.evaluated:
                raise ValueError("an unavailable pose cannot be evaluated")
            if record.selected_pose is not None and not evaluation.evaluated:
                raise ValueError("an available selected pose requires evaluation")
        if self.stage0.pose is None:
            if any(record.status != "unavailable" for record in self.iterations):
                raise ValueError(
                    "a missing Stage-1 pose makes T0 through R3 unavailable"
                )
            if any(record.attempted for record in self.iterations[1:]):
                raise ValueError("R1--R3 cannot be attempted without Stage 0")
        elif self.iterations[0].selected_pose is not self.stage0.pose:
            raise ValueError("T0 selected pose must be the successful Stage-1 pose")

    def to_audit_record(self) -> dict[str, object]:
        self.validate()
        return {
            "frame_id": self.prepared.frame_id,
            "input_mask": {
                "enabled": self.condition.mask_input,
                "prompt": self.condition.mask_prompt,
                "status": self.prepared.input_mask_status,
                "input_artifact_id": self.prepared.input_artifact_id,
            },
            "stage0": {
                "status": self.stage0.status,
                "failure_reason": self.stage0.failure_reason,
                "orbit_render_count": self.stage0.orbit_render_count,
                "matcher_pair_count": self.stage0.matcher_pair_count,
                "match_batch_size": self.stage0.match_batch_size,
                "matcher_call_count": self.stage0.matcher_call_count,
                "determinism_seed": self.stage0.determinism_seed,
                "artifact_id": self.stage0.artifact_id,
            },
            "gate_id": GATE_ID,
            "iterations": [record.to_audit_record() for record in self.iterations],
            "evaluated_iterations": [
                {"iteration": record.iteration, "evaluated": record.evaluated}
                for record in self.evaluations
            ],
        }


class SamTable4Backend(Protocol[T]):
    """Concrete data/runtime operations required by one Table-4 condition.

    The adapter applies the fixed ``request.determinism_seed`` before real
    matching (or calls :func:`configure_sam_table4_determinism` once before
    its first Stage-0 request).  Every expected runtime failure is returned as
    an unavailable artifact, never as a silently changed batch size or policy.
    """

    def prepare_frame(self, request: SamFramePreparationRequest) -> SamPreparedFrame:
        """Resolve one opaque public ID into local data and runtime state."""

    def run_stage0(self, request: SamStage0Request) -> SamStage0Artifact:
        """Run six orbit views in one six-pair match call and select a T0 pose."""

    def run_refinement(self, request: SamRefinementRequest) -> SamRefinementArtifact:
        """Run exactly one parent-pose render and one-pair matcher call."""

    def evaluate(self, request: SamEvaluationRequest) -> T:
        """Evaluate the selected pose on the dataset's documented metric."""


PrepareFrameHook = Callable[[SamFramePreparationRequest], SamPreparedFrame]
Stage0Hook = Callable[[SamStage0Request], SamStage0Artifact]
RefinementHook = Callable[[SamRefinementRequest], SamRefinementArtifact]
EvaluationHook = Callable[[SamEvaluationRequest], T]


@dataclass(frozen=True)
class SamTable4Hooks(Generic[T]):
    """Small adapter for callers that have functions instead of a backend class."""

    prepare_frame_hook: PrepareFrameHook
    stage0_hook: Stage0Hook
    refinement_hook: RefinementHook
    evaluation_hook: EvaluationHook[T]

    def prepare_frame(self, request: SamFramePreparationRequest) -> SamPreparedFrame:
        return self.prepare_frame_hook(request)

    def run_stage0(self, request: SamStage0Request) -> SamStage0Artifact:
        return self.stage0_hook(request)

    def run_refinement(self, request: SamRefinementRequest) -> SamRefinementArtifact:
        return self.refinement_hook(request)

    def evaluate(self, request: SamEvaluationRequest) -> T:
        return self.evaluation_hook(request)


@dataclass(frozen=True)
class SamTable4Execution(Generic[T]):
    """One complete 900-frame condition execution and its portable audit data."""

    condition: SamAblationCondition
    frames: tuple[SamFrameExecution[T], ...]

    def validate(self) -> None:
        self.condition.validate()
        expected_ids = self.condition.manifest.frame_ids
        observed_ids = tuple(frame.prepared.frame_id for frame in self.frames)
        if observed_ids != expected_ids:
            raise ValueError("SAM Table 4 execution must preserve manifest frame order")
        for frame in self.frames:
            frame.validate()

    def summary(self) -> dict[str, object]:
        self.validate()
        stage0_available = sum(frame.stage0.pose is not None for frame in self.frames)
        iteration_statuses: dict[str, dict[str, int]] = {}
        for iteration in range(4):
            counts: dict[str, int] = {}
            for frame in self.frames:
                status = frame.iterations[iteration].status
                counts[status] = counts.get(status, 0) + 1
            iteration_statuses[f"iteration_{iteration:02d}"] = counts
        return {
            "requested_frames": len(self.frames),
            "stage0_available_frames": stage0_available,
            "stage0_unavailable_frames": len(self.frames) - stage0_available,
            "iteration_statuses": iteration_statuses,
        }

    def audit_record(self) -> dict[str, object]:
        self.validate()
        return {
            "execution_semantic_id": SAM_TABLE4_EXECUTION_SEMANTIC_ID,
            "split": self.condition.split,
            "manifest_sha256": self.condition.manifest.sha256(),
            "requested_frames": len(self.frames),
            "mask_input": self.condition.mask_input,
            "protocol": {
                "stage0": {
                    "orbit_render_count": 6,
                    "matcher_pair_count": 6,
                    "match_batch_size": 6,
                    "matcher_call_count": 1,
                    "determinism_seed": self.condition.determinism_seed,
                },
                "refinement": {
                    "requested_iterations": 3,
                    "match_render_count": 1,
                    "matcher_pair_count": 1,
                    "match_batch_size": 1,
                    "matcher_call_count": 1,
                    "determinism_seed": self.condition.determinism_seed,
                    "candidate_policy": GATE_ID,
                },
            },
            "summary": self.summary(),
            "frames": [frame.to_audit_record() for frame in self.frames],
        }

    def write_audit(self, path: Path) -> None:
        """Write only portable control/evidence records, never local runtime state."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.audit_record(),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )


BackendFactory = Callable[[SamAblationCondition], SamTable4Backend[T]]


def condition_plan(condition: SamAblationCondition) -> dict[str, object]:
    """Return the inspectable frozen plan without probing local assets."""

    condition.validate()
    return {
        "execution_semantic_id": SAM_TABLE4_EXECUTION_SEMANTIC_ID,
        "split": condition.split,
        "mask_input": condition.mask_input,
        "manifest_sha256": condition.manifest.sha256(),
        "requested_frames": len(condition.manifest.frame_ids),
        "stage0": {
            "orbit_render_count": 6,
            "matcher_pair_count": 6,
            "match_batch_size": 6,
            "matcher_call_count": 1,
            "determinism_seed": condition.determinism_seed,
        },
        "refinement": {
            "requested_iterations": 3,
            "match_render_count": 1,
            "matcher_pair_count": 1,
            "match_batch_size": 1,
            "matcher_call_count": 1,
            "determinism_seed": condition.determinism_seed,
            "candidate_policy": GATE_ID,
        },
        "input_mask": {
            "enabled": condition.mask_input,
            "prompt": condition.mask_prompt,
            "sam_no_detection_policy": "fallback_unmasked_and_recorded",
        },
    }


def _unavailable_iterations() -> tuple[SamIterationRecord, ...]:
    """Mirror d7's explicit T0--R3 records after Stage-1 PnP is unavailable."""

    records: list[SamIterationRecord] = []
    for iteration in range(4):
        records.append(
            SamIterationRecord(
                iteration=iteration,
                stage="stage1" if iteration == 0 else "pose_aligned_refine",
                status="unavailable",
                attempted=iteration == 0,
                accepted=False,
                reason="stage1_missing",
                selected_pose=None,
            )
        )
    return tuple(records)


def _trajectory_iteration_records(
    stage0: SamStage0Artifact,
    trajectory: GateTrajectory[SamPose],
    artifacts: Mapping[int, SamRefinementArtifact],
) -> tuple[SamIterationRecord, ...]:
    """Materialize the generic gate trajectory as the d7 T0/R1/R2/R3 schema."""

    if stage0.pose is None:
        raise ValueError("a gate trajectory requires a successful Stage-1 pose")
    records: list[SamIterationRecord] = [
        SamIterationRecord(
            iteration=0,
            stage="stage1",
            status="success",
            attempted=True,
            accepted=True,
            reason=None,
            selected_pose=stage0.pose,
        )
    ]
    for step in trajectory.steps:
        artifact = artifacts.get(step.iteration)
        records.append(
            SamIterationRecord(
                iteration=step.iteration,
                stage="pose_aligned_refine",
                status=step.status,
                attempted=step.status != "filled",
                accepted=step.accepted,
                reason=step.reason,
                selected_pose=step.selected,
                candidate_pose=step.candidate,
                evidence=step.evidence,
                refinement=artifact,
            )
        )
    if tuple(record.iteration for record in records) != (0, 1, 2, 3):
        raise AssertionError("gate-v1 trajectory must materialize T0 through R3")
    return tuple(records)


def _evaluate_iterations(
    backend: SamTable4Backend[T],
    condition: SamAblationCondition,
    prepared: SamPreparedFrame,
    iterations: tuple[SamIterationRecord, ...],
) -> tuple[SamEvaluationRecord[T], ...]:
    records: list[SamEvaluationRecord[T]] = []
    for iteration in iterations:
        if iteration.selected_pose is None:
            records.append(
                SamEvaluationRecord(
                    iteration=iteration.iteration,
                    evaluated=False,
                    value=None,
                )
            )
            continue
        request = SamEvaluationRequest(
            condition=condition,
            prepared=prepared,
            iteration=iteration,
        )
        request.validate()
        records.append(
            SamEvaluationRecord(
                iteration=iteration.iteration,
                evaluated=True,
                value=backend.evaluate(request),
            )
        )
    return tuple(records)


def execute_sam_table4_condition(
    condition: SamAblationCondition,
    backend: SamTable4Backend[T],
) -> SamTable4Execution[T]:
    """Execute one no-SAM or SAM arm without changing the frozen protocol.

    A backend must report expected PnP/render/match failures as unavailable
    artifacts with a reason.  Contract or programming errors deliberately
    propagate instead of being converted into a hidden fallback.  This mirrors
    d7's separation between an unavailable PnP candidate and a malformed run.
    """

    condition.validate()
    frame_runs: list[SamFrameExecution[T]] = []
    for frame_id in condition.manifest.frame_ids:
        preparation_request = SamFramePreparationRequest(
            condition=condition,
            frame_id=frame_id,
        )
        preparation_request.validate()
        prepared = backend.prepare_frame(preparation_request)
        if not isinstance(prepared, SamPreparedFrame):
            raise TypeError("SAM Table 4 prepare_frame must return SamPreparedFrame")
        prepared.validate(mask_input=condition.mask_input)
        if prepared.frame_id != frame_id:
            raise ValueError("SAM Table 4 prepare_frame returned the wrong frame ID")

        stage0_request = SamStage0Request(condition=condition, prepared=prepared)
        stage0 = backend.run_stage0(stage0_request)
        if not isinstance(stage0, SamStage0Artifact):
            raise TypeError("SAM Table 4 run_stage0 must return SamStage0Artifact")
        stage0.validate(stage0_request)

        if stage0.pose is None:
            iterations = _unavailable_iterations()
        else:
            attempts: dict[int, SamRefinementArtifact] = {}

            def candidate_attempt(
                parent: SamPose,
                iteration: int,
            ) -> CandidateAttempt[SamPose]:
                request = SamRefinementRequest(
                    condition=condition,
                    prepared=prepared,
                    parent_pose=parent,
                    iteration=iteration,
                )
                artifact = backend.run_refinement(request)
                if not isinstance(artifact, SamRefinementArtifact):
                    raise TypeError(
                        "SAM Table 4 run_refinement must return SamRefinementArtifact"
                    )
                artifact.validate(request)
                attempts[iteration] = artifact
                return artifact.candidate_attempt()

            trajectory = run_gate_v1_trajectory(
                stage0.pose,
                iterations=condition.refinement_iterations,
                candidate_attempt=candidate_attempt,
                pose_of=lambda pose: np.asarray(pose.world_to_camera, dtype=np.float64),
                robot_radius_m=prepared.robot_radius_m,
            )
            iterations = _trajectory_iteration_records(stage0, trajectory, attempts)

        evaluations = _evaluate_iterations(backend, condition, prepared, iterations)
        frame_execution = SamFrameExecution(
            condition=condition,
            prepared=prepared,
            stage0=stage0,
            iterations=iterations,
            evaluations=evaluations,
        )
        frame_execution.validate()
        frame_runs.append(frame_execution)
    execution = SamTable4Execution(condition=condition, frames=tuple(frame_runs))
    execution.validate()
    return execution


def execute_sam_table4_pair(
    pair: SamAblationPair,
    backend_factory: BackendFactory[T],
) -> tuple[SamTable4Execution[T], SamTable4Execution[T]]:
    """Execute a validated no-SAM/SAM pair with separate local backends.

    The pair is validated before either arm starts, so the only protocol-level
    difference is ``mask_input``.  A local factory is used because the two
    arms may need independently named output roots while resolving the exact
    same public frame IDs.
    """

    pair.validate()
    without_backend = backend_factory(pair.without_sam)
    with_backend = backend_factory(pair.with_sam)
    return (
        execute_sam_table4_condition(pair.without_sam, without_backend),
        execute_sam_table4_condition(pair.with_sam, with_backend),
    )


__all__ = [
    "BackendFactory",
    "EvaluationHook",
    "InputMaskStatus",
    "PrepareFrameHook",
    "RefinementHook",
    "SAM_TABLE4_EXECUTION_SEMANTIC_ID",
    "SamAblationCondition",
    "SamEvaluationRecord",
    "SamEvaluationRequest",
    "SamFrameExecution",
    "SamFramePreparationRequest",
    "SamPose",
    "SamPreparedFrame",
    "SamRefinementArtifact",
    "SamRefinementRequest",
    "SamStage0Artifact",
    "SamStage0Request",
    "SamTable4Backend",
    "SamTable4Execution",
    "SamTable4Hooks",
    "Stage0Hook",
    "configure_sam_table4_determinism",
    "condition_plan",
    "execute_sam_table4_condition",
    "execute_sam_table4_pair",
]
