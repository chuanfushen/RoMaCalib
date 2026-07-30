"""Pure closed-loop CTRNet-X batch replay state machine.

This module is deliberately an execution *control layer*, not a copy of an
archived CF launcher and not an alias for the legacy single-frame evaluator.
It makes the audited batch invariants explicit:

* iteration 0 aggregates only stage-1 frames whose individual PnP succeeded;
* iterations 1--3 render each requested frame once from the parent shared
  pose, and aggregate raw post-geometry correspondences into one episode PnP;
* every requested frame enters evaluation at every requested iteration; and
* a failed shared PnP is recorded and, when possible, retains the parent pose
  for evaluation rather than pretending an update succeeded.

Concrete render/match/metric code is supplied by hooks.  Keeping those hooks
outside this module prevents the public replay semantics from mutating the
upstream ``romav2`` pipeline while still making coverage and recovery auditable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
import re
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from ..common import is_public_relative_identifier
from .episode_pnp import (
    EpisodeFrameCorrespondences,
    EpisodePose,
    assert_complete_frame_coverage,
    solve_episode_pnp,
)


T = TypeVar("T")
ReplayRoundStatus = Literal[
    "updated",
    "retained_parent",
    "solver_failed",
    "skipped_no_parent",
]
ArtifactRecoveryState = Literal[
    "cache_hit",
    "recompute_cached_render",
    "rerender",
    "recovery_failed",
]
_PUBLIC_FAILURE_CODE = re.compile(r"^[a-z][a-z0-9_]*(?::[a-z0-9_]+)*$")


def _is_public_artifact_id(value: str | None) -> bool:
    """Keep optional artifact audit identifiers portable and path-free."""
    if value is None:
        return True
    return is_public_relative_identifier(value)


def is_public_failure_code(value: str | None) -> bool:
    """Return whether a public audit reason is a stable, path-free code.

    Public result records deliberately carry codes rather than exception text:
    backend-local tracebacks and absolute paths stay in the machine-local log.
    """
    return isinstance(value, str) and bool(_PUBLIC_FAILURE_CODE.fullmatch(value))


def _validate_public_failure_code(value: str | None, *, label: str) -> None:
    if not is_public_failure_code(value):
        raise ValueError(f"{label} must be a stable lowercase failure code")


def _validate_pose(pose: EpisodePose) -> None:
    """Reject malformed hook output before it can be reported as an update."""
    transform = np.asarray(pose.world_to_camera, dtype=np.float64)
    camera_matrix = np.asarray(pose.camera_matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("episode PnP hook returned a non-finite 4x4 pose")
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise ValueError("episode PnP hook returned an invalid camera matrix")
    if pose.correspondence_count < 1:
        raise ValueError("episode PnP hook returned a pose without correspondences")


def _recorded_seed(value: object, *, label: str) -> int:
    """Accept integer-like scalar seeds but not lossy strings/floats/bools."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class Stage1FrameResult:
    """Stage-1 state for exactly one requested frame.

    ``correspondences`` may be present even when the individual PnP failed.
    That reflects a real intermediate artifact, but it is intentionally never
    admitted to the iteration-0 shared episode PnP source set.
    """

    frame_id: str
    episode_id: str
    pnp_success: bool
    correspondences: EpisodeFrameCorrespondences | None = None
    failure_reason: str | None = None

    def validate(self) -> None:
        if not self.frame_id or not self.episode_id:
            raise ValueError("stage-1 frame_id and episode_id must be non-empty")
        if self.pnp_success and self.correspondences is None:
            raise ValueError("a successful stage-1 PnP requires correspondences")
        if self.pnp_success and self.failure_reason is not None:
            raise ValueError("a successful stage-1 PnP cannot have a failure reason")
        if self.failure_reason is not None:
            _validate_public_failure_code(
                self.failure_reason,
                label="stage-1 failure_reason",
            )
        if self.correspondences is not None:
            self.correspondences.validate()
            if self.correspondences.frame_id != self.frame_id:
                raise ValueError("stage-1 correspondence frame_id does not match result")
            if self.correspondences.episode_id != self.episode_id:
                raise ValueError("stage-1 correspondence episode_id does not match result")
            if self.pnp_success and len(self.correspondences.scores) == 0:
                raise ValueError("a successful stage-1 PnP needs non-empty correspondences")


@dataclass(frozen=True)
class ReplayFrameAttempt:
    """One parent-pose-aligned render/match/geometry attempt for one frame."""

    frame_id: str
    episode_id: str
    correspondences: EpisodeFrameCorrespondences | None
    failure_reason: str | None = None
    render_count: int = 1
    correspondence_space: str = "raw_post_geometry"
    artifact_id: str | None = None
    replacement_artifact_id: str | None = None
    artifact_recovery_state: ArtifactRecoveryState | None = None
    # The historical matcher RNG is per frame and per replay iteration.  Keep
    # the derived value with the attempt so a public audit never has to infer
    # it from backend-private state.
    match_seed: int | None = None

    def validate(self) -> None:
        if not self.frame_id or not self.episode_id:
            raise ValueError("replay frame_id and episode_id must be non-empty")
        if self.render_count != 1:
            raise ValueError("each active replay frame must contain exactly one render")
        if self.correspondence_space != "raw_post_geometry":
            raise ValueError(
                "CTRNet-X replay must aggregate raw_post_geometry correspondences"
            )
        if not _is_public_artifact_id(self.artifact_id):
            raise ValueError("artifact_id must be a portable relative identifier")
        if not _is_public_artifact_id(self.replacement_artifact_id):
            raise ValueError("replacement_artifact_id must be a portable relative identifier")
        if self.replacement_artifact_id is not None and self.artifact_id is None:
            raise ValueError("replacement_artifact_id requires artifact_id")
        if self.artifact_recovery_state not in {
            None,
            "cache_hit",
            "recompute_cached_render",
            "rerender",
            "recovery_failed",
        }:
            raise ValueError("unsupported artifact_recovery_state")
        if self.artifact_recovery_state is not None and self.artifact_id is None:
            raise ValueError("artifact_recovery_state requires artifact_id")
        if self.match_seed is not None:
            _recorded_seed(self.match_seed, label="match_seed")
        if self.correspondences is None:
            if not self.failure_reason:
                raise ValueError("an unavailable replay correspondence needs a failure reason")
            _validate_public_failure_code(
                self.failure_reason,
                label="replay failure_reason",
            )
            return
        if self.failure_reason is not None:
            raise ValueError("a replay correspondence cannot also have a failure reason")
        self.correspondences.validate()
        if self.correspondences.frame_id != self.frame_id:
            raise ValueError("replay correspondence frame_id does not match attempt")
        if self.correspondences.episode_id != self.episode_id:
            raise ValueError("replay correspondence episode_id does not match attempt")
        if len(self.correspondences.scores) == 0:
            raise ValueError("empty replay correspondences must be recorded as unavailable")


@dataclass(frozen=True)
class RecoveryEvent:
    """An explicit non-silent failure/recovery record for public audit."""

    iteration: int
    scope: Literal["frame", "episode"]
    code: str
    action: str
    reason: str
    frame_id: str | None = None
    artifact_id: str | None = None
    replacement_artifact_id: str | None = None
    artifact_recovery_state: ArtifactRecoveryState | None = None

    def validate(self) -> None:
        if self.iteration < 0:
            raise ValueError("recovery iteration must be non-negative")
        if not self.code or not self.action or not self.reason:
            raise ValueError("recovery events require code, action, and reason")
        _validate_public_failure_code(self.code, label="recovery code")
        _validate_public_failure_code(self.action, label="recovery action")
        _validate_public_failure_code(self.reason, label="recovery reason")
        if self.scope == "frame" and not self.frame_id:
            raise ValueError("frame recovery events require frame_id")
        if self.scope == "episode" and self.frame_id is not None:
            raise ValueError("episode recovery events cannot name one frame")
        if not _is_public_artifact_id(self.artifact_id):
            raise ValueError("artifact_id must be a portable relative identifier")
        if not _is_public_artifact_id(self.replacement_artifact_id):
            raise ValueError("replacement_artifact_id must be a portable relative identifier")
        if self.replacement_artifact_id is not None and self.artifact_id is None:
            raise ValueError("replacement_artifact_id requires artifact_id")
        if self.artifact_recovery_state not in {
            None,
            "cache_hit",
            "recompute_cached_render",
            "rerender",
            "recovery_failed",
        }:
            raise ValueError("unsupported artifact_recovery_state")
        if self.artifact_recovery_state is not None and self.artifact_id is None:
            raise ValueError("artifact_recovery_state requires artifact_id")

    def to_record(self) -> dict[str, object]:
        """Return the stable, serializable recovery-audit payload."""
        self.validate()
        return {
            "iteration": self.iteration,
            "scope": self.scope,
            "frame_id": self.frame_id,
            "code": self.code,
            "action": self.action,
            "reason": self.reason,
            "artifact_id": self.artifact_id,
            "replacement_artifact_id": self.replacement_artifact_id,
            "artifact_recovery_state": self.artifact_recovery_state,
        }


@dataclass(frozen=True)
class FrameEvaluationRequest:
    """The complete per-frame state handed to a metric/evaluation hook."""

    episode_id: str
    frame_id: str
    iteration: int
    status: ReplayRoundStatus
    parent_pose: EpisodePose | None
    selected_pose: EpisodePose | None
    stage1: Stage1FrameResult
    replay_attempt: ReplayFrameAttempt | None
    source_frame_ids: tuple[str, ...]
    solver_failure_reason: str | None


@dataclass(frozen=True)
class FrameEvaluationRecord(Generic[T]):
    """One evaluator return value, retained even when it is ``None``."""

    request: FrameEvaluationRequest
    value: T


@dataclass(frozen=True)
class EpisodeReplayRound(Generic[T]):
    """One shared-PnP iteration and its complete frame-level evaluation set."""

    iteration: int
    parent_pose: EpisodePose | None
    candidate_pose: EpisodePose | None
    selected_pose: EpisodePose | None
    status: ReplayRoundStatus
    source_frame_ids: tuple[str, ...]
    replay_attempts: tuple[ReplayFrameAttempt, ...]
    solver_failure_reason: str | None
    evaluations: tuple[FrameEvaluationRecord[T], ...]
    # ``None`` preserves the legacy generic hook API.  CTRNet-X's public
    # executor supplies the actual round-specific shared-PnP seed explicitly.
    pnp_seed: int | None = None

    def validate(self, requested_frame_ids: tuple[str, ...]) -> None:
        if self.iteration < 0:
            raise ValueError("replay iteration must be non-negative")
        if self.pnp_seed is not None:
            _recorded_seed(self.pnp_seed, label="pnp_seed")
        if self.solver_failure_reason is not None:
            _validate_public_failure_code(
                self.solver_failure_reason,
                label="solver_failure_reason",
            )
        if self.status == "updated":
            if self.candidate_pose is None or self.selected_pose is not self.candidate_pose:
                raise ValueError("an updated replay round must select its candidate pose")
            if self.solver_failure_reason is not None:
                raise ValueError("an updated replay round cannot contain a solver failure")
        elif self.status == "retained_parent":
            if self.parent_pose is None or self.selected_pose is not self.parent_pose:
                raise ValueError("retained_parent must evaluate with the parent pose")
            if self.candidate_pose is not None or not self.solver_failure_reason:
                raise ValueError("retained_parent requires a failed solver candidate")
        elif self.status == "solver_failed":
            if self.candidate_pose is not None or self.selected_pose is not None:
                raise ValueError("solver_failed cannot report an available shared pose")
            if not self.solver_failure_reason:
                raise ValueError("solver_failed requires a failure reason")
        elif self.status == "skipped_no_parent":
            if self.iteration == 0:
                raise ValueError("iteration 0 cannot be skipped for a missing parent")
            if self.parent_pose is not None:
                raise ValueError("skipped_no_parent requires no parent pose")
            if self.candidate_pose is not None or self.selected_pose is not None:
                raise ValueError("skipped_no_parent cannot report an episode pose")
            if self.solver_failure_reason != "no_parent_shared_pose":
                raise ValueError(
                    "skipped_no_parent must record no_parent_shared_pose"
                )
        else:  # pragma: no cover - Literal keeps static callers honest
            raise ValueError(f"unknown replay status: {self.status}")
        assert_complete_frame_coverage(
            requested_frame_ids,
            (record.request.frame_id for record in self.evaluations),
        )
        if self.iteration == 0 and self.replay_attempts:
            raise ValueError("iteration 0 must not contain replay render attempts")
        if self.iteration > 0 and self.parent_pose is not None:
            for attempt in self.replay_attempts:
                attempt.validate()
            assert_complete_frame_coverage(
                requested_frame_ids,
                (attempt.frame_id for attempt in self.replay_attempts),
            )
        if self.iteration > 0 and self.parent_pose is None and self.replay_attempts:
            raise ValueError("a parentless round cannot claim pose-aligned replay renders")


@dataclass(frozen=True)
class BatchReplayResult(Generic[T]):
    """Auditable R3 closed-loop replay result for one CTRNet-X episode."""

    episode_id: str
    requested_frame_ids: tuple[str, ...]
    rounds: tuple[EpisodeReplayRound[T], ...]
    recovery_events: tuple[RecoveryEvent, ...]

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if len(self.requested_frame_ids) == 0:
            raise ValueError("batch replay requires at least one requested frame")
        if len(set(self.requested_frame_ids)) != len(self.requested_frame_ids):
            raise ValueError("requested frame IDs must be unique")
        if tuple(round_.iteration for round_ in self.rounds) != (0, 1, 2, 3):
            raise ValueError("CTRNet-X batch replay must emit iteration 0 through R3")
        for round_ in self.rounds:
            round_.validate(self.requested_frame_ids)
        for event in self.recovery_events:
            event.validate()
            if event.frame_id is not None and event.frame_id not in self.requested_frame_ids:
                raise ValueError("recovery event names a frame outside the requested scope")

    def recovery_audit_records(self) -> tuple[dict[str, object], ...]:
        """Expose every recovery event without collapsing it into a success bit."""
        self.validate()
        return tuple(event.to_record() for event in self.recovery_events)


EpisodePnpSolver = Callable[..., EpisodePose]
ReplayFrameHook = Callable[[EpisodePose, str, int], ReplayFrameAttempt]
FrameEvaluator = Callable[[FrameEvaluationRequest], T]
PnpSeedForIteration = Callable[[int], int]


def _stage1_by_frame(
    episode_id: str,
    requested_frame_ids: tuple[str, ...],
    stage1_results: Iterable[Stage1FrameResult],
) -> dict[str, Stage1FrameResult]:
    materialized = tuple(stage1_results)
    for result in materialized:
        result.validate()
        if result.episode_id != episode_id:
            raise ValueError("stage-1 result episode_id does not match batch episode")
    assert_complete_frame_coverage(
        requested_frame_ids,
        (result.frame_id for result in materialized),
    )
    return {result.frame_id: result for result in materialized}


def _solver_failure_reason(error: Exception) -> str:
    # The error message can contain a private path, checkpoint name, or other
    # machine-local detail.  Public audit records retain only this stable code;
    # the invoking backend remains responsible for its local traceback/log.
    if isinstance(error, ValueError):
        return "episode_pnp_value_error"
    return "episode_pnp_runtime_error"


def _solve_shared_pose(
    solver: EpisodePnpSolver,
    sources: tuple[EpisodeFrameCorrespondences, ...],
    solver_kwargs: Mapping[str, Any],
) -> tuple[EpisodePose | None, str | None]:
    """Run exactly one shared solver call and turn expected PnP failures into audit."""
    if not sources:
        return None, "no_episode_pnp_sources"
    try:
        pose = solver(sources, **dict(solver_kwargs))
    except (RuntimeError, ValueError) as error:
        return None, _solver_failure_reason(error)
    if not isinstance(pose, EpisodePose):
        raise TypeError("episode PnP hook must return EpisodePose")
    _validate_pose(pose)
    return pose, None


def _round_solver_kwargs(
    base_kwargs: Mapping[str, Any],
    pnp_seed: int | None,
) -> dict[str, Any]:
    """Inject an explicit round seed without changing old solver callers."""
    kwargs = dict(base_kwargs)
    if pnp_seed is None:
        return kwargs
    seed_value = _recorded_seed(
        pnp_seed,
        label="pnp_seed_for_iteration result",
    )
    configured_seed = kwargs.get("seed")
    if configured_seed is not None and int(configured_seed) != seed_value:
        raise ValueError(
            "solver_kwargs.seed conflicts with the explicit round-specific PnP seed"
        )
    kwargs["seed"] = seed_value
    return kwargs


def _evaluate_all_frames(
    *,
    episode_id: str,
    requested_frame_ids: tuple[str, ...],
    iteration: int,
    status: ReplayRoundStatus,
    parent_pose: EpisodePose | None,
    selected_pose: EpisodePose | None,
    stage1_by_frame: Mapping[str, Stage1FrameResult],
    replay_by_frame: Mapping[str, ReplayFrameAttempt],
    source_frame_ids: tuple[str, ...],
    solver_failure_reason: str | None,
    evaluate_frame: FrameEvaluator[T],
) -> tuple[FrameEvaluationRecord[T], ...]:
    records: list[FrameEvaluationRecord[T]] = []
    for frame_id in requested_frame_ids:
        request = FrameEvaluationRequest(
            episode_id=episode_id,
            frame_id=frame_id,
            iteration=iteration,
            status=status,
            parent_pose=parent_pose,
            selected_pose=selected_pose,
            stage1=stage1_by_frame[frame_id],
            replay_attempt=replay_by_frame.get(frame_id),
            source_frame_ids=source_frame_ids,
            solver_failure_reason=solver_failure_reason,
        )
        records.append(FrameEvaluationRecord(request=request, value=evaluate_frame(request)))
    assert_complete_frame_coverage(
        requested_frame_ids,
        (record.request.frame_id for record in records),
    )
    return tuple(records)


def _failed_round_status(parent_pose: EpisodePose | None) -> ReplayRoundStatus:
    return "solver_failed" if parent_pose is None else "retained_parent"


def run_ctrnetx_closed_loop_batch_replay(
    episode_id: str,
    requested_frame_ids: Iterable[str],
    stage1_results: Iterable[Stage1FrameResult],
    *,
    replay_frame: ReplayFrameHook,
    evaluate_frame: FrameEvaluator[T],
    episode_pnp_solver: EpisodePnpSolver = solve_episode_pnp,
    solver_kwargs: Mapping[str, Any] | None = None,
    pnp_seed_for_iteration: PnpSeedForIteration | None = None,
    refinement_iterations: int = 3,
) -> BatchReplayResult[T]:
    """Run the frozen R3 shared-pose replay contract for one episode.

    ``replay_frame`` is called exactly once for every requested frame in every
    active iteration (1--3) that has a parent shared pose.  Its return must
    carry raw post-geometry correspondences or an explicit failure reason.
    ``evaluate_frame`` is called for *all* requested frames at i0--i3,
    including frames that failed stage 1, rendering, matching, or shared PnP.

    A solver ``RuntimeError`` or ``ValueError`` is an audited failed attempt.
    At i1--i3 the parent shared pose is retained for all frame evaluations;
    this is labelled ``retained_parent`` rather than ``updated``.  If i0 did
    not yield a shared pose, later rounds are explicitly
    ``skipped_no_parent``--they neither invent a render nor run an empty PnP.
    Other hook exceptions deliberately propagate so programming errors cannot
    become a silent fallback.

    ``pnp_seed_for_iteration`` is optional for backward compatibility with
    generic replay consumers.  When supplied, it is called for each recorded
    round (T0 through R3), its result is stored in ``EpisodeReplayRound``, and
    the same value is passed to the episode-PnP solver as ``seed``.  A static
    conflicting ``solver_kwargs['seed']`` is rejected rather than silently
    overriding a paper-reproduction seed schedule.
    """
    if not episode_id:
        raise ValueError("episode_id must be non-empty")
    if refinement_iterations != 3:
        raise ValueError("CTRNet-X closed-loop batch is frozen at R3")
    requested = tuple(requested_frame_ids)
    if len(requested) == 0:
        raise ValueError("batch replay requires at least one requested frame")
    if any(not frame_id for frame_id in requested):
        raise ValueError("requested frame IDs must be non-empty")
    if len(set(requested)) != len(requested):
        raise ValueError("requested frame IDs must be unique")
    stage1_by_frame = _stage1_by_frame(episode_id, requested, stage1_results)
    kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    def pnp_seed_for_round(iteration: int) -> int | None:
        if pnp_seed_for_iteration is None:
            return None
        return _recorded_seed(
            pnp_seed_for_iteration(iteration),
            label="pnp_seed_for_iteration result",
        )

    recovery_events: list[RecoveryEvent] = []
    stage1_sources: list[EpisodeFrameCorrespondences] = []
    for frame_id in requested:
        result = stage1_by_frame[frame_id]
        if result.pnp_success:
            # validate() ensures this is present and non-empty.
            assert result.correspondences is not None
            stage1_sources.append(result.correspondences)
        else:
            recovery_events.append(
                RecoveryEvent(
                    iteration=0,
                    scope="frame",
                    frame_id=frame_id,
                    code="stage1_pnp_unsuccessful_excluded",
                    action="exclude_from_i0_episode_pnp_but_evaluate",
                    reason=result.failure_reason or "stage1_pnp_unsuccessful",
                )
            )

    initial_pnp_seed = pnp_seed_for_round(0)
    initial_candidate, initial_failure = _solve_shared_pose(
        episode_pnp_solver,
        tuple(stage1_sources),
        _round_solver_kwargs(kwargs, initial_pnp_seed),
    )
    if initial_candidate is None:
        initial_status: ReplayRoundStatus = "solver_failed"
        recovery_events.append(
            RecoveryEvent(
                iteration=0,
                scope="episode",
                code="episode_pnp_failed",
                action="no_shared_pose_all_frames_still_evaluated",
                reason=initial_failure or "episode_pnp_failed",
            )
        )
    else:
        initial_status = "updated"
    initial_evaluations = _evaluate_all_frames(
        episode_id=episode_id,
        requested_frame_ids=requested,
        iteration=0,
        status=initial_status,
        parent_pose=None,
        selected_pose=initial_candidate,
        stage1_by_frame=stage1_by_frame,
        replay_by_frame={},
        source_frame_ids=tuple(source.frame_id for source in stage1_sources),
        solver_failure_reason=initial_failure,
        evaluate_frame=evaluate_frame,
    )
    rounds: list[EpisodeReplayRound[T]] = [
        EpisodeReplayRound(
            iteration=0,
            parent_pose=None,
            candidate_pose=initial_candidate,
            selected_pose=initial_candidate,
            status=initial_status,
            source_frame_ids=tuple(source.frame_id for source in stage1_sources),
            replay_attempts=(),
            solver_failure_reason=initial_failure,
            evaluations=initial_evaluations,
            pnp_seed=initial_pnp_seed,
        )
    ]
    parent_pose = initial_candidate

    for iteration in range(1, refinement_iterations + 1):
        round_pnp_seed = pnp_seed_for_round(iteration)
        if parent_pose is None:
            # A pose-aligned render is impossible without the preceding shared
            # pose.  This is not a dropped frame: every requested frame still
            # receives an explicit no-pose evaluation record.
            for frame_id in requested:
                recovery_events.append(
                    RecoveryEvent(
                        iteration=iteration,
                        scope="frame",
                        frame_id=frame_id,
                        code="parent_pose_unavailable",
                        action="record_skipped_no_parent_evaluation_without_replay",
                        reason="no_parent_shared_pose",
                    )
                )
            recovery_events.append(
                RecoveryEvent(
                    iteration=iteration,
                    scope="episode",
                    code="replay_skipped_no_parent",
                    action="no_pose_all_frames_still_evaluated",
                    reason="no_parent_shared_pose",
                )
            )
            evaluations = _evaluate_all_frames(
                episode_id=episode_id,
                requested_frame_ids=requested,
                iteration=iteration,
                status="skipped_no_parent",
                parent_pose=None,
                selected_pose=None,
                stage1_by_frame=stage1_by_frame,
                replay_by_frame={},
                source_frame_ids=(),
                solver_failure_reason="no_parent_shared_pose",
                evaluate_frame=evaluate_frame,
            )
            rounds.append(
                EpisodeReplayRound(
                    iteration=iteration,
                    parent_pose=None,
                    candidate_pose=None,
                    selected_pose=None,
                    status="skipped_no_parent",
                    source_frame_ids=(),
                    replay_attempts=(),
                    solver_failure_reason="no_parent_shared_pose",
                    evaluations=evaluations,
                    pnp_seed=round_pnp_seed,
                )
            )
            continue

        replay_by_frame: dict[str, ReplayFrameAttempt] = {}
        replay_sources: list[EpisodeFrameCorrespondences] = []
        for frame_id in requested:
            attempt = replay_frame(parent_pose, frame_id, iteration)
            attempt.validate()
            if attempt.frame_id != frame_id or attempt.episode_id != episode_id:
                raise ValueError("replay hook returned an attempt for a different frame")
            replay_by_frame[frame_id] = attempt
            if attempt.artifact_recovery_state is not None:
                recovery_events.append(
                    RecoveryEvent(
                        iteration=iteration,
                        scope="frame",
                        frame_id=frame_id,
                        code="render_artifact_recovery",
                        action=attempt.artifact_recovery_state,
                        reason=attempt.failure_reason
                        or f"recorded_{attempt.artifact_recovery_state}",
                        artifact_id=attempt.artifact_id,
                        replacement_artifact_id=attempt.replacement_artifact_id,
                        artifact_recovery_state=attempt.artifact_recovery_state,
                    )
                )
            if attempt.correspondences is None:
                recovery_events.append(
                    RecoveryEvent(
                        iteration=iteration,
                        scope="frame",
                        frame_id=frame_id,
                        code="replay_correspondences_unavailable",
                        action="exclude_from_episode_pnp_but_evaluate",
                        reason=attempt.failure_reason or "replay_correspondences_unavailable",
                        artifact_id=attempt.artifact_id,
                        replacement_artifact_id=attempt.replacement_artifact_id,
                        artifact_recovery_state=attempt.artifact_recovery_state,
                    )
                )
            else:
                replay_sources.append(attempt.correspondences)

        candidate_pose, solver_failure = _solve_shared_pose(
            episode_pnp_solver,
            tuple(replay_sources),
            _round_solver_kwargs(kwargs, round_pnp_seed),
        )
        if candidate_pose is None:
            status = _failed_round_status(parent_pose)
            selected_pose = parent_pose
            recovery_events.append(
                RecoveryEvent(
                    iteration=iteration,
                    scope="episode",
                    code="episode_pnp_failed",
                    action=(
                        "retain_parent_for_all_frame_evaluation"
                        if parent_pose is not None
                        else "no_shared_pose_all_frames_still_evaluated"
                    ),
                    reason=solver_failure or "episode_pnp_failed",
                )
            )
        else:
            status = "updated"
            selected_pose = candidate_pose

        source_frame_ids = tuple(source.frame_id for source in replay_sources)
        evaluations = _evaluate_all_frames(
            episode_id=episode_id,
            requested_frame_ids=requested,
            iteration=iteration,
            status=status,
            parent_pose=parent_pose,
            selected_pose=selected_pose,
            stage1_by_frame=stage1_by_frame,
            replay_by_frame=replay_by_frame,
            source_frame_ids=source_frame_ids,
            solver_failure_reason=solver_failure,
            evaluate_frame=evaluate_frame,
        )
        rounds.append(
            EpisodeReplayRound(
                iteration=iteration,
                parent_pose=parent_pose,
                candidate_pose=candidate_pose,
                selected_pose=selected_pose,
                status=status,
                source_frame_ids=source_frame_ids,
                replay_attempts=(
                    tuple(replay_by_frame[frame_id] for frame_id in requested)
                ),
                solver_failure_reason=solver_failure,
                evaluations=evaluations,
                pnp_seed=round_pnp_seed,
            )
        )
        parent_pose = selected_pose

    result = BatchReplayResult(
        episode_id=episode_id,
        requested_frame_ids=requested,
        rounds=tuple(rounds),
        recovery_events=tuple(recovery_events),
    )
    result.validate()
    return result


__all__ = [
    "BatchReplayResult",
    "EpisodeReplayRound",
    "FrameEvaluationRecord",
    "FrameEvaluationRequest",
    "RecoveryEvent",
    "ReplayFrameAttempt",
    "Stage1FrameResult",
    "is_public_failure_code",
    "is_public_relative_identifier",
    "run_ctrnetx_closed_loop_batch_replay",
]
