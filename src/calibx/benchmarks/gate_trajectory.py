"""Framework-independent gate-v1 trajectory control flow.

Rendering, matching, and PnP provide one candidate attempt per iteration.  This
module owns the frozen decision/early-stop semantics so every benchmark uses
the same trajectory behavior and records filled iterations explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np

from .gate_v1 import RefinementEvidence, is_small_update, is_two_cycle, select_refinement_candidate


T = TypeVar("T")


@dataclass(frozen=True)
class CandidateAttempt(Generic[T]):
    """The result of rendering/matching/PnP for one refinement iteration."""

    candidate: T | None
    evidence: RefinementEvidence | None = None
    failure_reason: str | None = None

    def validate(self) -> None:
        if self.candidate is None and self.evidence is not None:
            raise ValueError("an unavailable candidate cannot have gate evidence")
        if self.candidate is not None and self.evidence is None:
            raise ValueError("an available candidate requires gate evidence")


@dataclass(frozen=True)
class TrajectoryStep(Generic[T]):
    iteration: int
    selected: T
    candidate: T | None
    accepted: bool
    status: str
    reason: str
    evidence: RefinementEvidence | None


@dataclass(frozen=True)
class GateTrajectory(Generic[T]):
    initial: T
    final: T
    steps: tuple[TrajectoryStep[T], ...]
    stop_reason: str | None


def run_gate_v1_trajectory(
    initial: T,
    *,
    iterations: int,
    candidate_attempt: Callable[[T, int], CandidateAttempt[T]],
    pose_of: Callable[[T], np.ndarray],
    robot_radius_m: float,
) -> GateTrajectory[T]:
    """Execute gate-v1 accept/reject, small-update, and two-cycle behavior.

    An unavailable candidate, rejection, or two-cycle retains the parent and
    fills all remaining requested iterations.  A small accepted update is kept
    and likewise fills remaining iterations, exactly so a requested R3 result
    always has a defined state per iteration.
    """
    if not 0 <= iterations <= 3:
        raise ValueError("gate-v1 supports between zero and three iterations")
    if robot_radius_m <= 1e-6:
        raise ValueError("robot_radius_m must be greater than 1e-6")
    parent = initial
    accepted_history: list[T] = [initial]
    steps: list[TrajectoryStep[T]] = []
    stop_reason: str | None = None
    for iteration in range(1, iterations + 1):
        attempt = candidate_attempt(parent, iteration)
        attempt.validate()
        if attempt.candidate is None:
            reason = attempt.failure_reason or "candidate_unavailable"
            steps.append(
                TrajectoryStep(
                    iteration=iteration,
                    selected=parent,
                    candidate=None,
                    accepted=False,
                    status="unavailable",
                    reason=reason,
                    evidence=None,
                )
            )
            stop_reason = reason
            break
        two_steps_before = accepted_history[-2] if len(accepted_history) >= 2 else None
        two_cycle = is_two_cycle(
            pose_of(attempt.candidate),
            None if two_steps_before is None else pose_of(two_steps_before),
            robot_radius_m,
        )
        selected, accepted, reason = select_refinement_candidate(
            parent,
            attempt.candidate,
            attempt.evidence,
            two_cycle=two_cycle,
        )
        status = "accepted" if accepted else "rejected"
        steps.append(
            TrajectoryStep(
                iteration=iteration,
                selected=selected,
                candidate=attempt.candidate,
                accepted=accepted,
                status=status,
                reason=reason,
                evidence=attempt.evidence,
            )
        )
        parent = selected
        if not accepted:
            stop_reason = reason
            break
        accepted_history.append(selected)
        if is_small_update(attempt.evidence):
            stop_reason = "small_update"
            break
    if stop_reason is not None:
        for iteration in range(len(steps) + 1, iterations + 1):
            steps.append(
                TrajectoryStep(
                    iteration=iteration,
                    selected=parent,
                    candidate=None,
                    accepted=False,
                    status="filled",
                    reason=f"filled_after_{stop_reason}",
                    evidence=None,
                )
            )
    return GateTrajectory(
        initial=initial,
        final=parent,
        steps=tuple(steps),
        stop_reason=stop_reason,
    )


__all__ = [
    "CandidateAttempt",
    "GateTrajectory",
    "TrajectoryStep",
    "run_gate_v1_trajectory",
]
