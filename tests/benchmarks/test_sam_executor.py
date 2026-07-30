from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibx.benchmarks.ablations.sam import SamAblationCondition, SamAblationPair
from calibx.benchmarks.ablations.sam_executor import (
    SamEvaluationRequest,
    SamFramePreparationRequest,
    SamPose,
    SamPreparedFrame,
    SamRefinementArtifact,
    SamRefinementRequest,
    SamStage0Artifact,
    SamStage0Request,
    condition_plan,
    execute_sam_table4_condition,
)
from calibx.benchmarks.common import FrameManifest
from calibx.benchmarks.gate_v1 import DETERMINISTIC_SEED, RefinementEvidence


def _manifest() -> FrameManifest:
    return FrameManifest(
        protocol="sam_table4_test",
        dataset_id="panda",
        frame_ids=tuple(f"frames/{index:04d}" for index in range(900)),
    )


def _condition(*, mask_input: bool) -> SamAblationCondition:
    return SamAblationCondition(
        split="robot_in_view",
        manifest=_manifest(),
        mask_input=mask_input,
    )


def _pose(x: float = 0.0) -> SamPose:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = x
    return SamPose(transform, artifact_id=f"poses/{x:.2f}.npz")


def _evidence(*, accepted: bool) -> RefinementEvidence:
    return RefinementEvidence(
        finite_proper_se3=True,
        n_correspondences=20,
        n_inliers=20,
        inlier_ratio=1.0,
        positive_depth_ratio=1.0,
        reprojection_mean_px=1.0,
        reprojection_mean_diagonal_ratio=0.001,
        image_hull_area_ratio=0.1,
        world_second_spread_ratio=0.1,
        rotation_delta_deg=1.0,
        camera_center_delta_ratio=0.01,
        mask_iou_parent=None,
        mask_iou_candidate=None,
        V=accepted,
        O=accepted,
        L=accepted,
        accepted=accepted,
        reasons=() if accepted else ("gate_reject",),
    )


@dataclass
class _FakeBackend:
    fail_stage0: bool = False
    reject_first_refinement: bool = False
    bad_stage0_batch: bool = False
    preparation_requests: list[SamFramePreparationRequest] = field(default_factory=list)
    stage0_requests: list[SamStage0Request] = field(default_factory=list)
    refinement_requests: list[SamRefinementRequest] = field(default_factory=list)
    evaluation_requests: list[SamEvaluationRequest] = field(default_factory=list)

    def prepare_frame(self, request: SamFramePreparationRequest) -> SamPreparedFrame:
        self.preparation_requests.append(request)
        return SamPreparedFrame(
            frame_id=request.frame_id,
            robot_radius_m=1.0,
            input_mask_status=(
                "detected" if request.condition.mask_input else "disabled"
            ),
            input_artifact_id=f"inputs/{request.frame_id}.png",
        )

    def run_stage0(self, request: SamStage0Request) -> SamStage0Artifact:
        self.stage0_requests.append(request)
        if self.fail_stage0:
            return SamStage0Artifact(
                frame_id=request.prepared.frame_id,
                pose=None,
                failure_reason="stage1_pnp_failed",
            )
        return SamStage0Artifact(
            frame_id=request.prepared.frame_id,
            pose=_pose(),
            match_batch_size=1 if self.bad_stage0_batch else request.match_batch_size,
            artifact_id=f"stage0/{request.prepared.frame_id}/best_pose.npz",
        )

    def run_refinement(self, request: SamRefinementRequest) -> SamRefinementArtifact:
        self.refinement_requests.append(request)
        parent_x = float(request.parent_pose.world_to_camera[0, 3])
        accepted = not (self.reject_first_refinement and request.iteration == 1)
        return SamRefinementArtifact(
            frame_id=request.prepared.frame_id,
            iteration=request.iteration,
            candidate_pose=_pose(parent_x + 0.1),
            evidence=_evidence(accepted=accepted),
            match_artifact_id=(
                f"refinement/{request.prepared.frame_id}/R{request.iteration}/"
                "matches.npz"
            ),
        )

    def evaluate(self, request: SamEvaluationRequest) -> int:
        self.evaluation_requests.append(request)
        return request.iteration.iteration


def test_executor_keeps_six_view_stage0_and_one_pair_refinement_separate() -> None:
    condition = _condition(mask_input=True)
    backend = _FakeBackend()

    execution = execute_sam_table4_condition(condition, backend)

    assert len(backend.preparation_requests) == 900
    assert len(backend.stage0_requests) == 900
    assert len(backend.refinement_requests) == 900 * 3
    assert len(backend.evaluation_requests) == 900 * 4
    assert {
        (
            request.orbit_render_count,
            request.matcher_pair_count,
            request.match_batch_size,
            request.matcher_call_count,
            request.determinism_seed,
        )
        for request in backend.stage0_requests
    } == {(6, 6, 6, 1, DETERMINISTIC_SEED)}
    assert {
        (
            request.match_render_count,
            request.matcher_pair_count,
            request.match_batch_size,
            request.matcher_call_count,
            request.candidate_policy,
            request.determinism_seed,
        )
        for request in backend.refinement_requests
    } == {(1, 1, 1, 1, "gate-v1", DETERMINISTIC_SEED)}

    audit = execution.audit_record()
    assert audit["manifest_sha256"] == condition.manifest.sha256()
    assert audit["protocol"]["stage0"]["determinism_seed"] == DETERMINISTIC_SEED
    assert audit["protocol"]["refinement"]["match_batch_size"] == 1
    assert execution.summary()["iteration_statuses"]["iteration_01"] == {
        "accepted": 900
    }
    assert condition_plan(condition)["input_mask"]["enabled"] is True


def test_gate_reject_fills_the_remaining_requested_rounds() -> None:
    condition = _condition(mask_input=True)
    backend = _FakeBackend(reject_first_refinement=True)

    execution = execute_sam_table4_condition(condition, backend)

    assert len(backend.refinement_requests) == 900
    assert len(backend.evaluation_requests) == 900 * 4
    assert [row.status for row in execution.frames[0].iterations] == [
        "success",
        "rejected",
        "filled",
        "filled",
    ]
    assert [row.evaluated for row in execution.frames[0].evaluations] == [
        True,
        True,
        True,
        True,
    ]


def test_stage0_unavailable_records_all_rounds_without_replay_and_pairing_is_strict(
) -> None:
    without_sam = _condition(mask_input=False)
    with_sam = _condition(mask_input=True)
    SamAblationPair(without_sam, with_sam).validate()
    backend = _FakeBackend(fail_stage0=True)

    execution = execute_sam_table4_condition(without_sam, backend)

    assert len(backend.refinement_requests) == 0
    assert len(backend.evaluation_requests) == 0
    assert [row.status for row in execution.frames[0].iterations] == [
        "unavailable",
        "unavailable",
        "unavailable",
        "unavailable",
    ]
    assert [row.attempted for row in execution.frames[0].iterations] == [
        True,
        False,
        False,
        False,
    ]
    assert {row.prepared.input_mask_status for row in execution.frames} == {"disabled"}


def test_pair_requires_full_manifest_sha_provenance_not_only_frame_ids() -> None:
    frame_ids = _manifest().frame_ids
    without_manifest = FrameManifest(
        protocol="sam_table4_control",
        dataset_id="panda",
        frame_ids=frame_ids,
        selection_seed=11,
    )
    with_manifest = FrameManifest(
        protocol="sam_table4_treatment",
        dataset_id="panda",
        frame_ids=frame_ids,
        selection_seed=12,
    )
    try:
        SamAblationPair(
            SamAblationCondition("robot_in_view", without_manifest, False),
            SamAblationCondition("robot_in_view", with_manifest, True),
        ).validate()
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:  # pragma: no cover - makes missing provenance validation obvious
        raise AssertionError(
            "same-frame manifests with different provenance were accepted"
        )


def test_stage0_artifact_cannot_silently_change_its_six_pair_batch() -> None:
    try:
        execute_sam_table4_condition(
            _condition(mask_input=True),
            _FakeBackend(bad_stage0_batch=True),
        )
    except ValueError as error:
        assert "six-view matcher contract" in str(error)
    else:  # pragma: no cover - makes a missing validation failure obvious
        raise AssertionError("mixed Stage-0 batch size was accepted")
