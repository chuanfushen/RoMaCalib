"""Runtime adapters for the public CTRNet-X closed-loop batch protocol.

This module turns the audited protocol boundary into executable code without
pretending that the public repository knows a private dataset layout.  The
historical CF worker supplied the following operations for every episode:

* Stage 1 runs a six-view, SAM-masked independent-frame search.  Only frames
  whose *individual* PnP succeeded contribute their saved PnP input pairs to
  the iteration-0 shared solve.
* Each R1--R3 frame is rendered exactly once from the parent shared pose.  Its
  raw post-geometry pairs are collected even though frame-level PnP is
  deliberately disabled in this stage.
* One episode-level PnP consumes the stable top-K pairs per usable frame, and
  every requested frame is evaluated at every iteration.

``batch_replay`` owns the closed-loop state machine.  This file owns only the
runtime boundary around it: a portable frame-ID/metadata manifest, typed
Stage-1/render/match/evaluation hooks, deterministic seed scheduling, and
optional adapters to the existing MuJoCo renderer and core matcher.  Dataset
readers still resolve the opaque metadata locally; no server paths, outputs,
or source dataset files are encoded here.

The optional MuJoCo/RoMa adapters import their heavy dependencies only while a
real frame is running.  Offline planning and manifest validation therefore do
not require Torch, OpenCV, MuJoCo, a GPU, or private assets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from ..common import (
    is_public_metadata_key,
    is_public_relative_identifier as _shared_public_relative_identifier,
)
from .batch_replay import (
    ArtifactRecoveryState,
    BatchReplayResult,
    EpisodePnpSolver,
    FrameEvaluationRequest,
    ReplayFrameAttempt,
    Stage1FrameResult,
    is_public_failure_code,
    run_ctrnetx_closed_loop_batch_replay,
)
from .episode_pnp import EpisodeFrameCorrespondences, EpisodePose, solve_episode_pnp
from .protocol import CtrnetxBatchManifest, CtrnetxProtocol


T = TypeVar("T")
JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

CTRNETX_BATCH_EXECUTION_SEMANTIC_ID = "ctrnetx-closed-loop-shared-episode-pnp-v1"
CTRNETX_RUNTIME_FRAME_MANIFEST_PROTOCOL = "ctrnetx_closed_loop_batch_runtime_frames"


def _is_public_relative_identifier(value: str) -> bool:
    """Keep compatibility with existing local call sites for the shared guard."""
    return _shared_public_relative_identifier(value)


def _public_json_copy(value: object, *, label: str) -> JsonValue:
    """Copy JSON metadata after rejecting non-portable string values.

    A runtime manifest may carry opaque fields such as a camera label, sample
    ordinal, or a dataset-relative key.  It must not embed an absolute local
    path, a Windows drive-relative path, or traversal.  Actual host paths stay
    inside the dataset adapter supplied at execution time.
    """
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        copied: JsonValue = json.loads(encoded)
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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_camera_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    return matrix


def _safe_artifact_id(value: str | None, *, label: str) -> None:
    if value is not None and (not value or not _is_public_relative_identifier(value)):
        raise ValueError(f"{label} must be a portable relative identifier")


def _safe_failure_code(value: str, *, label: str) -> None:
    if not is_public_failure_code(value):
        raise ValueError(f"{label} must be a stable lowercase failure code")


@dataclass(frozen=True)
class CtrnetxRuntimeFrame:
    """One local-runtime frame addressed by public IDs and opaque metadata.

    ``metadata`` is intentionally not a path resolver.  A caller-provided
    dataset adapter translates it into local file handles, joint state, camera
    intrinsics, and other machine-local assets only when executing a frame.
    ``frame_ordinal`` is explicit because it feeds the historical matching
    seed formula and cannot safely be inferred from an arbitrary frame ID.
    """

    episode_id: str
    frame_id: str
    frame_ordinal: int
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.episode_id or not _is_public_relative_identifier(self.episode_id):
            raise ValueError("episode_id must be a public relative identifier")
        if not self.frame_id or not _is_public_relative_identifier(self.frame_id):
            raise ValueError("frame_id must be a public relative identifier")
        if self.frame_ordinal < 0:
            raise ValueError("frame_ordinal must be non-negative")
        _public_json_copy(dict(self.metadata), label="metadata")

    def to_dict(self) -> dict[str, JsonValue]:
        self.validate()
        return {
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "frame_ordinal": self.frame_ordinal,
            "metadata": _public_json_copy(dict(self.metadata), label="metadata"),
        }


@dataclass(frozen=True)
class CtrnetxRuntimeEpisode:
    """Ordered runtime frame records for one shared-PnP episode."""

    episode_id: str
    frames: tuple[CtrnetxRuntimeFrame, ...]

    def validate(self) -> None:
        if not self.episode_id or not _is_public_relative_identifier(self.episode_id):
            raise ValueError("episode_id must be a public relative identifier")
        if not self.frames:
            raise ValueError("runtime episode must contain at least one frame")
        for frame in self.frames:
            frame.validate()
            if frame.episode_id != self.episode_id:
                raise ValueError("runtime frame episode_id does not match its episode")
        frame_ids = tuple(frame.frame_id for frame in self.frames)
        ordinals = tuple(frame.frame_ordinal for frame in self.frames)
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("runtime frame_id values must be unique within an episode")
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("runtime frame_ordinal values must be unique within an episode")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "episode_id": self.episode_id,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True)
class CtrnetxRuntimeFrameManifest:
    """Portable runtime frame IDs layered on top of the count-only manifest.

    ``CtrnetxBatchManifest`` remains the canonical paper scope (60 episodes,
    17,306 requested frames).  This manifest adds only the per-frame identifiers
    and local-adapter metadata needed to execute that fixed scope.  The two
    manifests are validated together before a full batch or shard starts.
    """

    episodes: tuple[CtrnetxRuntimeEpisode, ...]
    protocol: str = CTRNETX_RUNTIME_FRAME_MANIFEST_PROTOCOL
    version: int = 1

    def validate(self) -> None:
        if self.protocol != CTRNETX_RUNTIME_FRAME_MANIFEST_PROTOCOL:
            raise ValueError("unexpected CTRNet-X runtime frame manifest protocol")
        if self.version != 1:
            raise ValueError(f"unsupported runtime frame manifest version: {self.version}")
        if not self.episodes:
            raise ValueError("runtime frame manifest must contain at least one episode")
        for episode in self.episodes:
            episode.validate()
        episode_ids = tuple(episode.episode_id for episode in self.episodes)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("runtime manifest episode_id values must be unique")

    def validate_against_count_manifest(self, count_manifest: CtrnetxBatchManifest) -> None:
        """Require identical episode order and exact frame counts.

        The historical shared-PnP seed includes a global episode ordinal.  We
        therefore require order equality as well as membership/count equality;
        reordering a valid set of episodes is a different reproducibility run.
        """
        self.validate()
        count_manifest.validate()
        runtime_ids = tuple(episode.episode_id for episode in self.episodes)
        count_ids = tuple(episode.episode_id for episode in count_manifest.episodes)
        if runtime_ids != count_ids:
            raise ValueError(
                "runtime manifest episode order must match the count manifest exactly"
            )
        for runtime_episode, count_episode in zip(
            self.episodes,
            count_manifest.episodes,
            strict=True,
        ):
            if len(runtime_episode.frames) != count_episode.frame_count:
                raise ValueError(
                    "runtime frame count does not match count manifest for "
                    f"episode {runtime_episode.episode_id!r}"
                )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "protocol": self.protocol,
            "version": self.version,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> "CtrnetxRuntimeFrameManifest":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid runtime frame manifest: {path}") from error
        try:
            episodes = tuple(
                CtrnetxRuntimeEpisode(
                    episode_id=str(episode["episode_id"]),
                    frames=tuple(
                        CtrnetxRuntimeFrame(
                            episode_id=str(frame["episode_id"]),
                            frame_id=str(frame["frame_id"]),
                            frame_ordinal=int(frame["frame_ordinal"]),
                            metadata=dict(frame.get("metadata", {})),
                        )
                        for frame in episode["frames"]
                    ),
                )
                for episode in payload["episodes"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid runtime frame manifest schema: {path}") from error
        manifest = cls(
            protocol=str(payload.get("protocol", CTRNETX_RUNTIME_FRAME_MANIFEST_PROTOCOL)),
            version=int(payload.get("version", 1)),
            episodes=episodes,
        )
        manifest.validate()
        return manifest

    def episode(self, episode_id: str) -> CtrnetxRuntimeEpisode:
        self.validate()
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode
        raise KeyError(f"unknown runtime episode_id: {episode_id}")


def ctrnetx_match_seed(base_seed: int, frame_ordinal: int, iteration: int) -> int:
    """Return the audited per-frame/per-iteration matcher seed."""
    if frame_ordinal < 0 or iteration < 0:
        raise ValueError("frame_ordinal and iteration must be non-negative")
    return (int(base_seed) + int(frame_ordinal) * 1009 + int(iteration)) % (2**31 - 1)


def ctrnetx_batch_pnp_seed(base_seed: int, episode_ordinal: int, iteration: int) -> int:
    """Return the audited shared-PnP seed for T0 or an R1--R3 replay round."""
    if episode_ordinal < 0 or iteration < 0:
        raise ValueError("episode_ordinal and iteration must be non-negative")
    ordinal = episode_ordinal if iteration == 0 else episode_ordinal * 10 + iteration
    return (int(base_seed) + ordinal * 1009) % (2**31 - 1)


def configure_ctrnetx_match_rng(seed: int) -> None:
    """Apply the audited matcher seed immediately before a real match call.

    This deliberately mirrors the historical worker's Torch/NumPy/OpenCV
    ordering.  Imports remain local so the protocol module is usable for
    offline planning and tests without the optional evaluation environment.
    """
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("CTRNet-X matcher seed must be a non-negative integer")
    try:
        import cv2
        import torch
    except ImportError as error:  # pragma: no cover - depends on eval extra
        raise RuntimeError(
            "CTRNet-X matching requires the eval PyTorch and OpenCV runtime"
        ) from error
    seed_value = int(seed)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    cv2.setRNGSeed(seed_value)


@dataclass(frozen=True)
class CtrnetxBatchExecutionConfig:
    """Frozen paper settings needed by the runtime adapter.

    These values match the audited CF batch configuration.  They are validated
    here rather than silently translated into a generic single-frame runner.
    """

    stage1_views: int = 6
    stage1_match_batch_size: int = 6
    replay_views: int = 1
    visual_geom_group: int = 2
    refinement_iterations: int = 3
    mask_input: bool = True
    mask_prompt: str = "robotic arm"
    match_seed: int = 90
    topk_per_frame: int = 64
    min_source_frames: int = 2
    min_pnp_correspondences: int = 6
    reprojection_threshold_px: float = 5.0
    pnp_iterations: int = 10_000
    pnp_confidence: float = 0.999

    @classmethod
    def from_paper_config(cls, config: Mapping[str, object]) -> "CtrnetxBatchExecutionConfig":
        """Build the one frozen execution configuration from public TOML data."""
        protocol = dict(config.get("protocol", {}))
        stage1 = dict(config.get("stage1", {}))
        replay = dict(config.get("replay_render", {}))
        matching = dict(config.get("matching", {}))
        episode_pnp = dict(config.get("episode_pnp", {}))
        instance = cls(
            stage1_views=int(stage1.get("views", -1)),
            stage1_match_batch_size=int(stage1.get("match_batch_size", -1)),
            replay_views=int(replay.get("views", -1)),
            visual_geom_group=int(replay.get("visual_geom_group", -1)),
            refinement_iterations=int(protocol.get("iterations", -1)),
            mask_input=bool(matching.get("mask_input", False)),
            mask_prompt=str(matching.get("mask_prompt", "")),
            match_seed=int(episode_pnp.get("seed", -1)),
            topk_per_frame=int(episode_pnp.get("topk_per_frame", -1)),
            min_source_frames=int(episode_pnp.get("min_source_frames", -1)),
            min_pnp_correspondences=int(
                episode_pnp.get("min_pnp_correspondences", -1)
            ),
            reprojection_threshold_px=float(
                episode_pnp.get("reprojection_threshold_px", float("nan"))
            ),
            pnp_iterations=int(episode_pnp.get("iterations", -1)),
            pnp_confidence=float(episode_pnp.get("confidence", float("nan"))),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        CtrnetxProtocol(
            mode="batch",
            stage1_views=self.stage1_views,
            stage1_match_batch_size=self.stage1_match_batch_size,
            replay_views=self.replay_views,
            refinement_iterations=self.refinement_iterations,
            mask_input=self.mask_input,
        ).validate()
        if self.visual_geom_group != 2:
            raise ValueError("CTRNet-X paper batch is frozen at visual_geom_group=2")
        if self.mask_prompt != "robotic arm":
            raise ValueError("CTRNet-X paper batch is frozen at mask_prompt='robotic arm'")
        if self.match_seed != 90:
            raise ValueError("CTRNet-X paper batch is frozen at match_seed=90")
        if self.topk_per_frame != 64:
            raise ValueError("CTRNet-X paper batch is frozen at topk_per_frame=64")
        if self.min_source_frames != 2:
            raise ValueError("CTRNet-X paper batch is frozen at min_source_frames=2")
        if self.min_pnp_correspondences != 6:
            raise ValueError(
                "CTRNet-X paper batch is frozen at min_pnp_correspondences=6"
            )
        if self.reprojection_threshold_px != 5.0:
            raise ValueError(
                "CTRNet-X paper batch is frozen at reprojection_threshold_px=5.0"
            )
        if self.pnp_iterations != 10_000:
            raise ValueError("CTRNet-X paper batch is frozen at pnp_iterations=10000")
        if self.pnp_confidence != 0.999:
            raise ValueError("CTRNet-X paper batch is frozen at pnp_confidence=0.999")

    def to_record(self) -> dict[str, object]:
        """Expose every frozen value actually handed to runtime hooks/solver."""
        self.validate()
        return {
            "stage1": {
                "views": self.stage1_views,
                "match_batch_size": self.stage1_match_batch_size,
            },
            "replay_render": {
                "views": self.replay_views,
                "visual_geom_group": self.visual_geom_group,
            },
            "matching": {
                "mask_input": self.mask_input,
                "mask_prompt": self.mask_prompt,
                "match_seed": self.match_seed,
            },
            "episode_pnp": {
                "topk_per_frame": self.topk_per_frame,
                "min_source_frames": self.min_source_frames,
                "min_pnp_correspondences": self.min_pnp_correspondences,
                "reprojection_threshold_px": self.reprojection_threshold_px,
                "iterations": self.pnp_iterations,
                "confidence": self.pnp_confidence,
            },
            "refinement_iterations": self.refinement_iterations,
        }

    def pnp_solver_kwargs(self) -> dict[str, int | float]:
        """Return static solver options; the replay records its own round seed."""
        self.validate()
        return {
            "topk_per_frame": self.topk_per_frame,
            "min_source_frames": self.min_source_frames,
            "min_pnp_correspondences": self.min_pnp_correspondences,
            "reprojection_threshold": self.reprojection_threshold_px,
            "iterations": self.pnp_iterations,
            "confidence": self.pnp_confidence,
        }

    def pnp_seed(self, *, episode_ordinal: int, iteration: int) -> int:
        """Expose the exact historic seed as data, not private runner state."""
        self.validate()
        return ctrnetx_batch_pnp_seed(self.match_seed, episode_ordinal, iteration)


@dataclass(frozen=True)
class Stage1Request:
    """One six-view Stage-1 request supplied to a dataset/runtime adapter."""

    frame: CtrnetxRuntimeFrame
    match_seed: int
    views: int = 6
    match_batch_size: int = 6
    mask_input: bool = True
    mask_prompt: str = "robotic arm"
    iteration: int = 0

    def validate(self) -> None:
        self.frame.validate()
        if self.iteration != 0:
            raise ValueError("Stage 1 must be recorded as iteration 0")
        if self.views != 6 or self.match_batch_size != 6:
            raise ValueError("CTRNet-X Stage 1 requires six rendered/matched views")
        if not self.mask_input:
            raise ValueError("CTRNet-X Stage 1 requires SAM input masking")
        if self.mask_prompt != "robotic arm":
            raise ValueError("CTRNet-X Stage 1 requires the archived SAM prompt")


@dataclass(frozen=True)
class Stage1FrameArtifact:
    """A Stage-1 result plus portable audit fields.

    On individual PnP success, ``result.correspondences`` must be the saved
    PnP *input* pairs (the historical ``_pose`` arrays), not the raw matcher
    pairs.  A failed Stage 1 may retain raw pairs for its own diagnostics, but
    ``batch_replay`` will exclude it from the T0 shared solve.
    """

    result: Stage1FrameResult
    artifact_id: str | None = None
    render_count: int = 6
    match_batch_size: int = 6
    match_seed: int | None = None

    def validate(self) -> None:
        self.result.validate()
        if self.render_count != 6 or self.match_batch_size != 6:
            raise ValueError("CTRNet-X Stage 1 artifact must contain six-view matching")
        _safe_artifact_id(self.artifact_id, label="Stage-1 artifact_id")


@dataclass(frozen=True)
class PoseAlignedRenderRequest:
    """One R1--R3 parent-pose-aligned rendering request."""

    frame: CtrnetxRuntimeFrame
    parent_pose: EpisodePose
    iteration: int
    match_seed: int
    views: int = 1
    visual_geom_group: int = 2

    def validate(self) -> None:
        self.frame.validate()
        if self.iteration not in {1, 2, 3}:
            raise ValueError("pose-aligned replay iterations must be in [1, 3]")
        if self.views != 1:
            raise ValueError("each active replay frame requires exactly one render")
        if self.visual_geom_group != 2:
            raise ValueError("pose-aligned replay requires visual_geom_group=2")
        transform = np.asarray(self.parent_pose.world_to_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("parent shared pose must contain a finite 4x4 transform")


@dataclass(frozen=True)
class PoseAlignedRenderArtifact:
    """A one-render runtime artifact; ``handle`` is intentionally local-only."""

    episode_id: str
    frame_id: str
    artifact_id: str | None
    handle: object = None
    render_count: int = 1
    replacement_artifact_id: str | None = None
    artifact_recovery_state: ArtifactRecoveryState | None = None

    def validate(self) -> None:
        if not self.episode_id or not _is_public_relative_identifier(self.episode_id):
            raise ValueError("render artifact episode_id must be public-safe")
        if not self.frame_id or not _is_public_relative_identifier(self.frame_id):
            raise ValueError("render artifact frame_id must be public-safe")
        if self.render_count != 1:
            raise ValueError("each pose-aligned artifact must contain exactly one render")
        _safe_artifact_id(self.artifact_id, label="render artifact_id")
        _safe_artifact_id(
            self.replacement_artifact_id,
            label="render replacement_artifact_id",
        )
        if self.replacement_artifact_id is not None and self.artifact_id is None:
            raise ValueError("render replacement_artifact_id requires artifact_id")
        if self.artifact_recovery_state not in {
            None,
            "cache_hit",
            "recompute_cached_render",
            "rerender",
            "recovery_failed",
        }:
            raise ValueError("unsupported render artifact recovery state")
        if self.artifact_recovery_state is not None and self.artifact_id is None:
            raise ValueError("render artifact recovery state requires artifact_id")


@dataclass(frozen=True)
class RawPostGeometryMatchRequest:
    """A matcher/geometry request after its one pose-aligned render.

    ``forbid_frame_level_pnp`` is not advisory: an adapter that wraps the
    legacy core must set its minimum per-frame PnP correspondence count high
    enough that this request only exposes raw post-geometry correspondences.
    """

    frame: CtrnetxRuntimeFrame
    parent_pose: EpisodePose
    iteration: int
    match_seed: int
    render: PoseAlignedRenderArtifact
    correspondence_space: str = "raw_post_geometry"
    forbid_frame_level_pnp: bool = True

    def validate(self) -> None:
        self.frame.validate()
        self.render.validate()
        if self.render.episode_id != self.frame.episode_id:
            raise ValueError("render artifact episode_id does not match runtime frame")
        if self.render.frame_id != self.frame.frame_id:
            raise ValueError("render artifact frame_id does not match runtime frame")
        if self.iteration not in {1, 2, 3}:
            raise ValueError("raw post-geometry matching is only valid at R1--R3")
        if self.correspondence_space != "raw_post_geometry":
            raise ValueError("CTRNet-X replay must use raw_post_geometry correspondences")
        if not self.forbid_frame_level_pnp:
            raise ValueError("CTRNet-X replay must not run frame-level PnP")


@dataclass(frozen=True)
class RawPostGeometryMatchArtifact:
    """The raw mesh-picked pairs from one replay render, or a safe failure."""

    correspondences: EpisodeFrameCorrespondences | None
    failure_reason: str | None = None
    artifact_id: str | None = None
    replacement_artifact_id: str | None = None
    artifact_recovery_state: ArtifactRecoveryState | None = None

    def validate(self, *, frame: CtrnetxRuntimeFrame) -> None:
        _safe_artifact_id(self.artifact_id, label="match artifact_id")
        _safe_artifact_id(
            self.replacement_artifact_id,
            label="match replacement_artifact_id",
        )
        if self.replacement_artifact_id is not None and self.artifact_id is None:
            raise ValueError("match replacement_artifact_id requires artifact_id")
        if self.artifact_recovery_state not in {
            None,
            "cache_hit",
            "recompute_cached_render",
            "rerender",
            "recovery_failed",
        }:
            raise ValueError("unsupported match artifact recovery state")
        if self.artifact_recovery_state is not None and self.artifact_id is None:
            raise ValueError("match artifact recovery state requires artifact_id")
        if self.correspondences is None:
            if not self.failure_reason:
                raise ValueError("unavailable raw correspondences require a failure reason")
            _safe_failure_code(
                self.failure_reason,
                label="raw post-geometry failure_reason",
            )
            return
        if self.failure_reason is not None:
            raise ValueError("available raw correspondences cannot carry a failure reason")
        self.correspondences.validate()
        if self.correspondences.episode_id != frame.episode_id:
            raise ValueError("raw correspondence episode_id does not match runtime frame")
        if self.correspondences.frame_id != frame.frame_id:
            raise ValueError("raw correspondence frame_id does not match runtime frame")
        if len(self.correspondences.scores) == 0:
            raise ValueError("empty raw correspondences must be recorded as unavailable")


@dataclass(frozen=True)
class CtrnetxFrameEvaluationRequest:
    """Runtime frame context paired with the replay state handed to metrics."""

    frame: CtrnetxRuntimeFrame
    replay_request: FrameEvaluationRequest

    def validate(self) -> None:
        self.frame.validate()
        if self.replay_request.episode_id != self.frame.episode_id:
            raise ValueError("evaluation replay episode_id does not match runtime frame")
        if self.replay_request.frame_id != self.frame.frame_id:
            raise ValueError("evaluation replay frame_id does not match runtime frame")


class CtrnetxFrameRuntimeFailure(RuntimeError):
    """An explicitly recoverable per-frame render/match failure.

    Hooks should use a stable, path-free ``code`` instead of leaking exception
    text that may contain private dataset or server locations.  Unexpected
    exceptions are intentionally propagated rather than converted into a
    silent recovery event.
    """

    def __init__(
        self,
        code: str,
        *,
        artifact_id: str | None = None,
        replacement_artifact_id: str | None = None,
        artifact_recovery_state: ArtifactRecoveryState | None = None,
    ) -> None:
        _safe_failure_code(code, label="runtime failure code")
        _safe_artifact_id(artifact_id, label="runtime failure artifact_id")
        _safe_artifact_id(
            replacement_artifact_id,
            label="runtime failure replacement_artifact_id",
        )
        if replacement_artifact_id is not None and artifact_id is None:
            raise ValueError("runtime failure replacement_artifact_id requires artifact_id")
        if artifact_recovery_state not in {
            None,
            "cache_hit",
            "recompute_cached_render",
            "rerender",
            "recovery_failed",
        }:
            raise ValueError("unsupported runtime failure artifact recovery state")
        if artifact_recovery_state is not None and artifact_id is None:
            raise ValueError("runtime failure artifact recovery state requires artifact_id")
        self.code = code
        self.artifact_id = artifact_id
        self.replacement_artifact_id = replacement_artifact_id
        self.artifact_recovery_state = artifact_recovery_state
        super().__init__(code)


Stage1Executor = Callable[[Stage1Request], Stage1FrameArtifact]
PoseAlignedRenderer = Callable[[PoseAlignedRenderRequest], PoseAlignedRenderArtifact]
RawPostGeometryMatcher = Callable[
    [RawPostGeometryMatchRequest], RawPostGeometryMatchArtifact
]
FrameEvaluator = Callable[[CtrnetxFrameEvaluationRequest], T]


@dataclass(frozen=True)
class CtrnetxBatchHooks(Generic[T]):
    """Runtime operations deliberately kept outside the public dataset schema."""

    run_stage1: Stage1Executor
    render_pose_aligned: PoseAlignedRenderer
    match_raw_post_geometry: RawPostGeometryMatcher
    evaluate_frame: FrameEvaluator[T]

    def validate(self) -> None:
        for name, callback in (
            ("run_stage1", self.run_stage1),
            ("render_pose_aligned", self.render_pose_aligned),
            ("match_raw_post_geometry", self.match_raw_post_geometry),
            ("evaluate_frame", self.evaluate_frame),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")


@dataclass(frozen=True)
class CtrnetxBatchEpisodeExecution(Generic[T]):
    """Auditable execution result for one shared-pose episode."""

    semantic_id: str
    episode_id: str
    episode_ordinal: int
    stage1_artifacts: tuple[Stage1FrameArtifact, ...]
    replay: BatchReplayResult[T]
    pnp_seed_by_iteration: tuple[int, int, int, int]

    def validate(self) -> None:
        if self.semantic_id != CTRNETX_BATCH_EXECUTION_SEMANTIC_ID:
            raise ValueError("unexpected CTRNet-X batch execution semantic ID")
        if self.episode_ordinal < 0:
            raise ValueError("episode_ordinal must be non-negative")
        if not self.episode_id or not _is_public_relative_identifier(self.episode_id):
            raise ValueError("episode_id must be a public relative identifier")
        if len(self.pnp_seed_by_iteration) != 4:
            raise ValueError("batch execution must expose T0 through R3 PnP seeds")
        for artifact in self.stage1_artifacts:
            artifact.validate()
            if artifact.result.episode_id != self.episode_id:
                raise ValueError("Stage-1 artifact episode_id does not match execution")
        stage1_ids = tuple(artifact.result.frame_id for artifact in self.stage1_artifacts)
        if stage1_ids != self.replay.requested_frame_ids:
            raise ValueError("Stage-1 artifact order must match replay requested frames")
        self.replay.validate()
        if self.replay.episode_id != self.episode_id:
            raise ValueError("replay episode_id does not match execution")

    def audit_record(self) -> dict[str, object]:
        """Return serializable protocol/audit data without local paths or arrays."""
        self.validate()
        return {
            "semantic_id": self.semantic_id,
            "episode_id": self.episode_id,
            "episode_ordinal": self.episode_ordinal,
            "requested_frame_count": len(self.replay.requested_frame_ids),
            "pnp_seed_by_iteration": list(self.pnp_seed_by_iteration),
            "stage1": [
                {
                    "frame_id": artifact.result.frame_id,
                    "pnp_success": artifact.result.pnp_success,
                    "failure_reason": artifact.result.failure_reason,
                    "correspondence_count": (
                        0
                        if artifact.result.correspondences is None
                        else len(artifact.result.correspondences.scores)
                    ),
                    "artifact_id": artifact.artifact_id,
                    "match_seed": artifact.match_seed,
                }
                for artifact in self.stage1_artifacts
            ],
            "rounds": [
                {
                    "iteration": round_.iteration,
                    "pnp_seed": round_.pnp_seed,
                    "status": round_.status,
                    "source_frame_ids": list(round_.source_frame_ids),
                    "solver_failure_reason": round_.solver_failure_reason,
                    "evaluated_frame_count": len(round_.evaluations),
                    "replay_attempts": [
                        {
                            "frame_id": attempt.frame_id,
                            "match_seed": attempt.match_seed,
                            "render_count": attempt.render_count,
                            "correspondence_count": (
                                0
                                if attempt.correspondences is None
                                else len(attempt.correspondences.scores)
                            ),
                            "artifact_id": attempt.artifact_id,
                            "replacement_artifact_id": attempt.replacement_artifact_id,
                            "artifact_recovery_state": attempt.artifact_recovery_state,
                            "failure_reason": attempt.failure_reason,
                        }
                        for attempt in round_.replay_attempts
                    ],
                }
                for round_ in self.replay.rounds
            ],
            "recovery_events": list(self.replay.recovery_audit_records()),
        }


def _bound_stage1_artifact(
    artifact: Stage1FrameArtifact,
    request: Stage1Request,
) -> Stage1FrameArtifact:
    if not isinstance(artifact, Stage1FrameArtifact):
        raise TypeError("run_stage1 hook must return Stage1FrameArtifact")
    artifact.validate()
    if artifact.result.episode_id != request.frame.episode_id:
        raise ValueError("Stage-1 hook returned an artifact for a different episode")
    if artifact.result.frame_id != request.frame.frame_id:
        raise ValueError("Stage-1 hook returned an artifact for a different frame")
    if artifact.match_seed is not None and artifact.match_seed != request.match_seed:
        raise ValueError("Stage-1 artifact match_seed does not match the request")
    return replace(artifact, match_seed=request.match_seed)


def _failed_stage1_artifact_from_runtime_failure(
    request: Stage1Request,
    failure: CtrnetxFrameRuntimeFailure,
) -> Stage1FrameArtifact:
    """Preserve a recoverable T0 frame failure and continue episode coverage."""
    return Stage1FrameArtifact(
        result=Stage1FrameResult(
            frame_id=request.frame.frame_id,
            episode_id=request.frame.episode_id,
            pnp_success=False,
            correspondences=None,
            failure_reason=failure.code,
        ),
        artifact_id=failure.artifact_id,
        match_seed=request.match_seed,
    )


def _replay_attempt_from_hooks(
    request: PoseAlignedRenderRequest,
    hooks: CtrnetxBatchHooks[Any],
) -> ReplayFrameAttempt:
    """Execute one render then one raw-matcher call for an active replay frame."""
    request.validate()
    try:
        render = hooks.render_pose_aligned(request)
    except CtrnetxFrameRuntimeFailure as failure:
        return ReplayFrameAttempt(
            frame_id=request.frame.frame_id,
            episode_id=request.frame.episode_id,
            correspondences=None,
            failure_reason=failure.code,
            artifact_id=failure.artifact_id,
            replacement_artifact_id=failure.replacement_artifact_id,
            artifact_recovery_state=failure.artifact_recovery_state,
            match_seed=request.match_seed,
        )
    if not isinstance(render, PoseAlignedRenderArtifact):
        raise TypeError("render_pose_aligned hook must return PoseAlignedRenderArtifact")
    render.validate()
    if render.episode_id != request.frame.episode_id or render.frame_id != request.frame.frame_id:
        raise ValueError("render hook returned an artifact for a different frame")

    match_request = RawPostGeometryMatchRequest(
        frame=request.frame,
        parent_pose=request.parent_pose,
        iteration=request.iteration,
        match_seed=request.match_seed,
        render=render,
    )
    try:
        match = hooks.match_raw_post_geometry(match_request)
    except CtrnetxFrameRuntimeFailure as failure:
        return ReplayFrameAttempt(
            frame_id=request.frame.frame_id,
            episode_id=request.frame.episode_id,
            correspondences=None,
            failure_reason=failure.code,
            artifact_id=failure.artifact_id or render.artifact_id,
            replacement_artifact_id=(
                failure.replacement_artifact_id or render.replacement_artifact_id
            ),
            artifact_recovery_state=(
                failure.artifact_recovery_state or render.artifact_recovery_state
            ),
            match_seed=request.match_seed,
        )
    if not isinstance(match, RawPostGeometryMatchArtifact):
        raise TypeError(
            "match_raw_post_geometry hook must return RawPostGeometryMatchArtifact"
        )
    match.validate(frame=request.frame)
    artifact_id = match.artifact_id or render.artifact_id
    replacement_artifact_id = (
        match.replacement_artifact_id or render.replacement_artifact_id
    )
    recovery_state = match.artifact_recovery_state or render.artifact_recovery_state
    return ReplayFrameAttempt(
        frame_id=request.frame.frame_id,
        episode_id=request.frame.episode_id,
        correspondences=match.correspondences,
        failure_reason=match.failure_reason,
        artifact_id=artifact_id,
        replacement_artifact_id=replacement_artifact_id,
        artifact_recovery_state=recovery_state,
        match_seed=request.match_seed,
    )


def run_ctrnetx_batch_episode(
    episode: CtrnetxRuntimeEpisode,
    *,
    episode_ordinal: int,
    config: CtrnetxBatchExecutionConfig,
    hooks: CtrnetxBatchHooks[T],
    episode_pnp_solver: EpisodePnpSolver = solve_episode_pnp,
) -> CtrnetxBatchEpisodeExecution[T]:
    """Execute T0 plus R1--R3 for one runtime episode.

    This is the concrete connection from real Stage-1 artifacts, one-render
    replay matching, shared episode PnP, and all-frame metrics into
    :func:`run_ctrnetx_closed_loop_batch_replay`.  It neither reads a dataset
    directly nor mutates the core ``romav2`` implementation.
    """
    episode.validate()
    config.validate()
    hooks.validate()
    if episode_ordinal < 0:
        raise ValueError("episode_ordinal must be non-negative")
    if not callable(episode_pnp_solver):
        raise TypeError("episode_pnp_solver must be callable")

    frames_by_id = {frame.frame_id: frame for frame in episode.frames}
    stage1_artifacts: list[Stage1FrameArtifact] = []
    for frame in episode.frames:
        request = Stage1Request(
            frame=frame,
            match_seed=ctrnetx_match_seed(config.match_seed, frame.frame_ordinal, 0),
            views=config.stage1_views,
            match_batch_size=config.stage1_match_batch_size,
            mask_input=config.mask_input,
            mask_prompt=config.mask_prompt,
        )
        request.validate()
        try:
            artifact = hooks.run_stage1(request)
        except CtrnetxFrameRuntimeFailure as failure:
            artifact = _failed_stage1_artifact_from_runtime_failure(request, failure)
        stage1_artifacts.append(_bound_stage1_artifact(artifact, request))

    def replay_frame(parent_pose: EpisodePose, frame_id: str, iteration: int) -> ReplayFrameAttempt:
        frame = frames_by_id[frame_id]
        return _replay_attempt_from_hooks(
            PoseAlignedRenderRequest(
                frame=frame,
                parent_pose=parent_pose,
                iteration=iteration,
                match_seed=ctrnetx_match_seed(
                    config.match_seed,
                    frame.frame_ordinal,
                    iteration,
                ),
                views=config.replay_views,
                visual_geom_group=config.visual_geom_group,
            ),
            hooks,
        )

    def evaluate_frame(request: FrameEvaluationRequest) -> T:
        runtime_request = CtrnetxFrameEvaluationRequest(
            frame=frames_by_id[request.frame_id],
            replay_request=request,
        )
        runtime_request.validate()
        return hooks.evaluate_frame(runtime_request)

    replay = run_ctrnetx_closed_loop_batch_replay(
        episode.episode_id,
        tuple(frame.frame_id for frame in episode.frames),
        tuple(artifact.result for artifact in stage1_artifacts),
        replay_frame=replay_frame,
        evaluate_frame=evaluate_frame,
        episode_pnp_solver=episode_pnp_solver,
        solver_kwargs=config.pnp_solver_kwargs(),
        pnp_seed_for_iteration=lambda iteration: config.pnp_seed(
            episode_ordinal=episode_ordinal,
            iteration=iteration,
        ),
        refinement_iterations=config.refinement_iterations,
    )
    result = CtrnetxBatchEpisodeExecution(
        semantic_id=CTRNETX_BATCH_EXECUTION_SEMANTIC_ID,
        episode_id=episode.episode_id,
        episode_ordinal=episode_ordinal,
        stage1_artifacts=tuple(stage1_artifacts),
        replay=replay,
        pnp_seed_by_iteration=tuple(
            config.pnp_seed(episode_ordinal=episode_ordinal, iteration=iteration)
            for iteration in range(4)
        ),
    )
    result.validate()
    return result


def run_ctrnetx_batch_shard(
    count_manifest: CtrnetxBatchManifest,
    runtime_manifest: CtrnetxRuntimeFrameManifest,
    *,
    episode_ids: Sequence[str],
    config: CtrnetxBatchExecutionConfig,
    hooks: CtrnetxBatchHooks[T],
    episode_pnp_solver: EpisodePnpSolver = solve_episode_pnp,
) -> tuple[CtrnetxBatchEpisodeExecution[T], ...]:
    """Run an explicit shard after validating it against the full paper scope."""
    count_manifest.validate()
    runtime_manifest.validate_against_count_manifest(count_manifest)
    if not episode_ids:
        raise ValueError("batch shard must contain at least one episode_id")
    requested = tuple(episode_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("batch shard episode_id values must be unique")
    ordinal_by_id = {
        episode.episode_id: ordinal
        for ordinal, episode in enumerate(runtime_manifest.episodes)
    }
    unknown = [episode_id for episode_id in requested if episode_id not in ordinal_by_id]
    if unknown:
        raise KeyError(f"batch shard includes unknown episode IDs: {unknown[:3]}")
    return tuple(
        run_ctrnetx_batch_episode(
            runtime_manifest.episode(episode_id),
            episode_ordinal=ordinal_by_id[episode_id],
            config=config,
            hooks=hooks,
            episode_pnp_solver=episode_pnp_solver,
        )
        for episode_id in requested
    )


def run_ctrnetx_full_batch(
    count_manifest: CtrnetxBatchManifest,
    runtime_manifest: CtrnetxRuntimeFrameManifest,
    *,
    config: CtrnetxBatchExecutionConfig,
    hooks: CtrnetxBatchHooks[T],
    episode_pnp_solver: EpisodePnpSolver = solve_episode_pnp,
) -> tuple[CtrnetxBatchEpisodeExecution[T], ...]:
    """Run the validated 60-episode, 17,306-frame paper scope sequentially.

    Shard scheduling remains an explicit caller concern: use the four
    assignments produced by ``plan_batch_shards.py`` and call
    :func:`run_ctrnetx_batch_shard` once per worker.  This function exists for
    a single-worker formal reproduction and deliberately has no hidden process
    pool or GPU selection policy.
    """
    return run_ctrnetx_batch_shard(
        count_manifest,
        runtime_manifest,
        episode_ids=tuple(episode.episode_id for episode in runtime_manifest.episodes),
        config=config,
        hooks=hooks,
        episode_pnp_solver=episode_pnp_solver,
    )


# -- Evidence-preserving adapters for existing core output -----------------

def _core_array(source: Mapping[str, Any], name: str, width: int) -> np.ndarray:
    if name not in source:
        raise ValueError(f"core match record is missing {name!r}")
    array = np.asarray(source[name], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise ValueError(f"core match record {name!r} must be finite [N, {width}]")
    return array


def _core_scores(source: Mapping[str, Any], count: int) -> np.ndarray:
    if "scores" not in source or source["scores"] is None:
        raise ValueError("core match record is missing 'scores'")
    try:
        scores = np.asarray(source["scores"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("core match record scores must be numeric") from error
    if len(scores) != count or not np.isfinite(scores).all():
        raise ValueError("core match record scores must be finite [N]")
    return scores


def _core_correspondences(
    *,
    frame: CtrnetxRuntimeFrame,
    camera_matrix: np.ndarray,
    source: Mapping[str, Any],
    prefix: str,
) -> EpisodeFrameCorrespondences:
    image_points = _core_array(source, f"{prefix}image_points", 2)
    world_points = _core_array(source, f"{prefix}world_points", 3)
    if len(image_points) != len(world_points):
        raise ValueError("core image_points and world_points lengths differ")
    scores = _core_scores(
        {
            "scores": source.get(f"{prefix}scores"),
        },
        len(image_points),
    )
    return EpisodeFrameCorrespondences(
        frame_id=frame.frame_id,
        episode_id=frame.episode_id,
        image_points=image_points,
        world_points=world_points,
        scores=scores,
        camera_matrix=_validate_camera_matrix(camera_matrix),
    )


def stage1_artifact_from_core_best_view(
    request: Stage1Request,
    best_view: Mapping[str, Any],
    camera_matrix: np.ndarray,
    *,
    artifact_id: str | None = None,
) -> Stage1FrameArtifact:
    """Convert a core six-view selected record into the T0 source artifact.

    This preserves the audited selection distinction: a successful individual
    PnP contributes the PnP input arrays stored in ``best_view['_pose']``;
    it does *not* contribute raw post-geometry arrays.  Failed Stage-1 frames
    may retain raw arrays in their artifact but remain ineligible for T0.
    """
    request.validate()
    pnp = best_view.get("pnp")
    if not isinstance(pnp, Mapping) or not isinstance(pnp.get("status"), str):
        raise ValueError("core best view must provide pnp.status")
    pnp_success = pnp["status"] == "success"
    if pnp_success:
        pose = best_view.get("_pose")
        if not isinstance(pose, Mapping):
            raise ValueError("successful core best view is missing _pose arrays")
        correspondences = _core_correspondences(
            frame=request.frame,
            camera_matrix=camera_matrix,
            source=pose,
            prefix="",
        )
        if len(correspondences.scores) == 0:
            raise ValueError("successful core Stage-1 PnP cannot have empty input pairs")
        failure_reason = None
    else:
        # Retain the raw pairs only as a diagnostic artifact.  batch_replay's
        # Stage1FrameResult is the sole authority that excludes this frame from
        # the initial shared solve.
        correspondences = _core_correspondences(
            frame=request.frame,
            camera_matrix=camera_matrix,
            source=best_view,
            prefix="_",
        )
        status_code = str(pnp["status"])
        if not is_public_failure_code(status_code):
            status_code = "unsuccessful"
        failure_reason = f"stage1_individual_pnp_{status_code}"
    return Stage1FrameArtifact(
        result=Stage1FrameResult(
            frame_id=request.frame.frame_id,
            episode_id=request.frame.episode_id,
            pnp_success=pnp_success,
            correspondences=correspondences,
            failure_reason=failure_reason,
        ),
        artifact_id=artifact_id,
        match_seed=request.match_seed,
    )


def raw_post_geometry_from_core_best_view(
    request: RawPostGeometryMatchRequest,
    best_view: Mapping[str, Any],
    camera_matrix: np.ndarray,
    *,
    artifact_id: str | None = None,
    replacement_artifact_id: str | None = None,
    artifact_recovery_state: ArtifactRecoveryState | None = None,
) -> RawPostGeometryMatchArtifact:
    """Convert a one-render core result into a replay raw-correspondence artifact."""
    request.validate()
    pnp = best_view.get("pnp")
    if isinstance(pnp, Mapping) and pnp.get("status") == "success":
        raise ValueError(
            "CTRNet-X replay adapter received a frame-level PnP success; "
            "disable frame-level PnP before extracting raw correspondences"
        )
    correspondences = _core_correspondences(
        frame=request.frame,
        camera_matrix=camera_matrix,
        source=best_view,
        prefix="_",
    )
    if len(correspondences.scores) == 0:
        return RawPostGeometryMatchArtifact(
            correspondences=None,
            failure_reason="no_raw_post_geometry_correspondences",
            artifact_id=artifact_id,
            replacement_artifact_id=replacement_artifact_id,
            artifact_recovery_state=artifact_recovery_state,
        )
    return RawPostGeometryMatchArtifact(
        correspondences=correspondences,
        artifact_id=artifact_id,
        replacement_artifact_id=replacement_artifact_id,
        artifact_recovery_state=artifact_recovery_state,
    )


@dataclass(frozen=True)
class MujocoPoseAlignedFrameContext:
    """Machine-local inputs required by the optional MuJoCo replay renderer.

    This context is supplied by a dataset adapter after it restores the frame's
    robot state.  ``output_dir`` and all handles are runtime-only and never
    enter a public manifest or :meth:`CtrnetxBatchEpisodeExecution.audit_record`.
    """

    episode_id: str
    frame_id: str
    model: object
    data: object
    camera_matrix: np.ndarray
    settings: object
    output_dir: Path
    stem: str = "projected"

    def validate(self, *, frame: CtrnetxRuntimeFrame) -> None:
        if self.episode_id != frame.episode_id or self.frame_id != frame.frame_id:
            raise ValueError("MuJoCo context does not match the requested runtime frame")
        _validate_camera_matrix(self.camera_matrix)
        if not self.stem or Path(self.stem).name != self.stem:
            raise ValueError("MuJoCo render stem must be one file-name component")


@dataclass(frozen=True)
class MujocoPoseAlignedRenderHandle:
    """Local-only paths/mask passed directly from the renderer to the matcher."""

    render_path: Path
    mask_path: Path
    camera_path: Path
    render_mask: np.ndarray


MujocoFrameContextFactory = Callable[[CtrnetxRuntimeFrame], MujocoPoseAlignedFrameContext]


def _default_replay_artifact_id(request: PoseAlignedRenderRequest) -> str:
    return (
        f"replay/{request.frame.episode_id}/{request.frame.frame_id}/"
        f"iteration_{request.iteration:02d}/projected.png"
    )


def make_mujoco_pose_aligned_renderer(
    context_for_frame: MujocoFrameContextFactory,
) -> PoseAlignedRenderer:
    """Build a real one-render adapter using ``gate_rendering`` at runtime.

    The factory itself is light-weight.  MuJoCo and PIL are imported only when
    the returned callback renders a frame, so offline manifest tooling stays
    dependency-free.
    """
    if not callable(context_for_frame):
        raise TypeError("context_for_frame must be callable")

    def render(request: PoseAlignedRenderRequest) -> PoseAlignedRenderArtifact:
        request.validate()
        context = context_for_frame(request.frame)
        if not isinstance(context, MujocoPoseAlignedFrameContext):
            raise TypeError("context_for_frame must return MujocoPoseAlignedFrameContext")
        context.validate(frame=request.frame)
        # Lazy by design: importing the optional renderer must not affect
        # --help, manifest validation, or unit tests without a GPU stack.
        from calibx.benchmarks.gate_rendering import render_pose_aligned_artifacts

        render_path, mask_path, camera_path, render_mask = render_pose_aligned_artifacts(
            context.model,
            context.data,
            context.settings,
            _validate_camera_matrix(context.camera_matrix),
            np.asarray(request.parent_pose.world_to_camera, dtype=np.float64),
            Path(context.output_dir),
            stem=context.stem,
        )
        return PoseAlignedRenderArtifact(
            episode_id=request.frame.episode_id,
            frame_id=request.frame.frame_id,
            artifact_id=_default_replay_artifact_id(request),
            handle=MujocoPoseAlignedRenderHandle(
                render_path=Path(render_path),
                mask_path=Path(mask_path),
                camera_path=Path(camera_path),
                render_mask=np.asarray(render_mask, dtype=bool),
            ),
        )

    return render


@dataclass(frozen=True)
class CoreRawPostGeometryContext:
    """Machine-local inputs for the optional existing-core matcher adapter."""

    episode_id: str
    frame_id: str
    observed_rgb: np.ndarray
    matcher: object
    model: object
    data: object
    camera_matrix: np.ndarray
    matcher_args: object
    output_dir: Path

    def validate(self, *, frame: CtrnetxRuntimeFrame) -> None:
        if self.episode_id != frame.episode_id or self.frame_id != frame.frame_id:
            raise ValueError("core matcher context does not match the runtime frame")
        image = np.asarray(self.observed_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("observed_rgb must have shape [H, W, 3]")
        _validate_camera_matrix(self.camera_matrix)


CoreRawPostGeometryContextFactory = Callable[[CtrnetxRuntimeFrame], CoreRawPostGeometryContext]


def make_core_raw_post_geometry_matcher(
    context_for_frame: CoreRawPostGeometryContextFactory,
) -> RawPostGeometryMatcher:
    """Build a one-render raw-correspondence adapter around existing core code.

    The caller provides an already SAM-masked observation and a restored model
    state.  The adapter invokes the existing ``calibx.pose.match_one_render``
    exactly once, forces its frame-level PnP threshold beyond reach, and then
    extracts only its raw post-geometry arrays.  No ``romav2`` source is
    changed, and all heavy imports remain inside the returned callback.
    """
    if not callable(context_for_frame):
        raise TypeError("context_for_frame must be callable")

    def match(request: RawPostGeometryMatchRequest) -> RawPostGeometryMatchArtifact:
        request.validate()
        context = context_for_frame(request.frame)
        if not isinstance(context, CoreRawPostGeometryContext):
            raise TypeError("context_for_frame must return CoreRawPostGeometryContext")
        context.validate(frame=request.frame)
        handle = request.render.handle
        if not isinstance(handle, MujocoPoseAlignedRenderHandle):
            raise TypeError(
                "core raw matcher requires a MujocoPoseAlignedRenderHandle from "
                "make_mujoco_pose_aligned_renderer"
            )
        # ``calibx.pose`` imports Torch/OpenCV; defer it until a genuine run.
        from copy import copy

        from calibx.pose import match_one_render

        args = copy(context.matcher_args)
        # This is the crucial historical replay boundary: match/geometry runs,
        # but the per-frame PnP decision is not allowed to become a replay
        # source.  The episode solver below consumes the raw pairs instead.
        setattr(args, "min_pnp_correspondences", 1_000_000_000)
        configure_ctrnetx_match_rng(request.match_seed)
        output_dir = Path(context.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_view = match_one_render(
            np.asarray(context.observed_rgb),
            handle.render_path,
            context.matcher,
            context.model,
            context.data,
            _validate_camera_matrix(context.camera_matrix),
            args,
            output_dir,
        )
        return raw_post_geometry_from_core_best_view(
            request,
            best_view,
            context.camera_matrix,
            artifact_id=request.render.artifact_id,
            replacement_artifact_id=request.render.replacement_artifact_id,
            artifact_recovery_state=request.render.artifact_recovery_state,
        )

    return match


__all__ = [
    "CTRNETX_BATCH_EXECUTION_SEMANTIC_ID",
    "CTRNETX_RUNTIME_FRAME_MANIFEST_PROTOCOL",
    "CoreRawPostGeometryContext",
    "CoreRawPostGeometryContextFactory",
    "CtrnetxBatchEpisodeExecution",
    "CtrnetxBatchExecutionConfig",
    "CtrnetxBatchHooks",
    "CtrnetxFrameEvaluationRequest",
    "CtrnetxFrameRuntimeFailure",
    "CtrnetxRuntimeEpisode",
    "CtrnetxRuntimeFrame",
    "CtrnetxRuntimeFrameManifest",
    "MujocoFrameContextFactory",
    "MujocoPoseAlignedFrameContext",
    "MujocoPoseAlignedRenderHandle",
    "PoseAlignedRenderArtifact",
    "PoseAlignedRenderRequest",
    "RawPostGeometryMatchArtifact",
    "RawPostGeometryMatchRequest",
    "Stage1FrameArtifact",
    "Stage1Request",
    "ctrnetx_batch_pnp_seed",
    "ctrnetx_match_seed",
    "configure_ctrnetx_match_rng",
    "make_core_raw_post_geometry_matcher",
    "make_mujoco_pose_aligned_renderer",
    "raw_post_geometry_from_core_best_view",
    "run_ctrnetx_batch_episode",
    "run_ctrnetx_batch_shard",
    "run_ctrnetx_full_batch",
    "stage1_artifact_from_core_best_view",
]
