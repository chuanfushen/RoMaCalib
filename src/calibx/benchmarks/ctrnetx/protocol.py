"""Panda Manipulation / CTRNet-X protocol definitions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping

from ..common import is_public_relative_identifier

CTRNETX_SPLITS = (
    "robot_in_view_full_body",
    "robot_in_view_partial",
    "robot_in_and_out",
)
CTRNETX_SINGLE_FRAME_COUNTS = {
    "robot_in_view_full_body": 1_864,
    "robot_in_view_partial": 6_264,
    "robot_in_and_out": 9_178,
}
CTRNETX_SINGLE_TOTAL_FRAMES = sum(CTRNETX_SINGLE_FRAME_COUNTS.values())
CTRNETX_BATCH_EPISODES = 60


def _is_public_episode_id(value: str) -> bool:
    return is_public_relative_identifier(value)


@dataclass(frozen=True)
class CtrnetxEpisodeScope:
    """Public episode ID and requested-frame count for batch planning."""

    episode_id: str
    frame_count: int

    def validate(self) -> None:
        if not _is_public_episode_id(self.episode_id):
            raise ValueError("episode_id must be a public relative identifier")
        if self.frame_count < 1:
            raise ValueError("episode frame_count must be positive")


@dataclass(frozen=True)
class CtrnetxBatchManifest:
    """Minimal public input needed to reproduce closed-loop shard assignment."""

    episodes: tuple[CtrnetxEpisodeScope, ...]
    protocol: str = "ctrnetx_closed_loop_batch"
    version: int = 1

    def validate(self) -> None:
        if self.protocol != "ctrnetx_closed_loop_batch":
            raise ValueError("unexpected CTRNet-X batch manifest protocol")
        if self.version != 1:
            raise ValueError(f"unsupported CTRNet-X batch manifest version: {self.version}")
        if len(self.episodes) != CTRNETX_BATCH_EPISODES:
            raise ValueError(
                f"CTRNet-X batch requires {CTRNETX_BATCH_EPISODES} episode records"
            )
        for episode in self.episodes:
            episode.validate()
        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("CTRNet-X batch manifest episode_id values must be unique")
        requested_frames = sum(episode.frame_count for episode in self.episodes)
        if requested_frames != CTRNETX_SINGLE_TOTAL_FRAMES:
            raise ValueError(
                "CTRNet-X batch manifest must total "
                f"{CTRNETX_SINGLE_TOTAL_FRAMES} requested frames"
            )

    def to_dict(self) -> dict:
        self.validate()
        return {
            "protocol": self.protocol,
            "version": self.version,
            "episodes": [
                {"episode_id": episode.episode_id, "frame_count": episode.frame_count}
                for episode in self.episodes
            ],
        }

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def read(cls, path: Path) -> "CtrnetxBatchManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(
            protocol=str(payload.get("protocol", "ctrnetx_closed_loop_batch")),
            version=int(payload.get("version", 1)),
            episodes=tuple(
                CtrnetxEpisodeScope(
                    episode_id=str(item["episode_id"]),
                    frame_count=int(item["frame_count"]),
                )
                for item in payload["episodes"]
            ),
        )
        manifest.validate()
        return manifest


def assert_full_ctrnetx_single_scope(split_frame_counts: Mapping[str, int]) -> None:
    """Validate the archived three-split CF single-frame full-run scope."""
    normalized = {str(split): int(count) for split, count in split_frame_counts.items()}
    if normalized != CTRNETX_SINGLE_FRAME_COUNTS:
        raise ValueError(
            "CTRNet-X single-frame full scope must use the archived split "
            f"counts: {CTRNETX_SINGLE_FRAME_COUNTS}"
        )


@dataclass(frozen=True)
class CtrnetxProtocol:
    """Frozen CF settings for independent-frame or closed-loop replay."""

    mode: Literal["single", "batch"]
    stage1_views: int = 6
    stage1_match_batch_size: int = 6
    replay_views: int | None = None
    refinement_iterations: int = 3
    mask_input: bool = True

    def validate(self) -> None:
        if self.mode not in {"single", "batch"}:
            raise ValueError(f"unknown CTRNet-X mode: {self.mode}")
        if self.stage1_views != 6:
            raise ValueError("CTRNet-X paper protocols are frozen at six stage-1 views")
        if self.stage1_match_batch_size != 6:
            raise ValueError(
                "CTRNet-X formal stage-1 uses match_batch_size=6"
            )
        if self.mode == "batch" and self.replay_views != 1:
            raise ValueError(
                "CTRNet-X batch requires one pose-aligned replay render per iteration"
            )
        if self.mode == "single" and self.replay_views is not None:
            raise ValueError("CTRNet-X single-frame mode does not use replay_views")
        if self.refinement_iterations != 3:
            raise ValueError("CTRNet-X paper protocols are frozen at R3")
        if not self.mask_input:
            raise ValueError("CTRNet-X paper protocols require SAM input masking")


__all__ = [
    "CTRNETX_BATCH_EPISODES",
    "CtrnetxBatchManifest",
    "CtrnetxEpisodeScope",
    "CTRNETX_SINGLE_FRAME_COUNTS",
    "CTRNETX_SINGLE_TOTAL_FRAMES",
    "CTRNETX_SPLITS",
    "CtrnetxProtocol",
    "assert_full_ctrnetx_single_scope",
]
