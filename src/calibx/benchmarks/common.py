"""Deterministic public frame manifests and benchmark sharding helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np


_PUBLIC_RELATIVE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_PUBLIC_METADATA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_public_relative_identifier(value: str) -> bool:
    """Accept only portable opaque IDs, never paths, URIs, or traversal."""
    if not isinstance(value, str) or not _PUBLIC_RELATIVE_IDENTIFIER.fullmatch(value):
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def is_public_metadata_key(value: str) -> bool:
    """Keep public metadata keys semantic and incapable of naming a path."""
    return isinstance(value, str) and bool(_PUBLIC_METADATA_KEY.fullmatch(value))


def _is_public_frame_id(value: str) -> bool:
    return is_public_relative_identifier(value)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class FrameManifest:
    """An ordered, public-safe frame selection for one protocol run.

    ``frame_ids`` are dataset-relative opaque identifiers.  The dataset adapter
    owns resolving them into local files, which keeps host paths and raw data
    outside source control.
    """

    protocol: str
    dataset_id: str
    frame_ids: tuple[str, ...]
    selection_seed: int | None = None
    selection_method: str = "explicit"
    version: int = 1

    def validate(self) -> None:
        if not self.protocol:
            raise ValueError("protocol must not be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if self.version != 1:
            raise ValueError(f"unsupported manifest version: {self.version}")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique")
        invalid = [frame_id for frame_id in self.frame_ids if not _is_public_frame_id(frame_id)]
        if invalid:
            raise ValueError(f"frame_ids must be public relative identifiers: {invalid[:3]}")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> "FrameManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(
            protocol=str(payload["protocol"]),
            dataset_id=str(payload["dataset_id"]),
            frame_ids=tuple(str(value) for value in payload["frame_ids"]),
            selection_seed=(
                None
                if payload.get("selection_seed") is None
                else int(payload["selection_seed"])
            ),
            selection_method=str(payload.get("selection_method", "explicit")),
            version=int(payload.get("version", 1)),
        )
        manifest.validate()
        return manifest

    @classmethod
    def deterministic_sample(
        cls,
        *,
        protocol: str,
        dataset_id: str,
        available_frame_ids: list[str],
        count: int,
        seed: int,
    ) -> "FrameManifest":
        """Mirror the historical ``default_rng(seed).choice`` selection rule."""
        if count < 1:
            raise ValueError("count must be at least one")
        if count > len(available_frame_ids):
            raise ValueError("count cannot exceed the number of available frames")
        positions = np.random.default_rng(seed).choice(
            len(available_frame_ids),
            size=count,
            replace=False,
        )
        return cls(
            protocol=protocol,
            dataset_id=dataset_id,
            frame_ids=tuple(available_frame_ids[int(index)] for index in sorted(positions)),
            selection_seed=seed,
            selection_method="numpy_default_rng_choice_sorted_positions",
        )

    def shard(
        self,
        *,
        shard_index: int,
        shard_count: int,
        strategy: Literal["contiguous", "round_robin"] = "contiguous",
    ) -> "FrameManifest":
        """Return a deterministic shard while retaining the parent selection."""
        if shard_count < 1:
            raise ValueError("shard_count must be at least one")
        if not 0 <= shard_index < shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        if strategy == "contiguous":
            start = len(self.frame_ids) * shard_index // shard_count
            stop = len(self.frame_ids) * (shard_index + 1) // shard_count
            frame_ids = self.frame_ids[start:stop]
        elif strategy == "round_robin":
            frame_ids = self.frame_ids[shard_index::shard_count]
        else:
            raise ValueError(f"unknown shard strategy: {strategy}")
        return replace(
            self,
            frame_ids=frame_ids,
            selection_method=(
                f"{self.selection_method};{strategy}_shard="
                f"{shard_index}_of_{shard_count}"
            ),
        )


def assert_paired_manifests(
    first: FrameManifest,
    second: FrameManifest,
) -> None:
    """Reject ablations that silently compare different requested frames."""
    first.validate()
    second.validate()
    if first.dataset_id != second.dataset_id or first.frame_ids != second.frame_ids:
        raise ValueError("paired ablations require exactly the same dataset and frames")
