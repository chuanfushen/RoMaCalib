"""Small, public-safe provenance records for paper reproductions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def _is_public_relative_path(value: str | None) -> bool:
    if value is None:
        return True
    if not value:
        return False
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and not windows.root
        and ".." not in posix.parts
        and ".." not in windows.parts
    )


@dataclass(frozen=True)
class ExperimentProvenance:
    """Provenance that can be stored in a public repository.

    Paths are repository-relative references only; data roots, checkpoint
    caches, server names, and output directories intentionally do not belong
    here.  Historical records may document a dirty source snapshot, whereas a
    record marked reproducible must originate from a clean commit.
    """

    experiment_id: str
    protocol: str
    source_commit: str
    source_dirty: bool
    status: str
    requested_frames: int
    config: str
    manifest: str | None = None
    summary: str | None = None
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not _IDENTIFIER.fullmatch(self.experiment_id):
            raise ValueError("experiment_id must use lowercase letters, digits, _ or -")
        if not _IDENTIFIER.fullmatch(self.protocol):
            raise ValueError("protocol must use lowercase letters, digits, _ or -")
        if not _GIT_SHA.fullmatch(self.source_commit):
            raise ValueError("source_commit must be a 7-40 character lowercase SHA")
        if self.status not in {"historical", "reproducible"}:
            raise ValueError("status must be historical or reproducible")
        if self.status == "reproducible" and self.source_dirty:
            raise ValueError("a reproducible record cannot cite a dirty source tree")
        if self.requested_frames < 0:
            raise ValueError("requested_frames must be non-negative")
        for name, value in {
            "config": self.config,
            "manifest": self.manifest,
            "summary": self.summary,
        }.items():
            if not _is_public_relative_path(value):
                raise ValueError(f"{name} must be a repository-relative path")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
