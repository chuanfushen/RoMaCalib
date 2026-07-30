"""Public-safe manifests for explicitly qualitative paper figures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath


def _is_public_identifier(value: str) -> bool:
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
class QualitativeItem:
    """One displayed result, without a raw image path or private asset ID."""

    result_id: str
    display_frame_id: str
    source_result_id: str | None = None
    pose_reused: bool = False
    label: str | None = None

    def validate(self) -> None:
        for name, value in {
            "result_id": self.result_id,
            "display_frame_id": self.display_frame_id,
            "source_result_id": self.source_result_id,
        }.items():
            if value is not None and not _is_public_identifier(value):
                raise ValueError(f"{name} must be a public relative identifier")
        if self.pose_reused and not self.source_result_id:
            raise ValueError("a reused pose must record source_result_id")


@dataclass(frozen=True)
class QualitativeFigureManifest:
    """A selection record that prevents qualitative figures being misreported.

    ``selection_rule`` is mandatory so a best-case display remains visibly
    distinct from a random or exhaustive benchmark.  Raw RGB paths, outputs,
    and checkpoints do not appear in this public schema.
    """

    figure_id: str
    source_protocol: str
    selection_rule: str
    items: tuple[QualitativeItem, ...]
    status: str = "historical"
    figure_kind: str = "qualitative"
    version: int = 1

    def validate(self) -> None:
        if self.figure_kind != "qualitative":
            raise ValueError("paper selection manifests are qualitative-only")
        if self.status not in {"historical", "reproducible"}:
            raise ValueError("status must be historical or reproducible")
        if self.version != 1:
            raise ValueError(f"unsupported selection manifest version: {self.version}")
        if not _is_public_identifier(self.figure_id):
            raise ValueError("figure_id must be a public relative identifier")
        if not _is_public_identifier(self.source_protocol):
            raise ValueError("source_protocol must be a public relative identifier")
        if not self.selection_rule.strip():
            raise ValueError("selection_rule must not be empty")
        if not self.items:
            raise ValueError("a qualitative figure needs at least one item")
        result_ids = [item.result_id for item in self.items]
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("qualitative result_ids must be unique")
        for item in self.items:
            item.validate()

    def to_dict(self) -> dict:
        self.validate()
        return {
            **asdict(self),
            "items": [asdict(item) for item in self.items],
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "QualitativeFigureManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(
            figure_id=str(payload["figure_id"]),
            source_protocol=str(payload["source_protocol"]),
            selection_rule=str(payload["selection_rule"]),
            items=tuple(QualitativeItem(**item) for item in payload["items"]),
            status=str(payload.get("status", "historical")),
            figure_kind=str(payload.get("figure_kind", "qualitative")),
            version=int(payload.get("version", 1)),
        )
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class DroidFigure3Candidate:
    """Normalized candidate input for the Figure 3 qualitative selector."""

    result_id: str
    episode_id: str
    display_frame_id: str
    ours_iou: float
    baseline_iou: float

    @property
    def iou_delta(self) -> float:
        return self.ours_iou - self.baseline_iou

    def validate(self) -> None:
        for name, value in {
            "result_id": self.result_id,
            "episode_id": self.episode_id,
            "display_frame_id": self.display_frame_id,
        }.items():
            if not _is_public_identifier(value):
                raise ValueError(f"{name} must be a public relative identifier")
        for name, value in {
            "ours_iou": self.ours_iou,
            "baseline_iou": self.baseline_iou,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1]")


def select_droid_figure3_candidates(
    candidates: list[DroidFigure3Candidate],
    *,
    minimum_ours_iou: float = 0.80,
    limit: int = 6,
) -> tuple[DroidFigure3Candidate, ...]:
    """Apply the exact Figure 3 qualitative-selection rule.

    The final figure is not a top-IoU sample: candidates first require Ours
    silhouette IoU of at least 0.80, are ranked by improvement over metadata
    RT, and are capped at one result per episode.
    """
    if not 0.0 <= minimum_ours_iou <= 1.0:
        raise ValueError("minimum_ours_iou must be in [0, 1]")
    if limit < 1:
        raise ValueError("limit must be at least one")
    for candidate in candidates:
        candidate.validate()
    selected: list[DroidFigure3Candidate] = []
    seen_episodes: set[str] = set()
    ranked = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.ours_iou >= minimum_ours_iou
        ),
        key=lambda candidate: (
            -candidate.iou_delta,
            candidate.episode_id,
            candidate.result_id,
        ),
    )
    for candidate in ranked:
        if candidate.episode_id in seen_episodes:
            continue
        selected.append(candidate)
        seen_episodes.add(candidate.episode_id)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(
            f"Figure 3 needs {limit} qualifying distinct episodes; found {len(selected)}"
        )
    return tuple(selected)


def make_droid_figure3_manifest(
    candidates: list[DroidFigure3Candidate],
) -> QualitativeFigureManifest:
    """Return the public manifest corresponding to the Figure 3 selection."""
    selected = select_droid_figure3_candidates(candidates)
    return QualitativeFigureManifest(
        figure_id="fig3-droid",
        source_protocol="droid_qualitative",
        selection_rule=(
            "ours silhouette IoU >= 0.80; rank by IoU gain over metadata RT; "
            "at most one result per episode"
        ),
        items=tuple(
            QualitativeItem(
                result_id=candidate.result_id,
                display_frame_id=candidate.display_frame_id,
                label=(
                    f"ours_iou={candidate.ours_iou:.4f}; "
                    f"iou_delta={candidate.iou_delta:.4f}"
                ),
            )
            for candidate in selected
        ),
    )


@dataclass(frozen=True)
class RefinementRound:
    """One coarse/refinement state used by a DROID Figure 5 candidate."""

    round_index: int
    status: str
    silhouette_iou: float | None

    def validate(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if self.status not in {"success", "failed", "skipped"}:
            raise ValueError("round status must be success, failed, or skipped")
        if self.status == "success":
            if self.silhouette_iou is None or not math.isfinite(self.silhouette_iou):
                raise ValueError("a successful refinement round needs a finite IoU")
            if not 0.0 <= self.silhouette_iou <= 1.0:
                raise ValueError("silhouette_iou must be in [0, 1]")


def select_best_successful_round(rounds: list[RefinementRound]) -> RefinementRound:
    """Choose the best successful refinement state for a preselected Fig. 5 case."""
    for round_item in rounds:
        round_item.validate()
    successful = [round_item for round_item in rounds if round_item.status == "success"]
    if not successful:
        raise ValueError("a Figure 5 case needs at least one successful refinement round")
    return min(
        successful,
        key=lambda round_item: (-float(round_item.silhouette_iou), round_item.round_index),
    )


@dataclass(frozen=True)
class CtrnetxQualitativeCase:
    """A public-safe summary needed to select CTRNet-X qualitative examples."""

    frame_id: str
    status: str
    add_error_mm: float | None
    prediction_visible: bool

    def validate(self) -> None:
        if not _is_public_identifier(self.frame_id):
            raise ValueError("frame_id must be a public relative identifier")
        if self.status not in {"success", "failed", "error"}:
            raise ValueError("status must be success, failed, or error")
        if self.status == "success":
            if self.add_error_mm is None or not math.isfinite(self.add_error_mm):
                raise ValueError("a successful case needs a finite ADD error")
            if self.add_error_mm < 0.0:
                raise ValueError("ADD error must be non-negative")


def select_ctrnetx_cases(
    cases: list[CtrnetxQualitativeCase],
    *,
    top_k: int = 4,
) -> tuple[tuple[CtrnetxQualitativeCase, ...], tuple[CtrnetxQualitativeCase, ...]]:
    """Return best cases and visible high-error cases without calling them failures."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    for case in cases:
        case.validate()
    successful = [case for case in cases if case.status == "success"]
    if len(successful) < top_k:
        raise ValueError("not enough successful cases for the requested best-case panel")
    best = tuple(sorted(successful, key=lambda case: (float(case.add_error_mm), case.frame_id))[:top_k])
    best_ids = {case.frame_id for case in best}
    high_error_visible = [
        case
        for case in successful
        if case.prediction_visible and case.frame_id not in best_ids
    ]
    if len(high_error_visible) < top_k:
        raise ValueError("not enough visible high-error cases for the requested panel")
    high_error = tuple(
        sorted(
            high_error_visible,
            key=lambda case: (-float(case.add_error_mm), case.frame_id),
        )[:top_k]
    )
    return best, high_error
