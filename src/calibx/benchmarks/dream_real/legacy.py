"""Explicit adapter for the historical full-paper refinement protocol.

This module deliberately preserves the *old* unconditional refinement CLI
contract for historical regression.  It must not be confused with d7's
gate-v1 trajectory: a successful old-protocol PnP candidate replaced the
parent pose directly, and a failed refinement terminated that frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LEGACY_REFINEMENT_SEMANTIC_ID = "legacy-unconditional-pnp-replace-v1"


def _boolean_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


@dataclass(frozen=True)
class LegacyEvaluatorRequest:
    """All arguments used by the historical full-run evaluator adapter."""

    dataset_dir: Path
    mujoco_xml: Path
    prerender_dir: Path
    output_dir: Path
    sam3_checkpoint: Path
    visual_geom_group: int
    sample_count: int
    sample_seed: int
    views: int
    match_batch_size: int
    refinement_iterations: int
    device: str
    mask_prompt: str
    mask_input: bool
    resume: bool
    save_visualizations: bool
    save_all_matches: bool
    visual_body_names: tuple[str, ...] = ()
    match_seed: int | None = None
    frame_indices: tuple[int, ...] = ()
    initial_results_dir: Path | None = None

    def validate(self) -> None:
        if self.visual_geom_group < 0:
            raise ValueError("visual_geom_group must be non-negative")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least one")
        if self.views < 1 or self.match_batch_size < 1:
            raise ValueError("views and match_batch_size must be at least one")
        if not 0 <= self.refinement_iterations <= 3:
            raise ValueError("refinement_iterations must be in [0, 3]")
        if not self.device:
            raise ValueError("device must not be empty")

    def to_legacy_argv(self) -> list[str]:
        """Return argv for the legacy evaluator, with no host path defaults."""
        self.validate()
        argv = [
            "--dataset-dir",
            str(self.dataset_dir),
            "--mujoco-xml",
            str(self.mujoco_xml),
            "--visual-geom-group",
            str(self.visual_geom_group),
            "--prerender-dir",
            str(self.prerender_dir),
            "--output-dir",
            str(self.output_dir),
            "--sam3-checkpoint",
            str(self.sam3_checkpoint),
            "--sample-count",
            str(self.sample_count),
            "--sample-seed",
            str(self.sample_seed),
            "--views",
            str(self.views),
            "--match-batch-size",
            str(self.match_batch_size),
            "--refinement-iterations",
            str(self.refinement_iterations),
            "--device",
            self.device,
            "--mask-prompt",
            self.mask_prompt,
            _boolean_flag("mask-input", self.mask_input),
            _boolean_flag("resume", self.resume),
            _boolean_flag("save-visualizations", self.save_visualizations),
            _boolean_flag("save-all-matches", self.save_all_matches),
        ]
        if self.visual_body_names:
            argv.extend(("--visual-body-names", *self.visual_body_names))
        if self.match_seed is not None:
            argv.extend(("--match-seed", str(self.match_seed)))
        if self.frame_indices:
            argv.extend(("--frame-indices", *(str(index) for index in self.frame_indices)))
        if self.initial_results_dir is not None:
            argv.extend(("--initial-results-dir", str(self.initial_results_dir)))
        return argv


def request_from_paper_config(config: dict, dataset_key: str) -> LegacyEvaluatorRequest:
    """Map a host-neutral paper TOML dictionary to the historical request.

    This is intentionally a pure mapping.  A caller resolves local paths via
    the ordinary configuration layer before constructing the request.
    """
    dataset = config["datasets"][dataset_key]
    paths = config["paths"]
    evaluation = config["evaluation"]
    robot = config.get("robot", {})
    sample_count = int(dataset.get("sample_count", evaluation["sample_count"]))
    output_name = dataset.get(
        "output_name",
        f"{dataset['name']}_sample{sample_count}",
    )
    initial_root = paths.get("initial_results_root")
    initial_name = dataset.get("initial_results_name", output_name)
    initial_results_dir = (
        None if initial_root is None else Path(initial_root) / initial_name
    )
    return LegacyEvaluatorRequest(
        dataset_dir=Path(dataset["path"]),
        mujoco_xml=Path(paths["mujoco_xml"]),
        prerender_dir=Path(paths["prerender_root"]) / dataset["name"],
        output_dir=Path(paths["output_root"]) / output_name,
        sam3_checkpoint=Path(paths["sam3_checkpoint"]),
        visual_geom_group=int(robot.get("visual_geom_group", 2)),
        sample_count=sample_count,
        sample_seed=int(dataset.get("sample_seed", evaluation["sample_seed"])),
        views=int(evaluation["views"]),
        match_batch_size=int(evaluation["match_batch_size"]),
        refinement_iterations=int(evaluation.get("refinement_iterations", 0)),
        device=str(evaluation["device"]),
        mask_prompt=str(evaluation["mask_prompt"]),
        mask_input=bool(evaluation["mask_input"]),
        resume=bool(evaluation["resume"]),
        save_visualizations=bool(evaluation["save_visualizations"]),
        save_all_matches=bool(evaluation["save_all_matches"]),
        visual_body_names=tuple(str(name) for name in robot.get("visual_body_names", ())),
        match_seed=(
            None
            if evaluation.get("match_seed") is None
            else int(evaluation["match_seed"])
        ),
        frame_indices=tuple(int(index) for index in evaluation.get("frame_indices", ())),
        initial_results_dir=initial_results_dir,
    )


__all__ = [
    "LEGACY_REFINEMENT_SEMANTIC_ID",
    "LegacyEvaluatorRequest",
    "request_from_paper_config",
]
