#!/usr/bin/env python3
"""Plan or execute the archived DREAM-real R0-artifact → R1--R3 protocol.

Without ``--execute`` this prints a locally resolved plan.  With ``--execute`` it
loads the existing Stage-1 result for every selected frame and invokes the
dedicated legacy executor; it never dispatches to the generic ``calibx run``
pipeline or reruns SAM/Stage 1.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from calibx.benchmarks.dream_real.legacy import (
    LEGACY_REFINEMENT_SEMANTIC_ID,
    request_from_paper_config,
)
from calibx.benchmarks.dream_real.legacy_executor import (
    CalibxLegacyIterationBackend,
    LegacyExecutionOptions,
    execute_legacy_batch,
)
from calibx.benchmarks.dream_real.protocol import (
    DreamRealFullProtocol,
    is_full_dream_real_scope,
)
from calibx.configuration import load_config, project_path, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Configured DREAM-real camera subset; this always marks a non-full run.",
    )
    parser.add_argument(
        "--frame-index",
        action="append",
        type=int,
        default=None,
        help="Restrict an execution to explicit numeric frame IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the first selected frames (useful for a remote smoke test).",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help=(
            "Explicitly acknowledge a camera subset or smoke run. Without this flag, "
            "only the four-camera 49,619-frame paper scope is accepted."
        ),
    )
    parser.add_argument(
        "--runtime-identity-file",
        type=Path,
        default=None,
        help=(
            "Machine-local text/JSON identity record for safe resume (for example, "
            "the committed code, weight hashes, and environment). Required when "
            "the local config sets resume=true."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the dedicated artifact-replay executor instead of printing a plan.",
    )
    return parser.parse_args()


def _selected_datasets(config: dict, requested: list[str] | None) -> tuple[str, ...]:
    protocol = config.get("protocol", {})
    if protocol.get("semantic_id") != LEGACY_REFINEMENT_SEMANTIC_ID:
        raise ValueError("config does not declare the historical legacy semantic ID")
    configured = tuple(str(value) for value in config["evaluation"].get("datasets", ()))
    if len(set(configured)) != len(configured):
        raise ValueError("evaluation.datasets must not contain duplicate DREAM-real cameras")
    selected = configured if requested is None else tuple(requested)
    if not selected:
        raise ValueError("at least one DREAM-real dataset must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("--dataset must not select the same DREAM-real camera twice")
    unknown = [name for name in selected if name not in config["datasets"]]
    if unknown:
        raise ValueError(f"unknown configured DREAM-real dataset keys: {unknown}")
    disabled = [name for name in selected if name not in configured]
    if disabled:
        raise ValueError(f"dataset keys are not enabled by evaluation.datasets: {disabled}")
    return selected


def _resolved_options(
    config: dict,
    dataset_key: str,
    *,
    frame_indices: tuple[int, ...],
    limit: int | None,
    runtime_identity_file: Path | None,
) -> LegacyExecutionOptions:
    request = request_from_paper_config(config, dataset_key)
    # ``load_config`` resolves dataset paths below data_root.  The remaining
    # host-neutral paths are project-relative in the public TOML template.
    request = replace(
        request,
        dataset_dir=project_path(request.dataset_dir),
        mujoco_xml=project_path(request.mujoco_xml),
        prerender_dir=project_path(request.prerender_dir),
        output_dir=project_path(request.output_dir),
        sam3_checkpoint=project_path(request.sam3_checkpoint),
        initial_results_dir=(
            None
            if request.initial_results_dir is None
            else project_path(request.initial_results_dir)
        ),
    )
    prerender = config["prerender"]
    options = LegacyExecutionOptions.from_request(
        request,
        width=(None if prerender.get("width") is None else int(prerender["width"])),
        height=(
            None if prerender.get("height") is None else int(prerender["height"])
        ),
        distance_scale=float(prerender.get("distance_scale", 2.8)),
        min_distance=float(prerender.get("min_distance", 1.2)),
        elevation=float(prerender.get("elevation", -20.0)),
        azimuth_offset=float(prerender.get("azimuth_offset", 0.0)),
        limit=limit,
        runtime_identity_file=(
            None
            if runtime_identity_file is None
            else project_path(runtime_identity_file)
        ),
    )
    effective_frame_indices = frame_indices or request.frame_indices
    if effective_frame_indices:
        options = replace(
            options,
            frame_indices=effective_frame_indices,
            sample_count=len(effective_frame_indices),
        )
    options.validate()
    return options


def _declared_frame_counts(config: dict, datasets: tuple[str, ...]) -> dict[str, int]:
    evaluation = config["evaluation"]
    return {
        dataset: int(config["datasets"][dataset].get("sample_count", evaluation["sample_count"]))
        for dataset in datasets
    }


def _effective_frame_indices(
    config: dict, cli_frame_indices: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the actual fixed-frame selection after CLI precedence."""
    if cli_frame_indices:
        return cli_frame_indices
    return tuple(int(index) for index in config["evaluation"].get("frame_indices", ()))


def _scope_payload(
    config: dict,
    datasets: tuple[str, ...],
    *,
    explicit_dataset_selection: bool,
    frame_indices: tuple[int, ...],
    limit: int | None,
) -> dict[str, object]:
    restricted = explicit_dataset_selection or bool(frame_indices) or limit is not None
    declared_counts = _declared_frame_counts(config, datasets)
    paper_full = not restricted and is_full_dream_real_scope(declared_counts)
    return {
        "kind": "paper_full" if paper_full else "explicit_subset_or_smoke",
        "paper_full_eligible": paper_full,
        "declared_camera_frame_counts": declared_counts,
        "restriction": (
            None
            if paper_full
            else {
                "explicit_dataset_selection": explicit_dataset_selection,
                "frame_indices": list(frame_indices),
                "limit": limit,
            }
        ),
    }


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    validate_config(config)
    datasets = _selected_datasets(config, args.dataset)
    frame_indices = () if args.frame_index is None else tuple(args.frame_index)
    effective_frame_indices = _effective_frame_indices(config, frame_indices)
    scope = _scope_payload(
        config,
        datasets,
        explicit_dataset_selection=args.dataset is not None,
        frame_indices=effective_frame_indices,
        limit=args.limit,
    )
    if not scope["paper_full_eligible"] and not args.subset:
        raise ValueError(
            "this command may call a run paper-full only when it uses all four "
            "archived cameras and exactly 49,619 declared frames; pass --subset "
            "for a camera subset or frame-limited smoke run"
        )
    payload: dict[str, object] = {
        "config": str(config_path),
        "execute": args.execute,
        "scope": scope,
        "runs": [],
    }
    for dataset_key in datasets:
        evaluation = config["evaluation"]
        DreamRealFullProtocol(
            camera=dataset_key,
            views=int(evaluation["views"]),
            refinement_iterations=int(evaluation["refinement_iterations"]),
            mask_input=bool(evaluation["mask_input"]),
            match_batch_size=int(evaluation["match_batch_size"]),
        ).validate()
        options = _resolved_options(
            config,
            dataset_key,
            frame_indices=frame_indices,
            limit=args.limit,
            runtime_identity_file=args.runtime_identity_file,
        )
        if not args.execute:
            payload["runs"].append({"dataset": dataset_key, "plan": options.to_plan()})
            continue
        execution = execute_legacy_batch(
            options,
            CalibxLegacyIterationBackend(options),
        )
        payload["runs"].append(
            {
                "dataset": dataset_key,
                "requested_frames": len(execution.frames),
                "output_dir": str(options.output_dir),
                "summary": execution.summary["summary"],
            }
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
