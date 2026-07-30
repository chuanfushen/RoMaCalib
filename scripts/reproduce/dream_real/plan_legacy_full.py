"""Validate archived DREAM-real legacy plans and print their old evaluator argv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibx.benchmarks.dream_real.legacy import (
    LEGACY_REFINEMENT_SEMANTIC_ID,
    request_from_paper_config,
)
from calibx.benchmarks.dream_real.protocol import (
    DreamRealFullProtocol,
    is_full_dream_real_scope,
)
from calibx.configuration import load_config, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Camera subset to plan. Supplying it always marks a non-full plan.",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help=(
            "Explicitly acknowledge a camera subset. Without this flag, only the "
            "four-camera 49,619-frame paper scope is accepted."
        ),
    )
    return parser.parse_args()


def _planned_datasets(config: dict, requested: list[str] | None) -> list[str]:
    protocol = config.get("protocol", {})
    if protocol.get("semantic_id") != LEGACY_REFINEMENT_SEMANTIC_ID:
        raise ValueError("config does not declare the historical legacy refinement semantic")
    configured = [str(name) for name in config["evaluation"].get("datasets", ())]
    if len(set(configured)) != len(configured):
        raise ValueError("evaluation.datasets must not contain duplicate DREAM-real cameras")
    selected = configured if requested is None else requested
    if not selected:
        raise ValueError("config must declare at least one evaluation dataset")
    if len(set(selected)) != len(selected):
        raise ValueError("--dataset must not select the same DREAM-real camera twice")
    unknown = [name for name in selected if name not in config["datasets"]]
    if unknown:
        raise ValueError(f"unknown requested DREAM-real datasets: {unknown}")
    outside = [name for name in selected if name not in configured]
    if outside:
        raise ValueError(f"requested datasets are not enabled by evaluation.datasets: {outside}")
    return selected


def _declared_frame_counts(config: dict, datasets: list[str]) -> dict[str, int]:
    evaluation = config["evaluation"]
    return {
        dataset: int(config["datasets"][dataset].get("sample_count", evaluation["sample_count"]))
        for dataset in datasets
    }


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    validate_config(config)
    datasets = _planned_datasets(config, args.dataset)
    evaluation = config["evaluation"]
    for dataset in datasets:
        DreamRealFullProtocol(
            camera=dataset,
            views=int(evaluation["views"]),
            refinement_iterations=int(evaluation["refinement_iterations"]),
            mask_input=bool(evaluation["mask_input"]),
            match_batch_size=int(evaluation["match_batch_size"]),
        ).validate()
    declared_counts = _declared_frame_counts(config, datasets)
    paper_full = args.dataset is None and is_full_dream_real_scope(declared_counts)
    if not paper_full and not args.subset:
        raise ValueError(
            "this plan may call itself paper-full only when it uses all four archived "
            "cameras and exactly 49,619 declared frames; pass --subset for a camera subset"
        )
    requests = [request_from_paper_config(config, dataset) for dataset in datasets]
    print(
        json.dumps(
            {
                "config": str(config_path),
                "semantic_id": LEGACY_REFINEMENT_SEMANTIC_ID,
                "scope": {
                    "kind": "paper_full" if paper_full else "explicit_subset",
                    "paper_full_eligible": paper_full,
                    "declared_camera_frame_counts": declared_counts,
                },
                "requested_frames": sum(request.sample_count for request in requests),
                "runs": [
                    {
                        "dataset": dataset,
                        "requested_frames": request.sample_count,
                        "legacy_evaluator_argv": request.to_legacy_argv(),
                    }
                    for dataset, request in zip(datasets, requests, strict=True)
                ],
                "note": (
                    "This is an inspectable compatibility plan, not a gate-v1 "
                    "or execution alias."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
