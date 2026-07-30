"""Validate the three-split CF CTRNet-X single-frame legacy plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibx.benchmarks.ctrnetx.protocol import (
    CtrnetxProtocol,
    assert_full_ctrnetx_single_scope,
)
from calibx.benchmarks.dream_real.legacy import (
    LEGACY_REFINEMENT_SEMANTIC_ID,
    request_from_paper_config,
)
from calibx.configuration import load_config, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="One split to plan. Omit to plan all three archived full-run splits.",
    )
    return parser.parse_args()


def _planned_datasets(config: dict, requested: list[str] | None) -> list[str]:
    protocol = config.get("protocol", {})
    if protocol.get("semantic_id") != LEGACY_REFINEMENT_SEMANTIC_ID:
        raise ValueError("config does not declare the historical legacy refinement semantic")
    configured = [str(name) for name in config["evaluation"].get("datasets", ())]
    selected = configured if requested is None else requested
    if not selected:
        raise ValueError("config must declare at least one evaluation dataset")
    unknown = [name for name in selected if name not in config["datasets"]]
    if unknown:
        raise ValueError(f"unknown requested CTRNet-X splits: {unknown}")
    outside = [name for name in selected if name not in configured]
    if outside:
        raise ValueError(f"requested splits are not enabled by evaluation.datasets: {outside}")
    return selected


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    validate_config(config)
    datasets = _planned_datasets(config, args.dataset)
    evaluation = config["evaluation"]
    CtrnetxProtocol(
        mode="single",
        stage1_views=int(evaluation["views"]),
        stage1_match_batch_size=int(evaluation["match_batch_size"]),
        refinement_iterations=int(evaluation["refinement_iterations"]),
        mask_input=bool(evaluation["mask_input"]),
    ).validate()
    if args.dataset is None:
        assert_full_ctrnetx_single_scope(
            {
                dataset: int(config["datasets"][dataset].get("sample_count", evaluation["sample_count"]))
                for dataset in datasets
            }
        )
    requests = [request_from_paper_config(config, dataset) for dataset in datasets]
    print(
        json.dumps(
            {
                "config": str(config_path),
                "semantic_id": LEGACY_REFINEMENT_SEMANTIC_ID,
                "mode": "independent_single_frame",
                "requested_frames": sum(request.sample_count for request in requests),
                "runs": [
                    {
                        "split": dataset,
                        "requested_frames": request.sample_count,
                        "legacy_evaluator_argv": request.to_legacy_argv(),
                    }
                    for dataset, request in zip(datasets, requests, strict=True)
                ],
                "note": (
                    "Compatibility plan. Execute the separate typed single-frame "
                    "boundary with run_single_legacy.py and a portable runtime "
                    "frame manifest; do not dispatch this protocol to batch/gate code."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
