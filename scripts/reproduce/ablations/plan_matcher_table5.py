"""Validate the pending Table 5 matcher-ablation plan without publishing values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from calibx.benchmarks.ablations.matchers import (
    MatcherAblationCondition,
    TABLE5_MATCHERS,
    assert_matcher_comparable,
    matcher_ablation_provenance,
)
from calibx.benchmarks.common import FrameManifest
from calibx.configuration import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_manifest(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    protocol = config.get("protocol", {})
    if protocol.get("id") != "matcher_ablation_candidate":
        raise ValueError("unexpected Table 5 matcher-ablation protocol id")
    if int(protocol.get("requested_frames", -1)) != 300:
        raise ValueError("Table 5 plan must request 300 frames")
    if int(protocol.get("views", -1)) != 6 or int(
        protocol.get("refinement_iterations", -1)
    ) != 3:
        raise ValueError("Table 5 plan requires six views and R3")
    if protocol.get("gate_id") != "gate-v1":
        raise ValueError("Table 5 plan requires gate-v1")
    if protocol.get("mask_input") is not True:
        raise ValueError("Table 5 plan requires the audited SAM input mask")
    if int(protocol.get("shard_count", -1)) != 2 or int(
        protocol.get("frames_per_shard", -1)
    ) != 150:
        raise ValueError("Table 5 plan requires two 150-frame shards")
    if protocol["shard_count"] * protocol["frames_per_shard"] != 300:
        raise ValueError("Table 5 shard scope must cover the 300 requested frames")
    if protocol.get("pairing") != "identical_frame_manifest":
        raise ValueError("Table 5 plan requires an identical frame manifest")
    if protocol.get("status") != "pending_paper_value_verification":
        raise ValueError("Table 5 must remain pending value verification")
    matcher_config = config.get("matchers", {})
    baseline = str(matcher_config.get("baseline", ""))
    candidates = tuple(str(value) for value in matcher_config.get("candidates", ()))
    if baseline not in TABLE5_MATCHERS or any(
        matcher not in TABLE5_MATCHERS for matcher in candidates
    ):
        raise ValueError("Table 5 plan declares an unsupported matcher")
    requirement = str(matcher_config.get("dependency_requirement", ""))
    if not requirement:
        raise ValueError("Table 5 plan must declare external dependency requirements")
    manifest_value = config.get("manifests", {}).get("frame_manifest")
    manifest_path = args.manifest or (
        None if manifest_value is None else _resolve_manifest(str(manifest_value))
    )
    manifest_sha256 = None
    shard_manifest_sha256 = None
    if manifest_path is not None:
        manifest = FrameManifest.read(manifest_path)
        baseline_condition = MatcherAblationCondition(
            baseline,
            manifest,
            gate_id=str(protocol["gate_id"]),
            mask_input=bool(protocol["mask_input"]),
            shard_count=int(protocol["shard_count"]),
            frames_per_shard=int(protocol["frames_per_shard"]),
            dependency_requirement=requirement,
        )
        for matcher in candidates:
            assert_matcher_comparable(
                baseline_condition,
                MatcherAblationCondition(
                    matcher,
                    manifest,
                    gate_id=str(protocol["gate_id"]),
                    mask_input=bool(protocol["mask_input"]),
                    shard_count=int(protocol["shard_count"]),
                    frames_per_shard=int(protocol["frames_per_shard"]),
                    dependency_requirement=requirement,
                ),
            )
        provenance = matcher_ablation_provenance(manifest)
        manifest_sha256 = provenance.parent_manifest_sha256
        shard_manifest_sha256 = provenance.shard_manifest_sha256
    print(
        json.dumps(
            {
                "config": str(args.config),
                "status": protocol["status"],
                "baseline": baseline,
                "candidates": candidates,
                "manifest_sha256": manifest_sha256,
                "shard_manifest_sha256": shard_manifest_sha256,
                "note": (
                    "Planning contract only: all matchers must share the parent and "
                    "two shard manifest hashes; external matcher installation and clean "
                    "paper-value verification remain pending."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
