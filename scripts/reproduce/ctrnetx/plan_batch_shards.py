"""Validate a CF CTRNet-X batch manifest and create its four shard assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from calibx.benchmarks.ctrnetx.episode_pnp import balanced_episode_shards
from calibx.benchmarks.ctrnetx.protocol import (
    CTRNETX_SINGLE_TOTAL_FRAMES,
    CtrnetxBatchManifest,
    CtrnetxProtocol,
)
from calibx.benchmarks.ctrnetx.batch_executor import (
    CtrnetxBatchExecutionConfig,
    CtrnetxRuntimeFrameManifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument(
        "--runtime-frame-manifest",
        type=Path,
        default=None,
        help=(
            "Optional portable per-frame manifest. When supplied, validate its "
            "episode order and counts against --episode-manifest."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Must match the archived config when supplied.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _validate_config(config: dict) -> tuple[int, CtrnetxBatchExecutionConfig]:
    protocol = config.get("protocol", {})
    if protocol.get("id") != "ctrnetx_closed_loop_batch":
        raise ValueError("unexpected CTRNet-X batch protocol id")
    if int(protocol.get("requested_frames", -1)) != CTRNETX_SINGLE_TOTAL_FRAMES:
        raise ValueError("CTRNet-X batch config must request 17,306 frames")
    if int(protocol.get("episodes", -1)) != 60:
        raise ValueError("CTRNet-X batch config must declare 60 episodes")
    if int(protocol.get("iterations", -1)) != 3:
        raise ValueError("CTRNet-X batch config must declare R3")
    execution_config = CtrnetxBatchExecutionConfig.from_paper_config(config)
    sharding = config.get("sharding", {})
    shard_count = int(sharding.get("shard_count", -1))
    if shard_count != 4:
        raise ValueError("the archived CTRNet-X batch protocol uses four shards")
    if sharding.get("strategy") != "largest_first_min_load":
        raise ValueError("unexpected CTRNet-X batch sharding strategy")
    return shard_count, execution_config


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    configured_shard_count, execution_config = _validate_config(config)
    shard_count = configured_shard_count if args.shard_count is None else args.shard_count
    if shard_count != configured_shard_count:
        raise ValueError("--shard-count must match the archived batch configuration")

    manifest = CtrnetxBatchManifest.read(args.episode_manifest)
    runtime_manifest = None
    if args.runtime_frame_manifest is not None:
        runtime_manifest = CtrnetxRuntimeFrameManifest.read(args.runtime_frame_manifest)
        runtime_manifest.validate_against_count_manifest(manifest)
    counts = {episode.episode_id: episode.frame_count for episode in manifest.episodes}
    shards = balanced_episode_shards(counts, shard_count)
    loads = [sum(counts[episode_id] for episode_id in shard) for shard in shards]
    expected_loads = config["sharding"].get("historical_frame_counts")
    if expected_loads is not None and sorted(loads, reverse=True) != sorted(
        (int(value) for value in expected_loads),
        reverse=True,
    ):
        raise ValueError(
            "episode manifest does not reproduce the archived four-shard frame counts"
        )
    output = {
        "config": str(args.config),
        "protocol": manifest.protocol,
        "episode_manifest_sha256": manifest.sha256(),
        "runtime_frame_manifest_sha256": (
            None if runtime_manifest is None else runtime_manifest.sha256()
        ),
        "requested_frames": sum(counts.values()),
        "episodes": len(manifest.episodes),
        "strategy": "largest_first_min_load",
        "effective_execution_config": execution_config.to_record(),
        "shards": [list(shard) for shard in shards],
        "loads": loads,
        "note": (
            "Planning only. Execute a shard through "
            "calibx.benchmarks.ctrnetx.batch_executor with typed runtime hooks."
        ),
    }
    serialized = json.dumps(output, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
