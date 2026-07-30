"""Validate one Table 4 SAM-ablation pair against its frozen public config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from calibx.benchmarks.ablations.sam import SamAblationCondition, SamAblationPair
from calibx.benchmarks.common import FrameManifest
from calibx.benchmarks.gate_v1 import DETERMINISTIC_SEED
from calibx.configuration import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("robot_in_view", "robot_in_and_out"),
        required=True,
    )
    parser.add_argument("--without-sam", type=Path, default=None)
    parser.add_argument("--with-sam", type=Path, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_manifest(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_config(config: dict, split: str) -> tuple[int, int, int, int, bool, bool]:
    protocol = config.get("protocol", {})
    if protocol.get("id") != "sam_foreground_ablation_table4":
        raise ValueError("unexpected SAM ablation protocol id")
    if int(protocol.get("robot_in_view_requested_frames", -1)) != 900:
        raise ValueError("Table 4 robot_in_view arm must contain 900 requested frames")
    if int(protocol.get("robot_in_and_out_requested_frames", -1)) != 900:
        raise ValueError(
            "Table 4 robot_in_and_out arm must contain 900 requested frames"
        )
    views = int(protocol.get("views", -1))
    refinement_iterations = int(protocol.get("refinement_iterations", -1))
    if protocol.get("pairing") != "identical_frame_manifest":
        raise ValueError("Table 4 requires an identical frame manifest for both arms")
    arms = (config.get("without_sam", {}), config.get("with_sam", {}))
    without_mask = bool(arms[0].get("mask_input", True))
    with_mask = bool(arms[1].get("mask_input", False))
    if str(arms[0].get("mask_prompt", "")) != str(arms[1].get("mask_prompt", "")):
        raise ValueError("Table 4 SAM arms must use the identical mask_prompt")
    stage1_match_batch_size = int(protocol.get("stage1_match_batch_size", -1))
    refinement_match_batch_size = int(
        protocol.get("refinement_match_batch_size", -1)
    )
    determinism_seed = int(protocol.get("determinism_seed", -1))
    if determinism_seed != DETERMINISTIC_SEED:
        raise ValueError(
            "Table 4 requires the frozen gate-v1 determinism_seed="
            f"{DETERMINISTIC_SEED}"
        )
    if not (split in config.get("manifests", {})):
        raise ValueError(f"config does not declare manifest locations for {split}")
    return (
        views,
        refinement_iterations,
        stage1_match_batch_size,
        refinement_match_batch_size,
        without_mask,
        with_mask,
    )


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    (
        views,
        refinement_iterations,
        stage1_match_batch_size,
        refinement_match_batch_size,
        without_mask,
        with_mask,
    ) = _validate_config(config, args.split)
    determinism_seed = int(config["protocol"]["determinism_seed"])
    manifest_paths = config["manifests"][args.split]
    without_path = args.without_sam or _resolve_manifest(manifest_paths["without_sam"])
    with_path = args.with_sam or _resolve_manifest(manifest_paths["with_sam"])
    pair = SamAblationPair(
        without_sam=SamAblationCondition(
            split=args.split,
            manifest=FrameManifest.read(without_path),
            mask_input=without_mask,
            mask_prompt=str(config["without_sam"]["mask_prompt"]),
            views=views,
            refinement_iterations=refinement_iterations,
            stage1_match_batch_size=stage1_match_batch_size,
            refinement_match_batch_size=refinement_match_batch_size,
            determinism_seed=determinism_seed,
        ),
        with_sam=SamAblationCondition(
            split=args.split,
            manifest=FrameManifest.read(with_path),
            mask_input=with_mask,
            mask_prompt=str(config["with_sam"]["mask_prompt"]),
            views=views,
            refinement_iterations=refinement_iterations,
            stage1_match_batch_size=stage1_match_batch_size,
            refinement_match_batch_size=refinement_match_batch_size,
            determinism_seed=determinism_seed,
        ),
    )
    pair.validate()
    without_sha256 = pair.without_sam.manifest.sha256()
    with_sha256 = pair.with_sam.manifest.sha256()
    if without_sha256 != with_sha256:
        raise AssertionError("validated SAM pair has mismatched manifest SHA-256")
    print(
        json.dumps(
            {
                "config": str(args.config),
                "split": args.split,
                "without_sam_manifest_sha256": without_sha256,
                "with_sam_manifest_sha256": with_sha256,
                "shared_manifest_sha256": without_sha256,
                "requested_frames_per_arm": len(pair.with_sam.manifest.frame_ids),
                "determinism_seed": determinism_seed,
                "note": (
                    "Pair validation only; use run_sam_table4.py with an "
                    "explicit local typed backend to execute."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
