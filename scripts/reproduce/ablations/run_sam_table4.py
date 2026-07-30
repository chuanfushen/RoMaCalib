#!/usr/bin/env python3
"""Plan or execute the paired Table 4 SAM foreground-mask ablation.

The default mode is intentionally asset-free: it checks the frozen scalar
protocol and prints four planned arms (two 900-frame splits x no-SAM/SAM).
``--validate-manifests`` additionally loads the two public frame manifests for
each requested split.  A real run requires ``--execute`` and an explicitly
supplied local typed backend factory; this script does not infer a private
dataset layout or redirect execution through the generic runner.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import tomllib
from typing import Any, Callable

from calibx.benchmarks.ablations.sam import SamAblationCondition, SamAblationPair
from calibx.benchmarks.ablations.sam_executor import (
    SamTable4Backend,
    condition_plan,
    execute_sam_table4_condition,
)
from calibx.benchmarks.common import FrameManifest
from calibx.benchmarks.gate_v1 import DETERMINISTIC_SEED
from calibx.configuration import PROJECT_ROOT


SPLITS = ("robot_in_view", "robot_in_and_out")
ARMS = ("without_sam", "with_sam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        default=None,
        help="Restrict to one split; omit to plan or validate both 900-frame splits.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        choices=ARMS,
        default=None,
        help="Restrict to one arm; omit to plan or validate no-SAM and SAM.",
    )
    parser.add_argument(
        "--validate-manifests",
        action="store_true",
        help="Load and pair-check public manifests before printing the plan.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run an explicit local typed backend after pair validation.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        metavar="MODULE:FACTORY",
        help=(
            "Local factory called as factory(condition) for --execute. It must "
            "return a SamTable4Backend and is never imported for plan/help."
        ),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Optional destination for portable per-arm control/evidence JSON.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_manifest(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_scalar_protocol(config: dict[str, Any], split: str) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    if protocol.get("id") != "sam_foreground_ablation_table4":
        raise ValueError("unexpected SAM ablation protocol id")
    if int(protocol.get("robot_in_view_requested_frames", -1)) != 900:
        raise ValueError("Table 4 robot_in_view arm must contain 900 requested frames")
    if int(protocol.get("robot_in_and_out_requested_frames", -1)) != 900:
        raise ValueError(
            "Table 4 robot_in_and_out arm must contain 900 requested frames"
        )
    if int(protocol.get("views", -1)) != 6:
        raise ValueError("Table 4 Stage 0 requires views=6")
    if int(protocol.get("refinement_iterations", -1)) != 3:
        raise ValueError("Table 4 requires exactly three refinement iterations")
    if int(protocol.get("stage1_match_batch_size", -1)) != 6:
        raise ValueError("Table 4 Stage 0 requires match_batch_size=6")
    if int(protocol.get("refinement_match_batch_size", -1)) != 1:
        raise ValueError("Table 4 R1--R3 require one-pair match_batch_size=1")
    if int(protocol.get("determinism_seed", -1)) != DETERMINISTIC_SEED:
        raise ValueError(
            "Table 4 requires the frozen gate-v1 determinism_seed="
            f"{DETERMINISTIC_SEED}"
        )
    if protocol.get("pairing") != "identical_frame_manifest":
        raise ValueError("Table 4 requires an identical frame manifest for both arms")
    if split not in config.get("manifests", {}):
        raise ValueError(f"config does not declare manifests for {split}")
    without = config.get("without_sam", {})
    with_sam = config.get("with_sam", {})
    if bool(without.get("mask_input", True)):
        raise ValueError("without_sam.mask_input must be false")
    if not bool(with_sam.get("mask_input", False)):
        raise ValueError("with_sam.mask_input must be true")
    prompt = str(without.get("mask_prompt", ""))
    if not prompt or prompt != str(with_sam.get("mask_prompt", "")):
        raise ValueError("Table 4 SAM arms must share one non-empty mask_prompt")
    return protocol


def _build_pair(config: dict[str, Any], split: str) -> SamAblationPair:
    protocol = _validate_scalar_protocol(config, split)
    manifest_paths = config["manifests"][split]
    without = config["without_sam"]
    with_sam = config["with_sam"]
    pair = SamAblationPair(
        without_sam=SamAblationCondition(
            split=split,
            manifest=FrameManifest.read(
                _resolve_manifest(manifest_paths["without_sam"])
            ),
            mask_input=bool(without["mask_input"]),
            mask_prompt=str(without["mask_prompt"]),
            views=int(protocol["views"]),
            refinement_iterations=int(protocol["refinement_iterations"]),
            stage1_match_batch_size=int(protocol["stage1_match_batch_size"]),
            refinement_match_batch_size=int(protocol["refinement_match_batch_size"]),
            determinism_seed=int(protocol["determinism_seed"]),
        ),
        with_sam=SamAblationCondition(
            split=split,
            manifest=FrameManifest.read(_resolve_manifest(manifest_paths["with_sam"])),
            mask_input=bool(with_sam["mask_input"]),
            mask_prompt=str(with_sam["mask_prompt"]),
            views=int(protocol["views"]),
            refinement_iterations=int(protocol["refinement_iterations"]),
            stage1_match_batch_size=int(protocol["stage1_match_batch_size"]),
            refinement_match_batch_size=int(protocol["refinement_match_batch_size"]),
            determinism_seed=int(protocol["determinism_seed"]),
        ),
    )
    pair.validate()
    return pair


def _scalar_plan(config: dict[str, Any], split: str, arm: str) -> dict[str, object]:
    protocol = _validate_scalar_protocol(config, split)
    condition = config[arm]
    manifest = config["manifests"][split][arm]
    return {
        "split": split,
        "arm": arm,
        "requested_frames": int(protocol[f"{split}_requested_frames"]),
        "manifest": str(manifest),
        "mask_input": bool(condition["mask_input"]),
        "mask_prompt": str(condition["mask_prompt"]),
        "determinism_seed": int(protocol["determinism_seed"]),
        "stage0": {
            "orbit_render_count": 6,
            "matcher_pair_count": 6,
            "match_batch_size": 6,
            "matcher_call_count": 1,
        },
        "refinement": {
            "requested_iterations": 3,
            "match_render_count": 1,
            "matcher_pair_count": 1,
            "match_batch_size": 1,
            "matcher_call_count": 1,
            "candidate_policy": "gate-v1",
        },
    }


def _load_backend_factory(
    spec: str,
) -> Callable[[SamAblationCondition], SamTable4Backend]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--backend must use MODULE:FACTORY")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError("--backend factory must be callable")
    return factory


def _selected(values: list[str] | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    selected = defaults if values is None else tuple(values)
    if len(set(selected)) != len(selected):
        raise ValueError("repeat --split/--arm values are not allowed")
    return selected


def main() -> None:
    args = parse_args()
    if args.execute and not args.backend:
        raise ValueError("--execute requires --backend MODULE:FACTORY")
    if args.backend and not args.execute:
        raise ValueError("--backend is meaningful only together with --execute")
    config = _load_config(args.config)
    splits = _selected(args.split, SPLITS)
    arms = _selected(args.arm, ARMS)
    require_manifests = args.validate_manifests or args.execute
    pair_records: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "config": str(args.config),
        "execute": args.execute,
        "manifest_validation": require_manifests,
        "pairs": pair_records,
        "runs": [],
    }

    backend_factory = None if not args.execute else _load_backend_factory(args.backend)
    for split in splits:
        if not require_manifests:
            payload["runs"].extend(_scalar_plan(config, split, arm) for arm in arms)
            continue
        pair = _build_pair(config, split)
        without_sha256 = pair.without_sam.manifest.sha256()
        with_sha256 = pair.with_sam.manifest.sha256()
        if without_sha256 != with_sha256:
            raise AssertionError("validated SAM pair has mismatched manifest SHA-256")
        pair_records.append(
            {
                "split": split,
                "without_sam_manifest_sha256": without_sha256,
                "with_sam_manifest_sha256": with_sha256,
                "shared_manifest_sha256": without_sha256,
            }
        )
        arm_conditions = {
            "without_sam": pair.without_sam,
            "with_sam": pair.with_sam,
        }
        for arm in arms:
            condition = arm_conditions[arm]
            if not args.execute:
                payload["runs"].append({"arm": arm, "plan": condition_plan(condition)})
                continue
            assert backend_factory is not None
            backend = backend_factory(condition)
            execution = execute_sam_table4_condition(condition, backend)
            audit_path = None
            if args.audit_dir is not None:
                audit_path = args.audit_dir / split / f"{arm}_audit.json"
                if audit_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite existing audit record: {audit_path}"
                    )
                execution.write_audit(audit_path)
            payload["runs"].append(
                {
                    "arm": arm,
                    "plan": condition_plan(condition),
                    "summary": execution.summary(),
                    "audit": None if audit_path is None else str(audit_path),
                }
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
