#!/usr/bin/env python3
"""Plan or execute the historical CTRNet-X Panda independent-frame protocol.

This entry deliberately uses the dedicated typed single-frame boundary rather
than ``calibx.runner`` or the CTRNet-X closed-loop batch runner.  R0 is a
six-view/batch-six search and each R1--R3 round is one pose-aligned render with
the old unconditional-replace / fail-and-stop semantics.

``--execute`` requires a caller-provided hook factory because this public
repository intentionally does not guess a private CTRNet-X dataset layout.
The factory is imported only for execution and must expose a callable with the
following keyword-only contract::

    def build_hooks(*, config, manifest, frames) -> CtrnetxSingleHooks: ...

It resolves local data, models and output locations outside the portable
runtime manifest.  The default mode validates and prints the plan only.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from calibx.benchmarks.ctrnetx.protocol import (
    CTRNETX_SINGLE_TOTAL_FRAMES,
    CTRNETX_SPLITS,
    CtrnetxProtocol,
    assert_full_ctrnetx_single_scope,
)
from calibx.benchmarks.ctrnetx.single_executor import (
    CTRNETX_SINGLE_EXECUTION_SEMANTIC_ID,
    CtrnetxSingleExecutionConfig,
    CtrnetxSingleHooks,
    CtrnetxSingleRuntimeFrame,
    CtrnetxSingleRuntimeManifest,
    run_ctrnetx_single_formal_scope,
    run_ctrnetx_single_frames,
)
from calibx.benchmarks.dream_real.legacy import LEGACY_REFINEMENT_SEMANTIC_ID
from calibx.configuration import load_config, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--runtime-frame-manifest",
        type=Path,
        required=True,
        help="Portable CTRNet-X frame-ID/metadata manifest; never a data-root list.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Configured complete split to run; omit for all three full splits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first selected frames; requires --allow-partial.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly mark a selected subset as a smoke test, never a paper result.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Import --hook-factory and invoke typed local runtime hooks.",
    )
    parser.add_argument(
        "--hook-factory",
        default=None,
        metavar="MODULE:CALLABLE",
        help="Local adapter factory used only with --execute.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Optional destination for compact, path-free execution audit JSON.",
    )
    return parser.parse_args()


def _selected_splits(config: dict[str, Any], requested: list[str] | None) -> tuple[str, ...]:
    configured = tuple(str(split) for split in config["evaluation"].get("datasets", ()))
    selected = configured if requested is None else tuple(requested)
    if not selected:
        raise ValueError("at least one CTRNet-X split must be selected")
    unknown = [split for split in selected if split not in CTRNETX_SPLITS]
    if unknown:
        raise ValueError(f"unknown CTRNet-X split names: {unknown}")
    disabled = [split for split in selected if split not in configured]
    if disabled:
        raise ValueError(f"CTRNet-X split names are not enabled by config: {disabled}")
    if len(set(selected)) != len(selected):
        raise ValueError("CTRNet-X split names must not repeat")
    return selected


def _execution_config(config: dict[str, Any]) -> CtrnetxSingleExecutionConfig:
    protocol = config.get("protocol", {})
    if protocol.get("id") != "ctrnetx_single_legacy":
        raise ValueError("config does not declare ctrnetx_single_legacy")
    if protocol.get("semantic_id") != LEGACY_REFINEMENT_SEMANTIC_ID:
        raise ValueError("config does not declare the historical legacy semantic ID")
    if int(protocol.get("requested_frames", -1)) != CTRNETX_SINGLE_TOTAL_FRAMES:
        raise ValueError("CTRNet-X single config must declare 17,306 requested frames")
    evaluation = config["evaluation"]
    CtrnetxProtocol(
        mode="single",
        stage1_views=int(evaluation["views"]),
        stage1_match_batch_size=int(evaluation["match_batch_size"]),
        refinement_iterations=int(evaluation["refinement_iterations"]),
        mask_input=bool(evaluation["mask_input"]),
    ).validate()
    split_counts = {
        split: int(config["datasets"][split].get("sample_count", -1))
        for split in CTRNETX_SPLITS
    }
    assert_full_ctrnetx_single_scope(split_counts)
    runtime = CtrnetxSingleExecutionConfig(
        stage0_views=int(evaluation["views"]),
        stage0_match_batch_size=int(evaluation["match_batch_size"]),
        refinement_iterations=int(evaluation["refinement_iterations"]),
        mask_input=bool(evaluation["mask_input"]),
        mask_prompt=str(evaluation["mask_prompt"]),
        match_seed=int(evaluation["match_seed"]),
        visual_geom_group=int(config.get("robot", {}).get("visual_geom_group", 2)),
    )
    runtime.validate()
    return runtime


def _load_hook_factory(specification: str):
    if ":" not in specification:
        raise ValueError("--hook-factory must use MODULE:CALLABLE")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("--hook-factory must use MODULE:CALLABLE")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError("--hook-factory target must be callable")
    return factory


def _serialize(payload: dict[str, object], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(text, end="")
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.limit is not None and not args.allow_partial:
        raise ValueError("--limit requires --allow-partial so it cannot be mislabelled")
    if args.execute != (args.hook_factory is not None):
        raise ValueError("--execute and --hook-factory must be supplied together")

    config, config_path = load_config(args.config)
    validate_config(config)
    runtime_config = _execution_config(config)
    manifest = CtrnetxSingleRuntimeManifest.read(args.runtime_frame_manifest)
    splits = _selected_splits(config, args.split)
    if not args.allow_partial:
        manifest.validate_full_scope()
    frames: tuple[CtrnetxSingleRuntimeFrame, ...] = manifest.select(
        splits=splits,
        limit=args.limit,
    )
    payload: dict[str, object] = {
        "config": str(config_path),
        "runtime_frame_manifest_sha256": manifest.sha256(),
        "execution_semantic_id": CTRNETX_SINGLE_EXECUTION_SEMANTIC_ID,
        "legacy_semantic_id": LEGACY_REFINEMENT_SEMANTIC_ID,
        "execute": args.execute,
        "paper_scope_validated": not args.allow_partial,
        "selected_splits": list(splits),
        "selected_frames": len(frames),
        "stage0": {
            "views": runtime_config.stage0_views,
            "match_batch_size": runtime_config.stage0_match_batch_size,
            "match_seed": runtime_config.match_seed,
        },
        "refinement": {
            "iterations": runtime_config.refinement_iterations,
            "renders_per_iteration": 1,
            "candidate_policy": "unconditional_replace_on_pnp_success",
            "pnp_failure_policy": "frame_failed_then_later_iterations_skipped",
            "sam_rerun": False,
        },
    }
    if not args.execute:
        payload["note"] = (
            "Plan only. --execute needs a local typed hook factory; no private "
            "CTRNet-X dataset path is inferred by this repository."
        )
        _serialize(payload, args.audit_output)
        return

    factory = _load_hook_factory(args.hook_factory)
    hooks = factory(config=config, manifest=manifest, frames=frames)
    if not isinstance(hooks, CtrnetxSingleHooks):
        raise TypeError("hook factory must return CtrnetxSingleHooks")
    if args.allow_partial:
        executions = run_ctrnetx_single_frames(
            frames,
            config=runtime_config,
            hooks=hooks,
        )
    else:
        executions = run_ctrnetx_single_formal_scope(
            manifest,
            config=runtime_config,
            hooks=hooks,
            splits=splits,
        )
    payload["frames"] = [execution.audit_record() for execution in executions]
    _serialize(payload, args.audit_output)


if __name__ == "__main__":
    main()
