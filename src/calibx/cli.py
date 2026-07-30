"""Command-line interface for Calib-X experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .configuration import PROJECT_ROOT, load_config, project_path


def _flag(enabled: bool, name: str) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _execute(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _execute_many(commands: list[list[str]], dry_run: bool) -> None:
    for command in commands:
        print(shlex.join(command))
    if dry_run:
        return
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [
            executor.submit(
                subprocess.run,
                command,
                check=True,
                cwd=PROJECT_ROOT,
            )
            for command in commands
        ]
        for future in futures:
            future.result()


def _prerender_commands(config: dict, key: str) -> list[list[str]]:
    dataset = config["datasets"][key]
    render = config["prerender"]
    base_command = [
        sys.executable,
        "-m",
        "calibx.prerender",
        "--dataset-dir",
        str(project_path(dataset["path"])),
        "--mujoco-xml",
        str(project_path(config["paths"]["mujoco_xml"])),
        "--output-dir",
        str(project_path(config["paths"]["prerender_root"]) / dataset["name"]),
        "--views",
        str(config["evaluation"]["views"]),
        "--width",
        str(render["width"]),
        "--height",
        str(render["height"]),
        "--distance-scale",
        str(render["distance_scale"]),
        "--min-distance",
        str(render["min_distance"]),
        "--elevation",
        str(render["elevation"]),
        "--azimuth-offset",
        str(render["azimuth_offset"]),
    ]
    workers = int(render.get("workers", 1))
    if workers < 1:
        raise ValueError("prerender.workers must be >= 1")
    return [
        base_command
        + [
            "--num-shards",
            str(workers),
            "--shard-index",
            str(shard_index),
        ]
        for shard_index in range(workers)
    ]


def _evaluation_command(config: dict, key: str) -> list[str]:
    dataset = config["datasets"][key]
    evaluation = config["evaluation"]
    output_name = f"{dataset['name']}_sample{evaluation['sample_count']}"
    command = [
        sys.executable,
        "-m",
        "calibx.runner",
        "--dataset-dir",
        str(project_path(dataset["path"])),
        "--mujoco-xml",
        str(project_path(config["paths"]["mujoco_xml"])),
        "--prerender-dir",
        str(project_path(config["paths"]["prerender_root"]) / dataset["name"]),
        "--output-dir",
        str(project_path(config["paths"]["output_root"]) / output_name),
        "--sam3-checkpoint",
        str(project_path(config["paths"]["sam3_checkpoint"])),
        "--sample-count",
        str(evaluation["sample_count"]),
        "--sample-seed",
        str(evaluation["sample_seed"]),
        "--views",
        str(evaluation["views"]),
        "--match-batch-size",
        str(evaluation["match_batch_size"]),
        "--device",
        str(evaluation["device"]),
        "--mask-prompt",
        str(evaluation["mask_prompt"]),
        _flag(evaluation["mask_input"], "mask-input"),
        _flag(evaluation["resume"], "resume"),
        _flag(
            evaluation["save_visualizations"],
            "save-visualizations",
        ),
        _flag(evaluation["save_all_matches"], "save-all-matches"),
    ]
    return command


def _run(args: argparse.Namespace) -> int:
    config, config_path = load_config(args.config)
    paper_protocol = config.get("protocol", {}).get("id")
    if paper_protocol in {
        "dream_real_full_legacy",
        "ctrnetx_single_legacy",
        "ctrnetx_closed_loop_batch",
        "sam_foreground_ablation_table4",
        "matcher_ablation_candidate",
    }:
        raise ValueError(
            f"{paper_protocol} is a paper protocol plan, not a calibx.runner config; "
            "use the matching scripts/reproduce entry point"
        )
    datasets = args.datasets or config["evaluation"]["datasets"]
    print(f"Using config: {config_path}")
    for key in datasets:
        if key not in config["datasets"]:
            raise ValueError(f"Unknown dataset key: {key}")
        if args.prerender:
            _execute_many(
                _prerender_commands(config, key),
                args.dry_run,
            )
        else:
            _execute(
                _evaluation_command(config, key),
                args.dry_run,
            )
    return 0


def _summarize(args: argparse.Namespace) -> int:
    from .metrics import summarize

    run_dir = args.run_dir.resolve()
    records = [
        json.loads(path.read_text())
        for path in sorted(run_dir.glob("*/frame_summary.json"))
    ]
    if not records:
        raise RuntimeError(f"No frame summaries found under {run_dir}")
    summary = summarize(records, args.auc_threshold, args.auc_delta)
    print(json.dumps(summary, indent=2))
    return 0


def _paper_summarize(args: argparse.Namespace) -> int:
    from .reporting.records import (
        load_top_level_frame_records,
        summarize_refinement_records,
    )

    records = load_top_level_frame_records(args.run_dirs)
    summary = summarize_refinement_records(
        records,
        requested_frames=args.requested_frames,
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a configured evaluation.")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--datasets", nargs="*", default=None)
    run.add_argument("--prerender", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=_run)

    aggregate = subparsers.add_parser(
        "summarize",
        help="Recompute metrics from saved frame records.",
    )
    aggregate.add_argument("run_dir", type=Path)
    aggregate.add_argument("--auc-threshold", type=float, default=0.1)
    aggregate.add_argument("--auc-delta", type=float, default=1e-4)
    aggregate.set_defaults(handler=_summarize)

    paper_aggregate = subparsers.add_parser(
        "paper-summarize",
        help="Aggregate d7/Table 4 refinement records with all-frame AUC.",
    )
    paper_aggregate.add_argument("run_dirs", type=Path, nargs="+")
    paper_aggregate.add_argument("--requested-frames", type=int, default=None)
    paper_aggregate.set_defaults(handler=_paper_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
