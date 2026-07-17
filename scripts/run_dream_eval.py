#!/usr/bin/env python3
"""Run configured DREAM evaluations with the native RoMaV2 matcher."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from romav2.dream_config import PROJECT_ROOT, load_config, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", choices=("real", "dr", "photo"), default=None)
    parser.add_argument("--prerender", action="store_true", help="Generate configured MuJoCo views instead of evaluating.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def flag(enabled: bool, name: str) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def execute(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def execute_many(commands: list[list[str]], dry_run: bool) -> None:
    for command in commands:
        print(shlex.join(command))
    if dry_run:
        return
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(subprocess.run, command, check=True, cwd=PROJECT_ROOT) for command in commands]
        for future in futures:
            future.result()


def run_prerender(config: dict, key: str, dry_run: bool = False) -> None:
    dataset = config["datasets"][key]
    render = config["prerender"]
    visual_geom_group = int(config.get("robot", {}).get("visual_geom_group", 2))
    source_visual_geom_group = config.get("robot", {}).get(
        "source_visual_geom_group"
    )
    visual_body_names = list(config.get("robot", {}).get("visual_body_names", []))
    base_command = [
        sys.executable,
        "-m", "romav2.benchmarks.utils.prerender",
        "--dataset-dir", str(project_path(dataset["path"])),
        "--mujoco-xml", str(project_path(config["paths"]["mujoco_xml"])),
        "--visual-geom-group", str(visual_geom_group),
        "--output-dir", str(project_path(config["paths"]["prerender_root"]) / dataset["name"]),
        "--views", str(config["evaluation"]["views"]),
        "--width", str(render["width"]),
        "--height", str(render["height"]),
        "--distance-scale", str(render["distance_scale"]),
        "--min-distance", str(render["min_distance"]),
        "--elevation", str(render["elevation"]),
        "--azimuth-offset", str(render["azimuth_offset"]),
        "--sample-count", str(config["evaluation"]["sample_count"]),
        "--sample-seed", str(config["evaluation"]["sample_seed"]),
    ]
    if visual_body_names:
        base_command.extend(["--visual-body-names", *map(str, visual_body_names)])
    if source_visual_geom_group is not None:
        base_command.extend(
            ["--source-visual-geom-group", str(int(source_visual_geom_group))]
        )
    if render.get("azimuths") is not None:
        base_command.extend(["--azimuths", *map(str, render["azimuths"])])
    workers = int(render.get("workers", 1))
    if workers < 1:
        raise ValueError("prerender.workers must be >= 1")
    commands = [
        base_command + ["--num-shards", str(workers), "--shard-index", str(shard_index)]
        for shard_index in range(workers)
    ]
    execute_many(commands, dry_run)


def run_evaluation(config: dict, key: str, dry_run: bool = False) -> None:
    dataset = config["datasets"][key]
    evaluation = config["evaluation"]
    refinement = config.get("refinement", {})
    camera = config.get("camera", {})
    visual_geom_group = int(config.get("robot", {}).get("visual_geom_group", 2))
    source_visual_geom_group = config.get("robot", {}).get(
        "source_visual_geom_group"
    )
    visual_body_names = list(config.get("robot", {}).get("visual_body_names", []))
    output_name = f"{dataset['name']}_sample{evaluation['sample_count']}"
    command = [
        sys.executable,
        "-m", "romav2.benchmarks.utils.evaluator",
        "--dataset-dir", str(project_path(dataset["path"])),
        "--mujoco-xml", str(project_path(config["paths"]["mujoco_xml"])),
        "--visual-geom-group", str(visual_geom_group),
        "--prerender-dir", str(project_path(config["paths"]["prerender_root"]) / dataset["name"]),
        "--output-dir", str(project_path(config["paths"]["output_root"]) / output_name),
        "--sam3-checkpoint", str(project_path(config["paths"]["sam3_checkpoint"])),
        "--sample-count", str(evaluation["sample_count"]),
        "--sample-seed", str(evaluation["sample_seed"]),
        "--views", str(evaluation["views"]),
        "--match-batch-size", str(evaluation["match_batch_size"]),
        "--device", str(evaluation["device"]),
        "--mask-prompt", str(evaluation["mask_prompt"]),
        flag(evaluation["mask_input"], "mask-input"),
        flag(evaluation["resume"], "resume"),
        flag(evaluation["save_visualizations"], "save-visualizations"),
        flag(evaluation["save_all_matches"], "save-all-matches"),
    ]
    if config.get("_config_path") is not None:
        command.extend(["--config-source", str(config["_config_path"])])
    if visual_body_names:
        command.extend(["--visual-body-names", *map(str, visual_body_names)])
    if source_visual_geom_group is not None:
        command.extend(
            ["--source-visual-geom-group", str(int(source_visual_geom_group))]
        )
    if refinement.get("enabled", False):
        process_width = int(
            refinement.get("process_width", config["prerender"]["width"])
        )
        process_height = int(
            refinement.get("process_height", config["prerender"]["height"])
        )
        command.extend(
            [
                "--refinement-enabled",
                "--refine-iterations",
                str(int(refinement.get("iterations", 0))),
                flag(
                    bool(refinement.get("save_artifacts", True)),
                    "save-refinement-artifacts",
                ),
                "--distortion-state",
                str(camera.get("distortion_state", "unknown")),
                "--width",
                str(process_width),
                "--height",
                str(process_height),
            ]
        )
        if camera.get("distortion_coefficients") is not None:
            command.extend(
                [
                    "--distortion-coefficients",
                    *map(str, camera["distortion_coefficients"]),
                ]
            )
        for key in ("fx", "fy", "cx", "cy"):
            if camera.get(key) is not None:
                command.extend([f"--{key}", str(camera[key])])
        if camera.get("settings") is not None:
            command.extend(
                ["--camera-settings", str(project_path(camera["settings"]))]
            )
    execute(command, dry_run)


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    config["_config_path"] = config_path
    datasets = args.datasets or config["evaluation"]["datasets"]
    print(f"Using config: {config_path}")
    for key in datasets:
        print(f"{'Pre-rendering' if args.prerender else 'Evaluating'} DREAM dataset: {key}")
        (run_prerender if args.prerender else run_evaluation)(config, key, args.dry_run)


if __name__ == "__main__":
    main()
