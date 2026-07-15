#!/usr/bin/env python3
"""Download configured DREAM datasets and SAM3 weights."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from romav2.dream_config import load_config, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", choices=("real", "dr", "photo"), default=None)
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_sam3(config: dict, force: bool) -> None:
    from modelscope import snapshot_download

    target = project_path(config["paths"]["sam3_dir"])
    checkpoint = project_path(config["paths"]["sam3_checkpoint"])
    if checkpoint.is_file() and not force:
        print(f"SAM3 already present: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        config["download"]["sam3_model_id"],
        revision=config["download"].get("sam3_revision", "master"),
        local_dir=str(target),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ModelScope download completed but SAM3 checkpoint is missing: {checkpoint}")


def download_dataset(key: str, dataset: dict, force: bool) -> None:
    import gdown

    target = project_path(dataset["path"])
    if target.exists() and any(target.rglob("*.rgb.jpg")) and not force:
        print(f"DREAM {key} already present: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    archive_dir = target.parent / ".downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(
        id=dataset["google_drive_id"],
        output=str(archive_dir) + "/",
        quiet=False,
    )
    if not downloaded:
        raise RuntimeError(f"Failed to download DREAM dataset {key}")
    archive = Path(downloaded)
    shutil.unpack_archive(str(archive), str(target))
    archive.unlink()
    print(f"Extracted DREAM {key} to {target}")


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    print(f"Using config: {config_path}")
    if not args.skip_sam3:
        download_sam3(config, args.force)
    selected = args.datasets
    if selected is None:
        selected = [key for key, value in config["datasets"].items() if value.get("enabled", False)]
    for key in selected:
        download_dataset(key, config["datasets"][key], args.force)


if __name__ == "__main__":
    main()
