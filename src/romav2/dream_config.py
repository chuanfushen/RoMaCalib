"""Load the shared DREAM evaluation TOML configuration."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/dream_eval.toml"

ENV_OVERRIDES = {
    "DREAM_DATA_ROOT": ("paths", "data_root"),
    "DREAM_SAM3_DIR": ("paths", "sam3_dir"),
    "DREAM_SAM3_CHECKPOINT": ("paths", "sam3_checkpoint"),
    "DREAM_MUJOCO_XML": ("paths", "mujoco_xml"),
    "DREAM_PRERENDER_ROOT": ("paths", "prerender_root"),
    "DREAM_OUTPUT_ROOT": ("paths", "output_root"),
    "DREAM_REAL_DATASET": ("datasets", "real", "path"),
    "DREAM_DR_DATASET": ("datasets", "dr", "path"),
    "DREAM_PHOTO_DATASET": ("datasets", "photo", "path"),
}


def load_config(path: str | Path | None = None) -> tuple[dict, Path]:
    config_path = Path(path or os.environ.get("DREAM_EVAL_CONFIG", DEFAULT_CONFIG)).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    for environment, keys in ENV_OVERRIDES.items():
        if value := os.environ.get(environment):
            target = config
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
    data_root = Path(config["paths"]["data_root"])
    for dataset in config["datasets"].values():
        dataset_path = Path(dataset["path"])
        if not dataset_path.is_absolute():
            dataset["path"] = str(data_root / dataset_path)
    return config, config_path


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
