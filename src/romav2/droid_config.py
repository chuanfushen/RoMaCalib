"""Load the DROID session-level calibration experiment configuration."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/droid_eval.toml"

ENV_OVERRIDES = {
    "DROID_DATA_ROOT": ("paths", "data_root"),
    "DROID_MUJOCO_XML": ("paths", "mujoco_xml"),
    "DROID_SAM3_CHECKPOINT": ("paths", "sam3_checkpoint"),
    "DROID_OUTPUT_ROOT": ("paths", "output_root"),
    "DROID_CALIBALL_PYTHON": ("paths", "caliball_python"),
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> tuple[dict, Path]:
    config_path = Path(path or os.environ.get("DROID_EVAL_CONFIG", DEFAULT_CONFIG)).expanduser()
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
    return config, config_path
