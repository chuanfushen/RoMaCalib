"""Compatibility imports for the former DREAM configuration module."""

from calibx.configuration import (
    DEFAULT_CONFIG,
    ENV_OVERRIDES,
    PROJECT_ROOT,
    load_config,
    project_path,
    validate_config,
)

__all__ = [
    "DEFAULT_CONFIG",
    "ENV_OVERRIDES",
    "PROJECT_ROOT",
    "load_config",
    "project_path",
    "validate_config",
]
