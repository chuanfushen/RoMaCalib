"""Compatibility wrapper for :mod:`calibx.rendering`."""

from calibx.rendering import *  # noqa: F401,F403
from calibx.rendering import main as _main

if __name__ == "__main__":
    _main()
