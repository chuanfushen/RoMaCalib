"""Compatibility wrapper for :mod:`calibx.prerender`."""

from calibx.prerender import *  # noqa: F401,F403
from calibx.prerender import main as _main

if __name__ == "__main__":
    _main()
