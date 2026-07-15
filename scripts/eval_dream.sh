#!/usr/bin/env bash
set -euo pipefail

uv run --extra eval python scripts/run_dream_eval.py "$@"
