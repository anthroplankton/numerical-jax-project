#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60

uv run python -m jax_tpu_project.cli devices
