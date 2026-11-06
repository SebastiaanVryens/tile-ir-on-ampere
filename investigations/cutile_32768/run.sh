#!/usr/bin/env bash
# Run one script from this folder with the Triton Tile IR build available.
# Usage, from the activated venv: bash investigations/cutile_32768/run.sh factorial.py
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
prefix=$(python -c 'import sys; print(sys.prefix)')

PYTHONPATH="$prefix/opt/triton_tileir${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}" ENABLE_TILE=1 \
    exec python "$here/$1" "${@:2}"
