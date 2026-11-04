#!/usr/bin/env bash
# Build the pinned environment for every kernel path inside WSL2 Ubuntu 24.04.
# Safe to re-run: completed steps are skipped or repeated without harm.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Kept on the Linux filesystem: /mnt/c is slow and does not preserve Linux permissions.
VENV_DIR="${CUTILE_VENV:-$HOME/.venvs/cutile}"
TILEIR_SITE="$VENV_DIR/opt/triton_tileir"

TILEIR_TAG="v3.7.1-tileir-13.4.0"
TILEIR_WHEEL="triton_tileir-3.7.1+tileir13.4.0-cp312-cp312-linux_x86_64.whl"
TILEIR_SHA256="0e7828ae3d0489b7df1d91ff70652b40d9e045e439a7f4a863695350394dafa2"
TILEIR_URL="https://github.com/triton-lang/Triton-to-tile-IR/releases/download/$TILEIR_TAG/$TILEIR_WHEEL"

# df on / only reports the virtual disk's maximum size. Its backing file grows on C:,
# so the Windows drive is the real limit.
MIN_FREE_GB=15

fail() {
  echo "error: $*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found, so the GPU is not visible in WSL"

python_version="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$python_version" == "3.12" ]] || fail "the Tile IR wheel needs Python 3.12, found $python_version"

if [[ -d /mnt/c ]]; then
  free_gb="$(df --output=avail -BG /mnt/c | tail -1 | tr -dc '0-9')"
  ((free_gb >= MIN_FREE_GB)) || fail "only ${free_gb} GB free on C:, need ${MIN_FREE_GB} GB"
fi

# Triton compiles a small C launcher per kernel at runtime, so gcc and the Python headers
# are runtime dependencies here, not just build tools.
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential python3-dev python3-venv

python3 -m venv "$VENV_DIR"
# --no-cache-dir because the WSL disk file does not shrink on its own: cached wheels would
# keep costing space on C: after they stop being useful.
"$VENV_DIR/bin/python" -m pip install --no-cache-dir -r "$REPO_ROOT/requirements.txt"

if [[ ! -d "$TILEIR_SITE/triton" ]]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  curl -fL --retry 3 -o "$tmp_dir/$TILEIR_WHEEL" "$TILEIR_URL"
  echo "$TILEIR_SHA256  $tmp_dir/$TILEIR_WHEEL" | sha256sum --check --quiet
  # A separate --target leaves upstream triton intact, so PYTHONPATH selects between them.
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir --no-deps \
    --target "$TILEIR_SITE" "$tmp_dir/$TILEIR_WHEEL"
fi

"$VENV_DIR/bin/python" -m pip install --no-cache-dir --no-deps -e "$REPO_ROOT"

"$VENV_DIR/bin/python" -m pip freeze --exclude-editable >"$REPO_ROOT/requirements.lock"

echo
echo "Environment ready. Next:"
echo "  source $VENV_DIR/bin/activate"
echo "  python probes/run.py"
echo "  pytest"
echo "  python -m tilebench"
