#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="$(basename "$APP_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$APP_NAME}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$APP_DIR"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Preparing venv: $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "==> Creating venv..."
  uv venv "$VENV_DIR" --python "$PYTHON_BIN"
else
  echo "==> Reusing existing venv"
fi

echo "==> Installing/upgrading dependencies..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
uv pip install --upgrade pip setuptools wheel
uv pip install --upgrade -r requirements.txt

echo "==> Verifying GLiNER2.5 runtime imports..."
python - <<'PY'
import numpy
import torch
import transformers
from gliner2 import Extractor

# Accessing Extractor (rather than only importing the top-level package) is
# intentional: gliner2 resolves its GLiNER2.5 inference engine lazily.
assert Extractor is not None

print(
    "Runtime imports OK:",
    f"numpy={numpy.__version__}",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
)
PY

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "==> Created .env from .env.example"
else
  echo "==> Keeping existing .env"
fi

echo
echo "OK."
echo "Run:"
echo " source run.sh 0.0.0.0 8088"
