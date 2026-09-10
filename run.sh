#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="$(basename "$APP_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$APP_NAME}"

cd "$APP_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${1:-${HOST:-0.0.0.0}}"
PORT="${2:-${PORT:-8088}}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export HOST="$HOST"
export PORT="$PORT"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec uvicorn app:app \
  --host "$HOST" \
  --port "$PORT" \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --log-level "${LOG_LEVEL:-info}"
