#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/build/Hush.app"
VENV_DIR="$ROOT_DIR/engine/.venv"

ensure_venv() {
    if [[ -x "$VENV_DIR/bin/sottovox-server" ]]; then
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 "$VENV_DIR"
        uv pip install --python "$VENV_DIR/bin/python" -e "$ROOT_DIR/engine[dev]"
    else
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install -e "$ROOT_DIR/engine[dev]"
    fi
}

if [[ "${1:-}" == "--cli" ]]; then
    shift
    ensure_venv
    exec "$VENV_DIR/bin/sottovox-cli" "$@"
fi

if [[ "${1:-}" == "--server" ]]; then
    shift
    ensure_venv
    exec "$VENV_DIR/bin/sottovox-server" --warm "$@"
fi

ensure_venv
if [[ ! -d "$APP_DIR" ]]; then
    "$ROOT_DIR/mac/build_app.sh"
fi

open "$APP_DIR"
