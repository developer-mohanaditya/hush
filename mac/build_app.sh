#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAC_DIR="$ROOT_DIR/mac"
APP_DIR="$ROOT_DIR/build/Hush.app"
CONTENTS="$APP_DIR/Contents"

if [[ -d "/Applications/Xcode.app/Contents/Developer" ]]; then
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
fi

cd "$MAC_DIR"
swift build -c release
BIN_PATH="$(swift build -c release --show-bin-path)/Sottovox"

rm -rf "$APP_DIR"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources/engine"
cp "$BIN_PATH" "$CONTENTS/MacOS/Hush"
cp "$MAC_DIR/Resources/Info.plist" "$CONTENTS/Info.plist"
cp "$MAC_DIR/Resources/AppIcon.icns" "$CONTENTS/Resources/AppIcon.icns"
cp "$ROOT_DIR/engine/pyproject.toml" "$CONTENTS/Resources/engine/pyproject.toml"
cp -R "$ROOT_DIR/engine/sottovox_engine" "$CONTENTS/Resources/engine/sottovox_engine"
find "$CONTENTS/Resources/engine" -name "__pycache__" -o -name "*.pyc" -o -name ".DS_Store" | xargs rm -rf
codesign --force --sign - "$APP_DIR" >/dev/null 2>&1 || true

echo "Built $APP_DIR"
