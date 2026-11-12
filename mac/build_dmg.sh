#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/build/Hush.app"
STAGE_DIR="$ROOT_DIR/build/dmg_stage"
DMG_PATH="$ROOT_DIR/build/Hush.dmg"

"$ROOT_DIR/mac/build_app.sh"

rm -rf "$STAGE_DIR" "$DMG_PATH"
mkdir -p "$STAGE_DIR"
cp -R "$APP_DIR" "$STAGE_DIR/Hush.app"
ln -s /Applications "$STAGE_DIR/Applications"

hdiutil create \
    -volname "Hush" \
    -srcfolder "$STAGE_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH" >/dev/null

rm -rf "$STAGE_DIR"
echo "Built $DMG_PATH ($(du -h "$DMG_PATH" | cut -f1))"
