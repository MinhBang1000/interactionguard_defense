#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKUP_DIR="${BACKUP_DIR:-backups}"
LABEL="${1:-current_results}"
PYTHON_BIN="${PYTHON:-python}"

exec "$PYTHON_BIN" scripts/backup_current_results.py "$LABEL" --backup-dir "$BACKUP_DIR"
