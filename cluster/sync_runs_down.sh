#!/usr/bin/env bash

set -euo pipefail

# Sync only the runs/ folder from a remote t-zero checkout down to local.
# Usage:
#   SOURCE=user@run.node:/raid/you/t-zero bash cluster/sync_runs_down.sh
# Optional:
#   LOCAL_ROOT=/path/to/local/t-zero
#   --dry-run  (pass as argument to preview without transferring)

DRY_RUN=""
for arg in "$@"; do
	[[ "$arg" == "--dry-run" ]] && DRY_RUN="--dry-run"
done

if [[ -z "${SOURCE:-}" ]]; then
	echo "ERROR: SOURCE is not set. Example: SOURCE=user@host:/raid/you/t-zero" >&2
	exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
	echo "ERROR: rsync not found in PATH." >&2
	exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$ROOT_DIR}"

mkdir -p "$LOCAL_ROOT"

echo "Syncing runs/ from $SOURCE -> $LOCAL_ROOT${DRY_RUN:+ (dry run)}"
rsync -avhz --prune-empty-dirs ${DRY_RUN} \
	--include='runs/' \
	--include='runs/**' \
	--exclude='*' \
	"${SOURCE%/}/" \
	"$LOCAL_ROOT/"

echo "Sync complete."