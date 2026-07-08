#!/usr/bin/env bash

set -euo pipefail

# Sync the t-zero project to a remote destination.
# Usage:
#   DESTINATION=user@run.node:~/t-zero bash cluster/sync_run_node.sh

if [[ -z "${DESTINATION:-}" ]]; then
	echo "ERROR: DESTINATION is not set. Example: DESTINATION=user@host:~/t-zero" >&2
	exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
	echo "ERROR: rsync not found in PATH." >&2
	exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Syncing $ROOT_DIR -> $DESTINATION"
rsync -avhz \
	--exclude='.git/' \
	--exclude='data/' \
	--exclude='__pycache__/' \
	--exclude='*.pyc' \
	--exclude='runs' \
	--exclude='logs' \
	--exclude='wandb' \
	"$ROOT_DIR/" \
	"$DESTINATION/"

echo "Sync complete."
