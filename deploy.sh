#!/bin/bash
# Simple deployment script with rollback.
# Copies the backend to a deployment directory.
# Usage: ./deploy.sh /path/to/deploy

set -e

TARGET=$1
if [ -z "$TARGET" ]; then
  echo "Usage: $0 /path/to/deploy" >&2
  exit 1
fi

BACKUP_DIR="$TARGET.bak"
if [ -d "$TARGET" ]; then
  echo "Creating backup at $BACKUP_DIR"
  rm -rf "$BACKUP_DIR"
  mv "$TARGET" "$BACKUP_DIR"
fi

mkdir -p "$TARGET"
cp -r backend frontend phase1_vm_enhancements.py enhanced_vm_interface.py "$TARGET"/
cp -r config.yaml "$TARGET"/

echo "Deployed to $TARGET"
