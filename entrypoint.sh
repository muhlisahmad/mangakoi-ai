#!/usr/bin/env bash
set -e

# 1. Take ownership of the network volume so appuser can write to it
# This only works if we start as root, which we will configure in Docker
echo "Setting permissions for /runpod-volume..."
chown -R appuser:appuser /runpod-volume

# 2. Execute the main command as the non-root user
# 'gosu' or 'runuser' is used to switch the user permanently
exec runuser -u appuser -- python3 handler.py
