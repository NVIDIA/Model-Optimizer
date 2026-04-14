#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File-based command relay server.
# Run this inside the Docker container. It watches for command files from the
# client, executes them, and writes results back.
#
# Usage: bash server.sh [--relay-dir <path>] [--workdir <path>]

set -euo pipefail

RELAY_DIR=""
WORKDIR=""
POLL_INTERVAL=1

# Derive the modelopt repo root from the location of this script (tools/debugger/server.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKDIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    echo "Usage: $0 [--relay-dir <path>] [--workdir <path>]"
    echo ""
    echo "Options:"
    echo "  --relay-dir  Path to relay directory (default: <script_dir>/.relay)"
    echo "  --workdir    Working directory for commands (default: auto-detected repo root)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --relay-dir) RELAY_DIR="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Default relay dir is .relay next to this script
if [[ -z "$RELAY_DIR" ]]; then
    RELAY_DIR="$SCRIPT_DIR/.relay"
fi

# Default workdir is the repo root (two levels up from tools/debugger/)
if [[ -z "$WORKDIR" ]]; then
    WORKDIR="$DEFAULT_WORKDIR"
fi

CMD_DIR="$RELAY_DIR/cmd"
RESULT_DIR="$RELAY_DIR/result"

cleanup() {
    echo "[server] Shutting down..."
    # Kill any child processes in our process group
    pkill -P $$ 2>/dev/null || true
    rm -f "$RELAY_DIR/server.ready"
    rm -f "$RELAY_DIR/handshake.done"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Set environment
export PYTHONPATH="$WORKDIR"

# Check for an already-running server
if [[ -f "$RELAY_DIR/server.ready" ]]; then
    old_pid=$(cut -d: -f2 "$RELAY_DIR/server.ready")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "[server] ERROR: Another server (PID $old_pid) is already running."
        exit 1
    fi
fi

# Initialize relay directories
rm -rf "$RELAY_DIR"
mkdir -p "$CMD_DIR" "$RESULT_DIR"

# Install modelopt in editable mode (skip if already editable-installed from WORKDIR)
if python -c "
import modelopt, os
assert os.path.realpath(modelopt.__path__[0]).startswith(os.path.realpath('$WORKDIR'))
" 2>/dev/null; then
    echo "[server] modelopt already editable-installed from $WORKDIR, skipping pip install."
else
    echo "[server] Installing modelopt (pip install -e .[dev]) ..."
    (cd "$WORKDIR" && pip install -e ".[dev]") || {
        echo "[server] WARNING: pip install failed (exit=$?), continuing anyway."
    }
    echo "[server] Install done."
fi

# Signal that server is ready
echo "$(hostname):$$:$(date -Iseconds)" > "$RELAY_DIR/server.ready"
echo "[server] Ready. Relay dir: $RELAY_DIR"
echo "[server] Workdir: $WORKDIR"
echo "[server] Waiting for client handshake..."

# Wait for client handshake
while [[ ! -f "$RELAY_DIR/client.ready" ]]; do
    sleep "$POLL_INTERVAL"
done

CLIENT_INFO=$(cat "$RELAY_DIR/client.ready")
echo "[server] Client connected: $CLIENT_INFO"
echo "$(hostname):$$:$(date -Iseconds)" > "$RELAY_DIR/handshake.done"
echo "[server] Handshake complete. Listening for commands..."

# Main loop: watch for command files and re-handshake requests
shopt -s nullglob
while true; do
    # Detect re-handshake (client flushed and reconnected)
    if [[ -f "$RELAY_DIR/client.ready" && ! -f "$RELAY_DIR/handshake.done" ]]; then
        CLIENT_INFO=$(cat "$RELAY_DIR/client.ready")
        echo "[server] Client re-connected: $CLIENT_INFO"
        echo "$(hostname):$$:$(date -Iseconds)" > "$RELAY_DIR/handshake.done"
        echo "[server] Re-handshake complete."
    fi

    for cmd_file in "$CMD_DIR"/*.sh; do
        cmd_id="$(basename "$cmd_file" .sh)"
        echo "[server] Executing command $cmd_id..."

        # Execute the command, tee stdout+stderr to console and result file
        (cd "$WORKDIR" && bash "$cmd_file" 2>&1) | tee "$RESULT_DIR/$cmd_id.log" || true
        exit_code=${PIPESTATUS[0]}

        # Atomic write of exit code (signal to client that result is ready)
        echo "$exit_code" > "$RESULT_DIR/$cmd_id.exit.tmp"
        mv "$RESULT_DIR/$cmd_id.exit.tmp" "$RESULT_DIR/$cmd_id.exit"

        # Remove the command file to mark it as processed
        rm -f "$cmd_file"

        echo "[server] Command $cmd_id finished (exit=$exit_code)"
    done

    sleep "$POLL_INTERVAL"
done
