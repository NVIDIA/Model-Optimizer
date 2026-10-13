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

echo "[server] Workdir: $WORKDIR"

cleanup() {
    echo "[server] Shutting down..."
    # Kill any running command (guard all reads with || true to prevent set -e
    # from aborting the trap and leaving stale marker files)
    running_pid=$(cut -d: -f2 "$RELAY_DIR/running" 2>/dev/null) || true
    if [[ -n "$running_pid" ]]; then
        kill -- -"$running_pid" 2>/dev/null || kill "$running_pid" 2>/dev/null || true
    fi
    # Kill any child processes in our process group
    pkill -P $$ 2>/dev/null || true
    rm -f "$RELAY_DIR/server.ready"
    rm -f "$RELAY_DIR/handshake.done"
    rm -f "$RELAY_DIR/running"
    rm -f "$RELAY_DIR/cancel"
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

# Ensure modelopt is editable-installed from WORKDIR
check_modelopt_local() {
    python -c "
import modelopt, os, sys
actual = os.path.realpath(modelopt.__path__[0])
expected = os.path.realpath('$WORKDIR')
if os.path.commonpath([actual, expected]) != expected:
    print(f'modelopt loaded from {actual}, expected under {expected}', file=sys.stderr)
    sys.exit(1)
" 2>&1
}

if check_modelopt_local >/dev/null 2>&1; then
    echo "[server] modelopt already editable-installed from $WORKDIR, skipping pip install."
else
    echo "[server] Installing modelopt (pip install -e .[dev]) ..."
    (cd "$WORKDIR" && pip install -e ".[dev]")
    if ! check_modelopt_local; then
        echo "[server] ERROR: modelopt is not running from the local folder ($WORKDIR)."
        echo "[server] Try: pip install -e '.[dev]' inside the container, then restart the server."
        exit 1
    fi
    echo "[server] Install done."
fi

# Signal that server is ready
echo "$(hostname):$$:$(date -Iseconds)" > "$RELAY_DIR/server.ready"
echo "[server] Ready. Relay dir: $RELAY_DIR"
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
        # Guard against command files deleted by the client between glob expansion
        # and processing (e.g., client timeout on a queued command)
        [[ -f "$cmd_file" ]] || continue

        cmd_id="$(basename "$cmd_file" .sh)"
        # Tolerate file disappearing between guard and read (TOCTOU with client timeout)
        cmd_content=$(cat "$cmd_file" 2>/dev/null) || continue
        # Remove command file immediately after reading to prevent re-execution
        # and to avoid TOCTOU with client timeout deleting it during execution
        rm -f "$cmd_file"
        echo "[server] Executing command $cmd_id: $cmd_content"

        # Clear any stale cancel file from a previous timed-out client
        rm -f "$RELAY_DIR/cancel"

        # Create log file and stream output to server console via tail
        : > "$RESULT_DIR/$cmd_id.log"
        tail -f "$RESULT_DIR/$cmd_id.log" &
        tail_pid=$!

        # Run in a new process group (setsid) for clean cancellation of entire process tree
        (cd "$WORKDIR" && exec setsid bash -c "$cmd_content") >> "$RESULT_DIR/$cmd_id.log" 2>&1 &
        cmd_pid=$!

        # Track the running command (ID and PID) — atomic write to prevent partial reads
        echo "$cmd_id:$cmd_pid" > "$RELAY_DIR/running.tmp"
        mv "$RELAY_DIR/running.tmp" "$RELAY_DIR/running"

        # Wait for completion or cancellation
        cancelled=""
        while kill -0 "$cmd_pid" 2>/dev/null; do
            if [[ -f "$RELAY_DIR/cancel" ]]; then
                # Verify cancel targets this command (reject empty or mismatched signals)
                cancel_target=$(cat "$RELAY_DIR/cancel" 2>/dev/null) || true
                if [[ "$cancel_target" != "$cmd_id" ]]; then
                    rm -f "$RELAY_DIR/cancel"
                    sleep "$POLL_INTERVAL"
                    continue
                fi
                echo "[server] Cancelling command $cmd_id (PID $cmd_pid)..."
                # Kill entire process group (negative PID) for full tree cleanup
                kill -- -"$cmd_pid" 2>/dev/null || kill "$cmd_pid" 2>/dev/null || true
                # Wait up to 5s for graceful exit, then escalate to SIGKILL
                for _ in $(seq 1 5); do
                    kill -0 "$cmd_pid" 2>/dev/null || break
                    sleep 1
                done
                if kill -0 "$cmd_pid" 2>/dev/null; then
                    echo "[server] Process $cmd_pid did not exit, sending SIGKILL..."
                    kill -9 -- -"$cmd_pid" 2>/dev/null || kill -9 "$cmd_pid" 2>/dev/null || true
                fi
                wait "$cmd_pid" 2>/dev/null || true
                cancelled="true"
                rm -f "$RELAY_DIR/cancel"
                echo "[cancelled]" >> "$RESULT_DIR/$cmd_id.log"
                echo "[server] Command $cmd_id cancelled."
                break
            fi
            sleep "$POLL_INTERVAL"
        done

        # Determine exit code (|| exit_code=$? prevents set -e from killing the
        # server when the command exits non-zero)
        if [[ -n "$cancelled" ]]; then
            exit_code=130
        else
            exit_code=0
            wait "$cmd_pid" 2>/dev/null || exit_code=$?
        fi

        # Stop console streaming
        kill "$tail_pid" 2>/dev/null || true
        wait "$tail_pid" 2>/dev/null || true

        # Write exit code BEFORE removing the running marker, so any observer
        # that sees running disappear can immediately find the result
        echo "$exit_code" > "$RESULT_DIR/$cmd_id.exit.tmp"
        mv "$RESULT_DIR/$cmd_id.exit.tmp" "$RESULT_DIR/$cmd_id.exit"

        # Now safe to remove markers
        rm -f "$RELAY_DIR/running"
        rm -f "$RELAY_DIR/cancel"

        echo "[server] Command $cmd_id finished (exit=$exit_code)"
    done

    sleep "$POLL_INTERVAL"
done
