# File-Based Command Relay (Debugger)

A lightweight client/server system for running commands inside a Docker container from the host,
using only a shared filesystem — no networking required.

## Overview

```text
Host (Claude Code)                      Docker Container
┌─────────────┐                         ┌─────────────────┐
│ client.sh   │  writes cmd file        │ server.sh       │
│   run "X"   │ ───────────────────►    │   detects cmd   │
│             │                         │   executes X    │
│  reads      │  writes result file     │   writes result │
│  result     │ ◄───────────────────    │                 │
└─────────────┘                         └─────────────────┘
          └──── shared filesystem (.relay/) ────┘
```

## Assumptions

- The ModelOpt repo is accessible from both host and container (e.g., bind-mounted)
- **HuggingFace models** are mounted at `/hf-local`
- The server auto-detects the repo root from the location of `server.sh`

## Quick Start

### 1. Start the server (inside Docker)

```bash
# The server auto-detects the repo root (two levels up from tools/debugger/)
bash /path/to/modelopt/tools/debugger/server.sh
```

The server automatically sets the working directory to the repo root. You can override with `--workdir`.

### 2. Connect from the host

```bash
bash tools/debugger/client.sh handshake
```

### 3. Run commands

```bash
# Run a simple command
bash tools/debugger/client.sh run "echo hello"

# Run a test script
bash tools/debugger/client.sh run "bash llm_ptq/scripts/huggingface_example.sh"

# Run with a long timeout (default is 600s)
bash tools/debugger/client.sh --timeout 1800 run "python my_long_test.py"

# Cancel a running command
bash tools/debugger/client.sh cancel

# Check status
bash tools/debugger/client.sh status
```

## Protocol

The relay uses a directory at `tools/debugger/.relay/` with this structure:

```text
.relay/
├── server.ready      # Written by server on startup
├── client.ready      # Written by client during handshake
├── handshake.done    # Written by server to confirm handshake
├── running           # Written by server while a command is executing (cmd_id:pid)
├── cancel            # Written by client to request cancellation of the running command
├── cmd/              # Client writes command .sh files here
│   └── <id>.sh       # Command to execute
└── result/           # Server writes results here
    ├── <id>.log      # stdout + stderr
    └── <id>.exit     # Exit code
```

### Handshake

1. Server starts, creates `.relay/server.ready`
2. Client writes `.relay/client.ready`
3. Server detects it, writes `.relay/handshake.done`
4. Both sides are now connected

### Command Execution

1. Client writes a command to `.relay/cmd/<id>.sh`
2. Server detects the file, reads the command content, and removes the `.sh` file
3. Server runs `bash -c <content>` in a new process group, writes `.relay/running`
4. Server writes `.relay/result/<id>.exit` and `.relay/result/<id>.log`, then removes `.relay/running`
5. Client reads results and cleans up

### Cancellation

1. Client writes the target `cmd_id` to `.relay/cancel`
2. Server verifies the `cmd_id` matches, then kills the command's process group
3. Server writes exit code 130 and removes `.relay/running` and `.relay/cancel`
4. Client-side timeout also triggers cancellation automatically

## Options

### Server

| Flag | Default | Description |
|------|---------|-------------|
| `--relay-dir` | `<script_dir>/.relay` | Relay directory path |
| `--workdir` | Auto-detected repo root | Working directory for commands |

### Client

| Flag | Default | Description |
|------|---------|-------------|
| `--relay-dir` | `<script_dir>/.relay` | Relay directory path |
| `--timeout` | `600` | Seconds to wait for command result |

## Notes

- The `.relay/` directory is in `.gitignore` — it is not checked in.
- Only one server should run at a time (startup clears the relay directory).
- Commands run sequentially in the order the server discovers them.
- A running command can be cancelled via `client.sh cancel`. Cancelled commands exit with code 130.
- Client-side timeouts automatically cancel the running command on the server.
