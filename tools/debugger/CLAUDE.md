# Remote Command Relay

This directory contains a file-based command relay for executing commands inside a remote Docker
container from the host machine (where Claude Code runs).

## How to Use (for Claude Code)

### Setup (one-time per session)

The user must start the server inside Docker first:

```bash
# Inside Docker container (auto-detects repo root from script location):
bash /path/to/modelopt/tools/debugger/server.sh
```

Then Claude Code performs the handshake:

```bash
bash tools/debugger/client.sh handshake
```

### Running Commands

```bash
# Run any command in the Docker container (workdir = auto-detected repo root):
bash tools/debugger/client.sh run "<command>"

# For long-running tasks, increase timeout:
bash tools/debugger/client.sh --timeout 1800 run "<command>"
```

### Key Paths Inside Docker

| Path | Description |
|------|-------------|
| Repo root (auto-detected) | ModelOpt source, used as workdir |
| `/hf-local` | HuggingFace model cache |

### Examples

```bash
# Run PTQ test
bash tools/debugger/client.sh run "bash llm_ptq/scripts/huggingface_example.sh"

# Run pytest
bash tools/debugger/client.sh run "python -m pytest tests/gpu -k test_quantize"

# Check GPU
bash tools/debugger/client.sh run "nvidia-smi"

# Use HF models from local cache
bash tools/debugger/client.sh run "python script.py --model /hf-local/Qwen/Qwen3-8B"
```

### Important Notes

- The server must be started by the user manually inside Docker before the handshake.
- Default command timeout is 600 seconds (10 minutes). Use `--timeout` for longer tasks.
- Commands execute sequentially — one at a time.
- All commands run with the auto-detected repo root as the working directory.
- The `.relay/` directory is ephemeral and git-ignored.
