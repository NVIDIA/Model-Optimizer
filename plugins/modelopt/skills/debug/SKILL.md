---
name: debug
description: Run commands inside a remote Docker container via the file-based command relay (tools/debugger). Use when the user says "run in Docker", "run on GPU", "debug remotely", "run test in container", "check nvidia-smi", "run pytest in Docker", or needs to execute any command inside a Docker container that shares the repo filesystem. Requires the user to have started server.sh inside the container first.
---

# Remote Docker Debugger

Execute commands inside a Docker container from the host using the file-based command relay.

**Read `tools/debugger/CLAUDE.md` for full usage details** — it has the protocol and examples.

## Quick Reference

```bash
# Check connection
bash tools/debugger/client.sh status

# Connect to server (user must start server.sh in Docker first)
bash tools/debugger/client.sh handshake

# Run a command
bash tools/debugger/client.sh run "<command>"

# Long-running command (default timeout is 600s)
bash tools/debugger/client.sh --timeout 1800 run "<command>"

# Cancel the currently running command
bash tools/debugger/client.sh cancel

# Reconnect after server restart
bash tools/debugger/client.sh flush
bash tools/debugger/client.sh handshake
```
