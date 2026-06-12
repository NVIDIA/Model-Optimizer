# modelopt-mcp

MCP server exposing the ModelOpt launcher (`tools/launcher/`) as typed tools for codex / Claude Code agents.

Anchor design: [OMNIML-5123](https://jirasw.nvidia.com/browse/OMNIML-5123).

## What this is

A thin MCP wrapper around `tools/launcher/core.py`. Agents that want to submit ModelOpt jobs — PTQ, QAT, training, evaluation — call typed tools here instead of shelling out to `uv run launch.py --yaml ...` and parsing prose output.

Two executors, one tool surface:

* **Docker** (local GPU) — `submit_job(yaml_path, hf_local=...)`. The launcher runs the job in a container on the local machine.
* **Slurm** (remote cluster via SSH) — `submit_job(yaml_path, cluster_host=..., cluster_user=..., identity=...)`. The launcher tunnels in and submits via sbatch.

Mode is determined by which args you pass, not by which tool you call. One tool, two backends.

## Tool surface

| Tool | Description |
|---|---|
| `list_examples` | Enumerate bundled launcher YAMLs under `tools/launcher/examples/` with model + description metadata extracted from each YAML. Discovery primitive — call this first when you don't know which YAML to launch. |
| `verify_setup(executor, ...)` | Fail-fast probe for the named executor. Docker: `docker info` (daemon up) + GPU passthrough check. Slurm: `ssh -o BatchMode=yes -o ConnectTimeout=5` to the cluster login node. Returns structured failure on auth / network / daemon issues — no exception. |
| `submit_job(yaml_path, hf_local? \| cluster_host?, ...)` | Submit a launcher YAML. Mode resolved from mutually-exclusive args. Returns `experiment_id` (Slurm) or PID (Docker) immediately; the actual job runs detached. Auto-runs `verify_setup` first by default (skippable). |
| `job_status(experiment_id)` | Filesystem-based status from nemo_run's experiment dir (`_DONE`, `status_*.out`). Returns `done` / `failed` / `running` plus per-task statuses. No in-memory registry; survives MCP server restarts. |
| `job_logs(experiment_id, task?, tail?)` | Read `log_<task>.out` from the experiment dir. Per-task filtering + optional tail to truncate. |

## Install

Two paths, both **from source via uv**. No PyPI wheel — OMNIML-5123 deliberately picks `uvx`-from-git over publication overhead.

### End-user install (recommended)

```bash
# Claude Code
claude mcp add modelopt -- uvx --from \
  "git+https://github.com/NVIDIA/Model-Optimizer.git#subdirectory=tools/mcp" \
  modelopt-mcp

# Codex
codex mcp add modelopt -- uvx --from \
  "git+https://github.com/NVIDIA/Model-Optimizer.git#subdirectory=tools/mcp" \
  modelopt-mcp
```

`uvx` clones the whole repo to its cache, installs `tools/mcp/` as the entry point, and resolves the sibling `modelopt-launcher` dep via `[tool.uv.sources]` (path → `../launcher`) inside the cloned tree.

### Dev install (local checkout)

```bash
uv pip install -e tools/launcher    # sibling dep first
uv pip install -e tools/mcp         # then this package
modelopt-mcp                         # stdio server entry on PATH
```

Both packages share the launcher's `core.py` orchestrator. The dev path relies on `[tool.uv.sources]` to point `modelopt-launcher` at `../launcher`.

### Why no plain `pip install` today

`modelopt-mcp` and `modelopt-launcher` are not on PyPI. Plain `pip` doesn't read `[tool.uv.sources]`, so even from a local checkout, `pip install -e tools/mcp` fails to resolve the bare `modelopt-launcher` name. Stick with `uv` / `uvx` while we're git-only.

To enable `pip install` later, two options:

| Path | Tradeoff |
|---|---|
| **Publish to PyPI** — versioned wheels for both packages | Clean `pip install`, but requires release machinery + version cadence |
| **PEP-440 direct URL** — `"modelopt-launcher @ git+...#subdirectory=tools/launcher"` | Works with pip + uv, but double-clones the repo on install |

Out of scope for Phase 1.

## Example workflow

Agent picking a bundled example and running it on a remote cluster:

```python
# 1. Discover available YAMLs
examples = mcp__modelopt__list_examples()
# {"ok": True, "count": 47, "examples": [{"path": "launcher/examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml", "model": "Qwen/Qwen3-8B", ...}, ...]}

# 2. Pre-flight check before submission
mcp__modelopt__verify_setup(
    executor="slurm",
    cluster_host="cw-dfw-cs-001-login-01.nvidia.com",
    cluster_user="alice",
)
# {"ok": True, "ssh_ok": True, "whoami": "alice", "remote_hostname": "cw-dfw-cs-001-login-01"}

# 3. Submit
result = mcp__modelopt__submit_job(
    yaml_path="launcher/examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml",
    cluster_host="cw-dfw-cs-001-login-01.nvidia.com",
    cluster_user="alice",
    identity="/home/alice/.ssh/id_ed25519",
    skip_verify=True,  # we just probed
)
# {"ok": True, "experiment_id": "cicd_1781240000", "slurm_job_id": "12345", ...}

# 4. Poll until done
while True:
    status = mcp__modelopt__job_status(experiment_id="cicd_1781240000")
    if status["status"] in ("done", "failed"):
        break

# 5. Fetch logs
logs = mcp__modelopt__job_logs(
    experiment_id="cicd_1781240000",
    task="task_0",
    tail=200,
)
```

For local Docker execution, drop `cluster_host`/`cluster_user`/`identity` and pass `hf_local=<path>` to `submit_job` instead.

## Required env vars

| Var | When | Notes |
|---|---|---|
| `NEMORUN_HOME` | submit + status + logs | Where the launcher writes experiment artifacts. Defaults to cwd if unset. `job_status` / `job_logs` search `$NEMORUN_HOME/experiments/<id>/`. |
| `MODELOPT_MCP_LOG` | (optional) server | Log level. Defaults to `INFO`. Logs go to stderr — stdout is the MCP wire. |
| `MODELOPT_MCP_SKIP_GPU_CHECK` | (optional) `verify_setup(executor='docker')` | Set to skip the `docker run --gpus all nvidia-smi` step. Useful for CI hosts without GPUs. |
| `MODELOPT_LAUNCHER_EXAMPLES_DIR` | (optional) `list_examples` | Override the examples directory location. Defaults to `../launcher/examples/` relative to this package. |

## Design principles

Three constants drive the surface here:

1. **Single `submit_job` with mode by args.** Not separate `submit_docker` / `submit_slurm` tools. The LLM tool catalog stays compact; the mutual-exclusion is a runtime check that returns structured failure when both or neither mode is specified.
2. **Filesystem is the source of truth.** Status + logs both read from nemo_run's experiment dir. No in-memory registry — survives MCP server restarts. The bridge module never holds per-job state across calls.
3. **`verify_setup` is auto-called by `submit_job` by default.** The probe takes ~1 second; the cost of a misconfigured submission is 30+ seconds of cluster timeout or container-pull. Always-on verification pays back immediately. Callers can pass `skip_verify=True` when they just probed.

## Internal companion (NVIDIA only)

For NVIDIA-internal users running on the in-house clusters, there's a companion server [`nmm-sandbox-mcp`](https://gitlab-master.nvidia.com/omniml/integration/nmm-sandbox/-/tree/main/tools/mcp) that adds:

* `resolve_cluster_factory(name)` — turn `"cw_dfw"` into the `{cluster_host, cluster_user, identity, account, ...}` dict that `submit_job` consumes, so internal users supply 1 arg instead of 6.
* `submit_via_gitlab_ci(...)` — alternative submission path that triggers the nmm-sandbox CI's `intern-step` pipeline. Useful when the operator doesn't have direct cluster access.

Both servers are registered in the operator's `.mcp.json` side by side. The agent threads results from one into calls to the other. No Python coupling between them.

## Layout

```text
tools/mcp/
├── pyproject.toml              # name: modelopt-mcp, console_script
├── README.md                   # ← this file
├── modelopt_mcp/
│   ├── __init__.py
│   ├── server.py               # FastMCP entry; 5 tool definitions
│   └── bridge.py               # thin wrapper over launcher's core.py
│                               # + filesystem status/log helpers
└── tests/
    └── test_bridge.py          # 19 unit tests, fully hermetic
                                # (mocked subprocess + tmp_path fixtures)
```

## Phase 2

Out of scope for this PR; tracked separately:

* Capture `experiment_id` from Docker subprocess output (Phase 1 returns PID; nemo_run's id only appears in stdout after a few seconds).
* `wait_for_experiment(experiment_id, timeout_sec)` for blocking polls — pairs with `job_status` to avoid pause-loops in the agent.
* Extract the verify + submit helpers into a shared lib that `nmm-sandbox-mcp` can consume if it ever needs deeper Python integration.
