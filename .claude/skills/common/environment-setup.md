# Environment Setup

Common setup for all ModelOpt skills. Run through this before any task.

## 1. Get ModelOpt source

```bash
ls examples/llm_ptq/hf_ptq.py 2>/dev/null && echo "ModelOpt source found"
```

If not found:

```bash
git clone https://github.com/NVIDIA/Model-Optimizer.git && cd Model-Optimizer
```

## 2. Detect execution environment

### Where: local or remote?

- **Local**: The user wants to run on the current machine
- **Remote**: The user mentions a hostname, cluster, SSH, or a cluster config exists:

  ```bash
  cat ~/.config/modelopt/clusters.yaml 2>/dev/null || cat .claude/clusters.yaml 2>/dev/null
  ```

If remote and no config exists, ask the user for: hostname, SSH username, SSH key path, remote working directory. Then create `~/.config/modelopt/clusters.yaml`:

```yaml
clusters:
  <alias>:
    login_node: <hostname>
    user: <username>
    ssh_key: <key_path>
    workspace: <remote_workdir>
default_cluster: <alias>
```

### How: SLURM, Docker, or bare metal?

Run these checks on the **target machine** (local, or via SSH if remote):

```bash
which srun sbatch 2>/dev/null && echo "SLURM"
docker info 2>/dev/null | grep -qi nvidia && echo "Docker+GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
```

Use `nvidia-smi` for GPU detection — it's more reliable than `torch.cuda` which depends on the Python environment having CUDA-enabled PyTorch installed.

### Execution context summary

After detection, you should know which row you're in:

| Environment | Launcher? | Execution method |
| --- | --- | --- |
| Remote SLURM | Yes | `SLURM_HOST=<host> uv run launch.py --yaml <cfg> --yes` |
| Local SLURM | Yes | `SLURM_HOST=$(hostname) uv run launch.py --yaml <cfg> --yes` |
| Local Docker + GPU | Yes | `uv run launch.py --yaml <cfg> hf_local=<cache> --yes` |
| Remote bare GPU (SSH) | No | SSH + run scripts directly. See `references/remote-execution.md` |
| Local bare GPU | No | Run scripts directly |
| No GPU anywhere | — | **Stop**: task requires a CUDA GPU |

## 3. Launcher setup (if applicable)

The launcher (`tools/launcher/`) handles SLURM and Docker execution for PTQ, deployment, evaluation, and benchmarking. Read `tools/launcher/CLAUDE.md` for full docs.

Check if it's available:

```bash
ls tools/launcher/launch.py 2>/dev/null && echo "Launcher available"
```

For SLURM, set these env vars:

| Variable | Required | Example |
| --- | --- | --- |
| `SLURM_HOST` | Yes | `cluster-login.example.com` |
| `SLURM_ACCOUNT` | Yes | `my_account` |
| `SLURM_HF_LOCAL` | Optional | `/lustre/hf-cache` |
| `HF_TOKEN` | If gated models | `hf_abc...` |

For local Docker, pass `hf_local=<path>` to the launcher CLI.

## 4. Multi-user / Slack bot

If `MODELOPT_WORKSPACE_ROOT` is set, you are in a multi-user environment. Read `skills/common/workspace-management.md` before proceeding.
