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

- **Local**: The user wants to run on the current machine (no cluster/hostname mentioned)
- **Remote**: The user mentions a hostname, cluster name, SSH, or wants to run on a specific cluster

**If the user names a cluster or hostname**, check for cluster config:

```bash
cat ~/.config/modelopt/clusters.yaml 2>/dev/null || cat .claude/clusters.yaml 2>/dev/null
```

**If the cluster exists in config → use remote execution tools:**

```bash
source .claude/skills/common/remote_exec.sh
remote_load_cluster <cluster_name>    # loads config into env vars
remote_check_ssh                      # establishes persistent SSH session
remote_detect_env                     # detects SLURM/Docker/bare on remote
```

This sets `REMOTE_HOST`, `REMOTE_USER`, `REMOTE_ENV_TYPE`, and SLURM defaults. Use `remote_run` for all subsequent remote commands. See `references/remote-execution.md` for full details.

**If remote and no config exists**, ask the user for: hostname, SSH username, SSH key path, remote working directory. Then create `~/.config/modelopt/clusters.yaml`:

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

**Skip this if you already ran `remote_detect_env` above** — it sets `REMOTE_ENV_TYPE` for you.

Only run these checks for **local** execution (no remote cluster):

```bash
which srun sbatch 2>/dev/null && echo "SLURM"
docker info 2>/dev/null | grep -qi nvidia && echo "Docker+GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
```

Use `nvidia-smi` for GPU detection — it's more reliable than `torch.cuda` which depends on the Python environment having CUDA-enabled PyTorch installed.

### Execution context summary

After detection, you should know which row you're in:

| Environment | How detected | Launcher? | Execution method |
| --- | --- | --- | --- |
| Remote SLURM | `remote_detect_env` → `REMOTE_ENV_TYPE=slurm` | Yes | `SLURM_HOST=$REMOTE_HOST SLURM_ACCOUNT=<acct> uv run launch.py --yaml <cfg> --yes` |
| Local SLURM | `which srun` succeeds locally | Yes | `SLURM_HOST=$(hostname) uv run launch.py --yaml <cfg> --yes` |
| Local Docker + GPU | `docker info` shows nvidia locally | Yes | `uv run launch.py --yaml <cfg> hf_local=<cache> --yes` |
| Remote bare GPU | `remote_detect_env` → `REMOTE_ENV_TYPE=bare` | No | Use `remote_run` to run scripts. See `references/remote-execution.md` |
| Local bare GPU | `nvidia-smi` succeeds, no Docker/SLURM | No | Run scripts directly |
| No GPU anywhere | — | — | **Stop**: task requires a CUDA GPU |

For Remote SLURM, get `SLURM_ACCOUNT` from the cluster config's `slurm.default_account` field.

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
