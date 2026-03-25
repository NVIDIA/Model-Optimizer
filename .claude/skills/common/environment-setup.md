# Environment Setup

Common detection for all ModelOpt skills. After this, you know what's available.

## 1. Get ModelOpt source

```bash
ls examples/llm_ptq/hf_ptq.py 2>/dev/null && echo "Source found"
```

If not found: `git clone https://github.com/NVIDIA/Model-Optimizer.git && cd Model-Optimizer`

## 2. Local or remote?

- **Local**: User wants to run on the current machine
- **Remote**: User mentions a hostname, cluster, or SSH

If remote, check for cluster config:

```bash
cat ~/.config/modelopt/clusters.yaml 2>/dev/null || cat .claude/clusters.yaml 2>/dev/null
```

If no config, ask user for: hostname, SSH username, SSH key path, remote workdir. Create `~/.config/modelopt/clusters.yaml` (see `skills/common/remote-execution.md` for format).

Then connect:

```bash
source .claude/skills/common/remote_exec.sh
remote_load_cluster <cluster_name>
remote_check_ssh
remote_detect_env    # sets REMOTE_ENV_TYPE = slurm / docker / bare
```

## 3. What compute is available?

Run on the **target machine** (local, or via `remote_run` if remote):

```bash
which srun sbatch 2>/dev/null && echo "SLURM"
docker info 2>/dev/null | grep -qi nvidia && echo "Docker+GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
```

Also check:

```bash
ls tools/launcher/launch.py 2>/dev/null && echo "Launcher available"
```

No GPU anywhere → **stop**: task requires a CUDA GPU.

## Summary

After this, you should know:

- ModelOpt source location
- Local or remote (+ cluster config if remote)
- SLURM / Docker+GPU / bare GPU
- Launcher availability
- GPU model and count

Return to the skill's SKILL.md for the execution path based on these results.

## Multi-user / Slack bot

If `MODELOPT_WORKSPACE_ROOT` is set, read `skills/common/workspace-management.md` before proceeding.
