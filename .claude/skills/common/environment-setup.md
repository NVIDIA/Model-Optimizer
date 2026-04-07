# Environment Setup

Common detection for all ModelOpt skills. After this, you know what's available.

## Env-1. Get ModelOpt source

```bash
ls examples/llm_ptq/hf_ptq.py 2>/dev/null && echo "Source found"
```

If not found: `git clone https://github.com/NVIDIA/Model-Optimizer.git && cd Model-Optimizer`

If found, ensure the source is up to date:

```bash
git pull origin main
```

If previous runs left patches in `modelopt/` (from 4C unlisted model work), check whether they should be kept. Reset only if starting a completely new task: `git checkout main`.

## Env-2. Local or remote?

1. **User explicitly requests local or remote** → follow the user's choice
2. **User doesn't specify** → check for cluster config:

```bash
cat ~/.config/modelopt/clusters.yaml 2>/dev/null || cat .claude/clusters.yaml 2>/dev/null
```

If a cluster config exists with content → **use the remote cluster** (do not fall back to local even if local GPUs are available — the cluster config indicates the user's preferred execution environment). Otherwise → **local execution**.

For remote, connect:

```bash
source .claude/skills/common/remote_exec.sh
remote_load_cluster <cluster_name>
remote_check_ssh
remote_detect_env    # sets REMOTE_ENV_TYPE = slurm / docker / bare
```

If remote but no config, ask user for: hostname, SSH username, SSH key path, remote workdir. Create `~/.config/modelopt/clusters.yaml` (see `skills/common/remote-execution.md` for format).

## Env-3. What compute is available?

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

**No GPU detected?**

- If local with no GPU and no cluster config → ask the user:
  *"No local GPU detected. Do you have a remote machine or cluster with GPUs? If so, I'll need connection details (hostname, SSH username, key path, remote workdir) to run there."*
- If user provides remote info → create `clusters.yaml`, go back to Env-2
- If user has no GPU anywhere → **stop**: this task requires a CUDA GPU

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
