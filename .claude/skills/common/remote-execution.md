# Remote Execution

Read this when Claude Code runs on a different machine than the target GPU cluster/workstation. This covers SSH connectivity, cluster config, persistent sessions, and remote command execution.

---

## 1. Cluster Config

Config locations (checked in order, first found wins):

1. `~/.config/modelopt/clusters.yaml` — user-level (not committed, recommended)
2. `.claude/clusters.yaml` — project-level (can be committed for shared defaults)
3. Interactive input — if neither file exists, ask the user (see SKILL.md Step 0) and write `~/.config/modelopt/clusters.yaml` before proceeding

```yaml
clusters:
  my-cluster:
    login_node: cluster-login.example.com   # SSH hostname or SSH config alias
    user: username                           # SSH user
    ssh_key: ~/.ssh/id_rsa                   # (optional) SSH key path
    ssh_proxy: "socat - PROXY:localhost:%h:%p,proxyport=3128"  # (optional) proxy
    workspace: /absolute/path/to/workdir     # Remote working directory
    gpu_type: H100                           # For quant format recommendation
    slurm:                                   # (optional) pre-fill SLURM defaults
      default_account: my_account
      default_partition: batch_short

default_cluster: my-cluster
```

See `.claude/clusters.yaml.example` for a fully annotated example with multiple cluster types.

---

## 2. Connect and Establish Persistent Session

```bash
source .claude/skills/common/remote_exec.sh
remote_load_cluster <cluster_name>    # or omit name to use default_cluster
remote_check_ssh                      # validates connectivity + starts persistent session
```

`remote_check_ssh` starts an SSH **ControlMaster** connection. All subsequent `remote_run` / `remote_sync_*` / SCP calls reuse this single connection:

- ~180ms per command (vs 5-15s per new connection)
- Eliminates flaky proxy timeouts
- Auto-cleaned up when the shell exits

---

## 3. Detect Remote Environment

```bash
remote_detect_env
```

Auto-discovers whether the remote has SLURM, Docker, or bare-metal GPUs. Sets `REMOTE_ENV_TYPE` to `slurm`, `docker`, `bare`, or `unknown`.

After detection, proceed with the environment-specific setup:

- **SLURM** → prefix all commands with `remote_run`. For SLURM job scripts, see the skill's own references.
- **Docker** → use `remote_docker_run <container> "<command>"`
- **Bare metal** → use `remote_run` directly

---

## 4. Running Commands Remotely

### Single commands

```bash
remote_run "nvidia-smi"
remote_run "python --version"
remote_run "sbatch /path/to/job.sh"
```

`remote_run` uses base64 encoding internally, so special characters (`%`, `$`, quotes) work without escaping. It retries up to 3 times on SSH failures.

### Syncing files

```bash
# Local → remote
remote_sync_to /local/path remote_subdir

# Remote → local
remote_sync_from remote_subdir /local/path
```

Both use rsync over the persistent SSH session with default excludes (`.git`, `__pycache__`, `.claude`, `*.pyc`, `node_modules`, `*.egg-info`). The `.claude` directory is intentionally excluded — skills and config should not be synced to the remote machine.

### SCP (alternative to rsync)

SCP also reuses the persistent session automatically via ControlMaster:

```bash
scp /local/script.sh ${REMOTE_USER}@${REMOTE_HOST}:/remote/path/
```

---

## 5. The Two-Script Pattern

When submitting SLURM jobs remotely, write **two files** locally to avoid shell escaping issues:

1. **SLURM wrapper** (e.g., `job_slurm.sh`) — `#SBATCH` directives + `srun` with container
2. **Inner runner** (e.g., `run.sh`) — the actual work (runs inside the container)

Then upload both and submit:

```bash
remote_sync_to /local/scripts/ scripts/
JOBID=$(remote_run "sbatch /remote/path/scripts/job_slurm.sh" | grep -o '[0-9]\+' | tail -1)
```

---

## 6. Verifying Results Remotely

```bash
remote_run "ls -lh <output_path>/"
remote_run "cat <output_path>/hf_quant_config.json"
```

Or fetch results to local:

```bash
remote_sync_from <remote_output_subdir> /local/output/
```

---

## 7. Troubleshooting

| Problem | Cause | Fix |
| ------- | ----- | --- |
| `Connection timed out during banner exchange` | Proxy/login node overloaded | `remote_run` retries 3x automatically; use persistent session to avoid |
| SSH proxy completely unreachable (`Network is unreachable`) | VPN/proxy host is down or not running on this machine | Check if VPN is connected; verify `socat`/proxy service is running locally; try direct SSH by temporarily removing `ssh_proxy` from config |
| `unix_listener: cannot bind to path ... Read-only file system` | SSH ControlMaster socket in non-writable `/tmp` | `remote_exec.sh` auto-finds writable dir; ensure `TMPDIR` or `/tmp/claude-*` exists |
| `cd: /home/user/~/path: No such file or directory` | `~` not expanding on remote | Use absolute paths in `workspace` config, not `~/...` |
| Login nodes resolve home dirs differently | Symlinked home dirs vary by node | Use absolute lustre/NFS paths (e.g., `/lustre/fs1/...`) in job scripts |
| `#!` becomes `#\!` in scripts | Shell environment mangles shebang | Fix with `sed -i 's\|^#\\\\!\|#!\|' script.sh` after writing |

## Reference Files

- **`skills/common/remote_exec.sh`** — Full utility library (session, run, sync, SLURM, Docker helpers)
- **`.claude/clusters.yaml`** — Active cluster configuration
- **`.claude/clusters.yaml.example`** — Annotated example config
