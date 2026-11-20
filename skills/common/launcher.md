# Running Jobs with the ModelOpt Launcher

The launcher (`tools/launcher/`) runs a pipeline YAML on SLURM or in local Docker. Read `tools/launcher/CLAUDE.md` for the YAML format and full docs; the calling skill covers its own task config.

## Quick Start

```bash
cd tools/launcher
uv run launch.py --yaml <config.yaml> --dryrun --yes -v       # preview the resolved config
uv run launch.py --yaml <config.yaml> --yes                   # SLURM (SLURM_HOST set)
uv run launch.py --yaml <config.yaml> hf_local=<cache> --yes  # Local Docker
```

To start a new pipeline, `ls tools/launcher/examples/`, copy the closest match and modify it.

## SLURM vs Local Docker

| Mode | Invocation |
| --- | --- |
| Remote SLURM | `SLURM_HOST=<host> SLURM_ACCOUNT=<acct> uv run launch.py --yaml <cfg> user=<ssh_user> identity=<ssh_key> --yes` |
| Local SLURM | `SLURM_HOST=$(hostname) SLURM_ACCOUNT=<acct> uv run launch.py --yaml <cfg> --yes` |
| Local Docker | `uv run launch.py --yaml <cfg> hf_local=<cache> --yes` |

The launcher SSHes to `SLURM_HOST` via `nemo_run.SSHTunnel`. If `identity` is omitted, it uses `~/.ssh/id_rsa`.

> **`SLURM_HF_LOCAL` — HF cache bind-mount.** `slurm_factory` mounts the host HF
> cache at `/hf-local` from `$SLURM_HF_LOCAL` (default `/hf-local`). If that host
> dir is missing, pyxis fails at container start (`enroot-mount: failed to mount:
> /hf-local`). Export a real path (`SLURM_HF_LOCAL=/lustre/.../hf-local`).

**If using `clusters.yaml`**: read the cluster config and map fields to launcher args:

| `clusters.yaml` field | Launcher arg/env |
| --- | --- |
| `login_node` | `SLURM_HOST` env var |
| `user` | `user=` CLI arg |
| `ssh_key` | `identity=` CLI arg |
| `workspace` | `SLURM_JOB_DIR` env var (default: `~/experiments`) |
| `slurm.default_account` | `SLURM_ACCOUNT` env var |
| `slurm.default_partition` | `SLURM_PARTITION` env var (default: `batch`), for every task; `pipeline.task_<N>.slurm_config.partition=<name>` overrides one task |

## Sizing the Job

> **Match `gpus_per_node` to the cluster's node GPU count / QOS minimum.** If it
> is below what the QOS requires (many clusters mandate a full node), `sbatch`
> rejects the job with `QOSMinGRES` or `Requested node configuration is not
> available`. e.g. GB300 nodes have 4 GPUs and require the full node → set
> `gpus_per_node: 4`; B300/B200 nodes have 8. Check with `sinfo -o '%P %G'`.
>
> Also: the launcher emits `#SBATCH --mem=0` (all node RAM) unless `SLURM_MEM` is
> set, so clusters that scale RAM per GPU reject a partial-GPU alloc (`requested …
> RAM, but only <N> GPUs`). Requesting the full node fixes it.

## Output Location

A task's `/scratchspace` is its experiment's folder on the host, `<job_dir>/cicd/<experiment_id>`; you cannot change the host path. Every task of one YAML shares it, and a later experiment sees it as `/cicd/<experiment_id>/`.

**Local Docker** — `<job_dir>` is `tools/launcher/local_experiments` on the local host.

**Remote SLURM** — `<job_dir>` is `SLURM_JOB_DIR` (default `~/experiments`) on the remote host. Use `remote_run "find ..."` (see `remote-execution.md`) or check the job log for the path.

## Known Issues

- **UID mapping in Docker**: May cause `getpwuid` failures. Add `USER=user` and `LOGNAME=user` to environment.
