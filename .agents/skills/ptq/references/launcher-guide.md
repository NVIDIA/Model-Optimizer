# Using the ModelOpt Launcher for PTQ

The launcher (`tools/launcher/`) handles SLURM and Docker execution. Read `tools/launcher/CLAUDE.md` for full docs.

## Quick Start

```bash
cd tools/launcher
uv run launch.py --yaml <config.yaml> --yes          # SLURM (SLURM_HOST set)
uv run launch.py --yaml <config.yaml> hf_local=<cache> --yes  # Local Docker
```

## HF Transformers PTQ Config

The launcher provides `common/hf/ptq.sh` which wraps `hf_ptq.py`. Configure via environment variables:

```yaml
job_name: <Model>_<Format>
pipeline:
  task_0:
    script: common/hf/ptq.sh
    environment:
      - HF_MODEL: <HuggingFace model ID, e.g. Qwen/Qwen3-0.6B>
      - QFORMAT: <format, e.g. nvfp4, fp8, int4_awq>
      - CALIB_SIZE: "512"
      - EXPORT_PATH: /scratchspace/exported_model
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 1
      gpus_per_node: <num_gpus>
```

Extra `hf_ptq.py` flags can be passed via `args`:

```yaml
    args:
      - --batch_size 2
      - --trust_remote_code
```

## Output Location

`EXPORT_PATH` controls the path inside the container (default: `/scratchspace/exported_model`). The launcher mounts `/scratchspace` to a host directory automatically — you cannot change the host path.

**Local Docker** — find the checkpoint on the local host:

```bash
find tools/launcher/local_experiments -name "config.json" -path "*/exported_model/*" 2>/dev/null
```

**Remote SLURM** — the checkpoint is on the remote machine. Check the launcher's experiment directory on the remote host (typically `~/experiments/cicd/...`). Use `remote_run "find ..."` or check the job log for the export path.

## SLURM vs Local Docker

| Mode | Invocation |
| --- | --- |
| Remote SLURM | `SLURM_HOST=<host> SLURM_ACCOUNT=<acct> uv run launch.py --yaml <cfg> user=<ssh_user> identity=<ssh_key> --yes` |
| Local SLURM | `SLURM_HOST=$(hostname) SLURM_ACCOUNT=<acct> uv run launch.py --yaml <cfg> --yes` |
| Local Docker | `uv run launch.py --yaml <cfg> hf_local=<cache> --yes` |

The launcher SSHes to `SLURM_HOST` via `nemo_run.SSHTunnel`. If `identity` is omitted, it uses `~/.ssh/id_rsa`.

**If using `clusters.yaml`**: read the cluster config and map fields to launcher args:

| `clusters.yaml` field | Launcher arg/env |
| --- | --- |
| `login_node` | `SLURM_HOST` env var |
| `user` | `user=` CLI arg |
| `ssh_key` | `identity=` CLI arg |
| `workspace` | `SLURM_JOB_DIR` env var (default: `~/experiments`) |
| `slurm.default_account` | `SLURM_ACCOUNT` env var |
| `slurm.default_partition` | `pipeline.task_0.slurm_config.partition=<name>` CLI override (default: `batch`) |

## Known Issues

- **UID mapping in Docker**: May cause `getpwuid` failures. Add `USER=user` and `LOGNAME=user` to environment.
- **Megatron-LM submodule**: Only needed for `MegatronLMQuantizeTask` (Megatron models). HF PTQ via `common/hf/ptq.sh` does not require it.

## Dry Run

```bash
uv run launch.py --yaml <config> --dryrun --yes -v
```

## Examples

```bash
ls tools/launcher/examples/
```

Copy and modify the closest match.
