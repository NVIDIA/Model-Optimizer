# ModelOpt Launcher

Submit ModelOpt quantization, training, and evaluation jobs to Slurm clusters or run them locally with Docker.

## Quick Start

```bash
# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
git submodule update --init --recursive

# Run locally (requires local GPUs and Docker)
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml hf_local=/mnt/hf-local --yes

# Run on a Slurm cluster
export SLURM_HOST=login-node.example.com
export SLURM_ACCOUNT=my_account
export SLURM_HF_LOCAL=/shared/hf-local
export SLURM_JOB_DIR=/shared/experiments
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes
```

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `SLURM_HOST` | Slurm login node hostname | Yes (remote jobs) |
| `SLURM_ACCOUNT` | Slurm account for billing | Yes (remote jobs) |
| `SLURM_JOB_DIR` | Remote directory for job artifacts | Yes (remote jobs) |
| `SLURM_HF_LOCAL` | Path to HuggingFace model cache on the cluster | Yes (remote jobs) |
| `HF_TOKEN` | HuggingFace API token | No |
| `NEMORUN_HOME` | NeMo Run home directory (default: cwd) | No |

## Directory Structure

```text
launcher/
├── launch.py                    # Main entrypoint
├── core.py                      # Shared logic (also used by nmm-sandbox's slurm.py)
├── slurm_config.py              # SlurmConfig dataclass and factory
├── pyproject.toml               # Dependencies (nemo-run, pyyaml)
├── services/                    # Shell scripts executed on the cluster
│   ├── service_utils.sh         # Error handling, MPI rank utilities
│   └── megatron-lm/quantize/
│       ├── quantize.sh          # PTQ quantization + MMLU evaluation
│       └── Qwen3-8B.yaml        # Task config for Qwen3-8B
├── Qwen/Qwen3-8B/              # Example pipeline config
│   └── megatron_lm_ptq.yaml
└── modules/                     # Git submodules
    ├── Megatron-LM/             # NVIDIA Megatron-LM training framework
    └── Model-Optimizer/         # NVIDIA ModelOpt library
```

## YAML Config Format

A config YAML defines the job name, pipeline metadata, and one or more tasks:

```yaml
job_name: Qwen3-8B_NVFP4_DEFAULT_CFG
pipeline:
  skip: false
  allow_to_fail: false
  note:

  task_0:
    script: services/megatron-lm/quantize/quantize.sh
    args:
      - --calib-dataset-path-or-name /hf-local/abisee/cnn_dailymail
      - --calib-size 32
    environment:
      - MLM_MODEL_CFG: Qwen/Qwen3-8B
      - QUANT_CFG: NVFP4_DEFAULT_CFG
      - TP: 1
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 4
      gpus_per_node: 4
```

### Multi-task Pipeline

Tasks run sequentially — `task_1` starts only after `task_0` completes:

```yaml
job_name: Qwen3-8B_quantize_export
pipeline:
  global_vars:
    hf_model: /hf-local/Qwen/Qwen3-8B

  task_0:
    script: services/megatron-lm/quantize/quantize.sh
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1

  task_1:
    script: services/megatron-lm/export/export.sh
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
```

The `<<global_vars.X>>` syntax shares values across tasks.

### `--yaml` vs `pipeline=@`

There are two ways to load a config:

**`--yaml config.yaml`** (recommended) — the YAML maps top-level keys to function arguments.
The file contains both `job_name` and `pipeline`:

```yaml
# config.yaml — used with: uv run launch.py --yaml config.yaml --yes
job_name: Qwen3-8B_NVFP4
pipeline:
  task_0:
    script: services/megatron-lm/quantize/quantize.sh
    slurm_config:
      _factory_: "slurm_factory"
```

**`pipeline=@config.yaml`** — the YAML is a bare `SandboxPipeline` (no `job_name` or `pipeline` wrapper).
This is useful for reusing pipeline configs across different job names:

```yaml
# bare_pipeline.yaml — used with: uv run launch.py pipeline=@bare_pipeline.yaml --yes
task_0:
  script: services/megatron-lm/quantize/quantize.sh
  slurm_config:
    _factory_: "slurm_factory"
```

```bash
# With pipeline=@, set job_name separately
uv run launch.py pipeline=@bare_pipeline.yaml job_name=my_job --yes
```

### Overriding Parameters

Any parameter can be overridden from the command line:

```bash
# Change the number of nodes
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml \
    pipeline.task_0.slurm_config.nodes=2 --yes

# Change the container image
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml \
    pipeline.task_0.slurm_config.container=nvcr.io/nvidia/tensorrt-llm/release:1.3.0 --yes
```

### Useful Flags

| Flag | Description |
|---|---|
| `--yes` / `-y` | Skip confirmation prompt |
| `-v` | Verbose output |
| `--dryrun` | Resolve and print the full config without running |
| `--to-yaml output.yaml` | Dump the resolved config to a YAML file without running |
| `detach=true` | Submit the job and return immediately (don't wait for completion) |

```bash
# Preview the resolved config (all factory defaults expanded)
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml --dryrun --yes -v

# Dump resolved config to file for inspection or reproducibility
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml --to-yaml resolved.yaml

# Submit and detach
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml detach=true --yes
```

## Adding a New Model

1. Create a directory: `<Organization>/<ModelName>/`
2. Add a YAML config (e.g., `megatron_lm_ptq.yaml`) following the format above
3. Set `MLM_MODEL_CFG` to the HuggingFace model ID
4. Choose `QUANT_CFG` (e.g., `NVFP4_DEFAULT_CFG`, `INT8_DEFAULT_CFG`)
5. Set `nodes`, `ntasks_per_node`, `gpus_per_node` based on model size

## How It Works

1. `launch.py` parses the YAML and creates a `SandboxPipeline` with tasks and `SlurmConfig`
2. Code is packaged via `PatternPackager` — only `modules/Megatron-LM/`, `modules/Model-Optimizer/`, and `services/` are synced
3. For remote jobs: code is rsynced to the cluster, an sbatch script is generated and submitted via SSH
4. For local jobs: a Docker container is launched with the same container image and mounts
5. The `code/` directory on the cluster mirrors the launcher structure:

```text
code/
├── modules/
│   ├── Megatron-LM/megatron/...
│   └── Model-Optimizer/modelopt/...
└── services/...
```

## Reporting Bugs

When filing a bug report, please include:

1. **Version summary** — printed at the start of every run:

   ```text
   ============================================================
   Version Report
   ============================================================
     Launcher                       d28acd33     (main)
     Megatron-LM                    1e064f361    (main)
     Model-Optimizer                69c0d479     (main)
   ============================================================
   ```

2. **Reproducible config** — dump with `--to-yaml`:

   ```bash
   uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml --to-yaml bug_report.yaml
   ```

3. **Error output** — the relevant error message or traceback from the job log.

File issues at: <https://github.com/NVIDIA/Model-Optimizer/issues>

## Compatibility with nmm-sandbox

This launcher produces the same `code/` layout as [nmm-sandbox](https://gitlab-master.nvidia.com/omniml/integration/nmm-sandbox)'s `slurm.py`. The same YAML configs work with both:

```bash
# From nmm-sandbox (internal)
uv run slurm.py --yaml modules/Model-Optimizer/launcher/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes

# From Model-Optimizer/launcher (public)
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes
```
