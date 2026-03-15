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

## Model and Dataset Storage (`hf_local`)

Pipeline YAMLs use a `global_vars.hf_local` path prefix for model weights and datasets. This should be a **self-managed directory that mirrors the HuggingFace Hub hierarchy**:

```text
/hf-local/
├── Qwen/Qwen3-8B/              # model weights
├── meta-llama/Llama-3.1-8B/    # model weights
├── abisee/cnn_dailymail/        # calibration dataset
└── cais/mmlu/                   # evaluation dataset
```

Using a dedicated folder is preferred over the HuggingFace cache (`~/.cache/huggingface`) to avoid cache corruption from concurrent jobs writing to the same cache directory.

You can populate it by copying or symlinking from an existing HuggingFace download:

```bash
# Example: download a model and copy to hf_local
huggingface-cli download Qwen/Qwen3-8B --local-dir /hf-local/Qwen/Qwen3-8B
```

Override `hf_local` in any YAML via CLI:

```bash
# Use a different local path
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml \
    pipeline.global_vars.hf_local=/mnt/my-models/ --yes

# Download from HuggingFace Hub directly (no local cache)
uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml \
    pipeline.global_vars.hf_local="" --yes
```

For Slurm clusters, `SLURM_HF_LOCAL` sets the container mount path (e.g., `/lustre/.../hf-local:/hf-local`).

## Directory Structure

```text
launcher/
├── launch.py                    # Main entrypoint
├── core.py                      # Shared logic (also used by nmm-sandbox's slurm.py)
├── slurm_config.py              # SlurmConfig dataclass and factory
├── pyproject.toml               # Dependencies (nemo-run, pyyaml)
├── common/                      # Shared scripts executed on the cluster
│   ├── service_utils.sh         # Error handling, MPI rank utilities
│   ├── query.py                 # OpenAI-compatible query client
│   ├── megatron-lm/quantize/
│   │   └── quantize.sh          # PTQ quantization + MMLU evaluation
│   ├── tensorrt-llm/query.sh    # TRT-LLM server launch + query
│   ├── vllm/query.sh            # vLLM server launch + query
│   ├── eagle3/                  # EAGLE3 speculative decoding scripts
│   └── specdec_bench/           # Speculative decoding benchmark
├── Qwen/Qwen3-8B/              # Example configs
│   ├── megatron_lm_ptq.yaml     # PTQ quantization pipeline
│   └── hf_offline_eagle3.yaml   # EAGLE3 offline pipeline
└── modules/                     # Dependencies
    ├── Megatron-LM/             # Git submodule: NVIDIA Megatron-LM
    └── Model-Optimizer -> ../.. # Symlink to parent (auto-created if missing)
```

> **Note:** `modules/Model-Optimizer` is a symlink to the parent directory (`../..`),
> not a submodule. This avoids recursive nesting. `launch.py` auto-creates
> the symlink on first run if it's missing.

## YAML Config Format

A config YAML defines the job name, pipeline metadata, and one or more tasks:

```yaml
job_name: Qwen3-8B_NVFP4_DEFAULT_CFG
pipeline:
  skip: false
  allow_to_fail: false
  note:

  task_0:
    script: common/megatron-lm/quantize/quantize.sh
    args:
      - --calib-dataset-path-or-name /hf-local/abisee/cnn_dailymail
      - --calib-size 32
    environment:
      - MLM_MODEL_CFG: Qwen/Qwen3-8B
      - QUANT_CFG: NVFP4_DEFAULT_CFG
      - TP: 4
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 4
      gpus_per_node: 4
```

### Multi-task Pipeline

Tasks run sequentially — `task_1` starts only after `task_0` completes.
Example (illustrative — export script may not exist yet):

```yaml
job_name: Qwen3-8B_quantize_export
pipeline:
  global_vars:
    hf_model: /hf-local/Qwen/Qwen3-8B

  task_0:
    script: common/megatron-lm/quantize/quantize.sh
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1

  task_1:
    script: common/megatron-lm/export/export.sh
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
    script: common/megatron-lm/quantize/quantize.sh
    slurm_config:
      _factory_: "slurm_factory"
```

**`pipeline=@config.yaml`** — the YAML is a bare `SandboxPipeline` (no `job_name` or `pipeline` wrapper).
This is useful for reusing pipeline configs across different job names:

```yaml
# bare_pipeline.yaml — used with: uv run launch.py pipeline=@bare_pipeline.yaml --yes
task_0:
  script: common/megatron-lm/quantize/quantize.sh
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
2. Code is packaged via `PatternPackager` — `modules/Megatron-LM/`, `modules/Model-Optimizer/` (via symlink), and `common/` are synced
3. For remote jobs: code is rsynced to the cluster, an sbatch script is generated and submitted via SSH
4. For local jobs: a Docker container is launched with the same container image and mounts
5. The `code/` directory on the cluster mirrors the launcher structure:

```text
code/
├── modules/
│   ├── Megatron-LM/megatron/...
│   └── Model-Optimizer/modelopt/...
└── common/...
```

## Running Tests

```bash
cd launcher
uv pip install pytest
uv run python3 -m pytest ../tests/unit/launcher/ -v -o "addopts=" \
    --confcutdir=../tests/unit/launcher
```

64 unit tests cover core dataclasses, factory registry, YAML parsing, Docker/Slurm executor construction, environment merging, and end-to-end Docker launch.

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

Verified: identical MMLU results (0.719 local, 0.730 OCI-HSG) from both launchers.

For architecture details, factory system, and Claude Code workflows, see [ADVANCED.md](ADVANCED.md).
