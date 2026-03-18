# Configuration

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `SLURM_HOST` | Slurm login node hostname | Yes (remote) |
| `SLURM_ACCOUNT` | Slurm account for billing | Yes (remote) |
| `SLURM_JOB_DIR` | Remote directory for job artifacts | Yes (remote) |
| `SLURM_HF_LOCAL` | Path to HuggingFace model cache on the cluster | Yes (remote) |
| `HF_TOKEN` | HuggingFace API token | No |
| `NEMORUN_HOME` | NeMo Run home directory (default: cwd) | No |

## YAML Config Format

### Typed Task Config (recommended)

Uses a typed task class with named, documented fields:

```yaml
job_name: Qwen3-8B_NVFP4_DEFAULT_CFG
pipeline:
  task_0:
    _target_: common.megatron_lm.quantize.task.MegatronLMQuantizeTask
    config:
      model: Qwen/Qwen3-8B
      quant_cfg: NVFP4_DEFAULT_CFG
      tp: 4
      calib_size: 32
      hf_local: /hf-local/
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 4
      gpus_per_node: 4
```

### Raw SandboxTask Config

For full control or scripts without a typed task class:

```yaml
job_name: Qwen3-8B_NVFP4_DEFAULT_CFG
pipeline:
  task_0:
    script: common/megatron_lm/quantize/quantize.sh
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
    script: common/megatron_lm/quantize/quantize.sh
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1

  task_1:
    script: common/megatron_lm/export/export.sh
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
```

The `<<global_vars.X>>` syntax shares values across tasks.

## `--yaml` vs `pipeline=@`

**`--yaml config.yaml`** (recommended) — maps top-level keys to function arguments.
Contains `job_name` and `pipeline`:

```bash
uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes
```

**`pipeline=@config.yaml`** — bare `SandboxPipeline` without `job_name` wrapper:

```bash
uv run launch.py pipeline=@bare_pipeline.yaml job_name=my_job --yes
```

## CLI Overrides

Any parameter can be overridden:

```bash
# Change nodes
uv run launch.py --yaml config.yaml pipeline.task_0.slurm_config.nodes=2 --yes

# Change container
uv run launch.py --yaml config.yaml \
    pipeline.task_0.slurm_config.container=nvcr.io/nvidia/tensorrt-llm/release:1.2.0 --yes

# Change typed config field
uv run launch.py --yaml config.yaml pipeline.task_0.config.tp=1 --yes
```

## Useful Flags

| Flag | Description |
|---|---|
| `--yes` / `-y` | Skip confirmation prompt |
| `-v` | Verbose output |
| `--dryrun` | Print resolved config without running |
| `--to-yaml output.yaml` | Dump resolved config to file |
| `detach=true` | Submit and return immediately |

## Model and Dataset Storage (`hf_local`)

Pipeline YAMLs use `hf_local` as a path prefix for model weights and datasets. This should be a **self-managed directory that mirrors the HuggingFace Hub hierarchy**:

```text
/hf-local/
├── Qwen/Qwen3-8B/
├── meta-llama/Llama-3.1-8B/
├── abisee/cnn_dailymail/
└── cais/mmlu/
```

Using a dedicated folder is preferred over the HuggingFace cache (`~/.cache/huggingface`) to avoid cache corruption from concurrent jobs.

```bash
# Populate
huggingface-cli download Qwen/Qwen3-8B --local-dir /hf-local/Qwen/Qwen3-8B

# Override via CLI
uv run launch.py --yaml config.yaml pipeline.task_0.config.hf_local=/mnt/models/ --yes

# Download from Hub directly (no local cache)
uv run launch.py --yaml config.yaml pipeline.task_0.config.hf_local="" --yes
```

For Slurm clusters, `SLURM_HF_LOCAL` sets the container mount path.
