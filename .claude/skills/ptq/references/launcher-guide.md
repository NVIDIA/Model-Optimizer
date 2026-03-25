# Using the ModelOpt Launcher for PTQ

The launcher (`tools/launcher/`) handles SLURM, Docker, and local execution. Read `tools/launcher/CLAUDE.md` for full documentation. This guide covers PTQ-specific usage.

## Quick Start

```bash
cd tools/launcher
uv run launch.py --yaml <config.yaml> --yes
```

## Writing a PTQ Config

### For supported models (typed task)

Use the `MegatronLMQuantizeTask` for clean configs:

```yaml
job_name: <Model>_<Format>
pipeline:
  task_0:
    _target_: common.megatron_lm.quantize.task.MegatronLMQuantizeTask
    config:
      model: <HuggingFace model ID>
      quant_cfg: <QUANT_CFG name, e.g., NVFP4_DEFAULT_CFG>
      tp: <tensor parallelism>
      calib_dataset: abisee/cnn_dailymail
      calib_size: 512
      hf_local: /hf-local/
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: <tp>
      gpus_per_node: <tp>
```

Available `quant_cfg` values — check `modelopt/torch/quantization/config.py` for the full list.

### For custom scripts (raw SandboxTask)

When using a custom PTQ script (e.g., unsupported models):

```yaml
job_name: <Model>_custom_ptq
pipeline:
  task_0:
    script: <path_to_your_script.sh>
    args:
      - --model <model_path>
      - --output <output_path>
    environment:
      - HF_TOKEN: <token>
      - CUDA_VISIBLE_DEVICES: "0"
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 1
      gpus_per_node: 1
```

Place custom scripts in `tools/launcher/common/` so the packager includes them.

## SLURM vs Local

The launcher auto-detects based on environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `SLURM_HOST` | Login node for SSH submission | `cluster-login.example.com` |
| `SLURM_ACCOUNT` | SLURM account | `my_account` |
| `SLURM_PARTITION` | SLURM partition | `batch` |
| `HF_TOKEN` | HuggingFace token for gated models | `hf_abc...` |

If `SLURM_HOST` is set → SLURM execution. Otherwise → local Docker.

For local Docker, pass `hf_local=` to specify the model cache:

```bash
uv run launch.py --yaml <config> hf_local=/mnt/hf-local --yes
```

## GPU Sizing Guide

| Model size | TP | GPUs | Nodes |
|------------|-----|------|-------|
| < 15B | 1 | 1 | 1 |
| 15B-40B | 2-4 | 2-4 | 1 |
| 40B-100B | 4-8 | 4-8 | 1 |
| 100B+ | 8+ | 8+ | 2+ (use FSDP2 or multi-node) |

## Dry Run and Debug

Preview what the launcher will do without running:

```bash
uv run launch.py --yaml <config> --dryrun --yes -v
```

Export resolved config:

```bash
uv run launch.py --yaml <config> --to-yaml resolved.yaml
```

## Example Configs

Check `tools/launcher/examples/` for working configs:

```bash
ls tools/launcher/examples/
```

Copy and modify the closest match for your model.
