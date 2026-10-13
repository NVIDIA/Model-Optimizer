# QAD Training Scripts

Quantization-Aware Distillation (QAD) training scripts for language models using Megatron-LM. These scripts enable training quantized (e.g., NVFP4) student models with knowledge distillation from full-precision teacher models.

## Overview

| Script | Purpose |
|--------|---------|
| `qad.sh` | Main training script (run inside container) |
| `sbatch_qad.sh` | SLURM batch submission wrapper |
| `configs/*.conf` | Model-specific configuration files |

## Requirements

### Clone Required Repositories

```bash
# Set your workspace directory
export WORKSPACE=/path/to/your/workspace

# Clone Megatron-LM (with ModelOpt integration)
git clone https://github.com/NVIDIA/Megatron-LM.git ${WORKSPACE}/Megatron-LM

# Clone Model-Optimizer
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git ${WORKSPACE}/Model-Optimizer
```

### Prepare Checkpoints

You need the following checkpoints before training:

1. **Student checkpoint**: Quantized (e.g., NVFP4) model in Megatron-LM format
2. **Teacher checkpoint**: Full-precision (BF16) model in Megatron-LM format
3. **Teacher config YAML**: Model architecture configuration

See [Megatron-LM ModelOpt examples](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt) for checkpoint conversion from HuggingFace format.

## Creating a Configuration

### Available Templates

| Config | Model | Type |
|--------|-------|------|
| `qwen3-30b-a3b-instruct-2507-moe_template.conf` | Qwen3-30B-A3B-Instruct | MoE |
| `qwen3-8b_template.conf` | Qwen3-8B | Dense |

### Create Your Config

1. Copy a template:

   ```bash
   # For MoE models
   cp configs/qwen3-30b-a3b-instruct-2507-moe_template.conf configs/my-experiment.conf
   
   # For Dense models
   cp configs/qwen3-8b_template.conf configs/my-experiment.conf
   ```

2. Fill in required fields:

   **Checkpoints** (required):

   | Variable | Description |
   |----------|-------------|
   | `STUDENT_CKPT` | Path to quantized student MLM checkpoint |
   | `TEACHER_CKPT` | Path to teacher MLM checkpoint |
   | `TEACHER_MODEL_CONFIG` | Path to teacher YAML config (see below) |

   **Paths** (required):

   | Variable | Description |
   |----------|-------------|
   | `MLM_DIR` | Path to Megatron-LM directory |
   | `BLEND_PATH` | Path to datablend JSON (from dataset generation) |

   **Parallelism** (adjust for your hardware):

   | Variable | Dense Model | MoE Model |
   |----------|-------------|-----------|
   | `IS_MOE` | `false` | `true` |
   | `TP_SIZE` | `1` | `2` |
   | `EP_SIZE` | `1` | `4` |
   | `MBS` | `4` | `2` |

   **Training** (tune as needed):

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `LR` | `1e-5` | Learning rate |
   | `GBS` | `256` | Global batch size |
   | `SAVE_INTERVAL` | `200` | Checkpoint interval |

### Teacher Model Config (YAML)

Create a YAML file with teacher model architecture (example: `configs/Qwen3-30B-A3B-teacher.yaml`):

```yaml
num_layers: 48
hidden_size: 2048
num_attention_heads: 32
num_query_groups: 4
kv_channels: 128
ffn_hidden_size: 6144
```

## Dataset Generation

Use the one-button script to generate the default datablend:

```bash
cd data_utils/

bash generate_dataset.sh \
    --output-dir /path/to/datasets \
    --mlm-path /path/to/Megatron-LM \
    --tokenizer <HF-model>  # e.g., Qwen/Qwen3-30B-A3B-Instruct-2507
```

**Requirements**: HuggingFace token for `nvidia/Nemotron-Post-Training-Dataset-v2`. Login first: `huggingface-cli login`

**Output**: Creates `datablend_combined.json` with OpenScience + Nemotron-v2 datasets. Set `BLEND_PATH` in your config to point to this file.

## Quick Start

### SLURM Batch Submission (Recommended)

First, update `sbatch_qad.sh` SLURM header with your cluster settings:

- `--account=<your-account>`
- `--nodes`, `--gres=gpu`, `-t` as needed

```bash
# Submit training job (override account on command line)
sbatch --account=<your-account> sbatch_qad.sh --config configs/my-experiment.conf

# With HuggingFace token (for gated models)
sbatch --account=<your-account> sbatch_qad.sh --hf-token $HF_TOKEN --config configs/my-experiment.conf

# Adjust nodes and time
sbatch --account=<your-account> --nodes=4 -t 8:00:00 sbatch_qad.sh --config configs/my-experiment.conf
```

### Interactive Mode

```bash
# Get interactive node
srun -A <account> --nodes=1 -p batch --mpi=pmix \
    --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
    --container-mounts="..." \
    -t 4:0:0 --pty bash

# Run training
bash qad.sh --config configs/qwen3-8b.conf
```

## Resuming Training

Training automatically resumes from checkpoints. To force a fresh start:

```bash
rm -rf /path/to/checkpoints/*/latest_checkpointed_iteration.txt
```

## Troubleshooting

### OOM Errors

- Reduce `MBS`
- Increase `EP_SIZE`, `TP_SIZE`, `PP_SIZE`
- Add more nodes
