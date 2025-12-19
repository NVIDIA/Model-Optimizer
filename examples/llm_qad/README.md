# QAD Training Scripts

Quantization-Aware Distillation (QAD) training scripts for language models using Megatron-LM. These scripts enable training quantized (NVFP4) student models with knowledge distillation from full-precision teacher models.

## Overview

| Script | Purpose |
|--------|---------|
| `qad.sh` | Main training script (run inside container) |
| `sbatch_qad.sh` | SLURM batch submission wrapper |
| `configs/*.conf` | Model-specific configuration files |

## Requirements

### Software Dependencies

- **Container**: Nvidia PyTorch container (tested with `nvcr.io/nvidia/pytorch:25.06-py3`)
- **Python**: 3.10+
- **transformers**: 4.54+ (installed automatically)

### Clone Required Repositories

```bash
# Set your workspace directory
export WORKSPACE=/path/to/your/workspace

# Clone Megatron-LM (with ModelOpt integration)
git clone https://github.com/NVIDIA/Megatron-LM.git ${WORKSPACE}/Megatron-LM
cd ${WORKSPACE}/Megatron-LM
git checkout <modelopt-branch>  # Use branch with ModelOpt support

# Clone Model-Optimizer
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git ${WORKSPACE}/Model-Optimizer
```

### Prepare Container

For SLURM with Pyxis/Enroot, create a squashfs container:

```bash
# Pull and convert Docker image to sqsh
enroot import docker://nvcr.io/nvidia/pytorch:25.06-py3
mv nvidia+pytorch+25.06-py3.sqsh /path/to/containers/pytorch_25.06.sqsh
```

### Prepare Checkpoints

You need the following checkpoints before training:

1. **Student checkpoint**: Quantized (NVFP4) model in Megatron-LM format
2. **Teacher checkpoint**: Full-precision (BF16) model in Megatron-LM format
3. **Teacher config YAML**: Model architecture configuration

See [Megatron-LM ModelOpt examples](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt) for checkpoint conversion from HuggingFace format. 

## Creating a Configuration

### Use Template Configs

Template configurations are provided in `configs/`:

| Config | Model | Description |
|--------|-------|-------------|
| `qwen3-30b-a3b-instruct-2507-moe_template.conf` | Qwen3-30B-A3B-Instruct | MoE template (start here) |
| `qwen3-8b.conf` | Qwen3-8B | Dense model example |

### Create Your Config

1. Copy the template:
   ```bash
   cp configs/qwen3-30b-a3b-instruct-2507-moe_template.conf configs/my-experiment.conf
   ```

2. Fill in required empty fields:
   - `STUDENT_CKPT` - Path to quantized student MLM checkpoint
   - `TEACHER_CKPT` - Path to teacher MLM checkpoint  
   - `TEACHER_MODEL_CONFIG` - Path to teacher YAML config (see below)
   - `MLM_DIR` - Path to your Megatron-LM clone

3. Optionally adjust:
   - `QAD_CHECKPOINT_ROOT`, `DATACACHE_DIR` - output paths
   - `CONTAINER_IMAGE`, `CONTAINER_MOUNTS` - container settings
   - `BLEND_PATH` - dataset path

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

Set `TEACHER_MODEL_CONFIG` in your config to point to this file.

## Dataset Generation

QAD training requires preprocessed datasets in Megatron-LM format. Use the one-button script to generate datasets:

```bash
cd data_utils/

bash generate_dataset.sh \
    --output-dir /path/to/datasets \
    --mlm-path /path/to/Megatron-LM \
    --tokenizer <HF-model> (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)
```

### Requirements

- HuggingFace token to access `nvidia/Nemotron-Post-Training-Dataset-v2`
- Login first: `huggingface-cli login`

### What It Does

1. Downloads OpenScience + Nemotron-v2 datasets
2. Preprocesses to Megatron-LM format
3. Creates combined datablend JSON with weights:
   - 30% Nemotron-v2 code
   - 20% Nemotron-v2 math
   - 20% Nemotron-v2 stem
   - 10% Nemotron-v2 chat
   - 20% OpenScience

### Output

```
/path/to/datasets/
├── openscience_splits_preprocessed/  # Megatron format
├── nemotron_v2_preprocessed/         # Megatron format
└── datablend_combined.json           # Combined config
```

Set `BLEND_PATH` in your config to point to `datablend_combined.json`.

## Quick Start

### SLURM Batch Submission (Recommended)

```bash
# Submit training job
sbatch sbatch_qad.sh --config configs/qwen3-30b-a3b-instruct-2507-moe.conf

# With HuggingFace token (for gated models)
sbatch sbatch_qad.sh --hf-token $HF_TOKEN --config configs/qwen3-30b-a3b-thinking-2507-moe.conf

# Multi-node (override SLURM header)
sbatch --nodes=4 sbatch_qad.sh --config configs/qwen3-30b-a3b-instruct-2507-moe.conf
```

### Interactive Mode

```bash
# Get interactive node first
srun -A <account> --nodes=1 -p batch --mpi=pmix \
    -J qad:dev \
    --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
    --container-mounts="..." \
    -t 4:0:0 --pty bash

# Run training
bash qad.sh --config configs/qwen3-8b.conf
```

## Required Config Variables

### Model Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `STUDENT_MODEL` | Student model name (for logging) | `Qwen3-30B-A3B` |
| `TEACHER_MODEL` | Teacher model name (for logging) | `Qwen3-30B-A3B` |
| `TOKENIZER_MODEL` | HuggingFace tokenizer path | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `IS_MOE` | Whether model is Mixture of Experts | `true` or `false` |

### Checkpoint Paths

| Variable | Description |
|----------|-------------|
| `STUDENT_CKPT` | Path to quantized student MLM checkpoint |
| `TEACHER_CKPT` | Path to teacher MLM checkpoint |
| `TEACHER_MODEL_CONFIG` | Path to teacher model YAML config |
| `STUDENT_CONFIG_FILE` | Path to student model args script (in Megatron-LM) |

### Training Hyperparameters

| Variable | Description | Example |
|----------|-------------|---------|
| `LR` | Learning rate | `1e-5` |
| `GBS` | Global batch size | `256` |
| `MIN_LR` | Minimum learning rate | `0.0` |
| `LR_DECAY_STYLE` | LR decay schedule | `constant`, `cosine` |
| `SAVE_INTERVAL` | Checkpoint save interval (iterations) | `200` |
| `LOG_INTERVAL` | Logging interval (iterations) | `10` |

### Data Configuration

| Variable | Description |
|----------|-------------|
| `DATASET_NAME` | Dataset identifier (for output naming) |
| `BLEND_PATH` | Path to datablend JSON file |
| `TRAIN_SAMPLES` | Number of training samples |

### Parallelism

| Variable | Description | Example |
|----------|-------------|---------|
| `TP_SIZE` | Tensor parallelism size | `1`, `2`, `4` |
| `PP_SIZE` | Pipeline parallelism size | `1` |
| `EP_SIZE` | Expert parallelism (MoE only) | `4`, `8` |
| `MBS` | Micro-batch size | `1`, `2` |
| `NUM_GPUS` | GPUs per node | `4`, `8` |

### Required Paths

| Variable | Description |
|----------|-------------|
| `MLM_DIR` | Path to Megatron-LM directory |
| `MODELOPT_DIR` | Path to Model-Optimizer directory |
| `QAD_CHECKPOINT_ROOT` | Root directory for checkpoints |
| `DATACACHE_DIR` | Directory for data cache |

### Container Configuration

| Variable | Description |
|----------|-------------|
| `CONTAINER_IMAGE` | Path to container sqsh file |
| `CONTAINER_MOUNTS` | Container mount points |
| `CONTAINER_WORKDIR` | Working directory inside container |

## Optional Config Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MASTER_PORT` | `29500` | Distributed training port |
| `MAX_SEQ` | Model default | Override sequence length |
| `KD_CFG_PATH` | Auto-generated | Custom KD config YAML |
| `RUN_TAG` | Empty | Custom tag for output naming |

## Parallelism Settings

### Dense Models (e.g., Qwen3-8B)

```bash
export IS_MOE=false
export TP_SIZE=1
export EP_SIZE=1
export MBS=4
```

### MoE Models (e.g., Qwen3-30B-A3B)

```bash
export IS_MOE=true
export TP_SIZE=2
export EP_SIZE=4
export MBS=2
```

**Note**: MoE models require loading both student and teacher models, which increases memory requirements significantly.

### GPU Requirements

| Model | TP | EP | Nodes (4 GPU/node) | Total GPUs |
|-------|----|----|---------------------|------------|
| Qwen3-8B | 1 | 1 | 1 | 4-8 |
| Qwen3-30B-A3B | 2 | 4 | 2-4 | 8-16 |

## MoE Performance Optimizations

For MoE models, the script automatically enables performance optimizations:

- `--moe-token-dispatcher-type alltoall`
- `--moe-shared-expert-overlap`
- `--moe-permute-fusion`
- `--moe-grouped-gemm`
- `--cross-entropy-loss-fusion`

To disable (if causing issues):
```bash
export ENABLE_MOE_PERF=0
```

## Output Structure

```
$QAD_CHECKPOINT_ROOT/
├── <student>-NVFP4-Teacher-<teacher>-Data-<dataset>-lr<lr>-minlr<min>-decay<style>-gbs<gbs>-si<save>-li<log>/
│   ├── checkpoints/<model>/
│   │   ├── iter_0000200/
│   │   ├── iter_0000400/
│   │   └── latest_checkpointed_iteration.txt
│   ├── tensorboard/<model>/
│   └── logs/
│       ├── _qad_<datetime>.log
│       └── _<datetime>.env.log
└── logs_slurm/
    ├── <job-name>_<jobid>_<datetime>.log
    └── err_<job-name>_<jobid>_<datetime>.log
```

## Resuming Training

Training automatically resumes from checkpoints:

1. **Fresh start**: If no checkpoint exists, loads from `STUDENT_CKPT` with `--finetune`
2. **Resume**: If `latest_checkpointed_iteration.txt` exists, resumes from there

To force a fresh start:
```bash
rm -rf /path/to/checkpoints/*/latest_checkpointed_iteration.txt
```

## Job Dependencies

Chain jobs to run sequentially:

```bash
# Submit first job
JOB1=$(sbatch --parsable sbatch_qad.sh --config ...)

# Submit dependent job (runs after JOB1 finishes)
sbatch --dependency=afterany:$JOB1 sbatch_qad.sh --config ...
```

## Troubleshooting

### OOM Errors

1. **Reduce MBS**: Set `MBS=1`
2. **Increase parallelism**: Increase `EP_SIZE` or `TP_SIZE`
3. **Add more nodes**: Increase `SLURM --nodes`
4. **Disable log-params-norm**: Set `LOG_PARAMS_NORM=0`

### Triton Cache Errors

Clear corrupted cache:
```bash
rm -rf ~/.triton/cache
```

The scripts automatically use per-job Triton cache directories.

## See Also

- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [MoE Optimization Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html)
