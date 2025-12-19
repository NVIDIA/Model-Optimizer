# Qwen QAD Training Scripts

Quantization-Aware Distillation (QAD) training scripts for Qwen models using Megatron-LM. These scripts enable training quantized (NVFP4) student models with knowledge distillation from full-precision teacher models.

## Overview

| Script | Purpose |
|--------|---------|
| `qwen_qad.sh` | Main training script (interactive/Docker) |
| `sbatch_qwen_qad.sh` | SLURM batch submission wrapper |
| `configs/*.conf` | Model-specific configuration files |

## Quick Start

### SLURM Batch Submission (Recommended) for H100 x 8

```bash
# With HuggingFace token (for gated models)
sbatch sbatch_qwen_qad.sh --hf-token $HF_TOKEN --config configs/qwen3-30b-a3b-thinking-2507-moe.conf
```

### Interactive Mode

```bash
# Get interactive node first
srun -A coreai_dlalgo_modelopt --nodes=1 -p batch --mpi=pmix \
    -J qwen-qad:dev \
    --container-image=/lustre/.../pytorch_25.06-py3.sqsh \
    --container-mounts="/lustre/fsw:/lustre/fsw" \
    -t 4:0:0 --pty bash

# Run training
bash qwen_qad.sh --config configs/qwen3-8b-default.conf
```

## Configuration Files

Configuration files in `configs/` define model architecture, parallelism, and checkpoint paths.

### Required Config Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `STUDENT_MODEL` | Student model name | `Qwen3-8B` |
| `TEACHER_MODEL` | Teacher model name | `Qwen3-8B` |
| `STUDENT_CKPT` | Path to quantized student checkpoint | `/path/to/Qwen3-8B-NVFP4-TP1-MLM` |
| `TEACHER_CKPT` | Path to teacher checkpoint | `/path/to/Qwen3-8B-TP1-MLM` |
| `TEACHER_MODEL_CONFIG` | Teacher model YAML config | `/path/to/Qwen3-8B-teacher.yaml` |
| `TP_SIZE` | Tensor parallelism size | `1`, `4`, `8` |
| `MBS` | Micro-batch size | `1`, `2`, `4` |

### Optional Config Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PP_SIZE` | `1` | Pipeline parallelism size |
| `EP_SIZE` | `1` | Expert parallelism (MoE models) |
| `NUM_GPUS` | `8` | GPUs per node |
| `LR` | `1e-6` | Learning rate |
| `DATASET_NAME` | `openscience` | Training dataset |
| `TRAIN_SAMPLES` | Auto | Override training samples |
| `BLEND_PATH` | Auto | Override datablend path |

### Example Config Structure

```bash
# configs/qwen3-8b-default.conf
export STUDENT_MODEL="Qwen3-8B"
export TEACHER_MODEL="Qwen3-8B"
export STUDENT_CKPT="/path/to/Qwen3-8B-NVFP4-TP1-MLM"
export TEACHER_CKPT="/path/to/Qwen3-8B-TP1-MLM"
export TEACHER_MODEL_CONFIG="/path/to/Qwen3-8B-teacher.yaml"
export TP_SIZE=1
export PP_SIZE=1
export MBS=4
export NUM_GPUS=8
export DATASET_NAME="combined_v2_cot_chat"
```

## Dataset Options

### Naming Convention

Datasets follow this naming pattern:

- **Plain text**: `datablend_<dataset>.json`
- **With COT** (chain-of-thought): `datablend_<dataset>_cot.json`
- **With chat template**: `datablend_<dataset>_chat.json`
- **COT + chat**: `datablend_<dataset>_cot_chat.json`

### Available Datasets

#### Nemotron-v1 (Large scale, ~25M samples full)

| Name | Samples | Description |
|------|---------|-------------|
| `nemotron_30pct` | ~7.5M | ALL subjects @ 30% |
| `nemotron_30pct_cot_chat` | ~7.5M | ALL @ 30% + COT + Chat |
| `nemotron_stem_cot_chat` | ~5M | STEM only + COT + Chat |
| `nemotron_v1_math_30pct_cot_chat` | ~583K | Math split |
| `nemotron_v1_code_30pct_cot_chat` | ~540K | Code split |

#### Nemotron-v2 (High quality, ~400K samples @ 30%)

| Name | Samples | Description |
|------|---------|-------------|
| `nemotron_v2_30pct` | ~398K | English @ 30% |
| `nemotron_v2_cot_chat` | ~398K | English + COT + Chat |
| `nemotron_v2_stem_30pct_cot_chat` | ~101K | STEM split |
| `nemotron_v2_math_30pct_cot_chat` | ~68K | Math split |
| `nemotron_v2_code_30pct_cot_chat` | ~50K | Code split |

#### OpenScience

| Name | Samples | Description |
|------|---------|-------------|
| `openscience` | ~300K | Plain text |
| `openscience_chat` | ~300K | With chat template |

#### Combined Datasets (Recommended)

| Name | Samples | Description |
|------|---------|-------------|
| `combined_cot_chat` | ~8.2M | 20% OpenScience + 50% v1 + 30% v2 |
| `combined_v2_cot_chat` | ~1M | Code & Math focused blend |

## Parallelism Settings

### Dense Models (Qwen3-8B)

```bash
TP_SIZE=1   # Single GPU per tensor
PP_SIZE=1   # No pipeline parallelism
EP_SIZE=1   # Not MoE
MBS=4       # Can use larger micro-batch
```

### MoE Models (Qwen3-30B-A3B)

```bash
TP_SIZE=4   # Tensor parallel across 4 GPUs
PP_SIZE=1   # No pipeline parallelism  
EP_SIZE=8   # 128 experts / 8 = 16 experts per rank
MBS=1       # Small MBS for large vocab KD loss
```

**Note**: MoE models with EP=8 require 4 nodes (32 GPUs total).

### GPU Requirements

| Model | TP | EP | Nodes | Total GPUs |
|-------|----|----|-------|------------|
| Qwen3-8B | 1 | 1 | 1 | 8 |
| Qwen3-30B-A3B | 4 | 4 | 2 | 16 |
| Qwen3-30B-A3B | 4 | 8 | 4 | 32 |

## Multi-Node Training

### SLURM Multi-Node

```bash
# Set nodes in sbatch header or command line
#SBATCH --nodes=4

# Or override at submission
sbatch --nodes=4 sbatch_qwen_qad.sh --config configs/qwen3-30b-a3b-thinking-2507-moe.conf
```

The script automatically:

- Detects `SLURM_NNODES` and `SLURM_JOB_NODELIST`
- Sets `MASTER_ADDR` to first node
- Exports `NODE_RANK` per process

### Manual Multi-Node (Interactive)

On each node, set:

```bash
export NNODES=4
export NODE_RANK=0  # 0, 1, 2, 3 for each node
export MASTER_ADDR=<first-node-hostname>
export MASTER_PORT=29500
bash qwen_qad.sh --config configs/your-config.conf
```

## Resuming Training

Training automatically resumes from checkpoints:

1. **Fresh start**: Loads from `STUDENT_CKPT` with `--finetune`
2. **Resume**: If `CHECKPOINT_DIR/latest_checkpointed_iteration.txt` exists, loads from there

To force fresh start, remove the checkpoint directory:

```bash
rm -rf /path/to/checkpoints/*/latest_checkpointed_iteration.txt
```

## Job Dependencies

Chain jobs to run sequentially:

```bash
# Submit first job
JOB1=$(sbatch --parsable sbatch_qwen_qad.sh --config ...)

# Submit dependent job (runs after JOB1 finishes, regardless of success/failure)
sbatch --dependency=afterany:$JOB1 sbatch_qwen_qad.sh --config ...
```

Dependency options:

- `afterany:jobid` - Run after job finishes (success or failure)
- `afterok:jobid` - Run only if job succeeds
- `afternotok:jobid` - Run only if job fails

## Environment Variables

### HuggingFace Authentication

```bash
# Via argument (recommended - not logged)
sbatch sbatch_qwen_qad.sh --hf-token $HF_TOKEN --config ...

# Via environment
export HF_TOKEN=hf_xxx
sbatch sbatch_qwen_qad.sh --config ...
```

### Path Overrides

```bash
export MLM_DIR=/path/to/Megatron-LM
export MODELOPT_DIR=/path/to/TensorRT-Model-Optimizer
export MODELS_ROOT=/path/to/models
export QAD_CHECKPOINT_ROOT=/path/to/checkpoints
export DATACACHE_DIR=/path/to/data_cache
```

### Training Overrides

```bash
export LR=1e-5                    # Learning rate
export DATASET_NAME=nemotron_v2   # Dataset
export TRAIN_SAMPLES=100000       # Override sample count
export ITERATIONS_TO_SKIP=100     # Skip first N iterations
```

## Output Structure

```bash
$QAD_CHECKPOINT_ROOT/
├── <student>-Teacher-<teacher>-Data-<dataset>-lr<lr>/
│   ├── checkpoints/<model-name>/
│   │   ├── iter_0000200/
│   │   ├── iter_0000400/
│   │   └── latest_checkpointed_iteration.txt
│   ├── tensorboard/<model-name>/
│   └── logs/
│       ├── <model>_qad_<datetime>.log
│       └── <model>_<datetime>.env.log
└── logs_slurm/
    ├── coreai_dlalgo_modelopt-qwen.qad_<jobid>_<datetime>.log
    └── err_coreai_dlalgo_modelopt-qwen.qad_<jobid>_<datetime>.log
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir /path/to/tensorboard/ --port 6006 --bind_all
```

### Check Job Status

```bash
squeue -u $USER                    # List your jobs
squeue -j <jobid>                  # Check specific job
sacct -j <jobid> --format=...      # Job accounting info
```

### Estimated Time

```bash
squeue -j <jobid> -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10L %.6D %R"
# %.10L shows time left
```

## Troubleshooting

### OOM Errors

1. **Reduce MBS**: Set `MBS=1` in config
2. **Increase EP**: For MoE, increase `EP_SIZE` (requires more nodes)
3. **Disable log-params-norm**: Set `LOG_PARAMS_NORM=0` in config

### Rate Limiting (429 Errors)

Use HuggingFace token:

```bash
sbatch sbatch_qwen_qad.sh --hf-token $HF_TOKEN --config ...
```

### Shape Mismatch Errors

Ensure teacher model config has correct GQA settings:

```yaml
num_query_groups: 4    # For Qwen3-30B-A3B
kv_channels: 128
```

### Gradient Norm Spikes

Isolated spikes are normal with heterogeneous data. Monitor if:

- Spikes are persistent (every few iterations)
- Loss doesn't recover after spike
- Training diverges

## Advanced Usage

### Custom KD Config

```bash
bash qwen_qad.sh --config configs/... 1e-6 Qwen3-8B dataset Qwen3-8B /path/to/kd_config.yaml
```

### Skip Iterations

Resume but skip specific iterations:

```bash
export ITERATIONS_TO_SKIP=100
sbatch sbatch_qwen_qad.sh --config ...
```

### Custom Datablend

```bash
export BLEND_PATH=/path/to/custom_datablend.json
export TRAIN_SAMPLES=500000
sbatch sbatch_qwen_qad.sh --config ...
```

## Requirements

- **Container**: PyTorch 25.06+ with CUDA support
- **Megatron-LM**: With ModelOpt integration
- **TensorRT-Model-Optimizer**: Latest version
- **transformers**: 4.54+

## See Also

- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [MoE Optimization Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html)
