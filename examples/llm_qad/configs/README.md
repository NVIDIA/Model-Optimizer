# QAD Training Configuration Files

Configuration files for QAD (Quantization-Aware Distillation) training.
Works with both `sbatch_qwen_qad.sh` (SLURM) and `qwen_qad.sh` (Docker/Interactive).

## Quick Start

### SLURM Batch Mode
```bash
sbatch --nodes=4 -t 4:00:00 sbatch_qwen_qad.sh --config configs/qwen3-8b-default.conf
sbatch --nodes=4 -t 8:00:00 sbatch_qwen_qad.sh --config configs/qwen3-8b-nemotron.conf
sbatch --nodes=8 -t 8:00:00 sbatch_qwen_qad.sh --config configs/qwen3-30b-a3b-moe.conf
```

### Docker/Interactive Mode
```bash
bash qwen_qad.sh --config configs/qwen3-8b-default.conf
bash qwen_qad.sh --config configs/qwen3-8b-nemotron.conf

# Override config values
LR=1e-5 bash qwen_qad.sh --config configs/qwen3-8b-default.conf
```

## Available Configs

| Config | Model | Dataset | Recommended SLURM |
|--------|-------|---------|-------------------|
| `qwen3-8b-default.conf` | Qwen3-8B | openscience | `--nodes=4 -t 4:00:00` |
| `qwen3-8b-nemotron.conf` | Qwen3-8B | nemotron | `--nodes=4 -t 8:00:00` |
| `qwen3-30b-a3b-moe.conf` | Qwen3-30B-A3B | nemotron | `--nodes=8 -t 8:00:00` |

## Creating Custom Configs

```bash
cp configs/template.conf configs/my-experiment.conf
# Edit my-experiment.conf (set STUDENT_CKPT and TEACHER_CKPT)
sbatch --nodes=4 -t 4:00:00 sbatch_qwen_qad.sh --config configs/my-experiment.conf
```

## Configuration Variables

### Model
| Variable | Description | Required |
|----------|-------------|----------|
| `STUDENT_MODEL` | Student model architecture | Yes |
| `TEACHER_MODEL` | Teacher model architecture | Yes |

### Checkpoints (REQUIRED)
| Variable | Description | Required |
|----------|-------------|----------|
| `STUDENT_CKPT` | Path to student checkpoint (FP4 for QAD) | **Yes** |
| `TEACHER_CKPT` | Path to teacher checkpoint (BF16) | **Yes (QAD)** |
| `TEACHER_MODEL_CONFIG` | Path to teacher config YAML | No (auto) |

### Training
| Variable | Description | Default |
|----------|-------------|---------|
| `LR` | Learning rate | 1e-6 |
| `DATASET_NAME` | Dataset to use | openscience |
| `KD_CFG_PATH` | Custom KD config YAML | (empty) |
| `TRAIN_SAMPLES` | Override sample count | (auto) |

### Parallelism
| Variable | Description | Default |
|----------|-------------|---------|
| `TP_SIZE` | Tensor parallelism | 8 |
| `PP_SIZE` | Pipeline parallelism | 1 |
| `EP_SIZE` | Expert parallelism (MoE) | 1 |
| `MBS` | Micro-batch size | 16 |
| `NUM_GPUS` | GPUs per node | 8 |
| `MASTER_PORT` | Distributed port | 29500 |

### Paths
| Variable | Description |
|----------|-------------|
| `MLM_DIR` | Megatron-LM directory |
| `MODELOPT_DIR` | ModelOpt directory |
| `MODELS_ROOT` | Model checkpoints root |
| `QAD_CHECKPOINT_ROOT` | Output root |
| `DATACACHE_DIR` | Data cache |

### Container
| Variable | Description |
|----------|-------------|
| `CONTAINER_IMAGE` | Container squashfs |
| `CONTAINER_MOUNTS` | Mount points |
| `CONTAINER_WORKDIR` | Working directory |

## Output Directory Naming

Output directories are named using the checkpoint directory names:
```
{QAD_CHECKPOINT_ROOT}/{STUDENT_CKPT_NAME}-Teacher-{TEACHER_CKPT_NAME}-Data-{DATASET}-lr{LR}/
```

Example:
```
/checkpoints/Qwen3-8B-NVFP4-TP8-MLM-Teacher-Qwen3-8B-TP8-MLM-Data-nemotron-lr1e-6/
```

## SLURM Options

SLURM parameters should be passed via `sbatch` command:

```bash
sbatch --nodes=4 -t 4:00:00 sbatch_qwen_qad.sh --config ...
sbatch --nodes=8 -t 8:00:00 -p batch -A myaccount sbatch_qwen_qad.sh --config ...
```

## Variable Priority

```
Script defaults < Config file < Environment variables < Command line args
```
