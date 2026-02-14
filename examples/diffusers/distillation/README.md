# LTX-2 Distillation Training with ModelOpt

Knowledge distillation for LTX-2 DiT models using NVIDIA ModelOpt. A frozen **teacher** guides a trainable **student** through a combined loss:

```
L_total = α × L_task + (1-α) × L_distill
```

Currently supported:
- **Quantization-Aware Distillation (QAD)** — student uses ModelOpt fake quantization 

Planned:
- **Sparsity-Aware Distillation (SAD)** — student uses ModelOpt sparsity

## Installation

```bash
# From the distillation example directory
cd examples/diffusers/distillation

# Install Model-Optimizer (from repo root)
pip install -e ../../..

# Install all dependencies (ltx-trainer, ltx-core, ltx-pipelines, omegaconf)
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

Use the ltx-trainer preprocessing to extract latents and text embeddings:

```bash
python -m ltx_trainer.preprocess \
    --input_dir /path/to/videos \
    --output_dir /path/to/preprocessed \
    --model_path /path/to/ltx2/checkpoint.safetensors
```

### 2. Configure

Copy and edit the example config:

```bash
cp configs/distillation_example.yaml configs/my_experiment.yaml
```

Key settings to update:

```yaml
model:
  model_path: "/path/to/ltx2/checkpoint.safetensors"
  text_encoder_path: "/path/to/gemma/model"

data:
  preprocessed_data_root: "/path/to/preprocessed/data"

distillation:
  distillation_alpha: 0.5       # 1.0 = pure task loss, 0.0 = pure distillation
  quant_cfg: "FP8_DEFAULT_CFG"  # or INT8_DEFAULT_CFG, NVFP4_DEFAULT_CFG, null

# IMPORTANT: disable ltx-trainer's built-in quantization
acceleration:
  quantization: null
```

### 3. Run Training

**Single GPU:**

```bash
python distillation_trainer.py --config configs/my_experiment.yaml
```

**Multi-GPU (single node) with accelerate:**

```bash
accelerate launch \
    --config_file configs/accelerate/fsdp.yaml \
    --num_processes 8 \
    distillation_trainer.py --config configs/my_experiment.yaml
```

**Multi-node Training with Accelerate**

To launch on multiple nodes, make sure to set the following environment variables on each node:

- `NUM_NODES`: Total number of nodes
- `GPUS_PER_NODE`: Number of GPUs per node
- `NODE_RANK`: Unique rank/index of this node (0-based)
- `MASTER_ADDR`: IP address of the master node (rank 0)
- `MASTER_PORT`: Communication port (e.g., 29500)

Then run this (on every node):

```bash
accelerate launch \
    --config_file configs/accelerate/fsdp.yaml \
    --num_machines $NUM_NODES \
    --num_processes $((NUM_NODES * GPUS_PER_NODE)) \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    distillation_trainer.py --config configs/my_experiment.yaml
```

**Config overrides** can be passed via CLI using dotted notation:

```bash
accelerate launch ... distillation_trainer.py \
    --config configs/my_experiment.yaml \
    ++distillation.distillation_alpha=0.6 \
    ++distillation.quant_cfg=INT8_DEFAULT_CFG \
    ++optimization.learning_rate=1e-5
```

## Configuration Reference

### Calibration

Before training begins, calibration runs full denoising inference to collect activation statistics for accurate quantizer scales. This is cached as a step-0 checkpoint and reused on subsequent runs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `calibration_prompts_file` | null | Text file with one prompt per line. Use the HuggingFace dataset 'Gustavosta/Stable-Diffusion-Prompts' if not null. |
| `calibration_size` | 128 | Number of prompts (each runs a full denoising loop) |
| `calibration_n_steps` | 30 | Denoising steps per prompt |
| `calibration_guidance_scale` | 4.0 | CFG scale (should match inference-time) |

### Checkpoint Resume

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resume_from_checkpoint` | null | `"latest"` to auto-detect, or explicit path |
| `must_save_by` | null | Minutes after which to save and exit (for Slurm time limits) |
| `restore_quantized_checkpoint` | null | Restore a pre-quantized model (skips calibration) |
| `save_quantized_checkpoint` | null | Path to save the final quantized model |

### Custom Quantization Configs

To define custom quantization configs, add entries to `CUSTOM_QUANT_CONFIGS` in `distillation_trainer.py`:

```python
CUSTOM_QUANT_CONFIGS["MY_FP8_CFG"] = {
    "quant_cfg": mtq.FP8_DEFAULT_CFG["quant_cfg"],
    "algorithm": "max",
}
```

Then reference it in your YAML: `quant_cfg: MY_FP8_CFG`.
