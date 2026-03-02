# LTX-2 QAD Example (Quantization-Aware Distillation)

This example demonstrates **Quantization-Aware Distillation (QAD)** for [LTX-2](https://github.com/Lightricks/LTX-2) using the native LTX training loop and [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer). It combines:

- **LTX packages**: training loop, datasets, and strategies (masked loss, audio/video split)
- **NVIDIA ModelOpt**: PTQ calibration (`mtq.quantize`), distillation (`mtd.convert`), and NVFP4 quantization

Combined loss (same idea as the full distillation trainer):

```text
L_total = α × L_task + (1−α) × L_distill
```

For the **full-stage QAD** implementation (LTX-2 DiT with ModelOpt quantization, full calibration options, checkpoint resume, and multi-node training), see the NVIDIA Model-Optimizer distillation example:

- **Full distillation trainer**: [distillation_trainer.py](https://github.com/NVIDIA/Model-Optimizer/blob/ca1f9687bd741a0c73791c093692eff0f95d2d46/examples/diffusers/distillation/distillation_trainer.py)  
- **Example docs**: [examples/diffusers/distillation](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers/distillation) (README, configs, and usage there).

## Requirements

- Python 3.10+
- CUDA-capable GPU(s)
- [Accelerate](https://huggingface.co/docs/accelerate) (for FSDP multi-GPU training)

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package | Source |
|--------|--------|
| **ltx-core** | `git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core` |
| **ltx-pipelines** | `git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines` |
| **ltx-trainer** | `git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-trainer` |
| **nvidia-modelopt** | PyPI (`nvidia-modelopt`) |

You may also need to install PyTorch, Accelerate, safetensors, and PyYAML if not already present:

```bash
pip install torch accelerate safetensors pyyaml
```

## Project layout

| File | Description |
|------|-------------|
| `sample_example_qad_diffusers.py` | Main script: QAD training and inference checkpoint creation |
| `ltx2_qad.yaml` | LTX training config (model, data, optimization, QAD options) |
| `fsdp_custom.yaml` | Accelerate FSDP config for multi-GPU training |

## Usage

### 1. Prepare your dataset

Run the LTX preprocessing script to extract latents and text embeddings from your videos. Use `preprocess_dataset.py` with the following arguments (matching the LTX training pipeline):

```bash

    https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/scripts/process_dataset.py /path/to/videos \
    --resolution-buckets=384x256x97,256x160x121 \
    --output-dir=/path/to/preprocessed \
    --model-source=/path/to/ltx2/checkpoint.safetensors \
    --batch-size=4 \
    --encoder-type=gemma \
    --text-encoder-path=/path/to/gemma \
    --with-audio \
    --decode
```

- **Positional**: path to input videos (directory).
- **Required**: `--resolution-buckets`, `--output-dir`, `--model-source`, `--encoder-type`, `--text-encoder-path`.
- **Optional**: `--batch-size` (default 4), `--with-audio`, `--decode` (decode and save videos).

Set `data.preprocessed_data_root` in your config (step 2) to the same path as `--output-dir`.

On a **Slurm cluster**, run the same script via `srun` and `torchrun` (set `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE` from Slurm and use `--nnodes=$SLURM_NNODES` and `--nproc_per_node=8`).

### 2. Configure paths

Edit `ltx2_qad.yaml` and set:

- `model.model_source` – path to base LTX checkpoint (e.g. `.safetensors`)
- `model.text_encoder_path` – path to Gemma text encoder
- `data.preprocessed_data_root` – path to preprocessed LTX dataset

Adjust `qad` section as needed: `calib_size`, `kd_loss_weight`, `exclude_blocks`, `skip_inference_ckpt`.

#### Hyperparameters controllable via YAML (`ltx2_qad.yaml`)

All of the following can be set in `ltx2_qad.yaml`. QAD-specific options can also be overridden from the CLI (see step 3).

| Section | Key | Default (example) | Description |
|--------|-----|--------------------|-------------|
| **qad** | `calib_size` | `512` | Number of calibration batches for PTQ (more = better scale estimates, slower startup). |
| **qad** | `kd_loss_weight` | `0.5` | Weight for distillation loss in combined loss; `0` = task loss only, `1` = distillation only. |
| **qad** | `exclude_blocks` | `[0, 1, 46, 47]` | Transformer block indices to exclude from quantization (e.g. first/last blocks). |
| **qad** | `skip_inference_ckpt` | `false` | If `true`, do not build the inference checkpoint after training. |
| **optimization** | `learning_rate` | `1e-6` | Learning rate (low is typical for QAD/distillation). |
| **optimization** | `steps` | `300` | Total training steps. |
| **optimization** | `batch_size` | `1` | Per-device batch size. |
| **optimization** | `gradient_accumulation_steps` | `4` | Gradient accumulation steps (effective batch = batch_size × accumulation × num_gpus). |
| **optimization** | `optimizer_type` | `"adamw"` | Optimizer (`adamw`, etc.). |
| **checkpoints** | `interval` | `100` | Save a checkpoint every N steps; `null` to disable. |
| (root) | `output_dir` | `"outputs/ltx2_qad"` | Where to write checkpoints and logs. |

### 3. Run QAD training

Using Accelerate with the provided FSDP config:

```bash
accelerate launch --config_file fsdp_custom.yaml sample_example_qad_diffusers.py train \
    --config ltx2_qad.yaml \
```

Checkpoints are saved under `output_dir` (e.g. `outputs/ltx2_qad/checkpoints/`) as safetensors plus optional amax and modelopt state files.

### 4. Create inference checkpoint (ComfyUI-compatible)

To build a single inference checkpoint compatible with ComfyUI, use the PTQ checkpoint merger:

```bash
python -m ltx2.tools.ptq.checkpoint_merger \
    --artefact /path/to/amax_artifact.json \
    --checkpoint /path/to/ltx2_qad_bf16.safetensors \
    --config /path/to/config.yaml \
    --output /path/to/comfyui_checkpoints/nvfp4_qad_inference.safetensors
```

- **`--artefact`** – Path to the amax artifact JSON (from calibration / QAD training).
- **`--checkpoint`** – Path to the trained QAD weights (e.g. `ltx2_qad_bf16.safetensors` from your run).
- **`--config`** – Path to the merger config YAML.
- **`--output`** – Output path for the ComfyUI-ready `.safetensors` file.

This produces a single `.safetensors` file you can load in ComfyUI.

## How it works

1. **Model load** – Base transformer is loaded via `ltx_core.model_loader.load_transformer`.
2. **PTQ calibration** – ModelOpt `mtq.quantize` runs a calibration loop using the LTX dataset and training strategy; NVFP4 config excludes sensitive layers and optionally specific blocks.
3. **Distillation** – A full-precision teacher (same checkpoint) is loaded and the quantized model is wrapped with ModelOpt `mtd.convert` (KD loss).
4. **Training** – Standard LTX training loop with an overridden `_training_step` that adds KD loss via ModelOpt’s loss balancer.
5. **Checkpoint save** – Checkpoints are filtered (no teacher/loss/quantizer state), optionally dtype-matched to the base model, and saved as safetensors; amax and modelopt state can be saved separately.

## References

- LTX-2: [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
- NVIDIA ModelOpt: [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- Full-stage QAD / distillation trainer: [distillation_trainer.py](https://github.com/NVIDIA/Model-Optimizer/blob/ca1f9687bd741a0c73791c093692eff0f95d2d46/examples/diffusers/distillation/distillation_trainer.py) in Model-Optimizer (`examples/diffusers/distillation`)
