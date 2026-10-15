# Quantization-Aware Distillation for Video Diffusion Models

> [!WARNING]
> **Third-Party License Notice — LTX-2**
>
> LTX-2 is a third-party model and set of packages developed and provided by Lightricks. LTX-2
> is **not** covered by the Apache 2.0 license that governs NVIDIA Model Optimizer.
>
> By installing and using LTX-2 packages (`ltx-core`, `ltx-pipelines`, `ltx-trainer`) with
> NVIDIA Model Optimizer, you **must** comply with the
> [LTX Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).
>
> Any derivative models or fine-tuned weights produced from LTX-2 using NVIDIA Model Optimizer
> (including quantized or distilled checkpoints) remain subject to the LTX Community License
> Agreement and are **not** covered by Apache 2.0.

Unified trainer for quantization-aware distillation (QAD) of video diffusion
models using NVIDIA ModelOpt. A frozen full-precision **teacher** guides a
quantized **student** to recover quality lost from quantization:

```text
L = alpha * L_task + (1 - alpha) * L_distill
```

Optionally, layer-wise distillation matches intermediate transformer block
outputs for stronger gradient signal:

```text
L_distill = (1 - gamma) * L_output + gamma * L_layer
```

Supported models:

| Backend | Model Family | Variants |
|---------|-------------|----------|
| `wan`   | Wan2.2      | `ti2v-5B`, `t2v-A14B`, `i2v-A14B` |
| `ltx2`  | LTX-2       | -- |

---

## Installation

```bash
cd examples/diffusers/train

# Install the distillation package + ONE model backend:
pip install -e ".[wan]"    # For Wan2.2
pip install -e ".[ltx2]"   # For LTX-2
```

> **Note:** Install only one backend at a time. Their transitive dependencies
> may conflict.

---

## Quick Start

### Smoke Test (Mock Data)

Verify the pipeline works end-to-end without real data:

```bash
accelerate launch \
    --config_file configs/accelerate/fsdp_wan.yaml \
    --num_processes 8 \
    train.py \
    --config configs/wan_distillation.yaml \
    model.model_path=/path/to/Wan2.2-TI2V-5B \
    distillation.use_mock_data=true \
    optimization.steps=100 \
    wandb.enabled=false
```

### Prepare Real Data

Preprocess raw videos + captions into latent space:

```bash
python -m distill.preprocess \
    --model_name wan \
    --model_variant ti2v-5B \
    --model_path /path/to/Wan2.2-TI2V-5B \
    --dataset /path/to/metadata.json \
    --output_dir /path/to/preprocessed
```

The `--model_variant` flag selects the correct VAE for encoding (e.g. `ti2v-5B`
uses a different VAE than `t2v-A14B`). See the model table above for valid variants.
The metadata file is a JSON array of `{"video": "/path/to/video.mp4", "caption": "..."}` entries.

### QAD Training

Quantize the student with NVFP4 and distill from the full-precision teacher:

```bash
accelerate launch \
    --config_file configs/accelerate/fsdp_wan.yaml \
    --num_processes 8 \
    train.py \
    --config configs/wan_distillation.yaml \
    model.model_path=/path/to/Wan2.2-TI2V-5B \
    data.preprocessed_data_root=/path/to/preprocessed \
    distillation.quant_cfg=NVFP4_DEFAULT_CFG
```

### Layer-Wise Distillation

Add intermediate layer matching on top of QAD:

```bash
accelerate launch \
    --config_file configs/accelerate/fsdp_wan.yaml \
    --num_processes 8 \
    train.py \
    --config configs/wan_distillation.yaml \
    model.model_path=/path/to/Wan2.2-TI2V-5B \
    data.preprocessed_data_root=/path/to/preprocessed \
    distillation.quant_cfg=NVFP4_DEFAULT_CFG \
    'distillation.layer_distillation_modules=[blocks.9,blocks.19,blocks.29]' \
    distillation.layer_distillation_weight=0.5
```

`layer_distillation_modules` accepts module paths (e.g. `blocks.9` for Wan,
`transformer_blocks.5` for LTX-2). Since QAD uses the same architecture for
teacher and student, the same module name is used for both.

---

## Configuration

### Config Inheritance

Model configs inherit shared defaults from `configs/default.yaml`:

```text
configs/
├── default.yaml              # Shared: optimization, checkpoints, flow matching, ...
├── wan_distillation.yaml     # Wan-specific: model, validation dims, wandb tags
└── ltx2_distillation.yaml    # LTX-2-specific
```

Create a new experiment by overriding the base:

```yaml
# configs/my_experiment.yaml
defaults: default.yaml

model:
  model_name: "wan"
  model_path: "/path/to/checkpoint"

distillation:
  quant_cfg: "NVFP4_DEFAULT_CFG"
```

### CLI Overrides

Any config field can be overridden via dotted OmegaConf syntax:

```bash
train.py --config configs/wan_distillation.yaml \
    optimization.learning_rate=1e-5 \
    optimization.steps=5000
```

### Key Fields

| Field | Default | Description |
|-------|---------|-------------|
| `distillation.quant_cfg` | null | ModelOpt quantization config (e.g. `NVFP4_DEFAULT_CFG`) |
| `distillation.distillation_alpha` | 0.0 | Loss mixing: 0 = pure distillation, 1 = pure task loss |
| `distillation.layer_distillation_modules` | null | Module paths for layer distillation (null = disabled) |
| `distillation.layer_distillation_weight` | 0.0 | gamma: mixing ratio between output (0) and layer (1) distillation |
| `distillation.layer_distillation_normalize` | true | L2-normalize features before layer loss (scale-invariant) |
| `distillation.resume_from_checkpoint` | "latest" | Resume path or "latest" for auto-resume |
| `optimization.learning_rate` | 2e-6 | Learning rate |
| `optimization.steps` | 10000 | Total training steps |
| `checkpoints.must_save_by` | null | Minutes; save and exit for Slurm time limits |

---

## Slurm

`slurm_example.sh` is a ready-to-use launcher that handles container setup,
dependency installation, and multi-node coordination. Edit the default config
section inside the script (paths, partition, account) for your cluster, then:

```bash
# LTX-2 single node
./slurm_example.sh --model ltx2 \
    --model-path /path/to/ltx-2-19b-dev.safetensors \
    --text-encoder-path /path/to/gemma-3-12b-it

# Wan multi-node
./slurm_example.sh --model wan --model-path /path/to/Wan2.2-TI2V-5B --nodes 2

# Dry run (print sbatch script without submitting)
./slurm_example.sh --model wan --model-path /path/to/Wan2.2-TI2V-5B --dry-run

# With config overrides
./slurm_example.sh --model wan --model-path /path/to/Wan2.2-TI2V-5B \
    optimization.steps=5000 optimization.learning_rate=1e-5
```

### Time-Limit-Aware Training

For long runs across multiple Slurm jobs:

```bash
./slurm_example.sh --model wan --model-path /path/to/Wan2.2-TI2V-5B \
    --time 04:00:00 \
    checkpoints.must_save_by=230 \
    distillation.resume_from_checkpoint=latest
```

The trainer saves a checkpoint after 230 minutes, and subsequent jobs resume
automatically.

---

## Outputs

```text
outputs/
├── model_weights_final.safetensors   # Inference-ready weights
├── checkpoints/
│   ├── step_000500/                  # Training state (FSDP shards, optimizer, ...)
│   └── step_001000/
└── validation/
    ├── step_000500/
    │   ├── video_000.mp4
    │   └── video_001.mp4
    └── step_001000/
```

---

## Adding a New Model

1. Create `src/models/<name>/` with `loader.py`, `adapter.py`, `pipeline.py`
2. Register with `@register_backend("name")` in `src/models/__init__.py`
3. Add dependency extra in `pyproject.toml`
4. Add import guard in `src/models/_deps.py`
5. Create config YAML (with `defaults: default.yaml`) and FSDP config
