# Configuration Reference

The trainer uses structured Pydantic models for configuration, making it easy to customize training parameters.
This guide covers all available configuration options and their usage.

## 📋 Overview

The main configuration class is [`LtxvTrainerConfig`](../src/ltxv_trainer/config.py), which includes the following sub-configurations:

- **ModelConfig**: Base model and training mode settings
- **LoraConfig**: LoRA training parameters
- **ConditioningConfig**: Video conditioning settings (reference videos, first frame conditioning)
- **OptimizationConfig**: Learning rate, batch sizes, and scheduler settings
- **AccelerationConfig**: Mixed precision and other optimization settings
- **DataConfig**: Data loading parameters
- **ValidationConfig**: Validation and inference settings
- **CheckpointsConfig**: Checkpoint saving frequency and retention settings
- **HubConfig**: Hugging Face Hub integration settings
- **FlowMatchingConfig**: Timestep sampling parameters

## 📄 Example Configuration Files

Check out our example configurations in the `configs` directory. You can use these as templates for your training runs:

- 📄 [LTXV 2B Full Model Fine-tuning Example](../configs/ltxv_2b_full.yaml)
- 📄 [LTXV 2B LoRA training Example](../configs/ltxv_2b_lora.yaml)
- 📄 [LTXV 13B LoRA training Example](../configs/ltxv_13b_lora_cakeify.yaml)
- 📄 [LTXV 2B LoRA Fine-tuning Example (Low VRAM)](../configs/ltxv_2b_lora_low_vram.yaml) - Optimized for GPUs with 24GB VRAM
- 📄 [LTXV 13B IC-LoRA Training Example](../configs/ltxv_13b_ic_lora.yaml) - Video-to-video transformation training

## ⚙️ Configuration Sections

### ModelConfig

Controls the base model and training mode settings.

```yaml
model:
  model_source: "LTXV_13B_097_DEV"  # Model version, HuggingFace repo, or local path
  training_mode: "lora"             # "lora" or "full"
  load_checkpoint: null             # Path to checkpoint file/directory to resume from
```

**Key parameters:**
- `model_source`: Model to use - can be a model version (see [model_loader.py](../src/ltxv_trainer/model_loader.py)), HuggingFace repo ID, or local path
- `training_mode`: Training approach - either `"lora"` for LoRA training or `"full"` for full-rank model fine-tuning
- `load_checkpoint`: Optional path to a checkpoint to resume the training from

#### Available Model Versions

**LTXV Models (using T5 Text Encoder):**
- `"LTXV_2B_0.9.0"` - 2B model, publicly available
- `"LTXV_2B_0.9.1"` - 2B model, publicly available
- `"LTXV_2B_0.9.5"` - 2B model, publicly available
- `"LTXV_2B_0.9.6_DEV"` - 2B development model
- `"LTXV_2B_0.9.6_DISTILLED"` - 2B distilled model
- `"LTXV_13B_097_DEV"` - 13B development model (default)
- `"LTXV_13B_097_DISTILLED"` - 13B distilled model

**LTX2 Models (using Gemma3 Text Encoder):**
- `"LTX2"` - Next-generation 13B model using Gemma3 as a text encoder

#### Text Encoder Differences

The trainer automatically selects the appropriate text encoder based on your model choice:

- **LTXV models** use **T5 text encoder** (4096 embedding dimensions)
- **LTX2 models** use **Gemma3 text encoder** (3840 embedding dimensions)

> **Note:** LTX2 models are currently local-only and require you to have the model weights and Gemma text encoder installed locally.
> LTXV models can be downloaded automatically from HuggingFace.

### LoraConfig

LoRA-specific fine-tuning parameters (only used when `training_mode: "lora"`).

```yaml
lora:
  rank: 64                       # LoRA rank (higher = more parameters, more flexibility)
  alpha: 64                      # LoRA alpha scaling factor
  dropout: 0.0                   # Dropout probability (0.0-1.0)
  target_modules:                # Modules to apply LoRA to
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"
```

**Key parameters:**
- `rank`: LoRA rank - higher values mean more trainable parameters and potentially more flexibility (typical range: 16-128)
- `alpha`: Alpha scaling factor - usually set equal to rank
- `dropout`: Dropout probability for regularization
- `target_modules`: List of transformer modules (can include wildchar characters) to apply LoRA adapters to.

### ConditioningConfig

Video conditioning settings for specialized training modes.

```yaml
conditioning:
  mode: "none"                            # "none" or "reference_video"
  first_frame_conditioning_p: 0.1         # Probability of first-frame conditioning
  reference_latents_dir: "reference_latents"  # Directory for reference video latents
```

**Key parameters:**
- `mode`: Conditioning type - `"none"` for standard training, `"reference_video"` for IC-LoRA
- `first_frame_conditioning_p`: Probability of using first frame as conditioning (0.0-1.0)
- `reference_latents_dir`: Directory name for reference video latents (IC-LoRA only)

### OptimizationConfig

Training optimization parameters including learning rates, batch sizes, and schedulers.

```yaml
optimization:
  learning_rate: 1e-4              # Learning rate
  steps: 3000                      # Total training steps
  batch_size: 2                    # Batch size per GPU
  gradient_accumulation_steps: 1   # Steps to accumulate gradients
  max_grad_norm: 1.0              # Gradient clipping threshold
  optimizer_type: "adamw"         # "adamw" or "adamw8bit"
  scheduler_type: "linear"        # Scheduler type
  scheduler_params: {}            # Additional scheduler parameters
  enable_gradient_checkpointing: false  # Memory optimization at cost of speed
```

**Key parameters:**
- `learning_rate`: Learning rate for optimization (typical range: 1e-5 to 1e-3)
- `steps`: Total number of training steps
- `batch_size`: Batch size per GPU (reduce if running out of memory)
- `gradient_accumulation_steps`: Accumulate gradients over multiple steps (increases effective batch size)
- `scheduler_type`: Learning rate scheduler - `"constant"`, `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`
- `enable_gradient_checkpointing`: Trade training speed for GPU memory savings (required for LTXV 13B)

### AccelerationConfig

Hardware acceleration and compute optimization settings.

```yaml
acceleration:
  mixed_precision_mode: "bf16"     # "no", "fp16", or "bf16"
  quantization: null               # Quantization options
  load_text_encoder_in_8bit: false  # Load text encoder in 8-bit
```

**Key parameters:**
- `mixed_precision_mode`: Precision mode - `"bf16"` recommended for modern GPUs, `"fp16"` for older ones
- `quantization`: Quantization precision for model weights.
  Options include `null` (no quantization), `"int8-quanto"`, `"int4-quanto"`, `"int2-quanto"`, `"fp8-quanto"`, and `"fp8uz-quanto"`.
  Use quantization to reduce memory usage, especially for large models or limited hardware.
- `load_text_encoder_in_8bit`: Load the text encoder in 8-bit to save GPU memory

### DataConfig

Data loading and processing configuration.

```yaml
data:
  preprocessed_data_root: "path/to/preprocessed/data"  # Path to precomputed dataset directory
  num_dataloader_workers: 2                           # Background data loading workers
```

**Key parameters:**
- `preprocessed_data_root`: Path to your preprocessed dataset (contains the `.precomputed` directory)
- `num_dataloader_workers`: Number of parallel data loading processes (0 = synchronous loading)

### ValidationConfig

Validation and inference settings for monitoring training progress.

```yaml
validation:
  prompts:                        # Validation prompts
    - "A cat playing with a ball"
    - "A dog running in a field"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null                    # Optional list of image paths for image-to-video
  reference_videos: null          # Reference video paths (IC-LoRA only)
  video_dims: [704, 480, 161]     # Video dimensions [width, height, frames]
  seed: 42                        # Random seed for reproducibility
  inference_steps: 50             # Number of inference steps
  interval: 100                   # Steps between validation runs
  videos_per_prompt: 1            # Videos generated per prompt
  guidance_scale: 3.0             # CFG guidance strength
```

**Key parameters:**
- `prompts`: List of text prompts for validation video generation
- `images`: List of image paths for image-to-video validation (must match number of prompts)
- `interval`: Steps between validation runs (set to `null` to disable)
- `inference_steps`: Number of denoising steps for validation videos
- `video_dims`: Output video dimensions `[width, height, frames]`
- `reference_videos`: List of paths to reference videos. Required for IC-LoRA validation (must match number of prompts)

### CheckpointsConfig

Model checkpointing configuration.

```yaml
checkpoints:
  interval: null      # Steps between checkpoint saves (null = disabled)
  keep_last_n: 5      # Number of recent checkpoints to retain
```

**Key parameters:**
- `interval`: Steps between intermediate checkpoint saves (set to `null` to disable checkpoint saving)
- `keep_last_n`: Number of most recent checkpoints to keep (older ones are deleted)

### HubConfig

Hugging Face Hub integration for automatic model uploads.

```yaml
hub:
  push_to_hub: false                    # Enable Hub uploading
  hub_model_id: "username/model-name"   # Hub repository ID
```

**Key parameters:**
- `push_to_hub`: Whether to automatically push trained models to Hugging Face Hub
- `hub_model_id`: Repository ID in format `"username/repository-name"`

### FlowMatchingConfig

Flow matching training configuration for timestep sampling.

```yaml
flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"  # Timestep sampling strategy
  timestep_sampling_params: {}                    # Additional sampling parameters
```

**Key parameters:**
- `timestep_sampling_mode`: Sampling strategy - `"uniform"` or `"shifted_logit_normal"`
- `timestep_sampling_params`: Additional parameters for the sampling strategy

## 🚀 Next Steps

Once you've configured your training parameters:

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Choose your training approach in [Training Modes](training-modes.md)
- Start training with the [Training Guide](training-guide.md)
