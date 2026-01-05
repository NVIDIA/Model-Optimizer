# Training Guide

This guide covers how to run training jobs, from basic single-GPU training to advanced distributed setups and automatic model uploads.

## ⚡ Basic Training (Single GPU)

After preprocessing your dataset and preparing a configuration file, you can start training using the trainer script:

```bash
python scripts/train.py <PATH_TO_CONFIG_YAML_FILE>
```

The trainer will:
1. **Load your configuration** and validate all parameters
2. **Initialize models** and apply optimizations
3. **Run the training loop** with progress tracking
4. **Generate validation videos** (if configured)
5. **Save the trained weights** in your output directory

### Output Files

**For LoRA training:**
- `lora_weights.safetensors` - Main LoRA weights file
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

**For full model fine-tuning:**
- `model_weights.safetensors` - Full model weights
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

## 🖥️ Distributed / Multi-GPU Training

We use [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) directly for multi-GPU DDP and FSDP.

### Configure Accelerate

Run the interactive wizard once to set up your environment (DDP / FSDP, GPU count, etc.):

```bash
accelerate config
```

This stores your preferences in `~/.cache/huggingface/accelerate/default_config.yaml`.

### Use the provided Accelerate configs (recommended)

We include ready-to-use Accelerate config files in `configs/accelerate/`:

- [ddp.yaml](../configs/accelerate/ddp.yaml) — Standard DDP
- [ddp_compile.yaml](../configs/accelerate/ddp_compile.yaml) — DDP with `torch.compile` (Inductor)
- [fsdp.yaml](../configs/accelerate/fsdp.yaml) — Standard FSDP (auto-wraps `LTXVideoTransformerBlock`)
- [fsdp_compile.yaml](../configs/accelerate/fsdp_compile.yaml) — FSDP with `torch.compile` (Inductor)

Launch with a specific config using `--config_file`:

```bash
# DDP (2 GPUs shown as example)
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file configs/accelerate/ddp.yaml \
  scripts/train.py configs/ltxv_2b_lora.yaml

# DDP + torch.compile
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file configs/accelerate/ddp_compile.yaml \
  scripts/train.py configs/ltxv_2b_lora.yaml

# FSDP (4 GPUs shown as example)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file configs/accelerate/fsdp.yaml \
  scripts/train.py configs/ltxv_2b_full.yaml

# FSDP + torch.compile
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file configs/accelerate/fsdp_compile.yaml \
  scripts/train.py configs/ltxv_2b_full.yaml
```

Notes:

- The number of processes is taken from the Accelerate config (`num_processes`). You can override this with
  `--num_processes X` or restrict GPUs with `CUDA_VISIBLE_DEVICES` or edit the YAML.
- The compile variants enable `torch.compile` with the Inductor backend via Accelerate's `dynamo_config`.
- FSDP configs auto-wrap the transformer blocks (`fsdp_transformer_layer_cls_to_wrap: LTXVideoTransformerBlock`).

### Launch with your default Accelerate config

Use `accelerate launch` to run distributed jobs. Pass your training YAML to `scripts/train.py`:
If you prefer to use your default Accelerate profile, you can omit `--config_file`:

```bash
# Use settings from your default accelerate config
accelerate launch scripts/train.py configs/ltxv_2b_full.yaml

# Override number of processes on the fly (e.g., 2 GPUs)
accelerate launch --num_processes 2 scripts/train.py configs/ltxv_2b_lora.yaml

# Select specific GPUs
CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/train.py configs/ltxv_2b_lora.yaml
```

> Tip: You can disable the in-terminal progress bars with `--disable_progress_bars` flag in the trainer CLI if desired.

### Benefits of Distributed Training

- **Faster training**: Distribute workload across multiple GPUs
- **Larger effective batch sizes**: Combine gradients from multiple GPUs
- **Memory efficiency**: Each GPU handles a portion of the batch

> [!NOTE]
> Distributed training requires that all GPUs have sufficient memory for the model and batch size. The effective batch size becomes `batch_size × num_processes`.

## 🤗 Pushing Models to Hugging Face Hub

You can automatically push your trained models to the Hugging Face Hub by adding the following to your configuration YAML:

```yaml
hub:
  push_to_hub: true
  hub_model_id: "your-username/your-model-name"  # Your HF username and desired repo name
```

### Prerequisites

Before pushing, make sure you:

1. **Have a Hugging Face account** - Sign up at [huggingface.co](https://huggingface.co)
2. **Are logged in** via `huggingface-cli login` or have set the `HUGGING_FACE_HUB_TOKEN` environment variable
3. **Have write access** to the specified repository (it will be created if it doesn't exist)

### Login Options

**Option 1: Interactive login**
```bash
huggingface-cli login
```

**Option 2: Environment variable**
```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### What Gets Uploaded

The trainer will automatically:

- **Create a model card** with training details and sample outputs
- **Upload model weights**
- **Push sample videos as GIFs** in the model card
- **Include training configuration and prompts**

### Repository Structure

Your Hub repository will contain:
```
your-repo/
├── README.md                    # Auto-generated model card
├── lora_weights.safetensors     # Saved weights file
├── training_config.yaml        # Training configuration
└── sample_videos/              # Validation samples as GIFs
    ├── sample_001.gif
    └── sample_002.gif
```

## 🚀 Next Steps

After training completes:

- **Test your model** with validation prompts
- **Share your results** by pushing to Hugging Face Hub
- **Iterate and improve** based on validation results

## 💡 Tips for Successful Training

- **Start small**: Begin a small dataset and with a few hundred steps to verify everything works
- **Monitor validation**: Keep an eye on validation samples to catch overfitting
- **Adjust learning rate**: Lower learning rates often produce better results
- **Use gradient checkpointing**: Essential for LTXV 13B training on consumer GPUs
- **Save checkpoints**: Regular checkpoints help recover from interruptions

## Need Help?

If you encounter issues during training, see the [Troubleshooting Guide](troubleshooting.md).
