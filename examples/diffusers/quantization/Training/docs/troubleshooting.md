# Troubleshooting Guide

This guide covers common issues and solutions when training with LTX-Video-Trainer.

## 🔧 VRAM and Memory Issues

Memory management is crucial for successful training, especially with larger models like LTXV 13B.

### LTXV 13B Memory Requirements

When training with the LTXV 13B model, you **must** enable gradient checkpointing:

```yaml
optimization:
  enable_gradient_checkpointing: true  # Required for LTXV 13B
```

> **Note:** Gradient checkpointing trades training speed for memory savings. It's essential for training LTXV 13B on consumer GPUs.

### Memory Optimization Techniques

#### 1. Enable 8-bit Text Encoder

Load the text encoder in 8-bit precision to save GPU memory during training:

```yaml
quantization:
  load_text_encoder_in_8bit: true
```

This setting is also available in all data preparation scripts:

```bash
# Dataset preprocessing with 8-bit text encoder
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --load_text_encoder_in_8bit

# Caption generation with 8-bit quantization
python scripts/caption_videos.py videos/ \
    --output dataset.json \
    --use-8bit
```

#### 2. Reduce Batch Size

Lower the batch size if you encounter out-of-memory errors:

```yaml
data:
  batch_size: 1  # Start with 1 and increase gradually
```

#### 3. Use Lower Resolution

Reduce spatial or temporal dimensions to save memory:

```bash
# Smaller spatial resolution
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "512x512x49"

# Fewer frames
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25"  # 25 frames instead of 49
```

#### 4. Memory-Optimized Configuration

Use the low VRAM configuration as a starting point:

```yaml
# Based on configs/ltxv_2b_lora_low_vram.yaml
model_source: "LTXV_2B_0.9.6_DEV"

data:
  batch_size: 1

optimization:
  enable_gradient_checkpointing: true
  optimizer_type: "adamw8bit"  # 8-bit optimizer

quantization:
  load_text_encoder_in_8bit: true
```

### Memory Usage Guidelines

**Sequence Length Calculation:**
```
sequence_length = (H/32) * (W/32) * ((F-1)/8 + 1)
```

Where:
- H = Height, W = Width, F = Number of frames
- 32 = VAE spatial downsampling factor
- 8 = VAE temporal downsampling factor

**Examples:**
- `768x768x25`: sequence_length = 24 × 24 × 4 = 2,304
- `768x448x89`: sequence_length = 24 × 14 × 12 = 4,032
- `512x512x49`: sequence_length = 16 × 16 × 7 = 1,792

**Memory Requirements by Model:**
- **LTXV 2B**: ~16-40GB VRAM (depending on resolution and batch size)
- **LTXV 13B**: ~40GB+ VRAM (requires gradient checkpointing)
- **LTX2**: Similar to LTXV 13B (~40GB+ VRAM, requires gradient checkpointing)

---

## ⚠️ Common Usage Issues

### Issue: "No module named 'ltxv_trainer'" Error

**Solution:**
Ensure you're in the correct environment and have installed dependencies:

```bash
# Reinstall if needed
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Issue: Slow Training Speed

**Optimizations:**

1. **Disable gradient checkpointing** (if you have enough VRAM):
   ```yaml
   optimization:
     enable_gradient_checkpointing: false
   ```

2. **Increase batch size** (if memory allows):
   ```yaml
   data:
     batch_size: 2  # Or higher
   ```

3. **Enable Inductor via Accelerate** (optional):
   Configure `torch.compile` with Inductor globally using `accelerate config` instead of per-script flags.

### Issue: Poor Quality Validation Outputs

**Solutions:**

1. **Use Image-to-Video Validation Instead of Text-to-Video:**
   - For more reliable validation, use image-to-video (first-frame conditioning) rather than text-to-video. This is supported via the `images` field in your validation config (see `ValidationConfig` in `config.py`):
     ```yaml
     validation:
       prompts:
         - "a professional portrait video of a person with blurry bokeh background"
       images:
         - "/path/to/first_frame.png"  # One image per prompt
     ```
   - This approach provides a stronger conditioning signal and typically results in higher quality validation outputs.

2. **Note on Diffusers Inference Quality:**
   - The default inference pipeline in 🤗 Diffusers is suboptimal for LTXV models: it does **not** include STG (Spatio-Temporal Guidance) or other inference-time tricks that improve video quality.
   - For best results, use validation videos to track training progress, but for actual quality testing, export your LoRA and test it in ComfyUI using the recommended workflow:
     👉 [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)

3. **Other Tips:**
   - **Check caption quality:** Review and, if needed, manually edit captions for accuracy.
   - **Adjust LoRA rank:** Try higher values for `lora.rank` (e.g., 32, 64, 128) for more capacity:
     ```yaml
     lora:
       rank: 64
     ```
   - **Increase training steps:** Train longer if needed:
     ```yaml
     optimization:
       steps: 2000
     ```
---

## 🔍 Debugging Tools

### Monitor GPU Memory Usage

Track memory usage during training:

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Log memory usage to file
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=5 > memory_log.csv
```

### Verify Preprocessed Data

Decode latents to check to visualize the pre-processed videos:

```bash
python scripts/decode_latents.py dataset/.precomputed/latents \
    --output-dir debug_output
```

Compare decoded videos with originals to ensure quality.

---

## 💡 Best Practices

### Before Training

- [ ] Test preprocessing with a small subset first
- [ ] Verify all video files are accessible
- [ ] Check available GPU memory
- [ ] Review configuration against hardware capabilities

### During Training

- [ ] Monitor GPU memory usage
- [ ] Check loss convergence regularly
- [ ] Review validation samples periodically
- [ ] Save checkpoints frequently

### After Training

- [ ] Test trained model with diverse prompts
- [ ] Document training parameters and results
- [ ] Archive training data and configs

## 🆘 Getting Help

If you're still experiencing issues:

1. **Check logs:** Review console output and log files for error details
2. **Search issues:** Look through GitHub issues for similar problems
3. **Provide details:** When reporting issues, include:
   - Hardware specifications (GPU model, VRAM)
   - Configuration file used
   - Complete error message
   - Steps to reproduce the issue
---

## 🤝 Join the Community

Have questions, want to share your results, or need real-time help?
Join our [community Discord server](https://discord.gg/Mn8BRgUKKy) to connect with other users and the development team!

- Get troubleshooting help
- Share your training results and workflows
- Stay up to date with announcements and updates

We look forward to seeing you there!
