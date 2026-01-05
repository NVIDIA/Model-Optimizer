# Training Modes Guide

The trainer supports a few training modes, each suited for different use cases and requirements.

## 🎯 Standard LoRA Training

Standard LoRA (Low-Rank Adaptation) training fine-tunes the model by adding small, trainable adapter layers while keeping the base model frozen. This approach:

- **Requires significantly less memory and compute** than full fine-tuning
- **Produces small, portable weight files** (typically a few hundred MB)
- **Is ideal for learning specific styles, effects, or concepts**
- **Can be easily combined with other LoRAs** during inference

Configure standard LoRA training with:
```yaml
model:
  training_mode: "lora"
conditioning:
  mode: "none"
```

**Example configuration files:**
- 📄 [LTXV 13B Standard LoRA Training](../configs/ltxv_13b_lora_cakeify.yaml)
- 📄 [LTXV 2B Standard LoRA Training](../configs/ltxv_2b_lora.yaml)
- 📄 [LTXV 2B LoRA Training (Low VRAM)](../configs/ltxv_2b_lora_low_vram.yaml) - Optimized for GPUs with 24GB VRAM

**When to use Standard LoRA:**
- Learning specific concepts, styles or visual effects
- Limited GPU memory
- Want to create reusable, combinable models
- Experimenting with different concepts quickly

## 🔥 Full Model Fine-tuning

Full model fine-tuning updates all parameters of the base model, providing maximum flexibility but
requiring substantial computational resources and larger training datasets:

- **Offers the highest potential quality and capability improvements**
- **Requires significant GPU memory** (typically 40GB+ for larger models)
- **Produces large checkpoint files** (several GB)
- **Best for major model adaptations** or when LoRA limitations are reached

Configure full fine-tuning with:
```yaml
model:
  training_mode: "full"
conditioning:
  mode: "none"
```

**Example configuration file:**
- 📄 [LTXV 2B Full Model Fine-tuning](../configs/ltxv_2b_full.yaml)

**When to use Full Fine-tuning:**
- Need maximum model adaptation capability
- Have access to high-end GPUs (40GB+ VRAM)
- LoRA training doesn't achieve desired results
- Creating a specialized model for production use

## 🔄 In-Context LoRA (IC-LoRA) Training

IC-LoRA is a specialized training mode for video-to-video transformations.
Unlike standard training modes that learn from individual videos, IC-LoRA learns transformations from pairs of videos.
IC-LoRA enables a wide range of advanced video-to-video applications, such as:

- **Control adapters** (e.g., Depth, Pose): Learn to map from a control signal (like a depth map or pose skeleton) to a target video, enabling precise control over the generated output.
- **Video deblurring**: Train the model to transform blurry input videos into sharp, high-quality outputs.
- **Style transfer**: Apply the style of a reference video to a target video sequence.
- **Colorization**: Convert grayscale reference videos into colorized outputs.
- **Restoration and enhancement**: Denoise, upscale, or restore old or degraded videos using paired clean references.

By providing paired reference and target videos, IC-LoRA can learn complex transformations that go far beyond simple caption-based conditioning.

IC-LoRA training fundamentally differs from standard LoRA and full fine-tuning:

- **Reference videos** provide clean, unnoised conditioning input showing the "before" state
- **Target videos** are noised during training and represent the desired "after" state
- **The model learns transformations** from reference videos to target videos
- **Loss is applied only to the target portion**, not the reference
- **Training and inference time increase significantly** due to the doubled sequence length from concatenating reference and target video tokens

To enable IC-LoRA training, configure your YAML file with:

```yaml
model:
  training_mode: "lora"
conditioning:
  mode: "reference_video"
  reference_latents_dir: "reference_latents"  # Directory name for reference video latents
```

**Example configuration file:**
- 📄 [LTXV 13B IC-LoRA Training](../configs/ltxv_13b_ic_lora.yaml) - Basic video-to-video transformation training

### Dataset Requirements for IC-LoRA

- Your dataset must contain **paired videos** where each target video has a corresponding reference video
- Reference and target videos must have **identical resolution and length**
- Both reference and target videos should be **preprocessed together** using the same resolution buckets

### Generating Reference Videos

We provide an example script to generate Canny reference videos for a given dataset.
Note that it takes as an input a JSON file as the dataset configuration (eg. output of `caption_videos.py`), and edits the json to include the names of generated reference videos.

```bash
python scripts/compute_condition.py scenes_output_dir/ \
    --output scenes_output_dir/captions.json
```

To compute a different condition, simply implement the function `compute_condition()` inside this script.

### Configuration Requirements for IC-LoRA

- You must provide `reference_videos` in your validation configuration when using IC-LoRA training
- The number of reference videos must match the number of validation prompts

Example validation configuration for IC-LoRA:
```yaml
validation:
  prompts:
    - "First prompt"
    - "Second prompt"
  reference_videos:
    - "path/to/reference1.mp4"
    - "path/to/reference2.mp4"
```

**When to use IC-LoRA:**
- Learning video-to-video transformations
- Creating control adapters (e.g., Canny edge → video)
- Implementing style transfer effects
- Building conditional generation models

### Pretrained IC-LoRA Models

To help you get started with IC-LoRA training and understand the capabilities, we provide several pretrained control models:

**🎨 LTXV Control Adapters:**
- **[Depth Map Control](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7)** - Generate videos from depth maps
- **[Human Pose Control](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7)** - Generate videos from pose skeletons
- **[Canny Edge Control](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7)** - Generate videos from Canny edge maps

These models demonstrate the power of IC-LoRA for creating precise control over video generation.
You can use them directly for inference or as inspiration for training your own control adapters.

## 🚀 Next Steps

Once you've chosen your training mode:

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Configure your training parameters in [Configuration Reference](configuration-reference.md)
- Start training with the [Training Guide](training-guide.md)
