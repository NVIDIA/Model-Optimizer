---
tags:
- ltx-video
- image-to-video
pinned: true
language:
- en
license: other
pipeline_tag: text-to-video
library_name: diffusers
---

# {model_name}

This is a fine-tuned version of [`{base_model}`]({base_model_link}) trained on custom data.

## Model Details

- **Base Model:** [`{base_model}`]({base_model_link})
- **Training Type:** {training_type}
- **Training Steps:** {training_steps}
- **Learning Rate:** {learning_rate}
- **Batch Size:** {batch_size}

## Sample Outputs

| | | | |
|:---:|:---:|:---:|:---:|
{sample_grid}

## Usage

This model is designed to be used with the LTXV (Lightricks Text-to-Video) pipeline.

### 🔌 Using Trained LoRAs in ComfyUI
In order to use the trained lora in ComfyUI, follow these steps:
1. Copy your trained LoRA checkpoint (`.safetensors` file) to the `models/loras` folder in your ComfyUI installation.
2. In your ComfyUI workflow:
   - Add the "Load LoRA" node to choose your LoRA file
   - Connect it to the "Load Checkpoint" node to apply the LoRA to the base model

You can find reference Text-to-Video (T2V) and Image-to-Video (I2V) workflows in the [official LTXV ComfyUI repository](https://github.com/Lightricks/ComfyUI-LTXVideo).

### Example Prompts

{validation_prompts}


This model inherits the license of the base model ([`{base_model}`]({base_model_link})).

## Acknowledgments

- Base model by [Lightricks](https://huggingface.co/Lightricks)
- Training infrastructure: [LTX-Video-Trainer](https://github.com/Lightricks/ltx-video-trainer)
