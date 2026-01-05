# Dataset Preparation Guide

This guide covers the complete workflow for preparing and preprocessing your dataset for training.


## 📋 Overview

The general dataset preparation workflow is:

1. **(Optional)** Split long videos into scenes using `split_scenes.py`
2. **(Optional)** Generate captions for your videos using `caption_videos.py`
3. **Preprocess your dataset** using `preprocess_dataset.py` to compute and cache video latents and text embeddings
   This stage significantly accelerates training and reduces GPU memory usage.
4. **Run the trainer** with your preprocessed dataset

## 🎬 Step 1: Split Scenes

If you're starting with long-form videos (e.g., downloaded from YouTube), you should first split them into shorter, coherent scenes.
We provide a utility script to automate this process:

```bash
# Split a long video into scenes
python scripts/split_scenes.py input.mp4 scenes_output_dir/ \
    --filter-shorter-than 5s
```

This will create multiple video clips in `scenes_output_dir`.
These clips will be the input for the captioning step, if you choose to use it.

The script supports many configuration options for scene detection (detector algorithms, thresholds, minimum scene lengths, etc.).
To see all available options, run:

```bash
python scripts/split_scenes.py --help
```

## 📝 Step 2: Caption Videos

If your dataset doesn't include captions, you can automatically generate them using vision-language models (VLM).
Use the directory containing your video clips (either from step 1, or your own clips):

```bash
# Generate captions for all videos in the scenes directory
python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

If you're running into VRAM issues, try enabling 8-bit quantization to reduce memory usage:

```bash
python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json \
    --use-8bit
```

This will create a `dataset.json` file which contains video paths and their captions.
This JSON file will be used as input for the data preprocessing step.

By default, the script uses the Qwen2.5-VL model for media captioning.

> [!NOTE]
> The automatically generated captions may contain inaccuracies or hallucinated content, as VLMs can sometimes misinterpret visual information.
> We recommend reviewing and correcting the generated captions in your `dataset.json` file before proceeding to preprocessing, as accurate captions are important for effective training results.


## ⚡ Step 3: Dataset Preprocessing

This step preprocesses your video dataset by:

1. Resizing and cropping videos to fit specified resolution buckets
2. Computing and caching video latent representations
3. Computing and caching text embeddings for captions

Using the dataset.json file generated in step 2:

```bash
# Preprocess the dataset using the generated dataset.json
python scripts/preprocess_dataset.py scenes_output_dir/dataset.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path"
    --model-source "LTXV_13B_097_DEV"  # Optional: specify a specific version, defaults to latest
```

### 📊 Dataset Format

The trainer supports either videos or single images.
Note that your dataset must be homogeneous - either all videos or all images, mixing is not supported.
When using images, follow the same preprocessing steps and format requirements as with videos,
simply provide image files instead of video files.

The dataset must be a CSV, JSON, or JSONL metadata file with columns for captions and video paths:

**JSON format example:**
```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "media_path": "videos/cat_playing.mp4"
  },
  {
    "caption": "A dog running in the park",
    "media_path": "videos/dog_running.mp4"
  }
]
```

**JSONL format example:**
```jsonl
{"caption": "A cat playing with a ball of yarn", "media_path": "videos/cat_playing.mp4"}
{"caption": "A dog running in the park", "media_path": "videos/dog_running.mp4"}
```

**CSV format example:**
```csv
caption,media_path
"A cat playing with a ball of yarn","videos/cat_playing.mp4"
"A dog running in the park","videos/dog_running.mp4"
```


### 📐 Resolution Buckets

Videos are organized into "buckets" of specific dimensions (width × height × frames).
Each video is assigned to the nearest matching bucket.
You can preprocess with one or multiple resolution buckets.
When training with multiple resolution buckets, you must use a batch size of 1.

The dimensions of each bucket must follow these constraints due to LTX-Video's VAE architecture:

- Spatial dimensions (width and height) must be multiples of 32
- Number of frames must be a multiple of 8 plus 1 (e.g., 25, 33, 41, 49, etc.)

**Guidelines for choosing training resolution:**

- For high-quality, detailed videos: use larger spatial dimensions (e.g. 768x448) with fewer frames (e.g. 89)
- For longer, motion-focused videos: use smaller spatial dimensions (512×512) with more frames (121)
- Memory usage increases with both spatial and temporal dimensions

**Example usage:**

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "768x768x25"
```

Multiple buckets are supported by separating entries with `;`:

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "768x768x25;512x512x49"
```

This creates a bucket with:
- 768×768 resolution
- 25 frames

**Video processing workflow:**
1. Videos are **resized** maintaining aspect ratio until either width or height matches the target (768 in this example)
2. The larger dimension is **center cropped** to match the bucket's dimensions
3. Only the **first X frames are taken** to match the bucket's frame count (25 in this example), remaining frames are ignored.

> [!NOTE]
> The sequence length processed by the transformer model can be calculated as:
>
> ```
> sequence_length = (H/32) * (W/32) * ((F-1)/8 + 1)
> ```
>
> Where:
>
> - H = Height of video's latent
> - W = Width of video's latent
> - F = Number of latent frames
> - 32 = VAE's spatial downsampling factor
> - 8 = VAE's temporal downsampling factor
>
> For example, a 768×448×89 video would have sequence length:
>
> ```
> (768/32) * (448/32) * ((89-1)/8 + 1) = 24 * 14 * 12 = 4,032
> ```
>
> Keep this in mind when choosing video dimensions, as longer sequences require more GPU memory and computation power.

> [!WARNING]
> Multiple resolution buckets are supported. However, when training with multiple buckets, you must use
> a batch size of 1 (i.e., set `optimization.batch_size: 1` in your training config).

### 📁 Output Structure

The preprocessed data is saved in a `.precomputed` directory:

```
dataset/
└── .precomputed/
    ├── latents/           # Cached video latents
    ├── conditions/        # Cached text embeddings
    └── reference_latents/ # (only for IC-LoRA training) Cached reference video latents
```

## 🔄 IC-LoRA Reference Video Preprocessing

For IC-LoRA training, you can preprocess datasets that include reference videos.
Reference videos provide clean conditioning input while target videos represent the desired transformed output.

### Dataset Format with Reference Videos

**JSON format:**
```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "media_path": "videos/cat_playing.mp4",
    "reference_path": "references/cat_playing.mp4"
  }
]
```

**JSONL format:**
```jsonl
{"caption": "A cat playing with a ball of yarn", "media_path": "videos/cat_playing.mp4", "reference_path": "references/cat_playing.mp4"}
{"caption": "A dog running in the park", "media_path": "videos/dog_running.mp4", "reference_path": "references/dog_running.mp4"}
```

**CSV format:**
```csv
caption,media_path,reference_path
"A cat playing with a ball of yarn","videos/cat_playing.mp4","references/cat_playing.mp4"
```

### Preprocessing with Reference Videos

```bash
# Using JSON dataset
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path" \
    --reference-column "reference_path"

# Using CSV dataset
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path" \
    --reference-column "reference_path"
```

This will create an additional `reference_latents/` directory containing the preprocessed reference video latents.

### Example Dataset

For reference, see our **[🎯 Canny Control Dataset](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset)** which demonstrates proper IC-LoRA dataset structure with paired videos and Canny edge maps.

### Generating Reference Videos

**Dataset Requirements for IC-LoRA:**
- Your dataset must contain paired videos where each target video has a corresponding reference video
- Reference and target videos must have *identical* resolution and length
- Both reference and target videos should be preprocessed together using the same resolution buckets

We provide an example script, [`scripts/compute_condition.py`](../scripts/compute_condition.py), to generate Canny reference videos for a given dataset, similar to the approach used in our example dataset.
This script accepts a JSON file as the dataset configuration (such as the output from `caption_videos.py`) and updates it in-place by adding the filenames of the generated reference videos.

```bash
python scripts/compute_condition.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

If you want to generate a different type of condition, just modify or replace the `compute_condition()` function within this script.

## 🎯 LoRA Trigger Words

When training a LoRA, you can specify a trigger token that will be prepended to all captions:

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "1024x576x65" \
    --id-token "TOKEN"
```

This acts as a trigger word that activates the LoRA during inference when you include the same token in your prompts.

> **Note:**
> There is no need to manually insert the trigger word (id-token) into your dataset JSON/JSONL/CSV file.
> The trigger token specified with `--id-token` is automatically prepended to each caption during preprocessing.


## 🔍 Decoding Videos for Verification

If you add the `--decode-videos` flag, the script will VAE-decode the precomputed latents and save the resulting videos in `.precomputed/decoded_videos`.
This allows you to visually inspect and verify the processed data, which is helpful for debugging and confirming that your dataset has been handled correctly.

```bash
# Preprocess dataset and decode videos for verification
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "768x768x25" \
    --decode-videos

# For IC-LoRA datasets, this will also decode reference videos
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --reference-column "reference_path" \
    --decode-videos
```

For single-frame images, the decoded latents will be saved as PNG files rather than MP4 videos.

## 🚀 Next Steps

Once your dataset is preprocessed, you can proceed to:

- Configure your training parameters in [Configuration Reference](configuration-reference.md)
- Choose your training approach in [Training Modes](training-modes.md)
- Start training with the [Training Guide](training-guide.md)
