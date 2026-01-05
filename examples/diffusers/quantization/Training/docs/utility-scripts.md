# Utility Scripts Reference

This guide covers the various utility scripts available for preprocessing, conversion, and debugging tasks.

## 🎬 Dataset Processing Scripts

### Video Scene Splitting

The `scripts/split_scenes.py` script automatically splits long videos into shorter, coherent scenes.

```bash
# Basic scene splitting
python scripts/split_scenes.py input.mp4 output_dir/ --filter-shorter-than 5s
```

**Key features:**
- **Automatic scene detection**: Uses PySceneDetect for intelligent splitting
- **Multiple algorithms**: Content-based, adaptive, threshold, and histogram detection
- **Filtering options**: Remove scenes shorter than specified duration
- **Customizable parameters**: Thresholds, window sizes, and detection modes

**Common options:**
```bash
# See all available options
python scripts/split_scenes.py --help

# Use adaptive detection with custom threshold
python scripts/split_scenes.py video.mp4 scenes/ --detector adaptive --threshold 30.0

# Limit to maximum number of scenes
python scripts/split_scenes.py video.mp4 scenes/ --max-scenes 50
```

### Automatic Video Captioning

The `scripts/caption_videos.py` script generates captions for videos using vision-language models.

```bash
# Generate captions for all videos in a directory
python scripts/caption_videos.py scenes_output_dir/ --output captions.json

# Use 8-bit quantization to reduce VRAM usage
python scripts/caption_videos.py scenes_output_dir/ --output captions.json --use-8bit
```

**Key features:**
- **VLM-powered**: Uses Qwen2.5-VL for high-quality captions
- **Memory optimization**: 8-bit quantization option for limited VRAM
- **Batch processing**: Processes entire directories of videos
- **JSON output**: Creates structured dataset files

### Dataset Preprocessing

The `scripts/preprocess_dataset.py` script processes videos and caches latents for training.

```bash
# Basic preprocessing
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path"

# With video decoding for verification
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --decode-videos
```

You can provide multiple resolution buckets, separated by `;`:

```bash
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25;512x512x49" \
    --caption-column "caption" \
    --video-column "media_path"
```

> Note: When training with multiple resolution buckets, set your training `data.batch_size` to `1`.

For detailed usage, see the [Dataset Preparation Guide](dataset-preparation.md).

### Reference Video Generation

The `scripts/compute_condition.py` script provides a template for creating reference videos needed for IC-LoRA training.
This specific example generates reference videos using Canny edge detection.

> **Note:** You can edit the `scripts/compute_condition.py` script to generate other types of reference videos for IC-LoRA training.
> For example, you might implement colorization, depth maps, segmentation masks, or any custom video transformation by modifying the `compute_condition()` function.
> This flexibility allows you to tailor the conditioning signal to your specific research or creative needs.

```bash
# Generate Canny edge reference videos
python scripts/compute_condition.py videos_dir/ --output dataset.json
```

**Key features:**
- **Canny edge detection**: Creates edge-based reference videos
- **In-place editing**: Updates existing dataset JSON files
- **Customizable**: Modify the `compute_condition()` function for different conditions

## 🔍 Debugging and Verification Scripts

### Latents Decoding

The `scripts/decode_latents.py` script decodes precomputed video latents back into video files for visual inspection.

```bash
# Basic usage
python scripts/decode_latents.py /path/to/latents/dir --output-dir /path/to/output
```

**The script will:**
1. **Load the VAE model** from the specified path
2. **Process all `.pt` latent files** in the input directory
3. **Decode each latent** back into a video using the VAE
4. **Save resulting videos** as MP4 files in the output directory

**When to use:**
- **Verify preprocessing quality**: Check that your videos were encoded correctly
- **Debug training data**: Visualize what the model actually sees during training
- **Quality assessment**: Ensure latent encoding preserves important visual details

**Example workflow:**
```bash
# After preprocessing your dataset
python scripts/preprocess_dataset.py dataset.json --resolution-buckets "768x768x25"

# Decode some latents to verify quality
python scripts/decode_latents.py dataset/.precomputed/latents --output-dir decoded_samples

# Review the decoded videos to ensure quality
ls decoded_samples/
```

## 🚀 Training Scripts

### Basic and Distributed Training

Use `scripts/train.py` for both single GPU and multi-GPU runs. For distributed/FSDP runs, launch via Accelerate:

```bash
# Single-GPU training
python scripts/train.py config.yaml

# Multi-GPU (uses your accelerate config)
accelerate launch scripts/train.py config.yaml

# Override number of processes
accelerate launch --num_processes 4 scripts/train.py config.yaml
```

For detailed usage, see the [Training Guide](training-guide.md).

## 💡 Tips for Using Utility Scripts

- **Start with `--help`**: Always check available options for each script
- **Test on small datasets**: Verify workflows with a few files before processing large datasets
- **Use decode verification**: Always decode a few samples to verify preprocessing quality
- **Monitor VRAM usage**: Use `--use-8bit` flags when running into memory issues
- **Keep backups**: Make copies of important dataset files before running conversion scripts
