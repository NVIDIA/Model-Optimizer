# FVD (Fréchet Video Distance) Evaluation Tool

## Overview

This tool computes the Fréchet Video Distance (FVD) between two sets of videos using a pre-trained I3D model (Kinetics-400, RGB stream). FVD is a distribution-level metric that measures how similar two collections of videos are — lower values indicate more similar distributions (0 = identical).

**Primary Use Cases:**

1. **Model Optimization Validation** — Verify that quantized/pruned video generation models maintain output quality
2. **Precision Analysis** — Compare BF16 vs INT8 vs INT4 generated video outputs
3. **Framework Comparison** — Evaluate outputs across different inference backends

## Key Components

| Script | Purpose |
|--------|---------|
| `compute_fvd.py` | Main script — loads videos, extracts I3D features, computes FVD |
| `i3d_model.py` | I3D (Inception-v1 Inflated 3D) model architecture and weight loading |

### I3D Model Details

- **Architecture**: Inception-v1 inflated to 3D ([Carreira & Zisserman, CVPR 2017](https://arxiv.org/abs/1705.07750))
- **Weights**: `rgb_imagenet.pt` from [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) (~49 MB, auto-downloaded on first run)
- **Feature dimension**: 1024 (from the final average pooling layer)
- **Input**: 16-frame clips, center-cropped to 224×224, normalized to [-1, 1]

## Installation

### 1. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv fvd_env
source fvd_env/bin/activate        # Linux/macOS
# .\fvd_env\Scripts\Activate.ps1   # Windows PowerShell
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

Note: For GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
```

## Usage Examples

### Quick Start

Compare two directories of videos:

```bash
python compute_fvd.py \
    --ref-dir /path/to/reference/videos \
    --gen-dir /path/to/generated/videos
```

The I3D weights (~49 MB) are downloaded automatically on first run and cached in `~/.cache/fvd/rgb_imagenet.pt`.

### Save Results to JSON

```bash
python compute_fvd.py \
    --ref-dir /path/to/reference/videos \
    --gen-dir /path/to/generated/videos \
    --output results.json
```

### Use a Locally Downloaded I3D Checkpoint

```bash
python compute_fvd.py \
    --ref-dir /path/to/reference/videos \
    --gen-dir /path/to/generated/videos \
    --weights ./rgb_imagenet.pt
```

### Increase Sample Count with Multiple Clips per Video

```bash
python compute_fvd.py \
    --ref-dir /path/to/reference/videos \
    --gen-dir /path/to/generated/videos \
    --clips-per-video 4 \
    --output results.json
```

### Specify Device and Batch Size

```bash
python compute_fvd.py \
    --ref-dir ./real --gen-dir ./fake \
    --device cuda \
    --batch-size 16
```

### Explicit PCA Dimension

```bash
python compute_fvd.py \
    --ref-dir ./real --gen-dir ./fake \
    --pca-dim 64 \
    --output results.json
```

## Configuration Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--ref-dir` | Directory containing reference (real) videos |
| `--gen-dir` | Directory containing generated videos |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Path to I3D weights file | Auto-downloaded `rgb_imagenet.pt` |
| `--device` | Torch device (`cuda`, `cpu`, `cuda:0`) | Auto-detected |
| `--clip-length` | Number of frames per clip | 16 |
| `--clips-per-video` | Number of clips sampled per video | 1 |
| `--batch-size` | Batch size for I3D inference | 8 |
| `--pca-dim` | PCA dimension for features (0 to disable, auto-selected when clips < 1024) | Auto |
| `--output` | Path to save JSON results | None (prints to console) |

### Supported Video Formats

`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.m4v`

Videos are discovered recursively under the specified directories.

## Expected Output

### Console Output

```text
2025-01-15 10:30:00 | INFO | Device: cuda
2025-01-15 10:30:02 | INFO | I3D model loaded from rgb_imagenet.pt (1024-dim features)
2025-01-15 10:30:02 | INFO | Reference videos: 100
2025-01-15 10:30:02 | INFO | Generated videos: 100
Loading ref: 100%|██████████| 100/100 [00:15<00:00,  6.5video/s]
Loading gen: 100%|██████████| 100/100 [00:14<00:00,  6.8video/s]
2025-01-15 10:30:32 | INFO | Total clips — ref: 100, gen: 100
Extracting ref features: 100%|██████████| 13/13 [00:08<00:00,  1.5it/s]
Extracting gen features: 100%|██████████| 13/13 [00:07<00:00,  1.6it/s]
2025-01-15 10:30:48 | INFO | FVD = 12.3456
```

### JSON Output

```json
{
  "fvd": 12.3456,
  "ref_dir": "/path/to/reference/videos",
  "gen_dir": "/path/to/generated/videos",
  "num_ref_clips": 100,
  "num_gen_clips": 100,
  "clip_length": 16,
  "clips_per_video": 1,
  "feature_dim": 1024,
  "pca_dim": null,
  "model": "I3D (Kinetics-400, 1024-dim pool)"
}
```

## Benchmark Results

### LTX-2.3 Video Generation — PTQ vs QAD (BF16 Reference)

FVD scores comparing PTQ-quantized and QAD-quantized LTX-2.3 video generation outputs against BF16 baseline, evaluated across [VBench](https://github.com/Vchitect/VBench) dimensions. Lower is better.

| Category | FVD: PTQ vs BF16 ↓ | FVD: QAD vs BF16 ↓ |
|---|---|---|
| Temporal Flickering | 31.92 | 21.97 |
| Subject Dynamic Motion | 23.44 | 16.28 |
| Multiple Objects | 35.35 | 22.47 |
| Human Action | 30.08 | 21.82 |
| Object Class | 51.51 | 26.86 |
| Color | 36.52 | 25.09 |
| Spatial Relationship | 25.07 | 18.41 |
| Scene Background | 64.92 | 35.69 |
| Appearance Style | 31.08 | 20.82 |
| Temporal Style | 23.61 | 15.85 |
| Overall Consistency | 25.03 | 18.85 |
| **Average** | **34.41** | **22.19** |

**Takeaways:**

- QAD consistently outperforms PTQ across all 11 VBench dimensions, with an average FVD of **22.19** vs **34.41** (35% lower).
- The largest gap is on **Scene Background** (64.92 vs 35.69) and **Object Class** (51.51 vs 26.86), indicating PTQ degrades spatial detail fidelity more than QAD.
- Both methods perform best on **Temporal Style** and **Subject Dynamic Motion**, suggesting temporal dynamics are more robust to quantization.

## Key Insights

- **Lower is better**: FVD = 0 means identical distributions
- **Sample count matters**: FVD estimates are noisy below ~256 clips; 2048+ clips recommended for publishable results. Use `--clips-per-video` to increase sample count.
- **PCA auto-selection**: When the number of clips is less than the feature dimension (1024), PCA is automatically applied to avoid rank-deficient covariance matrices

## Troubleshooting

### CUDA Out of Memory

**Solutions:**

- Reduce batch size: `--batch-size 2`
- Use CPU: `--device cpu`
- Close other GPU applications

### No Videos Found

Ensure your video files have a supported extension (`.mp4`, `.avi`, etc.) and are located in or under the specified directory. The script searches recursively.

### Noisy / Unstable FVD Values

If FVD values vary significantly between runs, you likely have too few clips. Increase the sample count:

```bash
python compute_fvd.py --ref-dir ./real --gen-dir ./fake --clips-per-video 8
```

## References

- Unterthiner et al., ["FVD: A New Metric for Video Generation"](https://arxiv.org/abs/1812.01717), 2019
- Carreira & Zisserman, ["Quo Vadis, Action Recognition?"](https://arxiv.org/abs/1705.07750), CVPR 2017
- I3D PyTorch weights: [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
