# Skip-Softmax Sparse Attention for Diffusion Models

Skip-softmax exploits how Flash Attention processes K/V tiles left-to-right. It
maintains a **cumulative maximum** (cummax) of attention scores as it scans tiles.
If a tile's max score is far below the cummax, its contribution after softmax is
negligible and the tile can be skipped.

## Architecture

Two modes controlled by `_calibration_mode`:
- **Calibration**: eager attention with F.softmax patching to collect gap statistics
- **Inference**: Triton FA kernel with fused tile skipping and gap/log(seq_k) normalization

## Skip Decision Formula

For each tile `j` in the attention computation (per query row, per head):

```python
gap = cummax[j] - block_max[j]

skip tile j if:  gap >= threshold * log(seq_k)
```

Where:

- `block_max[j]` = max attention score in tile `j` (over all elements in the 128x128 block)
- `cummax[j]` = max of `block_max[0..j]` (running maximum across all tiles seen so far)
- `gap` = how far this tile's peak is below the global running peak (always >= 0)
- `seq_k` = total sequence length of keys
- `threshold` = a single scalar calibrated for a target sparsity
- `log` = natural logarithm

### Why `log(seq_k)`?

Longer sequences produce larger absolute gaps between block_max and cummax.
Dividing the gap by `log(seq_k)` normalizes for sequence length, so a threshold
calibrated at one resolution (e.g. 81 frames) generalizes to other resolutions
(121, 151, 251 frames) with minimal sparsity drift.

In the Triton kernel, this is computed once at launch:

```python
# Python wrapper computes effective threshold:
skip_threshold_log2 = -threshold * math.log2(seq_k)

# Kernel comparison (same as LLM path, different threshold value):
can_skip = tile_row_max < (row_max + skip_threshold_log2)
```

## Calibration

Calibration determines the threshold for a target sparsity using the
**percentile method**. One forward pass through the model is sufficient.

### Procedure

1. Run a calibration forward pass with eager attention (F.softmax patching)
2. For every attention call, compute: `normalized_gap = (cummax[j] - block_max[j]) / log(seq_k)`
3. Collect ALL normalized gap values into one flat list
4. Set the threshold as a percentile:
   ```python
   threshold = percentile(all_gaps, (1 - target_sparsity) * 100)
   ```
   For 20% target: `threshold = 80th percentile = 0.782`

### Reference Thresholds (LTX-2, calibrated on 81 frames, 30 steps)

| Target sparsity | Threshold | 81fr   | 121fr  | 151fr  | 251fr  | Max drift |
|-----------------|-----------|--------|--------|--------|--------|-----------|
| 10%             | 0.996     | 10.1%  | 10.5%  | 10.1%  |  8.2%  | 2.3%      |
| 20%             | 0.782     | 20.0%  | 21.5%  | 21.2%  | 19.5%  | 2.0%      |
| 30%             | 0.631     | 30.0%  | 32.5%  | 32.7%  | 31.8%  | 2.7%      |
| 50%             | 0.389     | 50.2%  | 54.0%  | 55.1%  | 56.6%  | 6.4%      |

## Usage

```bash
# With calibration (recommended)
python ltx2_skip_softmax.py --prompt "A cat playing piano" --output sparse.mp4 \
    --calibrate --target-sparsity 0.2

# Skip first/last layers for quality
python ltx2_skip_softmax.py --prompt "A cat playing piano" --output sparse.mp4 \
    --calibrate --target-sparsity 0.2 --skip-first-last 2
```

## LTX-2 Attention Module Naming

- `*.attn1` — video self-attention (apply sparsity here)
- `*.attn2` — text cross-attention (skip)
- `*.audio_attn1/2` — audio attention (skip)
- `*.audio_to_video_attn` / `*.video_to_audio_attn` — cross-modal (skip)
