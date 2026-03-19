# Skip-Softmax Sparse Attention for Diffusion Models

Skip-softmax exploits how Flash Attention processes K/V tiles left-to-right. It
maintains a **cumulative maximum** (cummax) of attention scores as it scans tiles.
If a tile's max score is far below the cummax, its contribution after softmax is
negligible and the tile can be skipped.

## Skip Decision Formula

For each tile `j` in the attention computation (per query row, per head):

```python
gap = cummax[j] - block_max[j]

skip tile j if:  gap >= threshold * log(seq_k)
```

Where:

- `block_max[j]` = max attention score in tile `j` (over all elements in the 128×128 block)
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

In a CUDA kernel, this is zero per-tile overhead — just multiply the threshold by
`log(seq_k)` once at kernel launch:

```cuda
// Precompute once at launch:
float scaled_threshold = threshold * logf(seq_k);

// Inside the tile loop (row_max is already tracked by Flash Attention):
float gap = row_max - block_max_j;
if (gap >= scaled_threshold)
    skip this tile;
```

### Comparison with raw gap (no normalization)

The raw gap `cummax - block_max >= constant` gives inconsistent sparsity when
sequence length changes:

| Target | Formula          | 81fr  | 121fr | 151fr | 251fr | Max drift |
|--------|------------------|-------|-------|-------|-------|-----------|
| 20%    | gap/log(seq_k)   | 20.0% | 21.5% | 21.2% | 19.5% | **2.0%** |
| 20%    | raw gap          | 20.0% | 23.7% | 24.5% | 26.3% | 6.3%     |
| 50%    | gap/log(seq_k)   | 50.2% | 54.0% | 55.1% | 56.6% | **6.4%** |
| 50%    | raw gap          | 50.2% | 55.7% | 57.6% | 61.5% | 11.2%    |

## Calibration

Calibration determines the threshold for a target sparsity. One forward pass
through the model is sufficient.

### Procedure

1. Run a calibration forward pass (e.g. 1 prompt, 10 denoising steps, short
   frame count like 81 frames for speed).
2. For every attention call during the forward pass, at every layer, head, step,
   and tile, compute:

   ```python
   normalized_gap = (cummax[j] - block_max[j]) / log(seq_k)
   ```

3. Collect ALL normalized gap values into one flat list (typically millions of values).
4. Set the threshold as a percentile:

   ```python
   threshold = percentile(all_gaps, (1 - target_sparsity) * 100)
   ```

   For 20% target: `threshold = 80th percentile = 0.782`
   For 50% target: `threshold = 50th percentile = 0.389`

### Reference Thresholds (LTX-2, calibrated on 81 frames, 30 steps)

| Target sparsity | Threshold | 81fr   | 121fr  | 151fr  | 251fr  | Max drift |
|-----------------|-----------|--------|--------|--------|--------|-----------|
| 10%             | 0.996     | 10.1%  | 10.5%  | 10.1%  |  8.2%  | 2.3%      |
| 15%             | 0.876     | 15.1%  | 15.9%  | 15.5%  | 13.6%  | 2.3%      |
| 20%             | 0.782     | 20.0%  | 21.5%  | 21.2%  | 19.5%  | 2.0%      |
| 25%             | 0.703     | 25.0%  | 26.9%  | 26.8%  | 25.6%  | 1.9%      |
| 30%             | 0.631     | 30.0%  | 32.5%  | 32.7%  | 31.8%  | 2.7%      |
| 40%             | 0.505     | 40.0%  | 43.2%  | 44.0%  | 44.3%  | 4.3%      |
| 50%             | 0.389     | 50.2%  | 54.0%  | 55.1%  | 56.6%  | 6.4%      |

Stability is best at low-to-moderate sparsity (10-30%, drift ≤ 2.7%). At higher
targets (50%+), longer sequences tend to be slightly more sparse than intended
(~6% drift).

Note: the target sparsity is the **average across all layers, heads, and
denoising steps**. In practice, early denoising steps will have much lower
sparsity than the target, and late steps will have higher sparsity. The
threshold is fixed — the attention patterns change as denoising progresses.

### Comparison with exponential calibration model

The existing calibration in `flash_skip_softmax.py` (lines 148-155) uses an
exponential model to map target sparsity to a threshold:

```python
scale_factor = a * exp(b * target_sparsity)    # exponential model
threshold_C = log(seq_k / scale_factor)         # = log(seq_k) - log(scale_factor)
skip if: gap >= threshold_C
```

Both approaches produce `gap >= C` at inference. The difference is how C is
computed from the target sparsity and seq_k:

| | Exponential model (additive) | Percentile model (multiplicative) |
|---|---|---|
| **Formula** | `C = log(seq_k) - log(a * exp(b * S))` | `C = threshold * log(seq_k)` |
| **Calibration** | Fit exponential `a * exp(b * S)` on trial thresholds | Percentile of all gaps / log(seq_k) |
| **seq_k scaling** | `log(seq_k) - constant` (additive shift) | `constant * log(seq_k)` (proportional scaling) |

Verified with actual tile data (a=646.58, b=3.06 from LTX-2 calibration,
targeting 20%):

| Frames | seq_k | C (exponential) | Actual sparsity | C (percentile) | Actual sparsity |
|--------|-------|-----------------|-----------------|----------------|-----------------|
| 81     | 4224  | 1.26            | **74.1%**       | 6.53           | **20.0%**       |
| 121    | 6144  | 1.64            | **74.0%**       | 6.82           | **21.5%**       |
| 151    | 7296  | 1.81            | **74.1%**       | 6.96           | **21.2%**       |
| 251    | 12288 | 2.33            | **71.9%**       | 7.36           | **19.5%**       |

The exponential model's fit (R²=0.60 on diffusion data) produces thresholds
(C=1.26–2.33) that are far too small, resulting in ~74% sparsity regardless of
the 20% target. The percentile model produces correct thresholds (C=6.53–7.36)
that hit the target.

Full comparison across all target sparsities:

| Target | Exponential model actual | Percentile model actual | Percentile max drift |
|--------|-------------------------|------------------------|---------------------|
| 5%     | 68%                     | 4-5%                   | 1.5%                |
| 15%    | 70-72%                  | 14-16%                 | 2.3%                |
| 20%    | 72-74%                  | 20-22%                 | 2.0%                |
| 25%    | 74-76%                  | 25-27%                 | 1.9%                |
| 50%    | 82-87%                  | 50-57%                 | 6.4%                |

The exponential model fails for diffusion because `a * exp(b * sparsity)` was
designed for LLMs where calibration data spans 0-90% sparsity. On diffusion,
the calibration data only covers 0-29% (due to uniform attention in many
heads/layers), so the extrapolation is unreliable.

### Comparison with LiteAttention

LiteAttention uses a similar tile-skipping approach but compares against the
**previous tile's max** instead of the cumulative max:

```python
LiteAttention:  skip if (block_max[j] - block_max[j-1]) * log2(e) < threshold
Skip-softmax:   skip if (cummax[j] - block_max[j]) >= threshold * log(seq_k)
```

Key differences:

- Skip-softmax uses cummax (global reference) — strictly more conservative
- LiteAttention uses previous tile only (local reference) — compensates with
  multi-pass refinement via double-buffered skip lists
- The `log2(e)` in LiteAttention is because the kernel uses `exp2` instead of
  `exp` for performance (single PTX instruction on NVIDIA GPUs)

## Diffusion-Specific Observations

### Sparsity varies per head, not just per layer

Within the same layer and timestep, different heads have vastly different sparsity:

| Layer | Head 0 (thr=0.1) | Head 31 (thr=0.1) |
|-------|-------------------|--------------------|
| 5     | 0% everywhere    | 91-99%             |
| 10    | 10-43%           | 61-81%             |
| 23    | 2-79%            | 81-91%             |

Head 0 has flat, uniform attention scores (range ~0.3). Head 31 has peaked
scores (range ~10). The skip decision is per-head — there is no cross-head
information.

### Sparsity varies by denoising timestep

Early denoising steps (noisy input, uniform attention) have low sparsity.
Late steps (structured attention, peaked scores) have higher sparsity.

Layer 23, threshold=0.1:

- Steps 0-5: 0-14% sparsity
- Steps 10-15: 25-48% sparsity
- Steps 20-29: 44-55% sparsity

### LTX-2 attention module naming

- `*.attn1` — video self-attention (apply sparsity here)
- `*.attn2` — text cross-attention (skip)
- `*.audio_attn1/2` — audio attention (skip)
- `*.audio_to_video_attn` / `*.video_to_audio_attn` — cross-modal (skip)

## Scripts

| Script | Purpose |
|--------|---------|
| `ltx2_skip_softmax.py` | End-to-end example: sparsify + generate video |
| `ltx2_baseline.py` | Dense baseline for comparison |
| `ltx2_capture_tile_data.py` | Capture block_max/cummax data to JSON for offline analysis |
| `ltx2_cummax_study.py` | Study cummax progression per layer across timesteps |
| `ltx2_cummax_seqlen_study.py` | Study cummax across different sequence lengths |
| `plot_seqlen_bmax_cummax.py` | Plot block_max vs cummax from saved JSON data |
| `skip_softmax_normalization_study.py` | Evaluate normalization formulas for seq_k invariance |
| `cross_calib_global_threshold.py` | Cross-calibration test with global threshold |
| `cross_calib_log_seqk.py` | Cross-calibration test with per-layer threshold |

## Usage

```bash
# Dense baseline
python ltx2_baseline.py --prompt "A cat playing piano" --output baseline.mp4

# Skip-softmax with static threshold (quick, no calibration)
python ltx2_skip_softmax.py --prompt "A cat playing piano" --output sparse.mp4 \
    --threshold 5e-4

# With calibration
python ltx2_skip_softmax.py --prompt "A cat playing piano" --output sparse.mp4 \
    --calibrate --target-sparsity 0.25

# Capture tile data for offline analysis
python ltx2_capture_tile_data.py --frames 81 121 151 251 --output-dir tile_data

# Plot from saved data (no GPU needed)
python plot_seqlen_bmax_cummax.py --layer 23 --head 0 --timestep 25
```
