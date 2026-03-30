# Skip-Softmax Sparse Attention Experiments

## Experiment Scripts

### LTX-2: `run_exps_skip_softmax_v25.py`

Supports skip-softmax V1 (`sparse_N`), V2.5 (`sparse_v25_N`), and various debug variants.

```bash
cd examples/diffusers/sparsity
PY=/home/jingyux/miniconda3/envs/modelopt/bin/python

# Baseline (Triton FA, no sparsity)
CUDA_VISIBLE_DEVICES=0 $PY run_exps_skip_softmax_v25.py \
    --num-frames 251 --calib-frames 251 \
    --output-dir experiment_outputs/skip_softmax/ltx2 \
    --experiment triton_baseline

# Skip-softmax V1: 25%, 50%, 75% sparsity
CUDA_VISIBLE_DEVICES=1 $PY run_exps_skip_softmax_v25.py \
    --num-frames 251 --calib-frames 251 \
    --output-dir experiment_outputs/skip_softmax/ltx2 \
    --experiment sparse_25

CUDA_VISIBLE_DEVICES=2 $PY run_exps_skip_softmax_v25.py \
    --num-frames 251 --calib-frames 251 \
    --output-dir experiment_outputs/skip_softmax/ltx2 \
    --experiment sparse_50

CUDA_VISIBLE_DEVICES=3 $PY run_exps_skip_softmax_v25.py \
    --num-frames 251 --calib-frames 251 \
    --output-dir experiment_outputs/skip_softmax/ltx2 \
    --experiment sparse_75

# Skip-softmax V2.5 (pool-K + fresh v_mean): 50% sparsity
CUDA_VISIBLE_DEVICES=4 $PY run_exps_skip_softmax_v25.py \
    --num-frames 251 --calib-frames 251 \
    --output-dir experiment_outputs/skip_softmax/ltx2 \
    --experiment sparse_v25_50
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | required | `baseline`, `triton_baseline`, `sparse_{5,25,50,75}`, `sparse_v25_{25,50,75}` |
| `--num-frames` | 121 | Inference frame count |
| `--calib-frames` | 81 | Calibration frame count |
| `--calib-steps` | 35 | Calibration denoising steps |
| `--skip-first-last` | 2 | Exclude first/last N transformer blocks from sparsity |
| `--output-dir` | `experiment_outputs` | Output directory for videos and logs |
| `--prompt` | "A cat playing piano" | Text prompt |
| `--seed` | 42 | Random seed |

**LTX-2 model details:**
- 48 transformer blocks, 32 heads, head_dim=128
- Self-attention: `*.attn1`, Cross-attention: `*.attn2`
- Audio/cross-modal attention disabled: `*audio_attn1*`, `*audio_attn2*`, `*audio_to_video_attn*`, `*video_to_audio_attn*`
- Model paths hardcoded in script (scratch storage)

**Seq lengths by frame count:**

| Frames | seq_k | KV tiles (bc=128) |
|--------|-------|-------------------|
| 81 | 2560 | 20 |
| 121 | 3840 | 30 |
| 251 | 7680 | 60 |

---

### Wan 2.2 14B: `run_exps_wan.py`

Supports both 5B (1 transformer) and 14B (2 transformers). For 14B, both
`pipe.transformer` and `pipe.transformer_2` are wrapped in an `nn.ModuleList`
and sparsified in a single `mtsa.sparsify()` call.

```bash
cd examples/diffusers/sparsity
PY=/home/jingyux/miniconda3/envs/modelopt/bin/python
MODEL=/home/scratch.omniml_data_2/jingyux/models/Wan2.2-T2V-A14B-Diffusers

# Baseline
CUDA_VISIBLE_DEVICES=4 $PY run_exps_wan.py \
    --model-id $MODEL --num-frames 81 --calib-frames 81 \
    --output-dir experiment_outputs/skip_softmax/wan2.2-14b \
    --experiment baseline --skip-first-last 3

# Sparse 25%, 50%, 75%
CUDA_VISIBLE_DEVICES=5 $PY run_exps_wan.py \
    --model-id $MODEL --num-frames 81 --calib-frames 81 \
    --output-dir experiment_outputs/skip_softmax/wan2.2-14b \
    --experiment sparse_25 --skip-first-last 3

CUDA_VISIBLE_DEVICES=6 $PY run_exps_wan.py \
    --model-id $MODEL --num-frames 81 --calib-frames 81 \
    --output-dir experiment_outputs/skip_softmax/wan2.2-14b \
    --experiment sparse_50 --skip-first-last 3

CUDA_VISIBLE_DEVICES=7 $PY run_exps_wan.py \
    --model-id $MODEL --num-frames 81 --calib-frames 81 \
    --output-dir experiment_outputs/skip_softmax/wan2.2-14b \
    --experiment sparse_75 --skip-first-last 3
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | required | `baseline`, `sparse_{25,50,75}` |
| `--model-id` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | HF model ID or local path |
| `--num-frames` | 81 | Inference frame count |
| `--calib-frames` | 81 | Calibration frame count |
| `--calib-steps` | 35 | Calibration denoising steps |
| `--num-steps` | 50 | Inference denoising steps |
| `--guidance-scale` | 5.0 | CFG scale |
| `--skip-first-last` | 3 | Exclude first/last N blocks from sparsity |
| `--height` | 704 | Video height |
| `--width` | 1280 | Video width |
| `--output-dir` | `experiment_outputs/skip_softmax/wan` | Output directory |

**Wan 2.2 model details:**

| | 5B | 14B |
|---|---|---|
| Transformers | 1 (`pipe.transformer`) | 2 (`pipe.transformer` + `pipe.transformer_2`) |
| Blocks per transformer | 30 | 40 |
| Heads | 24 | 40 |
| head_dim | 128 | 128 |
| Self-attention | `*.attn1` | `*.attn1` |
| Cross-attention | `*.attn2` (disabled) | `*.attn2` (disabled) |

**Seq lengths (704x1280):**

| Frames | seq_k (approx) |
|--------|----------------|
| 81 | ~27,720 |
| 121 | ~55,440 |
| 251 | ~55,440 |

---

## Calibration Fix (2026-03-26)

The diffusion calibration had a bug: it collected **per-row** normalized gaps
for percentile calibration, but the kernel skips tiles only when **all 128 rows
agree**. This caused the calibrated threshold to vastly overestimate achievable
tile-level sparsity.

**Fix** (`triton_skip_softmax_diffusion.py`): Changed calibration to collect
**per-tile min gaps** (`min over 128 rows`) before computing the percentile.
This matches the kernel's skip decision granularity.

```
Before: threshold = percentile(all_per_row_gaps, (1-target)*100)
After:  threshold = percentile(all_per_tile_min_gaps, (1-target)*100)
```

**Impact on calibrated thresholds (LTX-2, 251 frames):**

| Target sparsity | Threshold (per-tile min) | Notes |
|-----------------|--------------------------|-------|
| 25% | 0.189 | p75 of tile min gaps |
| 50% | 0.036 | p50 — very small, most tiles have a row near cummax |
| 75% | 0.000 | p25 is exactly 0 — 75%+ tiles have gap=0 for some row |

---

## Results

### LTX-2 (251 frames, skip-softmax V1, per-tile calibration)

| Experiment | Threshold | Time | Notes |
|---|---|---|---|
| Baseline (Triton, no skip) | — | 264.9s | |
| 25% target | 0.189 | 282.0s | Some tiles skipped but overhead > savings |
| 50% target | 0.036 | 277.5s | Very few tiles actually skip |
| 75% target | 0.000 | 277.6s | No tiles skip (threshold=0) |

**Conclusion**: Skip-softmax V1 is not effective for LTX-2. The all-128-rows-agree
requirement means very few tiles can be skipped, even with correct calibration.
With only 60-120 KV tiles, rows within a tile disagree too often.

### Wan 2.2 14B (81 frames, skip-softmax V1, per-tile calibration)

*(Results pending — experiments running)*

---

## Data Capture Scripts

### LTX-2: `capture_attn_inputs.py`

Captures raw Q/K/V tensors from LTX-2 self-attention layers for offline analysis.

```bash
# 121 frames (default)
$PY capture_attn_inputs.py --num-frames 121 --save-steps 0 9 19 29 39

# 251 frames
$PY capture_attn_inputs.py --num-frames 251 --save-steps 0 9 19 29 39 \
    --save-dir experiments/attn_input_251frames
```

### Wan 2.2: `capture_attn_inputs_wan.py`

Captures raw Q/K/V from Wan 2.2 self-attention (patches `WanAttnProcessor`).

```bash
$PY capture_attn_inputs_wan.py --num-frames 251 --save-steps 0 9 19 29 39 49 \
    --save-dir experiments/attn_input_wan22_251frames
```

### Visualization: `experiments/visual/viz_attention_tiles.py`

See `experiments/visual/README.md` for full documentation.

---

## Output Locations

```
experiment_outputs/skip_softmax/
    ltx2/                          # LTX-2 results
        log_baseline_251.txt
        log_sparse_{25,50,75}_251.txt
        triton_baseline.mp4
        sparse_{25,50,75}pct.mp4
    wan2.2-14b/                    # Wan 14B results
        log_baseline.txt
        log_sparse_{25,50,75}.txt
        baseline.mp4
        sparse_{25,50,75}pct.mp4
experiments/
    attn_input/                    # LTX-2 121 frames Q/K/V
    attn_input_251frames/          # LTX-2 251 frames Q/K/V
    attn_input_wan22_251frames/    # Wan 2.2 251 frames Q/K/V
    visual/                        # Visualization scripts and plots
```
