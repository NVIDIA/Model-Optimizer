# Skip-Softmax Sparse Attention for Diffusion Models

> [!WARNING]
> **Third-Party License Notice — LTX-2**
>
> LTX-2 packages (`ltx-core`, `ltx-pipelines`, `ltx-trainer`) are third-party dependencies
> developed and provided by [Lightricks](https://github.com/Lightricks/LTX-2). They are
> **NOT** covered by the Apache 2.0 license governing NVIDIA Model Optimizer.
>
> You **MUST** comply with the
> [LTX Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE)
> when installing and using LTX-2 with NVIDIA Model Optimizer. Any derivative models or
> fine-tuned weights produced from LTX-2 (including quantized, distilled, or sparsified
> checkpoints) remain subject to the LTX Community License Agreement, not Apache 2.0.

Skip-softmax sparse attention (BLASST, <https://arxiv.org/pdf/2512.12087>) skips KV
tiles whose attention scores are negligible during the FlashAttention computation,
reducing FLOPs without retraining.

Two modes are supported:
- **Fixed raw threshold** — pass a log2-space threshold directly to the Triton
  kernel. No calibration needed. Good for quick testing and sweeps.
- **Calibrated threshold** — an exponential model
  (`scale_factor = a * exp(b * target_sparsity)`) is calibrated once via the
  Triton calibration kernel, then the target sparsity can be adjusted at runtime
  without recalibration. Log-space fitting (`fit_logspace=True`) is recommended
  for diffusion models where scale_factors span many orders of magnitude.

## Supported Models

| Model | Script | Notes |
|-------|--------|-------|
| WAN 2.2 5B | `wan22_skip_softmax.py` | Single transformer, self-attention only |
| WAN 2.2 14B | `wan22_skip_softmax.py` | Dual transformer (auto-detected) |
| LTX-2 | (coming soon) | Via `ltx_triton_attention.py` backend |

## Quick Start

```bash
# Fixed raw threshold (no calibration, fast)
python wan22_skip_softmax.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --raw-threshold -0.7 \
    --prompt "A cat playing piano" --output out.mp4

# With calibration
python wan22_skip_softmax.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --calibrate --target-sparsity 0.5 \
    --prompt "A cat playing piano" --output out.mp4

# Dense baseline (no sparsity, for comparison)
python wan22_skip_softmax.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --baseline \
    --prompt "A cat playing piano" --output baseline.mp4

# Report runtime sparsity (per-layer tile skip ratios)
python wan22_skip_softmax.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --raw-threshold -0.7 --report-avg-sparsity \
    --prompt "A cat playing piano" --output out.mp4
```

## Threshold Modes

| Mode | How threshold reaches the kernel | Use case |
|------|----------------------------------|----------|
| **Raw threshold** (`--raw-threshold -0.7`) | Passed directly as `skip_threshold_log2` — no conversion | Quick testing, sweeps |
| **Calibrated** (`--calibrate --target-sparsity 0.5`) | `scale_factor = a * exp(b * target)`, then backend computes `threshold = scale_factor / seq_k`, then kernel converts `log2(threshold) * sm_scale` | Production use with automatic seqlen adaptation |
| **Static lambda** (default `skip_softmax_threshold=0.1`) | `log2(lambda) * sm_scale` | Fallback when neither raw nor calibrated |

## Known Issues

- **14B dual transformer calibration**: Transformers are calibrated sequentially — transformer_2's calibration runs while transformer_1 is already sparsified, introducing asymmetric calibration conditions.
- **Minimum achievable sparsity**: Even the strictest threshold may yield 30-40% sparsity on diffusion models (many tiles are inherently negligible). Targets below this floor cause extrapolation; an inference-time warning is emitted.
