# Sparse Attention for Diffusion Models

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

Two sparse-attention methods are supported under
`modelopt.torch.sparsity.attention_sparsity` (`mtsa`):

| Method | When to use | Calibration |
|--------|-------------|-------------|
| **Skip-Softmax** (BLASST) | Drop low-impact KV tiles inside FlashAttention. Works on any transformer with bidirectional attention. | Optional (exponential model) |
| **VSA** (Video Sparse Attention) | Block-level two-branch attention tuned for video models with long 3D token sequences. | None (fixed `top_k_ratio`) |

Switching between methods is a CLI/config change — the pipelines, APIs,
and plugins are shared.

| Model | Script | Methods |
|-------|--------|---------|
| Wan 2.2 5B / 14B | `wan22_sparse_attn.py` | `--method skip_softmax` (default), `--method vsa` |
| LTX-2            | `ltx2_vsa.py`          | VSA only (LTX-2 uses a custom attention module; skip-softmax backend in progress) |

---

## Skip-Softmax Sparse Attention

Skip-softmax (BLASST, <https://arxiv.org/pdf/2512.12087>) skips KV tiles whose attention
scores are negligible during the FlashAttention computation, reducing FLOPs without
retraining.

Two threshold modes are supported:

- **Fixed raw threshold** — pass a log2-space threshold directly to the Triton
  kernel. No calibration needed. Good for quick testing and sweeps.
- **Calibrated threshold** — an exponential model
  (`scale_factor = a * exp(b * target_sparsity)`) is calibrated once via the
  Triton calibration kernel, then the target sparsity can be adjusted at runtime
  without recalibration. Log-space fitting (`fit_logspace=True`) is recommended
  for diffusion models where scale_factors span many orders of magnitude.

### Quick Start (Wan 2.2)

```bash
# Fixed raw threshold (no calibration, fast)
python wan22_sparse_attn.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --raw-threshold -0.7 \
    --prompt "A cat playing piano" --output out.mp4

# With calibration
python wan22_sparse_attn.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --calibrate --target-sparsity 0.5 \
    --prompt "A cat playing piano" --output out.mp4

# Dense baseline (no sparsity, for comparison)
python wan22_sparse_attn.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --baseline \
    --prompt "A cat playing piano" --output baseline.mp4

# Report runtime sparsity (per-layer tile skip ratios)
python wan22_sparse_attn.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --raw-threshold -0.7 --report-avg-sparsity \
    --prompt "A cat playing piano" --output out.mp4
```

`--method skip_softmax` is the default, so it doesn't need to be passed
explicitly when using skip-softmax flags.

### Threshold Modes

| Mode | How threshold reaches the kernel | Use case |
|------|----------------------------------|----------|
| **Raw threshold** (`--raw-threshold -0.7`) | Passed directly as `skip_threshold_log2` — no conversion | Quick testing, sweeps |
| **Calibrated** (`--calibrate --target-sparsity 0.5`) | `scale_factor = a * exp(b * target)`, then backend computes `threshold = scale_factor / seq_k`, then kernel converts `log2(threshold) * sm_scale` | Production use with automatic seqlen adaptation |
| **Static lambda** (default `skip_softmax_threshold=0.1`) | `log2(lambda) * sm_scale` | Fallback when neither raw nor calibrated |

### Known Issues

- **14B dual transformer calibration**: Transformers are calibrated sequentially —
  transformer_2's calibration runs while transformer_1 is already sparsified,
  introducing asymmetric calibration conditions.
- **Minimum achievable sparsity**: Even the strictest threshold may yield 30–40%
  sparsity on diffusion models (many tiles are inherently negligible). Targets
  below this floor cause extrapolation; an inference-time warning is emitted.

---

## Video Sparse Attention (VSA)

VSA is a two-branch sparse attention architecture tailored for video diffusion
models:

1. **Compression branch** — averages tokens within a 3D block (default `4,4,4` =
   64 tokens) and computes coarse-grained block-level attention for global context.
2. **Sparse branch** — selects the top-K most important blocks by the compression
   branch's attention scores and computes fine-grained attention only on those.
3. **Gate blend** — `output = compression * gate_compress + sparse`. On models
   without a learned `gate_compress` (Wan 2.2, and LTX-2 until fine-tuned), VSA
   passes a zero tensor so `output = 0 * compression + sparse = sparse`. This
   makes VSA at `top_k_ratio=1.0` (keep all blocks) mathematically equivalent to
   dense attention, modulo `bfloat16` kernel rounding (~10⁻⁵ per call on a 75k
   token sequence).

VSA is **calibration-free** — sparsity is controlled by a fixed `top_k_ratio`
(`0.5` keeps 50% of blocks, `0.3` keeps 30%).

### Quick Start

```bash
# Wan 2.2 — VSA with default 50% top-K ratio (video_shape auto-derived)
python wan22_sparse_attn.py --method vsa \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --top-k-ratio 0.5 \
    --prompt "A cat playing piano" --output vsa.mp4

# Wan 2.2 — aggressive 30% top-K (70% sparsity), keep first/last 2 layers dense
python wan22_sparse_attn.py --method vsa \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --top-k-ratio 0.3 --skip-first-last 2 \
    --prompt "A cat playing piano" --output vsa.mp4

# Wan 2.2 — 720p+ / 81+ frames can OOM during VAE decode since VSA reserves
# ~15 GB of GPU memory for its tile buffers. Enable VAE tiling to recover.
python wan22_sparse_attn.py --method vsa \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --top-k-ratio 0.5 --enable-vae-tiling \
    --num-frames 81 --height 720 --width 1280 \
    --prompt "A cat playing piano" --output vsa.mp4

# LTX-2 — VSA
python ltx2_vsa.py \
    --checkpoint /path/to/ltx2.safetensors \
    --text-encoder-path /path/to/gemma \
    --top-k-ratio 0.5 \
    --prompt "A cat playing piano" --output vsa.mp4

# LTX-2 — baseline (no VSA)
python ltx2_vsa.py \
    --checkpoint /path/to/ltx2.safetensors \
    --text-encoder-path /path/to/gemma \
    --no-vsa --output baseline.mp4
```

### Requirements

- `fastvideo_kernel` at runtime (the Triton VSA kernel). Install with
  `pip install fastvideo_kernel`. VSA imports this lazily, so the modelopt
  sparsity API loads without it, but a VSA forward will raise a clear
  `ImportError` if missing.
- For LTX-2 only: `ltx_core`, `ltx_trainer`, `ltx_pipelines` (see LICENSE
  notice above).

### Programmatic API

```python
import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.config import VSA_DEFAULT

# Apply with the pre-built default (50% top-K, self-attention only)
transformer = mtsa.sparsify(transformer, VSA_DEFAULT)

# Or with a custom config
config = {
    "sparse_cfg": {
        "*.attn1*": {
            "method": "vsa",
            "block_size_3d": (4, 4, 4),   # 3D tile (T, H, W); 64 tokens per block
            "top_k_ratio": 0.3,           # 70% sparsity
            "enable": True,
            # "video_shape": (T, H, W),   # optional; auto-derived by the plugin
        },
        "*.attn2*": {"enable": False},    # skip cross-attention
        "default": {"enable": False},
    },
}
transformer = mtsa.sparsify(transformer, config)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size_3d` | `(4, 4, 4)` | Video tile dims (T, H, W) — default creates 64-token blocks |
| `top_k_ratio` | `0.5` | Fraction of blocks kept in the sparse branch (0 < ratio ≤ 1). `1.0` keeps all blocks = degenerate dense mode |
| `video_shape` | `None` | Post-patchify video shape (T, H, W). Plugins auto-derive it — set explicitly only to override. |
| `enable` | `True` | Per-layer toggle |

### How VSA Routes Through the Sparse-Attention API

- **Wan 2.2** uses diffusers' `WanAttention` whose processor calls
  `F.scaled_dot_product_attention` — VSA's SDPA patch in
  `SparseAttentionModule._forward_with_vsa_sdpa_patch` intercepts that call and
  replaces the computation with the Triton VSA kernel. The Wan 2.2 plugin
  registers a forward pre-hook that reads `hidden_states.shape = (B, C, T, H, W)`
  and sets `video_shape = (T // p_t, H // p_h, W // p_w)` on each VSA method
  instance before the transformer runs.
- **LTX-2** uses its native `LTXSelfAttention` whose forward signature is
  `(x, context, pe, k_pe)` and does **not** call `F.scaled_dot_product_attention`.
  The LTX-2 plugin installs a `_LTX2SparseAttention` wrapper that computes
  Q/K/V (with LTX-2's RMSNorm and `ltx_core` RoPE), an optional trainable
  `gate_compress` (zero-init), and then calls `VSA.forward_attention` directly.
  A forward pre-hook on the root `LTXModel` extracts `video_shape` from
  `Modality.positions`.
- Cross-attention is detected via Q/K sequence-length mismatch and falls
  through to the original attention path (no behaviour change).

### Verifying the Setup on Wan 2.2

A good sanity check is to compare `top_k_ratio=1.0` to the dense baseline —
since VSA without a learned gate becomes pure sparse attention and a full
mask is mathematically equivalent to dense, the two outputs should be close.
On a Wan 2.2 14B run at 720×1280 / 81 frames / 40 steps we measured:

| Comparison | First-frame PSNR |
|---|---|
| baseline vs baseline w/ VAE tiling | 40.5 dB |
| baseline vs VSA `top_k_ratio=1.0` | 23.9 dB |
| baseline vs VSA `top_k_ratio=0.5` | 13.1 dB |

The ~24 dB degrade at `top_k=1.0` is error accumulation (6400 attention
calls × bf16 rounding through the denoising loop) — a single-call PSNR vs
dense SDPA is 50 dB on random inputs.

### Known Limits

- **Peak memory on 720p+**: VSA's tile buffers reserve ~15 GB of GPU memory
  on top of the transformer, which can OOM the one-shot VAE decode at 720p /
  81 frames. Pass `--enable-vae-tiling` (or set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) to recover.
- **Token count ≥ 16 tiles (≈1024 tokens)**: VSA's setup overhead dominates for
  tiny sequences. For LTX-2, use ≥121 frames at ≥512×768 for meaningful speedups.
- **Mixing with skip-softmax**: VSA patches SDPA globally per module, while
  skip-softmax needs `attn_implementation="eager"`. `conversion.py` rejects
  configs that mix the two — run them separately.
- **Training**: `to_gate_compress` is zero-initialised and trainable, but no
  training loop is wired up yet. This example covers inference only.
