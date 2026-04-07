# Skip-Softmax Sparse Attention for Diffusion Models

Skip-softmax sparse attention (BLASST) skips KV tiles whose attention scores
are negligible during the FlashAttention computation, reducing FLOPs without
retraining. An exponential model (`scale_factor = a * exp(b * target_sparsity)`)
is calibrated once, then the target sparsity can be adjusted at runtime without
recalibration.

## Changes from Main Branch

### Core Triton Kernel (`modelopt/torch/kernels/`)

| File | Change |
|------|--------|
| `triton_fa.py` | Added `_attn_fwd_calibrate` kernel: computes full attention while measuring skip decisions for multiple thresholds via atomic counters. Added `attention_calibrate()` Python API. |
| `__init__.py` | Export `attention_calibrate` alongside `attention`. |

The kernel has two modes:
- **Inference** (`_attn_fwd`): Autotuned, single threshold, actual tile skipping.
- **Calibration** (`_attn_fwd_calibrate`): Fixed block sizes (128×64), multi-threshold measurement, no skipping (full attention output).

### Sparse Attention Methods (`modelopt/torch/sparsity/attention_sparsity/methods/`)

| File | Change |
|------|--------|
| `triton_skip_softmax.py` | Extended with calibration support: `_triton_calibration_context()` sets Triton calibration mode and collects counters; `_triton_inference_context()` activates diffusers backend with calibrated threshold; `_get_diffusers_backend_context()` activates `modelopt_triton` attention backend. |
| `flash_skip_softmax.py` | Enhanced `get_sparse_context()` with `ExitStack` to also activate diffusers eager backend for calibration. |
| `registry.py` | Added `set_calibration_mode()` to base `SparseAttentionMethod` class. |
| `__init__.py` | Updated imports. |

### Kernel Backends (`modelopt/torch/sparsity/attention_sparsity/kernels/`)

| File | Change |
|------|--------|
| `__init__.py` | Added thread-local context (`set_skip_softmax_context` / `get_skip_softmax_context`), lazy imports for diffusers/LTX backends with `contextlib.suppress(ImportError, RuntimeError)`. |
| `diffusers_triton_attention.py` | **New.** Registers `modelopt_triton` backend in diffusers. Two modes: inference calls `attention()`, calibration calls `attention_calibrate()`. Accumulates counters across attention calls. |
| `diffusers_eager_attention.py` | **New.** Registers `modelopt_skip_softmax` eager backend for LLM calibration (explicit `F.softmax` for patching). |
| `ltx_triton_attention.py` | **New.** Patches `ltx_core.Attention` modules for Triton dispatch. Supports calibration and inference modes. |
| `ltx_eager_attention.py` | **New.** Patches `ltx_core.Attention` for eager attention calibration. |

### Calibration (`modelopt/torch/sparsity/attention_sparsity/calibration/`)

| File | Change |
|------|--------|
| `calibrate.py` | Skip RULER dataset generation when user provides `forward_loop` (required for diffusion models). Guard `from transformers import AutoTokenizer` as lazy import. |
| `calibrator.py` | `_set_thresholds()` detects method type — sets `_threshold_trials` for `triton_skip_softmax`, `thresholds` for `flash_skip_softmax`. |

### Conversion & Config

| File | Change |
|------|--------|
| `conversion.py` | Added `_register_diffusers_backends_if_needed()` — auto-registers diffusers/LTX backends on `sparsify()`. Updated export config and summary display. |
| `config.py` | Added `skip_softmax_threshold` field to `SparseAttentionAttributeConfig`. |
| `plugins/huggingface.py` | Added diffusers `ModelMixin` support in `_is_supported_model()`. Lazy `import transformers`. |
| `stats_manager.py` | Made `sparse_blocks` optional in `collect()`. Preserve `normalized_gaps` in calibration stats. |
| `sparse_attention.py` | (Changes from main for VSA support also present.) |

### Example Scripts

| File | Description |
|------|-------------|
| `wan22_skip_softmax.py` | **New.** Wan 2.2 text-to-video with skip-softmax. Supports 5B (single transformer) and 14B (dual transformer). Uses `triton_skip_softmax` with Triton calibration kernel. Calibration prompts from OpenVid-1M. |

### Tests

| File | Description |
|------|-------------|
| `test_kernel_backends.py` | **New.** Unit tests for diffusers kernel backends with mocked dependencies (no GPU required). |

## Usage

```bash
# Wan 2.2 5B — calibrate + generate
python wan22_skip_softmax.py \
    --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --calibrate --target-sparsity 0.5 --calib-size 4 \
    --calib-frames 151 --calib-steps 40 \
    --prompt "A cat sitting on a windowsill" --output out.mp4

# Wan 2.2 14B — both transformers sparsified
python wan22_skip_softmax.py \
    --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --calibrate --target-sparsity 0.5 --calib-size 4 \
    --calib-frames 151 --calib-steps 40 \
    --prompt "A sunset over mountains" --output out.mp4

# Calibrate only (no video generation)
python wan22_skip_softmax.py \
    --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --calibrate --target-sparsity 0.5 --calib-size 4
```

## Architecture

```text
mtsa.sparsify(transformer, config, forward_loop)
  │
  ├─ apply_mode() → replace attention with SparseAttentionModule
  │
  └─ calibrate()
       │
       ├─ DynamicThresholdCalibrator._set_thresholds()
       │    └─ sets method._threshold_trials = [1e-6, ..., 9.9e-1]
       │
       ├─ forward_loop(model)
       │    │
       │    └─ SparseAttentionModule.forward()
       │         │
       │         └─ triton_skip_softmax._triton_calibration_context()
       │              ├─ set_triton_skip_softmax_config(calibration_mode=True)
       │              ├─ attention_backend("modelopt_triton")
       │              ├─ _diffusers_triton_attention() → attention_calibrate()
       │              │    └─ _attn_fwd_calibrate kernel (full attn + atomic counters)
       │              └─ _collect_calibration_stats() → module._last_stats
       │
       ├─ Fit: scale_factor = a * exp(b * sparsity)
       │
       └─ Apply a, b to all modules
            │
            └─ Inference: triton_skip_softmax._triton_inference_context()
                 ├─ threshold = a * exp(b * target) / seqlen
                 └─ attention() with skip_softmax_threshold → actual tile skipping
```
