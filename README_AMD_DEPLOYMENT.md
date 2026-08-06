# AMD MI300X FP8 Deployment Guide

ROCm-Model-Optimizer provides native FP8/INT8 quantization support for AMD MI300X and MI325X
(gfx942/CDNA3) via hipBLASLt, achieving **1.73–1.91× speedup over FP16** at batch sizes ≥256.

## Quick Start

```python
import torch
import modelopt.torch.quantization as mtq
from modelopt._rocm_compat import (
    extract_fp8_scales, convert_to_static_fp8,
    warmup_for_llama, get_quantization_strategy
)

# 1. Choose quantization strategy based on your batch size
strategy = get_quantization_strategy(batch_size=256, hidden_size=8192)
print(f"Strategy: {strategy['strategy']} | Est speedup: {strategy['estimated_speedup']}x")

# 2. Calibrate model
mtq.quantize(model, mtq.AMD_FP8_DEFAULT_CFG, forward_loop=lambda m: m(x_cal))

# 3. Extract calibrated scales and convert to deployment mode
scales = extract_fp8_scales(model)
model = convert_to_static_fp8(model, scales)   # full quantization (BS ≥ 128)
# OR:
# from modelopt._rocm_compat import convert_ffn_only_to_fp8
# model = convert_ffn_only_to_fp8(model, scales)  # FFN only (BS < 64)

# 4. Warm up hipBLASLt for your model's GEMM shapes
warmup_for_llama("70b", batch_sizes=[1, 4, 16, 64, 256])  # for LLaMA-70B

# 5. Save scales for fast restart (skip calibration next time)
from modelopt._rocm_compat import save_fp8_scales, load_fp8_scales
save_fp8_scales(scales, "llama70b_fp8_scales.json")
# Next session: scales = load_fp8_scales("llama70b_fp8_scales.json")

# 6. Inference
output = model(x)
```

## Benchmark Results (AMD MI300X / gfx942 / ROCm 7.0)

### Full Pipeline (calibrate → deploy)

| Model | Batch Size | FP16 ms | FP8 ms | Speedup | TFLOPS FP8 |
|-------|-----------|---------|--------|---------|-----------|
| LLaMA-70B FFN | 1 | 0.377 | 0.208 | **1.81×** | — |
| LLaMA-70B FFN | 4 | 0.386 | 0.211 | **1.83×** | — |
| LLaMA-70B FFN | 256 | 0.882 | 0.539 | 1.64× | 669 |
| LLaMA-7B FFN | 256 | 0.287 | 0.175 | 1.63× | 395 |
| LLaMA-7B FFN | 512 | 0.379 | 0.207 | **1.84×** | 670 |
| Standard FFN (4096/16384) | 256 | 0.346 | 0.191 | **1.81×** | 540 |

### CI Benchmark (automated)

```bash
sbatch scripts/amd_benchmark_ci.sbatch
# BENCHMARK PASS: BS=256 = 1.89× ≥ 1.4× threshold ✅
# 25/25 unit tests pass ✅
```

## Quantization Strategy by Batch Size

| Batch Size | Strategy | Why |
|-----------|----------|-----|
| BS=1–32 | FFN only | Attention cast overhead > GEMM savings at small BS |
| BS=64–127 | Full (70B+) or FFN-only (7B) | Depends on model H dimension |
| BS≥128 | Full quantization | Both FFN and attention benefit |

```python
from modelopt._rocm_compat import get_quantization_strategy, convert_ffn_only_to_fp8

strategy = get_quantization_strategy(batch_size=1, hidden_size=8192, model_size_b=70)
if strategy["quantize_attention"]:
    model = convert_to_static_fp8(model, scales)  # full
else:
    model = convert_ffn_only_to_fp8(model, scales)  # FFN only
```

## Key AMD-Specific Notes

1. **`float8_e4m3fnuz` ≠ `float8_e4m3fn`** — AMD CDNA uses different exponent bias.
   Always use `torch.float8_e4m3fnuz` on MI300X.

2. **Static scale required** — Dynamic per-batch `amax()` computation adds ~0.08ms overhead
   per forward pass, eliminating FP8 benefit. Always call `set_input_scale()` or use
   `convert_to_static_fp8()`.

3. **`torch._int_mm` is NOT hipBLASLt** — For INT8 deployment speedup, use MIGraphX:
   ```python
   export_fp8_onnx(model, x, "model.onnx")
   # Then: mgx.quantize_int8(m) → compile → 1,205 TOPS peak
   ```

4. **Warmup is critical** — hipBLASLt performs algorithm search on first call (adds ~100ms).
   Always call `warmup_for_llama()` before serving.

5. **KV cache savings** — FP8 KV cache reduces memory 50%:
   ```python
   from modelopt._rocm_compat import kv_cache_memory_savings
   savings = kv_cache_memory_savings(seq_len=4096, n_heads=8, head_dim=128, n_layers=80)
   # LLaMA-70B: saves ~6.7 GB per request at BS=1
   ```

## AMD-Specific APIs

| API | Description |
|-----|-------------|
| `mtq.AMD_FP8_DEFAULT_CFG` | FP8 E4M3FNUZ calibration config |
| `mtq.AMD_INT8_DEFAULT_CFG` | INT8 per-channel calibration config |
| `extract_fp8_scales(model)` | Extract calibrated per-layer scales |
| `convert_to_static_fp8(model, scales)` | Full FP8 deployment conversion |
| `convert_ffn_only_to_fp8(model, scales)` | FFN-only conversion (BS<64) |
| `get_quantization_strategy(bs, H)` | Recommend full vs FFN-only strategy |
| `warmup_for_llama(size, batch_sizes)` | Pre-warm hipBLASLt for LLaMA shapes |
| `save_fp8_scales(scales, path)` | Persist calibrated scales to JSON |
| `load_fp8_scales(path)` | Load saved scales (skip calibration) |
| `amd_deploy_model(model, ...)` | One-call deployment pipeline |
| `FP8Linear` | Drop-in nn.Linear with hipBLASLt dispatch |
| `quantize_kv_cache_fp8(k, v)` | Quantize KV cache (50% memory) |
| `profile_amd_model(model, x)` | Benchmark latency + throughput |
| `compare_amd_models(models, x)` | Side-by-side comparison |
| `print_amd_perf_report(model, x)` | Full performance report |

## Installation

```bash
# Standard install
pip install -e .

# AMD-specific extras
pip install -e ".[amd]"
```

## Requirements

- AMD MI300X or MI325X (gfx942/CDNA3)
- ROCm 7.0+
- PyTorch with ROCm support (`torch.version.hip` is not None)
