# Quantization Formats Reference

## Format Table

All formats available in `hf_ptq.py` via `--qformat`:

| CLI Name | Config Constant | Bits | Quantizes | Accuracy Drop | Best For |
|----------|----------------|------|-----------|---------------|----------|
| `fp8` | `FP8_DEFAULT_CFG` | 8 (E4M3) | Weights + activations | < 1% | Hopper/Blackwell, best accuracy |
| `int8` | `INT8_DEFAULT_CFG` | 8 (int) | Weights + activations | 1-2% | General purpose |
| `int8_sq` | `INT8_SMOOTHQUANT_CFG` | 8 (int) | Weights + activations | 1-2% | Ampere, SmoothQuant balancing |
| `int8_wo` | `INT8_WEIGHT_ONLY_CFG` | 8 (int) | Weights only | < 1% | Memory savings, minimal accuracy loss |
| `int4_awq` | `INT4_AWQ_CFG` | 4 (int) | Weights (AWQ) | 2-4% | Max compression, Ampere+ |
| `w4a8_awq` | `W4A8_AWQ_BETA_CFG` | W4/A8 | Mixed precision | 2-3% | Balance of compression and accuracy |
| `nvfp4` | `NVFP4_DEFAULT_CFG` | 4 (E2M1) | Weights + activations | 2-5% | Blackwell native HW support |
| `nvfp4_awq` | `NVFP4_AWQ_LITE_CFG` | 4 (E2M1) | Weights + activations | 2-4% | Better accuracy than nvfp4 |
| `nvfp4_mse` | `NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG` | 4 | Weights + activations | 2-4% | MSE-optimized NVFP4 |
| `fp8_pb_wo` | `FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG` | 8 | Weights only (blockwise) | < 1% | Fine-grained FP8 |
| `fp8_pc_pt` | `FP8_PER_CHANNEL_PER_TOKEN_CFG` | 8 | Per-channel/per-token | < 1% | Precise FP8 |
| `mxfp8` | `MXFP8_DEFAULT_CFG` | 8 (MX) | Weights + activations | < 1% | Microscaling FP8 |
| `nvfp4_mlp_only` | `NVFP4_MLP_ONLY_CFG` | 4 | MLP layers only | 1-3% | Selective quantization |
| `nvfp4_svdquant` | `NVFP4_SVDQUANT_DEFAULT_CFG` | 4 | SVD-based | 2-4% | SVDQuant algorithm |

## KV Cache Quantization

Separate from weight/activation quantization. Applied via `--kv_cache_qformat`:

| Name | Config | Description |
|------|--------|-------------|
| `none` | (no KV quant) | Default, no KV cache quantization |
| `fp8` | `FP8_KV_CFG` | FP8 KV cache |
| `fp8_affine` | `FP8_AFFINE_KV_CFG` | FP8 with affine transform |
| `nvfp4` | `NVFP4_KV_CFG` | NVFP4 KV cache |
| `nvfp4_affine` | `NVFP4_AFFINE_KV_CFG` | NVFP4 with affine |
| `nvfp4_rotate` | `NVFP4_KV_ROTATE_CFG` | NVFP4 with rotation |

## Deployment Compatibility

| Format | vLLM | TRT-LLM | SGLang |
|--------|------|---------|--------|
| fp8 | Yes (auto-detect) | Yes | Yes |
| int8_sq | Yes | Yes | Partial |
| int4_awq | Yes (auto-detect) | Yes | Yes |
| w4a8_awq | Yes | Yes | Partial |
| nvfp4 | Yes (v0.8+) | Yes | Partial |
| mxfp8 | Partial | Yes | No |

vLLM auto-detects the quantization format from `hf_quant_config.json` in the checkpoint directory.

## Quick Selection Guide

- **"I want the best accuracy"** → `fp8`
- **"I want the smallest model"** → `int4_awq` or `nvfp4`
- **"I have a Blackwell GPU"** → `nvfp4`
- **"I have a Hopper GPU"** → `fp8`
- **"I have an Ampere GPU"** → `int8_sq` or `int4_awq`
- **"I'm not sure"** → Start with `fp8`, iterate if needed
