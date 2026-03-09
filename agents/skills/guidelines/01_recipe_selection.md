# Recipe Selection Guide

## Decision Tree

```
Step 1: What GPU generation?
├── Blackwell (B100/B200) → nvfp4 (native HW support, best perf)
├── Hopper (H100/H200)   → fp8 (native HW support, best accuracy)
└── Ampere (A100/A10) or older → int8_sq or int4_awq

Step 2: Model size vs available GPUs?
├── < 3B params   → TP=1 (single GPU)
├── 3B - 13B      → TP=1 (fp8/nvfp4) or TP=2 (int8/bf16)
├── 13B - 70B     → TP=2-4
└── > 70B         → TP=4-8

Step 3: Accuracy priority?
├── Highest accuracy   → fp8 (< 1% drop)
├── Good balance       → w4a8_awq or int8_sq (1-3% drop)
└── Max compression    → nvfp4 or int4_awq (2-5% drop)

Step 4: Deployment target?
├── vLLM     → fp8, int4_awq, nvfp4, w4a8_awq (all auto-detected)
├── TRT-LLM  → all formats supported
└── SGLang   → fp8, int4_awq
```

## Quick Rules

- **When in doubt, start with fp8.** It has the best accuracy and is widely supported.
- **For Blackwell GPUs, try nvfp4 first.** Native hardware support makes it fast.
- **If accuracy is too low, move up the chain** (see iteration table below).
- **KV cache quantization is optional** — add it for extra memory savings with minimal accuracy impact.

## Iteration Strategy

When the user rejects accuracy, propose the next lighter format:

| Current | Try Next | Expected Improvement |
|---------|----------|---------------------|
| `nvfp4` | `fp8` | Significant — 2x more bits |
| `nvfp4_awq` | `fp8` | Significant |
| `int4_awq` | `w4a8_awq` | Moderate — 8-bit activations help |
| `w4a8_awq` | `int8_sq` | Moderate — full 8-bit |
| `int8_sq` | `fp8` | Small — FP8 often edges out INT8 |
| `fp8` | (stop) | Already the lightest quantization |

If `fp8` accuracy is still unacceptable, the model may not be suitable for quantization at this size. Consider:

- Using a larger model variant
- Fine-tuning after quantization (QAT)
- Selective quantization (`nvfp4_mlp_only` — quantize only MLP layers)

## Format Comparison Summary

| Format | Compression | Accuracy | Speed Gain | GPU Requirement |
|--------|-------------|----------|------------|-----------------|
| `fp8` | ~2x | Best | 1.5-2x | Hopper+ |
| `int8_sq` | ~2x | Good | 1.5-2x | Any |
| `w4a8_awq` | ~2-3x | Good | 2-3x | Any |
| `int4_awq` | ~3-4x | Moderate | 2-3x | Any |
| `nvfp4` | ~3-4x | Moderate | 3-4x | Blackwell |

"Compression" is approximate memory reduction. "Speed Gain" is typical throughput improvement over BF16.
