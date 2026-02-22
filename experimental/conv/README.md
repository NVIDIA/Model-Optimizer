# Conv3D Implicit GEMM (Experimental)

Experimental Conv3D kernel prototype using implicit GEMM, with optional fused FP4 fake quantization for activations.

This code is kept under `experimental/` by design and is **not** part of the stable `modelopt.torch.quantization` API.

## Model Support

| Model/Framework | Supported | Notes |
|-----------------|-----------|-------|
| Video diffusion backbones using Conv3D | Partial | Intended for experimentation and microbenchmarking |
| Generic LLM backbones | No | Conv3D path is not relevant |
| End-to-end ModelOpt PTQ/QAT pipeline | No | Not wired into formal quantization/export/compress flows |

## Deployment

| Framework | Supported | Notes |
|-----------|-----------|-------|
| TensorRT-LLM | No | No formal export integration for this kernel path |
| vLLM | No | No integration |
| SGLang | No | No integration |
| PyTorch runtime (CUDA) | Yes (experimental) | JIT-compiles CUDA extension on first use |

## Usage

```python
import torch

from experimental.conv.implicit_gemm_cuda import conv3d_implicit_gemm_cuda
from modelopt.torch.quantization.tensor_quant import dynamic_block_quantize_op

x = torch.randn(1, 128, 21, 60, 106, device="cuda")
w = torch.randn(512, 128, 3, 3, 3, device="cuda")
block_size = 128

# Without FP4 activation quantization (drop-in-style Conv3D call)
out = conv3d_implicit_gemm_cuda(x, w, stride=(1, 1, 1), padding=(1, 1, 1))

# Optional block quantization of weights for experiments
w_q = dynamic_block_quantize_op(
    w,
    block_size,
    w.abs().max().unsqueeze(0),
    4,  # num_bits
    2,  # exponent_bits
    8,  # scale_num_bits
    4,  # scale_exponent_bits
)

# With FP4 activation fake quantization
out_q = conv3d_implicit_gemm_cuda(
    x,
    w_q,
    stride=(1, 1, 1),
    padding=(1, 1, 1),
    act_amax=x.abs().max().unsqueeze(0),
    quant_act=True,
    fp4_block_size=block_size,  # 128 or 256
)
```

## API

Function: `conv3d_implicit_gemm_cuda(...)` from `experimental/conv/implicit_gemm_cuda.py`

| Parameter | Description |
|-----------|-------------|
| `x` | Input tensor `[N, Cin, D, H, W]` |
| `w` | Weight tensor `[Cout, Cin, kD, kH, kW]` |
| `bias` | Optional bias `[Cout]` |
| `stride` | Convolution stride `(D, H, W)` |
| `padding` | Convolution padding `(D, H, W)` |
| `dilation` | Convolution dilation `(D, H, W)` |
| `act_amax` | Activation abs-max scalar tensor (required when `quant_act=True`) |
| `quant_act` | Enable FP4 fake quantization on activations |
| `FP4_BLOCK_SIZE` | FP4 quantization block size (`128` or `256`) |

## Status

Current state: **Prototype**

Known limitations:

- API is unstable and may change without notice.
- Not registered in core quantization module registries.
- Not covered by formal export/compress integration.
- CUDA extension compile latency on first invocation.
- Validation and performance coverage are limited to local experiments.

## Notes

- The CUDA kernel is JIT-compiled on first call (can take several seconds).
- Output shape matches `torch.nn.functional.conv3d`.
- FP4 path applies quantize-dequantize in-kernel for activation tiles.

## References

- Implicit GEMM-based convolution design patterns in GPU kernels.
- ModelOpt FP4-related quantization utilities in `modelopt.torch.quantization.tensor_quant`.
