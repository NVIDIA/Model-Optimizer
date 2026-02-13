# Conv3D Implicit GEMM Kernels

CUDA and Triton kernels for Conv3D via implicit GEMM with optional FP4 fake quantization.

## Usage

```python
import torch
from modelopt.torch.quantization.conv_gemm.implicit_gemm_cuda import conv3d_implicit_gemm_cuda

x = torch.randn(1, 128, 21, 60, 106, device="cuda")
w = torch.randn(512, 128, 3, 3, 3, device="cuda")

# Without quantization (drop-in replacement for F.conv3d)
out = conv3d_implicit_gemm_cuda(x, w, stride=(1,1,1), padding=(1,1,1))

# With FP4 quantization
out = conv3d_implicit_gemm_cuda(
    x, w,
    stride=(1,1,1),
    padding=(1,1,1),
    act_amax=x.abs().max().unsqueeze(0),
    quant_act=True,
    FP4_BLOCK_SIZE=128,  # 128 or 256
)
```

The Triton kernel has the same API:

```python
from modelopt.torch.quantization.conv_gemm.implicit_gemm import conv3d_implicit_gemm_triton

out = conv3d_implicit_gemm_triton(x, w, stride=(1,1,1), padding=(1,1,1))
```

## Parameters

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
| `FP4_BLOCK_SIZE` | Quantization block size: `128` or `256` |

## Notes

- The CUDA kernel is JIT-compiled on first call (takes a few seconds).
- Both kernels return the same shape as `torch.nn.functional.conv3d`.
- FP4 quantization fuses the quantize-dequantize into the GEMM tile load, so there is minimal overhead vs the non-quantized path.