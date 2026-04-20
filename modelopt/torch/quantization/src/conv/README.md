# Conv3D Implicit GEMM

Conv3D kernel using implicit GEMM with BF16 WMMA tensor cores and optional fused FP4 (E2M1) fake quantization.

This kernel is integrated into `modelopt.torch.quantization` via `_QuantConv3d` — when NVFP4 quantization is applied to an `nn.Conv3d` layer through ModelOpt PTQ, the implicit GEMM path is used automatically. We have only tested it on VAE Conv3D layers from video generation models (e.g. Wan2.2).

## Requirements

- **GPU:** SM80+ (Ampere or newer) for BF16 WMMA tensor cores
- **PyTorch:** CUDA toolkit with JIT C++ extension support (`torch.utils.cpp_extension`)
- **Grouped convolution is not supported** (groups must be 1)

## Data Types

| Stage | Precision |
|-------|-----------|
| Input / output tensors | FP32, FP16, or BF16 (dtype is preserved) |
| Internal compute | BF16 via WMMA m16n16k16 tensor cores |
| Accumulation | FP32 |
| FP4 activation quantization | E2M1 values, FP8 E4M3 scales |

## Integration with ModelOpt Quantization

When NVFP4 quantization is configured on a `Conv3d` layer via ModelOpt PTQ, the implicit GEMM kernel is used automatically during quantized inference. The integration is in `_QuantConv3d` (`modelopt/torch/quantization/nn/modules/quant_conv.py`):

- During **calibration**, the standard cuDNN path is used (faster).
- During **quantized inference** with NVFP4 input and weight quantizers, the kernel fuses activation FP4 quantization inside the GEMM.
- For all other quantization configs, the default cuDNN path is used as fallback.

## Usage

```python
import torch

from modelopt.torch.quantization.src.conv.implicit_gemm_cuda import conv3d_implicit_gemm_cuda
from modelopt.torch.quantization.tensor_quant import dynamic_block_quantize_op

x = torch.randn(1, 128, 21, 60, 106, device="cuda")
w = torch.randn(512, 128, 3, 3, 3, device="cuda")
block_size = 128

# Without FP4 activation quantization (drop-in-style Conv3D call)
out = conv3d_implicit_gemm_cuda(x, w, stride=(1, 1, 1), padding=(1, 1, 1))

# Optional FP4 block quantization of weights along the GEMM K dimension.
# The kernel's A-tile (activations) is quantized along K = Cin*kD*kH*kW,
# so weights must be flattened to [Cout, K] before quantizing to match.
Cout, Cin = w.shape[:2]
K = Cin * w.shape[2] * w.shape[3] * w.shape[4]
w_flat = w.reshape(Cout, K)
w_q_flat = dynamic_block_quantize_op(
    w_flat,
    block_size,
    w_flat.abs().max().unsqueeze(0),
    4,  # num_bits
    2,  # exponent_bits
    8,  # scale_num_bits
    4,  # scale_exponent_bits
)
w_q = w_q_flat.reshape_as(w)

# With FP4 activation fake quantization
out_q = conv3d_implicit_gemm_cuda(
    x,
    w_q,
    stride=(1, 1, 1),
    padding=(1, 1, 1),
    act_amax=x.abs().max().unsqueeze(0),
    quant_act=True,
    fp4_block_size=block_size,  # 16, 32, 64, 128, or 256
)
```

## API

### `conv3d_implicit_gemm_cuda`

`from modelopt.torch.quantization.src.conv.implicit_gemm_cuda import conv3d_implicit_gemm_cuda`

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
| `fp4_block_size` | FP4 quantization block size (`16`, `32`, `64`, `128`, or `256`) |

### `fp4_fake_quant`

`from modelopt.torch.quantization.src.conv.implicit_gemm_cuda import fp4_fake_quant`

Standalone FP4 (E2M1) blockwise fake quantization with FP8 E4M3 scale quantization. Uses the same CUDA device functions as the fused path inside the GEMM kernel.

| Parameter | Description |
|-----------|-------------|
| `x` | Input tensor (any shape; `numel` must be divisible by `block_size`) |
| `global_amax` | Scalar tensor — global abs max for scale computation |
| `block_size` | Number of elements per FP4 quantization block (default `16`) |

## Testing

```bash
# Run tests (requires GPU)
python -m pytest tests/gpu/torch/quantization/kernels/test_implicit_gemm.py -v
```

## Status

Current state: **Integrated** (registered in `QuantModuleRegistry`, auto-dispatched for NVFP4 Conv3D)

Known limitations:

- CUDA extension compile latency on first invocation (~seconds).
- Grouped convolution (`groups > 1`) is not supported. In the ModelOpt E2E flow, `_QuantConv3d` automatically falls back to the default cuDNN path for grouped convolutions.
- BF16 rounding error accumulates with the K dimension — expect max abs diff scaling roughly as `sqrt(K)` compared to cuDNN FP32.
- Inference only (`@torch.no_grad`) — not suitable for QAT backward pass.

## Notes

- The CUDA kernel is JIT-compiled on first call via `torch.utils.cpp_extension.load()`.
- Output shape matches `torch.nn.functional.conv3d`.
- FP4 path applies quantize-dequantize in-kernel for activation tiles (no extra global memory pass).
- Tile config: BLOCK_M=64, BLOCK_N=64, BLOCK_K=256, 8 warps (256 threads), ~70 KB shared memory per block.
- The kernel body is guarded by `#if __CUDA_ARCH__ >= 800` so it compiles as an empty stub when nvcc targets pre-Ampere archs (PyTorch's default `-gencode` list can include sm_75, which lacks BF16 WMMA fragments). Dispatch is enforced at runtime by `_get_cuda_module()` via `_MIN_SM_MAJOR = 8`.

## Files

| File | Role |
|------|------|
| `implicit_gemm_cuda.py` | Python API and JIT compilation |
| `implicit_gemm_kernel.cu` | CUDA kernel (BF16 WMMA + FP4 quantization) |
| `implicit_gemm_binding.cpp` | PyTorch C++ extension binding |

## References

- Implicit GEMM-based convolution design patterns in GPU kernels.
- ModelOpt FP4-related quantization utilities in `modelopt.torch.quantization.tensor_quant`.
