# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conv3D Implicit GEMM with BF16 WMMA Tensor Cores and optional fused FP4 quantization.

CUDA kernel source: implicit_gemm_kernel.cu
C++ binding:        implicit_gemm_binding.cpp
"""

import os

import torch
import torch.nn.functional as F

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

_cuda_module = None


_MIN_SM_MAJOR = 8  # BF16 WMMA tensor cores require SM80+ (Ampere and newer)


def _get_cuda_module():
    """Get or compile the CUDA module."""
    global _cuda_module
    if _cuda_module is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This kernel requires a CUDA GPU.")
        major, minor = torch.cuda.get_device_capability()
        if major < _MIN_SM_MAJOR:
            raise RuntimeError(
                f"This kernel requires SM{_MIN_SM_MAJOR}0+ (Ampere or newer) for BF16 WMMA "
                f"tensor cores, but the current GPU has SM{major}{minor}."
            )

        from torch.utils.cpp_extension import load

        _cuda_module = load(
            name="conv3d_implicit_gemm_cuda_v20_wmma",
            sources=[
                os.path.join(_KERNEL_DIR, "implicit_gemm_binding.cpp"),
                os.path.join(_KERNEL_DIR, "implicit_gemm_kernel.cu"),
            ],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "--ptxas-options=-v",
                "-std=c++17",
            ],
        )
    return _cuda_module


def _triple(v) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    assert len(v) == 3
    return (int(v[0]), int(v[1]), int(v[2]))


def _pad6(padding) -> tuple[int, int, int, int, int, int]:
    if isinstance(padding, int):
        p = int(padding)
        return (p, p, p, p, p, p)
    if len(padding) == 3:
        pd, ph, pw = map(int, padding)
        return (pw, pw, ph, ph, pd, pd)
    assert len(padding) == 6
    return tuple(map(int, padding))  # type: ignore[return-value]


@torch.no_grad()
def conv3d_implicit_gemm_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    act_amax: torch.Tensor | None = None,
    quant_act: bool = False,
    fp4_block_size: int = 256,
) -> torch.Tensor:
    """Conv3D via implicit GEMM with BF16 WMMA tensor cores.

    Args:
        x: Input tensor [N, Cin, D, H, W]
        w: Weight tensor [Cout, Cin, kD, kH, kW]
        bias: Optional bias tensor [Cout]
        stride: Convolution stride (D, H, W)
        padding: Convolution padding (D, H, W)
        dilation: Convolution dilation (D, H, W)
        act_amax: Activation max value for FP4 quantization
        quant_act: Whether to apply FP4 quantization to activations
        fp4_block_size: FP4 quantization block size (16, 32, 64, 128, or 256)

    Returns:
        Output tensor [N, Cout, OD, OH, OW]

    Raises:
        ValueError: If fp4_block_size is not one of {16, 32, 64, 128, 256}.
    """
    valid_block_sizes = {16, 32, 64, 128, 256}
    if fp4_block_size not in valid_block_sizes:
        raise ValueError(
            f"fp4_block_size must be one of {sorted(valid_block_sizes)}, got {fp4_block_size}"
        )

    cuda_mod = _get_cuda_module()

    if x.ndim != 5 or w.ndim != 5:
        raise ValueError(f"Expected 5D tensors, got x.ndim={x.ndim}, w.ndim={w.ndim}")
    n_batch, cin, d, h, w_in = x.shape
    cout, cin_w, kd, kh, kw = w.shape
    if cin_w != cin:
        raise ValueError(
            f"Grouped convolution is not supported (x has {cin} input channels, "
            f"w has {cin_w}). This kernel requires groups=1."
        )

    sd, sh, sw = _triple(stride)
    dd, dh, dw = _triple(dilation)
    pad_wl, pad_wr, pad_hl, pad_hr, pad_dl, pad_dr = _pad6(padding)

    x_pad = F.pad(x, (pad_wl, pad_wr, pad_hl, pad_hr, pad_dl, pad_dr))
    dp = d + pad_dl + pad_dr
    hp = h + pad_hl + pad_hr
    wp = w_in + pad_wl + pad_wr

    od = (dp - (dd * (kd - 1) + 1)) // sd + 1
    oh = (hp - (dh * (kh - 1) + 1)) // sh + 1
    ow = (wp - (dw * (kw - 1) + 1)) // sw + 1

    m = n_batch * od * oh * ow
    k = cin * kd * kh * kw

    w_flat = w.reshape(cout, k).transpose(0, 1).contiguous()

    x_pad = x_pad.float().contiguous()
    w_flat = w_flat.float().contiguous()

    has_bias = bias is not None
    bias_t = bias.float().contiguous() if has_bias else torch.empty(0, device=x.device)  # type: ignore[union-attr]

    if quant_act and act_amax is None:
        raise ValueError("act_amax is required when quant_act=True")

    do_quant = quant_act
    amax_t = act_amax.float().contiguous() if do_quant else torch.empty(0, device=x.device)  # type: ignore[union-attr]

    y_flat = cuda_mod.conv3d_implicit_gemm_cuda(
        x_pad,
        w_flat,
        bias_t,
        amax_t,
        n_batch,
        cin,
        dp,
        hp,
        wp,
        cout,
        od,
        oh,
        ow,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        dd,
        dh,
        dw,
        m,
        k,
        do_quant,
        has_bias,
        fp4_block_size,
    )

    y = y_flat.view(n_batch, od, oh, ow, cout).permute(0, 4, 1, 2, 3).contiguous()
    return y.to(x.dtype)


@torch.no_grad()
def fp4_fake_quant(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Standalone FP4 fake quantization using the same CUDA device functions as the GEMM kernel.

    Applies blockwise FP4 (E2M1) quantize-dequantize with FP8 E4M3 scale quantization.

    Args:
        x: Input tensor (any shape, numel must be divisible by block_size).
        global_amax: Scalar tensor — global abs max for scale computation.
        block_size: Number of elements per FP4 quantization block.

    Returns:
        Fake-quantized tensor with same shape and dtype as input.
    """
    cuda_mod = _get_cuda_module()

    orig_shape = x.shape
    orig_dtype = x.dtype
    x_f32 = x.float().contiguous()
    amax_f32 = global_amax.float().contiguous()

    assert x_f32.numel() % block_size == 0, (
        f"numel ({x_f32.numel()}) must be divisible by block_size ({block_size})"
    )

    y = cuda_mod.fp4_fake_quant_cuda(x_f32, amax_f32, block_size)
    return y.view(orig_shape).to(orig_dtype)
