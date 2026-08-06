# SPDX-FileCopyrightText: Copyright (c) 2024 AMD, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AMD FP8 fused matmul — fuses input scaling + FP8 cast + GEMM in one kernel.

Falls back to torch._scaled_mm if Triton FP8 is not available.
"""
from __future__ import annotations
import torch

_TRITON_AVAILABLE = False
_TRITON_FP8_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
    try:
        _ = tl.float8e4m3fnuz
        _TRITON_FP8_AVAILABLE = True
    except AttributeError:
        try:
            _ = tl.float8e4nv
            _TRITON_FP8_AVAILABLE = True
        except AttributeError:
            pass
except ImportError:
    pass


def fused_fp8_mm_available() -> bool:
    """Check if the fused Triton FP8 matmul is available."""
    return _TRITON_AVAILABLE and _TRITON_FP8_AVAILABLE


def _torch_scaled_mm_fallback(x: "torch.Tensor",
                               W_fp8: "torch.Tensor",
                               scale_x: "torch.Tensor",
                               scale_w: "torch.Tensor") -> "torch.Tensor":
    """Fallback: use torch._scaled_mm (no Triton kernel)."""
    x_fp8 = x.contiguous().to(torch.float8_e4m3fnuz)
    return torch._scaled_mm(x_fp8, W_fp8.T,
                            scale_a=scale_x, scale_b=scale_w,
                            out_dtype=torch.float16)


def amd_fp8_fused_mm(x: "torch.Tensor",
                      W_fp8: "torch.Tensor",
                      scale_x: "torch.Tensor",
                      scale_w: "torch.Tensor") -> "torch.Tensor":
    """Fused FP8 matmul — scale+cast+GEMM in one pass when Triton FP8 available.
    
    Args:
        x: Input [M, K] float16
        W_fp8: Weight [N, K] float8_e4m3fnuz (pre-quantized)
        scale_x: Input scale scalar (float32)
        scale_w: Weight scale scalar (float32)
    
    Returns:
        Output [M, N] float16
    """
    if not _TRITON_FP8_AVAILABLE:
        return _torch_scaled_mm_fallback(x, W_fp8, scale_x, scale_w)
    
    # Use Triton kernel only if shapes work cleanly
    M, K = x.shape
    N = W_fp8.shape[0]
    
    # Triton kernel: fuse scale, clamp, fp8-cast into the GEMM
    try:
        import triton
        import triton.language as tl
        
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 32
        
        @triton.jit
        def _kernel(xp, wp, op, sx, sw,
                    M, N, K,
                    sxm, sxk, swn, swk, som, son,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
            pm = tl.program_id(0); pn = tl.program_id(1)
            om = pm*BM + tl.arange(0,BM)
            on = pn*BN + tl.arange(0,BN)
            acc = tl.zeros((BM,BN), tl.float32)
            for k in range(tl.cdiv(K,BK)):
                ok = k*BK + tl.arange(0,BK)
                xt = tl.load(xp+om[:,None]*sxm+ok[None,:]*sxk,
                             mask=(om[:,None]<M)&(ok[None,:]<K), other=0.0).to(tl.float32)
                xt_fp8 = (xt/sx).clamp(-448.,448.).to(tl.float8e4m3fnuz)
                wt = tl.load(wp+on[:,None]*swn+ok[None,:]*swk,
                             mask=(on[:,None]<N)&(ok[None,:]<K), other=0.0)
                acc += tl.dot(xt_fp8, wt.T, allow_tf32=False).to(tl.float32)
            out = (acc*sx*sw).to(tl.float16)
            tl.store(op+om[:,None]*som+on[None,:]*son, out,
                     mask=(om[:,None]<M)&(on[None,:]<N))
        
        out = torch.empty((M,N), dtype=torch.float16, device=x.device)
        grid = (triton.cdiv(M,BLOCK_M), triton.cdiv(N,BLOCK_N))
        _kernel[grid](x, W_fp8, out, float(scale_x), float(scale_w),
                      M, N, K,
                      x.stride(0), x.stride(1),
                      W_fp8.stride(0), W_fp8.stride(1),
                      out.stride(0), out.stride(1),
                      BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K)
        return out
    except Exception:
        return _torch_scaled_mm_fallback(x, W_fp8, scale_x, scale_w)
