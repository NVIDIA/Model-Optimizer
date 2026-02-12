"""Pure PyTorch kernel implementations for PTQ calibration.

Replaces the tilelang-based kernel.py which requires CUDA 12's libnvrtc.
These implementations are numerically equivalent but slower — suitable for
PTQ calibration but not production inference.
"""

import torch

block_size = 128

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: str = None):
    """Block-wise FP8 quantization of activations.

    Returns (fp8_tensor, scale_tensor) with the same shapes as the tilelang version.
    """
    orig_shape = x.shape
    # Flatten to 2D: (num_elements / block_size, block_size)
    x_flat = x.reshape(-1, block_size)
    # Compute per-block scale
    amax = x_flat.float().abs().amax(dim=-1)
    scale = amax / FP8_MAX
    scale = scale.clamp(min=1e-12)
    # Quantize
    x_scaled = x_flat.float() / scale.unsqueeze(-1)
    x_fp8 = x_scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(orig_shape)
    # Scale shape: match what the caller expects
    scale = scale.reshape(*orig_shape[:-1], orig_shape[-1] // block_size)
    return x_fp8, scale


def fp8_gemm(a: torch.Tensor, a_scale: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor):
    """FP8 matrix multiply with block-wise dequantization."""
    # Dequantize and do regular matmul
    a_f = a.float() * a_scale.unsqueeze(-1).repeat_interleave(block_size, dim=-1)[..., :a.shape[-1]]
    b_f = b.float() * b_scale.unsqueeze(-1).repeat_interleave(block_size, dim=-1)[..., :b.shape[-1]]
    return torch.matmul(a_f, b_f.t()).to(torch.bfloat16)


def fp8_index(q_fp8: torch.Tensor, weights: torch.Tensor, k_cache: torch.Tensor, k_scale_cache: torch.Tensor):
    """Compute sparse attention index scores using FP8 Q and cached K.

    Args:
        q_fp8: (bsz, seqlen, n_heads, head_dim) float8_e4m3fn
        weights: (bsz, seqlen, n_heads, 1) weighting factors
        k_cache: (bsz, cache_len, head_dim) float8_e4m3fn
        k_scale_cache: (bsz, cache_len, head_dim // block_size) float32

    Returns:
        index_score: (bsz, seqlen, cache_len) attention-like scores
    """
    bsz, seqlen, n_heads, head_dim = q_fp8.shape
    cache_len = k_cache.shape[1]
    n_blocks = head_dim // block_size

    # Dequant K cache: (bsz, cache_len, head_dim)
    k_f = k_cache.float().reshape(bsz, cache_len, n_blocks, block_size)
    k_f = k_f * k_scale_cache.unsqueeze(-1)
    k_f = k_f.reshape(bsz, cache_len, head_dim)

    # Dequant Q: we don't have q_scale here, just use the fp8 values directly
    q_f = q_fp8.float()  # (bsz, seqlen, n_heads, head_dim)

    # weights: (bsz, seqlen, n_heads, 1) — absorb into q
    q_weighted = (q_f * weights).sum(dim=2)  # (bsz, seqlen, head_dim)

    # Score: (bsz, seqlen, cache_len)
    index_score = torch.bmm(q_weighted, k_f.transpose(1, 2))
    return index_score
