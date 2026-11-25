#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 AMD, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
AMD LLaMA FP8 Inference Example
================================
Demonstrates full FP8 quantization of a LLaMA-style model on AMD MI300X.
Shows both attention QKV and FFN projection quantization.

Usage:
    python examples/amd_llama_fp8_inference.py
    python examples/amd_llama_fp8_inference.py --model-size 70b --batch-size 256
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

if not getattr(torch.version, "hip", None):
    raise RuntimeError("AMD ROCm required. Run on MI300X/MI325X.")

import modelopt.torch.quantization as mtq
from modelopt._rocm_compat import (
    extract_fp8_scales, convert_to_static_fp8,
    warmup_for_llama, get_llama_ffn_shapes, 
    kv_cache_memory_savings, quantize_kv_cache_fp8
)


class LLaMAAttention(nn.Module):
    """LLaMA-style multi-head attention with GQA support."""
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # Repeat KV heads for GQA
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn.transpose(1, 2).reshape(B, T, -1))


class LLaMAFFN(nn.Module):
    """LLaMA SwiGLU FFN."""
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up   = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LLaMALayer(nn.Module):
    def __init__(self, hidden, intermediate, n_heads, n_kv_heads):
        super().__init__()
        self.attn = LLaMAAttention(hidden, n_heads, n_kv_heads)
        self.ffn  = LLaMAFFN(hidden, intermediate)
        self.norm1 = nn.RMSNorm(hidden)
        self.norm2 = nn.RMSNorm(hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


MODEL_CONFIGS = {
    "7b":  dict(hidden=4096, intermediate=11008, n_heads=32, n_kv_heads=32, n_layers=2),
    "13b": dict(hidden=5120, intermediate=13824, n_heads=40, n_kv_heads=40, n_layers=2),
    "70b": dict(hidden=8192, intermediate=28672, n_heads=64, n_kv_heads=8,  n_layers=2),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="7b", choices=list(MODEL_CONFIGS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len",   type=int, default=1)
    parser.add_argument("--iters",     type=int, default=200)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    H, I = cfg["hidden"], cfg["intermediate"]
    BS, T = args.batch_size, args.seq_len
    
    print(f"\n{'='*60}")
    print(f" AMD LLaMA-{args.model_size} FP8 Inference  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f" BS={BS}, T={T}, H={H}, I={I}")
    print(f"{'='*60}")

    # Build a 2-layer model (representative of one full decode step per layer)
    model = nn.Sequential(*[
        LLaMALayer(H, I, cfg["n_heads"], cfg["n_kv_heads"])
        for _ in range(cfg["n_layers"])
    ]).cuda().half()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters (2 layers): {params/1e6:.1f}M")

    # Calibrate
    print("\n── Calibrating with AMD_FP8_DEFAULT_CFG ──")
    x_cal = torch.randn(4, T, H, device="cuda", dtype=torch.float16)
    mtq.quantize(model, mtq.AMD_FP8_DEFAULT_CFG, forward_loop=lambda m: m(x_cal))
    scales = extract_fp8_scales(model)
    print(f"   {len(scales)} layer scales extracted")
    
    # Rebuild fresh model and convert to FP8 deployment
    model_fp16 = nn.Sequential(*[
        LLaMALayer(H, I, cfg["n_heads"], cfg["n_kv_heads"])
        for _ in range(cfg["n_layers"])
    ]).cuda().half()
    
    model_fp8 = nn.Sequential(*[
        LLaMALayer(H, I, cfg["n_heads"], cfg["n_kv_heads"])
        for _ in range(cfg["n_layers"])
    ]).cuda().half()
    model_fp8 = convert_to_static_fp8(model_fp8, scales)
    
    # Warmup hipBLASLt
    print(f"\n── Warming hipBLASLt for LLaMA-{args.model_size} ──")
    warmup_for_llama(args.model_size, batch_sizes=[BS])

    # KV cache savings estimate
    savings = kv_cache_memory_savings(
        seq_len=2048, n_heads=cfg["n_kv_heads"],
        head_dim=H // cfg["n_heads"],
        batch_size=BS, n_layers=32
    )
    print(f"\n── KV Cache Savings (BS={BS}, seq=2048, 32 layers) ──")
    print(f"   FP16: {savings['fp16_gb']:.2f} GB → FP8: {savings['fp8_gb']:.2f} GB "
          f"(saves {savings['savings_gb']:.2f} GB = {savings['savings_ratio']*100:.0f}%)")

    # Benchmark
    print(f"\n── Benchmark (BS={BS}, T={T}) ──")
    x = torch.randn(BS, T, H, device="cuda", dtype=torch.float16)
    WARMUP = 20

    for _ in range(WARMUP): model_fp16(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters): model_fp16(x)
    torch.cuda.synchronize()
    t16 = (time.perf_counter() - t0) / args.iters * 1000

    for _ in range(WARMUP): model_fp8(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters): model_fp8(x)
    torch.cuda.synchronize()
    t8 = (time.perf_counter() - t0) / args.iters * 1000

    spd = t16 / t8
    # Approximate TFLOPS for 2-layer model (attn + ffn GEMMs)
    # FFN: 3 GEMMs per layer | Attn: 4 GEMMs per layer
    gemm_flops = 2 * cfg["n_layers"] * (
        3 * BS * T * H * I +                           # FFN
        BS * T * (H*H + 2*H*H//cfg["n_heads"]*cfg["n_kv_heads"])  # QKV+O approx
    )
    tfl8 = gemm_flops / (t8 / 1000) / 1e12

    print(f"   FP16:    {t16:.3f} ms/iter")
    print(f"   FP8:     {t8:.3f} ms/iter")
    print(f"   Speedup: {spd:.2f}x")
    print(f"   FP8 TFLOPS (approx): {tfl8:.1f}")

    print(f"\n{'='*60}")
    print(f" Result: {spd:.2f}x FP16 on AMD {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
