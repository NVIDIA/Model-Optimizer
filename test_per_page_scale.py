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

# ruff: noqa: N803 — B/KV/NKV are batch/shape names by convention, matching the kernel files

"""Numerical correctness + accuracy benefit of per-page (method 2) NVFP4 KV$ scaling.

Method 2 = each KV-cache page (one BLOCK_N=128 tile) carries its OWN global scale (amax/(6*448)):
complete pages are baked once on write with their own amax; the in-progress (last) page stays raw
bf16 (the "master pool") and is re-quantized on read each step from its own amax. Contrast with
method 1 = a single per-tensor scale frozen after prefill.

Test 1 (correctness): per-page on-WRITE+skip must equal per-page on-READ (every tile FQ'd with its
own amax), since baking a complete tile is the SAME op the on-read path runs. Expect fp32 bit-exact;
bf16 = pure storage residual (a real FP4 cache -> 0).

Test 2 (benefit): with an outlier in a LATER page, a scale frozen from the FIRST page saturates the
E4M3 block scales of the outlier page (method 1), while per-page (method 2) re-centers per page.
Expect: per-page attention is CLOSER to the no-fakequant reference than the frozen scale.
"""

import torch

from modelopt.torch.kernels.common.attention.decode_attention import (
    attention_decode,
    fake_quant_kv_onwrite,
)
from modelopt.torch.kernels.common.attention.triton_fa import attention as triton_attention

DEV, HEAD_DIM, NQ, NKV, PAGE = "cuda", 128, 64, 8, 16
TILE = 128  # the per-page (BLOCK_N) scale granularity in the kernels
SCALE = HEAD_DIM**-0.5
torch.manual_seed(0)


def _setup(B, KV, dtype):
    nblk = (KV + PAGE - 1) // PAGE
    tb = B * nblk
    q = torch.randn(B, NQ, HEAD_DIM, device=DEV, dtype=dtype)
    kc = torch.randn(tb, PAGE, NKV, HEAD_DIM, device=DEV, dtype=dtype)
    vc = torch.randn(tb, PAGE, NKV, HEAD_DIM, device=DEV, dtype=dtype)
    bt = torch.arange(tb, device=DEV, dtype=torch.int32).view(B, nblk)
    sk = torch.full((B,), KV, device=DEV, dtype=torch.int32)
    return q, kc, vc, bt, sk, nblk


def _bake_per_page(kc, vc, bt, KV, B, nvfp4):
    """On-write: bake every COMPLETE 128-tile in place with its own per-tile amax (method 2)."""
    lo = torch.zeros(B, device=DEV, dtype=torch.int32)
    hi = torch.full((B,), (KV // TILE) * TILE, device=DEV, dtype=torch.int32)
    fake_quant_kv_onwrite(
        kc, vc, bt, lo, hi, lo, hi, page_size=PAGE, nvfp4=nvfp4, decode=False, per_page_scale=True
    )


def _run_correctness(dtype, B, KV, nvfp4):
    q, kc, vc, bt, sk, _ = _setup(B, KV, dtype)
    # Reference: per-page applied purely on READ (cache stays raw -> cache_quantized=False forces the
    # decode kernel to FQ every tile, each with its own in-kernel amax).
    ref = attention_decode(
        q,
        kc.clone(),
        vc.clone(),
        bt,
        sk,
        softmax_scale=SCALE,
        page_size=PAGE,
        nvfp4=nvfp4,
        per_page_scale=True,
    )
    # Optimized: bake complete pages on write, then read them as-is (cache_quantized=True); only the
    # trailing page is FQ'd on read.
    kc2, vc2 = kc.clone(), vc.clone()
    _bake_per_page(kc2, vc2, bt, KV, B, nvfp4 & {"k", "v"})
    opt = attention_decode(
        q,
        kc2,
        vc2,
        bt,
        sk,
        softmax_scale=SCALE,
        page_size=PAGE,
        nvfp4=nvfp4,
        per_page_scale=True,
        k_cache_quantized=("k" in nvfp4),
        v_cache_quantized=("v" in nvfp4),
    )
    d = (ref.float() - opt.float()).abs()
    return d.max().item(), d.mean().item()


def _attn_frozen(q, kc, vc, bt, sk, nvfp4, gk, gv):
    """Method 1: single frozen per-tensor scale (gk/gv) applied on read (no baking)."""
    return attention_decode(
        q,
        kc.clone(),
        vc.clone(),
        bt,
        sk,
        softmax_scale=SCALE,
        page_size=PAGE,
        nvfp4=nvfp4,
        attn_global_scales={"k": gk, "v": gv},
    )


if __name__ == "__main__":
    print(f"GPU {torch.cuda.get_device_name(0)} | head_dim={HEAD_DIM} q={NQ} kv={NKV} page={PAGE}")

    # ---- Test 1: on-write+skip == on-read, per page ----
    print("\n[1] per-page on-write+skip vs on-read (expect fp32 BIT-EXACT)")
    cases = [
        (1, 4096),
        (4, 8192),
        (8, 16384),
        (1, 130),
        (3, 200),
        (2, 33),
        (1, 127),
        (1, 256),
        (5, 4097),
    ]
    bad = 0
    for nvfp4 in ({"k", "v"}, {"q", "k", "p", "v"}):
        for dtype in (torch.float32, torch.bfloat16):
            tag = "".join(sorted(nvfp4))
            worst_max = worst_mean = 0.0
            for B, KV in cases:
                mx, mn = _run_correctness(dtype, B, KV, nvfp4)
                worst_max, worst_mean = max(worst_max, mx), max(worst_mean, mn)
            # fp32: bit-exact up to Triton FMA-context epsilon (~1 ULP = 2^-24); the ref/opt paths
            # compile differently (K/V_CACHE_QUANTIZED False vs True) so the fp32 dot can reorder by
            # 1 ULP. bf16: pure bf16-cache storage residual (a real FP4 cache -> 0).
            gate = 1e-5 if dtype == torch.float32 else 5e-2
            ok = worst_max <= gate
            bad += not ok
            kind = "bit-exact(<=1ULP)" if dtype == torch.float32 else "storage-residual"
            print(
                f"  nvfp4={tag:5s} {dtype!s:14s} max={worst_max:.3e} mean={worst_mean:.3e}"
                f"  [{kind}] {'OK' if ok else 'FAIL'}"
            )

    # ---- Test 2: per-page beats a first-page-frozen scale under a later-page outlier ----
    print("\n[2] later-page outlier: per-page should be CLOSER to no-FQ than a frozen scale")
    B, KV, nvfp4 = 2, 512, {"k", "v"}  # 4 pages of 128
    q, kc, vc, bt, sk, _ = _setup(B, KV, torch.float32)
    # Inject a large outlier into the LAST page (keys [384, 512)) of both K and V.
    last_lo = (KV // TILE - 1) * TILE
    flat_k = kc.view(B, -1, NKV, HEAD_DIM)  # [B, KV, NKV, HEAD_DIM] (contiguous arange block_table)
    flat_v = vc.view(B, -1, NKV, HEAD_DIM)
    flat_k[:, last_lo:KV] *= 8.0
    flat_v[:, last_lo:KV] *= 8.0
    exact = attention_decode(q, kc.clone(), vc.clone(), bt, sk, softmax_scale=SCALE, page_size=PAGE)
    # Method 1 frozen scale seeded from the FIRST page only (mimics freeze-on-first-chunk before the
    # outlier page arrives).
    gk = flat_k[:, :TILE].float().abs().max() / (6 * 448) + 1e-30
    gv = flat_v[:, :TILE].float().abs().max() / (6 * 448) + 1e-30
    frozen = _attn_frozen(q, kc, vc, bt, sk, nvfp4, gk, gv)
    kc2, vc2 = kc.clone(), vc.clone()
    _bake_per_page(kc2, vc2, bt, KV, B, nvfp4)
    perpage = attention_decode(
        q,
        kc2,
        vc2,
        bt,
        sk,
        softmax_scale=SCALE,
        page_size=PAGE,
        nvfp4=nvfp4,
        per_page_scale=True,
        k_cache_quantized=True,
        v_cache_quantized=True,
    )
    e_frozen = (exact - frozen).abs().mean().item()
    e_perpage = (exact - perpage).abs().mean().item()
    better = e_perpage < e_frozen
    bad += not better
    print(
        f"  |exact-frozen|={e_frozen:.4e}  |exact-perpage|={e_perpage:.4e}"
        f"  -> per-page {'BETTER' if better else 'NOT better'}"
        f" ({e_frozen / max(e_perpage, 1e-30):.1f}x)"
    )

    # ---- Test 3: prefill per-page is consistent with decode per-page ----
    # A single-token prefill over the same baked cache must match the (validated) decode path. The
    # boundary page uses SCALE_PAGE=128 so prefill reads baked 128-pages as-is even with a smaller
    # autotuned BLOCK_N. Compare on a 128-ALIGNED KV (no trailing page) so the only difference is
    # tl.dot-vs-tl.sum kernel numerics; the unaligned case differs only on the <128-token tail page
    # (prefill quantizes it per-BLOCK_N, a finer-but-consistent sub-page scale) and is reported FYI.
    print("\n[3] prefill(per-page) vs decode(per-page): aligned must match (kernel numerics)")
    torch.backends.cuda.matmul.allow_tf32 = False  # isolate per-page logic from TF32 dot error
    for KV, gated in ((512, True), (600, False)):
        q, kc, vc, bt, sk, _ = _setup(2, KV, torch.float32)
        nvfp4 = {"k", "v"}
        kc2, vc2 = kc.clone(), vc.clone()
        _bake_per_page(kc2, vc2, bt, KV, 2, nvfp4)
        dec = attention_decode(
            q,
            kc2.clone(),
            vc2.clone(),
            bt,
            sk,
            softmax_scale=SCALE,
            page_size=PAGE,
            nvfp4=nvfp4,
            per_page_scale=True,
            k_cache_quantized=True,
            v_cache_quantized=True,
        )
        pre = triton_attention(
            q,
            k=torch.empty(0, NKV, HEAD_DIM, device=DEV),
            v=torch.empty(0, NKV, HEAD_DIM, device=DEV),
            b_start_loc=torch.arange(2, device=DEV, dtype=torch.int32),
            b_seq_len=torch.ones(2, device=DEV, dtype=torch.int32),
            max_input_len=1,
            is_causal=False,
            softmax_scale=SCALE,
            b_start_loc_k=None,
            b_seq_len_k=sk,
            max_input_len_k=KV,
            k_cache=kc2,
            v_cache=vc2,
            block_table=bt,
            page_size=PAGE,
            nvfp4=nvfp4,
            per_page_scale=True,
        )
        d = (dec.float() - pre.float()).abs()
        tail = "aligned (no tail)" if KV % TILE == 0 else f"tail={KV % TILE} (per-BLOCK_N, FYI)"
        if gated:
            ok = d.max().item() < 5e-3
            bad += not ok
            print(
                f"  KV={KV} {tail}: max={d.max().item():.3e} mean={d.mean().item():.3e}"
                f"  {'OK' if ok else 'FAIL'}"
            )
        else:
            print(f"  KV={KV} {tail}: max={d.max().item():.3e} mean={d.mean().item():.3e}  [FYI]")

    print(f"\n{'ALL PASS' if bad == 0 else f'{bad} FAILURE(S)'}")
