# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ruff: noqa: N803, N806 — Triton kernel args and attention dims use uppercase by convention

"""GPU tests for in-kernel NVFP4 fake quantization of the attention BMM operands.

Validates, against PyTorch references using ``torch.float8_e4m3fn``:
  - the fp32 E4M3 emulation (block scale) is bit-exact to hardware fp8;
  - the decode and prefill kernels' in-kernel NVFP4 reproduce a materialized NVFP4
    attention for every operand subset {q, k, p, v}.
"""

import pytest
import torch

pytest.importorskip("triton")

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    import triton
    import triton.language as tl

    from modelopt.torch.kernels.common.attention.decode_attention import attention_decode
    from modelopt.torch.kernels.common.attention.triton_fa import attention
    from modelopt.torch.kernels.quantization.attention.nvfp4_fakequant import e4m3_emulate

    @triton.jit
    def _e4m3_emulate_kernel(X, Y, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Y + i, e4m3_emulate(tl.load(X + i)))


E2M1, FP8, BLK = 6.0, 448.0, 16


# --- PyTorch NVFP4 reference (matches the kernels' E2M1 + E4M3-block + global scheme) ---
def _e4m3(x):
    ax = x.abs().clamp(max=FP8).float()
    o = torch.zeros_like(ax)
    nz = ax > 0
    e = torch.floor(torch.log2(ax[nz])).clamp(min=-6.0)
    s = torch.exp2(e - 3.0)
    o[nz] = torch.round(ax[nz] / s) * s
    return torch.sign(x) * o.clamp(max=FP8)


def _e2m1(b):
    return torch.where(
        b <= 0.25,
        0.0,
        torch.where(
            b < 0.75,
            0.5,
            torch.where(
                b <= 1.25,
                1.0,
                torch.where(
                    b < 1.75,
                    1.5,
                    torch.where(
                        b <= 2.5, 2.0, torch.where(b < 3.5, 3.0, torch.where(b <= 5.0, 4.0, 6.0))
                    ),
                ),
            ),
        ),
    )


def _qdq(x, axis, per_row=False):
    xt = x.transpose(axis, -1).float()
    gmax = xt.abs().amax(-1, keepdim=True) if per_row else xt.abs().max()
    g = gmax / (E2M1 * FP8)
    n = xt.shape[-1]
    xr = xt.reshape(*xt.shape[:-1], n // BLK, BLK)
    bamax = xr.abs().amax(-1, keepdim=True)
    gb = g.unsqueeze(-1) if per_row else g
    scale = _e4m3((bamax / (E2M1 * gb)).clamp(max=FP8)) * gb
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    xq = torch.sign(xr) * _e2m1(xr.abs() / scale) * scale
    return xq.reshape(xt.shape).transpose(axis, -1)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
def test_e4m3_emulation_bit_exact():
    """The fp32 E4M3 emulation used for the block scale matches torch.float8_e4m3fn."""
    torch.manual_seed(0)
    x = (torch.rand(4096, device="cuda") * 900 - 450).float()  # full range incl. >448
    y = torch.empty_like(x)
    _e4m3_emulate_kernel[(1,)](x, y, N=4096)
    ref = x.clamp(-FP8, FP8).to(torch.float8_e4m3fn).to(torch.float32)
    assert (y - ref).abs().max().item() == 0.0


def _paged_decode(k, v, seq_len, page=16):
    b, kvh, _, d = k.shape
    nb = (seq_len + page - 1) // page
    kc = torch.zeros(b * nb, page, kvh, d, device=k.device, dtype=k.dtype)
    vc = torch.zeros_like(kc)
    bt = torch.arange(b * nb, device=k.device, dtype=torch.int32).view(b, nb)
    for bb in range(b):
        for blk in range(nb):
            ts, te = blk * page, min((blk + 1) * page, seq_len)
            kc[bb * nb + blk, : te - ts] = k[bb, :, ts:te].transpose(0, 1)
            vc[bb * nb + blk, : te - ts] = v[bb, :, ts:te].transpose(0, 1)
    return kc, vc, bt


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
@pytest.mark.parametrize("ops", [(), ("k", "v"), ("q", "k"), ("p", "v"), ("q", "k", "p", "v")])
def test_decode_kernel_nvfp4(ops):
    """Decode kernel NVFP4 == materialized NVFP4 decode (single tile, num_kv_splits=1)."""
    b, H, KVH, S, d = 2, 4, 2, 112, 64
    sm = 1.0 / (d**0.5)
    torch.manual_seed(0)
    q = torch.randn(b, H, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, KVH, S, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, KVH, S, d, device="cuda", dtype=torch.float16)
    kc, vc, bt = _paged_decode(k, v, S)
    sl = torch.full((b,), S, device="cuda", dtype=torch.int32)
    got = attention_decode(
        q, kc, vc, bt, sl, softmax_scale=sm, page_size=16, num_kv_splits=1, nvfp4=set(ops)
    )
    g = H // KVH
    ref = torch.empty_like(q)
    for bb in range(b):
        for h in range(H):
            qh, kh, vh = q[bb, h].float(), k[bb, h // g].float(), v[bb, h // g].float()
            if "q" in ops:
                qh = _qdq(qh, 0)
            if "k" in ops:
                kh = _qdq(kh, -1)
            p = torch.softmax((qh @ kh.t()) * sm, dim=-1)
            if "p" in ops:
                p = _qdq(p, 0)  # 1-D row -> per-row global
            if "v" in ops:
                vh = _qdq(vh, 0)
            ref[bb, h] = (p @ vh).to(ref.dtype)
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-3, atol=3e-3)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
@pytest.mark.parametrize("ops", [(), ("k", "v"), ("q", "k"), ("p", "v"), ("q", "k", "p", "v")])
def test_prefill_kernel_nvfp4(ops):
    """Prefill kernel NVFP4 == materialized NVFP4 attention (single tile, non-causal)."""
    S, H, KVH, d = 16, 4, 2, 64
    sm = 1.0 / (d**0.5)
    torch.manual_seed(0)
    q = torch.randn(S, H, d, device="cuda", dtype=torch.float16)
    k = torch.randn(S, KVH, d, device="cuda", dtype=torch.float16)
    v = torch.randn(S, KVH, d, device="cuda", dtype=torch.float16)
    bsl = torch.tensor([0], device="cuda", dtype=torch.int32)
    bseq = torch.tensor([S], device="cuda", dtype=torch.int32)
    got = attention(q, k, v, bsl, bseq, S, is_causal=False, softmax_scale=sm, nvfp4=set(ops))
    g = H // KVH
    ref = torch.empty_like(q)
    for h in range(H):
        qh, kh, vh = q[:, h].float(), k[:, h // g].float(), v[:, h // g].float()
        if "q" in ops:
            qh = _qdq(qh, -1)
        if "k" in ops:
            kh = _qdq(kh, -1)
        sc = (qh.half().float() @ kh.half().float().t()) * sm
        p = torch.softmax(sc, dim=-1)
        if "p" in ops:
            p = _qdq(p, -1, per_row=True)  # per-query-token global (flash homogeneity)
        if "v" in ops:
            vh = _qdq(vh, 0)
        ref[:, h] = (p.half().float() @ vh.half().float()).to(ref.dtype)
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
