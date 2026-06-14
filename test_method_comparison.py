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

"""Compare NVFP4 KV-cache global-scale strategies against the no-fakequant bf16 baseline.

All run the SAME decode kernel over the SAME bf16 cache, fakequanting {k,v} (Q/P left
full-precision to isolate the KV$ scale effect); they differ only in the per-tensor NVFP4 global
scale for K/V:

  1. existing (1st-chunk frozen) : scale = amax(FIRST prefill chunk)/(6*448), frozen — this is what
       the on-write plugin actually does (``_running_max`` seeds on the first launch and never
       updates), so a later token/page bigger than the first chunk SATURATES the E4M3 block scales.
  2. constant 1.0                : global scale fixed at 1.0 (no calibration) — the lower bound.
  3. per-page                    : each 128-key page uses its own amax/(6*448) — method 2.
  (ref) ideal frozen             : amax(WHOLE cache)/(6*448) — the best single scale (never
       saturates); shown to separate "per-page vs a good frozen scale" from "vs the deployed one".

Baseline = same kernel, nvfp4 disabled (no fakequant) on the bf16 cache. We report mean/max abs
output error vs that baseline (smaller = more faithful). Regimes vary magnitude ACROSS pages: when
the first chunk is representative (uniform) all are close; when later pages exceed it (growing /
outlier) the 1st-chunk-frozen scale saturates and per-page pulls ahead.
"""

import torch

from modelopt.torch.kernels.common.attention.decode_attention import attention_decode

DEV, HEAD_DIM, NQ, NKV, PAGE = "cuda", 128, 64, 8, 16
TILE = 128  # per-page scale granularity
FIRST_CHUNK = 2 * TILE  # the prefill chunk that seeds the deployed frozen scale (256 tokens)
SCALE = HEAD_DIM**-0.5
KV_NVFP4 = {"k", "v"}
torch.manual_seed(0)


def _setup(B, KV, dtype=torch.bfloat16):
    nblk = (KV + PAGE - 1) // PAGE
    tb = B * nblk
    q = torch.randn(B, NQ, HEAD_DIM, device=DEV, dtype=dtype)
    kc = torch.randn(tb, PAGE, NKV, HEAD_DIM, device=DEV, dtype=dtype)
    vc = torch.randn(tb, PAGE, NKV, HEAD_DIM, device=DEV, dtype=dtype)
    bt = torch.arange(tb, device=DEV, dtype=torch.int32).view(B, nblk)
    sk = torch.full((B,), KV, device=DEV, dtype=torch.int32)
    return q, kc, vc, bt, sk


def _flat(t, B):
    return t.view(B, -1, NKV, HEAD_DIM)  # [B, KV_padded, NKV, HEAD_DIM] (contiguous arange bt)


def _gscale(t):
    return t.float().abs().amax() / (6 * 448) + 1e-30  # 0-d device tensor


def _out(q, kc, vc, bt, sk, *, nvfp4=None, gs=None, per_page=False):
    return attention_decode(
        q,
        kc.clone(),
        vc.clone(),
        bt,
        sk,
        softmax_scale=SCALE,
        page_size=PAGE,
        nvfp4=(nvfp4 or set()),
        attn_global_scales=gs,
        per_page_scale=per_page,
    ).float()


def _err(out, base):
    d = (out - base).abs()
    return d.mean().item(), d.max().item()


def _compare(name, q, kc, vc, bt, sk, B):
    base = _out(q, kc, vc, bt, sk)  # no fakequant, bf16 -> reference
    fk, fv = _flat(kc, B), _flat(vc, B)
    one = q.new_full((), 1.0, dtype=torch.float32)
    fc = {"k": _gscale(fk[:, :FIRST_CHUNK]), "v": _gscale(fv[:, :FIRST_CHUNK])}  # 1st-chunk frozen
    ideal = {"k": _gscale(kc), "v": _gscale(vc)}  # whole-cache frozen (never saturates)
    rows = [
        ("1 existing (1st-chunk frozen)", {"nvfp4": KV_NVFP4, "gs": fc}),
        ("2 constant 1.0", {"nvfp4": KV_NVFP4, "gs": {"k": one, "v": one}}),
        ("3 per-page", {"nvfp4": KV_NVFP4, "per_page": True}),
        ("(ref) ideal frozen (whole-amax)", {"nvfp4": KV_NVFP4, "gs": ideal}),
    ]
    print(f"\n=== {name} | ref(no-FQ) mean|out|={base.abs().mean().item():.4e} ===")
    print(f"  {'method':33s} {'mean|err|':>11s} {'max|err|':>11s} {'rel-mean':>9s}")
    results = {}
    for label, kw in rows:
        mean, mx = _err(_out(q, kc, vc, bt, sk, **kw), base)  # type: ignore[arg-type]
        results[label] = mean
        print(f"  {label:33s} {mean:11.4e} {mx:11.4e} {mean / base.abs().mean().item():8.2%}")
    return results


if __name__ == "__main__":
    print(
        f"GPU {torch.cuda.get_device_name(0)} | head_dim={HEAD_DIM} q={NQ} kv={NKV} page={PAGE} (bf16)"
    )
    print(f"first prefill chunk (seeds the deployed frozen scale) = {FIRST_CHUNK} tokens")
    B, KV = 2, 1024  # 8 pages of 128
    npages = KV // TILE
    summary = {}

    # A: uniform N(0,1) — first chunk representative of the whole sequence. Expect all close.
    q, kc, vc, bt, sk = _setup(B, KV)
    summary["A uniform"] = _compare("A: uniform N(0,1)", q, kc, vc, bt, sk, B)

    # B: magnitude GROWS with page position (page p scaled x2^p) — the first chunk (pages 0-1) is the
    # smallest, so the deployed 1st-chunk scale is far too small for later pages -> saturation.
    q, kc, vc, bt, sk = _setup(B, KV)
    fk, fv = _flat(kc, B), _flat(vc, B)
    for p in range(npages):
        fk[:, p * TILE : (p + 1) * TILE] *= 2.0**p
        fv[:, p * TILE : (p + 1) * TILE] *= 2.0**p
    summary["B growing"] = _compare(
        "B: magnitude grows with position (x2^page)", q, kc, vc, bt, sk, B
    )

    # C: a single large outlier page (x32) AFTER the first chunk — classic later-context outlier.
    q, kc, vc, bt, sk = _setup(B, KV)
    fk, fv = _flat(kc, B), _flat(vc, B)
    fk[:, 6 * TILE : 7 * TILE] *= 32.0
    fv[:, 6 * TILE : 7 * TILE] *= 32.0
    summary["C late-outlier"] = _compare("C: late outlier page (x32, page 6)", q, kc, vc, bt, sk, B)

    print("\n=== summary: mean|err| vs bf16 baseline (lower = better) ===")
    print(
        f"  {'regime':14s} {'1 existing':>12s} {'2 const1.0':>12s} {'3 per-page':>12s} {'ideal-frozen':>13s}"
    )
    for name, r in summary.items():
        v = list(r.values())
        print(f"  {name:14s} {v[0]:12.3e} {v[1]:12.3e} {v[2]:12.3e} {v[3]:13.3e}")
    print(
        "\nReading: per-page (3) tracks the IDEAL frozen scale (it never saturates either), and "
        "beats the DEPLOYED 1st-chunk-frozen (1) + constant-1.0 (2) exactly when later pages exceed "
        "the first chunk (B/C). Uniform (A) -> all comparable."
    )
