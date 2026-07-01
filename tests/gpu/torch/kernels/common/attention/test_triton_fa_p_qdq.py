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

"""GPU tests for the softmax quant-dequant (P_QDQ) feature of the Triton FA kernel."""

# ruff: noqa: N803 — Triton JIT wrapper uses uppercase for constexpr and tensor args

import pytest
import torch
from conftest import make_qkv, make_varlen_meta, sdpa_reference

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor, e2m1_values
from modelopt.torch.quantization.tensor_quant import fp8_eager

if TRITON_KERNEL_AVAILABLE:
    import triton
    import triton.language as tl

    from modelopt.torch.kernels.common.attention import attention
    from modelopt.torch.kernels.common.attention.triton_fa import LOG2E
    from modelopt.torch.kernels.quantization.common.nvfp4_quant import nvfp4_scalar_qdq

    @triton.jit
    def _nvfp4_qdq_block_kernel(X, OUT, GS, N: tl.constexpr):
        """Run the P/V in-kernel quant primitive ``nvfp4_scalar_qdq`` on one N-block."""
        i = tl.arange(0, N)
        xv = tl.load(X + i)
        block_amax = tl.max(tl.abs(xv))
        tl.store(OUT + i, nvfp4_scalar_qdq(xv, block_amax, tl.load(GS), N))


# The kernel runs with a single pinned config under pytest (see _FWD_CONFIGS):
# BLOCK_M=128, BLOCK_N=64. The tile-looped reference below relies on it.
BLOCK_N = 64
FP8_E4M3_MAX = 448.0

# NVFP4's E4M3 block scale uses ``tl.float8e4nv``, which requires SM89+ (Ada/Hopper/Blackwell).
_HAS_SM89 = (
    TRITON_KERNEL_AVAILABLE
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9)
)


def _qdq_fp8(p, scale=1.0):
    """Per-tensor FP8 E4M3 qdq, mirroring the kernel's fp8_scalar_qdq.

    Reuses modelopt's ``fp8_eager`` (native E4M3 cast). The kernel uses the
    quantizer convention ``q = cast(p / scale) * scale``; fp8_eager parametrizes
    by amax with ``scale = amax / 448``, so the equivalent amax is ``scale * 448``.
    """
    return fp8_eager(p, torch.tensor(scale * FP8_E4M3_MAX, device=p.device))


def _fp4_round(x):
    """Round to the nearest E2M1 value, reusing modelopt's ``NVFP4QTensor._cast_fp4``.

    ``_cast_fp4`` implements the same round-half-to-even on the FP4 grid as the
    kernel's Triton ``fp4_round_magnitude`` (verified bit-for-bit on the grid
    boundaries and a dense random sweep), so this also guards that the two stay in
    sync. ``_cast_fp4`` does an in-place ``abs_`` and returns uint4 indices, so pass
    a clone and map the indices back through ``e2m1_values``.
    """
    idx = NVFP4QTensor._cast_fp4(x.clone()).long()
    return e2m1_values.to(device=x.device, dtype=x.dtype)[idx]


def _qdq_nvfp4(p, global_scale=1.0):
    """NVFP4 qdq with per-16 E4M3 block scales, mirroring _p_qdq_nvfp4.

    p: [..., n] with n % 16 == 0 and p >= 0. The per-block E4M3 scale is the
    kernel's ``fp8_quantize_scale(block_amax, global_scale)``, which equals
    ``fp8_eager(block_amax / 6, amax=448 * global_scale)`` — so the FP8 scale
    quantization is reused from modelopt rather than re-spelled here.
    """
    shape = p.shape
    g = p.reshape(*shape[:-1], shape[-1] // 16, 16)
    block_amax = g.amax(dim=-1, keepdim=True)  # p >= 0, so max == amax
    scale = fp8_eager(block_amax / 6.0, torch.tensor(FP8_E4M3_MAX * global_scale, device=p.device))
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    q = _fp4_round(g / scale) * scale
    return q.reshape(shape)


def _apply_qdq(p, mode, qdq_scale=1.0):
    if mode == "fp8":
        return _qdq_fp8(p, qdq_scale)
    assert mode == "nvfp4"
    return _qdq_nvfp4(p, qdq_scale)


def qdq_attention_reference(q, k, v, scale, mode, is_causal=True, amax=1.0):
    """Tile-looped online-softmax reference replicating kernel P_QDQ semantics.

    Single sequence: q [s, h, d], k/v [s_kv, h_kv, d] (fp16). Walks KV tiles
    of BLOCK_N exactly like the kernel, keeps the softmax denominator
    unquantized, applies qdq to the unnormalized p of each tile, and mirrors
    the kernel's ``p.to(v.dtype)`` cast before the P @ V dot.
    Returns [s, h, d] float32.

    ``amax`` mirrors the kernel's ``p_qdq_amax`` and is converted to the same
    per-mode scale the wrapper uses: ``amax / 448`` (FP8) or ``amax / (6 * 448)``
    (NVFP4 global scale).
    """
    qdq_scale = amax / 448.0 if mode == "fp8" else amax / (6.0 * 448.0)
    s, h, d = q.shape
    s_kv = k.shape[0]
    r = h // k.shape[1]
    kk = k.repeat_interleave(r, dim=1) if r > 1 else k
    vv = v.repeat_interleave(r, dim=1) if r > 1 else v

    # Scores in the kernel's scaled log2 space: Q K^T * scale * log2(e)
    t = torch.einsum("qhd,khd->hqk", q.float(), kk.float()) * (scale * LOG2E)
    if is_causal:
        offset = s_kv - s
        causal = (
            torch.arange(s, device=q.device)[:, None] + offset
            >= torch.arange(s_kv, device=q.device)[None, :]
        )
        t = t.masked_fill(~causal[None], float("-inf"))

    row_max = torch.full((h, s), float("-inf"), device=q.device)
    row_sum = torch.zeros(h, s, device=q.device)
    acc = torch.zeros(h, s, d, device=q.device)
    for start in range(0, s_kv, BLOCK_N):
        tile = t[:, :, start : start + BLOCK_N]
        m_new = torch.maximum(row_max, tile.amax(dim=-1))
        p = torch.exp2(tile - m_new[..., None])
        l_new = p.sum(dim=-1)
        corr = torch.exp2(row_max - m_new)
        row_sum = row_sum * corr + l_new
        acc = acc * corr[..., None]
        p = _apply_qdq(p, mode, qdq_scale)
        # Kernel casts p to v.dtype for the BMM2 dot
        p = p.to(v.dtype).float()
        acc = acc + torch.einsum("hqk,khd->hqd", p, vv[start : start + BLOCK_N].float())
        row_max = m_new
    out = acc / row_sum[..., None]
    return out.permute(1, 0, 2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSoftmaxQdqForward:
    """Forward correctness of FP8/NVFP4 softmax quant-dequant."""

    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    def test_prefill_matches_tile_reference(self, mode):
        """Kernel qdq output matches the tile-looped torch reference."""
        seq_len, num_heads, num_kv_heads, head_dim = 128, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(7)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len])

        o = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale, p_qdq=mode)
        ref = qdq_attention_reference(q, k, v, scale, mode)
        torch.testing.assert_close(o.float(), ref, rtol=5e-3, atol=5e-3)

        # The feature must actually change the output vs dense attention.
        o_dense = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale)
        assert not torch.equal(o, o_dense)

    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    def test_varlen_partial_tiles(self, mode):
        """Variable-length batch with partial KV tiles (seq % BLOCK_N != 0)."""
        seq_lens = [96, 80]
        total = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(11)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim, dtype=torch.float16)
        locs, lens = make_varlen_meta(seq_lens)

        o = attention(q, k, v, locs, lens, max(seq_lens), softmax_scale=scale, p_qdq=mode)
        for b, n in enumerate(seq_lens):
            s = int(locs[b].item())
            ref = qdq_attention_reference(q[s : s + n], k[s : s + n], v[s : s + n], scale, mode)
            torch.testing.assert_close(o[s : s + n].float(), ref, rtol=5e-3, atol=5e-3)

    @pytest.mark.parametrize(("mode", "tol"), [("fp8", 5e-2), ("nvfp4", 0.25)])
    def test_qdq_close_to_dense(self, mode, tol):
        """Quantization is an approximation: output stays near dense attention."""
        seq_len, num_heads, num_kv_heads, head_dim = 128, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(13)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len])

        o = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale, p_qdq=mode)
        ref = sdpa_reference(q, k, v, locs, lens)
        torch.testing.assert_close(o, ref, rtol=tol, atol=tol)

    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    def test_decode(self, mode):
        """Decode (seq_q=1 vs KV cache) matches the non-causal tile reference."""
        batch, num_heads, num_kv_heads, head_dim = 2, 4, 2, 64
        seq_lens_k = [80, 64]
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(17)
        q = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
        total_kv = sum(seq_lens_k)
        k = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        locs_k, lens_k = make_varlen_meta(seq_lens_k)
        out = attention(
            q,
            k,
            v,
            torch.arange(batch, device="cuda", dtype=torch.int32),
            torch.ones(batch, device="cuda", dtype=torch.int32),
            1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=locs_k,
            b_seq_len_k=lens_k,
            max_input_len_k=max(seq_lens_k),
            p_qdq=mode,
        )

        for b, n in enumerate(seq_lens_k):
            s = int(locs_k[b].item())
            ref = qdq_attention_reference(
                q[b : b + 1], k[s : s + n], v[s : s + n], scale, mode, is_causal=False
            )
            torch.testing.assert_close(out[b : b + 1].float(), ref, rtol=5e-3, atol=5e-3)

    @pytest.mark.parametrize(
        ("mode", "amax"),
        [
            # Non-power-of-2 amax vs the default of 1.0: a pure power-of-2 change
            # is a fixed point of FP quantization (exponent shift only) and would
            # not change the output, so use non-power-of-2 values. amax < 1
            # (0.3) also exercises FP8 saturation (P entries above amax clamp).
            ("fp8", 0.3),
            ("fp8", 3.0),
            ("nvfp4", 0.7),
        ],
    )
    def test_custom_amax_matches_tile_reference(self, mode, amax):
        """User-supplied p_qdq_amax changes the grid and matches the reference."""
        seq_len, num_heads, num_kv_heads, head_dim = 128, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(31)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len])

        o = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            p_qdq=mode,
            p_qdq_amax=amax,
        )
        ref = qdq_attention_reference(q, k, v, scale, mode, amax=amax)
        torch.testing.assert_close(o.float(), ref, rtol=5e-3, atol=5e-3)

        # The amax knob must actually change the quantization grid vs the default (amax=1).
        o_default = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale, p_qdq=mode)
        assert not torch.equal(o, o_default)

    def test_invalid_amax_raises(self):
        q, k, v = make_qkv(8, 2, 2, 32, dtype=torch.float16)
        locs, lens = make_varlen_meta([8])
        with pytest.raises(ValueError, match="p_qdq_amax"):
            attention(q, k, v, locs, lens, 8, p_qdq="fp8", p_qdq_amax=0.0)

    def test_composes_with_skip_softmax(self):
        """p_qdq composes with the skip-softmax feature in one launch."""
        seq_len, num_heads, num_kv_heads, head_dim = 256, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(19)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len])

        o = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            p_qdq="fp8",
            skip_softmax_threshold=1e-3,
        )
        ref = sdpa_reference(q, k, v, locs, lens)
        torch.testing.assert_close(o, ref, rtol=5e-2, atol=5e-2)

    def test_invalid_mode_raises(self):
        q, k, v = make_qkv(8, 2, 2, 32, dtype=torch.float16)
        locs, lens = make_varlen_meta([8])
        with pytest.raises(ValueError, match="p_qdq"):
            attention(q, k, v, locs, lens, 8, p_qdq="int8")


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSoftmaxQdqBackward:
    """Backward uses the straight-through estimator (no qdq re-applied)."""

    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    def test_ste_gradients_close_to_dense(self, mode):
        seq_len, num_heads, num_kv_heads, head_dim = 128, 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(23)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        locs, lens = make_varlen_meta([seq_len])

        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
        attention(q1, k1, v1, locs, lens, seq_len, softmax_scale=scale, p_qdq=mode).sum().backward()

        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))
        attention(q2, k2, v2, locs, lens, seq_len, softmax_scale=scale).sum().backward()

        # The backward recomputes the unquantized P (straight-through estimator),
        # so gradients match dense attention up to the quantization perturbation
        # that enters through the saved output (delta = rowsum(O * dO)). A few
        # individual elements can shift; the overall norm error stays small.
        for g_qdq, g_dense in ((q1.grad, q2.grad), (k1.grad, k2.grad), (v1.grad, v2.grad)):
            assert torch.isfinite(g_qdq).all()
            rel_err = (g_qdq - g_dense).norm() / g_dense.norm()
            assert rel_err < 5e-2, f"relative gradient error too large: {rel_err:.4f}"


@pytest.mark.skipif(not _HAS_SM89, reason="NVFP4 E4M3 block scale (tl.float8e4nv) requires SM89+")
class TestNvfp4ScaleUnderflow:
    """Boundary behavior of the NVFP4 block-scale underflow guard (P/V path).

    A block whose amax is small enough that the E4M3 block scale underflows to
    zero must not yield NaN/inf: ``nvfp4_scalar_quant``'s ``scale_safe`` maps a
    ``0``/NaN/inf scale to ``1.0``, and the tiny entries then round to 0 on the E2M1
    grid. Tiny softmax-P blocks occur naturally in long context (keys many
    log2-units below the row max), so this guard is hit in real workloads.

    With P's fixed global scale ``1/(6*448)``, ``fp8_quantize_scale`` forms the
    in-range value ``block_amax * 448``; it underflows E4M3 (rounds to 0) once
    ``block_amax`` drops below ~2.2e-6.
    """

    P_GLOBAL_SCALE = 1.0 / (6.0 * 448.0)

    def _run(self, x):
        gs = torch.tensor(self.P_GLOBAL_SCALE, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        _nvfp4_qdq_block_kernel[(1,)](x, out, gs, N=x.shape[0])
        return out

    def test_tiny_block_underflow_is_finite_and_zeroed(self):
        # amax * 448 far below the E4M3 min subnormal (2^-9) -> scale rounds to 0
        # -> scale_safe guard -> 1.0 -> tiny values (< 0.25) round to 0. No NaN/inf.
        x = torch.full((16,), 1e-7, device="cuda", dtype=torch.float32)
        out = self._run(x)
        assert torch.isfinite(out).all()
        assert torch.all(out == 0.0)

    def test_zero_block_is_finite(self):
        # An all-zero block (amax 0 -> scale 0) must stay finite via the guard.
        x = torch.zeros(16, device="cuda", dtype=torch.float32)
        out = self._run(x)
        assert torch.isfinite(out).all()
        assert torch.all(out == 0.0)

    def test_in_range_block_is_finite_and_nonzero(self):
        # A normal block (amax 0.5) keeps its scale inside E4M3 range -> no guard,
        # round-trips to a nonzero, finite result.
        x = torch.full((16,), 0.5, device="cuda", dtype=torch.float32)
        out = self._run(x)
        assert torch.isfinite(out).all()
        assert torch.any(out != 0.0)
