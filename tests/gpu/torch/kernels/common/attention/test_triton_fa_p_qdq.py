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

"""GPU tests for P-QDQ and tile-sensitive behavior of the Triton FA kernel."""

import math
from unittest.mock import Mock

import pytest
import torch
from conftest import make_qkv, make_varlen_meta, sdpa_reference

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor, e2m1_values
from modelopt.torch.quantization.tensor_quant import fp8_eager

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention import attention, triton_fa
    from modelopt.torch.kernels.common.attention.triton_fa import LOG2E

NATIVE_E4M3_AVAILABLE = TRITON_KERNEL_AVAILABLE and torch.cuda.get_device_capability() >= (8, 9)
requires_native_e4m3 = pytest.mark.skipif(
    not NATIVE_E4M3_AVAILABLE, reason="Native E4M3 requires compute capability >= 8.9"
)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need triton")
def test_v_qdq_rejects_autograd_before_kernel_launch(monkeypatch):
    apply = Mock(side_effect=AssertionError("_Attention.apply reached"))
    monkeypatch.setattr(triton_fa._Attention, "apply", apply)
    q, k, v = (torch.zeros(1, 1, 16, requires_grad=True) for _ in range(3))
    locs = torch.zeros(1, dtype=torch.int32)
    lens = torch.ones(1, dtype=torch.int32)
    with pytest.raises(NotImplementedError, match=r"v_qdq.*autograd"):
        attention(q, k, v, locs, lens, 1, v_qdq="nvfp4")
    apply.assert_not_called()


# P-QDQ is tile-local, so this is the normal autotuned forward path's numerical
# contract rather than an implementation detail inferred from whichever
# configuration autotune selects. Direct measurement uses separate geometry.
P_QDQ_BLOCK_N = 32
FP8_E4M3_MAX = 448.0


class _FixedBlockMForward:
    """Launch the raw forward kernel with one selected query tile."""

    def __init__(self, autotuner, block_m):
        self.fn = autotuner.fn
        self._block_m = block_m

    def __getitem__(self, grid):
        resolved_grid = grid({"BLOCK_M": self._block_m}) if callable(grid) else grid

        def launch(*args, **kwargs):
            return self.fn[resolved_grid](
                *args,
                **kwargs,
                BLOCK_M=self._block_m,
                BLOCK_N=P_QDQ_BLOCK_N,
                num_stages=2,
                num_warps=4,
            )

        return launch


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need triton")
def test_sparse_dense_window_is_block_m_invariant(monkeypatch):
    """The dense recent-token policy must not depend on the compute query tile."""
    seq_len_q, seq_len_kv = 197, 233
    num_heads, head_dim = 2, 64
    torch.manual_seed(20260706)
    q = torch.randn(seq_len_q, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(seq_len_kv, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    q_locs, q_lens = make_varlen_meta([seq_len_q])
    kv_locs, kv_lens = make_varlen_meta([seq_len_kv])

    autotuner = triton_fa._attn_fwd
    outputs = {}
    for block_m in (16, 64, 128):
        monkeypatch.setattr(triton_fa, "_attn_fwd", _FixedBlockMForward(autotuner, block_m))
        outputs[block_m] = attention(
            q,
            k,
            v,
            q_locs,
            q_lens,
            seq_len_q,
            is_causal=True,
            b_start_loc_k=kv_locs,
            b_seq_len_k=kv_lens,
            max_input_len_k=seq_len_kv,
            sparsity_n=2,
            sparsity_m=4,
            dense_recent_tokens=64,
        )

    for block_m in (64, 128):
        torch.testing.assert_close(outputs[16], outputs[block_m], rtol=2e-3, atol=2e-4)


def _sparse_attention_reference(q, k, v, scale, dense_recent_tokens):
    """Differentiable 2:4 attention with token-exact dense recent positions."""
    seq_len_q, seq_len_kv = q.shape[0], k.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    q_abs = torch.arange(seq_len_q, device=q.device) + seq_len_kv - seq_len_q
    kv_abs = torch.arange(seq_len_kv, device=q.device)
    causal = q_abs[:, None] >= kv_abs[None, :]
    scores = scores.masked_fill(~causal[None, :, :], float("-inf"))

    grouped = scores.reshape(scores.shape[0], seq_len_q, seq_len_kv // 4, 4)
    kept = torch.zeros_like(grouped, dtype=torch.bool)
    kept.scatter_(-1, grouped.topk(2, dim=-1).indices, True)
    sparse_scores = grouped.masked_fill(~kept, float("-inf")).reshape_as(scores)

    distance = q_abs[:, None] - kv_abs[None, :]
    dense = (distance >= 0) & (distance < dense_recent_tokens)
    scores = torch.where(dense[None, :, :], scores, sparse_scores)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need triton")
def test_sparse_dense_window_forward_and_backward_match_token_reference():
    """Forward and both backward phases use the same tile-independent dense policy."""
    seq_len, num_heads, head_dim = 96, 1, 32
    scale = 1.0 / math.sqrt(head_dim)
    dense_recent_tokens = 17
    torch.manual_seed(31415)
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    weights = torch.randn_like(q)
    locs, lens = make_varlen_meta([seq_len])

    q_actual, k_actual, v_actual = (tensor.clone().requires_grad_(True) for tensor in (q, k, v))
    actual = attention(
        q_actual,
        k_actual,
        v_actual,
        locs,
        lens,
        seq_len,
        softmax_scale=scale,
        sparsity_n=2,
        sparsity_m=4,
        dense_recent_tokens=dense_recent_tokens,
    )
    (actual * weights).sum().backward()

    q_ref, k_ref, v_ref = (tensor.clone().requires_grad_(True) for tensor in (q, k, v))
    reference = _sparse_attention_reference(q_ref, k_ref, v_ref, scale, dense_recent_tokens)
    (reference * weights).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
    for actual_grad, reference_grad in (
        (q_actual.grad, q_ref.grad),
        (k_actual.grad, k_ref.grad),
        (v_actual.grad, v_ref.grad),
    ):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-2, atol=2e-2)


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
    scale_in_fp8_range = (block_amax / (6.0 * global_scale)).clamp(max=FP8_E4M3_MAX)
    scale = scale_in_fp8_range.to(torch.float8_e4m3fn).float() * global_scale
    scale_safe = torch.where(scale == 0, 1.0, scale)
    q = _fp4_round(g / scale_safe) * scale_safe
    q = torch.where(scale == 0, 0.0, q)
    return q.reshape(shape)


def _apply_qdq(p, mode, qdq_scale=1.0):
    if mode == "fp8":
        return _qdq_fp8(p, qdq_scale)
    assert mode == "nvfp4"
    return _qdq_nvfp4(p, qdq_scale)


def qdq_attention_reference(q, k, v, scale, mode, is_causal=True, amax=1.0, block_n=P_QDQ_BLOCK_N):
    """Tile-looped online-softmax reference replicating kernel P_QDQ semantics.

    Single sequence: q [s, h, d], k/v [s_kv, h_kv, d] (fp16). Walks KV tiles
    of ``block_n`` exactly like the kernel, keeps the softmax denominator
    unquantized, applies qdq to the unnormalized p of each tile, and mirrors
    the kernel's operand carrier before the P @ V dot. Native NVFP4 rounds P
    to the model dtype before packing, then keeps the QDQ result in FP32. FP8
    retains the legacy post-QDQ cast to V's dtype.
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
    for start in range(0, s_kv, block_n):
        tile = t[:, :, start : start + block_n]
        m_new = torch.maximum(row_max, tile.amax(dim=-1))
        p = torch.exp2(tile - m_new[..., None])
        l_new = p.sum(dim=-1)
        corr = torch.exp2(row_max - m_new)
        row_sum = row_sum * corr + l_new
        acc = acc * corr[..., None]
        if mode == "nvfp4":
            p = p.to(v.dtype).float()
        p = _apply_qdq(p, mode, qdq_scale)
        if mode == "fp8":
            p = p.to(v.dtype).float()
        acc = acc + torch.einsum("hqk,khd->hqd", p, vv[start : start + block_n].float())
        row_max = m_new
    out = acc / row_sum[..., None]
    return out.permute(1, 0, 2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSoftmaxQdqForward:
    """Forward correctness of FP8/NVFP4 softmax quant-dequant."""

    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    @requires_native_e4m3
    def test_prefill_matches_tile_reference(self, mode):
        """Kernel qdq output matches the tile-looped torch reference."""
        seq_len, num_heads, num_kv_heads, head_dim = 256, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(7)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len])

        o = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale, p_qdq=mode)
        ref = qdq_attention_reference(q, k, v, scale, mode)
        atol = 2e-2 if mode == "nvfp4" else 5e-3
        # Isolated Triton-vs-PyTorch exp2 differences can cross an NVFP4 quantization
        # boundary, so use the same NVFP4 tolerance as the partial-tile coverage.
        torch.testing.assert_close(o.float(), ref, rtol=5e-3, atol=atol)

        # The feature must actually change the output vs dense attention.
        o_dense = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale)
        assert not torch.equal(o, o_dense)

    @requires_native_e4m3
    def test_nvfp4_uses_native_p_input_and_fp32_carriers(self):
        """Dynamic-Q and P QDQ stay FP32 until their native MMA boundaries."""
        num_heads = num_kv_heads = 1
        head_dim = 16
        seq_len_kv = P_QDQ_BLOCK_N
        scale = 1.0 / math.sqrt(head_dim)

        q = torch.zeros(1, num_heads, head_dim, device="cuda", dtype=torch.float32)
        boundary_p = 0.1248
        q[..., 0] = -math.log2(boundary_p) / (scale * LOG2E)
        k = torch.zeros(seq_len_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        # The first block contains the tile max and P at the 1/24 decision
        # boundary. Rounding P to BF16 before packing moves it to the adjacent
        # E2M1 value; rounding the QDQ result afterward does not model this.
        k[1:, 0, 0] = -1.0
        v = torch.zeros_like(k)
        v[1:16, 0, 0] = 1.0
        q_locs = torch.zeros(1, device="cuda", dtype=torch.int32)
        q_lens = torch.ones(1, device="cuda", dtype=torch.int32)
        kv_locs = torch.zeros(1, device="cuda", dtype=torch.int32)
        kv_lens = torch.full((1,), seq_len_kv, device="cuda", dtype=torch.int32)

        out = attention(
            q,
            k,
            v,
            q_locs,
            q_lens,
            1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=kv_locs,
            b_seq_len_k=kv_lens,
            max_input_len_k=seq_len_kv,
            p_qdq="nvfp4",
        )
        reference = qdq_attention_reference(q, k, v, scale, "nvfp4", is_causal=False)

        # The Triton exp2 approximation shifts the unquantized denominator
        # slightly at this decision-boundary case; the old FP32 pack input is
        # still separated by more than two orders of magnitude.
        torch.testing.assert_close(out, reference, rtol=2e-3, atol=5e-4)

    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    @requires_native_e4m3
    def test_varlen_partial_tiles(self, mode):
        """Variable-length batch with partial KV tiles (seq % P_QDQ_BLOCK_N != 0)."""
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
            # BF16/FP16 pre-pack rounding makes NVFP4 code selection more
            # sensitive to Triton-vs-PyTorch exp2 differences near thresholds.
            atol = 2e-2 if mode == "nvfp4" else 5e-3
            torch.testing.assert_close(o[s : s + n].float(), ref, rtol=5e-3, atol=atol)

    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    @requires_native_e4m3
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
    @requires_native_e4m3
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

    @requires_native_e4m3
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

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"v_qdq": "int8"}, "v_qdq"),
            ({"v_qdq": "fp8"}, "v_qdq"),
            ({"v_qdq": "nvfp4", "v_qdq_amax": 0.0}, "v_qdq_amax"),
            ({"v_qdq": "nvfp4", "v_cache_quantized": True}, "v_cache_quantized"),
        ],
    )
    def test_invalid_v_qdq_configuration(self, kwargs, match):
        q, k, v = make_qkv(8, 2, 2, 32, dtype=torch.float16)
        locs, lens = make_varlen_meta([8])
        with pytest.raises(ValueError, match=match):
            attention(q, k, v, locs, lens, 8, **kwargs)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
@requires_native_e4m3
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
