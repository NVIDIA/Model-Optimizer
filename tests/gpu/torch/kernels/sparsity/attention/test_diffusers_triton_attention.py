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

"""GPU tests for diffusers and LTX Triton attention wrappers."""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.kernels.sparsity.attention import diffusers_triton_attention as diffusers_mod
from modelopt.torch.kernels.sparsity.attention import ltx_triton_attention as ltx_mod


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestDiffusersTritonAttention:
    """Exercise _diffusers_triton_attention on a real device."""

    @pytest.fixture(autouse=True)
    def _reset_thread_local(self):
        diffusers_mod.clear_triton_skip_softmax_config()
        yield
        diffusers_mod.clear_triton_skip_softmax_config()

    def _make_qkv(self, b=1, seq_q=128, seq_k=128, h=4, d=64, dtype=torch.float16):
        q = torch.randn(b, seq_q, h, d, device="cuda", dtype=dtype)
        k = torch.randn(b, seq_k, h, d, device="cuda", dtype=dtype)
        v = torch.randn(b, seq_k, h, d, device="cuda", dtype=dtype)
        return q, k, v

    def test_basic_forward(self):
        q, k, v = self._make_qkv()
        out = diffusers_mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_skip_softmax_threshold_path(self):
        diffusers_mod.set_triton_skip_softmax_config(threshold=0.01)
        q, k, v = self._make_qkv()
        out = diffusers_mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape

    def test_small_threshold_path(self):
        diffusers_mod.set_triton_skip_softmax_config(threshold=0.0009765625)
        q, k, v = self._make_qkv()
        out = diffusers_mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape

    def test_scale_factor_path(self):
        diffusers_mod.set_triton_skip_softmax_config(scale_factor=2.0)
        q, k, v = self._make_qkv()
        out = diffusers_mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape

    def test_measure_sparsity_returns_counts(self):
        diffusers_mod.set_triton_skip_softmax_config(threshold=0.1, measure_sparsity=True)
        q, k, v = self._make_qkv(seq_q=512, seq_k=512)
        diffusers_mod._diffusers_triton_attention(q, k, v)
        total, _skipped = diffusers_mod.get_sparsity_counters()
        assert total > 0

    def test_calibration_mode(self):
        diffusers_mod.set_triton_skip_softmax_config(
            calibration_mode=True,
            threshold_trials=[1e-3, 1e-2, 1e-1],
        )
        q, k, v = self._make_qkv(seq_q=128, seq_k=128)
        out = diffusers_mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape
        counters = diffusers_mod.get_calibration_counters()
        assert counters is not None
        assert counters.shape == (3, 2)
        assert diffusers_mod.get_calibration_seq_k() == 128

    def test_cross_attention_different_seq_lengths(self):
        q, k, v = self._make_qkv(seq_q=128, seq_k=256)
        out = diffusers_mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape

    def test_sparse_nm_differs_from_dense(self):
        q, k, v = self._make_qkv(seq_q=256, seq_k=256)
        out_dense = diffusers_mod._diffusers_triton_attention(q, k, v)
        diffusers_mod.set_triton_skip_softmax_config(sparsity_n=2, sparsity_m=4)
        out_sparse = diffusers_mod._diffusers_triton_attention(q, k, v)
        assert out_sparse.shape == q.shape
        assert not torch.isnan(out_sparse).any()
        assert not torch.allclose(out_sparse, out_dense, atol=1e-3)

    def test_sparse_nm_matches_core_kernel(self):
        """The dispatch forwards N:M params to the core kernel unchanged."""
        b, seq, h, d = 2, 256, 4, 64
        q, k, v = self._make_qkv(b=b, seq_q=seq, seq_k=seq, h=h, d=d)
        diffusers_mod.set_triton_skip_softmax_config(sparsity_n=2, sparsity_m=4)
        out = diffusers_mod._diffusers_triton_attention(q, k, v)

        from modelopt.torch.kernels.common.attention import attention

        locs = torch.arange(b, device="cuda", dtype=torch.int32) * seq
        lens = torch.full((b,), seq, device="cuda", dtype=torch.int32)
        ref = attention(
            q.reshape(b * seq, h, d).contiguous(),
            k.reshape(b * seq, h, d).contiguous(),
            v.reshape(b * seq, h, d).contiguous(),
            locs,
            lens,
            seq,
            is_causal=False,
            softmax_scale=1.0 / d**0.5,
            sparsity_n=2,
            sparsity_m=4,
            dense_sink_tokens=0,
            dense_recent_tokens=0,
        ).view(b, seq, h, d)
        assert torch.equal(out, ref)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestLTXTritonAttention:
    """Exercise _ltx_triton_attention (LTX layout [B, T, H*D])."""

    @pytest.fixture(autouse=True)
    def _reset_thread_local(self):
        ltx_mod.clear_ltx_triton_context()
        yield
        ltx_mod.clear_ltx_triton_context()

    def _make_qkv(self, b=1, seq_q=128, seq_k=128, heads=4, dim_head=64, dtype=torch.float16):
        dim = heads * dim_head
        q = torch.randn(b, seq_q, dim, device="cuda", dtype=dtype)
        k = torch.randn(b, seq_k, dim, device="cuda", dtype=dtype)
        v = torch.randn(b, seq_k, dim, device="cuda", dtype=dtype)
        return q, k, v

    def test_inference_basic(self):
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=4)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_inference_with_small_threshold(self):
        ltx_mod.set_ltx_triton_context(active=True, threshold=0.0009765625)
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_inference_with_scale_factor(self):
        ltx_mod.set_ltx_triton_context(active=True, scale_factor=5.0)
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_inference_with_static_threshold(self):
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=4, threshold=0.1)
        assert out.shape == q.shape

    def test_cross_attention_different_seq(self):
        q, k, v = self._make_qkv(seq_q=128, seq_k=256)
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_sparse_nm_differs_from_dense(self):
        q, k, v = self._make_qkv(seq_q=256, seq_k=256)
        out_dense = ltx_mod._ltx_triton_attention(q, k, v, heads=4)
        ltx_mod.set_ltx_triton_context(active=True, sparsity_n=2, sparsity_m=4)
        out_sparse = ltx_mod._ltx_triton_attention(q, k, v, heads=4)
        assert out_sparse.shape == q.shape
        assert not torch.isnan(out_sparse).any()
        assert not torch.allclose(out_sparse, out_dense, atol=1e-3)

    def test_calibration_mode(self):
        ltx_mod.set_ltx_triton_context(
            active=True,
            calibration_mode=True,
            threshold_trials=[1e-3, 1e-2, 1e-1],
        )
        q, k, v = self._make_qkv(seq_q=128, seq_k=128)
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=4)
        assert out.shape == q.shape
        counters = ltx_mod.get_calibration_counters()
        assert counters is not None
        assert counters.shape == (3, 2)
        assert ltx_mod.get_calibration_seq_k() == 128

    def test_wrapper_dispatch(self):
        """TritonLTXAttentionWrapper dispatches based on thread-local flag."""
        called = {"original": 0}

        def original_fn(q, k, v, heads, mask=None):
            called["original"] += 1
            return q

        wrapper = ltx_mod._TritonLTXAttentionWrapper(original_fn)
        q, k, v = self._make_qkv()

        # When inactive, original_fn is called
        ltx_mod.clear_ltx_triton_context()
        wrapper(q, k, v, heads=4)
        assert called["original"] == 1

        # When active, triton path runs
        ltx_mod.set_ltx_triton_context(active=True, threshold=0.1)
        out = wrapper(q, k, v, heads=4)
        assert called["original"] == 1  # unchanged
        assert out.shape == q.shape
