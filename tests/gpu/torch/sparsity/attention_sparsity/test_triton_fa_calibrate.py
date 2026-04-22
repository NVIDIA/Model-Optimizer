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

"""GPU tests for the Triton flash attention calibration kernel.

Exercises ``attention_calibrate`` which computes full attention while counting
how many KV tiles would be skipped at each threshold in ``threshold_trials``.
"""

import pytest
import torch
from conftest import make_qkv, make_varlen_meta

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.kernels import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels import attention, attention_calibrate


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestAttentionCalibrate:
    """Multi-threshold sparsity measurement kernel."""

    def _make_inputs(self, batch=1, seq_len=256, num_heads=4, head_dim=64):
        total = batch * seq_len
        torch.manual_seed(42)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len] * batch)
        return q, k, v, locs, lens

    def test_output_matches_dense(self):
        """Calibration kernel computes full attention — output should match dense."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 256, softmax_scale=scale, is_causal=False)
        out_calib, counters = attention_calibrate(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            is_causal=False,
            threshold_trials=[1e-3, 1e-2, 1e-1],
        )
        assert out_calib.shape == q.shape
        # Online softmax differences between dense and calibrate kernel are within a small tol
        torch.testing.assert_close(out_calib, out_dense, rtol=5e-3, atol=5e-3)

    def test_counter_shape_and_values(self):
        """Counters have shape [num_thresholds, 2] and sane values."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        trials = [1e-4, 1e-2, 1e-1, 5e-1]
        _, counters = attention_calibrate(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            is_causal=False,
            threshold_trials=trials,
        )
        assert counters.shape == (len(trials), 2)
        totals = counters[:, 0]
        skipped = counters[:, 1]
        # Totals are equal across thresholds (every tile evaluated for every threshold)
        assert (totals == totals[0]).all()
        # Skipped counts monotonically increase with threshold
        skipped_list = skipped.tolist()
        assert all(skipped_list[i] <= skipped_list[i + 1] for i in range(len(skipped_list) - 1))
        # No tile can be skipped more than total
        assert (skipped <= totals).all()

    def test_different_seq_q_seq_k(self):
        """Cross-attention varlen with separate Q and K/V metadata."""
        batch = 1
        seq_q, seq_k = 128, 256
        num_heads, head_dim = 4, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(11)
        q = torch.randn(seq_q * batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(seq_k * batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(seq_k * batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
        b_start_loc = torch.arange(batch, device="cuda", dtype=torch.int32) * seq_q
        b_seq_len = torch.full((batch,), seq_q, device="cuda", dtype=torch.int32)
        b_start_loc_k = torch.arange(batch, device="cuda", dtype=torch.int32) * seq_k
        b_seq_len_k = torch.full((batch,), seq_k, device="cuda", dtype=torch.int32)

        out, counters = attention_calibrate(
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            seq_q,
            softmax_scale=scale,
            is_causal=False,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=seq_k,
            threshold_trials=[1e-2, 1e-1],
        )
        assert out.shape == q.shape
        assert counters.shape == (2, 2)

    def test_threshold_order_doesnt_affect_counts(self):
        """Skipped counts at the same threshold are independent of trial ordering."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        _, c1 = attention_calibrate(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            is_causal=False,
            threshold_trials=[1e-3, 1e-1],
        )
        _, c2 = attention_calibrate(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            is_causal=False,
            threshold_trials=[1e-1, 1e-3],
        )
        # Both runs measure the same two thresholds — the skipped counts should match
        # after permuting back to the same order.
        assert c1[0, 1].item() == c2[1, 1].item()
        assert c1[1, 1].item() == c2[0, 1].item()


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestMeasureSparsity:
    """Runtime sparsity counters during inference."""

    def test_measure_sparsity_returns_counts(self):
        """measure_sparsity=True attaches _sparsity_total/_sparsity_skipped to output."""
        torch.manual_seed(99)
        batch, seq_len, num_heads, head_dim = 1, 1024, 4, 64
        total = batch * seq_len
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len] * batch)
        scale = 1.0 / (head_dim**0.5)

        out = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            is_causal=False,
            skip_softmax_threshold=0.5,
            measure_sparsity=True,
        )
        assert hasattr(out, "_sparsity_total")
        assert hasattr(out, "_sparsity_skipped")
        assert out._sparsity_total > 0
        assert out._sparsity_skipped <= out._sparsity_total

    def test_measure_sparsity_without_skip_is_noop(self):
        """Without skip-softmax, measure_sparsity doesn't attach counters."""
        q, k, v = make_qkv(256, 4, 4, 64, dtype=torch.float16)
        locs, lens = make_varlen_meta([256])
        scale = 1.0 / (64**0.5)

        out = attention(
            q, k, v, locs, lens, 256, softmax_scale=scale, is_causal=False, measure_sparsity=True
        )
        # No skip-softmax active => counters should not be attached
        assert not hasattr(out, "_sparsity_total")

    def test_raw_threshold_path(self):
        """Raw threshold is passed directly to the kernel without conversion."""
        q, k, v = make_qkv(256, 4, 4, 64, dtype=torch.float16)
        locs, lens = make_varlen_meta([256])
        scale = 1.0 / (64**0.5)
        out_raw = attention(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            is_causal=False,
            skip_softmax_raw_threshold=-20.0,
        )
        # With a very negative raw threshold, almost no tiles are skipped
        # Output should be close to dense
        out_dense = attention(q, k, v, locs, lens, 256, softmax_scale=scale, is_causal=False)
        torch.testing.assert_close(out_raw, out_dense, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestBackwardWithSparsity:
    """Backward pass with skip-softmax (covers _attn_bwd_dq / _attn_bwd_dkdv)."""

    def test_backward_with_skip_softmax(self):
        """Backward pass runs without error when skip-softmax is active."""
        seq_len, num_heads, head_dim = 128, 4, 64
        scale = 1.0 / (head_dim**0.5)
        torch.manual_seed(7)
        q, k, v = make_qkv(seq_len, num_heads, num_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = make_varlen_meta([seq_len])

        out = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            is_causal=True,
            skip_softmax_threshold=1e-3,
        )
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_backward_with_sparsity_nm(self):
        """Backward pass with 2:4 N:M sparsity runs without error."""
        seq_len, num_heads, head_dim = 128, 4, 64
        scale = 1.0 / (head_dim**0.5)
        torch.manual_seed(13)
        q, k, v = make_qkv(seq_len, num_heads, num_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = make_varlen_meta([seq_len])

        out = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            is_causal=True,
            sparsity_n=2,
            sparsity_m=4,
        )
        out.sum().backward()
        assert q.grad is not None
