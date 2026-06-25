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

import os
import subprocess
import sys

import pytest
import torch
from conftest import make_qkv, make_varlen_meta, scatter_to_paged_cache

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention import attention
    from modelopt.torch.kernels.sparsity.attention.calibrate import attention_calibrate


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

    def test_decode_skips_padding_rows(self):
        """Decode (seq_q=1) skips real KV tiles once padding Q rows are excluded.

        With BLOCK_M=128, 127/128 query rows are padding. Before the padding-row
        fix their ~0 gap forced zero skips; after it the largest threshold skips a
        meaningful number of KV tiles.
        """
        seq_q, seq_k, num_heads, head_dim = 1, 512, 4, 64
        scale = 1.0 / (head_dim**0.5)
        torch.manual_seed(0)
        q = torch.randn(seq_q, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(seq_k, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(seq_k, num_heads, head_dim, device="cuda", dtype=torch.float16)
        b_start_loc = torch.zeros(1, device="cuda", dtype=torch.int32)
        b_seq_len = torch.ones(1, device="cuda", dtype=torch.int32)
        b_start_loc_k = torch.zeros(1, device="cuda", dtype=torch.int32)
        b_seq_len_k = torch.full((1,), seq_k, device="cuda", dtype=torch.int32)

        _, counters = attention_calibrate(
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
            threshold_trials=[1e-2, 1e-1, 5e-1, 9e-1],
        )
        skipped = counters[:, 1]
        assert (skipped[1:] >= skipped[:-1]).all()  # monotonic non-decreasing
        assert (skipped <= counters[:, 0]).all()
        assert skipped[-1] > 0  # padding-row fix makes this non-zero

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

    def test_threshold_semantics_match_runtime_counts(self):
        """Calibration threshold trials use the same lambda semantics as runtime."""
        batch, seq_len, num_heads, head_dim = 1, 256, 1, 64
        total = batch * seq_len
        scale = 1.0 / (head_dim**0.5)
        qk_scale = scale * 1.44269504088896
        threshold = 0.1

        q = torch.zeros(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        q[:, :, 0] = 1.0
        k[128:, :, 0] = -1.0 / qk_scale
        v[128:] = 1.0
        locs = torch.zeros(batch, device="cuda", dtype=torch.int32)
        lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

        out = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            is_causal=False,
            skip_softmax_threshold=threshold,
            measure_sparsity=True,
        )
        _, counters = attention_calibrate(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            is_causal=False,
            threshold_trials=[threshold],
        )

        assert counters[0, 0].item() == out._sparsity_total
        assert counters[0, 1].item() == out._sparsity_skipped


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestAttentionCalibratePaged:
    """Paged KV cache calibration must match the contiguous reference exactly.

    This is the path the vLLM integration calibrates through: KV lives in a
    paged cache addressed by a block table rather than in contiguous tensors.
    """

    def test_prefill_paged_matches_contiguous(self):
        """Causal prefill: paged counters and output equal the contiguous run."""
        seq, num_heads, num_kv_heads, head_dim, page_size = 384, 4, 2, 64, 16
        scale = 1.0 / (head_dim**0.5)
        trials = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 9e-1]

        torch.manual_seed(0)
        # A dominant sink at position 0 (q·k[0] huge, all other scores ~0) makes
        # later KV tiles negligible, so later query tiles skip them — gives nonzero
        # counters to compare, beyond the trivially-equal all-dense case.
        q = torch.ones(seq, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.zeros(seq, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        k[0] = 20.0
        v = torch.randn(seq, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        locs, lens = make_varlen_meta([seq])

        out_ref, c_ref = attention_calibrate(
            q, k, v, locs, lens, seq, softmax_scale=scale, is_causal=True, threshold_trials=trials
        )

        k_cache, v_cache, block_table = scatter_to_paged_cache(
            k, v, locs, lens, num_kv_heads, head_dim, page_size
        )
        k_dummy = torch.empty(0, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        out_pg, c_pg = attention_calibrate(
            q,
            k_dummy,
            k_dummy,
            locs,
            lens,
            seq,
            softmax_scale=scale,
            is_causal=True,
            b_seq_len_k=lens,
            max_input_len_k=seq,
            threshold_trials=trials,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )
        assert torch.equal(c_ref, c_pg), (c_ref.tolist(), c_pg.tolist())
        assert c_pg[-1, 1] > 0  # the sink makes some tiles skippable
        torch.testing.assert_close(out_pg, out_ref, rtol=5e-3, atol=5e-3)

    def test_decode_paged_matches_contiguous(self):
        """Decode (seq_q=1) against a long paged cache equals the contiguous run."""
        seq_k, num_heads, num_kv_heads, head_dim, page_size = 2048, 4, 2, 64, 16
        scale = 1.0 / (head_dim**0.5)
        trials = [1e-3, 1e-2, 1e-1, 5e-1, 9e-1]

        torch.manual_seed(1)
        q = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(seq_k, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(seq_k, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        locs_q = torch.zeros(1, device="cuda", dtype=torch.int32)
        len_q = torch.ones(1, device="cuda", dtype=torch.int32)
        locs_k = torch.zeros(1, device="cuda", dtype=torch.int32)
        len_k = torch.full((1,), seq_k, device="cuda", dtype=torch.int32)

        out_ref, c_ref = attention_calibrate(
            q,
            k,
            v,
            locs_q,
            len_q,
            1,
            softmax_scale=scale,
            is_causal=False,
            b_start_loc_k=locs_k,
            b_seq_len_k=len_k,
            max_input_len_k=seq_k,
            threshold_trials=trials,
        )

        k_cache, v_cache, block_table = scatter_to_paged_cache(
            k, v, locs_k, len_k, num_kv_heads, head_dim, page_size
        )
        k_dummy = torch.empty(0, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        out_pg, c_pg = attention_calibrate(
            q,
            k_dummy,
            k_dummy,
            locs_q,
            len_q,
            1,
            softmax_scale=scale,
            is_causal=False,
            b_seq_len_k=len_k,
            max_input_len_k=seq_k,
            threshold_trials=trials,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )
        assert torch.equal(c_ref, c_pg), (c_ref.tolist(), c_pg.tolist())
        # Full cache scanned: total == num_heads * ceil(seq_k / 128).
        assert int(c_pg[0, 0]) == num_heads * (seq_k // 128)
        torch.testing.assert_close(out_pg, out_ref, rtol=5e-3, atol=5e-3)

    def test_paged_requires_block_table(self):
        """Passing a cache without a block table is a hard error, not a silent run."""
        q, k, v = make_qkv(256, 4, 4, 64, dtype=torch.float16)
        locs, lens = make_varlen_meta([256])
        k_cache, v_cache, _ = scatter_to_paged_cache(k, v, locs, lens, 4, 64, 16)
        with pytest.raises(ValueError, match="block_table"):
            attention_calibrate(
                q,
                k,
                v,
                locs,
                lens,
                256,
                softmax_scale=1.0 / (64**0.5),
                is_causal=False,
                threshold_trials=[1e-2],
                k_cache=k_cache,
                v_cache=v_cache,
            )


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestCalibrateVsPytorchReference:
    """The Triton calibration kernel must measure the same sparsity as PyTorch.

    ``attention_calibrate`` (contiguous and paged) and the PyTorch
    ``flash_skip_softmax`` calibration both use 128x128 block-level skip logic
    (keep a block iff some query row's block-max stays within ``log(threshold)``
    of the running max). This is the contract that lets vLLM calibration produce
    the same ``(a, b)`` as the established PyTorch path — assert the per-threshold
    skipped-tile fractions agree on identical inputs.
    """

    _TRIALS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 9e-1]

    @staticmethod
    def _pytorch_sparsity(q4, k4, v4, trials, scale, is_causal):
        """Per-threshold skipped-block fraction from PyTorch flash_skip_softmax."""
        from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import (
            FlashSkipSoftmax,
        )

        seq_q, seq_k = q4.shape[2], k4.shape[2]
        scores = torch.matmul(q4, k4.transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_q, seq_k, device=q4.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask[None, None], float("-inf"))
        method = FlashSkipSoftmax(
            method_config={
                "thresholds": {"prefill": trials, "decode": trials},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": is_causal,
            }
        )
        method._calibration_mode = True
        method.thresholds = trials
        _, stats = method.calc_correction_factor_and_p(scores, "prefill" if seq_q > 1 else "decode")
        return stats["sparsity"]

    @staticmethod
    def _triton_sparsity(counters):
        return (counters[:, 1].float() / counters[:, 0].clamp(min=1)).tolist()

    @staticmethod
    def _graded_qkv(seq, num_heads, head_dim, seed):
        """Localized-decay attention (sink + distance decay) -> graded sparsity."""
        torch.manual_seed(seed)
        q4 = torch.randn(1, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)
        pos = torch.arange(seq, device="cuda").float()
        decay = torch.exp(-pos / (seq * 0.15))[None, None, :, None]
        k4 = (torch.randn(1, num_heads, seq, head_dim, device="cuda") * decay).to(torch.float16)
        k4[:, :, 0] = 8.0
        v4 = torch.randn(1, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)
        return q4, k4, v4

    def _triton_paged_sparsity(self, q4, k4, v4, trials, scale):
        seq, num_heads, head_dim = q4.shape[2], q4.shape[1], q4.shape[3]
        qf = q4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        kf = k4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        vf = v4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        locs, lens = make_varlen_meta([seq])
        k_cache, v_cache, block_table = scatter_to_paged_cache(
            kf, vf, locs, lens, num_heads, head_dim, 16
        )
        k_dummy = torch.empty(0, num_heads, head_dim, device="cuda", dtype=torch.float16)
        _, counters = attention_calibrate(
            qf,
            k_dummy,
            k_dummy,
            locs,
            lens,
            seq,
            softmax_scale=scale,
            is_causal=True,
            b_seq_len_k=lens,
            max_input_len_k=seq,
            threshold_trials=trials,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=16,
        )
        return self._triton_sparsity(counters)

    def test_fitted_ab_matches_pytorch(self):
        """End-to-end: the fitted exponential (a, b) is the same for both paths.

        Measures per-length sparsity over several lengths with PyTorch
        flash_skip_softmax and with the paged (vLLM) Triton kernel, fits each set
        through DynamicThresholdCalibrator, and asserts the calibration results
        (a, b) agree — the property that lets vLLM-calibrated checkpoints serve
        identically to HF-calibrated ones.
        """
        from modelopt.torch.sparsity.attention_sparsity.calibration.calibrator import (
            DynamicThresholdCalibrator,
        )

        num_heads, head_dim = 4, 64
        scale = 1.0 / (head_dim**0.5)
        trials = [1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 9e-1]

        pt_stats, triton_stats = [], []
        # Non-128-multiple lengths so the partial last block-row exercises the
        # padding-row handling (where flash previously diverged from the kernel).
        for seed, seq in enumerate([500, 776, 1000, 1500, 2000]):
            q4, k4, v4 = self._graded_qkv(seq, num_heads, head_dim, seed)
            pt_stats.append(
                {
                    "sparsity": self._pytorch_sparsity(q4, k4, v4, trials, scale, True),
                    "sample_length": seq,
                }
            )
            triton_stats.append(
                {
                    "sparsity": self._triton_paged_sparsity(q4, k4, v4, trials, scale),
                    "sample_length": seq,
                }
            )

        pt_fit = DynamicThresholdCalibrator(threshold_trials=trials).calibrate_from_stats(
            pt_stats, "prefill"
        )
        triton_fit = DynamicThresholdCalibrator(threshold_trials=trials).calibrate_from_stats(
            triton_stats, "prefill"
        )

        # Both fits must succeed and agree (same measured sparsity -> same fit).
        assert pt_fit and triton_fit, (pt_fit, triton_fit)
        assert pt_fit["a"] == pytest.approx(triton_fit["a"], rel=1e-3)
        assert pt_fit["b"] == pytest.approx(triton_fit["b"], rel=1e-3)
        # Sanity: a real (non-degenerate) exponential fit on enough valid points.
        assert pt_fit["a"] > 0 and pt_fit["b"] > 0
        assert pt_fit["num_data_points"] >= 10

    def test_graded_prefill_matches_pytorch(self):
        """Localized-decay attention sweeps sparsity 0->~0.7; all three paths agree.

        Compares PyTorch flash_skip_softmax, contiguous ``attention_calibrate``,
        and the paged (vLLM) ``attention_calibrate`` — the graded sweep makes this
        a discriminating test rather than a trivially-0% / 100% one.

        ``seq`` is deliberately *not* a multiple of the 128 block size: the last
        query-block-row is partial, exercising the padding-row handling where the
        flash block method previously diverged from the kernel (it counted
        dtype-min-padded rows as "keep" and never skipped the last block row).
        """
        num_heads, head_dim, seq, page_size = 4, 64, 1000, 16
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(2)
        q4 = torch.randn(1, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)
        # Key norm decays with position (+ a sink at 0) so distant tiles fall below
        # the threshold gradually as it grows — a smooth sparsity sweep.
        pos = torch.arange(seq, device="cuda").float()
        decay = torch.exp(-pos / (seq * 0.15))[None, None, :, None]
        k4 = (torch.randn(1, num_heads, seq, head_dim, device="cuda") * decay).to(torch.float16)
        k4[:, :, 0] = 8.0
        v4 = torch.randn(1, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)

        pt = self._pytorch_sparsity(q4, k4, v4, self._TRIALS, scale, is_causal=True)

        qf = q4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        kf = k4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        vf = v4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        locs, lens = make_varlen_meta([seq])

        _, c_contig = attention_calibrate(
            qf,
            kf,
            vf,
            locs,
            lens,
            seq,
            softmax_scale=scale,
            is_causal=True,
            threshold_trials=self._TRIALS,
        )

        k_cache, v_cache, block_table = scatter_to_paged_cache(
            kf, vf, locs, lens, num_heads, head_dim, page_size
        )
        k_dummy = torch.empty(0, num_heads, head_dim, device="cuda", dtype=torch.float16)
        _, c_paged = attention_calibrate(
            qf,
            k_dummy,
            k_dummy,
            locs,
            lens,
            seq,
            softmax_scale=scale,
            is_causal=True,
            b_seq_len_k=lens,
            max_input_len_k=seq,
            threshold_trials=self._TRIALS,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        triton_contig = self._triton_sparsity(c_contig)
        triton_paged = self._triton_sparsity(c_paged)

        # The sweep must actually exercise the intermediate (fit-relevant) range.
        assert any(0.1 < s < 0.9 for s in pt), pt
        # Paged is the vLLM path; it must equal the contiguous kernel exactly.
        assert triton_paged == triton_contig, (triton_paged, triton_contig)
        # Triton (both layouts) matches PyTorch flash_skip_softmax block-for-block.
        for s_pt, s_tr in zip(pt, triton_contig):
            assert abs(s_pt - s_tr) <= 0.02, (pt, triton_contig)

    def test_dominant_sink_matches_pytorch_exactly(self):
        """A dominant sink puts gaps far from any threshold boundary -> exact match."""
        num_heads, head_dim, seq = 4, 64, 512
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(0)
        q4 = torch.ones(1, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)
        k4 = torch.zeros(1, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)
        k4[:, :, 0] = 20.0
        v4 = torch.randn(1, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)

        pt = self._pytorch_sparsity(q4, k4, v4, self._TRIALS, scale, is_causal=True)

        qf = q4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        kf = k4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        vf = v4.permute(0, 2, 1, 3).reshape(seq, num_heads, head_dim).contiguous()
        locs, lens = make_varlen_meta([seq])
        _, counters = attention_calibrate(
            qf,
            kf,
            vf,
            locs,
            lens,
            seq,
            softmax_scale=scale,
            is_causal=True,
            threshold_trials=self._TRIALS,
        )
        triton = self._triton_sparsity(counters)
        assert max(pt) > 0.0  # the sink makes blocks skippable
        # Gaps are far from any threshold boundary, so the skipped-block counts
        # are identical; only the fraction's fp repr differs (fp64 vs fp32), so a
        # single-block disagreement (>= 1/40 = 0.025 here) would still fail.
        for s_pt, s_tr in zip(pt, triton):
            assert abs(s_pt - s_tr) < 1e-5, (pt, triton)


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

    def test_first_measured_call_has_real_tile_count_with_autotune(self):
        """Counters from the first measured call should not include autotune trials."""
        script = r"""
import torch
from modelopt.torch.kernels.common.attention import attention

batch, seq_len, num_heads, head_dim = 1, 256, 1, 64
total = batch * seq_len
scale = 1.0 / (head_dim**0.5)
q = torch.zeros(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
k = torch.zeros_like(q)
v = torch.zeros_like(q)
locs = torch.zeros(batch, device="cuda", dtype=torch.int32)
lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
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
torch.cuda.synchronize()
print(f"TOTAL={out._sparsity_total}")
"""
        env = os.environ.copy()
        env.pop("PYTEST_VERSION", None)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        totals = [line for line in result.stdout.splitlines() if line.startswith("TOTAL=")]
        assert totals, result.stdout
        # seq_len=256, _MEASURE_BLOCK_M = _MEASURE_BLOCK_N = 128, non-causal:
        # Q tiles = ceil(256/128) = 2, KV tiles = ceil(256/128) = 2, total = 4.
        assert int(totals[-1].split("=", maxsplit=1)[1]) == 4

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

    def test_tiny_threshold_path(self):
        """A tiny lambda threshold keeps output close to dense."""
        q, k, v = make_qkv(256, 4, 4, 64, dtype=torch.float16)
        locs, lens = make_varlen_meta([256])
        scale = 1.0 / (64**0.5)
        out_skip = attention(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            is_causal=False,
            skip_softmax_threshold=2**-20,
        )
        # A near-zero threshold skips very few tiles, so output stays close to dense.
        out_dense = attention(q, k, v, locs, lens, 256, softmax_scale=scale, is_causal=False)
        torch.testing.assert_close(out_skip, out_dense, rtol=1e-2, atol=1e-2)


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
