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

"""CPU unit tests for dflash_losses.py, the DFlash family's shared loss components.

Each test here calls a helper directly; the variants' use of them is covered by the
model tests in test_hf_dflash.py, test_hf_dspark.py and the other plugin test files.
"""

import pytest
import torch

from modelopt.torch.speculative.plugins.dflash_losses import _dpace_position_weights, _tvd_per_token


class TestDPaceWeights:
    """The D-PACE position-weighting objective (arXiv:2605.18810)."""

    @staticmethod
    def _reference_weights(conf, alpha):
        """Paper closed form computed by explicit summation (Eq.7-8).

        q~_i = (1-a)q_i + a; C_m = prod_{i<=m} q~_i; w_j = sum_{m>=j} C_m.
        Deliberately a plain double loop so it is an independent oracle for the
        vectorized implementation under test.
        """
        smoothed = alpha + (1.0 - alpha) * conf
        length = smoothed.shape[-1]
        cum = torch.ones_like(smoothed)
        running = torch.ones(smoothed.shape[:-1])
        for m in range(length):
            running = running * smoothed[..., m]
            cum[..., m] = running
        expected = torch.zeros_like(smoothed)
        for j in range(length):
            expected[..., j] = cum[..., j:].sum(dim=-1)
        return expected

    def test_weights_match_paper_formula(self):
        """Eq.7-8, pinned both to a hand-worked value and an independent loop oracle.

        conf=[0.8, 0.5], alpha=0.5 -> q~=[0.9, 0.75] -> prefix=[0.9, 0.675]
        -> w=[0.9+0.675, 0.675]=[1.575, 0.675].
        """
        hand = _dpace_position_weights(torch.tensor([[0.8, 0.5]]), alpha=0.5)
        assert torch.allclose(hand, torch.tensor([[1.575, 0.675]]), atol=1e-6)
        conf = torch.tensor([[0.9, 0.6, 0.3, 0.8]])
        assert torch.allclose(
            _dpace_position_weights(conf, 0.5), self._reference_weights(conf, 0.5), atol=1e-6
        )

    def test_mask_makes_invalid_positions_noops(self):
        """Invalid positions neither shrink the prefix product nor add to the sum."""
        alpha = 0.5
        conf = torch.tensor([[0.9, 0.2, 0.3, 0.8]])
        mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        masked = _dpace_position_weights(conf, alpha, valid_mask=mask)
        # Dropping the invalid slot entirely must give the same weights at the kept slots.
        kept = _dpace_position_weights(conf[:, [0, 2, 3]], alpha)
        assert torch.allclose(masked[:, [0, 2, 3]], kept, atol=1e-6)

    def test_weights_are_detached(self):
        """Weights must carry no gradient (paper Eq.9 detaches them)."""
        conf = torch.rand(2, 3, 5, requires_grad=True)
        weights = _dpace_position_weights(conf, 0.5)
        assert not weights.requires_grad

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="dflash_dpace_alpha"):
            _dpace_position_weights(torch.rand(1, 4), alpha=1.5)


class TestTvdPerTokenChunking:
    """``_tvd_per_token`` must be chunk-size-invariant, and must chunk via ``split``.

    Calls the helper directly: the model fixtures never reach a second chunk.
    """

    @staticmethod
    def _run(chunk_size, n=12, vocab=32):
        """Forward + backward through ``_tvd_per_token`` from a fixed seed."""
        torch.manual_seed(0)
        final = torch.randn(n, vocab, requires_grad=True)
        teacher = torch.randn(n, vocab)
        out = _tvd_per_token(final, teacher, chunk_size=chunk_size)
        # Weight the rows unequally, so a cat that reassembles the chunks in the wrong
        # order cannot cancel out in the reduction.
        (out * torch.arange(1, n + 1, dtype=out.dtype)).sum().backward()
        return out.detach(), final.grad.detach()

    # 12 rows: 5 and 7 leave a ragged last chunk (5+5+2, 7+5); 13 and 1024 exceed n.
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 12, 13, 1024])
    def test_chunk_size_invariant(self, chunk_size):
        ref_out, ref_grad = self._run(12)  # one chunk == the unchunked reference
        out, grad = self._run(chunk_size)
        assert torch.equal(out, ref_out), f"TVD value changed at chunk_size={chunk_size}"
        assert torch.equal(grad, ref_grad), f"TVD grad changed at chunk_size={chunk_size}"

    def test_chunks_via_split_not_slice(self):
        """Pin the split itself: slicing gives identical values, only the graph differs."""
        final = torch.randn(8, 4, requires_grad=True)
        teacher = torch.randn(8, 4)
        out = _tvd_per_token(final, teacher, chunk_size=2)

        # `alive` keeps visited nodes referenced: `.next_functions` returns fresh wrappers,
        # and a freed wrapper's id() can be reused, which would truncate the walk.
        seen, visited, alive, stack = set(), set(), [], [out.grad_fn]
        while stack:
            fn = stack.pop()
            if fn is None or id(fn) in visited:
                continue
            visited.add(id(fn))
            alive.append(fn)
            seen.add(type(fn).__name__)
            stack.extend(nxt for nxt, _ in fn.next_functions)

        assert any(name.startswith("SplitBackward") for name in seen), (
            f"expected a SplitBackward node in the graph, saw: {sorted(seen)}"
        )
        assert not any(name.startswith("SliceBackward") for name in seen), (
            f"chunking regressed to per-chunk slicing, saw: {sorted(seen)}"
        )

    def test_no_grad_path_matches_grad_path(self):
        """The ``requires_grad=False`` branch skips checkpointing; values must not move."""
        torch.manual_seed(0)
        final = torch.randn(12, 32)
        teacher = torch.randn(12, 32)
        with torch.no_grad():
            plain = _tvd_per_token(final, teacher, chunk_size=5)
        grad_out = _tvd_per_token(final.requires_grad_(True), teacher, chunk_size=5)
        assert torch.equal(plain, grad_out.detach())
