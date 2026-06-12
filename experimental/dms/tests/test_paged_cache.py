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

"""Tests for DMS paged cache layer."""

import pytest
import torch

from experimental.dms.tests.utils import add_dms_to_path

try:
    from dms.cache import DMSPagedCacheLayer
except ImportError:
    add_dms_to_path()
    from dms.cache import DMSPagedCacheLayer

# ---------------------------------------------------------------------------
# Test helper: ExtendedDMSPagedCacheLayer
# ---------------------------------------------------------------------------


def _recent_position_size(cache_seq_lengths, dms_window_size):
    """Return the number of tokens stored in the recent (sliding-window) region."""
    return torch.clamp(cache_seq_lengths, max=dms_window_size)


def _first_recent_position(recent_info_position, cache_seq_lengths, dms_window_size):
    """Return the ring-buffer index of the oldest token in the recent region."""
    return recent_info_position - _recent_position_size(cache_seq_lengths, dms_window_size)


class ExtendedDMSPagedCacheLayer(DMSPagedCacheLayer):
    """DMSPagedCacheLayer extended with a get_contiguous_cache method for test verification.

    Reconstructs a dense (contiguous) view of the paged KV cache so that tests can
    compare the internal block-based storage against a naive tracked reference.
    """

    def get_contiguous_cache(self, right_padded: bool = True):
        """Return a dense view of keys, values, seq lengths, and eviction info.

        Args:
            right_padded: If True (default), valid tokens are left-aligned and the
                right side is zero-padded.  If False, valid tokens are right-aligned
                (left-padded).

        Returns:
            Tuple of (keys, values, seq_lengths, eviction_info), each shaped with
            leading dims (batch_size, num_heads, ...).
        """
        assert self.key_blocks is not None
        num_blocks_per_seq = self.cache_seq_lengths.max().item() // self.block_size + 1
        blocks_to_retrieve = self.block_table[:, :num_blocks_per_seq]
        max_length = self.cache_seq_lengths.max().item()

        def _gather_and_reorder(blocks):
            """Gather blocks into a flat sequence and reorder by window position."""
            retrieved = blocks[blocks_to_retrieve].reshape(
                self.page_batch,
                num_blocks_per_seq * self.block_size,
                self.head_dim,
            )
            retrieved = retrieved[:, :max_length, :]

            permutation = torch.arange(max_length, device=blocks.device, dtype=torch.int32)[None, :]
            permutation = torch.minimum(permutation, self.cache_seq_lengths[:, None] - 1)
            permutation = permutation.broadcast_to(self.page_batch, max_length)

            page_batch_idx = torch.arange(self.page_batch, device=self.device, dtype=torch.int32)

            if not self.disable_eviction:
                recent_size = torch.clamp(self.cache_seq_lengths, max=self.dms_window_size)
                window_idx = torch.arange(
                    self.dms_window_size, device=self.device, dtype=torch.int32
                )
                adjusted_window_idx = torch.minimum(
                    window_idx[None, :],
                    torch.clamp(recent_size[:, None] - 1, min=0),
                )

                last_pos_ptr = (
                    self.recent_info_position[:, None] - 1 - adjusted_window_idx
                ) % self.dms_window_size

                window_positions = self.recent_info[page_batch_idx[:, None], last_pos_ptr, 0]

                perm_idx = self.cache_seq_lengths[:, None] - 1 - adjusted_window_idx
                assert (perm_idx >= 0).all()

                non_window = permutation[:, :, None] != window_positions[:, None, :]
                non_window = non_window.to(torch.int32).min(dim=-1).values.to(torch.bool)

                result_perm = torch.zeros_like(permutation)
                result_perm[page_batch_idx[:, None], perm_idx] = window_positions
            else:
                non_window = torch.ones_like(permutation, dtype=torch.bool)
                result_perm = torch.zeros_like(permutation)

            num_non_window = non_window.to(torch.int32).sum(dim=-1).cpu().tolist()
            for i in range(self.page_batch):
                result_perm[i, : num_non_window[i]] = permutation[i, non_window[i]]

            return retrieved[page_batch_idx[:, None], result_perm]

        retrieved_keys = _gather_and_reorder(self.key_blocks)
        retrieved_values = _gather_and_reorder(self.value_blocks)
        seq_lengths = self.cache_seq_lengths

        # Retrieve eviction info from the ring buffer
        if not self.disable_eviction:
            page_batch_idx = torch.arange(self.page_batch, device=self.device, dtype=torch.int32)

            ei_idx = torch.arange(self.dms_window_size, device=self.device, dtype=torch.int32)
            ei_idx = torch.minimum(
                ei_idx[None, :],
                _recent_position_size(self.cache_seq_lengths, self.dms_window_size)[:, None] - 1,
            )
            ei_idx = (
                _first_recent_position(
                    self.recent_info_position, self.cache_seq_lengths, self.dms_window_size
                )[:, None]
                + ei_idx
            ) % self.dms_window_size

            eviction_info = self.recent_info[page_batch_idx[:, None], ei_idx, 1]

        # Convert to left-padded layout if requested
        if not right_padded:

            def _left_pad(x, lens):
                total = x.shape[1]
                lens_list = lens.cpu().tolist()
                padded = []
                for i in range(self.page_batch):
                    valid = x[i, : lens_list[i]]
                    pad = torch.zeros(
                        total - lens_list[i], *valid.shape[1:], device=x.device, dtype=x.dtype
                    )
                    padded.append(torch.cat([pad, valid], dim=0))
                return torch.stack(padded, dim=0)

            retrieved_keys = _left_pad(retrieved_keys, seq_lengths)
            retrieved_values = _left_pad(retrieved_values, seq_lengths)
            seq_lengths = seq_lengths.reshape(self.batch_size, self.num_heads)
            if not self.disable_eviction:
                eviction_info = _left_pad(
                    eviction_info,
                    _recent_position_size(self.cache_seq_lengths, self.dms_window_size),
                )

        retrieved_keys = retrieved_keys.reshape(
            self.batch_size, self.num_heads, max_length, self.head_dim
        ).contiguous()
        retrieved_values = retrieved_values.reshape(
            self.batch_size, self.num_heads, max_length, self.head_dim
        ).contiguous()
        seq_lengths = seq_lengths.reshape(self.batch_size, self.num_heads)

        if not self.disable_eviction:
            eviction_info = eviction_info.reshape(self.batch_size, self.num_heads, -1)
        else:
            eviction_info = torch.zeros(
                self.batch_size,
                self.num_heads,
                max_length,
                device=retrieved_values.device,
                dtype=torch.int32,
            )

        return retrieved_keys, retrieved_values, seq_lengths, eviction_info


# ---------------------------------------------------------------------------
# Tests: fast_update_ignore_eviction
# ---------------------------------------------------------------------------


class TestFastUpdateIgnoreEviction:
    """Verify fast_update_ignore_eviction produces the same cache state as the regular update path."""

    @pytest.mark.parametrize("seed", range(5))
    def test_fast_update_matches_regular_update(self, seed):
        """fast_update_ignore_eviction should produce identical cache contents to update()."""
        torch.manual_seed(seed)

        max_elem = 32
        max_seq_len_bound = 128

        max_seq_len = torch.randint(2, max_seq_len_bound, (1,)).item()
        block_size = torch.randint(2, max_elem - 1, (1,)).item()
        dms_window_size = torch.randint(block_size + 1, max_elem, (1,)).item()
        batch_size = torch.randint(1, max_elem, (1,)).item()
        head = torch.randint(1, 3, (1,)).item()
        head_dim = torch.randint(1, 4, (1,)).item()
        max_context_length = 10 * max_seq_len

        cache_regular = ExtendedDMSPagedCacheLayer(
            dms_window_size=dms_window_size,
            max_context_length=max_context_length,
            block_size=block_size,
            accommodate_min_initial_context_length=max_context_length,
            disable_eviction=True,
        )
        cache_fast = ExtendedDMSPagedCacheLayer(
            dms_window_size=dms_window_size,
            max_context_length=max_context_length,
            block_size=block_size,
            accommodate_min_initial_context_length=max_context_length,
            disable_eviction=True,
        )

        for _ in range(5):
            seq_len = torch.randint(0, max_seq_len, (batch_size, head))
            key_states = torch.randint(0, 100, (batch_size, head, max_seq_len, head_dim))
            value_states = torch.randint(0, 100, (batch_size, head, max_seq_len, head_dim))

            if seq_len.max() == 0:
                continue

            cache_regular.update(
                key_states,
                value_states,
                {
                    "eviction_info": torch.zeros(batch_size, head, max_seq_len),
                    "sequence_lengths": seq_len,
                    "cumulative_length": max_seq_len,
                },
            )
            cache_fast.fast_update_ignore_eviction(key_states, value_states, seq_len)

            cont_regular = cache_regular.get_contiguous_cache()
            cont_fast = cache_fast.get_contiguous_cache()

            for regular_tensor, fast_tensor in zip(cont_regular, cont_fast):
                assert (regular_tensor == fast_tensor).all()


# ---------------------------------------------------------------------------
# Tests: paged cache update correctness
# ---------------------------------------------------------------------------


def _run_paged_cache_update_test(seed, disable_eviction):
    """Run a multi-step paged cache update test against a naive tracked reference.

    At each step, random KV pairs with eviction decisions are fed into the cache.
    A naive Python tracker mirrors the expected cache state, and the two are compared
    after each step for keys, values, eviction info, and left/right padding consistency.
    """
    torch.manual_seed(seed)
    max_val = 16
    upper_bound_seq_len = 100
    batch_size = torch.randint(1, max_val, (1,)).item()
    num_heads = torch.randint(1, max_val, (1,)).item()
    block_size = torch.randint(1, max_val, (1,)).item()
    head_dim = torch.randint(1, max_val, (1,)).item()
    dms_window_size = torch.randint(block_size + 1, max_val + 1, (1,)).item()

    cache = ExtendedDMSPagedCacheLayer(
        dms_window_size=dms_window_size,
        max_context_length=32768,
        block_size=block_size,
        accommodate_min_initial_context_length=torch.randint(1, max_val, (1,)).item(),
        disable_eviction=disable_eviction,
    )

    page_batch = batch_size * num_heads
    tracked_keys = [[] for _ in range(page_batch)]
    tracked_values = [[] for _ in range(page_batch)]
    tracked_eviction = [[] for _ in range(page_batch)]

    for step in range(10):
        max_seq_len = torch.randint(1, upper_bound_seq_len, (1,)).item()
        keys = torch.randint(0, 1_000_000_000, (batch_size, num_heads, max_seq_len, head_dim))
        values = torch.randint(0, 1_000_000_000, (batch_size, num_heads, max_seq_len, head_dim))
        eviction_info = (
            torch.randint(0, 2, (batch_size, num_heads, max_seq_len))
            if not disable_eviction
            else torch.zeros(batch_size, num_heads, max_seq_len)
        )

        seq_lengths = torch.randint(1, max_seq_len + 1, (batch_size, num_heads))

        # Occasionally force a sequence length of 1 for edge-case coverage
        if max_seq_len > 1 and torch.randint(0, 2, (1,)).item() == 1:
            p = torch.randint(0, batch_size, (1,)).item()
            q = torch.randint(0, num_heads, (1,)).item()
            seq_lengths[p, q] = 1

        keys_flat = keys.reshape(page_batch, max_seq_len, head_dim)
        values_flat = values.reshape(page_batch, max_seq_len, head_dim)
        eviction_flat = eviction_info.reshape(page_batch, max_seq_len)
        seq_lengths_flat = seq_lengths.reshape(page_batch)

        # --- Update naive tracked reference ---
        for j in range(page_batch):
            sl = seq_lengths_flat[j]
            if sl == 0:
                continue

            for s in range(max_seq_len - sl, max_seq_len):
                tracked_keys[j].append(keys_flat[j, [s]])
                tracked_values[j].append(values_flat[j, [s]])
                if len(tracked_eviction[j]) > 0:
                    tracked_eviction[j][-1] = eviction_flat[j, [s]]
                tracked_eviction[j].append(torch.zeros_like(eviction_flat[j, [s]]))

                if len(tracked_keys[j]) > dms_window_size:
                    if tracked_eviction[j][-dms_window_size - 1] == 1:
                        del tracked_keys[j][-dms_window_size - 1]
                        del tracked_values[j][-dms_window_size - 1]
                        del tracked_eviction[j][-dms_window_size - 1]

        # --- Update actual cache ---
        cache.update(
            key_states=keys,
            value_states=values,
            cache_kwargs={
                "eviction_info": eviction_info,
                "sequence_lengths": seq_lengths,
                "cumulative_length": 1,
            },
        )

        # --- Retrieve and verify ---
        (ret_keys, ret_values, cache_seq_lens, ret_eviction) = cache.get_contiguous_cache()
        (ret_keys_lp, ret_values_lp, cache_seq_lens_lp, ret_eviction_lp) = (
            cache.get_contiguous_cache(right_padded=False)
        )

        ret_keys = ret_keys.reshape(page_batch, -1, head_dim)
        ret_keys_lp = ret_keys_lp.reshape(page_batch, -1, head_dim)
        ret_values = ret_values.reshape(page_batch, -1, head_dim)
        ret_values_lp = ret_values_lp.reshape(page_batch, -1, head_dim)
        cache_seq_lens = cache_seq_lens.reshape(page_batch)
        cache_seq_lens_lp = cache_seq_lens_lp.reshape(page_batch)
        ret_eviction = ret_eviction.reshape(page_batch, -1)
        ret_eviction_lp = ret_eviction_lp.reshape(page_batch, -1)

        def _assert_tracked_matches(tracked, retrieved, j):
            """Assert that the tracked reference matches the retrieved cache for head j."""
            tracked_cat = torch.concat(tracked[j], dim=0)
            sl = cache_seq_lens[j].item()
            retrieved_trimmed = retrieved[j, :sl]

            assert len(retrieved_trimmed) == sl
            assert len(tracked_cat) == len(retrieved_trimmed), (
                f"step: {step}, j: {j}\n"
                f" tracked {tracked_cat.shape}: {tracked_cat}\n"
                f" retrieved {retrieved_trimmed.shape}: {retrieved_trimmed}"
            )

            # Recent window must match exactly
            a = tracked_cat[-dms_window_size:]
            b = retrieved_trimmed[-dms_window_size:]
            assert torch.allclose(a, b), (
                f"dms_window_size: {dms_window_size}, step: {step}, j: {j}\n"
                f" tracked {a.shape}: {a}\n retrieved {b.shape}: {b}"
            )

            # Older tokens may be reordered by the paged allocator but must all be present
            a = tracked_cat[:-dms_window_size]
            b = retrieved_trimmed[:-dms_window_size]
            if len(a) > 0:
                cmp = a[:, None, :] == b[None, :, :]
                cmp = cmp.to(torch.int32).min(dim=-1).values.max(dim=-1).values.to(torch.bool)
                assert cmp.all(), (
                    f"pref: dms_window_size: {dms_window_size}, step: {step}, j: {j}\n"
                    f" tracked {a.shape}: {a}\n retrieved {b.shape}: {b}"
                )

        for j in range(page_batch):
            sl = cache_seq_lens[j].item()
            sl_lp = cache_seq_lens_lp[j].item()

            # Left-padded view should match right-padded content
            assert (ret_keys_lp[j][-sl_lp:] == ret_keys[j][:sl]).all()
            assert (ret_values_lp[j][-sl_lp:] == ret_values[j][:sl]).all()
            assert (ret_eviction_lp[j][-sl_lp:] == ret_eviction[j][:sl]).all()

            # Compare tracked reference against retrieved cache
            _assert_tracked_matches(tracked_keys, ret_keys, j)
            _assert_tracked_matches(tracked_values, ret_values, j)

            # Verify eviction info
            a = ret_eviction[j][:sl]
            b = torch.concat(tracked_eviction[j][-len(a) :], dim=0)
            assert b[-1] == 0
            assert (a[:-1] == b[:-1]).all(), (
                f"step: {step}, j: {j}\n a {a.shape}: {a}\n b {b.shape}: {b}"
            )


class TestPagedCacheUpdate:
    """Verify paged cache update against a naive element-by-element tracked reference."""

    @pytest.mark.parametrize("seed", range(5))
    def test_update_with_eviction(self, seed):
        """Cache update with eviction enabled should match the tracked reference."""
        _run_paged_cache_update_test(seed, disable_eviction=False)

    @pytest.mark.parametrize("seed", range(5))
    def test_update_without_eviction(self, seed):
        """Cache update with eviction disabled should match the tracked reference."""
        _run_paged_cache_update_test(seed, disable_eviction=True)
