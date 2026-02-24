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

"""Paged DMS cache layer implementation with block-based memory management."""

import gc
import math
from typing import Any

import torch
from transformers import CacheLayerMixin


def ceil_int_div(a: int, b: int) -> int:
    """Return the ceiling of integer division a / b."""
    return (a + b - 1) // b


def float_ceil(a: float):
    """Return the ceiling of a float value."""
    return math.ceil(a)


def _aux_potential_eviction(
    vals_for_replacement: torch.Tensor,
    to_be_evicted_table_block_id: torch.Tensor,
    to_be_evicted_position_within_block: torch.Tensor,
    to_be_evicted_mask: torch.Tensor,
    block_table: torch.Tensor,
    blocks: torch.Tensor,
    page_batch_index: torch.Tensor,
    last_table_block_id: torch.Tensor,
    next_position_within_block: torch.Tensor,
):
    """Adding a new element to KV cache may lead to eviction of the last element in the DMS sliding window."""
    # For each batch element the block table contains a list of blocks allocated for this batch element
    block_ids = block_table[page_batch_index, to_be_evicted_table_block_id]

    # Override the last element of the sliding window with the new element if the last element of the sliding window
    # is marked for the eviction and the window is full
    blocks[block_ids, to_be_evicted_position_within_block, :, :] = (
        blocks[block_ids, to_be_evicted_position_within_block, :, :]
        * (1 - to_be_evicted_mask[:, None, None])
        + vals_for_replacement[:, 0, None, :] * to_be_evicted_mask[:, None, None]
    )

    # Otherwise write the new element to the next position within the last allocated block
    block_ids = block_table[page_batch_index, last_table_block_id]
    blocks[block_ids, next_position_within_block, :, :] = blocks[
        block_ids, next_position_within_block, :, :
    ] * to_be_evicted_mask[:, None, None] + vals_for_replacement[:, 0, None, :] * (
        1 - to_be_evicted_mask[:, None, None]
    )


def _aux_no_eviction(
    vals_for_replacement: torch.Tensor,
    block_table: torch.Tensor,
    blocks: torch.Tensor,
    page_batch_index: torch.Tensor,
    last_table_block_id: torch.Tensor,
    next_position_within_block: torch.Tensor,
):
    """Adding new element to kv cache without eviction of the last element."""
    # otherwise write the new element to the next position within the last
    # allocated block
    block_ids = block_table[page_batch_index, last_table_block_id]
    blocks[block_ids, next_position_within_block, :, :] = vals_for_replacement[:, 0, None, :]


@torch.compile()
def _aux_update_single(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    eviction_info: torch.Tensor,
    recent_info: torch.Tensor | None,
    recent_info_position: torch.Tensor | None,
    block_table: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    cache_seq_lengths: torch.Tensor,
    page_batch_index: torch.Tensor,
) -> torch.Tensor:
    """Updates the paged cache during token by token generation."""
    # page_batch, seq_len, head_dim = key_states.size()
    # page_batch, seq_len = eviction_info.size()
    # page_batch_index is a tensor of shape (page_batch,): 0, 1, 2, ... page_batch - 1
    block_size = key_blocks.size(1)

    last_table_block_id = cache_seq_lengths // block_size
    next_position_within_block = cache_seq_lengths % block_size

    # `recent_info_position` points to the next position in the sliding window; when the sliding window is full,
    # it points to the first position. Not-filled elements are not zeroed out and not marked for eviction
    # (see recent_info initialization).
    if recent_info is not None:  # DMS eviction is enabled
        assert recent_info_position is not None
        eviction_candidate_info_position = recent_info_position % recent_info.size(1)

        eviction_candidate_info = recent_info[
            page_batch_index, eviction_candidate_info_position
        ]  # Note that this is zeroed out in the beginning

        # `eviction_candidate_info[:, 1]` is 1 when the element is marked for eviction and 0 otherwise
        # `block_table[eviction_candidate_info[:, 0] // block_size]` is the block id where the element resides
        # and `eviction_candidate_info[:, 0] % block_size` is the position (offset) within the block
        to_be_evicted = eviction_candidate_info[:, 1] == 1
        to_be_evicted_kv = to_be_evicted.to(key_blocks.dtype)
        to_be_evicted_int = to_be_evicted.to(torch.int32)
        to_be_evicted_position = eviction_candidate_info[:, 0]
        to_be_evicted_table_block_id = to_be_evicted_position // block_size
        to_be_evicted_position_within_block = to_be_evicted_position % block_size

        _aux_potential_eviction(
            vals_for_replacement=key_states,
            to_be_evicted_table_block_id=to_be_evicted_table_block_id,
            to_be_evicted_position_within_block=to_be_evicted_position_within_block,
            to_be_evicted_mask=to_be_evicted_kv,
            block_table=block_table,
            blocks=key_blocks,
            page_batch_index=page_batch_index,
            last_table_block_id=last_table_block_id,
            next_position_within_block=next_position_within_block,
        )

        _aux_potential_eviction(
            vals_for_replacement=value_states,
            to_be_evicted_table_block_id=to_be_evicted_table_block_id,
            to_be_evicted_position_within_block=to_be_evicted_position_within_block,
            to_be_evicted_mask=to_be_evicted_kv,
            block_table=block_table,
            blocks=value_blocks,
            page_batch_index=page_batch_index,
            last_table_block_id=last_table_block_id,
            next_position_within_block=next_position_within_block,
        )

        final_position = to_be_evicted_position * to_be_evicted_int + (1 - to_be_evicted_int) * (
            cache_seq_lengths
        )

        previous_recent_info_position = (
            recent_info_position + recent_info.size(1) - 1
        ) % recent_info.size(1)

        # Update the eviction info for the previous element in the sliding window (if present)
        recent_info[page_batch_index, previous_recent_info_position, 1] = (
            eviction_info[:, 0] * (cache_seq_lengths > 0).to(torch.int32)
        ).to(torch.int32)

        # No info about eviction yet for the new element
        recent_info[page_batch_index, eviction_candidate_info_position, 1] = 0
        recent_info[page_batch_index, eviction_candidate_info_position, 0] = final_position

        recent_info_position[...] += 1

        cache_seq_lengths[...] = cache_seq_lengths + (1 - to_be_evicted_int)

        # At the beginning of this function call block_table[cache_seq_lengths // block_size] points to a block with
        # at least one free position; need to maintain this invariant by detecting filled blocks
        requires_free_page = torch.logical_and(
            (cache_seq_lengths % block_size) == 0, to_be_evicted_int == 0
        )

    else:  # DMS eviction is disabled
        _aux_no_eviction(
            vals_for_replacement=key_states,
            block_table=block_table,
            blocks=key_blocks,
            page_batch_index=page_batch_index,
            last_table_block_id=last_table_block_id,
            next_position_within_block=next_position_within_block,
        )
        _aux_no_eviction(
            vals_for_replacement=value_states,
            block_table=block_table,
            blocks=value_blocks,
            page_batch_index=page_batch_index,
            last_table_block_id=last_table_block_id,
            next_position_within_block=next_position_within_block,
        )
        cache_seq_lengths[...] = cache_seq_lengths + 1

        requires_free_page = (cache_seq_lengths % block_size) == 0

    return requires_free_page


def _aux_write_kv(
    block_table: torch.Tensor,
    blocks: torch.Tensor,
    write_positions: torch.Tensor,
    values: torch.Tensor,
    page_batch_index: torch.Tensor,
):
    _page_batch, _chunk_len = write_positions.size()
    block_size = blocks.size(1)
    block_table_id = write_positions // block_size
    position_within_block = write_positions % block_size

    block_id = block_table[page_batch_index[:, None], block_table_id]
    assert (block_id != -1).all(), f"block_id: {block_id} is -1"

    blocks[block_id, position_within_block, :, :] = values[:, :, None, :]


@torch.compile()
def _aux_update_many_handle_single_chunk(
    update_key_chunk: torch.Tensor,
    update_value_chunk: torch.Tensor,
    eviction_info_chunk: torch.Tensor,
    block_table: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    cache_seq_lengths: torch.Tensor,
    is_non_empty: torch.Tensor,
    recent_info: torch.Tensor | None,
    recent_info_position: torch.Tensor | None,
    page_batch_index: torch.Tensor,
    update_mask: torch.Tensor,
    true_update_size: torch.Tensor,
) -> torch.Tensor:
    """Used for prefilling the KV cache as each tensor has a fixed size.

    `true_update_size` represents the true number of elements to be added for each batch index.
    """
    assert update_key_chunk.size() == update_value_chunk.size(), (
        f"update_key_chunk.size: {update_key_chunk.size()} != update_value_chunk.size: {update_value_chunk.size()}"
    )
    page_batch, chunk_len, _head_dim = update_key_chunk.size()
    assert recent_info is None or chunk_len < recent_info.size(1), (
        f"recent_info {recent_info.shape} {chunk_len}"
    )

    assert eviction_info_chunk.size() == (page_batch, chunk_len), (
        f"eviction_info_chunk.size: {eviction_info_chunk.size()} != (page_batch, chunk_len): {(page_batch, chunk_len)}"
    )
    assert page_batch_index.size() == (page_batch,), (
        f"page_batch_index.size: {page_batch_index.size()} != (page_batch,): {(page_batch,)}"
    )

    block_size = key_blocks.size(1)

    device = update_key_chunk.device

    chunk_indexer = torch.arange(chunk_len, dtype=torch.int32, device=device)

    if recent_info is not None:  # DMS eviction is enabled
        assert recent_info_position is not None
        # First we update the eviction info for the previous element if present
        update_eviction_info_positions = (recent_info_position - 1) % recent_info.size(1)
        update_eviction_info_mask = (cache_seq_lengths > 0).to(torch.int32)

        recent_info[page_batch_index, update_eviction_info_positions, 1] = (
            eviction_info_chunk[:, 0] * update_eviction_info_mask
            + (1 - update_eviction_info_mask)
            * recent_info[page_batch_index, update_eviction_info_positions, 1]
        ).to(torch.int32)

        # The following trick handles variable lens: if the index is longer than true_update_size, then pad the index
        # with the last element within the true_update_size, e.g., [0, 1, 2, 3, 4, 5] and true_update_size = [3]
        # means that we have [0, 1, 2, 2, 2, 2] . This will later be used to write the same element multiple times
        # while preserving the constant shapes of the tensors.

        potential_eviction_positions_in_recent_info = (
            recent_info_position[:, None]
            + torch.minimum(chunk_indexer[None, :], true_update_size[:, None] - 1)
        ) % recent_info.size(1)

        potential_eviction_positions_in_seq = recent_info[
            page_batch_index[:, None], potential_eviction_positions_in_recent_info, 0
        ]
        confirmed_evictions_mask = (
            recent_info[
                page_batch_index[:, None],
                potential_eviction_positions_in_recent_info,
                1,
            ]
            == 1
        )

        confirmed_evictions_mask = torch.logical_and(
            confirmed_evictions_mask, is_non_empty[:, None]
        )

        # Account for the padding with the last element (as described above)
        # to get a proper count of confirmed evictions
        confirmed_evictions_mask[:, 1:] = torch.logical_and(
            confirmed_evictions_mask[:, 1:],
            potential_eviction_positions_in_recent_info[:, 1:]
            != potential_eviction_positions_in_recent_info[:, :-1],
        )

        confirmed_evictions_cum_sum = confirmed_evictions_mask.to(torch.int32).cumsum(dim=-1)
        confirmed_evictions_mask = torch.logical_and(
            confirmed_evictions_mask,
            confirmed_evictions_cum_sum <= true_update_size[:, None],
        )

        # Count how many new positions are needed for each element of the batch
        num_confirmed_evictions = confirmed_evictions_mask.to(torch.int32).sum(dim=-1)
        new_positions_used = true_update_size - num_confirmed_evictions

        assert (new_positions_used >= 0).all(), (
            f"new_positions_used: {new_positions_used} is less than 0"
        )
        assert new_positions_used.size() == (page_batch,), (
            f"new_positions_used.size: {new_positions_used.size()} != (page_batch,): {(page_batch,)}"
        )

        new_free_positions = cache_seq_lengths[:, None] + torch.clamp(
            torch.minimum(chunk_indexer[None, :], new_positions_used[:, None] - 1),
            min=0,
        )

        assert new_free_positions.size() == (page_batch, chunk_len), (
            f"new_free_positions.size: {new_free_positions.size()}"
            f" != (page_batch, chunk_len): {(page_batch, chunk_len)}"
        )
        assert new_free_positions.size() == potential_eviction_positions_in_seq.size(), (
            f"new_free_positions.size: {new_free_positions.size()}"
            f" != potential_eviction_positions_in_seq.size: {potential_eviction_positions_in_seq.size()}"
        )

        potential_eviction_positions_in_seq = torch.cat(
            [
                potential_eviction_positions_in_seq,
                new_free_positions,
            ],
            dim=-1,
        )

        # Padding below allows for constant shape ops to take prefix
        # of length new_positions_used from new_free_positions
        confirmed_evictions_padding = torch.zeros_like(confirmed_evictions_mask)
        padding_chunk_size = chunk_len - num_confirmed_evictions[:, None]
        indexer = torch.minimum(chunk_indexer[None, :], torch.clamp(padding_chunk_size - 1, min=0))

        confirmed_evictions_padding[page_batch_index[:, None], indexer] = True
        # If only post eviction positions are used, then have writing padding that ends in the last of those positions,
        # instead of the next free position
        confirmed_evictions_padding = torch.logical_and(
            confirmed_evictions_padding, padding_chunk_size > 0
        )

        confirmed_evictions_mask = torch.cat(
            [confirmed_evictions_mask, confirmed_evictions_padding], dim=-1
        )

        pad_selector = (new_positions_used > 0).to(torch.int32)[:, None]

        potential_eviction_positions_in_seq[:, chunk_len:] = (
            pad_selector * potential_eviction_positions_in_seq[:, chunk_len:]
            + (1 - pad_selector) * potential_eviction_positions_in_seq[:, [chunk_len - 1]]
        )

        new_write_positions = potential_eviction_positions_in_seq[confirmed_evictions_mask].reshape(
            page_batch, chunk_len
        )

        # Always perform dummy write for empty sequences
        new_write_positions = new_write_positions * is_non_empty[:, None] + cache_seq_lengths[
            :, None
        ] * (~is_non_empty[:, None])

        _aux_write_kv(
            block_table=block_table,
            blocks=key_blocks,
            write_positions=new_write_positions,
            values=update_key_chunk,
            page_batch_index=page_batch_index,
        )

        _aux_write_kv(
            block_table=block_table,
            blocks=value_blocks,
            write_positions=new_write_positions,
            values=update_value_chunk,
            page_batch_index=page_batch_index,
        )

        recent_indexer = torch.minimum(
            chunk_indexer[None, :], torch.clamp(true_update_size[:, None] - 1, min=0)
        )

        recent_info_indexer = (recent_info_position[:, None] + recent_indexer) % recent_info.size(1)

        # update the info about last window positions

        non_empty_update = (true_update_size[:, None] > 0).to(torch.int32)

        recent_info[page_batch_index[:, None], recent_info_indexer, 0] = (
            new_write_positions * non_empty_update
            + recent_info[page_batch_index[:, None], recent_info_indexer, 0]
            * (1 - non_empty_update)
        ).to(torch.int32)

        eviction_info_chunk = torch.cat(
            [
                eviction_info_chunk[:, 1:],
                torch.zeros_like(eviction_info_chunk[:, [0]]),
            ],
            dim=-1,
        )
        recent_info[page_batch_index[:, None], recent_info_indexer, 1] = (
            eviction_info_chunk[:, :] * non_empty_update
            + recent_info[page_batch_index[:, None], recent_info_indexer, 1]
            * (1 - non_empty_update)
        ).to(torch.int32)

        recent_info_position[...] += true_update_size

        cache_seq_lengths[...] += new_positions_used

        require_free_pages = torch.logical_and(
            new_positions_used > 0, cache_seq_lengths % block_size == 0
        )
    else:
        new_write_positions = cache_seq_lengths[:, None] + torch.clamp(
            torch.minimum(chunk_indexer[None, :], true_update_size[:, None] - 1), min=0
        )

        _aux_write_kv(
            block_table=block_table,
            blocks=key_blocks,
            write_positions=new_write_positions,
            values=update_key_chunk,
            page_batch_index=page_batch_index,
        )

        _aux_write_kv(
            block_table=block_table,
            blocks=value_blocks,
            write_positions=new_write_positions,
            values=update_value_chunk,
            page_batch_index=page_batch_index,
        )

        cache_seq_lengths[...] += true_update_size

        require_free_pages = torch.logical_and(
            true_update_size > 0, cache_seq_lengths % block_size == 0
        )

    return require_free_pages


class DMSPagedCacheLayer(CacheLayerMixin):
    """Paged cache layer with block-based storage and optional DMS eviction."""

    def __init__(
        self,
        dms_window_size: int,
        max_context_length: int,
        block_size: int = 256,
        growth_factor: float = 1.5,
        accommodate_min_initial_context_length: int = 4096,
        disable_eviction: bool = False,
    ):
        """Initialize the paged cache layer."""
        super().__init__()
        assert block_size <= dms_window_size, (
            f"block_size: {block_size} > dms_window_size: {dms_window_size}"
        )
        self.block_size = block_size
        self.dms_window_size = dms_window_size
        self.prefill_chunk_size = max(self.dms_window_size - 2, block_size)
        assert self.prefill_chunk_size > 0, (
            f"prefill_chunk_size: {self.prefill_chunk_size} is not greater than 0"
        )
        self.growth_factor = growth_factor
        self.min_initial_context_length = accommodate_min_initial_context_length
        self.disable_eviction = disable_eviction

        self.max_context_length = max_context_length

        self.max_blocks_per_sequence = ceil_int_div(self.max_context_length, self.block_size)

        self.key_blocks = None
        self.value_blocks = None
        self.block_table = None
        self.free_page_ids = None
        self.cache_seq_lengths = None
        self.recent_info = None  # Position and eviction info of last window_size keys/values
        self.recent_info_position = None

        self.device = None

        self.cumulative_length = 0

        self.prefill_mode = False

    def offload(self):
        """Offload cache tensors to CPU."""
        if self.key_blocks is not None:
            self.key_blocks = self.key_blocks.to("cpu", non_blocking=True)
            self.value_blocks = self.value_blocks.to("cpu", non_blocking=True)
            self.block_table = self.block_table.to("cpu", non_blocking=True)
            self.free_page_ids = self.free_page_ids.to("cpu", non_blocking=True)
            self.cache_seq_lengths = self.cache_seq_lengths.to("cpu", non_blocking=True)
            if self.recent_info is not None:
                self.recent_info = self.recent_info.to("cpu", non_blocking=True)
                self.recent_info_position = self.recent_info_position.to("cpu", non_blocking=True)

    def enable_prefill_mode(self):
        """Enable prefill mode and disable eviction."""
        self.prefill_mode = True
        self.disable_eviction = True
        self.recent_info = None
        self.recent_info_position = None

    def disable_prefill_mode(self, disable_eviction: bool):
        """Disable prefill mode and optionally re-enable eviction."""
        self.prefill_mode = False
        self.disable_eviction = disable_eviction
        if self.key_blocks is not None and self.recent_info is None and (not disable_eviction):
            self._initialize_recent_info()

    def prefetch(self):
        """Prefetch cache tensors back to the original device."""
        if self.key_blocks is not None and self.key_blocks.device != self.device:
            self.key_blocks = self.key_blocks.to(self.device, non_blocking=True)
            self.value_blocks = self.value_blocks.to(self.device, non_blocking=True)
            self.block_table = self.block_table.to(self.device, non_blocking=True)
            self.free_page_ids = self.free_page_ids.to(self.device, non_blocking=True)
            self.cache_seq_lengths = self.cache_seq_lengths.to(self.device, non_blocking=True)
            if self.recent_info is not None:
                self.recent_info = self.recent_info.to(self.device, non_blocking=True)
                self.recent_info_position = self.recent_info_position.to(
                    self.device, non_blocking=True
                )

    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""
        if self.key_blocks is not None:
            self.key_blocks = None
            self.value_blocks = None
            self.block_table = None
            self.free_page_ids = None
            self.cache_seq_lengths = None
            self.recent_info = None
            self.recent_info_position = None
            gc.collect()
            torch.cuda.empty_cache()
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        raise NotImplementedError("Beam search is not supported")

    def _get_free_pages(self, num_pages: int):
        assert self.free_page_ids is not None
        assert self.key_blocks is not None
        assert self.value_blocks is not None
        while len(self.free_page_ids) < num_pages:

            def expand_blocks(blocks: torch.Tensor):
                return torch.cat(
                    [
                        blocks,
                        torch.zeros(
                            (
                                float_ceil(blocks.size(0) * self.growth_factor) - blocks.size(0),
                                blocks.size(1),
                                blocks.size(2),
                                blocks.size(3),
                            ),
                            dtype=blocks.dtype,
                            device=blocks.device,
                        ),
                    ],
                    dim=0,
                )

            old_num_blocks = self.key_blocks.size(0)
            self.key_blocks = expand_blocks(self.key_blocks)
            self.value_blocks = expand_blocks(self.value_blocks)
            assert self.key_blocks.size(0) == self.value_blocks.size(0), (
                f"key_blocks.size: {self.key_blocks.size(0)} != value_blocks.size: {self.value_blocks.size(0)}"
            )
            self.free_page_ids = torch.cat(
                [
                    self.free_page_ids,
                    torch.arange(
                        old_num_blocks,
                        self.key_blocks.size(0),
                        dtype=torch.int32,
                        device=self.device,
                    ),
                ],
                dim=0,
            )

        result = self.free_page_ids[:num_pages]
        assert result.size() == (num_pages,), (
            f"result.size: {result.size()} != (num_pages,): {(num_pages,)}"
        )
        self.free_page_ids = self.free_page_ids[num_pages:]
        return result

    def _initialize_recent_info(self):
        assert self.cache_seq_lengths is not None
        self.recent_info = torch.zeros(
            (self.page_batch, self.dms_window_size, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self.recent_info_position = self.cache_seq_lengths.clone()

    def lazy_initialization(self, key_states: torch.Tensor):
        """Lazily initialize cache storage based on key state shape."""
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads, _, self.head_dim = key_states.shape

        self.page_batch = self.batch_size * self.num_heads

        initial_num_blocks = max(
            ceil_int_div(self.min_initial_context_length, self.block_size) * self.page_batch,
            self.page_batch,
        )

        self.block_table = -torch.ones(
            self.page_batch,
            self.max_blocks_per_sequence + 1,  # +1 for handling full cache case
            dtype=torch.int32,
            device=self.device,
        )
        self.key_blocks = torch.zeros(
            (initial_num_blocks, self.block_size, 1, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.value_blocks = torch.zeros(
            (initial_num_blocks, self.block_size, 1, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self.free_page_ids = torch.arange(
            0, initial_num_blocks, dtype=torch.int32, device=self.device
        )

        self.cache_seq_lengths = torch.zeros(self.page_batch, dtype=torch.int32, device=self.device)

        if not self.disable_eviction:
            self._initialize_recent_info()

        assert self.block_table is not None
        self.block_table[:, 0] = self._get_free_pages(self.block_table.size(0))

    def _handle_page_allocation(
        self, requires_free_page: torch.Tensor, page_batch_index: torch.Tensor
    ):
        assert self.block_table is not None
        assert self.cache_seq_lengths is not None
        if requires_free_page.any():
            req_free_pages = page_batch_index[requires_free_page]
            free_pages = self._get_free_pages(len(req_free_pages))

            self.block_table[
                req_free_pages,
                self.cache_seq_lengths[req_free_pages] // self.block_size,
            ] = free_pages

    def _update_single(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        eviction_info: torch.Tensor,
    ):
        batch_x_head, seq_len, _head_dim = key_states.size()
        page_batch_index = torch.arange(batch_x_head, dtype=torch.int32, device=self.device)

        assert seq_len == 1, f"seq_len: {seq_len} != 1"

        requires_free_page = _aux_update_single(
            key_states=key_states,
            value_states=value_states,
            eviction_info=eviction_info,
            recent_info=self.recent_info,
            recent_info_position=self.recent_info_position,
            block_table=self.block_table,
            key_blocks=self.key_blocks,
            value_blocks=self.value_blocks,
            cache_seq_lengths=self.cache_seq_lengths,
            page_batch_index=page_batch_index,
        )

        self._handle_page_allocation(
            requires_free_page=requires_free_page, page_batch_index=page_batch_index
        )

    def _update_many(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        eviction_info: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ):
        assert self.cache_seq_lengths is not None
        # Assume key and value states are left padded, e.g., [_, _, _, 1, 2, 3, 4]

        page_batch, seq_len, head_dim = key_states.size()
        assert page_batch == self.page_batch, (
            f"page_batch: {page_batch} != self.page_batch: {self.page_batch}"
        )
        assert head_dim == self.head_dim, f"head_dim: {head_dim} != self.head_dim: {self.head_dim}"
        assert eviction_info.size() == (page_batch, seq_len), (
            f"eviction_info.size: {eviction_info.size()} != (page_batch, seq_len): {(page_batch, seq_len)}"
        )
        assert sequence_lengths.ndim == 1, f"sequence_lengths.ndim: {sequence_lengths.ndim} != 1"

        is_non_empty = sequence_lengths > 0

        start_positions = seq_len - sequence_lengths

        end_positions = start_positions + sequence_lengths

        page_batch_index = torch.arange(page_batch, dtype=torch.int32, device=self.device)

        while (start_positions < end_positions).any():
            chunk_indexer = torch.arange(
                self.prefill_chunk_size, dtype=torch.int32, device=self.device
            )[None, :]

            update_mask = chunk_indexer < (
                self.block_size - (self.cache_seq_lengths[:, None] % self.block_size)
            )

            chunk_indexer = start_positions[:, None] + chunk_indexer

            update_mask = torch.logical_and(update_mask, chunk_indexer < end_positions[:, None])

            chunk_indexer = torch.clamp(
                torch.minimum(chunk_indexer, end_positions[:, None] - 1), min=0
            )

            true_update_size = update_mask.to(torch.int32).sum(dim=1)

            chunk_indexer = torch.clamp(
                torch.minimum(
                    chunk_indexer,
                    start_positions[:, None] + true_update_size[:, None] - 1,
                ),
                min=0,
            )

            key_chunk = key_states[page_batch_index[:, None], chunk_indexer]
            value_chunk = value_states[page_batch_index[:, None], chunk_indexer]
            eviction_info_chunk = eviction_info[page_batch_index[:, None], chunk_indexer]

            requires_free_page = _aux_update_many_handle_single_chunk(
                update_key_chunk=key_chunk,
                update_value_chunk=value_chunk,
                eviction_info_chunk=eviction_info_chunk,
                is_non_empty=is_non_empty,
                block_table=self.block_table,
                key_blocks=self.key_blocks,
                value_blocks=self.value_blocks,
                cache_seq_lengths=self.cache_seq_lengths,
                recent_info=self.recent_info,
                recent_info_position=self.recent_info_position,
                page_batch_index=page_batch_index,
                update_mask=update_mask,
                true_update_size=true_update_size,
            )

            self._handle_page_allocation(
                requires_free_page=requires_free_page, page_batch_index=page_batch_index
            )

            start_positions[...] += true_update_size

    def fast_update_ignore_eviction(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ):
        """Bulk-write key-value pairs to the paged cache without eviction."""
        if self.key_blocks is None:
            self.lazy_initialization(key_states)

        assert self.key_blocks is not None
        assert self.value_blocks is not None
        assert self.block_table is not None
        assert self.cache_seq_lengths is not None

        assert self.disable_eviction, (
            "fast_update_ignore_eviction is only supported when eviction is disabled"
        )

        assert sequence_lengths.max().item() != 0, (
            f"sequence_lengths.max(): {sequence_lengths.max().item()} is 0"
        )

        batch, head, seq_len, head_dim = key_states.size()

        self.cumulative_length += seq_len
        assert value_states.size() == key_states.size(), (
            f"value_states.size: {value_states.size()} != key_states.size: {key_states.size()}"
        )

        assert sequence_lengths.size() == (batch, head), (
            f"sequence_lengths.size: {sequence_lengths.size()} != (batch, head): {(batch, head)}"
        )

        key_states = key_states.reshape(self.page_batch, seq_len, head_dim)
        value_states = value_states.reshape(self.page_batch, seq_len, head_dim)
        sequence_lengths = sequence_lengths.reshape(self.page_batch)

        assert self.cache_seq_lengths.size() == sequence_lengths.size(), (
            f"cache_seq_lengths.size: {self.cache_seq_lengths.size()}"
            f" != sequence_lengths.size: {sequence_lengths.size()}"
        )

        last_allocated_block_index = self.cache_seq_lengths // self.block_size
        last_to_allocate_block_index = (
            self.cache_seq_lengths + sequence_lengths
        ) // self.block_size

        per_page_batch_blocks_available = last_allocated_block_index + 1
        per_page_batch_blocks_required = last_to_allocate_block_index + 1

        blocks_to_alloc = per_page_batch_blocks_required - per_page_batch_blocks_available
        max_blocks_to_alloc = blocks_to_alloc.max().item()

        if max_blocks_to_alloc > 0:
            assert blocks_to_alloc.shape == (self.page_batch,), (
                f"blocks_to_alloc.shape: {blocks_to_alloc.shape} != (self.page_batch,): {(self.page_batch,)}"
            )
            assert (blocks_to_alloc >= 0).all(), (
                f"blocks_to_alloc: {blocks_to_alloc} is less than 0"
            )

            total_blocks_to_alloc = blocks_to_alloc.sum()

            free_blocks = self._get_free_pages(total_blocks_to_alloc)

            # Greedy assignment of free blocks
            free_block_indexer = blocks_to_alloc.cumsum(dim=0)

            free_block_write_indexer = torch.arange(
                max_blocks_to_alloc, device=self.device, dtype=torch.int32
            )
            # For writing into the block table
            free_block_write_indexer = (
                last_allocated_block_index[:, None] + 1 + free_block_write_indexer[None, :]
            )

            assert free_block_write_indexer.size() == (
                self.page_batch,
                max_blocks_to_alloc,
            ), (
                f"free_block_write_indexer.size: {free_block_write_indexer.size()}"
                f" != (self.page_batch, max_blocks_to_alloc):"
                f" {(self.page_batch, max_blocks_to_alloc)}"
            )

            # +1 acts as a sink for handling sizes < max_blocks_to_alloc
            # free_block_write_indexer[a, b] where to write b'th free block for a'th page batch
            free_block_write_indexer = torch.minimum(
                free_block_write_indexer,
                last_allocated_block_index[:, None] + blocks_to_alloc[:, None] + 1,
            )

            # free_blocks[free_block_get_indexer[a, b]] is the b'th free block for a'th page batch
            free_block_get_offset = torch.nn.functional.pad(free_block_indexer, (1, -1), value=0)
            free_block_get_indexer = torch.arange(
                max_blocks_to_alloc, device=self.device, dtype=torch.int32
            )
            free_block_get_indexer = torch.minimum(
                free_block_get_indexer[None, :], blocks_to_alloc[:, None]
            )
            free_block_get_indexer = free_block_get_offset[:, None] + free_block_get_indexer
            free_block_get_indexer = torch.clamp(
                free_block_get_indexer, max=total_blocks_to_alloc - 1
            )

            free_block_assignment = free_blocks[free_block_get_indexer]
            assert free_block_assignment.shape == (self.page_batch, max_blocks_to_alloc), (
                f"free_block_assignment.shape: {free_block_assignment.shape}"
                f" != (self.page_batch, max_blocks_to_alloc):"
                f" {(self.page_batch, max_blocks_to_alloc)}"
            )

            # If max_blocks_to_alloc is more than the number of blocks that we want

            mask = (free_block_write_indexer <= last_to_allocate_block_index[:, None]).to(
                torch.int32
            )

            masked_free_block_assignment = free_block_assignment * mask - (
                1 - mask
            ) * torch.ones_like(free_block_assignment)

            self.block_table.scatter_(
                dim=1, index=free_block_write_indexer, src=masked_free_block_assignment
            )

        write_seq_positions = torch.arange(
            seq_len,
            device=self.cache_seq_lengths.device,
            dtype=self.cache_seq_lengths.dtype,
        )
        write_seq_positions = self.cache_seq_lengths[:, None] + write_seq_positions[None, :]
        write_seq_positions = torch.minimum(
            write_seq_positions, (self.cache_seq_lengths + sequence_lengths)[:, None]
        )

        source_seq_positions = torch.arange(
            seq_len,
            device=self.cache_seq_lengths.device,
            dtype=self.cache_seq_lengths.dtype,
        )

        # Left padded input

        source_seq_positions = (seq_len - sequence_lengths)[:, None] + source_seq_positions[None, :]
        source_seq_positions = torch.clamp(source_seq_positions, max=seq_len - 1)
        assert source_seq_positions.size() == (self.page_batch, seq_len), (
            f"source_seq_positions.size: {source_seq_positions.size()}"
            f" != (self.page_batch, seq_len): {(self.page_batch, seq_len)}"
        )

        write_block_table_ids = write_seq_positions // self.block_size
        write_block_offsets = write_seq_positions % self.block_size

        assert write_block_table_ids.size() == (self.page_batch, seq_len), (
            f"write_block_table_ids.size: {write_block_table_ids.size()}"
            f" != (self.page_batch, seq_len): {(self.page_batch, seq_len)}"
        )
        assert write_block_offsets.size() == (self.page_batch, seq_len), (
            f"write_block_offsets.size: {write_block_offsets.size()}"
            f" != (self.page_batch, seq_len): {(self.page_batch, seq_len)}"
        )

        write_block_ids = self.block_table.gather(dim=1, index=write_block_table_ids)
        assert write_block_ids.size() == (self.page_batch, seq_len), (
            f"write_block_ids.size: {write_block_ids.size()}"
            f" != (self.page_batch, seq_len): {(self.page_batch, seq_len)}"
        )

        source_seq_positions = source_seq_positions[:, :, None].broadcast_to(
            self.page_batch, seq_len, head_dim
        )

        keys_to_write = key_states.gather(dim=1, index=source_seq_positions)
        self.key_blocks[write_block_ids, write_block_offsets, 0, :] = keys_to_write

        values_to_write = value_states.gather(dim=1, index=source_seq_positions)
        self.value_blocks[write_block_ids, write_block_offsets, 0, :] = values_to_write

        self.cache_seq_lengths += sequence_lengths

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any],
    ):
        """Update the paged cache with new key-value states."""
        eviction_info = cache_kwargs["eviction_info"]
        sequence_lengths = cache_kwargs["sequence_lengths"]
        cumulative_length = cache_kwargs["cumulative_length"]

        if self.key_blocks is None:
            self.lazy_initialization(key_states)

        batch, head, seq_len, head_dim = key_states.size()
        assert key_states.size() == value_states.size(), (
            f"key_states.size: {key_states.size()} != value_states.size: {value_states.size()}"
        )
        assert key_states.size()[:3] == eviction_info.size(), (
            f"key_states.size()[:3]: {key_states.size()[:3]} != eviction_info.size(): {eviction_info.size()}"
        )
        assert sequence_lengths is None or sequence_lengths.size() == (batch, head), (
            f"sequence_lengths.size: {sequence_lengths.size()} != (batch, head): {(batch, head)}"
        )

        assert batch * head == self.page_batch, (
            f"batch * head: {batch * head} != self.page_batch: {self.page_batch}"
        )
        assert self.head_dim == head_dim, f"self.head_dim: {self.head_dim} != head_dim: {head_dim}"

        key_states = key_states.reshape(self.page_batch, seq_len, head_dim)
        value_states = value_states.reshape(self.page_batch, seq_len, head_dim)
        eviction_info = eviction_info.reshape(self.page_batch, seq_len)
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths.reshape(self.page_batch)

        if seq_len == 1 and not self.prefill_mode:
            assert sequence_lengths is None or (sequence_lengths == 1).all()
            assert cumulative_length == 1
            self._update_single(
                key_states=key_states,
                value_states=value_states,
                eviction_info=eviction_info,
            )
        else:
            self._update_many(
                key_states=key_states,
                value_states=value_states,
                eviction_info=eviction_info,
                sequence_lengths=sequence_lengths,
            )

        self.cumulative_length += cumulative_length

        return None, None

    def get_block_table(self):
        """Get the block table mapping."""
        return self.block_table

    def get_key_blocks(self):
        """Get the key cache blocks."""
        return self.key_blocks

    def get_value_blocks(self):
        """Get the value cache blocks."""
        return self.value_blocks

    def get_seq_lengths(self):
        """Get the sequence lengths per batch element."""
        return self.cache_seq_lengths

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Returns the length and offset of the cache, used to generate the mask."""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object."""
        return self.max_context_length
