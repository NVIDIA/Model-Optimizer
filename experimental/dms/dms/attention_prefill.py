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

"""Exact DMS prefill attention with eviction-based sparse masking and cache rewriting."""

from collections.abc import Callable

import torch
from torch.nn.attention.flex_attention import AuxRequest, flex_attention

from dms.cache import DMSCombinedCacheLayer
from dms.logging import get_logger

logger = get_logger("AttentionPrefill")


# =============================================================================
# Cache rewriting utilities
# =============================================================================


def rewrite_cache_in_left_padding_style(
    compressed_attention_mask: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    eviction_info: torch.Tensor,
):
    """Rewrite cache entries in left-padded format, removing evicted tokens.

    Args:
      - compressed_attention_mask: torch.Tensor of shape (batch, kv_seq_len)
      that for each key specifies how far to the right the key is visible from the query.
      0 denotes attention masking.
      - key_states: torch.Tensor of shape (batch, heads_kv, kv_seq_len, head_dim)
      - value_states: torch.Tensor of shape (batch, heads_kv, kv_seq_len, head_dim)
      - eviction_info: torch.Tensor of shape (batch, heads_kv, kv_seq_len).

    Returns:
      - left padded, potentially pruned (eviction) version of the key, value and eviction info tensors
    """
    _batch, heads_kv, kv_seq_len, head_dim = key_states.shape
    assert heads_kv == 1, "kv heads should be merged into batch dim"

    new_space_size = kv_seq_len + 1

    new_key_states, new_value_states, new_eviction_info, how_many_to_maintain = (
        _rewrite_cache_in_left_padding_style_aux(
            compressed_attention_mask=compressed_attention_mask,
            key_states=key_states,
            value_states=value_states,
            eviction_info=eviction_info,
            heads_kv=heads_kv,
            head_dim=head_dim,
            new_space_size=new_space_size,
        )
    )
    return new_key_states, new_value_states, new_eviction_info, how_many_to_maintain


@torch.compile()
def _rewrite_cache_in_left_padding_style_aux(
    compressed_attention_mask: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    eviction_info: torch.Tensor,
    heads_kv: int,
    head_dim: int,
    new_space_size: int,
):
    batch, kv_seq_len = compressed_attention_mask.shape
    # elements that should be evicted before cache end are removed
    should_remove = compressed_attention_mask < kv_seq_len

    should_maintain = torch.logical_not(should_remove).to(torch.int32)

    maintain_id = should_maintain.cumsum(dim=1)
    how_many_to_maintain = maintain_id[:, -1]
    # write elements to their new positions omitting removed elements
    write_indexer = new_space_size - how_many_to_maintain[:, None] + maintain_id - 1
    # removed elements will be written to position 0
    write_indexer[should_remove] = 0

    new_key_states = torch.empty(
        batch, heads_kv, new_space_size, head_dim, device=key_states.device, dtype=key_states.dtype
    )
    new_value_states = torch.empty(
        batch,
        heads_kv,
        new_space_size,
        head_dim,
        device=value_states.device,
        dtype=value_states.dtype,
    )
    new_eviction_info = torch.empty(
        batch, new_space_size, device=eviction_info.device, dtype=eviction_info.dtype
    )

    assert write_indexer.shape == (batch, kv_seq_len), (
        f"write_indexer.shape: {write_indexer.shape} != (batch, kv_seq_len): {(batch, kv_seq_len)}"
    )
    assert eviction_info.shape == (batch, kv_seq_len), (
        f"eviction_info.shape: {eviction_info.shape} != (batch, kv_seq_len): {(batch, kv_seq_len)}"
    )
    new_eviction_info.scatter_(dim=1, index=write_indexer, src=eviction_info)

    write_indexer = write_indexer[:, None, :, None].broadcast_to(
        batch, heads_kv, kv_seq_len, key_states.shape[3]
    )
    new_key_states.scatter_(dim=2, index=write_indexer, src=key_states)
    new_value_states.scatter_(dim=2, index=write_indexer, src=value_states)

    is_padding = (
        torch.arange(new_space_size - 1, -1, -1, device=new_key_states.device, dtype=torch.int32)[
            None, :
        ]
        >= how_many_to_maintain[:, None]
    )

    kv_states_mask = is_padding[:, None, :, None].broadcast_to(
        batch, heads_kv, new_space_size, head_dim
    )

    new_key_states[kv_states_mask] = 0
    new_value_states[kv_states_mask] = 0
    new_eviction_info[is_padding[:, :]] = 0

    return new_key_states, new_value_states, new_eviction_info, how_many_to_maintain


# =============================================================================
# Prefill attention with FlexAttention
# =============================================================================


def wrapped_flex_attention(query, key, value, score_mod, scale, enable_gqa):
    """Run flex attention with LSE auxiliary output."""
    return flex_attention(
        query=query,
        key=key,
        value=value,
        score_mod=score_mod,
        scale=scale,
        enable_gqa=enable_gqa,
        return_aux=AuxRequest(lse=True),
    )


@torch.compile()
def compiled_flex_attention(*args, **kwargs):
    """Compile and run flex attention with LSE auxiliary output."""
    return wrapped_flex_attention(*args, **kwargs)


def get_mask(
    dms_window_size: int,
    compressed_attention_mask: torch.Tensor,
    q_seq_len: int,
    gqa_factor: int,
    flex_attention_fn: Callable,
):
    """Build a score modification function for DMS sparse attention masking."""
    _page_batch, kv_seq_len = compressed_attention_mask.shape
    q_offset = kv_seq_len - q_seq_len

    def score_mod(score, b, h, q_idx, k_idx):
        causal = q_idx + q_offset >= k_idx

        within_range = q_idx + q_offset < compressed_attention_mask[b, k_idx]
        can_attend = torch.logical_and(causal, within_range).to(score.dtype)

        return can_attend * score + (1 - can_attend) * (-1e5)

    return score_mod, flex_attention_fn


def dms_prefill_flex(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    decisions: torch.Tensor,
    attn_mask: torch.Tensor,
    cache: DMSCombinedCacheLayer,
    attn_scaling: float,
    flash_attn_fn: Callable,
    q_flash: torch.Tensor,
    flex_attention_fn: Callable = compiled_flex_attention,
):
    """Compute DMS prefill attention using FlexAttention with eviction-based sparse masking.

    This function performs attention over both recent (contiguous) and paged KV caches,
    applying eviction decisions to determine which KV pairs remain visible.
    It combines attention outputs from both cache types using softmax LSE rescaling,
    then updates the cache by evicting dms marked tokens and rewriting in left-padded format.

    Args:
        queries: Query tensor of shape (batch, heads_q, seq_len, head_dim).
        keys: Key tensor of shape (batch, heads_kv, seq_len, head_dim).
        values: Value tensor of shape (batch, heads_kv, seq_len, head_dim).
        decisions: Eviction decisions per token (0=keep, 1=evict after window).
        attn_mask: Boolean attention mask of shape (batch, seq_len).
        cache: Combined DMS cache containing recent and paged KV storage.
        attn_scaling: Scaling factor for attention scores.
        flash_attn_fn: Flash attention function for paged cache computation.
        q_flash: Query tensor in flash attention layout for paged cache.
        flex_attention_fn: Flex attention function for local cache computation.

    Returns:
        Attention output tensor of shape (batch, heads_q, seq_len, head_dim).
    """
    assert decisions.dtype in (torch.int32, torch.long), (
        f"decisions.dtype: {decisions.dtype} is not int32 or long"
    )
    batch_k, heads_kv, seq_len_k, head_dim_k = keys.size()
    batch_q, heads_q, seq_len_q, head_dim_q = queries.size()
    assert keys.size() == values.size(), f"keys.size: {keys.size()} != values.size: {values.size()}"
    assert batch_q == batch_k, f"batch_q: {batch_q} != batch_k: {batch_k}"
    assert seq_len_q == seq_len_k, (
        f"dms_prefill_flex handles cache by itself, "
        f"so query and key must have the same sequence length: q_seq_len: {seq_len_q} k_seq_len: {seq_len_k}"
    )
    assert head_dim_k == head_dim_q, f"head_dim_k: {head_dim_k} != head_dim_q: {head_dim_q}"
    assert heads_kv <= heads_q, f"heads_kv: {heads_kv} > heads_q: {heads_q}"
    assert decisions.size() == (batch_q, heads_kv, seq_len_k), (
        f"decisions.size: {decisions.size()} != (batch_q, heads_kv, seq_len_k): {(batch_q, heads_kv, seq_len_k)}"
    )

    batch = batch_q
    head_dim = head_dim_q

    page_batch = batch * heads_kv

    gqa_factor = heads_q // heads_kv

    keys = keys.reshape(page_batch, 1, seq_len_k, head_dim_k)
    values = values.reshape(page_batch, 1, seq_len_k, head_dim_k)
    queries = queries.reshape(page_batch, gqa_factor, seq_len_q, head_dim_q)

    decisions = decisions.reshape(page_batch, seq_len_k)

    assert attn_mask.size() == (
        batch,
        seq_len_k,
    ), f"Attention mask shape does not match: {attn_mask.size()} != {batch, seq_len_k}"

    assert attn_mask.dtype == torch.bool, f"Attention mask dtype is not bool: {attn_mask.dtype}"

    # transformers uses False to mask out positions
    # here we use True to mask out positions
    attn_mask = torch.logical_not(attn_mask)

    # used to zero out results for masked positions
    results_masking = attn_mask[:, None, :, None]

    attn_mask = (
        attn_mask[:, None, :]
        .broadcast_to(batch, heads_kv, seq_len_k)
        .reshape(batch_q * heads_kv, seq_len_k)
    )

    # eviction info about  i'th token is produced by i'th+1 token
    # and may require carrying over to the cached kv pairs
    eviction_info = decisions.clone()
    eviction_info_carry = eviction_info[:, 0]
    # we assume contiguous masking
    eviction_info = torch.nn.functional.pad(eviction_info[:, 1:], (0, 1), value=0)

    assert eviction_info.shape == attn_mask.shape, (
        f"eviction_info: {eviction_info.shape} attn_mask: {attn_mask.shape}"
    )
    eviction_info[attn_mask] = 2  # 0 - no eviction, 1 - dms eviction, 2 - attention mask

    assert isinstance(cache, DMSCombinedCacheLayer), (
        f"requires DMSCombinedCacheLayer, got {type(cache)}"
    )

    if cache.get_recent_cache_csize() > 0:
        past_keys, past_values, past_cache_seq_lengths, past_eviction_info = (
            cache.get_recent_cache()
        )

        past_keys = past_keys.reshape(page_batch, 1, -1, head_dim)
        past_values = past_values.reshape(page_batch, 1, -1, head_dim)
        past_cache_seq_lengths = past_cache_seq_lengths.reshape(page_batch)
        past_eviction_info = past_eviction_info.reshape(page_batch, -1)

        _, _, past_seq_len, _ = past_keys.shape

        keys = torch.cat([past_keys, keys], dim=2)
        values = torch.cat([past_values, values], dim=2)

        assert past_eviction_info.size() == (page_batch, past_seq_len)
        # cache should be left padded
        past_eviction_info = torch.nn.functional.pad(past_eviction_info[:, 1:], (0, 1), value=0)

        past_eviction_info[:, -1] = eviction_info_carry

        # mask out padding in the prefix
        padded_prefix_indexer = torch.arange(
            past_seq_len - 1, -1, -1, device=keys.device, dtype=torch.int32
        )
        assert past_cache_seq_lengths.size() == (page_batch,)
        padded_prefix_indexer = padded_prefix_indexer[None, :] >= past_cache_seq_lengths[:, None]
        assert past_eviction_info.size() == (page_batch, past_seq_len)
        past_eviction_info[padded_prefix_indexer] = 2  # 1 - dms eviction 2-attention mask

        eviction_info = torch.cat(
            [
                past_eviction_info.to(torch.int32),
                eviction_info,
            ],
            dim=1,
        )

    dms_window_size = cache.cont_cache.dms_window_size
    total_padded_seq_len = keys.shape[2]

    # paddle paddle flashmask style attention mask
    # that for each kv-pair we say till what position the it is visible from the query
    compressed_attention_mask = torch.full_like(
        eviction_info, total_padded_seq_len, dtype=torch.int32
    )
    assert compressed_attention_mask.shape == (page_batch, total_padded_seq_len), (
        f"compressed_attention_mask.shape: {compressed_attention_mask.shape}"
        f" != (page_batch, total_padded_seq_len): {(page_batch, total_padded_seq_len)}"
    )

    position_indexer = torch.arange(total_padded_seq_len, device=keys.device, dtype=torch.int32)
    position_indexer = position_indexer[None, :]
    position_indexer = position_indexer.broadcast_to(page_batch, total_padded_seq_len)

    compressed_attention_mask[eviction_info == 2] = 0  # attention mask/padding
    compressed_attention_mask[eviction_info == 1] = (
        position_indexer[eviction_info == 1]
        + dms_window_size  # Warning we do not support attention gaps
    )

    compressed_attention_mask = torch.clamp(compressed_attention_mask, max=total_padded_seq_len)

    score_mod_fn, attention_fn = get_mask(
        dms_window_size=dms_window_size,
        compressed_attention_mask=compressed_attention_mask,
        q_seq_len=seq_len_q,
        gqa_factor=gqa_factor,
        flex_attention_fn=flex_attention_fn,
    )

    attn_output_local, aux_request = attention_fn(
        query=queries,
        key=keys,
        value=values,
        score_mod=score_mod_fn,
        scale=attn_scaling,
        enable_gqa=True,
    )

    attn_output_local = attn_output_local.reshape(batch, heads_q, seq_len_q, head_dim_q)

    if cache.get_paged_cache_csize() > 0:
        paged_cache = cache.get_paged_cache()

        attention_output_paged, softmax_lse_paged = flash_attn_fn(
            q_flash,
            paged_cache.get_key_blocks(),
            paged_cache.get_value_blocks(),
            k=None,
            v=None,
            cache_seqlens=paged_cache.get_seq_lengths(),
            causal=False,
            softmax_scale=attn_scaling,
            block_table=paged_cache.get_block_table(),
            return_softmax_lse=True,
        )

        softmax_lse_paged = torch.where(torch.isinf(softmax_lse_paged), 0, softmax_lse_paged)

        attention_output_paged = attention_output_paged.reshape(
            batch, heads_kv, seq_len_q, gqa_factor, head_dim_k
        ).transpose(2, 3)

        attention_output_paged = attention_output_paged.reshape(batch, heads_q, seq_len_q, head_dim)

        softmax_lse_local = aux_request.lse
        denom_local = torch.exp(softmax_lse_local.float())
        denom_local = denom_local.reshape(batch, heads_q, seq_len_q)
        denom_paged = torch.exp(softmax_lse_paged.float())
        denom_paged = denom_paged.reshape(batch, heads_q, seq_len_q)

        new_denom = denom_local + denom_paged

        denom_changer_local = (denom_local / new_denom).to(attn_output_local.dtype)
        denom_changer_local = denom_changer_local[:, :, :, None]
        denom_changer_paged = (denom_paged / new_denom).to(attention_output_paged.dtype)
        denom_changer_paged = denom_changer_paged[:, :, :, None]

        assert denom_changer_local.ndim == attn_output_local.ndim, (
            f"denom_changer_local.ndim: {denom_changer_local.ndim} != attn_output_local.ndim: {attn_output_local.ndim}"
        )
        assert denom_changer_paged.ndim == attention_output_paged.ndim, (
            f"denom_changer_paged.ndim: {denom_changer_paged.ndim}"
            f" != attention_output_paged.ndim: {attention_output_paged.ndim}"
        )
        assert attention_output_paged.ndim == attn_output_local.ndim, (
            f"attention_output_paged.ndim: {attention_output_paged.ndim}"
            f" != attn_output_local.ndim: {attn_output_local.ndim}"
        )

        attn_output = (
            attn_output_local * denom_changer_local + attention_output_paged * denom_changer_paged
        )
    else:
        attn_output = attn_output_local

    attn_output[results_masking.broadcast_to(batch, heads_q, seq_len_q, head_dim_q)] = 0

    # performs eviction and rewrites cache in left padding style
    new_key_states, new_value_states, new_eviction_info, seq_lengths = (
        rewrite_cache_in_left_padding_style(
            compressed_attention_mask=compressed_attention_mask,
            key_states=keys,
            value_states=values,
            eviction_info=eviction_info,
        )
    )

    cache.update(
        key_states=new_key_states.reshape(batch, heads_kv, -1, head_dim),
        value_states=new_value_states.reshape(batch, heads_kv, -1, head_dim),
        cache_kwargs={
            "eviction_info": torch.nn.functional.pad(
                new_eviction_info[..., :-1].reshape(batch, heads_kv, -1),
                (1, 0),
                value=0,
            ),
            "sequence_lengths": seq_lengths.reshape(batch, heads_kv),
            "cumulative_length": seq_len_q,
        },
    )

    return attn_output


def dms_run_prefill_flex(
    q_flash: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    decisions: torch.Tensor,
    attn_mask: torch.Tensor,
    cache: DMSCombinedCacheLayer,
    attn_scaling: float,
    flash_attn_fn: Callable,
    flex_attention_fn: Callable = compiled_flex_attention,
):
    """Run DMS prefill using FlexAttention, reshaping tensors for flash layout."""
    _page_batch, seq_len_q, gqa_factor, head_dim_q = q_flash.size()
    batch, head_k, _seq_len_k, head_dim_k = keys.size()

    head_q = gqa_factor * head_k

    queries = q_flash.transpose(1, 2).reshape(batch, head_q, seq_len_q, head_dim_q)

    attn_output = dms_prefill_flex(
        queries=queries,
        keys=keys,
        values=values,
        decisions=decisions,
        attn_mask=attn_mask,
        cache=cache,
        attn_scaling=attn_scaling,
        flash_attn_fn=flash_attn_fn,
        q_flash=q_flash,
        flex_attention_fn=flex_attention_fn,
    )

    assert attn_output.shape == (batch, head_q, seq_len_q, head_dim_q), (
        f"attn_output.shape: {attn_output.shape} != {batch, head_q, seq_len_q, head_dim_q}"
    )

    attn_output = attn_output.reshape(batch, head_k, gqa_factor, seq_len_q, head_dim_k).transpose(
        2, 3
    )

    return attn_output
