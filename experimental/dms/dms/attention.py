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

"""DMS attention: dispatch, training mode (FlexAttention), and inference mode (Flash Attention)."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.attention.flex_attention import flex_attention

from dms.cache import DMSCache
from dms.logging import get_logger

if TYPE_CHECKING:
    from dms.cache import DMSCombinedCacheLayer

logger = get_logger("Attention")

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError as e:
    logger.warning(f"Error importing flash_attn_with_kvcache: {e}")
    flash_attn_with_kvcache = None

try:
    from dms.attention_prefill import dms_run_prefill_flex
except ImportError as e:
    logger.warning(f"Error importing dms_run_prefill_flex: {e}")
    dms_run_prefill_flex = None


# =============================================================================
# Dispatch
# =============================================================================


def dms_attention(
    new_q_flash: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    decisions: torch.Tensor,
    decision_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    dms_cache: DMSCache | None,
    attn_scaling: float,
    window_size: int,
    train_attn_kwargs: dict[str, Any] = {},
):
    """Handles prefill/inference of DMS (Dynamic Memory Sparsification).

    If dms_cache is None, we are in train mode, otherwise we are in eval mode.
    """
    if dms_cache is None:
        # train mode
        attn_output = dms_attn_train_mode(
            q_flash=new_q_flash,
            k=new_k,
            v=new_v,
            decision_logits=decision_logits,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            attn_scaling=attn_scaling,
            window_size=window_size,
            train_attn_kwargs=train_attn_kwargs,
        )
    else:
        # eval mode
        decisions = decisions.to(torch.int32)

        attn_output = dms_attn_eval_mode(
            new_q_flash=new_q_flash,
            new_k=new_k,
            new_v=new_v,
            decisions=decisions,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            dms_cache=dms_cache,
            attn_scaling=attn_scaling,
        )

    return attn_output


# =============================================================================
# Training mode (FlexAttention with soft gating)
# =============================================================================

MASK_VALUE = -50000.0  # score used to mask tokens in attention


def dms_attn_train_mode(
    q_flash: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decision_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    attn_scaling: float,
    window_size: int,
    train_attn_kwargs: dict[str, Any] = {},
):
    """Perform DMS attention in training mode using FlexAttention with soft gating."""
    page_batch, seq_len_qf, gqa_factor, head_dim_qf = q_flash.size()
    batch, head_k, seq_len_k, head_dim_k = k.size()
    assert page_batch == batch * head_k, (
        f"page_batch: {page_batch} != batch * head_k: {batch * head_k}"
    )
    assert seq_len_qf == seq_len_k, f"seq_len_qf: {seq_len_qf} != seq_len_k: {seq_len_k}"
    assert head_dim_qf == head_dim_k, f"head_dim_qf: {head_dim_qf} != head_dim_k: {head_dim_k}"

    seq_len = seq_len_k

    assert v.size() == k.size(), f"v: {v.size()} k: {k.size()}"

    assert decision_logits.size() == (batch, head_k, seq_len), (
        f"decision_logits.size: {decision_logits.size()} != (batch, head_k, seq_len): {(batch, head_k, seq_len)}"
    )
    assert attention_mask is None or attention_mask.ndim == 4, (
        f"attention_mask.ndim: {attention_mask.ndim} is not 4"
    )
    assert layer_idx >= 0, f"layer_idx: {layer_idx} is not >= 0"

    decision_logits = decision_logits.reshape(page_batch, seq_len)
    # note that dms has shifted the decision logits by 1
    decision_logits = torch.nn.functional.pad(decision_logits[..., 1:], (0, 1), value=0.0)
    dms_mask_values = torch.nn.functional.logsigmoid(-decision_logits)
    k = k.reshape(page_batch, 1, seq_len, head_dim_k)
    v = v.reshape(page_batch, 1, seq_len, head_dim_k)

    q_flash = q_flash.transpose(1, 2)

    def score_mod(score, b, h, q_idx, k_idx):
        causal = q_idx >= k_idx
        within_sliding_window = q_idx - k_idx <= window_size

        causal = causal.to(score.dtype)
        within_sliding_window = within_sliding_window.to(score.dtype)

        modified_score = within_sliding_window * score + (1 - within_sliding_window) * (
            dms_mask_values[b, k_idx] + score
        )

        return (1 - causal) * MASK_VALUE + causal * modified_score

    attn_output = torch.compile(flex_attention)(
        query=q_flash,
        key=k,
        value=v,
        score_mod=score_mod,
        scale=attn_scaling,
        enable_gqa=True,
        **train_attn_kwargs,
    )

    attn_output = attn_output.reshape(batch, head_k, gqa_factor, seq_len_qf, head_dim_qf).transpose(
        2, 3
    )

    return attn_output


# =============================================================================
# Inference mode (Flash Attention + paged KV cache)
# =============================================================================


def dms_attn_eval_mode(
    new_q_flash: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    decisions: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_idx: int,
    dms_cache: DMSCache,
    attn_scaling: float,
    flash_attn_fn: Callable = flash_attn_with_kvcache,
    prefill_attn_fn: Callable = dms_run_prefill_flex,
    prefill_attn_fn_kwargs: dict = {},
):
    """Perform DMS attention in evaluation mode using flash attention or flex prefill."""
    assert decisions.dtype in (torch.int32, torch.long), (
        f"decisions.dtype: {decisions.dtype} is not int32 or long"
    )
    batch, head_k, new_seq_len, head_dim_k = new_k.size()
    page_batch, seq_len_qf, gqa_factor, head_dim_qf = new_q_flash.size()

    assert page_batch == batch * head_k, (
        f"page_batch: {page_batch} != batch * head_k: {batch * head_k}"
    )
    assert seq_len_qf == new_seq_len, f"seq_len_qf: {seq_len_qf} != new_seq_len: {new_seq_len}"
    assert head_dim_qf == head_dim_k, f"head_dim_qf: {head_dim_qf} != head_dim_k: {head_dim_k}"

    assert new_v.size() == new_k.size(), f"new_v: {new_v.size()} new_k: {new_k.size()}"

    assert decisions.size() == (batch, head_k, new_seq_len), (
        f"decisions.size: {decisions.size()} != (batch, head_k, new_seq_len): {(batch, head_k, new_seq_len)}"
    )
    assert attention_mask is None or attention_mask.ndim == 4, (
        f"attention_mask.ndim: {attention_mask.ndim} is not 4"
    )
    assert layer_idx >= 0, f"layer_idx: {layer_idx} is not >= 0"

    layer_cache: DMSCombinedCacheLayer = dms_cache[layer_idx]

    if layer_cache.is_inference_mode():
        layer_cache.update(
            key_states=new_k,
            value_states=new_v,
            cache_kwargs={
                "eviction_info": decisions,
                "sequence_lengths": None,
                "cumulative_length": 1,
            },
        )

        attn_output = flash_attn_fn(
            new_q_flash,
            layer_cache.paged_cache.get_key_blocks(),
            layer_cache.paged_cache.get_value_blocks(),
            k=None,
            v=None,
            cache_seqlens=layer_cache.paged_cache.get_seq_lengths(),
            causal=True,
            softmax_scale=attn_scaling,
            block_table=layer_cache.paged_cache.get_block_table(),
        )

        attn_output = attn_output.reshape(batch, head_k, seq_len_qf, gqa_factor, head_dim_qf)

        return attn_output
    elif layer_cache.is_prefill_mode():
        if attention_mask is None:
            attention_mask = torch.ones((batch, 1, 1, new_seq_len), dtype=torch.bool).to(
                new_q_flash.device
            )
        attention_mask = attention_mask.to(torch.bool)
        assert attention_mask.ndim == 4, (
            f"attention_mask.ndim: {attention_mask.ndim} is not 4"
        )  # [batch, head or 1, q_seq_len, k_seq_len]

        attention_mask = attention_mask[:, 0, -1, -new_seq_len:]

        attention_output = prefill_attn_fn(
            q_flash=new_q_flash,
            keys=new_k,
            values=new_v,
            decisions=decisions,
            attn_mask=attention_mask,
            cache=layer_cache,
            attn_scaling=attn_scaling,
            flash_attn_fn=flash_attn_fn,
            **prefill_attn_fn_kwargs,
        )
        return attention_output
    else:
        raise ValueError(f"Invalid mode: {layer_cache.current_mode}")
