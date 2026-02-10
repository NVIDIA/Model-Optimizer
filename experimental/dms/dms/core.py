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

"""DMS core operations: attention I/O, gating, output types, chunked prefill, and training state."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from dms.cache import Mode
from dms.logging import get_logger

logger = get_logger("DMSCore")


# =============================================================================
# Setup utilities
# =============================================================================


def setup_compile_limit_for_dms(compile_limit: int = 72):
    """Set the torch.compile cache size limit for DMS layer compilation."""
    # we want to compile the prepare_attention_input and post_process_attention_output functions
    # for each layer
    if torch._dynamo.config.cache_size_limit != compile_limit:
        logger.info(f"Setting up compile limit for DMS to {compile_limit}")
        torch._dynamo.config.cache_size_limit = compile_limit


# =============================================================================
# Training state
# =============================================================================


@dataclass
class DMSTrainingStateAux:
    """Auxiliary state information for DMS training.

    Attributes:
        dms_teacher_mode: Whether the model is in teacher mode.
        target_frac_to_close: Target fraction of DMS key-value pairs to evict.
        current_step: Current training step.
        grad_acc_step: Gradient accumulation step.
        process_index: Index of the current process in distributed training.
        noise: Tensor of noise values per layer (shape: [num_layers, ...]).
        right_padding_size: Right padding size per batch item (shape: [batch_size]).
        kv_cache_shape: Shape tuple for KV cache (num_layers, batch_size, num_kv_heads, seq_length).
    """

    dms_teacher_mode: bool
    target_frac_to_close: float | None
    current_step: int
    grad_acc_step: int
    process_index: int
    noise: torch.Tensor | None
    right_padding_size: int
    kv_cache_shape: tuple[int, int, int, int]  # num_layers, batch_size, num_kv_heads, seq_length


# =============================================================================
# Output dataclasses
# =============================================================================


@dataclass
class DMSBaseModelOutputWithPastAndCR(BaseModelOutputWithPast):
    """DMS base model output with compression ratio.

    Args:
        cr: (`float`, *optional*, returned when DMS cache is used):
            Compression ratio, that is size of cache without compression
            divided by size of cache with compression
        dms_frac_closed: (`torch.Tensor`, *optional*, returned when DMS cache is used):
            Per head average number of tokens (soft) evicted by DMS, used for DMS loss computation.
    """

    cr: float | None = None
    dms_frac_closed: torch.Tensor | None = None


@dataclass
class DMSCausalLMOutputWithPastAndCR(CausalLMOutputWithPast):
    """DMS causal LM output with compression ratio.

    Args:
        cr: (`float`, *optional*, returned when DMS cache is used):
            Compression ratio, that is size of cache without compression
            divided by size of cache with compression
        dms_frac_closed: (`torch.Tensor`, *optional*, returned when DMS cache is used):
            Per head average number of tokens (soft) evicted by DMS, used for DMS loss computation.
    """

    cr: float | None = None
    dms_loss: torch.Tensor | None = None
    dms_frac_closed: torch.Tensor | None = None


# =============================================================================
# Attention input preparation and output processing
# =============================================================================


@torch.compile()
def prepare_attention_input(
    pre_attn_norm_hidden_states: torch.Tensor,
    post_attn_norm_hidden_states: torch.Tensor,
    q_proj_fn: torch.nn.Linear,
    k_proj_fn: torch.nn.Linear,
    v_proj_fn: torch.nn.Linear,
    q_norm_fn: torch.nn.Module,
    k_norm_fn: torch.nn.Module,
    head_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    dms_proj_alpha_norm_fn: torch.nn.Module | None,
    dms_proj_alpha_fn: torch.nn.Linear | None,
    dms_alpha_per: str,
    dms_decision_scale: float,
    dms_initial_decision_offset: float,
    dms_training: bool,
    dms_disable_eviction: bool,
    dms_tau: float,
    apply_rotary_pos_emb_fn: Callable,
    dms_teacher_mode: bool,
    dms_noise: torch.Tensor | None = None,
):
    """Prepare query, key, value, and DMS decision tensors for attention."""
    batch, seq_len, _hidden_dim = pre_attn_norm_hidden_states.size()

    query_states = q_norm_fn(
        q_proj_fn(post_attn_norm_hidden_states).view(batch, seq_len, -1, head_dim).transpose(1, 2)
    )

    key_states = k_norm_fn(
        k_proj_fn(post_attn_norm_hidden_states).view(batch, seq_len, -1, head_dim).transpose(1, 2)
    )
    value_states = (
        v_proj_fn(post_attn_norm_hidden_states).view(batch, seq_len, -1, head_dim).transpose(1, 2)
    )

    _, num_q_heads, _, _ = query_states.size()
    _, num_kv_heads, _, _ = key_states.size()
    gqa_factor = num_q_heads // num_kv_heads

    if dms_proj_alpha_fn is None:
        assert dms_proj_alpha_norm_fn is None, (
            "dms_proj_alpha_norm_fn is not None when dms_proj_alpha_fn is None"
        )
        decision_logits = (
            query_states[:, ::gqa_factor, :, -1].clone() * dms_decision_scale
            - dms_initial_decision_offset
        )
        assert decision_logits.shape == (batch, num_kv_heads, seq_len), (
            f"decision_logits.shape: {decision_logits.shape} != {(batch, num_kv_heads, seq_len)}"
        )

        query_states[:, ::gqa_factor, :, -1] = 0
        query_states, key_states = apply_rotary_pos_emb_fn(query_states, key_states, cos, sin)
        query_states[:, ::gqa_factor, :, -1] = 0
    else:
        assert dms_proj_alpha_norm_fn is not None, (
            "dms_proj_alpha_norm_fn is None when dms_proj_alpha_fn is not None"
        )
        decision_logits = (
            dms_proj_alpha_fn(dms_proj_alpha_norm_fn(pre_attn_norm_hidden_states))
            * dms_decision_scale
            - dms_initial_decision_offset
        )
        assert decision_logits.shape == (batch, seq_len, num_kv_heads), (
            f"decision_logits.shape: {decision_logits.shape} != {(batch, seq_len, num_kv_heads)}"
        )
        decision_logits = decision_logits.transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb_fn(query_states, key_states, cos, sin)

    if dms_training and not dms_teacher_mode:
        assert dms_noise is not None, "dms_noise is None when dms_training and not dms_teacher_mode"
        dms_noise = dms_noise.to(decision_logits.device)
        _probs, decisions, decision_logits = get_gating_with_noise(
            gating_weights=decision_logits, noise=dms_noise, tau=dms_tau
        )
    else:
        decisions = (decision_logits > 0).to(decision_logits.dtype)
    assert decisions.shape == (batch, num_kv_heads, seq_len), (
        f"decisions.shape: {decisions.shape} != {(batch, num_kv_heads, seq_len)}"
    )

    if dms_alpha_per == "head":
        decisions = decisions.broadcast_to(batch, num_kv_heads, seq_len)
        decision_logits = decision_logits.broadcast_to(batch, num_kv_heads, seq_len)
    elif dms_alpha_per == "layer":
        decisions = decisions[:, [0], :].broadcast_to(batch, num_kv_heads, seq_len)
        decision_logits = decision_logits[:, [0], :].broadcast_to(batch, num_kv_heads, seq_len)
    else:
        raise ValueError(f"Invalid dms_alpha_per: {dms_alpha_per}")

    flash_attn_query_states = query_states.reshape(
        batch * num_kv_heads, gqa_factor, seq_len, head_dim
    ).transpose(1, 2)

    if dms_disable_eviction or dms_teacher_mode:
        decisions = torch.zeros_like(decisions)
        decision_logits = torch.full_like(decision_logits, fill_value=-1000.0)

    return flash_attn_query_states, key_states, value_states, decisions, decision_logits


@torch.compile()
def post_process_attention_output(
    attn_output: torch.Tensor,
    o_proj: torch.nn.Linear,
):
    """Reshape attention output and apply output projection."""
    batch, heads_kv, seq_len_q, gqa_factor, head_dim = attn_output.size()

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(
        batch, seq_len_q, heads_kv * gqa_factor * head_dim
    ).contiguous()

    attn_output = o_proj(attn_output)

    return attn_output


def get_gating_with_noise(gating_weights: torch.Tensor, noise: torch.Tensor, tau: float):
    """Apply Gumbel noise to gating weights and return discretized decisions."""
    assert gating_weights.shape == noise.shape, (
        f"gating_weights.shape: {gating_weights.shape} != noise.shape {noise.shape}"
    )

    logits = (gating_weights + noise) / tau
    probs = torch.nn.functional.sigmoid(logits)

    discretized = (probs > 0.5).to(probs.dtype) - probs.detach() + probs

    return probs, discretized, logits


# =============================================================================
# Chunked prefill
# =============================================================================


def run_decoder_layers(
    decoder_layers: list[torch.nn.Module],
    hidden_states: torch.Tensor,
    **kwargs: Any,
):
    """Pass hidden states through decoder layers.

    Returns the final hidden states along with
    the per head average number of tokens (soft) evicted by DMS.
    """
    acc_dms_frac_closed = 0
    num_layers = 0
    for dl in decoder_layers:
        hidden_states, dms_frac_closed = dl(
            hidden_states,
            **kwargs,
        )
        if dms_frac_closed is not None:
            num_layers += 1
            acc_dms_frac_closed += dms_frac_closed

    # NOTE: assumption that each attention enabled layer has the same number of attention heads
    dms_frac_closed = acc_dms_frac_closed / num_layers if num_layers > 0 else None
    return hidden_states, dms_frac_closed


def dms_perform_chunked_prefill(
    decoder_layers: list[torch.nn.Module],
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values: Any,
    use_cache: bool,
    cache_position: torch.Tensor | None,
    position_embeddings: torch.Tensor | None,
    dms_manual_inference_mode: bool,
    dms_chunked_prefill: int | None,
    **kwargs: Any,
):
    """Used to chunk the input for transformer decoder layers.

    At this point in transformers most elements (masks, embeddings)
    are already constructed.
    """
    batch, seq_len, _hidden_dim = hidden_states.size()

    assert attention_mask is None or attention_mask.ndim == 4, (
        f"attention_mask.ndim: {attention_mask.ndim}"
    )

    if not dms_manual_inference_mode:
        if (
            seq_len == 1
            and past_key_values is not None
            and len(past_key_values) > 0
            and past_key_values[0].get_seq_length() > 0
        ):
            if past_key_values.current_mode != Mode.INFERENCE:
                past_key_values.inference_mode()
                logger.debug(
                    f"Setting inference mode for past_key_values with cr: {past_key_values.get_cr()}"
                )

        elif past_key_values is not None:
            logger.debug(
                f"Setting prefill mode for past_key_values with seq_length {past_key_values[0].get_seq_length()}"
            )
            past_key_values.prefill_mode()

    if seq_len > 1 and dms_chunked_prefill is not None:
        num_chunks = (seq_len + dms_chunked_prefill - 1) // dms_chunked_prefill

        hidden_states_chunks = []

        for chid in tqdm(
            range(num_chunks),
            desc=f"Chunked prefill for batch_size:{batch} seq_len:{seq_len} chunk_size:{dms_chunked_prefill}",
        ):
            start_pos = chid * dms_chunked_prefill
            end_pos = min(start_pos + dms_chunked_prefill, seq_len)
            hidden_states_chunk = hidden_states[:, start_pos:end_pos, :]
            unpadded_chunk_len = hidden_states_chunk.shape[1]

            if attention_mask is not None:
                # take attention mask from the last query
                assert attention_mask.shape[-1] == seq_len, (
                    f"attention_mask.shape[-1]: {attention_mask.shape[-1]} != {seq_len}"
                )
                attention_mask_chunk = attention_mask[:, :, [-1], start_pos:end_pos]
            else:
                attention_mask_chunk = None

            if position_ids is not None:
                assert position_ids.shape[-1] == seq_len, (
                    f"position_ids.shape[-1]: {position_ids.shape[-1]} != {seq_len}"
                )
                position_ids_chunk = position_ids[..., start_pos:end_pos]
            else:
                position_ids_chunk = None

            if cache_position is not None:
                assert cache_position.shape[-1] == seq_len, (
                    f"cache_position.shape[-1]: {cache_position.shape[-1]} != {seq_len}"
                )
                cache_position_chunk = cache_position[..., start_pos:end_pos]
            else:
                cache_position_chunk = None

            if position_embeddings is not None:
                assert isinstance(position_embeddings, tuple)
                for e in position_embeddings:
                    assert e.shape[1] == seq_len, f"e.shape[1]: {e.shape[1]} != {seq_len}"
                position_embeddings_chunk = tuple(
                    e[:, start_pos:end_pos, :] for e in position_embeddings
                )
            else:
                position_embeddings_chunk = None

            hidden_states_chunk, _dms_loss = run_decoder_layers(
                decoder_layers=decoder_layers,
                hidden_states=hidden_states_chunk,
                attention_mask=attention_mask_chunk,
                position_ids=position_ids_chunk,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position_chunk,
                position_embeddings=position_embeddings_chunk,
                **kwargs,
            )
            hidden_states_chunks.append(hidden_states_chunk[:, :unpadded_chunk_len, :])
        hidden_states_chunks = torch.cat(hidden_states_chunks, dim=1)

        return hidden_states_chunks, None
    else:
        return run_decoder_layers(
            decoder_layers=decoder_layers,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
