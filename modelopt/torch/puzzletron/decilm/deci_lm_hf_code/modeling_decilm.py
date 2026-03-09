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

# Copyright 2024 Nvidia Corporation, Google Inc, HuggingFace Inc, EleutherAI. All rights reserved.
#
# This code for Nvidia's model is based on the Llama modeling code by HuggingFace,
# which is in turn based on EleutherAI's GPT-NeoX library and the GPT-NeoX and
# OPT implementations in this library.
# Sliding window code based on Gemma2 by Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: ignore-errors

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from .block_config import AttentionConfig, FFNConfig, MambaConfig, MoEConfig
from .configuration_decilm import DeciLMConfig
from .megatron_lm__mamba_mixer import MambaMixerMegatron
from .transformers_4_44_2__activations import ACT2FN
from .transformers_4_44_2__cache_utils import Cache, StaticCache
from .transformers_4_44_2__modeling_attn_mask_utils import AttentionMaskConverter
from .transformers_4_44_2__modeling_flash_attention_utils_backward_compat import (
    _flash_attention_forward,
)
from .transformers_4_44_2__modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from .transformers_4_44_2__modeling_rope_utils import ROPE_INIT_FUNCTIONS
from .transformers_4_44_2__pytorch_utils import ALL_LAYERNORM_LAYERS
from .transformers_4_51_3__modeling_llama4_attention import Llama4TextAttention, Llama4TextConfig
from .variable_cache import VariableCache
from .vllm_yarn_utils import YaRNScalingRotaryEmbedding

# from transformers.models.llama4.modeling_llama4 import Llama4TextL2Norm
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[DeciLMConfig.model_type] = "DeciLMForCausalLM"
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeciLMConfig"


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or
            a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be
            as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to place the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class Llama4TextL2Norm(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

    def extra_repr(self):
        return f"eps={self.eps}"


class DeciLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeciLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(DeciLMRMSNorm)


class DeciLMRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: DeciLMConfig | None = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`DeciLMRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_impl = "rope" if config is None else config.position_embedding_type
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

    def _set_inv_freq_if_needed(self, device: torch.device) -> None:
        is_missing_inv_freq = not hasattr(self, "inv_freq")
        is_meta_mismatch = not is_missing_inv_freq and (
            str(device) != "meta" and self.inv_freq.is_meta
        )

        if is_missing_inv_freq or is_meta_mismatch:
            with torch.device(device):
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.config, device, **self.rope_kwargs
                )
                self.original_inv_freq = inv_freq
                self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        self._set_inv_freq_if_needed(x.device)

        if self.rope_impl == "rope_llama4":
            return self.llama4_forward(x, position_ids)
        else:
            return self.llama3_forward(x, position_ids)

    def llama3_forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def llama4_forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.polar(
                torch.ones_like(freqs), freqs
            )  # Convert to complex representation

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        freqs_cis = freqs_cis * self.attention_scaling
        return freqs_cis


class DeciMistralYarnRotaryEmbedding(nn.Module):
    def __init__(self, config: DeciLMConfig):
        super().__init__()
        self.config = config
        self.rope_scaling = config.rope_scaling
        self.base = config.rope_theta
        self.rope_impl = config.position_embedding_type
        self.head_size = config.hidden_size // config.num_attention_heads
        self.yarn = YaRNScalingRotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=self.head_size,
            max_position_embeddings=self.rope_scaling["original_max_position_embeddings"],
            base=self.base,
            is_neox_style=True,
            scaling_factor=self.rope_scaling["factor"],
            beta_fast=self.rope_scaling["beta_fast"],
            beta_slow=self.rope_scaling["beta_slow"],
            dtype=torch.float32,
        )
        self.attention_scaling = self.yarn.mscale
        self.scaling_factor = self.rope_scaling["factor"]
        self.rope_impl = "rope" if config is None else config.position_embedding_type
        self.rope_impl = "even_odd"

    def _set_inv_freq_if_needed(self, device: torch.device) -> None:
        is_missing_inv_freq = not hasattr(self, "inv_freq")
        is_meta_mismatch = not is_missing_inv_freq and (
            str(device) != "meta" and self.inv_freq.is_meta
        )

        if is_missing_inv_freq or is_meta_mismatch:
            with torch.device(device):
                inv_freq = self.yarn._compute_inv_freq(self.scaling_factor)
                self.register_buffer("inv_freq", inv_freq, persistent=False)

    def halves_forward(self, x, position_ids):
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )

        self._set_inv_freq_if_needed(x.device)

        # print(f"halves_forward")
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        inv_freq_expanded = inv_freq_expanded.to(x.device)
        # print(f"inv_freq_expanded: {inv_freq_expanded.device}")
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def forward(self, x, position_ids):
        if self.rope_impl == "halves":
            return self.halves_forward(x, position_ids)
        elif self.rope_impl == "even_odd":
            return self.even_odd_forward(x, position_ids)
        else:
            raise ValueError(f"Invalid rope implementation: {self.rope_impl}")

    def even_odd_forward(self, x, position_ids):
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )

        self._set_inv_freq_if_needed(x.device)

        # print(f"even_odd_forward")
        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.polar(
                torch.ones_like(freqs), freqs
            )  # Convert to complex representation

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        freqs_cis = freqs_cis * self.attention_scaling
        return freqs_cis


class DeciLMLinearScalingRotaryEmbedding(DeciLMRotaryEmbedding):
    """DeciLMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`DeciLMLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`DeciLMRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class DeciLMDynamicNTKScalingRotaryEmbedding(DeciLMRotaryEmbedding):
    """DeciLMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`DeciLMDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`DeciLMRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


rope_type_to_class = {
    "default": DeciLMRotaryEmbedding,
    "linear": DeciLMLinearScalingRotaryEmbedding,
    "dynamic": DeciLMDynamicNTKScalingRotaryEmbedding,
    "rope_llama4": DeciLMRotaryEmbedding,
    "rope": DeciLMRotaryEmbedding,
    "mistral_yarn": DeciMistralYarnRotaryEmbedding,
}


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs_cis, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        freqs_cis (`torch.Tensor`): The frequency tensor.
            a tuple of two tensors, cos and sin.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # print(f"applying first half-second half")
    cos, sin = freqs_cis
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def vllm_apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # print(f"freqs_cis: {freqs_cis.shape}, xq_: {xq_.shape}, xk_: {xk_.shape}")
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, None, :, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, None, :, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class DeciLMGatedMLP(nn.Module):
    def __init__(
        self,
        config: DeciLMConfig,
        ffn_config: FFNConfig,
    ):
        super().__init__()
        self.config = config
        self.ffn_config = ffn_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = ffn_config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[getattr(ffn_config, "hidden_act", "silu")]

        if ffn_config.sparsify is not None:
            self.register_full_backward_hook(sparsity_backward_hook)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class DeciLMVanillaMLP(nn.Module):
    def __init__(
        self,
        config: DeciLMConfig,
        ffn_config: FFNConfig,
    ):
        super().__init__()
        self.config = config
        self.ffn_config = ffn_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = ffn_config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[getattr(ffn_config, "hidden_act", "silu")]

        if ffn_config.sparsify is not None:
            self.register_full_backward_hook(sparsity_backward_hook)

        assert self.config.pretraining_tp == 1, (
            "Unsupported pretraining_tp != 1 for DeciLMVanillaMLP"
        )

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DeciLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: DeciLMConfig,
        attention_config: AttentionConfig,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.attention_config = attention_config  # type: AttentionConfig
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = attention_config.n_heads_in_group  # DeciLM-specific code
        self.num_key_value_heads = (
            self.num_heads // self.num_key_value_groups
        )  # DeciLM-specific code
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # llama4 attention specific
        self.llama4_attn_config = attention_config.llama4

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.o_proj_bias
        )

        if self.config.position_embedding_type in ["rope", "rope_llama4", "mistral_yarn"]:
            # TO DO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
            self.rotary_emb = rope_type_to_class[self.config.position_embedding_type](
                config=self.config
            )

        if attention_config.sparsify is not None:
            self.register_full_backward_hook(sparsity_backward_hook)

        self.is_llama4 = self.llama4_attn_config is not None
        if (
            self.is_llama4
            and self.llama4_attn_config.use_qk_norm
            and self.llama4_attn_config.use_rope
        ):
            self.qk_norm = Llama4TextL2Norm(self.config.rms_norm_eps)

        self.use_rope = (
            self.llama4_attn_config.use_rope
            if self.is_llama4
            else self.config.position_embedding_type in ["rope", "mistral_yarn"]
        )
        self.rope_impl = self.rotary_emb.rope_impl
        self.apply_rope_fn = (
            apply_rotary_emb
            if self.rope_impl in ["even_odd", "rope_llama4"]
            else apply_rotary_pos_emb
        )
        # self.apply_rope_fn = apply_rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.use_rope:
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE "
                    "embeddings internally through `position_ids` (2D tensor with the indexes of the "
                    "tokens), to using externally computed `position_embeddings` (Tuple of tensors, "
                    "containing cos and sin). In v4.45 `position_ids` will be removed and "
                    "`position_embeddings` will be mandatory."
                )
                freqs_cis = self.rotary_emb(value_states, position_ids)
            else:
                freqs_cis = position_embeddings

            query_states, key_states = self.apply_rope_fn(query_states, key_states, freqs_cis)

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        if self.is_llama4:
            query_states = self.apply_attention_scaling(input_shape, cache_position, query_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # print(f"cache_position: {cache_position}")
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def apply_attention_scaling(self, input_shape, cache_position, query_states):
        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.llama4_attn_config.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(
                    torch.floor(
                        (cache_position.float() + 1.0) / self.llama4_attn_config.floor_scale
                    )
                    + 1.0
                )
                * self.llama4_attn_config.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.view((*input_shape, 1, 1)).transpose(1, 2)
            query_states = (query_states * attn_scales).to(query_states.dtype)
            return query_states
        return query_states


class DeciLMFlashAttention2(DeciLMAttention):
    """
    DeciLM flash attention module. This module inherits from `DeciLMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is
        # bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is
        # used to handle this difference.
        # Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case
        # q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        self.sliding_window = self.attention_config.prefill_sliding_window

        self.pre_attention_identity_query = nn.Identity()  # for debugging hooks
        self.pre_attention_identity_key = nn.Identity()  # for debugging hooks

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.45
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.config.position_embedding_type in ["rope", "mistral_yarn"]:
            # llama4 doesn't use flash attention
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE "
                    "embeddings internally through `position_ids` (2D tensor with the indexes of the "
                    "tokens), to using externally computed `position_embeddings` (Tuple of tensors, "
                    "containing cos and sin). In v4.45 `position_ids` will be removed and "
                    "`position_embeddings` will be mandatory."
                )
                freqs_cis = self.rotary_emb(value_states, position_ids)
            else:
                freqs_cis = position_embeddings

            query_states, key_states = self.apply_rope_fn(query_states, key_states, freqs_cis)
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, freq_cis)
            # print(f"applying even odd rope")

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout
        # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV
        # cache to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeciLMRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = self.pre_attention_identity_query(query_states)
        key_states = self.pre_attention_identity_key(key_states)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=self.sliding_window,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


DECILM_ATTENTION_CLASSES = {
    "eager": DeciLMAttention,
    "flash_attention_2": DeciLMFlashAttention2,
}


class DeciLMLlama4TextAttention(Llama4TextAttention):
    def __init__(self, config: DeciLMConfig, layer_idx: int, attention_config: AttentionConfig):
        llama4_text_config = Llama4TextConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads // attention_config.n_heads_in_group,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            attn_scale=attention_config.llama4.attn_scale,
            floor_scale=attention_config.llama4.floor_scale,
            attn_temperature_tuning=attention_config.llama4.attn_temperature_tuning,
            attention_dropout=attention_config.llama4.attention_dropout,
            use_qk_norm=attention_config.llama4.use_qk_norm,
            use_rope=attention_config.llama4.use_rope,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            attn_implementation=config.llama4_attn_implementation,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            attention_chunk_size=attention_config.llama4.attention_chunk_size,
        )
        super().__init__(llama4_text_config, layer_idx, use_rope=attention_config.llama4.use_rope)


class DeciLMDecoderLayer(nn.Module):
    # DeciLM-specific code
    def __init__(self, config: DeciLMConfig, layer_idx: int | tuple[int, ...]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.block_config = config.get_block_config(layer_idx)

        self.attention_config = self.block_config.attention
        self.ffn_config = self.block_config.ffn
        self.layer_idx = layer_idx

        if not config._attn_implementation:
            config._attn_implementation = "eager"

        if not self.attention_config.no_op:
            self.input_layernorm = DeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.attention_config.replace_with_linear:
                self.self_attn = DeciLMLinearAttention(config)
            elif self.attention_config.is_mamba:
                self.self_attn = DeciLMMambaMixer(config, self.attention_config.mamba)
            elif not self.attention_config.is_llama4:
                self.self_attn = DECILM_ATTENTION_CLASSES[config._attn_implementation](
                    config=config, attention_config=self.attention_config, layer_idx=layer_idx
                )
            else:
                self.self_attn = DeciLMLlama4TextAttention(config, layer_idx, self.attention_config)

        if not (self.ffn_config.no_op or self.attention_config.is_mamba):
            if getattr(self.ffn_config, "hidden_act", None) is None:
                print(f"WARNING: FFN hidden_act is None for layer {layer_idx}")

            self.post_attention_layernorm = DeciLMRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            if self.ffn_config.replace_with_linear:
                self.mlp = DeciLMLinearMLP(config)
            elif self.ffn_config.is_moe:
                self.mlp = DeciLMMoe(config, self.ffn_config)
            else:
                self.mlp = (
                    DeciLMGatedMLP(config, self.ffn_config)
                    if self.ffn_config.gated
                    else DeciLMVanillaMLP(config, self.ffn_config)
                )

        self.is_sliding = self.attention_config.is_sliding
        self.sliding_window = self.attention_config.prefill_sliding_window
        self.return_only_hidden_states = self.config.block_return_only_hidden_states

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        output_router_logits: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> (
        tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]
        | torch.FloatTensor
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        paramz = list(self.parameters())
        device = paramz[0].device if len(paramz) > 0 else None
        if isinstance(hidden_states, tuple):
            # could happen when sewing kit sends the output of the previous layer
            # to this layer without going through the model forward unpacking code.
            # can be avoided by using config.block_return_only_hidden_states=True
            hidden_states = hidden_states[0]

        hidden_states = hidden_states.to(device)

        if cache_position is not None:
            cache_position = cache_position.to(device)

        if self.attention_config.llama4 is not None:
            # chunk_size = self.attention_config.llama4.attention_chunk_size
            # print(f"pre-llama4_update: {attention_mask=}")
            # causal_mask, chunk_causal_mask = self._llama4_update_causal_mask(
            #     attention_mask, hidden_states, cache_position, past_key_value, output_attentions, use_cache=use_cache,
            # )
            # attention_mask = causal_mask if (chunk_size is None) else chunk_causal_mask
            # if (past_key_value is not None) and isinstance(attention_mask, BlockMask):
            #     print(f"pre-adjust: {attention_mask.shape=}")
            #     print(f"pre-adjust: {hidden_states.shape=}")
            #     print(f"pre-adjust: {past_key_value.get_seq_length()=}")
            #     q_len = hidden_states.shape[1]
            #     kv_len = past_key_value.get_seq_length()
            #     if kv_len == 0:
            #         kv_len = q_len
            #     print(f"pre-adjust: {kv_len=} {q_len=}")
            #     print(f"post-adjust: {attention_mask.shape=}")
            assert self.config.llama4_attn_implementation != "flex_attention", (
                "We have a mask issue with flex attention"
            )

            causal_mask, chunk_causal_mask = self._llama4_update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_value,
                output_attentions,
                use_cache=use_cache,
            )
            is_chunked = self.attention_config.llama4.attention_chunk_size is not None
            attention_mask = (
                chunk_causal_mask if is_chunked and (chunk_causal_mask is not None) else causal_mask
            )

        else:
            attention_mask = self._llama3_update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_value, output_attentions
            )
            if self.attention_config.unshifted_sink and self.attention_config.is_sink:
                attention_mask = self._unshifted_sink_mask(
                    attention_mask,
                    hidden_states,
                    self.attention_config.window_length,
                    self.attention_config.num_sink_tokens,
                )
            else:
                attention_mask = self._gemma2_window_mask(
                    attention_mask, hidden_states, past_key_value
                )

        self_attn_weights = None
        present_key_value = past_key_value
        router_logits = None

        if self.attention_config.no_op:
            pass
        elif self.attention_config.replace_with_linear or self.attention_config.is_mamba:
            if self.attention_config.is_mamba:
                assert past_key_value is None, "DeciLM does not support generation with Mamba yet"
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states)
            hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_out = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states, self_attn_weights = attn_out[:2]
            if len(attn_out) > 2:
                present_key_value = attn_out[2]

            hidden_states = residual + hidden_states

        if not self.ffn_config.no_op:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

            # Handle MoE layers differently as they return router logits
            if self.ffn_config.is_moe:
                hidden_states, router_logits = self.mlp(hidden_states)
            else:
                hidden_states = self.mlp(hidden_states)

            hidden_states = residual + hidden_states

        if self.return_only_hidden_states:
            return hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits and router_logits is not None:
            outputs += (router_logits,)

        return outputs

    def _gemma2_window_mask(
        self,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
        past_key_value: VariableCache | None,
    ) -> torch.Tensor | None:
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.config._attn_implementation == "flash_attention_2":
                if past_key_value is not None:  # when decoding
                    attention_mask = attention_mask[:, -self.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]
        return attention_mask

    def _unshifted_sink_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        window_length: int,
        num_sink_tokens: int | None,
    ) -> torch.Tensor:
        assert self.config._attn_implementation == "eager", (
            "Unshifted sink is only supported in 'eager' mode."
        )
        assert attention_mask is not None, "The attention mask seems to not be prepared"

        attention_mask = attention_mask.clone()
        min_dtype = torch.finfo(hidden_states.dtype).min

        if window_length == 0:
            attention_mask = torch.full_like(attention_mask, fill_value=min_dtype)
        else:
            query_length = attention_mask.shape[-2]
            is_decode = query_length == 1
            if is_decode:
                attention_mask[:, :, :, :-window_length] = min_dtype
            else:
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-window_length
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)

        attention_mask[:, :, :, :num_sink_tokens] = 0
        return attention_mask

    def _llama3_update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is
        # 2D and of dynamic length even when the static KV cache is used. This is an issue for
        # torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic
        # shapes. (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow.
        # A workaround is `@torch.compiler.disable`, but this prevents using `fullgraph=True`.
        # See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        assert not isinstance(past_key_values, StaticCache), "DeciLM does not support StaticCache"
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if (
                AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                )
                and not self.is_sliding
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @torch.compiler.disable(recursive=False)  # the operations in this method are not compilable
    def _llama4_update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        output_attentions: bool = False,
        chunked_attention_mask=None,
        use_cache=True,
    ):
        attn_implementation = self.config.llama4_attn_implementation

        if attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return (
                    attention_mask,
                    attention_mask,
                )  # flash does not support chunked attn TODO support flash
            return None, None

        if attn_implementation not in ["sdpa", "flex_attention", "eager"]:
            return None, None

        sequence_length = input_tensor.shape[1]
        cache_position = cache_position.to(self.device)
        attention_chunk_size = self.attention_config.llama4.attention_chunk_size
        if attention_chunk_size is None:
            # let the function build some chunked mask, we won't use it since it's not a chunked
            # attention layer. We still need to know the chunk size for this if statement that
            # comes later on: if attn_implementation == "sdpa" and chunked_attention_mask is not None
            # otherwise the mask dtype is wrong for sdpa :bufo-wat:
            attention_chunk_size = self.config.get_min_attention_chunk_size()
            if attention_chunk_size is None:
                logger.warning_once(
                    "Could not infer attention_chunk_size since the model (or the model shard) "
                    "has no chunked attention, using 8192 as default for mask construction"
                )
                attention_chunk_size = 8192

        first_cache_position = cache_position[0]

        if past_key_values is not None:
            full_cache_length = past_key_values.get_max_cache_shape() or sequence_length
        else:
            full_cache_length = (
                attention_mask.shape[-1] if attention_mask is not None else sequence_length
            )

        cond1 = first_cache_position >= attention_chunk_size
        cond2 = (first_cache_position < attention_chunk_size) & (
            first_cache_position + sequence_length > attention_chunk_size
        )
        key_length = (
            torch.where(
                cond1,
                attention_chunk_size + sequence_length - 1,
                torch.where(cond2, first_cache_position + sequence_length, attention_chunk_size),
            )
            if use_cache
            else full_cache_length
        )

        if attn_implementation == "flex_attention":
            raise NotImplementedError("DeciLM Llama4 does not support flex attention")
            # if isinstance(attention_mask, torch.Tensor):
            #     offsets = (first_cache_position, max(first_cache_position - attention_chunk_size + 1, 0))
            #     chunked_attention_mask = make_flex_block_causal_mask(
            #         attention_mask, attention_chunk_size, sequence_length, key_length, offsets=offsets
            #     )
            #     attention_mask = make_flex_block_causal_mask(
            #         attention_mask,
            #         query_length=sequence_length,
            #         key_length=full_cache_length,
            #         offsets=(first_cache_position, 0),
            #     )
            #     return attention_mask, chunked_attention_mask
            # if isinstance(attention_mask, BlockMask):
            #     return attention_mask, chunked_attention_mask

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        dtype, device = input_tensor.dtype, input_tensor.device
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=max(full_cache_length, attention_chunk_size),
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            min_dtype=torch.finfo(dtype).min,
        )
        if full_cache_length > attention_chunk_size:
            start_idx = max(first_cache_position - attention_chunk_size + 1, 0)
            end_idx = start_idx + key_length
            chunked_attention_mask = self.create_chunked_attention_mask(
                attention_chunk_size,
                start=start_idx,  # same offset as with flex
                end=end_idx,
                device=device,
            )

            ### Deci: we added this code to patch a bug in transformers
            if attention_mask is None:
                if past_key_values is not None:
                    raise NotImplementedError("We only support attention_mask=None is prefill")
                attention_mask = torch.ones(
                    input_tensor.shape[0], input_tensor.shape[1], device=device, dtype=torch.long
                )

            local_attention_mask = attention_mask[:, start_idx:end_idx]  # offset here as well
            # It may be smaller than attention_chunk_size -> pad it
            requires_padding = local_attention_mask.shape[-1] < attention_chunk_size
            if requires_padding:
                local_attention_mask = nn.functional.pad(
                    local_attention_mask, (0, attention_chunk_size - local_attention_mask.shape[-1])
                )
            # Depending on the padding, take the query tokens from the end or the cache_position
            if not requires_padding:
                chunked_attention_mask = chunked_attention_mask[None, None, -sequence_length:, :]
            else:
                chunked_attention_mask = chunked_attention_mask[None, None, cache_position, :]

            chunked_attention_mask = chunked_attention_mask.expand(
                input_tensor.shape[0], -1, -1, -1
            )
            chunked_attention_mask = chunked_attention_mask * local_attention_mask[:, None, None, :]
            if attn_implementation == "eager":
                min_dtype = torch.finfo(dtype).min
                chunked_attention_mask = torch.where(
                    chunked_attention_mask == 0, min_dtype, 0.0
                ).to(dtype)

        # print(f"{output_attentions=}")

        if (
            attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and attention_mask.ndim == 4
            and not output_attentions  # Only unmask for 4d masks
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if attn_implementation == "sdpa" and chunked_attention_mask is not None:
            chunked_attention_mask = chunked_attention_mask.bool()
            causal_mask = causal_mask.bool()
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=first_cache_position,
                is_training=self.training,
            ):
                causal_mask = None
        return causal_mask, chunked_attention_mask

    def create_chunked_attention_mask(
        self, attention_chunk_size: int, start: int, end: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate the following:

        'What'      :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚    |
        '▁is'       :  1 ■ ■ ⬚ ⬚ ⬚ ⬚     |
        '▁ch'       :  2 ■ ■ ■ ⬚ ⬚ ⬚     |
        'unked'     :  3 ⬚ ⬚ ⬚ ■ ⬚ ⬚    |
        '▁attention':  4 ⬚ ⬚ ⬚ ■ ■ ⬚    |
        '?'         :  5 ⬚ ⬚ ⬚ ■ ■ ■     |

        If the chunk size is 3.
        This can just be appplied over the already created attention mask
        """
        arange_vector = torch.arange(start, end, device=device)
        block_pos = torch.abs(
            arange_vector.unsqueeze(0) // attention_chunk_size
            - arange_vector.unsqueeze(1) // attention_chunk_size
        )
        token_pos = arange_vector.unsqueeze(0) - arange_vector.unsqueeze(1)
        mask = (block_pos == 0) & (token_pos <= 0)
        return mask.to(device)


class DeciLMMultiDecoderLayer(nn.Module):
    def __init__(self, config: DeciLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        block_config = config.block_configs[layer_idx]
        assert block_config.parallel_blocks is not None
        num_parallel_blocks = len(block_config.parallel_blocks)
        self.parallel_blocks = nn.ModuleList(
            [
                DeciLMDecoderLayer(config, (layer_idx, internal_block_idx))
                for internal_block_idx in range(num_parallel_blocks)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        block_outputs = [block(hidden_states, *args, **kwargs) for block in self.parallel_blocks]
        output_hidden_states = [
            out[0].to(hidden_states.device)
            if isinstance(out, tuple)
            else out.to(hidden_states.device)
            for out in block_outputs
        ]
        output_hidden_states = torch.stack(output_hidden_states, dim=0).sum(dim=0)
        output_hidden_states = (
            output_hidden_states - (len(self.parallel_blocks) - 1) * hidden_states
        )

        if self.config.block_return_only_hidden_states:
            return output_hidden_states

        other_outputs = block_outputs[0][1:]
        outputs = (output_hidden_states, *other_outputs)
        return outputs


DECILM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeciLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare DeciLM Model outputting raw hidden-states without any specific head on top.",
    DECILM_START_DOCSTRING,
)
class DeciLMPreTrainedModel(PreTrainedModel):
    config_class = DeciLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeciLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True  # all the _supports_... flags refer to the Llama3 layers
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _prepare_generation_config(
        self,
        generation_config: GenerationConfig | None,
        *args,
        **kwargs,
    ) -> tuple[GenerationConfig, dict]:
        try:
            from transformers import cache_utils
            from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING

            need_setup_cache_classes_mapping = NEED_SETUP_CACHE_CLASSES_MAPPING
        except Exception:
            # older releases exposed it via generation.utils
            need_setup_cache_classes_mapping = {}

        # DeciLM-specific code
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, *args, **kwargs
        )
        # New transformers version, can reach only through cache_utils
        if need_setup_cache_classes_mapping == {}:
            cache_utils._CACHE_IMPLEMENTATION_MAPPING["variable"] = VariableCache
        else:
            need_setup_cache_classes_mapping["variable"] = VariableCache

        generation_config.cache_implementation = "variable"
        return generation_config, model_kwargs


DECILM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`VariableCache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            If passed to the forward function, past_key_values must be a VariableCache object (see imports).
            For generation purposes, this is already handled inside model.generate().

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare DeciLM Model outputting raw hidden-states without any specific head on top.",
    DECILM_START_DOCSTRING,
)
class DeciLMModel(DeciLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeciLMDecoderLayer`]

    Args:
        config: DeciLMConfig
    """

    def __init__(self, config: DeciLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                (
                    DeciLMDecoderLayer(config, layer_idx)
                    if (config.block_configs[layer_idx].parallel_blocks is None)
                    else DeciLMMultiDecoderLayer(config, layer_idx)
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.config.position_embedding_type in ["rope", "rope_llama4", "mistral_yarn"]:
            self.rotary_emb = rope_type_to_class[self.config.position_embedding_type](config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_final_layer_norm(self):
        return self.norm

    def set_final_layer_norm(self, value):
        self.norm = value

    @add_start_docstrings_to_model_forward(DECILM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        is_legacy_cache_format = (past_key_values is not None) and type(
            past_key_values
        ).__name__ != "VariableCache"
        # We use the __name__ instead of isinstance to support weird use cases
        # (init cache from a checkpoint dir and use it with local code)
        if is_legacy_cache_format:
            raise NotImplementedError(
                "DeciLMModel does not support legacy cache format, please use a newer "
                "transformers version or use VariableCache explicitly (see import in this file)."
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            # use default device
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = None
        if hasattr(self, "rotary_emb"):
            # rotary emb is created all devices, so we need to move position_ids to the correct device
            some_param = next(self.parameters())
            position_ids = position_ids.to(some_param.device)
            cache_position = cache_position.to(some_param.device)
            faux_hidden_states = position_ids.to(some_param.dtype)
            position_embeddings = self.rotary_emb(faux_hidden_states, position_ids)
            # print(f'START {position_embeddings.device=}') # HF hook will change the device
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            if self.config.block_return_only_hidden_states:
                hidden_states = layer_outputs
                next_decoder_cache = past_key_values

            else:
                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                # Extract router logits if they exist
                if output_router_logits:
                    router_logits_index = -1  # Router logits are always the last element
                    if len(layer_outputs) > (2 if output_attentions else 1) + (
                        1 if use_cache else 0
                    ):
                        all_router_logits += (layer_outputs[router_logits_index],)

        # Final layer norm
        hidden_states = hidden_states.to(next(self.parameters()).device)
        hidden_states = self.norm(hidden_states)

        # Add the last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Set the next cache
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            outputs = (hidden_states, next_cache, all_hidden_states, all_self_attns)
            if output_router_logits:
                outputs += (all_router_logits,)
            return outputs

        # Handle different return types based on whether router logits are requested
        if output_router_logits and all_router_logits:
            return MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                router_logits=all_router_logits,
            )
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


@add_start_docstrings(
    """
    The DeciLM Model transformer with a sequence classification head on top (linear layer).

    [`DeciLMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    DECILM_START_DOCSTRING,
)
class DeciLMForSequenceClassification(DeciLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeciLMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(DECILM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits, *transformer_outputs[1:])
            return (loss, *output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The DeciLM Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DECILM_START_DOCSTRING,
)
class DeciLMForQuestionAnswering(DeciLMPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->DeciLM
    def __init__(self, config):
        super().__init__(config)
        self.transformer = DeciLMModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(DECILM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        start_positions: torch.LongTensor | None = None,
        end_positions: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits, *outputs[2:])
            return (total_loss, *output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The DeciLM Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    DECILM_START_DOCSTRING,
)
class DeciLMForTokenClassification(DeciLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeciLMModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(DECILM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, *outputs[2:])
            return (loss, *output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


########################################################################
# DeciLM-specific code
########################################################################


def _find_multiple(n: int, k: int) -> int:
    # DeciLM-specific code
    if n % k == 0:
        return n
    return n + k - (n % k)


class DeciLMMoe(nn.Module):
    """
    Implementation of Mixture of Experts module for DeciLM.
    Equivalent to Llama4 MoE but implemented more frugally.
    """

    def __init__(self, config: DeciLMConfig, ffn_config: FFNConfig):
        super().__init__()
        self.config = config
        self.ffn_config = ffn_config

        # MoE parameters
        assert ffn_config.moe is not None, "MoE configuration must be provided to use DeciLMMoe"
        self.moe_config: MoEConfig = ffn_config.moe
        self.hidden_dim = config.hidden_size
        self.num_experts_per_tok = self.moe_config.num_experts_per_tok
        self.num_local_experts = self.moe_config.num_local_experts
        self.expert_intermediate_dim = self.moe_config.expert_intermediate_dim
        self.shared_expert_intermediate_dim = self.moe_config.shared_expert_intermediate_dim

        # Initialize experts and router
        routed_expert_ffn_config = FFNConfig(
            intermediate_size=self.expert_intermediate_dim,
        )

        self.experts = nn.ModuleList(
            [
                DeciLMGatedMLP(config, routed_expert_ffn_config)
                for _ in range(self.num_local_experts)
            ]
        )

        self.router = nn.Linear(config.hidden_size, self.num_local_experts, bias=False)

        # Initialize shared expert as a standard MLP
        shared_expert_ffn_config = FFNConfig(
            intermediate_size=self.moe_config.shared_expert_intermediate_dim
        )
        self.shared_expert = DeciLMGatedMLP(config, shared_expert_ffn_config)

        if ffn_config.sparsify is not None:
            self.register_full_backward_hook(sparsity_backward_hook)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch, seq_len, hidden_dim)
                - torch.Tensor: Router scores for loss computation
        """
        router_logits = self.router(hidden_states)

        routed_out = self.forward_routed_experts(hidden_states, router_logits)

        shared_out = self.shared_expert(hidden_states)

        moe_out = routed_out + shared_out

        return moe_out, router_logits

    def forward_routed_experts(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        For each expert:
        1. Build the input to the expert based on the router mask
        2. Run the expert
        3. Add the result of the expert into the total MoE result using +=
        """
        router_top_values, router_indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        router_scores = torch.sigmoid(router_top_values.float()).to(hidden_states.dtype)

        routed_out = torch.zeros_like(hidden_states)
        for i_expert in range(self.num_local_experts):
            expert_mask = router_indices == i_expert
            if expert_mask.any():
                is_token_routed_to_this_expert = expert_mask.any(dim=-1)
                relevant_hidden_states = hidden_states[is_token_routed_to_this_expert, :]
                relevant_scores = router_scores[expert_mask]
                expert_in = relevant_hidden_states * relevant_scores.unsqueeze(-1)

                expert_out = self.experts[i_expert](expert_in).to(hidden_states.device)

                routed_out[is_token_routed_to_this_expert, :] += expert_out

        return routed_out

    def extra_repr(self) -> str:
        return (
            f"(MoE): num_local_experts={self.num_local_experts}, "
            f"expert_intermediate_dim={self.expert_intermediate_dim},"
        )


class DeciLMLinearMLP(nn.Module):
    # DeciLM-specific code
    def __init__(
        self,
        config: DeciLMConfig,
    ):
        super().__init__()
        self.linear_mlp = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_mlp.forward(x)


class DeciLMLinearAttention(nn.Module):
    # DeciLM-specific code
    def __init__(
        self,
        config: DeciLMConfig,
    ):
        super().__init__()
        self.linear_attn = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_attn.forward(x)


def sparsity_backward_hook(*args, **kwargs):
    raise NotImplementedError(
        "No support for sparsity when training HF DeciLM (inference is ok though)"
    )


class DeciLMMambaMixer(nn.Module):
    def __init__(
        self,
        config: DeciLMConfig,
        mamba_config: MambaConfig,
    ):
        super().__init__()
        self.mamba_mixer = MambaMixerMegatron(
            d_model=config.hidden_size,
            d_state=mamba_config.state_dim,
            nheads=mamba_config.num_heads,
            headdim=mamba_config.head_dim,
            ngroups=mamba_config.num_groups,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x.permute([1, 0, 2])  # MambaMixerMegatron expects [Sequence, Batch, Embedding]
        out = self.mamba_mixer(x)
        out = out.permute([1, 0, 2])  # go back to [Batch, Sequence, Embedding]
        return out


class LMHead(nn.Linear):
    """
    Special class to allow FSDP wrapping without affecting other Linear layers in the model.
    """


class DeciLMForCausalLM(DeciLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeciLMConfig):
        super().__init__(config)
        self.model = DeciLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_router_aux_loss(self, router_logits):
        """
        Computes the auxiliary loss for router logits.
        This encourages load balancing across experts.

        Args:
            router_logits: List of router logits tensors from each MoE layer
                Each tensor has shape [batch_size, sequence_length, num_experts]

        Returns:
            Auxiliary loss tensor
        """
        aux_loss = torch.tensor(0.0, device=router_logits[0].device)

        for layer_idx, layer_router_logits in enumerate(router_logits):
            router_probs = torch.softmax(layer_router_logits, dim=-1)

            # Mean routing probability across batch and sequence dimensions
            mean_prob = router_probs.mean(dim=[0, 1])

            # Compute auxiliary loss: combination of load balancing and importance loss
            # Load balancing loss: variance of expert usage probabilities (should be uniform)
            num_experts = mean_prob.size(0)
            ideal_prob = 1.0 / num_experts
            balance_loss = torch.sum((mean_prob - ideal_prob) ** 2)

            # Add this layer's auxiliary loss to the total
            aux_loss = aux_loss + balance_loss

        # Average over all layers
        if len(router_logits) > 0:
            aux_loss = aux_loss / len(router_logits)

        return aux_loss

    @add_start_docstrings_to_model_forward(DECILM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Return:
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # Extract model outputs based on return type
        if isinstance(outputs, MoeModelOutputWithPast):
            hidden_states = outputs.last_hidden_state
            router_logits = outputs.router_logits
        elif return_dict:
            hidden_states = outputs.last_hidden_state
            router_logits = None  # No router logits in this case
        else:
            hidden_states = outputs[0]
            router_logits = outputs[4] if output_router_logits and len(outputs) > 4 else None

        # Generate logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # Calculate router aux loss if router logits are present
            if router_logits is not None and self.config.router_aux_loss_coef > 0:
                aux_loss = self.compute_router_aux_loss(router_logits)
                loss = loss + aux_loss * self.config.router_aux_loss_coef

        # Handle non-dict return
        if not return_dict:
            output = (logits,)
            if isinstance(outputs, tuple):
                output += outputs[1:]  # Add all other outputs
            return (loss, *output) if loss is not None else output

        # Different output types for MoE vs regular model
        if router_logits is not None:
            return MoeCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values if return_dict else outputs[1],
                hidden_states=outputs.hidden_states
                if return_dict
                else outputs[2]
                if output_hidden_states
                else None,
                attentions=outputs.attentions
                if return_dict
                else outputs[3]
                if output_attentions
                else None,
                router_logits=router_logits,
            )
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values if return_dict else outputs[1],
                hidden_states=outputs.hidden_states
                if return_dict
                else outputs[2]
                if output_hidden_states
                else None,
                attentions=outputs.attentions
                if return_dict
                else outputs[3]
                if output_attentions
                else None,
            )
