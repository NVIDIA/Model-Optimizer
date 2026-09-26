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

"""Analytical parameter count and memory footprint utilities for MCore GPT and Mamba/Hybrid models.

These are fast, model-free alternatives to forward-pass-based counters.

Layer conventions (validated against ``TELayerNormColumnParallelLinear`` and ``TEColumnParallelLinear`` MCore specs):

- Dense attention / MLP layers: ``input_layernorm`` / ``pre_mlp_layernorm`` are fused into
  ``linear_qkv`` / ``linear_fc1`` via ``TELayerNormColumnParallelLinear`` (their weight—and bias
  for LayerNorm—count as part of that linear module's parameters).
- MoE layers: ``pre_mlp_layernorm`` is a *separate* ``TENorm`` (not fused); routed expert
  ``linear_fc1`` uses plain ``TEColumnParallelLinear`` (no fused LN). Shared experts never have
  bias (``assert add_bias_linear == False`` in ``SharedExpertMLP``).
- Mamba layers: ``in_proj`` uses ``TELayerNormColumnParallelLinear`` (fused LN). The internal
  ``norm`` on ``d_inner`` is always RMSNorm regardless of the global ``normalization`` setting.
- GDN (``G``) layers are not currently supported and raise an error.

Hybrid pattern characters (from ``megatron.core.ssm.mamba_hybrid_layer_allocation.Symbols``):
  ``M`` = Mamba, ``*`` = Attention-only TransformerLayer, ``-`` = MLP-only TransformerLayer,
  ``E`` = MoE-only TransformerLayer, ``G`` = GDN (unsupported), ``|`` = PP boundary (ignored),
  ``/`` = MTP separator (everything from ``/`` onward is MTP and ignored).
"""

import io
import sys
from typing import Any

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.parallel_state import (
    get_expert_tensor_and_model_parallel_group,
    get_expert_tensor_parallel_rank,
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils import num2hrb, print_rank_0

__all__ = [
    "mcore_memory_footprint_mb",
    "mcore_param_count",
    "mcore_param_count_live",
    "parse_main_layer_chars",
    "print_mcore_model_stats",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HYBRID_MAMBA = "M"
_HYBRID_ATTN = "*"
_HYBRID_MLP = "-"
_HYBRID_MOE = "E"
_HYBRID_GDN = "G"


def _norm_params(size: int, normalization: str) -> int:
    """Number of parameters for a norm layer (weight + optional bias)."""
    return size * (2 if normalization == "LayerNorm" else 1)


def _attn_layer_params(
    hidden_size: int,
    num_attention_heads: int,
    num_query_groups: int,
    kv_channels: int,
    add_bias_linear: bool,
    normalization: str,
    qk_layernorm: bool,
    attention_output_gate: bool = False,
) -> int:
    """Params for a single attention sublayer.

    Includes ``linear_qkv`` (with fused ``input_layernorm``), ``linear_proj``, and optional QK layernorms.
    """
    # linear_qkv: hidden_size -> (Q + 2*KV) * kv_channels, with fused input_layernorm
    qkv_out = (num_attention_heads + 2 * num_query_groups) * kv_channels
    if attention_output_gate:
        qkv_out += kv_channels * num_attention_heads
    params = hidden_size * qkv_out
    if add_bias_linear:
        params += qkv_out
    params += _norm_params(hidden_size, normalization)  # fused input_layernorm

    # linear_proj: (num_attention_heads * kv_channels) -> hidden_size
    params += num_attention_heads * kv_channels * hidden_size
    if add_bias_linear:
        params += hidden_size

    # optional per-head QK layernorm (q_layernorm + k_layernorm)
    if qk_layernorm:
        params += 2 * _norm_params(kv_channels, normalization)

    return params


def _dense_mlp_params(
    hidden_size: int,
    ffn_hidden_size: int,
    gated_linear_unit: bool,
    add_bias_linear: bool,
    normalization: str,
) -> int:
    """Params for a dense MLP sublayer.

    ``pre_mlp_layernorm`` is fused into ``linear_fc1`` (TELayerNormColumnParallelLinear).
    """
    fc1_out = ffn_hidden_size * (2 if gated_linear_unit else 1)
    params = hidden_size * fc1_out
    if add_bias_linear:
        params += fc1_out
    params += _norm_params(hidden_size, normalization)  # fused pre_mlp_layernorm

    params += ffn_hidden_size * hidden_size
    if add_bias_linear:
        params += hidden_size

    return params


def _moe_layer_params(
    hidden_size: int,
    num_moe_experts: int,
    moe_router_topk: int,
    moe_ffn_hidden_size: int,
    gated_linear_unit: bool,
    add_bias_linear: bool,
    normalization: str,
    moe_shared_expert_intermediate_size: int | None,
    moe_shared_expert_gate: bool = False,
) -> tuple[int, int]:
    """Params for a MoE sublayer, returned as (total, active).

    ``pre_mlp_layernorm`` is a *separate* TENorm (not fused into expert fc1).
    Routed expert fc1/fc2 use ``TEColumnParallelLinear`` (no fused LN).
    Shared experts never carry bias regardless of ``add_bias_linear``.

    ``total`` counts all ``num_moe_experts`` routed experts; ``active`` counts only
    ``moe_router_topk`` (the experts actually used in each forward pass).  The router,
    pre-layernorm, and shared expert are always fully active and count equally in both.
    """
    # Always-active: pre_mlp_layernorm + router + shared expert
    always = _norm_params(hidden_size, normalization)
    always += num_moe_experts * hidden_size  # router weight
    if add_bias_linear:
        always += num_moe_experts  # router bias

    # Shared expert (SharedExpertMLP always has add_bias_linear=False)
    if moe_shared_expert_intermediate_size:
        s_fc1_out = moe_shared_expert_intermediate_size * (2 if gated_linear_unit else 1)
        always += hidden_size * s_fc1_out + moe_shared_expert_intermediate_size * hidden_size
        if moe_shared_expert_gate:
            always += hidden_size  # gate_weight: 1 x hidden_size

    # Per routed-expert params
    fc1_out = moe_ffn_hidden_size * (2 if gated_linear_unit else 1)
    per_expert = hidden_size * fc1_out + moe_ffn_hidden_size * hidden_size
    if add_bias_linear:
        per_expert += fc1_out + moe_ffn_hidden_size

    total = always + num_moe_experts * per_expert
    active = always + moe_router_topk * per_expert
    return total, active


def _mamba_layer_params(
    hidden_size: int,
    mamba_num_heads: int,
    mamba_head_dim: int,
    mamba_num_groups: int,
    mamba_state_dim: int,
    normalization: str,
    d_conv: int = 4,
) -> int:
    """Params for a single Mamba layer.

    ``in_proj`` uses TELayerNormColumnParallelLinear (fused input LN).
    The internal ``norm`` on ``d_inner`` is always RMSNorm (1 weight, no bias).
    """
    d_inner = mamba_num_heads * mamba_head_dim

    # in_proj: hidden_size -> (2*d_inner + 2*ngroups*d_state + nheads), no bias, fused input LN
    in_proj_out = 2 * d_inner + 2 * mamba_num_groups * mamba_state_dim + mamba_num_heads
    params = hidden_size * in_proj_out
    params += _norm_params(hidden_size, normalization)  # fused input_layernorm

    # out_proj: d_inner -> hidden_size, no bias
    params += d_inner * hidden_size

    # conv1d (depthwise) on (d_inner + 2*ngroups*d_state) channels: weight + bias
    conv_dim = d_inner + 2 * mamba_num_groups * mamba_state_dim
    params += conv_dim * d_conv + conv_dim

    # Scalar per-head params: A_log + dt_bias + D
    params += 3 * mamba_num_heads

    # Internal RMSNorm on d_inner (always RMSNorm, 1 weight only)
    params += d_inner

    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_ALLOWED_HYBRID_CHARS = {_HYBRID_MAMBA, _HYBRID_ATTN, _HYBRID_MLP, _HYBRID_MOE}


def parse_main_layer_chars(hybrid_layer_pattern: str, num_layers: int | None = None) -> list[str]:
    """Extract per-layer characters from the main (non-MTP) part of a hybrid pattern.

    Strips the MTP suffix (``/...``) and PP boundaries (``|``), returning one char per layer.
    When ``num_layers`` is provided the result length must equal it exactly.
    Raises ``ValueError`` for any character not in the supported set ``{M, *, -, E}``.
    """
    main = hybrid_layer_pattern.split("/")[0]
    chars = [c for c in main if c != "|"]
    unknown = {c for c in chars if c not in _ALLOWED_HYBRID_CHARS}
    if unknown:
        raise ValueError(
            f"Unsupported hybrid layer chars {unknown} in pattern '{hybrid_layer_pattern}'. "
            f"Supported: {_ALLOWED_HYBRID_CHARS}"
        )
    if num_layers is not None and len(chars) != num_layers:
        raise ValueError(
            f"Hybrid pattern '{hybrid_layer_pattern}' has {len(chars)} layers "
            f"but num_layers={num_layers}."
        )
    return chars


def mcore_param_count(
    config: Any,
    vocab_size: int,
    share_embeddings_and_output_weights: bool = False,
    hybrid_layer_pattern: str | None = None,
    **overrides: Any,
) -> tuple[int, int]:
    """Compute total and active parameter counts for an MCore GPT or Mamba/Hybrid model.

    For non-MoE models ``total == active``.  For MoE models, ``active`` counts only the
    ``moe_router_topk`` routed experts actually used per forward pass (router weights,
    pre-layernorm, and shared experts are always active and count in both figures).

    Args:
        config: MCore ``TransformerConfig`` (or any object exposing the same attributes).
        vocab_size: Vocabulary size.
        share_embeddings_and_output_weights: Whether the word-embedding and LM-head weights
            are tied (output_layer excluded from the count when True).
        hybrid_layer_pattern: Hybrid layer pattern string for Mamba/Hybrid models (e.g.
            ``"M*M*E-"``) or ``None`` for a pure GPT model.  Characters are the MCore
            ``Symbols`` values; ``|`` PP boundaries and ``/...`` MTP suffixes are stripped.
        **overrides: Per-call overrides for any ``config`` attribute.  Useful for computing
            counts for hypothetical configs without modifying the model.

    Returns: ``(total_params, active_params)``
    """

    def _get(attr: str, default: Any = None) -> Any:
        return overrides.get(attr, getattr(config, attr, default))

    hidden_size: int = _get("hidden_size")
    num_layers: int = _get("num_layers")
    num_attention_heads: int = _get("num_attention_heads")
    num_query_groups: int | None = _get("num_query_groups")
    kv_channels: int | None = _get("kv_channels")
    ffn_hidden_size: int | None = _get("ffn_hidden_size")
    num_moe_experts: int | None = _get("num_moe_experts")
    moe_router_topk: int = _get("moe_router_topk", 2)
    moe_ffn_hidden_size: int | None = _get("moe_ffn_hidden_size")
    moe_shared_expert_intermediate_size: int | None = _get("moe_shared_expert_intermediate_size")
    moe_shared_expert_gate: bool = _get("moe_shared_expert_gate", False)
    mamba_num_heads: int | None = _get("mamba_num_heads")
    mamba_head_dim: int | None = _get("mamba_head_dim")
    mamba_num_groups: int | None = _get("mamba_num_groups")
    mamba_state_dim: int | None = _get("mamba_state_dim")
    gated_linear_unit: bool = _get("gated_linear_unit", False)
    add_bias_linear: bool = _get("add_bias_linear", False)
    normalization: str = _get("normalization", "RMSNorm")
    qk_layernorm: bool = _get("qk_layernorm", False)
    attention_output_gate: bool = _get("attention_output_gate", False)
    moe_layer_freq: int | list[int] = _get("moe_layer_freq", 1)

    # Fill in derived defaults
    if num_query_groups is None:
        num_query_groups = num_attention_heads
    if kv_channels is None and num_attention_heads:
        kv_channels = hidden_size // num_attention_heads
    if moe_ffn_hidden_size is None and num_moe_experts is not None:
        moe_ffn_hidden_size = ffn_hidden_size

    # Embedding + final norm + output layer (always active)
    base = vocab_size * hidden_size
    base += _norm_params(hidden_size, normalization)  # final layernorm
    if not share_embeddings_and_output_weights:
        base += hidden_size * vocab_size

    total = base
    active = base

    if hybrid_layer_pattern is None:
        # ---- Pure GPT: all layers have attention + dense-MLP or attention + MoE ----
        assert kv_channels is not None, "kv_channels must be set for GPT attention layers"
        if isinstance(moe_layer_freq, list):
            moe_pattern = list(moe_layer_freq[:num_layers])
        else:
            moe_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]

        for i in range(num_layers):
            layer_t = layer_a = _attn_layer_params(
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                add_bias_linear,
                normalization,
                qk_layernorm,
                attention_output_gate,
            )
            if moe_pattern[i] and num_moe_experts:
                assert moe_ffn_hidden_size is not None, (
                    "moe_ffn_hidden_size must be set for MoE layers"
                )
                mt, ma = _moe_layer_params(
                    hidden_size,
                    num_moe_experts,
                    moe_router_topk,
                    moe_ffn_hidden_size,
                    gated_linear_unit,
                    add_bias_linear,
                    normalization,
                    moe_shared_expert_intermediate_size,
                    moe_shared_expert_gate,
                )
                layer_t += mt
                layer_a += ma
            else:
                assert ffn_hidden_size is not None, (
                    "ffn_hidden_size must be set for dense MLP layers"
                )
                mlp = _dense_mlp_params(
                    hidden_size,
                    ffn_hidden_size,
                    gated_linear_unit,
                    add_bias_linear,
                    normalization,
                )
                layer_t += mlp
                layer_a += mlp
            total += layer_t
            active += layer_a
    else:
        # ---- Hybrid / MambaModel: layer type is encoded in the pattern ----
        layer_chars = parse_main_layer_chars(hybrid_layer_pattern, num_layers)

        for char in layer_chars:
            if char == _HYBRID_MAMBA:
                assert mamba_num_heads is not None, "mamba_num_heads must be set for Mamba layers"
                assert mamba_head_dim is not None, "mamba_head_dim must be set for Mamba layers"
                assert mamba_num_groups is not None, "mamba_num_groups must be set for Mamba layers"
                assert mamba_state_dim is not None, "mamba_state_dim must be set for Mamba layers"
                t = a = _mamba_layer_params(
                    hidden_size,
                    mamba_num_heads,
                    mamba_head_dim,
                    mamba_num_groups,
                    mamba_state_dim,
                    normalization,
                )
            elif char == _HYBRID_ATTN:
                assert kv_channels is not None, "kv_channels must be set for attention layers"
                t = a = _attn_layer_params(
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    kv_channels,
                    add_bias_linear,
                    normalization,
                    qk_layernorm,
                    attention_output_gate,
                )
            elif char == _HYBRID_MLP:
                assert ffn_hidden_size is not None, "ffn_hidden_size must be set for MLP layers"
                t = a = _dense_mlp_params(
                    hidden_size,
                    ffn_hidden_size,
                    gated_linear_unit,
                    add_bias_linear,
                    normalization,
                )
            elif char == _HYBRID_MOE:
                assert num_moe_experts is not None, "num_moe_experts must be set for MoE layers"
                assert moe_ffn_hidden_size is not None, (
                    "moe_ffn_hidden_size must be set for MoE layers"
                )
                t, a = _moe_layer_params(
                    hidden_size,
                    num_moe_experts,
                    moe_router_topk,
                    moe_ffn_hidden_size,
                    gated_linear_unit,
                    add_bias_linear,
                    normalization,
                    moe_shared_expert_intermediate_size,
                    moe_shared_expert_gate,
                )
            else:
                raise ValueError(f"Unsupported hybrid layer character: {char}")
            total += t
            active += a

    return total, active


def mcore_param_count_live(model: GPTModel | MambaModel) -> int:
    """Count parameters in a live MCore GPTModel or MambaModel (reduced across TP, EP, ETP, and PP ranks)."""
    if isinstance(model, DynamicModule):
        raise RuntimeError(
            "mcore_param_count_live does not support DynamicModule. "
            "Use mcore_param_count() with the active hyperparameter values instead."
        )

    tp_rank = get_tensor_model_parallel_rank()
    # get_expert_tensor_parallel_rank() falls back to tp_rank when ETP is not configured.
    etp_rank = get_expert_tensor_parallel_rank()

    regular_params = 0  # allreduce=True (or unset): replicated / TP-sharded
    expert_params = 0  # allreduce=False: EP-sharded (and possibly ETP-sharded)

    for name, p in model.named_parameters():
        if model.share_embeddings_and_output_weights and "output_layer.weight" in name:
            continue
        is_expert_parallel = not getattr(p, "allreduce", True)
        is_tp_sharded = getattr(p, "tensor_model_parallel", False)
        if is_expert_parallel:
            # EP/ETP-sharded: ETP-sharded params are summed across all ETP ranks; non-ETP params
            # are counted only on ETP rank 0 to avoid multiplying by ETP size in the EPxETP reduce.
            if not is_tp_sharded and etp_rank != 0:
                continue
            expert_params += p.numel()
        else:
            # Non-expert: TP-sharded params are summed; replicated params counted on TP rank 0.
            if not is_tp_sharded and tp_rank != 0:
                continue
            regular_params += p.numel()

    device = next(model.parameters()).device

    regular_tensor = torch.tensor([regular_params], device=device)
    torch.distributed.all_reduce(regular_tensor, group=get_pipeline_model_parallel_group())
    torch.distributed.all_reduce(regular_tensor, group=get_tensor_model_parallel_group())

    ep_etp_group = get_expert_tensor_and_model_parallel_group(check_initialized=False)
    if ep_etp_group is not None:
        expert_tensor = torch.tensor([expert_params], device=device)
        torch.distributed.all_reduce(expert_tensor, group=get_pipeline_model_parallel_group())
        torch.distributed.all_reduce(expert_tensor, group=ep_etp_group)
        return int((regular_tensor + expert_tensor).item())

    return int(regular_tensor.item())


def mcore_memory_footprint_mb(
    config: Any,
    vocab_size: int,
    share_embeddings_and_output_weights: bool = False,
    hybrid_layer_pattern: str | None = None,
    dtype_bytes: int = 2,
    kv_cache_dtype_bytes: int | None = None,
    sequence_length: int = 4096,
    batch_size: int = 1,
    **overrides: Any,
) -> tuple[float, float, float, float]:
    """Compute inference memory footprint in MB for an MCore model.

    Covers three components:

    * **params**: model weights at ``dtype_bytes`` precision.
    * **kv_cache**: KV cache for all attention layers (2 tensors per layer at ``kv_cache_dtype_bytes``
      precision). This assumes full global attention; configs with sliding-window or local attention
      (e.g. some Nemotron variants) will have a smaller real KV cache — treat this as an upper bound.
    * **mamba_state**: recurrent SSM sliding-window state stored for all Mamba layers during
      generation (one buffer of size ``(d_inner + 2*ngroups*d_state) * d_conv`` per layer).

    Args:
        config: MCore ``TransformerConfig`` (or any object exposing the same attributes).
        vocab_size: Vocabulary size.
        share_embeddings_and_output_weights: Tied embedding/LM-head flag.
        hybrid_layer_pattern: Hybrid layer pattern (``None`` for pure GPT).
        dtype_bytes: Bytes per parameter (2 for fp16/bf16, 4 for fp32).
        kv_cache_dtype_bytes: Bytes per KV-cache element; defaults to ``dtype_bytes``.
        sequence_length: Context length for KV-cache sizing.
        batch_size: Batch size for KV-cache and Mamba-state sizing.
        **overrides: Config attribute overrides (same as :func:`mcore_param_count`).

    Returns: ``(params_mb, kv_cache_mb, mamba_state_mb, total_mb)``
    """
    if kv_cache_dtype_bytes is None:
        kv_cache_dtype_bytes = dtype_bytes

    def _get(attr: str, default: Any = None) -> Any:
        return overrides.get(attr, getattr(config, attr, default))

    hidden_size: int = _get("hidden_size")
    num_layers: int = _get("num_layers")
    num_attention_heads: int = _get("num_attention_heads")
    num_query_groups: int | None = _get("num_query_groups")
    kv_channels: int | None = _get("kv_channels")
    mamba_num_heads: int | None = _get("mamba_num_heads")
    mamba_head_dim: int | None = _get("mamba_head_dim")
    mamba_num_groups: int | None = _get("mamba_num_groups")
    mamba_state_dim: int | None = _get("mamba_state_dim")

    if num_query_groups is None:
        num_query_groups = num_attention_heads
    if kv_channels is None and num_attention_heads:
        kv_channels = hidden_size // num_attention_heads

    # Parameter memory
    total_params, _ = mcore_param_count(
        config,
        vocab_size,
        share_embeddings_and_output_weights,
        hybrid_layer_pattern,
        **overrides,
    )
    params_bytes = total_params * dtype_bytes

    # Count attention and Mamba layers from pattern
    if hybrid_layer_pattern is None:
        n_attn = num_layers
        n_mamba = 0
    else:
        chars = parse_main_layer_chars(hybrid_layer_pattern, num_layers)
        n_attn = chars.count(_HYBRID_ATTN)
        n_mamba = chars.count(_HYBRID_MAMBA)

    # KV cache: 2 tensors (K, V) per attention layer
    # each tensor: [batch_size, sequence_length, num_query_groups, kv_channels]
    kv_bytes = 0
    if n_attn > 0:
        if num_query_groups is None or kv_channels is None:
            raise ValueError(
                "num_query_groups and kv_channels must be set when attention layers exist."
            )
        kv_per_layer = 2 * batch_size * sequence_length * num_query_groups * kv_channels
        kv_bytes = n_attn * kv_per_layer * kv_cache_dtype_bytes

    # Mamba recurrent state per layer (both caches needed for autoregressive generation):
    #   conv1d sliding window: [batch, d_inner + 2*ngroups*d_state, d_conv - 1]
    #   SSM recurrent state:   [batch, nheads, d_head, d_state]
    mamba_bytes = 0
    if n_mamba > 0:
        if None in (mamba_num_heads, mamba_head_dim, mamba_num_groups, mamba_state_dim):
            raise ValueError(
                "mamba_num_heads, mamba_head_dim, mamba_num_groups, and mamba_state_dim "
                "must be set when Mamba layers exist."
            )
        d_inner = mamba_num_heads * mamba_head_dim
        d_conv = 4  # hardcoded in MambaMixer
        conv_dim = d_inner + 2 * mamba_num_groups * mamba_state_dim
        conv_state = batch_size * conv_dim * (d_conv - 1)
        ssm_state = batch_size * mamba_num_heads * mamba_head_dim * mamba_state_dim
        mamba_bytes = n_mamba * (conv_state + ssm_state) * dtype_bytes

    _mb = 1024**2
    params_mb = params_bytes / _mb
    kv_cache_mb = kv_bytes / _mb
    mamba_state_mb = mamba_bytes / _mb
    total_mb = params_mb + kv_cache_mb + mamba_state_mb
    return params_mb, kv_cache_mb, mamba_state_mb, total_mb


def print_mcore_model_stats(
    model: "GPTModel | MambaModel",
    label: str = "Model",
    seq_length: int = 4096,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> None:
    """Print total params, active params, and memory footprint for an MCore model.

    Args:
        model: GPTModel or MambaModel to print stats for.
        label: Label prefix for the output line (e.g. ``"Original"``, ``"Pruned"``).
        seq_length: Sequence length for KV-cache / Mamba-state memory estimate.
        batch_size: Batch size for KV-cache / Mamba-state memory estimate.
        dtype_bytes: Bytes per parameter for memory estimation (default: 2 for BF16).
    """
    hybrid_layer_pattern: str | None = None
    config_overrides: dict = {}
    if isinstance(model, MambaModel):
        hybrid_key = (
            "hybrid_override_pattern"
            if hasattr(model, "hybrid_override_pattern")
            else "hybrid_layer_pattern"
        )
        hybrid_layer_pattern = getattr(model, hybrid_key)
        # mamba_num_heads may not be stored in config when derived from model architecture;
        # fall back to reading it from the actual layer.
        if getattr(model.config, "mamba_num_heads", None) is None:
            for layer in model.decoder.layers:
                if hasattr(layer, "mixer") and hasattr(layer.mixer, "nheads"):
                    config_overrides["mamba_num_heads"] = layer.mixer.nheads
                    break

    total, active = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=hybrid_layer_pattern,
        **config_overrides,
    )
    params_mb, kv_cache_mb, mamba_state_mb, total_mb = mcore_memory_footprint_mb(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=hybrid_layer_pattern,
        dtype_bytes=dtype_bytes,
        sequence_length=seq_length,
        batch_size=batch_size,
        **config_overrides,
    )
    dtype_str = {1: "FP8", 2: "BF16", 4: "FP32"}.get(dtype_bytes, f"{dtype_bytes}B")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("Total Parameters", num2hrb(total))
    if active != total:
        grid.add_row("Active Parameters", num2hrb(active))

    mem_items = [f"weights: {params_mb:.1f} MB", f"kv_cache: {kv_cache_mb:.1f} MB"]
    if mamba_state_mb > 0:
        mem_items.append(f"mamba_state: {mamba_state_mb:.1f} MB")
    mem_items.append(f"[bold]Total: {total_mb:.1f} MB[/bold]")
    grid.add_row(f"Memory ({dtype_str}, {seq_length=}, {batch_size=})", ", ".join(mem_items))

    buf = io.StringIO()
    Console(file=buf, highlight=False, force_terminal=sys.stdout.isatty()).print(
        Panel(
            grid,
            title=f"[bold cyan]{label} Stats[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
            expand=False,
        )
    )
    print_rank_0()
    print_rank_0(buf.getvalue())
