# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Reversible compact forwards for physically sliced attention and GDN candidates."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

from .attention_ffn_surgery import slice_query_rows_by_head, sorted_attention_keep_indices
from .gated_delta_net import GDNShape, gated_delta_net_prefix_indices

__all__ = [
    "compact_gated_delta_net_forward",
    "compact_grouped_attention_forward",
    "resolve_compact_grouped_attention_target",
    "supports_compact_gated_delta_net",
    "supports_compact_grouped_attention",
]


_SUPPORTED_GROUPED_ATTENTION_TYPES = {
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5Attention"),
}
_SUPPORTED_GDN_TYPES = {
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5GatedDeltaNet"),
}


def _type_key(module) -> tuple[str, str]:
    module_type = type(module)
    return module_type.__module__, module_type.__qualname__


def _indices_on(indices: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return indices.to(device=tensor.device)


def _index_select(tensor: torch.Tensor | None, dim: int, indices: torch.Tensor):
    if tensor is None:
        return None
    return tensor.index_select(dim, _indices_on(indices, tensor)).contiguous()


@contextmanager
def _indexed_linear_forward(
    module,
    *,
    output_indices: torch.Tensor | None = None,
    input_indices: torch.Tensor | None = None,
):
    """Execute one linear layer through selected rows/columns without resizing parameters."""

    missing = object()
    original_instance_forward = vars(module).get("forward", missing)

    def forward(self, inputs):
        weight = self.weight
        bias = self.bias
        if output_indices is not None:
            weight = _index_select(weight, 0, output_indices)
            bias = _index_select(bias, 0, output_indices)
        if input_indices is not None:
            weight = _index_select(weight, 1, input_indices)
        return F.linear(inputs, weight, bias)

    module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        if original_instance_forward is missing:
            del module.forward
        else:
            module.forward = original_instance_forward


@contextmanager
def _temporary_attrs(module, updates: dict[str, object]):
    originals = {name: getattr(module, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def _has_compact_grouped_attention_layout(
    attention_module,
    *,
    orig_num_q: int,
    orig_num_kv: int,
    head_dim: int,
) -> bool:
    q_proj = getattr(attention_module, "q_proj", None)
    k_proj = getattr(attention_module, "k_proj", None)
    v_proj = getattr(attention_module, "v_proj", None)
    o_proj = getattr(attention_module, "o_proj", None)
    if any(
        module is None or not hasattr(module, "weight")
        for module in (q_proj, k_proj, v_proj, o_proj)
    ):
        return False
    assert q_proj is not None and k_proj is not None and v_proj is not None and o_proj is not None
    return (
        int(getattr(attention_module, "head_dim", -1)) == int(head_dim)
        and int(getattr(attention_module, "num_key_value_groups", -1))
        == int(orig_num_q) // int(orig_num_kv)
        and int(q_proj.weight.shape[0]) == 2 * int(orig_num_q) * int(head_dim)
        and int(k_proj.weight.shape[0]) == int(orig_num_kv) * int(head_dim)
        and int(v_proj.weight.shape[0]) == int(orig_num_kv) * int(head_dim)
        and int(o_proj.weight.shape[1]) == int(orig_num_q) * int(head_dim)
    )


def supports_compact_grouped_attention(
    attention_module,
    *,
    orig_num_q: int,
    orig_num_kv: int,
    head_dim: int,
) -> bool:
    """Return whether this exact tested Qwen attention layout supports compact execution."""

    return (
        _type_key(attention_module) in _SUPPORTED_GROUPED_ATTENTION_TYPES
        and orig_num_q > 0
        and orig_num_kv > 0
        and head_dim > 0
        and orig_num_q % orig_num_kv == 0
        and _has_compact_grouped_attention_layout(
            attention_module,
            orig_num_q=orig_num_q,
            orig_num_kv=orig_num_kv,
            head_dim=head_dim,
        )
    )


def resolve_compact_grouped_attention_target(layer, teacher_attention, child_attention):
    """Resolve one physically compact gated-attention target, if supported."""

    if (
        teacher_attention is None
        or child_attention is None
        or getattr(teacher_attention, "no_op", False)
        or getattr(child_attention, "no_op", False)
    ):
        return None
    attention_module = getattr(layer, "self_attn", None)
    if attention_module is None:
        return None
    orig_num_q = getattr(teacher_attention, "num_query_heads", None)
    orig_num_kv = getattr(teacher_attention, "num_kv_heads", None)
    target_num_q = getattr(child_attention, "num_query_heads", None)
    target_num_kv = getattr(child_attention, "num_kv_heads", None)
    head_dim = getattr(teacher_attention, "qk_head_dim", None) or getattr(
        attention_module, "head_dim", None
    )
    if (
        orig_num_q is None
        or orig_num_kv is None
        or target_num_q is None
        or target_num_kv is None
        or head_dim is None
    ):
        return None
    orig_num_q = int(orig_num_q)
    orig_num_kv = int(orig_num_kv)
    target_num_q = int(target_num_q)
    target_num_kv = int(target_num_kv)
    head_dim = int(head_dim)
    if min(orig_num_q, orig_num_kv, target_num_q, target_num_kv, head_dim) <= 0:
        raise ValueError("grouped-attention head counts and head dimension must be positive")
    if orig_num_q % orig_num_kv or target_num_q % target_num_kv:
        raise ValueError("grouped-attention query heads must be divisible by KV heads")
    if target_num_q > orig_num_q or target_num_kv > orig_num_kv:
        raise ValueError("grouped-attention target geometry cannot exceed the teacher")
    if target_num_q == orig_num_q and target_num_kv == orig_num_kv:
        return None
    if target_num_q // target_num_kv > orig_num_q // orig_num_kv:
        raise ValueError("grouped-attention target GQA ratio cannot exceed the teacher")
    has_compact_layout = _has_compact_grouped_attention_layout(
        attention_module,
        orig_num_q=orig_num_q,
        orig_num_kv=orig_num_kv,
        head_dim=head_dim,
    )
    if has_compact_layout and _type_key(attention_module) not in _SUPPORTED_GROUPED_ATTENTION_TYPES:
        module_name, type_name = _type_key(attention_module)
        raise RuntimeError(
            "Compact grouped-attention scoring is unsupported for "
            f"{module_name}.{type_name}; refusing to score reduced geometry"
        )
    if not supports_compact_grouped_attention(
        attention_module,
        orig_num_q=orig_num_q,
        orig_num_kv=orig_num_kv,
        head_dim=head_dim,
    ):
        return None
    return {
        "module": attention_module,
        "orig_num_q": orig_num_q,
        "orig_num_kv": orig_num_kv,
        "target_num_q": target_num_q,
        "target_num_kv": target_num_kv,
        "head_dim": head_dim,
    }


@contextmanager
def compact_grouped_attention_forward(
    attention_module,
    *,
    orig_num_q: int,
    orig_num_kv: int,
    target_num_q: int,
    target_num_kv: int,
    head_dim: int,
):
    """Run gated grouped attention with the exact projection geometry of export."""

    if not supports_compact_grouped_attention(
        attention_module,
        orig_num_q=orig_num_q,
        orig_num_kv=orig_num_kv,
        head_dim=head_dim,
    ):
        raise ValueError(
            f"Unsupported compact grouped-attention layout on {type(attention_module).__name__}"
        )
    if min(target_num_q, target_num_kv) <= 0:
        raise ValueError("target query and KV head counts must be positive")
    if target_num_q > orig_num_q or target_num_kv > orig_num_kv:
        raise ValueError("target query and KV head counts cannot exceed the teacher")
    if target_num_q % target_num_kv:
        raise ValueError(
            f"target query heads={target_num_q} must be divisible by KV heads={target_num_kv}"
        )
    orig_ratio = int(orig_num_q) // int(orig_num_kv)
    target_ratio = int(target_num_q) // int(target_num_kv)
    if target_ratio > orig_ratio:
        raise ValueError(
            f"target query-head ratio={target_ratio} exceeds teacher ratio={orig_ratio}"
        )

    keep_q, keep_kv = sorted_attention_keep_indices(int(target_num_kv), target_ratio, orig_ratio)
    q_proj = attention_module.q_proj
    q_rows = slice_query_rows_by_head(
        torch.arange(q_proj.weight.shape[0]), keep_q, int(head_dim), int(orig_num_q)
    )
    kv_rows = (keep_kv[:, None] * int(head_dim) + torch.arange(int(head_dim))[None, :]).reshape(-1)
    q_columns = (keep_q[:, None] * int(head_dim) + torch.arange(int(head_dim))[None, :]).reshape(-1)

    with ExitStack() as stack:
        stack.enter_context(_indexed_linear_forward(q_proj, output_indices=q_rows))
        stack.enter_context(
            _indexed_linear_forward(attention_module.k_proj, output_indices=kv_rows)
        )
        stack.enter_context(
            _indexed_linear_forward(attention_module.v_proj, output_indices=kv_rows)
        )
        stack.enter_context(
            _indexed_linear_forward(attention_module.o_proj, input_indices=q_columns)
        )
        stack.enter_context(
            _temporary_attrs(
                attention_module,
                {"num_key_value_groups": int(target_num_q) // int(target_num_kv)},
            )
        )
        yield


def _compact_gdn_norm(norm, hidden_states, gate, value_dim_indices: torch.Tensor):
    compact_weight = _index_select(norm.weight, 0, value_dim_indices)
    return torch.func.functional_call(
        norm,
        {"weight": compact_weight},
        (hidden_states, gate),
        strict=False,
    )


def supports_compact_gated_delta_net(gdn_module, *, teacher_shape: GDNShape) -> bool:
    """Return whether this exact tested Qwen GDN layout supports compact execution."""

    if _type_key(gdn_module) not in _SUPPORTED_GDN_TYPES:
        return False
    if "_fp32_params" in getattr(gdn_module, "_modules", {}):
        return False
    if not {"A_log", "dt_bias"}.issubset(getattr(gdn_module, "_parameters", {})):
        return False
    required = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "conv1d",
        "norm",
        "out_proj",
        "chunk_gated_delta_rule",
        "recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "conv_kernel_size",
        "activation",
        "layer_idx",
    )
    if any(not hasattr(gdn_module, name) for name in required):
        return False
    projection_width = (
        2 * teacher_shape.num_key_heads * teacher_shape.key_head_dim
        + teacher_shape.num_value_heads * teacher_shape.value_head_dim
    )
    value_width = teacher_shape.num_value_heads * teacher_shape.value_head_dim
    return (
        GDNShape.from_module(gdn_module) == teacher_shape
        and int(gdn_module.in_proj_qkv.weight.shape[0]) == projection_width
        and int(gdn_module.in_proj_z.weight.shape[0]) == value_width
        and int(gdn_module.in_proj_a.weight.shape[0]) == teacher_shape.num_value_heads
        and int(gdn_module.in_proj_b.weight.shape[0]) == teacher_shape.num_value_heads
        and int(gdn_module.conv1d.weight.shape[0]) == projection_width
        and int(gdn_module.norm.weight.shape[0]) == teacher_shape.value_head_dim
        and int(gdn_module.out_proj.weight.shape[1]) == value_width
        and int(gdn_module.A_log.shape[0]) == teacher_shape.num_value_heads
        and int(gdn_module.dt_bias.shape[0]) == teacher_shape.num_value_heads
    )


@contextmanager
def compact_gated_delta_net_forward(
    gdn_module,
    *,
    teacher_shape: GDNShape,
    target_shape: GDNShape,
):
    """Run Qwen-style GDN with the exact compact geometry used by materialization."""

    live_shape = GDNShape.from_module(gdn_module)
    if live_shape != teacher_shape:
        raise ValueError(f"live GDN shape {live_shape} does not match teacher {teacher_shape}")
    if not supports_compact_gated_delta_net(gdn_module, teacher_shape=teacher_shape):
        module_name, type_name = _type_key(gdn_module)
        raise RuntimeError(
            "Compact GDN scoring is unsupported for "
            f"{module_name}.{type_name}; refusing to score reduced geometry"
        )

    indices = gated_delta_net_prefix_indices(teacher_shape, target_shape)
    cidx = indices["cidx"]
    hidx = indices["hidx"]
    vidx = indices["vidx"]
    value_dim_indices = torch.arange(target_shape.value_head_dim)
    missing = object()
    original_instance_forward = vars(gdn_module).get("forward", missing)

    def compact_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: torch.Tensor | None = None,
        seq_idx=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported compact GDN forward arguments: {sorted(kwargs)}")
        if (
            attention_mask is not None
            and attention_mask.shape[1] > 1
            and attention_mask.shape[0] > 1
        ):
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx
        )
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states
            compact_projection_width = int(cidx.numel())
            expected_recurrent_shape = (
                target_shape.num_value_heads,
                target_shape.key_head_dim,
                target_shape.value_head_dim,
            )
            if int(conv_state.shape[-2]) != compact_projection_width:
                raise ValueError(
                    "Compact GDN cache convolution width does not match the target geometry: "
                    f"actual={int(conv_state.shape[-2])} expected={compact_projection_width}"
                )
            actual_recurrent_shape = tuple(int(size) for size in recurrent_state.shape[-3:])
            if actual_recurrent_shape != expected_recurrent_shape:
                raise ValueError(
                    "Compact GDN recurrent-state shape does not match the target geometry: "
                    f"actual={actual_recurrent_shape} expected={expected_recurrent_shape}"
                )

        mixed_qkv = F.linear(
            hidden_states,
            _index_select(self.in_proj_qkv.weight, 0, cidx),
            _index_select(self.in_proj_qkv.bias, 0, cidx),
        ).transpose(1, 2)
        z = F.linear(
            hidden_states,
            _index_select(self.in_proj_z.weight, 0, vidx),
            _index_select(self.in_proj_z.bias, 0, vidx),
        ).reshape(batch_size, seq_len, target_shape.num_value_heads, target_shape.value_head_dim)
        b = F.linear(
            hidden_states,
            _index_select(self.in_proj_b.weight, 0, hidx),
            _index_select(self.in_proj_b.bias, 0, hidx),
        )
        a = F.linear(
            hidden_states,
            _index_select(self.in_proj_a.weight, 0, hidx),
            _index_select(self.in_proj_a.bias, 0, hidx),
        )
        conv_weight = _index_select(self.conv1d.weight, 0, cidx)
        conv_bias = _index_select(self.conv1d.bias, 0, cidx)

        if use_precomputed_states and seq_len == 1:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weight.squeeze(1),
                conv_bias,
                self.activation,
            )
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=conv_weight.squeeze(1),
                    bias=conv_bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                )
            else:
                mixed_qkv = F.silu(
                    F.conv1d(
                        mixed_qkv,
                        conv_weight,
                        conv_bias,
                        stride=self.conv1d.stride,
                        padding=self.conv1d.padding,
                        dilation=self.conv1d.dilation,
                        groups=target_shape.num_key_heads * target_shape.key_head_dim * 2
                        + target_shape.num_value_heads * target_shape.value_head_dim,
                    )[:, :, : mixed_qkv.shape[-1]]
                )
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                target_shape.num_key_heads * target_shape.key_head_dim,
                target_shape.num_key_heads * target_shape.key_head_dim,
                target_shape.num_value_heads * target_shape.value_head_dim,
            ],
            dim=-1,
        )
        query = query.reshape(
            batch_size, seq_len, target_shape.num_key_heads, target_shape.key_head_dim
        )
        key = key.reshape(
            batch_size, seq_len, target_shape.num_key_heads, target_shape.key_head_dim
        )
        value = value.reshape(
            batch_size, seq_len, target_shape.num_value_heads, target_shape.value_head_dim
        )

        beta = b.sigmoid()
        g = -_index_select(self.A_log, 0, hidx).float().exp() * F.softplus(
            a.float() + _index_select(self.dt_bias, 0, hidx)
        )
        repeats = target_shape.num_value_heads // target_shape.num_key_heads
        if repeats > 1:
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, target_shape.value_head_dim)
        z = z.reshape(-1, target_shape.value_head_dim)
        core_attn_out = _compact_gdn_norm(self.norm, core_attn_out, z, value_dim_indices).reshape(
            batch_size, seq_len, -1
        )
        return F.linear(
            core_attn_out,
            _index_select(self.out_proj.weight, 1, vidx),
            self.out_proj.bias,
        )

    gdn_module.forward = MethodType(compact_forward, gdn_module)
    try:
        yield
    finally:
        if original_instance_forward is missing:
            del gdn_module.forward
        else:
            gdn_module.forward = original_instance_forward
