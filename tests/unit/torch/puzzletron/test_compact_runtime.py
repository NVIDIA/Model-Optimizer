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

"""Tests for reversible compact Qwen attention and GDN runtime projections."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.pruning.attention_ffn_surgery import (
    slice_attention_weights,
    sorted_attention_keep_indices,
)
from modelopt.torch.puzzletron.pruning.compact_runtime import (
    compact_gated_delta_net_forward,
    compact_grouped_attention_forward,
    resolve_compact_grouped_attention_target,
)
from modelopt.torch.puzzletron.pruning.gated_delta_net import (
    GDNShape,
    slice_gated_delta_net_state_dict,
)


def _qwen_config(*, layer_type: str):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        dtype=torch.float32,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        max_position_embeddings=32,
        vocab_size=32,
        layer_types=[layer_type],
        attn_implementation="eager",
    )


def test_compact_grouped_attention_target_requires_reduced_supported_geometry():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    config = _qwen_config(layer_type="full_attention")
    attention = Qwen3_5Attention(config, 0)
    layer = SimpleNamespace(self_attn=attention)
    teacher = SimpleNamespace(
        no_op=False,
        num_query_heads=4,
        num_kv_heads=2,
        qk_head_dim=8,
    )
    child = SimpleNamespace(
        no_op=False,
        num_query_heads=2,
        num_kv_heads=1,
    )

    target = resolve_compact_grouped_attention_target(layer, teacher, child)

    assert target == {
        "module": attention,
        "orig_num_q": 4,
        "orig_num_kv": 2,
        "target_num_q": 2,
        "target_num_kv": 1,
        "head_dim": 8,
    }
    assert resolve_compact_grouped_attention_target(layer, teacher, teacher) is None


@pytest.mark.parametrize(("target_num_q", "target_num_kv"), [(2, 1), (2, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_compact_grouped_attention_matches_physical_projection_geometry(
    target_num_q: int,
    target_num_kv: int,
    dtype: torch.dtype,
):
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    teacher_config = _qwen_config(layer_type="full_attention")
    teacher = Qwen3_5Attention(teacher_config, 0).to(dtype=dtype).eval()
    target_config = deepcopy(teacher_config)
    target_config.num_attention_heads = target_num_q
    target_config.num_key_value_heads = target_num_kv
    physical = Qwen3_5Attention(target_config, 0).to(dtype=dtype).eval()

    state = {name: tensor.detach().clone() for name, tensor in teacher.state_dict().items()}
    keep_q, keep_kv = sorted_attention_keep_indices(
        target_num_kv,
        target_num_q // target_num_kv,
        teacher_config.num_attention_heads // teacher_config.num_key_value_heads,
    )
    q, k, v, o = slice_attention_weights(
        state["q_proj.weight"],
        state["k_proj.weight"],
        state["v_proj.weight"],
        state["o_proj.weight"],
        keep_q,
        keep_kv,
        teacher_config.head_dim,
    )
    state.update(
        {
            "q_proj.weight": q,
            "k_proj.weight": k,
            "v_proj.weight": v,
            "o_proj.weight": o,
        }
    )
    physical.load_state_dict(state)

    hidden_states = torch.randn(1, 4, teacher_config.hidden_size, dtype=dtype)
    position_embeddings = (
        torch.ones(1, 4, teacher_config.head_dim, dtype=dtype),
        torch.zeros(1, 4, teacher_config.head_dim, dtype=dtype),
    )
    attention_mask = torch.zeros(1, 1, 4, 4, dtype=dtype)
    original_shapes = {name: tuple(tensor.shape) for name, tensor in teacher.state_dict().items()}

    with torch.no_grad():
        teacher_output = teacher(
            hidden_states,
            position_embeddings,
            attention_mask,
        )[0]
        physical_output = physical(
            hidden_states,
            position_embeddings,
            attention_mask,
        )[0]
        with compact_grouped_attention_forward(
            teacher,
            orig_num_q=4,
            orig_num_kv=2,
            target_num_q=target_num_q,
            target_num_kv=target_num_kv,
            head_dim=8,
        ):
            runtime_output = teacher(
                hidden_states,
                position_embeddings,
                attention_mask,
            )[0]

        physical_cache = DynamicCache(config=target_config)
        runtime_cache = DynamicCache(config=teacher_config)
        physical_prefill = physical(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=physical_cache,
        )[0]
        with compact_grouped_attention_forward(
            teacher,
            orig_num_q=4,
            orig_num_kv=2,
            target_num_q=target_num_q,
            target_num_kv=target_num_kv,
            head_dim=8,
        ):
            runtime_prefill = teacher(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=runtime_cache,
            )[0]
            next_hidden = torch.randn(1, 1, teacher_config.hidden_size, dtype=dtype)
            next_position_embeddings = (
                torch.ones(1, 1, teacher_config.head_dim, dtype=dtype),
                torch.zeros(1, 1, teacher_config.head_dim, dtype=dtype),
            )
            next_attention_mask = torch.zeros(1, 1, 1, 5, dtype=dtype)
            runtime_decode = teacher(
                next_hidden,
                next_position_embeddings,
                next_attention_mask,
                past_key_values=runtime_cache,
            )[0]
        physical_decode = physical(
            next_hidden,
            next_position_embeddings,
            next_attention_mask,
            past_key_values=physical_cache,
        )[0]
        restored_output = teacher(
            hidden_states,
            position_embeddings,
            attention_mask,
        )[0]

    assert torch.equal(runtime_output, physical_output)
    assert torch.equal(runtime_prefill, physical_prefill)
    assert torch.equal(runtime_decode, physical_decode)
    assert torch.equal(restored_output, teacher_output)
    assert {name: tuple(tensor.shape) for name, tensor in teacher.state_dict().items()} == (
        original_shapes
    )
    assert teacher.num_key_value_groups == 2


@pytest.mark.parametrize(
    "target_shape",
    [
        GDNShape(num_key_heads=1, num_value_heads=2, key_head_dim=8, value_head_dim=8),
        GDNShape(num_key_heads=2, num_value_heads=2, key_head_dim=8, value_head_dim=8),
        GDNShape(num_key_heads=2, num_value_heads=4, key_head_dim=4, value_head_dim=8),
        GDNShape(num_key_heads=2, num_value_heads=4, key_head_dim=8, value_head_dim=4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_compact_gdn_matches_physical_projection_and_kernel_geometry(
    target_shape: GDNShape,
    dtype: torch.dtype,
):
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    teacher_config = _qwen_config(layer_type="linear_attention")
    teacher = Qwen3_5GatedDeltaNet(teacher_config, 0).to(dtype=dtype).eval()
    teacher_shape = GDNShape.from_module(teacher)
    target_config = deepcopy(teacher_config)
    target_config.linear_num_key_heads = target_shape.num_key_heads
    target_config.linear_num_value_heads = target_shape.num_value_heads
    target_config.linear_key_head_dim = target_shape.key_head_dim
    target_config.linear_value_head_dim = target_shape.value_head_dim
    physical = Qwen3_5GatedDeltaNet(target_config, 0).to(dtype=dtype).eval()

    state = {
        f"gdn.{name}": tensor.detach().clone() for name, tensor in teacher.state_dict().items()
    }
    slice_gated_delta_net_state_dict(
        state,
        prefix="gdn",
        shape=teacher_shape,
        target=target_shape,
    )
    physical.load_state_dict({name.removeprefix("gdn."): tensor for name, tensor in state.items()})
    hidden_states = torch.randn(2, 4, teacher_config.hidden_size, dtype=dtype)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    original_shapes = {name: tuple(tensor.shape) for name, tensor in teacher.state_dict().items()}

    with torch.no_grad():
        teacher_output = teacher(hidden_states, attention_mask=attention_mask)
        physical_output = physical(hidden_states, attention_mask=attention_mask)
        with compact_gated_delta_net_forward(
            teacher,
            teacher_shape=teacher_shape,
            target_shape=target_shape,
        ):
            runtime_output = teacher(hidden_states, attention_mask=attention_mask)

        physical_cache = DynamicCache(config=target_config)
        runtime_cache = DynamicCache(config=teacher_config)
        physical_prefill = physical(
            hidden_states,
            cache_params=physical_cache,
            attention_mask=attention_mask,
        )
        with compact_gated_delta_net_forward(
            teacher,
            teacher_shape=teacher_shape,
            target_shape=target_shape,
        ):
            runtime_prefill = teacher(
                hidden_states,
                cache_params=runtime_cache,
                attention_mask=attention_mask,
            )
            next_hidden = torch.randn(2, 1, teacher_config.hidden_size, dtype=dtype)
            runtime_decode = teacher(next_hidden, cache_params=runtime_cache)
        physical_decode = physical(next_hidden, cache_params=physical_cache)
        restored_output = teacher(hidden_states, attention_mask=attention_mask)

    assert torch.equal(runtime_output, physical_output)
    assert torch.equal(runtime_prefill, physical_prefill)
    assert torch.equal(runtime_decode, physical_decode)
    assert torch.equal(restored_output, teacher_output)
    assert {name: tuple(tensor.shape) for name, tensor in teacher.state_dict().items()} == (
        original_shapes
    )
    assert GDNShape.from_module(teacher) == teacher_shape
