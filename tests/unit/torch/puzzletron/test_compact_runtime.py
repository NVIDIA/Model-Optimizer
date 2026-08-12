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
    supports_compact_gated_delta_net,
    supports_compact_grouped_attention,
)
from modelopt.torch.puzzletron.pruning.gated_delta_net import (
    GDNShape,
    slice_gated_delta_net_state_dict,
)


def _qwen_modeling():
    return pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


def _qwen_config(*, layer_type: str):
    _qwen_modeling()
    # Optional dependency: Qwen3.5 is available only in newer transformers builds.
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
    qwen_attention_cls = _qwen_modeling().Qwen3_5Attention

    config = _qwen_config(layer_type="full_attention")
    attention = qwen_attention_cls(config, 0)
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


def test_compact_grouped_attention_rejects_native_automodel_backend():
    # Optional dependency: native AutoModel Qwen modules are not installed in every test env.
    pytest.importorskip("nemo_automodel.components.models.qwen3_next.layers")
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    config = Qwen3NextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        layer_types=["full_attention"],
    )
    config.head_dim = 8
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    attention = Qwen3NextAttention(config, layer_idx=0, backend=backend)
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
    original_forward = attention.forward.__func__
    original_state = set(vars(attention))

    assert not supports_compact_grouped_attention(
        attention,
        orig_num_q=4,
        orig_num_kv=2,
        head_dim=8,
    )
    with pytest.raises(RuntimeError, match="refusing to score reduced geometry"):
        resolve_compact_grouped_attention_target(layer, teacher, child)

    assert attention.forward.__func__ is original_forward
    assert set(vars(attention)) == original_state


@pytest.mark.parametrize(("target_num_q", "target_num_kv"), [(2, 1), (2, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_compact_grouped_attention_matches_physical_projection_geometry(
    target_num_q: int,
    target_num_kv: int,
    dtype: torch.dtype,
):
    from transformers.cache_utils import DynamicCache

    qwen_attention_cls = _qwen_modeling().Qwen3_5Attention

    teacher_config = _qwen_config(layer_type="full_attention")
    teacher = qwen_attention_cls(teacher_config, 0).to(dtype=dtype).eval()
    target_config = deepcopy(teacher_config)
    target_config.num_attention_heads = target_num_q
    target_config.num_key_value_heads = target_num_kv
    physical = qwen_attention_cls(target_config, 0).to(dtype=dtype).eval()

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
    projection_modules = (teacher.q_proj, teacher.k_proj, teacher.v_proj, teacher.o_proj)
    assert all("forward" not in vars(module) for module in projection_modules)

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
            assert all("forward" in vars(module) for module in projection_modules)
            runtime_output = teacher(
                hidden_states,
                position_embeddings,
                attention_mask,
            )[0]

        # DynamicCache configs select attention types, not Q/K/V geometry; both are full attention.
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
    assert all("forward" not in vars(module) for module in projection_modules)


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

    qwen_gdn_cls = _qwen_modeling().Qwen3_5GatedDeltaNet

    teacher_config = _qwen_config(layer_type="linear_attention")
    teacher = qwen_gdn_cls(teacher_config, 0).to(dtype=dtype).eval()
    teacher_shape = GDNShape.from_module(teacher)
    target_config = deepcopy(teacher_config)
    target_config.linear_num_key_heads = target_shape.num_key_heads
    target_config.linear_num_value_heads = target_shape.num_value_heads
    target_config.linear_key_head_dim = target_shape.key_head_dim
    target_config.linear_value_head_dim = target_shape.value_head_dim
    physical = qwen_gdn_cls(target_config, 0).to(dtype=dtype).eval()

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
    assert supports_compact_gated_delta_net(teacher, teacher_shape=teacher_shape)
    assert "forward" not in vars(teacher)

    with torch.no_grad():
        teacher_output = teacher(hidden_states, attention_mask=attention_mask)
        physical_output = physical(hidden_states, attention_mask=attention_mask)
        with compact_gated_delta_net_forward(
            teacher,
            teacher_shape=teacher_shape,
            target_shape=target_shape,
        ):
            assert "forward" in vars(teacher)
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
    assert "forward" not in vars(teacher)


@pytest.mark.parametrize(
    "missing_attribute",
    [
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "conv_kernel_size",
        "activation",
        "layer_idx",
    ],
)
def test_compact_gdn_support_requires_every_forward_attribute(monkeypatch, missing_attribute):
    qwen_gdn_cls = _qwen_modeling().Qwen3_5GatedDeltaNet

    teacher = qwen_gdn_cls(_qwen_config(layer_type="linear_attention"), 0).eval()
    teacher_shape = GDNShape.from_module(teacher)
    monkeypatch.delattr(teacher, missing_attribute)

    assert not supports_compact_gated_delta_net(teacher, teacher_shape=teacher_shape)


def test_compact_gdn_rejects_cache_with_teacher_geometry():
    from transformers.cache_utils import DynamicCache

    qwen_gdn_cls = _qwen_modeling().Qwen3_5GatedDeltaNet

    teacher_config = _qwen_config(layer_type="linear_attention")
    teacher = qwen_gdn_cls(teacher_config, 0).eval()
    teacher_shape = GDNShape.from_module(teacher)
    target_shape = GDNShape(
        num_key_heads=teacher_shape.num_key_heads,
        num_value_heads=teacher_shape.num_value_heads,
        key_head_dim=teacher_shape.key_head_dim // 2,
        value_head_dim=teacher_shape.value_head_dim,
    )
    cache = DynamicCache(config=teacher_config)
    hidden_states = torch.randn(1, 2, teacher_config.hidden_size)
    next_hidden = torch.randn(1, 1, teacher_config.hidden_size)

    with torch.no_grad():
        teacher(hidden_states, cache_params=cache)
        with (
            compact_gated_delta_net_forward(
                teacher,
                teacher_shape=teacher_shape,
                target_shape=target_shape,
            ),
            pytest.raises(ValueError, match="cache convolution width.*target geometry"),
        ):
            teacher(next_hidden, cache_params=cache)
