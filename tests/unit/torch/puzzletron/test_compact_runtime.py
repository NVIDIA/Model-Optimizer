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
import torch.nn.functional as F
from torch import nn

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


class _NativeQwenAttention(nn.Module):
    """CPU-faithful projection and SDPA contract of AutoModel's Qwen attention."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.backend = SimpleNamespace(attn=attn_backend)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, 2 * num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.attn_module = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(batch_size, seq_len, -1, 2 * self.head_dim)
        query, gate = torch.chunk(query, 2, dim=-1)
        key = self.k_proj(hidden_states).reshape(batch_size, seq_len, -1, self.head_dim)
        value = self.v_proj(hidden_states).reshape(batch_size, seq_len, -1, self.head_dim)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(output * torch.sigmoid(gate.reshape(batch_size, seq_len, -1)))


_NativeQwenAttention.__name__ = "Qwen3NextAttention"
_NativeQwenAttention.__qualname__ = "Qwen3NextAttention"
_NativeQwenAttention.__module__ = "nemo_automodel.components.models.qwen3_next.layers"


class _NativeGatedRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return hidden_states * torch.sigmoid(gate) * self.weight


class _NativeSSMGate(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(num_heads))
        self.dt_bias = nn.Parameter(torch.randn(num_heads))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)


class _NativeQwenGatedDeltaNet(nn.Module):
    """Single-device contract of AutoModel's CP-aware Qwen GDN."""

    def __init__(self, shape: GDNShape, *, hidden_size: int = 8) -> None:
        super().__init__()
        self.num_k_heads = shape.num_key_heads
        self.num_v_heads = shape.num_value_heads
        self.head_k_dim = shape.key_head_dim
        self.head_v_dim = shape.value_head_dim
        self.key_dim = shape.num_key_heads * shape.key_head_dim
        self.value_dim = shape.num_value_heads * shape.value_head_dim
        projection_width = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(hidden_size, projection_width)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim)
        self.in_proj_a = nn.Linear(hidden_size, shape.num_value_heads)
        self.in_proj_b = nn.Linear(hidden_size, shape.num_value_heads)
        self.conv1d = nn.Conv1d(
            projection_width,
            projection_width,
            kernel_size=2,
            padding=1,
            groups=projection_width,
        )
        self.norm = _NativeGatedRMSNorm(shape.value_head_dim)
        self.out_proj = nn.Linear(self.value_dim, hidden_size)
        self._fp32_params = _NativeSSMGate(shape.num_value_heads)
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = self._unsupported_cache_update
        self.chunk_gated_delta_rule = self._delta_rule
        self.recurrent_gated_delta_rule = self._delta_rule
        self.conv_kernel_size = 2
        self.activation = "silu"
        self.layer_idx = 0
        self._cp_mesh = None

    @staticmethod
    def _unsupported_cache_update(*args, **kwargs):
        raise AssertionError("cache update is outside this single-device forward contract")

    @staticmethod
    def _delta_rule(query, key, value, *, g, beta, **kwargs):
        del kwargs
        interaction = query.mean(dim=-1, keepdim=True) * key.mean(dim=-1, keepdim=True)
        scale = torch.sigmoid(g).unsqueeze(-1) * beta.unsqueeze(-1)
        return value + interaction * scale, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask=None,
        position_ids=None,
        seq_index=None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, seq_index, kwargs
        if attention_mask is not None and attention_mask.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None]
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = F.silu(self.conv1d(self.in_proj_qkv(hidden_states).transpose(1, 2)))[
            :, :, :seq_len
        ].transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = self._fp32_params(self.in_proj_a(hidden_states))
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
        output, _ = self._delta_rule(query, key, value, g=g, beta=beta)
        output = self.norm(output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(output.reshape(batch_size, seq_len, -1))


_NativeQwenGatedDeltaNet.__name__ = "CPAwareGatedDeltaNet"
_NativeQwenGatedDeltaNet.__qualname__ = "CPAwareGatedDeltaNet"
_NativeQwenGatedDeltaNet.__module__ = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"


def _load_physically_sliced_native_gdn(
    teacher: _NativeQwenGatedDeltaNet,
    teacher_shape: GDNShape,
    target_shape: GDNShape,
) -> _NativeQwenGatedDeltaNet:
    state = {}
    for name, tensor in teacher.state_dict().items():
        if name.startswith("_fp32_params."):
            name = name.removeprefix("_fp32_params.")
        state[f"gdn.{name}"] = tensor.detach().clone()
    slice_gated_delta_net_state_dict(
        state,
        prefix="gdn",
        shape=teacher_shape,
        target=target_shape,
    )
    native_state = {}
    for name, tensor in state.items():
        name = name.removeprefix("gdn.")
        if name in {"A_log", "dt_bias"}:
            name = f"_fp32_params.{name}"
        native_state[name] = tensor
    physical = _NativeQwenGatedDeltaNet(target_shape)
    physical.load_state_dict(native_state)
    return physical


@pytest.mark.parametrize(("target_num_q", "target_num_kv"), [(6, 2), (4, 1), (3, 1)])
def test_native_automodel_attention_matches_physical_mild_geometry(
    target_num_q: int,
    target_num_kv: int,
) -> None:
    torch.manual_seed(7)
    teacher = _NativeQwenAttention(
        num_heads=8,
        num_kv_heads=2,
        head_dim=4,
        hidden_size=8,
    ).eval()
    physical = _NativeQwenAttention(
        num_heads=target_num_q,
        num_kv_heads=target_num_kv,
        head_dim=4,
        hidden_size=8,
    ).eval()
    state = {name: tensor.detach().clone() for name, tensor in teacher.state_dict().items()}
    keep_q, keep_kv = sorted_attention_keep_indices(
        target_num_kv,
        target_num_q // target_num_kv,
        teacher.num_heads // teacher.num_kv_heads,
    )
    q_proj, k_proj, v_proj, o_proj = slice_attention_weights(
        state["q_proj.weight"],
        state["k_proj.weight"],
        state["v_proj.weight"],
        state["o_proj.weight"],
        keep_q,
        keep_kv,
        teacher.head_dim,
    )
    physical.load_state_dict(
        {
            "q_proj.weight": q_proj,
            "k_proj.weight": k_proj,
            "v_proj.weight": v_proj,
            "o_proj.weight": o_proj,
        }
    )
    hidden_states = torch.randn(2, 3, 8)

    target = resolve_compact_grouped_attention_target(
        SimpleNamespace(self_attn=teacher),
        SimpleNamespace(
            no_op=False,
            num_query_heads=8,
            num_kv_heads=2,
            qk_head_dim=4,
        ),
        SimpleNamespace(
            no_op=False,
            num_query_heads=target_num_q,
            num_kv_heads=target_num_kv,
        ),
    )
    assert target is not None
    assert target.pop("module") is teacher
    assert supports_compact_grouped_attention(
        teacher,
        orig_num_q=8,
        orig_num_kv=2,
        head_dim=4,
    )
    with (
        torch.no_grad(),
        compact_grouped_attention_forward(
            teacher,
            **target,
        ),
    ):
        runtime_output = teacher(hidden_states)
    physical_output = physical(hidden_states)

    assert torch.isfinite(runtime_output).all()
    torch.testing.assert_close(runtime_output, physical_output, rtol=0, atol=0)
    assert physical.q_proj.weight.shape == (2 * target_num_q * 4, 8)
    assert physical.k_proj.weight.shape == (target_num_kv * 4, 8)
    assert physical.v_proj.weight.shape == (target_num_kv * 4, 8)
    assert physical.o_proj.weight.shape == (8, target_num_q * 4)


@pytest.mark.parametrize("failure", ["backend", "layout"])
def test_native_automodel_attention_rejects_unqualified_runtime(failure: str) -> None:
    attention = _NativeQwenAttention(
        num_heads=8,
        num_kv_heads=2,
        head_dim=4,
        hidden_size=8,
        attn_backend="te" if failure == "backend" else "sdpa",
    )
    if failure == "layout":
        attention.num_key_value_groups = 1
    layer = SimpleNamespace(self_attn=attention)
    teacher = SimpleNamespace(
        no_op=False,
        num_query_heads=8,
        num_kv_heads=2,
        qk_head_dim=4,
    )
    child = SimpleNamespace(no_op=False, num_query_heads=6, num_kv_heads=2)

    with pytest.raises(RuntimeError, match="refusing to score reduced geometry"):
        resolve_compact_grouped_attention_target(layer, teacher, child)


def test_native_automodel_gdn_matches_physical_mild_coupled_geometry() -> None:
    torch.manual_seed(11)
    teacher_shape = GDNShape(
        num_key_heads=16,
        num_value_heads=16,
        key_head_dim=128,
        value_head_dim=128,
    )
    target_shape = GDNShape(
        num_key_heads=14,
        num_value_heads=14,
        key_head_dim=112,
        value_head_dim=112,
    )
    teacher = _NativeQwenGatedDeltaNet(teacher_shape).eval()
    physical = _load_physically_sliced_native_gdn(teacher, teacher_shape, target_shape).eval()
    hidden_states = torch.randn(2, 2, 8)
    attention_mask = torch.tensor([[1, 1], [1, 0]])

    assert supports_compact_gated_delta_net(teacher, teacher_shape=teacher_shape)
    with (
        torch.no_grad(),
        compact_gated_delta_net_forward(
            teacher,
            teacher_shape=teacher_shape,
            target_shape=target_shape,
        ),
    ):
        runtime_output = teacher(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=torch.arange(2).unsqueeze(0),
            seq_index=torch.arange(2),
        )
    physical_output = physical(hidden_states, attention_mask=attention_mask)

    assert torch.isfinite(runtime_output).all()
    torch.testing.assert_close(runtime_output, physical_output, rtol=1e-6, atol=1e-7)
    assert physical.in_proj_qkv.weight.shape == (4704, 8)
    assert physical.in_proj_z.weight.shape == (1568, 8)
    assert physical.in_proj_a.weight.shape == (14, 8)
    assert physical.in_proj_b.weight.shape == (14, 8)
    assert physical.conv1d.weight.shape == (4704, 1, 2)
    assert physical.norm.weight.shape == (112,)
    assert physical.out_proj.weight.shape == (8, 1568)
    assert physical._fp32_params.A_log.shape == (14,)
    assert physical._fp32_params.dt_bias.shape == (14,)


def test_native_automodel_gdn_rejects_cp_and_packed_runtime() -> None:
    teacher_shape = GDNShape(
        num_key_heads=2,
        num_value_heads=2,
        key_head_dim=4,
        value_head_dim=4,
    )
    target_shape = GDNShape(
        num_key_heads=1,
        num_value_heads=1,
        key_head_dim=4,
        value_head_dim=4,
    )
    teacher = _NativeQwenGatedDeltaNet(teacher_shape)
    teacher._cp_mesh = SimpleNamespace(size=lambda: 2)

    assert not supports_compact_gated_delta_net(teacher, teacher_shape=teacher_shape)
    with (
        pytest.raises(RuntimeError, match="refusing to score reduced geometry"),
        compact_gated_delta_net_forward(
            teacher,
            teacher_shape=teacher_shape,
            target_shape=target_shape,
        ),
    ):
        pass

    teacher._cp_mesh = None
    with (
        compact_gated_delta_net_forward(
            teacher,
            teacher_shape=teacher_shape,
            target_shape=target_shape,
        ),
        pytest.raises(RuntimeError, match="packed execution is not supported"),
    ):
        teacher(
            torch.randn(1, 2, 8),
            cu_seqlens=torch.tensor([0, 2]),
            indices=torch.tensor([0, 1]),
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


def test_compact_grouped_attention_dispatches_native_automodel_sdpa_backend():
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

    assert supports_compact_grouped_attention(
        attention,
        orig_num_q=4,
        orig_num_kv=2,
        head_dim=8,
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
