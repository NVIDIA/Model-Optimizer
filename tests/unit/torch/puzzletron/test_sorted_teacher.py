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

"""CPU tests for the sorted-teacher state-dict transform (pure core, no checkpoint I/O)."""

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MLAConfig,
    MambaConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.pruning.sorted_teacher import build_layer_layouts, sort_state_dict


def _swiglu(gate, up, down, x):
    return (F.silu(x @ gate.t()) * (x @ up.t())) @ down.t()


def _gqa_attn(q, k, v, o, x, num_q, num_kv, head_dim):
    t = x.shape[0]
    qh, kh, vh = (
        (x @ q.t()).view(t, num_q, head_dim),
        (x @ k.t()).view(t, num_kv, head_dim),
        (x @ v.t()).view(t, num_kv, head_dim),
    )
    n_in = num_q // num_kv
    out = torch.empty(t, num_q, head_dim, dtype=x.dtype)
    for h in range(num_q):
        p = torch.softmax(qh[:, h] @ kh[:, h // n_in].t() / head_dim**0.5, dim=-1)
        out[:, h] = p @ vh[:, h // n_in]
    return out.reshape(t, num_q * head_dim) @ o.t()


def _rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def _mla_projections(sd, prefix, x, q_rank, kv_rank, rope_dim):
    q = _rms_norm(x @ sd[f"{prefix}.q_a_proj.weight"].t(), sd[f"{prefix}.q_a_layernorm.weight"])
    q = q @ sd[f"{prefix}.q_b_proj.weight"].t()
    compressed = x @ sd[f"{prefix}.kv_a_proj_with_mqa.weight"].t()
    kv, rope = compressed.split((kv_rank, rope_dim), dim=-1)
    kv = _rms_norm(kv, sd[f"{prefix}.kv_a_layernorm.weight"])
    kv = kv @ sd[f"{prefix}.kv_b_proj.weight"].t()
    return q, kv, rope


def test_build_layer_layouts_keys_and_skip_mamba():
    block_configs = [
        BlockConfig(
            subblock_configs=(
                FFNConfig(intermediate_size=12),
                AttentionConfig(num_query_heads=4, num_kv_heads=2),
            )
        ),
        # mamba/no-op-attention layer: attention skipped, ffn still prunable
        BlockConfig(
            subblock_configs=(
                FFNConfig(intermediate_size=12),
                MambaConfig(num_heads=4, head_dim=4),
            )
        ),
    ]
    layouts = build_layer_layouts(
        block_configs,
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=4,
        head_dim=4,
    )
    assert layouts[0].q_key == "model.layers.0.self_attn.q_proj.weight"
    assert layouts[0].down_key == "model.layers.0.mlp.down_proj.weight"
    assert layouts[0].num_kv_heads == 2 and layouts[0].num_q_heads == 4
    # Layer 1: mamba attention -> no attn keys, but ffn keys present.
    assert layouts[1].o_key is None and layouts[1].num_kv_heads is None
    assert layouts[1].down_key == "model.layers.1.mlp.down_proj.weight"


def test_build_layer_layouts_uses_per_layer_attention_head_dimension():
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(
                        num_query_heads=4,
                        num_kv_heads=2,
                        qk_head_dim=8,
                    ),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=4,
        head_dim=4,
    )

    assert layouts[0].head_dim == 8


def test_sort_state_dict_is_output_invariant():
    torch.manual_seed(0)
    num_q, num_kv, head_dim, hidden, inter = 6, 3, 4, 8, 12
    p = "model.layers.0"
    sd = {
        f"{p}.mlp.gate_proj.weight": torch.randn(inter, hidden, dtype=torch.float64),
        f"{p}.mlp.up_proj.weight": torch.randn(inter, hidden, dtype=torch.float64),
        f"{p}.mlp.down_proj.weight": torch.randn(hidden, inter, dtype=torch.float64),
        f"{p}.self_attn.q_proj.weight": torch.randn(num_q * head_dim, hidden, dtype=torch.float64),
        f"{p}.self_attn.k_proj.weight": torch.randn(num_kv * head_dim, hidden, dtype=torch.float64),
        f"{p}.self_attn.v_proj.weight": torch.randn(num_kv * head_dim, hidden, dtype=torch.float64),
        f"{p}.self_attn.o_proj.weight": torch.randn(hidden, num_q * head_dim, dtype=torch.float64),
    }
    block_configs = [
        BlockConfig(
            subblock_configs=(
                FFNConfig(intermediate_size=inter),
                AttentionConfig(num_query_heads=num_q, num_kv_heads=num_kv),
            )
        )
    ]
    layouts = build_layer_layouts(
        block_configs, layer_prefix_tmpl="model.layers.{i}", num_attention_heads=num_q, head_dim=head_dim
    )
    x = torch.randn(5, hidden, dtype=torch.float64)
    ffn_ref = _swiglu(sd[f"{p}.mlp.gate_proj.weight"], sd[f"{p}.mlp.up_proj.weight"], sd[f"{p}.mlp.down_proj.weight"], x)
    attn_ref = _gqa_attn(
        sd[f"{p}.self_attn.q_proj.weight"], sd[f"{p}.self_attn.k_proj.weight"],
        sd[f"{p}.self_attn.v_proj.weight"], sd[f"{p}.self_attn.o_proj.weight"], x, num_q, num_kv, head_dim,
    )

    sorted_sd, perms = sort_state_dict(
        sd,
        layouts,
        ffn_scores={0: torch.rand(inter)},
        attention_logs={
            0: {
                "kv_group_scores": torch.rand(num_kv),
                "query_head_scores": torch.rand(num_kv, num_q // num_kv),
            }
        },
    )

    # Sorted weights differ but reproduce the same outputs (permutation invariance).
    assert not torch.equal(sorted_sd[f"{p}.mlp.gate_proj.weight"], sd[f"{p}.mlp.gate_proj.weight"])
    ffn_got = _swiglu(
        sorted_sd[f"{p}.mlp.gate_proj.weight"], sorted_sd[f"{p}.mlp.up_proj.weight"],
        sorted_sd[f"{p}.mlp.down_proj.weight"], x,
    )
    attn_got = _gqa_attn(
        sorted_sd[f"{p}.self_attn.q_proj.weight"], sorted_sd[f"{p}.self_attn.k_proj.weight"],
        sorted_sd[f"{p}.self_attn.v_proj.weight"], sorted_sd[f"{p}.self_attn.o_proj.weight"], x, num_q, num_kv, head_dim,
    )
    assert torch.allclose(ffn_ref, ffn_got, atol=1e-10)
    assert torch.allclose(attn_ref, attn_got, atol=1e-10)
    assert "ffn.0" in perms and "attn.q.0" in perms and "attn.kv.0" in perms


def test_sort_state_dict_permutes_descriptor_declared_query_head_auxiliary_tensors():
    num_q, num_kv, head_dim, hidden = 4, 2, 2, 8
    prefix = "model.layers.0.self_attn"
    sinks = torch.arange(num_q, dtype=torch.float64)
    state = {
        f"{prefix}.q_proj.weight": torch.randn(num_q * head_dim, hidden),
        f"{prefix}.k_proj.weight": torch.randn(num_kv * head_dim, hidden),
        f"{prefix}.v_proj.weight": torch.randn(num_kv * head_dim, hidden),
        f"{prefix}.o_proj.weight": torch.randn(hidden, num_q * head_dim),
        f"{prefix}.sinks": sinks,
    }
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(num_query_heads=num_q, num_kv_heads=num_kv),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=num_q,
        head_dim=head_dim,
        attention_q_head_subnames=("sinks",),
    )
    query_scores = torch.tensor([[0.1, 0.8], [0.9, 0.2]])
    kv_scores = torch.tensor([0.2, 0.7])

    sorted_state, permutations = sort_state_dict(
        state,
        layouts,
        ffn_scores={},
        attention_logs={
            0: {
                "kv_group_scores": kv_scores,
                "query_head_scores": query_scores,
            }
        },
    )

    query_order = permutations["attn.q.0"]
    torch.testing.assert_close(sorted_state[f"{prefix}.sinks"], sinks[query_order])


def test_sort_state_dict_permutes_fused_expert_axis_with_router_order():
    num_experts = 3
    block = BlockConfig(
        subblock_configs=(MoEConfig(num_experts=num_experts, expert_intermediate_size=4),)
    )
    layouts = build_layer_layouts(
        [block],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=2,
        moe_router_subname="gate",
        moe_router_aux_subnames=("gate.bias",),
        moe_fused_expert_subnames=(
            "experts.gate_up_proj",
            "experts.down_proj",
        ),
    )
    prefix = "model.layers.0.mlp"
    state = {
        f"{prefix}.gate.weight": torch.arange(num_experts * 4).reshape(num_experts, 4),
        f"{prefix}.gate.bias": torch.arange(num_experts),
        f"{prefix}.experts.gate_up_proj": torch.arange(num_experts)[:, None, None].expand(-1, 8, 4).clone(),
        f"{prefix}.experts.down_proj": torch.arange(num_experts)[:, None, None].expand(-1, 4, 4).clone(),
    }
    scores = torch.tensor([0.2, 0.9, 0.4])

    sorted_state, permutations = sort_state_dict(
        state,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={f"{prefix}.gate": {"score": scores}},
    )

    order = torch.tensor([1, 2, 0])
    torch.testing.assert_close(permutations["moe.experts.0"], order)
    for key in state:
        torch.testing.assert_close(sorted_state[key], state[key].index_select(0, order))


def test_sort_state_dict_can_record_expert_order_without_reindexing_runtime_state():
    num_experts = 3
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    MoEConfig(num_experts=num_experts, expert_intermediate_size=4),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=2,
        moe_fused_expert_subnames=("experts.gate_up",),
        moe_expert_order_mode="metadata_only",
    )
    prefix = "model.layers.0.mlp"
    state = {
        f"{prefix}.gate.weight": torch.randn(num_experts, 4),
        f"{prefix}.experts.gate_up": torch.randn(num_experts, 8, 4),
    }

    sorted_state, permutations = sort_state_dict(
        state,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={f"{prefix}.gate": {"score": torch.tensor([0.2, 0.9, 0.4])}},
    )

    torch.testing.assert_close(permutations["moe.experts.0"], torch.tensor([1, 2, 0]))
    for key, value in state.items():
        torch.testing.assert_close(sorted_state[key], value)


def test_sort_state_dict_does_not_treat_non_latent_nemotron_experts_as_latent() -> None:
    num_experts, hidden, intermediate = 2, 4, 3
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    MoEConfig(
                        num_experts=num_experts,
                        expert_intermediate_size=intermediate,
                        latent_dim=None,
                    ),
                )
            )
        ],
        layer_prefix_tmpl="backbone.layers.{i}",
        num_attention_heads=2,
        head_dim=2,
        moe_module="mixer",
    )
    prefix = "backbone.layers.0.mixer"
    state = {
        f"{prefix}.experts.0.up_proj.weight": torch.randn(intermediate, hidden),
        f"{prefix}.experts.0.down_proj.weight": torch.randn(hidden, intermediate),
    }

    sorted_state, permutations = sort_state_dict(
        state,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={prefix: {"format_version": 3, "score": torch.ones(num_experts)}},
    )

    assert layouts[0].moe_latent_dim is None
    assert layouts[0].moe_fc1_latent_key is None
    assert layouts[0].moe_fc2_latent_key is None
    assert not any(key.startswith("moe.latent") for key in permutations)
    for key, value in state.items():
        torch.testing.assert_close(sorted_state[key], value)


@pytest.mark.parametrize("group_size", [1, 2])
def test_sort_state_dict_permutes_fused_expert_intermediate_channels(group_size):
    num_experts, intermediate, hidden = 2, 4, 3
    block = BlockConfig(
        subblock_configs=(
            MoEConfig(num_experts=num_experts, expert_intermediate_size=intermediate),
        )
    )
    layouts = build_layer_layouts(
        [block],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=2,
        moe_fused_expert_subnames=("experts.gate_up", "experts.down"),
        moe_fused_gate_up_subnames=("experts.gate_up",),
        moe_fused_down_subnames=("experts.down",),
        moe_expert_intermediate_group_size=group_size,
    )
    prefix = "model.layers.0.mlp"
    gate_up = torch.arange(num_experts * 2 * intermediate * hidden).reshape(
        num_experts, 2 * intermediate, hidden
    )
    down = torch.arange(
        num_experts * hidden * (intermediate // group_size)
    ).reshape(num_experts, hidden, intermediate // group_size)
    state = {
        f"{prefix}.experts.gate_up": gate_up,
        f"{prefix}.experts.down": down,
    }
    scores = {
        0: torch.tensor([0.1, 0.2, 0.9, 0.8]),
        1: torch.tensor([0.8, 0.7, 0.1, 0.2]),
    }

    sorted_state, permutations = sort_state_dict(
        state,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={
            f"{prefix}.experts": {
                "expert_stats_dict": {
                    expert: {"score": score} for expert, score in scores.items()
                }
            }
        },
    )

    expected_permutations = (
        {
            0: torch.tensor([2, 3, 1, 0]),
            1: torch.tensor([0, 1, 3, 2]),
        }
        if group_size == 1
        else {
            0: torch.tensor([2, 3, 0, 1]),
            1: torch.tensor([0, 1, 2, 3]),
        }
    )
    for expert, permutation in expected_permutations.items():
        torch.testing.assert_close(
            permutations[f"moe.expert_intermediate.0.{expert}"], permutation
        )
        gated = torch.cat((permutation, permutation + intermediate))
        torch.testing.assert_close(
            sorted_state[f"{prefix}.experts.gate_up"][expert],
            gate_up[expert].index_select(0, gated),
        )
        group_permutation = permutation.reshape(-1, group_size)[:, 0] // group_size
        torch.testing.assert_close(
            sorted_state[f"{prefix}.experts.down"][expert],
            down[expert].index_select(1, group_permutation),
        )


def test_sort_state_dict_applies_each_expert_channel_order_before_expert_order() -> None:
    num_experts, intermediate, hidden = 3, 4, 2
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    MoEConfig(
                        num_experts=num_experts,
                        expert_intermediate_size=intermediate,
                    ),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=2,
        moe_router_subname="gate",
        moe_fused_expert_subnames=("experts.gate_up", "experts.down"),
        moe_fused_gate_up_subnames=("experts.gate_up",),
        moe_fused_down_subnames=("experts.down",),
    )
    prefix = "model.layers.0.mlp"
    gate_up = torch.arange(num_experts * 2 * intermediate * hidden).reshape(
        num_experts, 2 * intermediate, hidden
    )
    down = torch.arange(num_experts * hidden * intermediate).reshape(
        num_experts, hidden, intermediate
    )
    state = {
        f"{prefix}.gate.weight": torch.arange(num_experts * hidden).reshape(
            num_experts, hidden
        ),
        f"{prefix}.experts.gate_up": gate_up,
        f"{prefix}.experts.down": down,
    }
    channel_scores = {
        0: torch.tensor([0.1, 0.4, 0.9, 0.2]),
        1: torch.tensor([0.8, 0.1, 0.3, 0.7]),
        2: torch.tensor([0.2, 0.9, 0.1, 0.6]),
    }
    expert_scores = torch.tensor([0.2, 0.9, 0.4])

    sorted_state, permutations = sort_state_dict(
        state,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={
            f"{prefix}.gate": {"score": expert_scores},
            f"{prefix}.experts": {
                "expert_stats_dict": {
                    expert: {"score": score}
                    for expert, score in channel_scores.items()
                }
            },
        },
    )

    expert_order = torch.argsort(expert_scores, descending=True)
    torch.testing.assert_close(permutations["moe.experts.0"], expert_order)
    for new_expert, old_expert in enumerate(expert_order.tolist()):
        channel_order = torch.argsort(channel_scores[old_expert], descending=True)
        gated_order = torch.cat((channel_order, channel_order + intermediate))
        torch.testing.assert_close(
            sorted_state[f"{prefix}.experts.gate_up"][new_expert],
            gate_up[old_expert].index_select(0, gated_order),
        )
        torch.testing.assert_close(
            sorted_state[f"{prefix}.experts.down"][new_expert],
            down[old_expert].index_select(1, channel_order),
        )


def test_sort_state_dict_composes_mamba_channels_with_groupwise_head_order() -> None:
    num_heads, head_dim, num_groups, state_dim, hidden = 4, 3, 2, 2, 5
    inner = num_heads * head_dim
    state_width = num_groups * state_dim
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    MambaConfig(
                        num_heads=num_heads,
                        head_dim=head_dim,
                        num_groups=num_groups,
                        state_dim=state_dim,
                    ),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=2,
        mamba_module="mixer",
    )
    prefix = "model.layers.0.mixer"
    in_rows = 2 * inner + 2 * state_width + num_heads
    state = {
        f"{prefix}.in_proj.weight": torch.arange(in_rows * hidden).reshape(in_rows, hidden),
        f"{prefix}.out_proj.weight": torch.arange(hidden * inner).reshape(hidden, inner),
        f"{prefix}.conv1d.weight": torch.arange((inner + 2 * state_width) * 2).reshape(
            inner + 2 * state_width, 2
        ),
        f"{prefix}.A_log": torch.arange(num_heads),
        f"{prefix}.D": torch.arange(num_heads) + 10,
        f"{prefix}.dt_bias": torch.arange(num_heads) + 20,
        f"{prefix}.norm.weight": torch.arange(inner),
    }
    head_scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
    channel_scores = torch.tensor(
        [
            [0.2, 0.9, 0.1],
            [0.8, 0.1, 0.7],
            [0.3, 0.2, 0.9],
            [0.6, 0.8, 0.1],
        ]
    )

    sorted_state, permutations = sort_state_dict(
        state,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={
            f"{prefix}.in_proj": {
                "mamba_head_scores": head_scores,
                "mamba_head_dim_scores": channel_scores,
            }
        },
    )

    head_order = permutations["mamba.heads.0"]
    heads_per_group = num_heads // num_groups
    torch.testing.assert_close(
        head_order // heads_per_group,
        torch.arange(num_heads) // heads_per_group,
    )
    within_head_order = torch.argsort(channel_scores[head_order], dim=-1, descending=True)
    combined_order = (
        head_order[:, None] * head_dim + within_head_order
    ).reshape(-1)
    torch.testing.assert_close(
        sorted_state[f"{prefix}.in_proj.weight"][:inner],
        state[f"{prefix}.in_proj.weight"][:inner].index_select(0, combined_order),
    )
    torch.testing.assert_close(
        sorted_state[f"{prefix}.in_proj.weight"][inner : 2 * inner],
        state[f"{prefix}.in_proj.weight"][inner : 2 * inner].index_select(
            0, combined_order
        ),
    )
    torch.testing.assert_close(
        sorted_state[f"{prefix}.out_proj.weight"],
        state[f"{prefix}.out_proj.weight"].index_select(1, combined_order),
    )
    torch.testing.assert_close(
        sorted_state[f"{prefix}.norm.weight"],
        state[f"{prefix}.norm.weight"].index_select(0, combined_order),
    )
    torch.testing.assert_close(
        sorted_state[f"{prefix}.conv1d.weight"][:inner],
        state[f"{prefix}.conv1d.weight"][:inner].index_select(0, combined_order),
    )
    dt_start = 2 * inner + 2 * state_width
    torch.testing.assert_close(
        sorted_state[f"{prefix}.in_proj.weight"][dt_start:],
        state[f"{prefix}.in_proj.weight"][dt_start:].index_select(0, head_order),
    )
    for suffix in ("A_log", "D", "dt_bias"):
        torch.testing.assert_close(
            sorted_state[f"{prefix}.{suffix}"],
            state[f"{prefix}.{suffix}"].index_select(0, head_order),
        )


def test_sort_state_dict_preserves_interleaved_gate_up_pairs():
    intermediate, hidden = 4, 3
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    MoEConfig(num_experts=1, expert_intermediate_size=intermediate),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=2,
        moe_fused_expert_subnames=("experts.gate_up", "experts.down"),
        moe_fused_gate_up_subnames=("experts.gate_up",),
        moe_fused_down_subnames=("experts.down",),
        moe_expert_intermediate_group_size=2,
        moe_fused_gate_layout="interleaved",
    )
    prefix = "model.layers.0.mlp"
    gate_up = torch.arange(2 * intermediate * hidden).reshape(
        1, 2 * intermediate, hidden
    )
    down = torch.arange(hidden * 2).reshape(1, hidden, 2)

    sorted_state, permutations = sort_state_dict(
        {
            f"{prefix}.experts.gate_up": gate_up,
            f"{prefix}.experts.down": down,
        },
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={
            f"{prefix}.experts": {
                "expert_stats_dict": {
                    0: {"score": torch.tensor([0.1, 0.2, 0.9, 0.8])}
                }
            }
        },
    )

    permutation = permutations["moe.expert_intermediate.0.0"]
    pair_indices = torch.stack((2 * permutation, 2 * permutation + 1), dim=1).reshape(-1)
    torch.testing.assert_close(
        sorted_state[f"{prefix}.experts.gate_up"][0],
        gate_up[0].index_select(0, pair_indices),
    )


def test_sort_state_dict_mla_latent_ranks_are_output_invariant() -> None:
    torch.manual_seed(7)
    hidden, q_rank, kv_rank, rope_dim = 5, 4, 3, 2
    prefix = "model.layers.0.self_attn"
    sd = {
        f"{prefix}.q_a_proj.weight": torch.randn(q_rank, hidden, dtype=torch.float64),
        f"{prefix}.q_a_layernorm.weight": torch.randn(q_rank, dtype=torch.float64),
        f"{prefix}.q_b_proj.weight": torch.randn(6, q_rank, dtype=torch.float64),
        f"{prefix}.kv_a_proj_with_mqa.weight": torch.randn(kv_rank + rope_dim, hidden, dtype=torch.float64),
        f"{prefix}.kv_a_layernorm.weight": torch.randn(kv_rank, dtype=torch.float64),
        f"{prefix}.kv_b_proj.weight": torch.randn(7, kv_rank, dtype=torch.float64),
    }
    layouts = build_layer_layouts(
        [BlockConfig(subblock_configs=(MLAConfig(q_lora_rank=q_rank, kv_lora_rank=kv_rank),))],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=3,
    )
    x = torch.randn(8, hidden, dtype=torch.float64)
    expected = _mla_projections(sd, prefix, x, q_rank, kv_rank, rope_dim)

    sorted_sd, perms = sort_state_dict(
        sd,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={
            f"{prefix}.q_a_layernorm": {"q_lora_rank_score": torch.tensor([0.1, 0.8, 0.3, 0.5])},
            f"{prefix}.kv_a_layernorm": {"kv_lora_rank_score": torch.tensor([0.2, 0.9, 0.4])},
        },
    )
    actual = _mla_projections(sorted_sd, prefix, x, q_rank, kv_rank, rope_dim)

    assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(actual, expected))
    assert set(perms) >= {"mla.q_lora.0", "mla.kv_lora.0"}


def test_sort_state_dict_mla_heads_permutes_q_decoded_kv_and_output_together() -> None:
    num_heads, qk_dim, decoded_kv_dim, v_dim = 3, 4, 5, 2
    q_rank, kv_rank, hidden = 2, 2, 7
    prefix = "model.layers.0.self_attn"
    sd = {
        f"{prefix}.q_b_proj.weight": torch.arange(
            num_heads * qk_dim * q_rank, dtype=torch.float32
        ).reshape(num_heads * qk_dim, q_rank),
        f"{prefix}.kv_b_proj.weight": torch.arange(
            num_heads * decoded_kv_dim * kv_rank, dtype=torch.float32
        ).reshape(num_heads * decoded_kv_dim, kv_rank),
        f"{prefix}.o_proj.weight": torch.arange(
            hidden * num_heads * v_dim, dtype=torch.float32
        ).reshape(hidden, num_heads * v_dim),
    }
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    MLAConfig(num_heads=num_heads, q_lora_rank=q_rank, kv_lora_rank=kv_rank),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=num_heads,
        head_dim=qk_dim,
    )

    sorted_sd, perms = sort_state_dict(
        sd,
        layouts,
        ffn_scores={},
        attention_logs={},
        score_logs={
            f"{prefix}.o_proj": {"kv_group_scores": torch.tensor([0.2, 0.9, 0.5])}
        },
    )
    permutation = torch.tensor([1, 2, 0])

    assert torch.equal(
        sorted_sd[f"{prefix}.q_b_proj.weight"],
        sd[f"{prefix}.q_b_proj.weight"].view(num_heads, qk_dim, q_rank)[
            permutation
        ].reshape(num_heads * qk_dim, q_rank),
    )
    assert torch.equal(
        sorted_sd[f"{prefix}.kv_b_proj.weight"],
        sd[f"{prefix}.kv_b_proj.weight"].view(
            num_heads, decoded_kv_dim, kv_rank
        )[permutation].reshape(num_heads * decoded_kv_dim, kv_rank),
    )
    assert torch.equal(
        sorted_sd[f"{prefix}.o_proj.weight"],
        sd[f"{prefix}.o_proj.weight"].view(hidden, num_heads, v_dim)[:, permutation].reshape(
            hidden, num_heads * v_dim
        ),
    )
    assert torch.equal(perms["mla.heads.0"], permutation)


def test_sort_state_dict_accepts_per_kv_attention_score():
    """independent_kv_head_contribution emits a [num_kv] score; sorting must still be invariant.

    This is the common-recipe case that previously crashed (the o_proj score was assumed to be
    per-query [num_q]).
    """
    torch.manual_seed(1)
    num_q, num_kv, head_dim, hidden, inter = 6, 3, 4, 8, 12
    p = "model.layers.0"
    sd = {
        f"{p}.mlp.gate_proj.weight": torch.randn(inter, hidden, dtype=torch.float64),
        f"{p}.mlp.up_proj.weight": torch.randn(inter, hidden, dtype=torch.float64),
        f"{p}.mlp.down_proj.weight": torch.randn(hidden, inter, dtype=torch.float64),
        f"{p}.self_attn.q_proj.weight": torch.randn(num_q * head_dim, hidden, dtype=torch.float64),
        f"{p}.self_attn.k_proj.weight": torch.randn(num_kv * head_dim, hidden, dtype=torch.float64),
        f"{p}.self_attn.v_proj.weight": torch.randn(num_kv * head_dim, hidden, dtype=torch.float64),
        f"{p}.self_attn.o_proj.weight": torch.randn(hidden, num_q * head_dim, dtype=torch.float64),
    }
    block_configs = [
        BlockConfig(
            subblock_configs=(
                FFNConfig(intermediate_size=inter),
                AttentionConfig(num_query_heads=num_q, num_kv_heads=num_kv),
            )
        )
    ]
    layouts = build_layer_layouts(
        block_configs, layer_prefix_tmpl="model.layers.{i}", num_attention_heads=num_q, head_dim=head_dim
    )
    x = torch.randn(5, hidden, dtype=torch.float64)
    attn_ref = _gqa_attn(
        sd[f"{p}.self_attn.q_proj.weight"], sd[f"{p}.self_attn.k_proj.weight"],
        sd[f"{p}.self_attn.v_proj.weight"], sd[f"{p}.self_attn.o_proj.weight"], x, num_q, num_kv, head_dim,
    )
    # Per-KV score (length num_kv, not num_q).
    sorted_sd, perms = sort_state_dict(
        sd,
        layouts,
        ffn_scores={},
        attention_logs={0: {"kv_group_scores": torch.rand(num_kv)}},
    )
    attn_got = _gqa_attn(
        sorted_sd[f"{p}.self_attn.q_proj.weight"], sorted_sd[f"{p}.self_attn.k_proj.weight"],
        sorted_sd[f"{p}.self_attn.v_proj.weight"], sorted_sd[f"{p}.self_attn.o_proj.weight"], x, num_q, num_kv, head_dim,
    )
    assert torch.allclose(attn_ref, attn_got, atol=1e-10)
    assert "attn.kv.0" in perms
