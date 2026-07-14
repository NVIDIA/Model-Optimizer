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

"""CPU tests for materialize-from-sorted-teacher (slice/merge -> smaller real weights)."""

import torch
import torch.nn.functional as F

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MLAConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.pruning.materialize import (
    BlockTarget,
    block_targets_from_replacements,
    materialize_solution_state_dict,
)
from modelopt.torch.puzzletron.pruning.sorted_teacher import build_layer_layouts


def test_block_targets_from_replacements():
    teacher = [
        BlockConfig(
            subblock_configs=(
                FFNConfig(intermediate_size=12),
                AttentionConfig(num_query_heads=24, num_kv_heads=4),
            )
        )
    ]
    # FFN-only child (attention None -> no attn target).
    reps_ffn = [{
        "parent_layer_indices": [0],
        "child_block_configs": [
            BlockConfig(subblock_configs=(FFNConfig(intermediate_size=8),))
        ],
    }]
    t = block_targets_from_replacements(reps_ffn, teacher, num_attention_heads=24)
    assert t[0].target_intermediate == 8 and t[0].target_num_kv is None

    # Attention removal child: (q=18, kv=3); FFN None.
    reps_attn = [{
        "parent_layer_indices": [0],
        "child_block_configs": [
            BlockConfig(
                subblock_configs=(AttentionConfig(num_query_heads=18, num_kv_heads=3),)
            )
        ],
    }]
    t = block_targets_from_replacements(reps_attn, teacher, num_attention_heads=24)
    assert (t[0].target_num_q, t[0].target_num_kv) == (18, 3)

    # A KV-only reduction removes the corresponding sorted query-head groups.
    reps_merge = [{
        "parent_layer_indices": [0],
        "child_block_configs": [
            BlockConfig(subblock_configs=(AttentionConfig(num_kv_heads=2),))
        ],
    }]
    t = block_targets_from_replacements(reps_merge, teacher, num_attention_heads=24)
    assert (t[0].target_num_q, t[0].target_num_kv) == (12, 2)


P = "model.layers.0"
HIDDEN, INTER, NUM_Q, NUM_KV, HEAD_DIM = 8, 12, 8, 4, 4


def _state():
    torch.manual_seed(0)
    return {
        f"{P}.mlp.gate_proj.weight": torch.randn(INTER, HIDDEN, dtype=torch.float64),
        f"{P}.mlp.up_proj.weight": torch.randn(INTER, HIDDEN, dtype=torch.float64),
        f"{P}.mlp.down_proj.weight": torch.randn(HIDDEN, INTER, dtype=torch.float64),
        f"{P}.self_attn.q_proj.weight": torch.randn(NUM_Q * HEAD_DIM, HIDDEN, dtype=torch.float64),
        f"{P}.self_attn.k_proj.weight": torch.randn(NUM_KV * HEAD_DIM, HIDDEN, dtype=torch.float64),
        f"{P}.self_attn.v_proj.weight": torch.randn(NUM_KV * HEAD_DIM, HIDDEN, dtype=torch.float64),
        f"{P}.self_attn.o_proj.weight": torch.randn(HIDDEN, NUM_Q * HEAD_DIM, dtype=torch.float64),
    }


def _layouts():
    bc = [
        BlockConfig(
            subblock_configs=(
                FFNConfig(intermediate_size=INTER),
                AttentionConfig(num_query_heads=NUM_Q, num_kv_heads=NUM_KV),
            )
        )
    ]
    return build_layer_layouts(
        bc, layer_prefix_tmpl="model.layers.{i}", num_attention_heads=NUM_Q, head_dim=HEAD_DIM
    )


def test_materialize_ffn_matches_masked_forward():
    sd, layouts = _state(), _layouts()
    keep = 5
    out = materialize_solution_state_dict(sd, layouts, {0: BlockTarget(target_intermediate=keep)})

    g, u, d = out[f"{P}.mlp.gate_proj.weight"], out[f"{P}.mlp.up_proj.weight"], out[f"{P}.mlp.down_proj.weight"]
    assert g.shape == (keep, HIDDEN) and d.shape == (HIDDEN, keep)
    x = torch.randn(3, HIDDEN, dtype=torch.float64)
    materialized = (F.silu(x @ g.t()) * (x @ u.t())) @ d.t()
    # Masked (score-time) forward on the full sorted weights, keeping prefix [:keep].
    fg, fu, fd = sd[f"{P}.mlp.gate_proj.weight"], sd[f"{P}.mlp.up_proj.weight"], sd[f"{P}.mlp.down_proj.weight"]
    h = F.silu(x @ fg.t()) * (x @ fu.t())
    h[:, keep:] = 0
    masked = h @ fd.t()
    assert torch.allclose(materialized, masked, atol=1e-10)


def test_materialize_attention_removal_and_merge_shapes():
    layouts = _layouts()
    # Removal to (6 q, 3 kv): drop 1 group, keep 2 q/group (m = 6//3).
    rem = materialize_solution_state_dict(_state(), layouts, {0: BlockTarget(target_num_q=6, target_num_kv=3)})
    assert rem[f"{P}.self_attn.q_proj.weight"].shape == (6 * HEAD_DIM, HIDDEN)
    assert rem[f"{P}.self_attn.k_proj.weight"].shape == (3 * HEAD_DIM, HIDDEN)
    assert rem[f"{P}.self_attn.o_proj.weight"].shape == (HIDDEN, 6 * HEAD_DIM)

    # KV-group reduction to (4 q, 2 kv) preserves two query heads per group.
    mg = materialize_solution_state_dict(_state(), layouts, {0: BlockTarget(target_num_q=4, target_num_kv=2)})
    assert mg[f"{P}.self_attn.q_proj.weight"].shape == (4 * HEAD_DIM, HIDDEN)
    assert mg[f"{P}.self_attn.k_proj.weight"].shape == (2 * HEAD_DIM, HIDDEN)
    assert mg[f"{P}.self_attn.o_proj.weight"].shape == (HIDDEN, 4 * HEAD_DIM)


def test_materialize_mla_rank_prefixes_keep_rope_rows() -> None:
    q_rank, kv_rank, rope_dim = 4, 3, 2
    prefix = f"{P}.self_attn"
    state = {
        f"{prefix}.q_a_proj.weight": torch.randn(q_rank, HIDDEN),
        f"{prefix}.q_a_proj.bias": torch.randn(q_rank),
        f"{prefix}.q_a_layernorm.weight": torch.randn(q_rank),
        f"{prefix}.q_b_proj.weight": torch.randn(12, q_rank),
        f"{prefix}.kv_a_proj_with_mqa.weight": torch.randn(kv_rank + rope_dim, HIDDEN),
        f"{prefix}.kv_a_proj_with_mqa.bias": torch.randn(kv_rank + rope_dim),
        f"{prefix}.kv_a_layernorm.weight": torch.randn(kv_rank),
        f"{prefix}.kv_b_proj.weight": torch.randn(12, kv_rank),
    }
    layouts = build_layer_layouts(
        [BlockConfig(subblock_configs=(MLAConfig(q_lora_rank=q_rank, kv_lora_rank=kv_rank),))],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=4,
    )

    result = materialize_solution_state_dict(
        state,
        layouts,
        {0: BlockTarget(target_q_lora_rank=2, target_kv_lora_rank=1)},
    )

    assert result[f"{prefix}.q_a_proj.weight"].shape == (2, HIDDEN)
    assert result[f"{prefix}.q_a_layernorm.weight"].shape == (2,)
    assert result[f"{prefix}.q_b_proj.weight"].shape == (12, 2)
    assert result[f"{prefix}.kv_a_proj_with_mqa.weight"].shape == (1 + rope_dim, HIDDEN)
    assert torch.equal(
        result[f"{prefix}.kv_a_proj_with_mqa.weight"][-rope_dim:],
        state[f"{prefix}.kv_a_proj_with_mqa.weight"][-rope_dim:],
    )
    assert result[f"{prefix}.kv_a_layernorm.weight"].shape == (1,)
    assert result[f"{prefix}.kv_b_proj.weight"].shape == (12, 1)


def test_materialize_mla_head_prefix_slices_all_coupled_head_axes() -> None:
    num_heads, qk_dim, decoded_kv_dim, v_dim = 4, 6, 5, 3
    q_rank, kv_rank = 8, 7
    prefix = f"{P}.self_attn"
    state = {
        f"{prefix}.q_b_proj.weight": torch.randn(num_heads * qk_dim, q_rank),
        f"{prefix}.kv_b_proj.weight": torch.randn(
            num_heads * decoded_kv_dim, kv_rank
        ),
        f"{prefix}.o_proj.weight": torch.randn(HIDDEN, num_heads * v_dim),
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

    result = materialize_solution_state_dict(
        state,
        layouts,
        {0: BlockTarget(target_mla_heads=2)},
    )

    assert result[f"{prefix}.q_b_proj.weight"].shape == (2 * qk_dim, q_rank)
    assert result[f"{prefix}.kv_b_proj.weight"].shape == (
        2 * decoded_kv_dim,
        kv_rank,
    )
    assert result[f"{prefix}.o_proj.weight"].shape == (HIDDEN, 2 * v_dim)


def test_materialize_fused_expert_count_and_grouped_intermediate_prefixes() -> None:
    experts, intermediate, hidden, group_size = 4, 8, 6, 2
    prefix = f"{P}.mlp"
    state = {
        f"{prefix}.router.weight": torch.randn(experts, hidden),
        f"{prefix}.router.bias": torch.randn(experts),
        f"{prefix}.experts.gate_up": torch.randn(experts, 2 * intermediate, hidden),
        f"{prefix}.experts.down": torch.randn(
            experts, hidden, intermediate // group_size
        ),
        f"{prefix}.experts.down_bias": torch.randn(experts, hidden),
    }
    layouts = build_layer_layouts(
        [
            BlockConfig(
                subblock_configs=(
                    MoEConfig(
                        num_experts=experts,
                        expert_intermediate_size=intermediate,
                    ),
                )
            )
        ],
        layer_prefix_tmpl="model.layers.{i}",
        num_attention_heads=2,
        head_dim=4,
        moe_router_subname="router",
        moe_router_aux_subnames=("router.bias",),
        moe_fused_expert_subnames=(
            "experts.gate_up",
            "experts.down",
            "experts.down_bias",
        ),
        moe_fused_gate_up_subnames=("experts.gate_up",),
        moe_fused_down_subnames=("experts.down",),
        moe_expert_intermediate_group_size=group_size,
    )

    result = materialize_solution_state_dict(
        state,
        layouts,
        {
            0: BlockTarget(
                target_num_experts=2,
                target_expert_intermediate=4,
                expert_keep_indices=(3, 1),
            )
        },
    )

    assert result[f"{prefix}.router.weight"].shape == (2, hidden)
    assert result[f"{prefix}.router.bias"].shape == (2,)
    assert result[f"{prefix}.experts.gate_up"].shape == (2, 8, hidden)
    assert result[f"{prefix}.experts.down"].shape == (2, hidden, 2)
    assert result[f"{prefix}.experts.down_bias"].shape == (2, hidden)
    torch.testing.assert_close(
        result[f"{prefix}.experts.down_bias"],
        state[f"{prefix}.experts.down_bias"][[3, 1]],
    )


def test_untargeted_layers_unchanged():
    sd, layouts = _state(), _layouts()
    out = materialize_solution_state_dict(sd, layouts, {})  # no targets
    for key in sd:
        assert torch.equal(out[key], sd[key])


def test_materialize_no_op_removes_sublayer_tensors_but_keeps_other_half():
    layouts = _layouts()
    ffn_removed = materialize_solution_state_dict(
        _state(), layouts, {0: BlockTarget(remove_ffn=True)}
    )
    assert not any(".mlp." in key for key in ffn_removed)
    assert any(".self_attn." in key for key in ffn_removed)

    attention_removed = materialize_solution_state_dict(
        _state(), layouts, {0: BlockTarget(remove_attention=True)}
    )
    assert not any(".self_attn." in key for key in attention_removed)
    assert any(".mlp." in key for key in attention_removed)
