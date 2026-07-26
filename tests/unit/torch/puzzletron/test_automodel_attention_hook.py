# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Puzzletron's sole grouped-attention ranking method."""

import torch
from torch import nn

from modelopt.torch.puzzletron.block_config import AttentionConfig
from modelopt.torch.puzzletron.plugins.automodel.hooks import GroupedAttentionScorer
from modelopt.torch.puzzletron.plugins.automodel.reduction import MeshGroups


def test_grouped_attention_ranks_kv_groups_and_query_heads_by_deletion_damage():
    num_q, num_kv, head_dim, hidden = 4, 2, 1, 2
    projection = nn.Linear(num_q * head_dim, hidden, bias=False)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0, 4.0],
                ]
            )
        )
    activations = torch.ones(1, 3, num_q * head_dim)
    scorer = GroupedAttentionScorer(
        projection,
        MeshGroups(),
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        validation_full_iters=1,
    )

    scorer(projection, (activations,), None)
    scorer.step_iteration()
    result = scorer.finalize()

    # Group 1 contributes [2, 4] and is more important than group 0's [1, 1].
    assert result["kv_groups_importance_ascending"].tolist() == [0, 1]
    # Within group 0 the heads tie; within group 1 the 4x head outranks the 2x head.
    assert result["query_heads_importance_ascending_per_group"][1].tolist() == [0, 1]
    assert result["kv_group_scores"].shape == (num_kv,)
    assert result["query_head_scores"].shape == (num_kv, num_q // num_kv)


def test_grouped_attention_axis_selection_is_explicit():
    projection = nn.Linear(4, 2, bias=False)
    scorer = GroupedAttentionScorer(
        projection,
        MeshGroups(),
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=1,
        validation_full_iters=1,
        scored_axes=("kv_groups",),
    )
    scorer(projection, (torch.ones(1, 2, 4),), None)
    scorer.step_iteration()
    assert set(scorer.finalize()) == {
        "kv_group_scores",
        "kv_groups_importance_ascending",
    }


def test_grouped_attention_excludes_canonical_padding():
    projection = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        projection.weight.fill_(1)
    activations = torch.tensor([[[1.0, 2.0], [1000.0, 1000.0]]])
    scorer = GroupedAttentionScorer(
        projection,
        MeshGroups(),
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=1,
        validation_full_iters=1,
    )
    scorer.set_batch_metadata(
        sequence_ids=torch.tensor([[0, -1]]),
        num_samples=1,
    )

    scorer(projection, (activations,), None)
    assert scorer._pending_count == 1
    scorer.step_iteration()
    assert scorer.finalize()["kv_groups_importance_ascending"].tolist() == [0, 1]


def test_grouped_attention_exact_checkpoint_finalizes_without_replay():
    projection = nn.Linear(4, 2, bias=False)
    scorer = GroupedAttentionScorer(
        projection,
        MeshGroups(),
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=1,
        validation_full_iters=1,
    )
    scorer(projection, (torch.ones(1, 2, 4),), None)
    scorer.step_iteration()
    expected = scorer.finalize()

    restored = GroupedAttentionScorer(
        projection,
        MeshGroups(),
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=1,
        validation_full_iters=1,
    )
    restored.load_checkpoint_state(scorer.checkpoint_state())
    actual = restored.finalize()

    torch.testing.assert_close(actual["kv_group_scores"], expected["kv_group_scores"])
    torch.testing.assert_close(actual["query_head_scores"], expected["query_head_scores"])


def test_attention_config_uses_canonical_grouped_head_fields():
    config = AttentionConfig(num_kv_heads=2, num_query_heads=8)
    assert (config.num_kv_heads, config.num_query_heads) == (2, 8)
    no_op = AttentionConfig(no_op=True, num_kv_heads=2, num_query_heads=8)
    assert no_op.num_kv_heads is None
    assert no_op.num_query_heads is None
