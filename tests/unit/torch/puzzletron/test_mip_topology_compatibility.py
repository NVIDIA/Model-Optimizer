# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.mip.run_puzzle import (
    _apply_forced_removals,
    _block_config_tp_compatible,
)


def _block(*subblocks):
    return BlockConfig(subblock_configs=tuple(subblocks))


def test_tp_compatibility_rejects_attention_heads_that_cannot_be_sharded():
    assert _block_config_tp_compatible(
        _block(AttentionConfig(num_kv_heads=2, num_query_heads=4)), 2
    )
    assert not _block_config_tp_compatible(
        _block(AttentionConfig(num_kv_heads=1, num_query_heads=4)), 2
    )
    assert not _block_config_tp_compatible(
        _block(AttentionConfig(num_kv_heads=1, num_query_heads=3)), 2
    )


def test_tp_compatibility_keeps_noops_and_unrelated_subblocks():
    assert _block_config_tp_compatible(_block(AttentionConfig(no_op=True)), 2)
    assert _block_config_tp_compatible(_block(FFNConfig(intermediate_size=1792)), 2)
    assert _block_config_tp_compatible(
        _block(AttentionConfig(num_kv_heads=1, num_query_heads=2)), 1
    )


def _duplicate_metrics(first: float, second: float):
    block = _block(FFNConfig(intermediate_size=16))
    return {
        "teacher": {
            "parent_layer_indices": [0],
            "block_config": block,
            "is_teacher": True,
            "metrics": {"score": first},
        },
        "candidate": {
            "parent_layer_indices": [0],
            "block_config": block,
            "is_teacher": False,
            "metrics": {"score": second},
        },
    }


def test_forced_removal_dedup_respects_minimization():
    result = _apply_forced_removals(
        _duplicate_metrics(3.0, 1.0),
        [],
        objective="metrics.score",
        bigger_is_better=False,
    )
    assert [row["metrics"]["score"] for row in result.values()] == [1.0]


def test_forced_removal_dedup_respects_maximization():
    result = _apply_forced_removals(
        _duplicate_metrics(3.0, 1.0),
        [],
        objective="metrics.score",
        bigger_is_better=True,
    )
    assert [row["metrics"]["score"] for row in result.values()] == [3.0]
