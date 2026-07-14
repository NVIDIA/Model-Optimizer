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

import math

import pytest

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.replacement_library.score_composition import (
    compose_full_block_metrics,
    composed_table_to_gathered_metrics,
)


def _block(attention_heads=8, ffn_size=16, *, reverse=False):
    subblocks = [
        AttentionConfig(num_query_heads=attention_heads, num_kv_heads=2),
        FFNConfig(intermediate_size=ffn_size),
    ]
    if reverse:
        subblocks.reverse()
    return BlockConfig(subblock_configs=tuple(subblocks))


def _solution(block):
    return {
        "single_sequence_replacement": {
            "weight_paths": [],
            "parent_layer_indices": [0],
            "child_block_configs": [block.to_dict()],
        }
    }


def _subblock_result(index, kind, block, value):
    return {
        "request_id": f"request-{index}",
        "i_solution": index,
        "puzzle_solution": {
            "subblock_replacement": {"layer_idx": 0, "kind": kind, "name": kind},
            **_solution(block),
        },
        "lm_loss": {"avg": value},
    }


def test_composes_two_subblock_deltas_relative_to_teacher():
    teacher = _block()
    candidate = _block(attention_heads=4, ffn_size=8)
    results = [
        _subblock_result(0, "attention", _block(attention_heads=4), 1.2),
        _subblock_result(1, "ffn", _block(ffn_size=8), 1.3),
    ]

    table = compose_full_block_metrics(
        [_solution(candidate)],
        results,
        teacher_blocks=[teacher],
        teacher_baseline={"lm_loss": 1.0},
    )

    record = table.records[0]
    assert record.metrics["lm_loss"] == pytest.approx(1.5)
    assert record.provenance == "additive_subblock"
    assert record.source_result_ids == ("request-0", "request-1")


def test_component_matching_ignores_serialized_subblock_order():
    teacher = _block()
    candidate = _block(attention_heads=4, reverse=True)

    table = compose_full_block_metrics(
        [_solution(candidate)],
        [_subblock_result(0, "attention", _block(attention_heads=4), 1.2)],
        teacher_blocks=[teacher],
        teacher_baseline={"lm_loss": 1.0},
    )

    assert table.records[0].metrics["lm_loss"] == pytest.approx(1.2)


def test_missing_subblock_component_is_rejected():
    with pytest.raises(ValueError, match="missing subblock score"):
        compose_full_block_metrics(
            [_solution(_block(attention_heads=4, ffn_size=8))],
            [_subblock_result(0, "attention", _block(attention_heads=4), 1.2)],
            teacher_blocks=[_block()],
            teacher_baseline={"lm_loss": 1.0},
        )


def test_nonfinite_component_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        compose_full_block_metrics(
            [_solution(_block(attention_heads=4))],
            [_subblock_result(0, "attention", _block(attention_heads=4), math.nan)],
            teacher_blocks=[_block()],
            teacher_baseline={"lm_loss": 1.0},
        )


def test_exact_block_result_takes_precedence():
    exact = {
        "request_id": "exact",
        "puzzle_solution": _solution(_block(attention_heads=4, ffn_size=8)),
        "lm_loss": {"avg": 1.1},
    }
    table = compose_full_block_metrics(
        [_solution(_block(attention_heads=4, ffn_size=8))],
        [
            _subblock_result(0, "attention", _block(attention_heads=4), 1.2),
            _subblock_result(1, "ffn", _block(ffn_size=8), 1.3),
        ],
        teacher_blocks=[_block()],
        teacher_baseline={"lm_loss": 1.0},
        exact_results=[exact],
    )

    assert table.records[0].metrics["lm_loss"] == pytest.approx(1.1)
    assert table.records[0].provenance == "exact_block"


def test_converts_composed_records_to_canonical_mip_candidates_with_provenance():
    teacher = _block()
    candidate = _block(attention_heads=4)
    canonical = [_solution(candidate)]
    table = compose_full_block_metrics(
        canonical,
        [_subblock_result(7, "attention", candidate, 1.2)],
        teacher_blocks=[teacher],
        teacher_baseline={"lm_loss": 1.0},
    )
    teacher_record = {
        "block_config": teacher,
        "parent_layer_indices": [0],
        "metrics": {"lm_loss": 1.0, "one_minus_lm_loss": 0.0},
        "is_teacher": True,
    }

    gathered = composed_table_to_gathered_metrics(table, canonical, [teacher_record])

    candidate_record = gathered["replacement_0"]
    assert candidate_record["block_config"] == candidate
    assert candidate_record["layer_replacement"]["parent_layer_indices"] == [0]
    assert candidate_record["metrics"] == pytest.approx({"lm_loss": 1.2, "one_minus_lm_loss": -0.2})
    assert candidate_record["score_provenance"] == {
        "granularity": "subblock",
        "method": "additive_subblock",
        "source_result_ids": ["request-7"],
    }
    assert gathered["teacher_0"] is teacher_record
