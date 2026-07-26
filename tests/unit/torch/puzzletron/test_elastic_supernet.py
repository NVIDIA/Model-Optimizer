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

"""CPU tests for the elastic-supernet per-step config sampler (pure parts)."""

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
)
from modelopt.torch.puzzletron.bypass_distillation import elastic_supernet
from modelopt.torch.puzzletron.bypass_distillation.elastic_supernet import (
    CanonicalBlockElastic,
    CanonicalCandidateMasker,
    ElasticSupernetMasker,
    build_subblock_elastics,
)
from modelopt.torch.puzzletron.pruning.sorted_teacher import build_layer_layouts

HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, INTER = 64, 24, 4, 4, 12


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


def _fake_param_fn(subblock_config):
    """Stand-in for calculate_subblock_params (which builds a meta layer); proportional counts."""
    if subblock_config.no_op:
        return 0
    if getattr(subblock_config, "num_kv_heads", None) is not None:  # AttentionConfig
        q, kv = subblock_config.num_query_heads, subblock_config.num_kv_heads
        return (2 * q + 2 * kv) * HEAD_DIM * HIDDEN
    return 3 * HIDDEN * subblock_config.intermediate_size  # FFNConfig


def test_build_subblock_elastics_includes_teacher_size():
    elastics = build_subblock_elastics(
        _layouts(), param_fn=_fake_param_fn, ffn_sizes=[4, 8], attn_targets=[(20, 4), (18, 3)]
    )
    kinds = {e.kind: e for e in elastics}
    assert set(kinds) == {"ffn", "attn"}
    # FFN: pruned sizes + teacher intermediate (12); all <= teacher.
    assert kinds["ffn"].sampler.sizes == [0, 4, 8, 12]
    # Attention: targets + teacher (24,4) included.
    assert (NUM_Q, NUM_KV) in kinds["attn"].sampler.sizes
    assert (18, 3) in kinds["attn"].sampler.sizes
    assert (0, 0) in kinds["attn"].sampler.sizes


def test_sample_targets_returns_valid_choices():
    elastics = build_subblock_elastics(
        _layouts(), param_fn=_fake_param_fn, ffn_sizes=[4, 8], attn_targets=[(20, 4), (18, 3)]
    )
    masker = ElasticSupernetMasker(elastics, head_dim=HEAD_DIM)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        t = masker.sample_targets(gen)
        assert t[0]["ffn"] in [0, 4, 8, 12]
        assert t[0]["attn"] in [(0, 0), (18, 3), (20, 4), (24, 4)]


def test_axis_endpoint_schedule_ignores_architecturally_absent_subblocks():
    parent = BlockConfig(
        subblock_configs=(
            MambaConfig(num_heads=8, head_dim=4, num_groups=2, state_dim=8),
            FFNConfig(no_op=True),
        )
    )
    reduced = BlockConfig(
        subblock_configs=(
            MambaConfig(num_heads=4, head_dim=4, num_groups=2, state_dim=8),
            FFNConfig(no_op=True),
        )
    )

    def candidate(identity, block, axes):
        return SimpleNamespace(
            identity=SimpleNamespace(value=identity),
            block_config=block,
            metadata={"slice_axes": axes},
        )

    elastic = CanonicalBlockElastic(
        layer_idx=0,
        parent_block_config=parent,
        sampler=SimpleNamespace(
            sizes=[
                candidate("teacher", parent, {}),
                candidate("reduced", reduced, {"mamba_heads": 4}),
            ]
        ),
    )
    masker = CanonicalCandidateMasker(
        [elastic], layouts_by_idx={}, head_dim=4, seed=42
    )

    schedule = masker.coverage_schedule_manifest()["layer_0"]
    assert [row["candidate_id"] for row in schedule] == ["teacher", "reduced"]
    assert all(row["disabled_subblocks"] == [] for row in schedule)
    assert masker.sample_targets(coverage_mode="axis_endpoints")[0].identity.value == "teacher"
    assert masker.sample_targets(coverage_mode="axis_endpoints")[0].identity.value == "reduced"


def test_coverage_then_uniform_visits_prelude_then_full_candidate_set():
    parent = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=16),
        )
    )

    def candidate(identity, block, axes):
        return SimpleNamespace(
            identity=SimpleNamespace(value=identity),
            block_config=block,
            metadata={"slice_axes": axes},
        )

    attention_reduced = parent.with_subblock(
        AttentionConfig(num_query_heads=4, num_kv_heads=1)
    )
    ffn_reduced = parent.with_subblock(FFNConfig(intermediate_size=8))
    attention_noop = parent.with_subblock(AttentionConfig(no_op=True))
    mixed = attention_reduced.with_subblock(FFNConfig(intermediate_size=8))
    sizes = [
        candidate("teacher", parent, {}),
        candidate("attention", attention_reduced, {"kv_groups": 1}),
        candidate("ffn", ffn_reduced, {"ffn_intermediate": 8}),
        candidate("attention_noop", attention_noop, {}),
        candidate(
            "mixed",
            mixed,
            {"kv_groups": 1, "ffn_intermediate": 8},
        ),
    ]
    elastic = CanonicalBlockElastic(
        layer_idx=0,
        parent_block_config=parent,
        sampler=SimpleNamespace(sizes=sizes),
    )
    first = CanonicalCandidateMasker(
        [elastic], layouts_by_idx={}, head_dim=4, seed=7
    )
    second = CanonicalCandidateMasker(
        [elastic], layouts_by_idx={}, head_dim=4, seed=7
    )
    first_generator = torch.Generator().manual_seed(11)
    second_generator = torch.Generator().manual_seed(11)

    schedule = first.coverage_schedule_manifest()["layer_0"]
    prelude = [
        first.sample_targets(
            first_generator, coverage_mode="coverage_then_uniform"
        )[0].identity.value
        for _ in schedule
    ]
    replay = [
        second.sample_targets(
            second_generator, coverage_mode="coverage_then_uniform"
        )[0].identity.value
        for _ in schedule
    ]
    assert prelude == [row["candidate_id"] for row in schedule]
    assert replay == prelude

    postlude = [
        first.sample_targets(
            first_generator, coverage_mode="coverage_then_uniform"
        )[0].identity.value
        for _ in range(50)
    ]
    assert set(postlude) <= {item.identity.value for item in sizes}
    assert len(set(postlude)) > 1


def test_canonical_masker_can_select_smallest_complete_candidate():
    parent = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=16),
        )
    )
    reduced = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=4, num_kv_heads=1),
            FFNConfig(intermediate_size=8),
        )
    )

    def candidate(identity, block, axes):
        return SimpleNamespace(
            identity=SimpleNamespace(value=identity),
            block_config=block,
            metadata={"slice_axes": axes},
        )

    elastic = CanonicalBlockElastic(
        layer_idx=0,
        parent_block_config=parent,
        sampler=SimpleNamespace(
            sizes=[
                candidate("teacher", parent, {}),
                candidate(
                    "smallest",
                    reduced,
                    {"kv_groups": 1, "ffn_intermediate": 8},
                ),
            ],
            probs=torch.tensor([0.2, 0.8]),
        ),
    )
    masker = CanonicalCandidateMasker(
        [elastic], layouts_by_idx={}, head_dim=4, seed=42
    )

    selected = masker.sample_targets(selection="smallest")

    assert selected[0].identity.value == "smallest"
    assert masker.coverage["smallest"]["visits"] == 1


def test_logical_data_lanes_are_model_parallel_connected_components():
    peer_sets = [
        ((0, 1), (0, 2)),
        ((0, 1), (1, 3)),
        ((2, 3), (0, 2)),
        ((2, 3), (1, 3)),
        ((4, 5), (4, 6)),
        ((4, 5), (5, 7)),
        ((6, 7), (4, 6)),
        ((6, 7), (5, 7)),
    ]

    assert [
        elastic_supernet.logical_data_lane_from_peer_sets(rank, peer_sets)
        for rank in range(8)
    ] == [(0, 2)] * 4 + [(1, 2)] * 4


def test_canonical_sampling_is_lane_diverse_and_exactly_reproducible():
    parent = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=16),
        )
    )

    def candidate(identity, block, axes):
        return SimpleNamespace(
            identity=SimpleNamespace(value=identity),
            block_config=block,
            metadata={"slice_axes": axes},
        )

    sizes = [
        candidate("teacher", parent, {}),
        candidate(
            "attention",
            parent.with_subblock(AttentionConfig(num_query_heads=4, num_kv_heads=1)),
            {"kv_groups": 1},
        ),
        candidate(
            "ffn",
            parent.with_subblock(FFNConfig(intermediate_size=8)),
            {"ffn_intermediate": 8},
        ),
        candidate(
            "attention-noop",
            parent.with_subblock(AttentionConfig(no_op=True)),
            {},
        ),
    ]
    elastic = CanonicalBlockElastic(
        layer_idx=0,
        parent_block_config=parent,
        sampler=SimpleNamespace(sizes=sizes, probs=torch.full((4,), 0.25)),
    )

    observed = []
    for lane in range(4):
        masker = CanonicalCandidateMasker(
            [elastic], layouts_by_idx={}, head_dim=4, seed=17
        )
        selected = masker.sample_targets(
            sample_index=lane,
            coverage_mode="coverage_then_uniform",
        )
        observed.append(selected[0].identity.value)

        replay = CanonicalCandidateMasker(
            [elastic], layouts_by_idx={}, head_dim=4, seed=17
        ).sample_targets(
            sample_index=lane,
            coverage_mode="coverage_then_uniform",
        )
        assert replay[0].identity.value == selected[0].identity.value

    assert observed == [
        row["candidate_id"]
        for row in CanonicalCandidateMasker(
            [elastic], layouts_by_idx={}, head_dim=4, seed=17
        ).coverage_schedule_manifest()["layer_0"]
    ]


def test_model_parallel_peers_must_agree_on_lane_architecture():
    elastic_supernet.validate_lane_architecture_assignments(
        [(0, "architecture-a"), (0, "architecture-a"), (1, "architecture-b")]
    )

    with pytest.raises(RuntimeError, match="lane 0.*architecture-a.*architecture-b"):
        elastic_supernet.validate_lane_architecture_assignments(
            [(0, "architecture-a"), (0, "architecture-b")]
        )
