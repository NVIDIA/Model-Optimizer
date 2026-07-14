from dataclasses import replace

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.candidates import build_candidate_library
from modelopt.torch.puzzletron.sampling.sparse import (
    SparseSamplingPolicy,
    sample_replacement_candidates,
    sample_subblock_configs,
)


def _candidates(*, layers: int = 5, width: int = 1024):
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2, qk_head_dim=128),
            FFNConfig(intermediate_size=16),
        )
    )
    search_space = {
        "axes": {
            "kv_groups": {"enabled": True, "sizes": [1]},
            "q_heads_per_group": {"enabled": True, "sizes": [2]},
            "qk_head_dim": {"enabled": True, "sizes": [64]},
            "ffn_intermediate": {"enabled": True, "sizes": [8]},
        }
    }
    return build_candidate_library(
        [block] * layers,
        search_space=search_space,
        parent_checkpoint_identity="teacher",
        include_self=True,
        include_noops=False,
        hidden_width=width,
    )


def test_subblock_sampling_keeps_anchor_all_singles_and_four_pairs():
    manifest = sample_subblock_configs(
        _candidates(),
        policy=SparseSamplingPolicy(max_pairwise_per_family=4),
    )

    attention = [row for row in manifest.selected if row.subblock_kind == "attention"]
    ffn = [row for row in manifest.selected if row.subblock_kind == "ffn"]
    assert sum(not row.changed_axes for row in attention) == 1
    assert {row.changed_axes for row in attention if len(row.changed_axes) == 1} == {
        ("kv_groups",),
        ("q_heads_per_group",),
        ("qk_head_dim",),
    }
    assert len([row for row in attention if len(row.changed_axes) == 2]) == 3
    assert all(len(row.changed_axes) <= 2 for row in attention)
    assert [(row.changed_axes, row.subblock_kind) for row in ffn] == [
        ((), "ffn"),
        (("ffn_intermediate",), "ffn"),
    ]


def test_subblock_sampling_is_deterministic_and_width_isolated():
    wide = _candidates(width=1024)
    narrow = _candidates(width=512)
    policy = SparseSamplingPolicy(seed=17)

    forward = sample_subblock_configs([*wide, *narrow], policy=policy)
    reverse = sample_subblock_configs(list(reversed([*wide, *narrow])), policy=policy)

    assert forward.identity == reverse.identity
    assert [row.sample_id for row in forward.selected] == [
        row.sample_id for row in reverse.selected
    ]
    assert {row.hidden_width for row in forward.selected} == {512, 1024}
    assert len({(row.hidden_width, row.sample_id) for row in forward.selected}) == len(
        forward.selected
    )


def test_subblock_sampling_rejects_noop_and_more_than_two_axis_runtime_entries():
    candidates = _candidates(layers=1)
    no_op = replace(
        candidates[0],
        block_config=BlockConfig(
            subblock_configs=(
                AttentionConfig(no_op=True),
                FFNConfig(intermediate_size=16),
            )
        ),
        source_kind="no_op",
    )

    manifest = sample_subblock_configs([*candidates, no_op])

    assert all(not row.no_op for row in manifest.selected)
    assert any(row.reason == "no_op" for row in manifest.excluded)
    assert any(row.reason == "more_than_two_axes" for row in manifest.excluded)


def test_replacement_sampling_caps_each_width_and_prioritizes_single_axes():
    policy = SparseSamplingPolicy(replacement_cap=7)
    manifest = sample_replacement_candidates(
        [*_candidates(layers=7, width=1024), *_candidates(layers=7, width=512)],
        policy=policy,
    )

    for width in (512, 1024):
        rows = [row for row in manifest.selected if row.hidden_width == width]
        assert len(rows) == 7
        assert all(row.changed_axes for row in rows)
        pair_index = next((i for i, row in enumerate(rows) if len(row.changed_axes) == 2), len(rows))
        assert all(len(row.changed_axes) == 1 for row in rows[:pair_index])
        assert {row.layer_idx for row in rows} >= {0, 3, 6}


def test_replacement_sampling_reserves_maximal_pruning_anchor_per_layer():
    manifest = sample_replacement_candidates(_candidates(layers=1))

    assert all(row.changed_axes for row in manifest.selected)
    assert max(len(row.changed_axes) for row in manifest.selected) == 4
    reasons = {row.reason for row in manifest.excluded}
    assert "teacher_anchor" in reasons


def test_sparse_sampling_ignores_architecturally_absent_noop_subblocks():
    teacher = BlockConfig(
        subblock_configs=(
            MambaConfig(num_heads=8, head_dim=64, state_dim=128),
            FFNConfig(no_op=True),
        )
    )
    candidates = build_candidate_library(
        [teacher] * 3,
        search_space={
            "axes": {
                "mamba_heads": {"enabled": True, "sizes": [4]},
                "mamba_head_dim": {"enabled": True, "sizes": [32]},
            }
        },
        parent_checkpoint_identity="teacher",
        include_self=True,
        include_noops=False,
        hidden_width=1024,
    )

    runtime = sample_subblock_configs(candidates)
    replacement = sample_replacement_candidates(candidates)

    assert {row.subblock_kind for row in runtime.selected} == {"mamba"}
    assert {row.changed_axes for row in runtime.selected} == {
        (),
        ("mamba_head_dim",),
        ("mamba_heads",),
        ("mamba_head_dim", "mamba_heads"),
    }
    assert replacement.selected
    assert all(row.subblock_kind == "mamba" for row in replacement.selected)


def test_cartesian_noops_exclude_whole_block_when_whole_block_is_disabled():
    teacher = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            MoEConfig(num_experts=8, expert_intermediate_size=16, top_k=2),
        )
    )
    candidates = build_candidate_library(
        [teacher],
        search_space={
            "no_op": {
                "subblocks": ["attention", "moe"],
                "whole_block": False,
                "cartesian": True,
            }
        },
        parent_checkpoint_identity="teacher",
        include_self=True,
        include_noops=True,
    )

    no_op_states = {
        (
            candidate.block_config.require_subblock("attention").no_op,
            candidate.block_config.require_subblock("moe").no_op,
        )
        for candidate in candidates
    }
    assert no_op_states == {(False, False), (True, False), (False, True)}
