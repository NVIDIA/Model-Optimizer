from copy import deepcopy

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.candidates import build_candidate_library
from modelopt.torch.puzzletron.sampling.sparse import SparseSamplingPolicy


def _fixture():
    teacher = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2, qk_head_dim=128),
            FFNConfig(intermediate_size=16),
        )
    )
    candidates = build_candidate_library(
        [teacher] * 20,
        search_space={
            "axes": {
                "kv_groups": {"enabled": True, "sizes": [1]},
                "q_heads_per_group": {"enabled": True, "sizes": [2]},
                "qk_head_dim": {"enabled": True, "sizes": [64]},
                "ffn_intermediate": {"enabled": True, "sizes": [8]},
            }
        },
        parent_checkpoint_identity="teacher",
        include_self=True,
        include_noops=False,
        hidden_width=1024,
    )
    solutions = [
        {
            "single_sequence_replacement": {
                "weight_paths": [],
                "parent_layer_indices": [candidate.layer_idx],
                "child_block_configs": [candidate.block_config.to_dict()],
            }
        }
        for candidate in candidates
        if candidate.source_kind != "self"
    ]
    return candidates, solutions


def test_sparse_replacement_payload_caps_and_annotates_one_width():
    from examples.puzzletron.prepare_sparse_replacement_scoring import (
        prepare_sparse_replacement_payload,
    )

    candidates, solutions = _fixture()
    manifest, selected = prepare_sparse_replacement_payload(
        candidates,
        solutions,
        hidden_width=1024,
        policy=SparseSamplingPolicy(replacement_cap=50),
    )

    assert len(selected) == 50
    assert len(manifest["selected"]) == 50
    assert all(solution["hidden_width"] == 1024 for solution in selected)
    assert len({solution["sparse_sample_id"] for solution in selected}) == 50
    first_pair = next(
        (
            index
            for index, row in enumerate(manifest["selected"])
            if len(row["changed_axes"]) == 2
        ),
        len(selected),
    )
    assert all(
        len(row["changed_axes"]) == 1 for row in manifest["selected"][:first_pair]
    )


def test_sparse_replacement_matching_ignores_serialized_subblock_order():
    from examples.puzzletron.prepare_sparse_replacement_scoring import (
        prepare_sparse_replacement_payload,
    )

    candidates, solutions = _fixture()
    reordered = deepcopy(solutions)
    for solution in reordered:
        block = solution["single_sequence_replacement"]["child_block_configs"][0]
        block["subblock_configs"] = list(reversed(block["subblock_configs"]))

    _manifest, selected = prepare_sparse_replacement_payload(
        candidates,
        reordered,
        hidden_width=1024,
        policy=SparseSamplingPolicy(replacement_cap=50),
    )

    assert len(selected) == 50


def test_sparse_replacement_matching_ignores_optional_null_fields():
    from examples.puzzletron.prepare_sparse_replacement_scoring import (
        prepare_sparse_replacement_payload,
    )

    candidates, solutions = _fixture()
    with_nulls = deepcopy(solutions)
    for solution in with_nulls:
        subblocks = solution["single_sequence_replacement"]["child_block_configs"][0][
            "subblock_configs"
        ]
        for subblock in subblocks:
            subblock["optional_runtime_field"] = None

    _manifest, selected = prepare_sparse_replacement_payload(
        candidates,
        with_nulls,
        hidden_width=1024,
        policy=SparseSamplingPolicy(replacement_cap=50),
    )

    assert len(selected) == 50
