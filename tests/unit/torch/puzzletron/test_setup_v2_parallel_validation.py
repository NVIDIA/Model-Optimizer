# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from puzzletron_setup.v2.parallel_validation import (
    geometry_scope,
    validate_automodel_parallelism,
    validate_vllm_parallelism,
)
from puzzletron_setup.v2.resources import ParallelProfile


MOE_INVENTORY = {
    "moe": True,
    "facts": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "num_experts": 64,
    },
    "axes": [
        {
            "axis_id": "hidden_width",
            "teacher_value": 4096,
            "values": [4096, 3072],
            "alignment": 256,
            "label": "Residual width",
        },
        {
            "axis_id": "kv_groups",
            "teacher_value": 8,
            "values": [8, 6],
            "alignment": 1,
            "label": "KV groups",
        },
        {
            "axis_id": "q_heads_per_group",
            "teacher_value": 4,
            "values": [4, 3],
            "alignment": 1,
            "label": "Q heads per group",
        },
        {
            "axis_id": "moe_experts",
            "teacher_value": 64,
            "values": [64, 48],
            "alignment": 16,
            "label": "Experts",
        },
    ],
}

SELECTED = {
    "axes": {
        "hidden_width": {"enabled": True, "values": [3072]},
        "kv_groups": {"enabled": True, "values": [6]},
        "q_heads_per_group": {"enabled": True, "values": [3]},
        "moe_experts": {"enabled": True, "values": [48]},
    }
}


def _profile(**changes):
    values = {
        "name": "test",
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "ep": 1,
        "sequence_parallel": False,
    }
    values.update(changes)
    return ParallelProfile(**values)


def _messages(issues):
    return "\n".join(issue.message for issue in issues)


def test_stage_scope_keeps_masking_stages_on_teacher_geometry():
    assert geometry_scope("depth_importance") == "teacher"
    assert geometry_scope("width_importance") == "teacher"
    assert geometry_scope("sort_sanity") == "teacher"
    assert geometry_scope("bypass") == "teacher"
    assert geometry_scope("bypass_sanity") == "teacher"
    assert geometry_scope("width_sanity") == "candidate"
    assert geometry_scope("slicing_sanity") == "candidate"
    assert geometry_scope("replacement_scoring") == "candidate"
    assert geometry_scope("post.params.eval", node_type="evaluation") == "candidate"
    assert geometry_scope("post.params.kd", node_type="global_kd") == "candidate"
    assert geometry_scope("post.params.serving", node_type="aiperf") == "candidate"


def test_width_sanity_rejects_tp_incompatible_with_selected_query_heads():
    issues = validate_automodel_parallelism(
        _profile(tp=8),
        MOE_INVENTORY,
        SELECTED,
        stage_id="width_sanity",
        sequence_length=1024,
    )

    assert "query-head counts [18, 24, 32]" in _messages(issues)
    assert "valid choices [1, 2]" in _messages(issues)


def test_bypass_accepts_tp_that_only_candidate_geometry_would_reject():
    issues = validate_automodel_parallelism(
        _profile(tp=8),
        MOE_INVENTORY,
        SELECTED,
        stage_id="bypass",
        sequence_length=1024,
    )

    assert issues == ()


def test_candidate_stage_rejects_ep_incompatible_with_selected_experts():
    issues = validate_automodel_parallelism(
        _profile(ep=64, dp_shard=64),
        MOE_INVENTORY,
        SELECTED,
        stage_id="replacement_scoring",
        sequence_length=1024,
    )

    assert "expert counts [48, 64]" in _messages(issues)


def test_context_parallel_rejects_nondivisible_sequence_length():
    issues = validate_automodel_parallelism(
        _profile(cp=3),
        MOE_INVENTORY,
        SELECTED,
        stage_id="depth_importance",
        sequence_length=1024,
    )

    assert "CP=3 does not divide sequence length 1024" in _messages(issues)


def test_dense_model_rejects_expert_parallelism():
    dense_inventory = {**MOE_INVENTORY, "moe": False}
    issues = validate_automodel_parallelism(
        _profile(ep=2, dp_shard=2),
        dense_inventory,
        SELECTED,
        stage_id="depth_importance",
        sequence_length=1024,
    )

    assert "Dense models require EP=1" in _messages(issues)


def test_pipeline_parallelism_is_not_checked_against_layer_count():
    inventory = {**MOE_INVENTORY, "num_layers": 5}
    issues = validate_automodel_parallelism(
        _profile(pp=7),
        inventory,
        SELECTED,
        stage_id="depth_importance",
        sequence_length=1024,
    )

    assert issues == ()


def test_candidate_gdn_and_mamba_head_counts_are_tp_constrained():
    inventory = {
        "moe": False,
        "facts": {"hidden_size": 4096},
        "axes": [
            {
                "axis_id": "gdn_key_groups",
                "teacher_value": 8,
                "values": [8, 6],
            },
            {
                "axis_id": "gdn_value_heads_per_group",
                "teacher_value": 2,
                "values": [2, 3],
            },
            {
                "axis_id": "gdn_key_head_dim",
                "teacher_value": 128,
                "values": [128, 96],
            },
            {
                "axis_id": "mamba_heads",
                "teacher_value": 32,
                "values": [32, 30],
            },
            {
                "axis_id": "mamba_head_dim",
                "teacher_value": 128,
                "values": [128, 96],
            },
        ],
    }
    pruning = {
        "axes": {
            "gdn_key_groups": {"enabled": True, "values": [6]},
            "gdn_value_heads_per_group": {"enabled": True, "values": [3]},
            "gdn_key_head_dim": {"enabled": True, "values": [96]},
            "mamba_heads": {"enabled": True, "values": [30]},
            "mamba_head_dim": {"enabled": True, "values": [96]},
        }
    }

    issues = validate_automodel_parallelism(
        _profile(tp=8),
        inventory,
        pruning,
        stage_id="width_sanity",
        sequence_length=1024,
    )

    messages = _messages(issues)
    assert "GDN key-head counts [6, 8]" in messages
    assert "GDN value-head counts [12, 16, 18, 24]" in messages
    assert "Mamba head counts [30, 32]" in messages
    assert "head dimension" not in messages


def test_vllm_rejects_tp_and_effective_ep_for_any_selected_candidate():
    issues = validate_vllm_parallelism(
        {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 8,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": True,
            "gpu_group_size": 32,
        },
        MOE_INVENTORY,
        SELECTED,
        stage_id="post.params.serving",
    )

    messages = _messages(issues)
    assert "query-head counts [18, 24, 32]" in messages
    assert "effective EP=32" in messages
    assert "expert counts [48, 64]" in messages


def test_vllm_decode_context_parallel_checks_each_grouped_attention_geometry():
    issues = validate_vllm_parallelism(
        {
            "tensor_parallel_size": 6,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 3,
            "enable_expert_parallel": False,
            "gpu_group_size": 6,
        },
        MOE_INVENTORY,
        SELECTED,
        stage_id="vllm_stats",
    )

    assert "DCP=3" in _messages(issues)
    assert "KV-head count 6" in _messages(issues)
