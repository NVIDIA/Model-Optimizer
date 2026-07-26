# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from puzzletron_setup.v2.state import WizardState
from puzzletron_setup.v2.validation import validate_state

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
        {"axis_id": "hidden_width", "teacher_value": 4096},
        {"axis_id": "kv_groups", "teacher_value": 8},
        {"axis_id": "q_heads_per_group", "teacher_value": 4},
        {"axis_id": "moe_experts", "teacher_value": 64},
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


def _profile(*, tp=1, ep=1, dp_shard=1):
    return {
        "name": "model",
        "tp": tp,
        "cp": 1,
        "pp": 1,
        "dp_shard": dp_shard,
        "dp_replicate": 1,
        "ep": ep,
        "sequence_parallel": False,
        "consumers": [],
    }


def _state(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    for path, value in (
        ("model.source", "/model"),
        ("data.source", "/data"),
        ("data.sequence_length", 1024),
        ("infrastructure.execution_contract.repository", "/repo"),
        ("output.result_root", "/results"),
    ):
        state.set_field(path, value)
    state.set_model({"source": "/model"}, MOE_INVENTORY)
    state.set_collection("pruning", SELECTED)
    return state


def _messages(state):
    return "\n".join(issue.message for issue in validate_state(state))


def test_persisted_static_resources_use_teacher_or_candidate_geometry(tmp_path):
    state = _state(tmp_path)
    state.set_collection("parallel_profiles", {"model": _profile(tp=8)})
    state.set_collection(
        "stage_resources",
        {
            "bypass": {"profile_name": "model"},
            "width_sanity": {"profile_name": "model"},
        },
    )

    messages = _messages(state)

    assert "query-head counts [18, 24, 32]" in messages
    assert "bypass" not in messages


def test_persisted_post_mip_automodel_resources_are_candidate_checked(tmp_path):
    state = _state(tmp_path)
    state.set_collection("parallel_profiles", {"model": _profile(tp=8)})
    state.set_collection(
        "stage_resources",
        {
            "post.run.evaluate": {"profile_name": "model"},
            "post.run.distill": {"profile_name": "model"},
        },
    )
    state.set_collection(
        "post_mip_flows",
        {
            "run": {
                "source": {"run": "run"},
                "nodes": {
                    "evaluate": {"type": "evaluation"},
                    "distill": {"type": "global_kd"},
                },
            }
        },
    )

    messages = _messages(state)

    assert messages.count("query-head counts [18, 24, 32]") == 2


def test_persisted_post_mip_aiperf_topology_is_candidate_checked(tmp_path):
    state = _state(tmp_path)
    topology = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 8,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "gpu_group_size": 32,
    }
    state.set_collection(
        "post_mip_flows",
        {
            "run": {
                "source": {"run": "run"},
                "nodes": {
                    "serve": {
                        "type": "aiperf",
                        "config": {"topology": topology},
                    }
                },
            }
        },
    )

    messages = _messages(state)

    assert "effective EP=32 (TP * DP)" in messages
    assert "expert counts [48, 64]" in messages


def test_persisted_vllm_measurement_topology_is_candidate_checked(tmp_path):
    state = _state(tmp_path)
    state.set_collection(
        "vllm_measurements",
        {
            "serving": {
                "prefill_seq_len": 1024,
                "generation_seq_len": 128,
                "batch_size": 1,
                "max_num_seqs": 1,
                "runtime_stats": {
                    "topology": {
                        "tensor_parallel_size": 8,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                        "prefill_context_parallel_size": 1,
                        "decode_context_parallel_size": 1,
                        "enable_expert_parallel": False,
                        "gpu_group_size": 8,
                    }
                },
            }
        },
    )

    messages = _messages(state)

    assert "query-head counts [18, 24, 32]" in messages
