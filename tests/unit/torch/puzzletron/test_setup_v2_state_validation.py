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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from puzzletron_setup import WORKER_REPOSITORY_PLACEHOLDER, WORKER_VENV_PLACEHOLDER
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
        ("infrastructure.execution_contract.venv", "/venv"),
        ("output.result_root", "/results"),
    ):
        state.set_field(path, value)
    state.set_model({"source": "/model"}, MOE_INVENTORY)
    state.set_collection("pruning", SELECTED)
    return state


def _messages(state):
    return "\n".join(issue.message for issue in validate_state(state))


def test_worker_path_placeholders_must_be_replaced(tmp_path):
    state = _state(tmp_path)
    state.set_field(
        "infrastructure.execution_contract.repository",
        WORKER_REPOSITORY_PLACEHOLDER,
    )
    state.set_field("infrastructure.execution_contract.venv", WORKER_VENV_PLACEHOLDER)

    issues = {
        issue.path: issue.message
        for issue in validate_state(state)
        if issue.path.startswith("infrastructure.execution_contract")
    }

    assert issues == {
        "infrastructure.execution_contract.repository": (
            "Replace the placeholder with a path visible on every worker."
        ),
        "infrastructure.execution_contract.venv": (
            "Replace the placeholder with a path visible on every worker."
        ),
    }


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


def test_hugging_face_subset_selection_rejects_invalid_weights_and_disabled_entries(
    tmp_path,
):
    state = _state(tmp_path)
    state.set_field("data.selected_source", "owner/dataset")
    state.set_collection(
        "hf_dataset_catalogs",
        {
            "owner/dataset@sha": {
                "source": "owner/dataset",
                "revision": "sha",
                "default_subset": "small",
                "subsets": [
                    {
                        "name": "small",
                        "num_rows": 10,
                        "num_bytes_original_files": 100,
                        "selectable": True,
                        "disabled_reason": None,
                    },
                    {
                        "name": "external",
                        "num_rows": 30,
                        "num_bytes_original_files": 300,
                        "selectable": False,
                        "disabled_reason": "external media required",
                    },
                ],
            }
        },
    )
    state.set_collection(
        "data_subset_selection",
        {
            "source": "owner/dataset",
            "revision": "sha",
            "subsets": [
                {
                    "name": "small",
                    "num_rows": 10,
                    "num_bytes_original_files": 100,
                    "weight": 0.25,
                },
                {
                    "name": "external",
                    "num_rows": 30,
                    "num_bytes_original_files": 300,
                    "weight": 0.5,
                },
            ],
        },
    )

    messages = _messages(state)

    assert "weights must sum to 1.0" in messages
    assert "external media required" in messages


def test_hugging_face_subset_selection_accepts_revision_locked_metadata(tmp_path):
    state = _state(tmp_path)
    state.set_field("data.selected_source", "owner/dataset")
    subset = {
        "name": "default",
        "num_rows": 10,
        "num_bytes_original_files": 100,
        "weight": 1.0,
    }
    state.set_collection(
        "hf_dataset_catalogs",
        {
            "owner/dataset@sha": {
                "source": "owner/dataset",
                "revision": "sha",
                "default_subset": "default",
                "subsets": [
                    {
                        **subset,
                        "selectable": True,
                        "disabled_reason": None,
                    }
                ],
            }
        },
    )
    state.set_collection(
        "data_subset_selection",
        {
            "source": "owner/dataset",
            "revision": "sha",
            "subsets": [subset],
        },
    )

    issues = [
        issue
        for issue in validate_state(state)
        if issue.path.startswith("data.subsets")
    ]

    assert issues == []
