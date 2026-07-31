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

from types import SimpleNamespace

from puzzletron_setup.v2.defaults import DefaultsResolver
from puzzletron_setup.v2.prompts import ScriptedBackend
from puzzletron_setup.v2.resources import ParallelProfile, ResourceProfileRegistry
from puzzletron_setup.v2.session import WizardSession
from puzzletron_setup.v2.state import WizardState
from puzzletron_setup.v2.wizard import (
    _configure_stage_resource,
    _profile_prompt,
    _serving_setting_prompt,
)

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


class RecordingBackend(ScriptedBackend):
    def __init__(self, answers):
        super().__init__(answers)
        self.messages = []

    def text(self, message, default):
        self.messages.append(message)
        return super().text(message, default)

    def select(self, message, choices, default):
        self.messages.append(message)
        return super().select(message, choices, default)

    def checkbox(self, message, choices, defaults):
        self.messages.append(message)
        return super().checkbox(message, choices, defaults)


def _session(tmp_path, answers):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.set_field("data.sequence_length", 1024)
    state.set_collection("pruning", SELECTED)
    backend = RecordingBackend(answers)
    session = WizardSession(state, backend)
    session.begin("parallel-test")
    return session, backend


def test_profile_prompt_rejects_incompatible_reuse_and_asks_again(
    tmp_path,
    capsys,
):
    session, backend = _session(
        tmp_path,
        ["reuse:bad", "reuse:good"],
    )
    registry = ResourceProfileRegistry(
        {
            "bad": ParallelProfile(name="bad", tp=8),
            "good": ParallelProfile(name="good", tp=2),
        }
    )
    model = SimpleNamespace(inventory=MOE_INVENTORY)

    profile = _profile_prompt(
        session,
        registry,
        "width_sanity",
        model,
    )

    assert profile.name == "good"
    assert backend.remaining == 0
    output = capsys.readouterr().out
    assert "query-head counts [18, 24, 32]" in output
    assert "Choose a different parallel setting." in output


def test_default_stages_reuse_first_compatible_profile_without_reprompting(
    tmp_path,
):
    session, backend = _session(tmp_path, [])
    registry = ResourceProfileRegistry(
        {
            "bad": ParallelProfile(name="bad", tp=8),
            "good": ParallelProfile(name="good", tp=2),
        }
    )
    session.state.set_collection("parallel_profiles", registry.to_dict())
    resolver = DefaultsResolver()
    model = SimpleNamespace(inventory=MOE_INVENTORY)

    first = _configure_stage_resource(
        session,
        resolver,
        model,
        "width_sanity",
        action="defaults",
        batch_default=8,
    )
    second = _configure_stage_resource(
        session,
        resolver,
        model,
        "replacement_scoring",
        action="defaults",
        batch_default=8,
    )

    assert first.profile.name == "good"
    assert second.profile.name == "good"
    assert session.state.collection("stage_resources")["width_sanity"]["profile_name"] == "good"
    assert (
        session.state.collection("stage_resources")["replacement_scoring"]["profile_name"] == "good"
    )
    assert backend.remaining == 0


def test_serving_prompt_asks_aiperf_inputs_and_boolean_expert_parallel(tmp_path):
    session, backend = _session(
        tmp_path,
        [
            128,
            32,
            "1, 4, 8",
            8,
            "best_per_concurrency",
            1,
            1,
            8,
            1,
            1,
            True,
        ],
    )

    result = _serving_setting_prompt(
        session,
        "post.params.serving",
        {
            "input_tokens": 128,
            "output_tokens": 32,
            "concurrency": 2,
            "request_count": 8,
        },
        inventory=MOE_INVENTORY,
        pruning=SELECTED,
        stage_id="post.params.serving",
    )

    assert result["input_tokens"] == 128
    assert result["output_tokens"] == 32
    assert result["concurrency"] == [1, 4, 8]
    assert result["request_count"] == 8
    assert result["best_selection_mode"] == "best_per_concurrency"
    assert result["topology"] == {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 8,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "gpu_group_size": 8,
        "distributed_executor_backend": "mp",
    }
    assert backend.remaining == 0
    assert any("Serving concurrency sweep" in message for message in backend.messages)
    assert any("How should the best models be selected?" in message for message in backend.messages)
    assert any("effective EP is TP * DP" in message for message in backend.messages)


def test_serving_prompt_rejects_duplicate_concurrencies_and_reprompts(
    tmp_path,
    capsys,
):
    session, backend = _session(
        tmp_path,
        [
            128,
            32,
            "1, 1",
            "1, 2",
            8,
            "individual_best",
            1,
            1,
            8,
            1,
            1,
            True,
        ],
    )

    result = _serving_setting_prompt(
        session,
        "post.params.serving",
        {
            "input_tokens": 128,
            "output_tokens": 32,
            "concurrency": [1],
            "request_count": 8,
        },
        inventory=MOE_INVENTORY,
        pruning=SELECTED,
        stage_id="post.params.serving",
    )

    assert result["concurrency"] == [1, 2]
    assert backend.remaining == 0
    assert "Concurrency values must be unique." in capsys.readouterr().out


def test_serving_prompt_rejects_incompatible_topology_and_reprompts(
    tmp_path,
    capsys,
):
    session, backend = _session(
        tmp_path,
        [
            128,
            32,
            "2",
            8,
            "individual_best",
            4,
            1,
            8,
            1,
            1,
            True,
            2,
            1,
            4,
            1,
            1,
            True,
        ],
    )

    result = _serving_setting_prompt(
        session,
        "post.params.serving",
        {
            "input_tokens": 128,
            "output_tokens": 32,
            "concurrency": 2,
            "request_count": 8,
        },
        inventory=MOE_INVENTORY,
        pruning=SELECTED,
        stage_id="post.params.serving",
    )

    assert result["topology"]["tensor_parallel_size"] == 2
    assert result["topology"]["data_parallel_size"] == 4
    assert backend.remaining == 0
    output = capsys.readouterr().out
    assert "effective EP=32" in output
    assert "Choose a different parallel setting." in output


def test_vllm_topology_prompt_rejects_incompatible_setting_and_reprompts(
    tmp_path,
    capsys,
):
    from puzzletron_setup.v2.wizard import _vllm_topology_prompt

    session, backend = _session(
        tmp_path,
        [
            4,
            1,
            8,
            1,
            1,
            True,
            2,
            1,
            1,
            1,
            1,
            True,
        ],
    )

    topology = _vllm_topology_prompt(
        session,
        "vllm.measurements.latency",
        {},
        inventory=MOE_INVENTORY,
        pruning=SELECTED,
        stage_id="vllm_stats",
    )

    assert topology == {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "gpu_group_size": 2,
        "distributed_executor_backend": "mp",
    }
    assert backend.remaining == 0
    output = capsys.readouterr().out
    assert "effective EP=32" in output
    assert "requires DP=1" in output
    assert "Choose a different parallel setting." in output
