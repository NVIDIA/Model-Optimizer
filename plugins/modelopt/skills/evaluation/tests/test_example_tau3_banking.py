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

import shlex
from functools import partial

import pytest
from _gym_template import EXAMPLES, block

TEMPLATE = EXAMPLES / "example_tau3_banking.yaml"
USER_SIM = "++gpt-5_4-mini-2026-03-17.responses_api_models.openai_model."
_block = partial(block, TEMPLATE)


def _folded_tokens(key):
    header, block = _block(key)
    assert header.split(":", 1)[1].strip() == ">-"  # a folded scalar
    folded = " ".join(line.strip() for line in block.splitlines() if line.strip())
    tokens = shlex.split(folded, comments=True)
    assert tokens == shlex.split(folded)  # no `#` turning the tail into a shell comment
    return tokens


@pytest.mark.parametrize("key", ["extra_args", "prepare_args", "run_args"])
def test_folded_scalars_have_no_shell_comments(key):
    assert _folded_tokens(key)


def test_deployment_enables_tool_calling():
    tokens = _folded_tokens("extra_args")
    assert "--enable-auto-tool-choice" in tokens
    assert "--tool-call-parser" in tokens


def test_run_args_keep_declared_repeats_and_user_simulator_placeholders():
    args = dict(t.split("=", 1) for t in _folded_tokens("run_args"))
    assert not any(k.startswith("++num_repeats") for k in args)  # declared 5 = AA shape
    assert args[USER_SIM + "openai_base_url"] == "<TAU3_USER_BASE_URL>"
    assert args[USER_SIM + "openai_model"] == "<TAU3_USER_MODEL>"
    assert args[USER_SIM + "openai_api_key"] == "$INFERENCE_API_KEY"  # expanded at run time


def test_bootstrap_verifies_pin_and_gates_limit():
    header, command = _block("command")
    assert header.strip() == "command: |"
    assert '[ "$actual_gym_sha" != "$expected_gym_sha" ]' in command
    assert "{% if config.params.limit_samples is not none %}" in command
    assert 'set -- "$@" --limit "{{config.params.limit_samples}}"' in command


def test_mlflow_export_keeps_logs_private():
    # The user-simulator key is expanded into Gym's argv; keep the exporter defaults
    # (required artifacts only, no logs) so Gym's own logs are never uploaded.
    _, export = _block("export")
    assert "log_logs: true" not in export
    assert "only_required: false" not in export
