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
"""Exercise generator control flow with a stub client, without requiring the OpenAI SDK."""

import copy
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/speculative_decoding"


class _ConnectionError(Exception):
    """Stand in for a network failure without importing the client SDK."""


class _StatusError(Exception):
    """Expose the HTTP status used by the generator's failure classifier."""

    def __init__(self, status_code):
        super().__init__(f"Request failed with status {status_code}")
        self.status_code = status_code


def _response(content="answer", finish_reason="stop", stop_reason=None, **message_fields):
    """Build only the response attributes consumed by the generator."""
    message = SimpleNamespace(
        content=content, tool_calls=None, function_call=None, **message_fields
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, stop_reason=stop_reason)
    return SimpleNamespace(choices=[choice])


def _sample(prompt):
    """Build a user-only conversation without loading a dataset."""
    return {"messages": [{"role": "user", "content": prompt}]}


def _read_jsonl(path):
    """Read the generated output or journal, treating absent files as empty."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def run_generator(monkeypatch, tmp_path):
    """Run the real CLI and files, replacing only the optional network client boundary."""
    data_path = tmp_path / "input.json"
    output_path = tmp_path / "shards/output.jsonl"

    def run(samples, responses, *options):
        data_path.write_text(json.dumps(samples))
        responses = iter(responses)
        requests = []

        def create(**kwargs):
            # Later turns mutate the same history list passed to the client.
            requests.append(copy.deepcopy(kwargs))
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        openai_stub = ModuleType("openai")
        openai_stub.OpenAI = Mock(return_value=client)
        openai_stub.APIConnectionError = _ConnectionError
        openai_stub.APIStatusError = _StatusError
        monkeypatch.setitem(sys.modules, "openai", openai_stub)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "server_generate.py",
                "--data_path",
                str(data_path),
                "--output_path",
                str(output_path),
                "--num_threads",
                "1",
                "--log_empty_conversations",
                *options,
            ],
        )
        exit_code = 0
        try:
            runpy.run_path(str(_EXAMPLE / "scripts/server_generate.py"), run_name="__main__")
        except SystemExit as exc:
            exit_code = exc.code
        return SimpleNamespace(
            exit_code=exit_code,
            rows=_read_jsonl(output_path),
            failures=_read_jsonl(Path(str(output_path) + ".failures")),
            requests=requests,
            client_factory=openai_stub.OpenAI,
            output_path=output_path,
        )

    return run


@pytest.mark.parametrize("sharegpt", [False, True])
def test_multi_turn_history_and_request_options(run_generator, sharegpt):
    """Regenerate each user turn using generated history and configured request options."""
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "discard this reference answer"},
        {"role": "user", "content": "second"},
    ]
    sample = {"messages": messages}
    if sharegpt:
        roles = {"system": "system", "user": "human", "assistant": "gpt"}
        sample = {
            "conversations": [{"from": roles[m["role"]], "value": m["content"]} for m in messages]
        }
    result = run_generator(
        [sample],
        [_response(" first answer ", reasoning="thinking"), _response("second answer")],
        "--max_tokens",
        "0",
        "--request_timeout",
        "90",
        "--extra_body",
        '{"chat_template_kwargs": {"enable_thinking": true}}',
    )
    generated = {"role": "assistant", "content": "first answer", "reasoning_content": "thinking"}
    assert result.exit_code == 0
    assert result.requests[0]["messages"] == messages[:2]
    assert result.requests[1]["messages"] == [*messages[:2], generated, messages[3]]
    assert all(request["max_tokens"] is None for request in result.requests)
    assert all(
        request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
        for request in result.requests
    )
    assert result.client_factory.call_args.kwargs["timeout"] == 90
    assert result.rows == [
        {
            "conversation_id": 0,
            "conversations": [
                *messages[:2],
                generated,
                messages[3],
                {"role": "assistant", "content": "second answer"},
            ],
        },
        {"finished": True},
    ]
    assert not result.failures


@pytest.mark.parametrize("has_system_message", [False, True])
def test_system_prompt_override_warns_only_when_replacing_input(
    run_generator, capsys, has_system_message
):
    """Warn when overriding a dataset system message without exposing its contents."""
    sample = _sample("question")
    if has_system_message:
        sample["messages"].insert(0, {"role": "system", "content": "dataset instructions"})
    result = run_generator([sample], [_response()], "--system_prompt", "override instructions")
    assert result.exit_code == 0
    assert result.requests[0]["messages"] == [
        {"role": "system", "content": "override instructions"},
        {"role": "user", "content": "question"},
    ]
    stderr = capsys.readouterr().err
    assert ("--system_prompt overrides the input system message" in stderr) == has_system_message
    assert "dataset instructions" not in stderr


@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("strict", [False, True])
def test_empty_answer_resume_depends_on_temperature(run_generator, temperature, strict):
    """Retry sampled empty answers on ordinary resume without keeping partial conversations."""
    samples = [
        {"messages": [*_sample("first")["messages"], *_sample("second")["messages"]]},
        _sample("keep me"),
    ]
    options = ["--temperature", str(temperature)]
    if strict:
        options.append("--fail_on_error")
    first = run_generator(
        samples,
        [_response(), _response(content=" ", reasoning_content="no final answer"), _response()],
        *options,
    )
    retryable = temperature > 0
    assert first.exit_code == int(strict)
    assert all(request["temperature"] == temperature for request in first.requests)
    assert [row["conversation_id"] for row in first.rows if "conversation_id" in row] == [1]
    assert len(first.failures) == 1
    assert first.failures[0]["conversation_id"] == 0
    assert first.failures[0]["retryable"] == retryable
    assert (first.rows[-1] == {"finished": True}) == (not retryable)

    responses = [_response(), _response()] if retryable else []
    resumed = run_generator(samples, responses, *options)
    assert resumed.exit_code == int(strict and not retryable)
    assert len(resumed.requests) == len(responses)
    assert [row["conversation_id"] for row in resumed.rows if "conversation_id" in row] == (
        [1, 0] if retryable else [1]
    )
    assert resumed.rows[-1] == {"finished": True}


@pytest.mark.parametrize(
    "error", [_ConnectionError("connection lost"), _StatusError(429), _StatusError(503)]
)
def test_transient_failure_resumes_without_duplicates(run_generator, error):
    """Retry transient failures while preserving previously written conversations."""
    samples = [_sample("retry me"), _sample("keep me")]
    first = run_generator(samples, [error, _response()])
    assert first.exit_code == 0
    assert [row["conversation_id"] for row in first.rows] == [1]
    assert len(first.failures) == 1
    assert first.failures[0]["conversation_id"] == 0
    assert first.failures[0]["retryable"] is True

    resumed = run_generator(samples, [_response()])
    assert resumed.exit_code == 0
    assert len(resumed.requests) == 1
    assert resumed.requests[0]["messages"] == samples[0]["messages"]
    assert [row["conversation_id"] for row in resumed.rows[:-1]] == [1, 0]
    assert resumed.rows[-1] == {"finished": True}
    completed = run_generator(samples, [], "--fail_on_error")
    assert completed.exit_code == 0
    assert completed.requests == []
    assert completed.rows == resumed.rows


@pytest.mark.parametrize(
    "error",
    [
        _StatusError(400),
        _StatusError(422),
        _response(content="", reasoning_content="no final answer"),
    ],
)
def test_rejected_sample_continues_and_requires_explicit_retry(
    run_generator, error, monkeypatch, tmp_path
):
    """Skip rejections on ordinary resume and keep their journal out of combined data."""
    samples = [_sample("reject me"), _sample("keep me")]
    first = run_generator(samples, [error, _response()])
    assert first.exit_code == 0
    assert [row.get("conversation_id") for row in first.rows] == [1, None]
    assert first.rows[-1] == {"finished": True}
    assert len(first.failures) == 1
    assert first.failures[0]["retryable"] is False
    skipped = run_generator(samples, [])
    assert skipped.exit_code == 0
    assert skipped.requests == []

    combined = tmp_path / "combined.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sharding_utils.py",
            "--combine",
            "--input_dir",
            str(first.output_path.parent),
            "--output_path",
            str(combined),
        ],
    )
    runpy.run_path(str(_EXAMPLE / "distributed_generate/sharding_utils.py"), run_name="__main__")
    assert _read_jsonl(combined) == [{"conversations": first.rows[0]["conversations"]}]

    retried = run_generator(samples, [_response()], "--retry_failed", "--fail_on_error")
    assert retried.exit_code == 0
    assert len(retried.requests) == 1
    assert [row["conversation_id"] for row in retried.rows if "conversation_id" in row] == [1, 0]
    assert run_generator(samples, [], "--fail_on_error").exit_code == 0


def test_retryable_failure_overrides_old_completion_marker(run_generator):
    """Resume a transient failure even when an earlier run wrote a completion marker."""
    samples = [_sample("retry me")]
    rejected = run_generator(samples, [_StatusError(400)])
    assert rejected.rows == [{"finished": True}]
    retry = run_generator(samples, [_StatusError(503)], "--retry_failed")
    assert retry.exit_code == 0
    assert retry.rows == rejected.rows
    assert retry.failures[-1]["retryable"] is True
    resumed = run_generator(samples, [_response()])
    assert resumed.exit_code == 0
    assert len(resumed.requests) == 1
    assert [row["conversation_id"] for row in resumed.rows if "conversation_id" in row] == [0]


def test_strict_mode_reports_all_failures_and_keeps_successes(run_generator, capsys):
    """Report the whole batch and return nonzero until all strict-mode failures resolve."""
    samples = [_sample("reject"), _sample("transient"), _sample("success")]
    first = run_generator(
        samples, [_StatusError(400), _StatusError(503), _response()], "--fail_on_error"
    )
    assert first.exit_code == 1
    assert {failure["conversation_id"] for failure in first.failures} == {0, 1}
    assert [row["conversation_id"] for row in first.rows] == [2]
    assert "2 conversations failed (1 retryable)" in capsys.readouterr().err
    resumed = run_generator(samples, [_response()], "--fail_on_error")
    assert resumed.exit_code == 1
    assert len(resumed.requests) == 1
    assert resumed.rows[-1] == {"finished": True}
    completed = run_generator(samples, [], "--fail_on_error")
    assert completed.exit_code == 1
    assert completed.requests == []


@pytest.mark.parametrize("status", [401, 404])
def test_fatal_request_error_exits_nonzero(run_generator, status):
    """Keep server configuration errors fatal even without strict mode."""
    result = run_generator([_sample("fatal")], [_StatusError(status)])
    assert result.exit_code == 1
    assert result.rows == []
    assert result.failures[0]["retryable"] is True


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [("length", None), ("repetition", None), ("stop", "repetition_detected")],
)
def test_truncation_stops_remaining_turns(run_generator, finish_reason, stop_reason):
    """Persist the incomplete response and its stop metadata without generating later turns."""
    sample = {"messages": [*_sample("first")["messages"], *_sample("second")["messages"]]}
    result = run_generator([sample], [_response("partial", finish_reason, stop_reason)])
    assert result.exit_code == 0
    assert len(result.requests) == 1
    assert result.rows[0] == {
        "conversation_id": 0,
        "conversations": [sample["messages"][0], {"role": "assistant", "content": "partial"}],
        "truncated": True,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
    }
    assert not result.failures


def test_unsupported_role_does_not_write_partial_conversation(run_generator):
    """Reject unsupported roles without persisting an earlier successful turn."""
    sample = {
        "messages": [*_sample("first")["messages"], {"role": "tool", "content": "unsupported"}]
    }
    result = run_generator(
        [sample, _sample("success")], [_response(), _response()], "--temperature", "0.7"
    )
    assert result.exit_code == 0
    assert [row.get("conversation_id") for row in result.rows] == [1, None]
    assert result.failures[0]["conversation_id"] == 0
    assert result.failures[0]["retryable"] is False
    assert "Unsupported message role" in result.failures[0]["error"]
