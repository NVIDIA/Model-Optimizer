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

"""Tests for the direct local-checkpoint lmms-eval CLI."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_PATH = _REPOSITORY_ROOT / "examples/puzzletron/evaluate_lmms_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_lmms_checkpoint", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
evaluate_lmms_checkpoint = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_lmms_checkpoint)


def test_runner_import_keeps_stdout_machine_readable(monkeypatch, capsys):
    runner = object()

    def fake_import_module(name):
        assert name == "modelopt.torch.puzzletron.evaluation"
        print("import-time diagnostic")
        return SimpleNamespace(run_lmms_eval_checkpoint=runner)

    monkeypatch.setattr(evaluate_lmms_checkpoint.importlib, "import_module", fake_import_module)

    assert evaluate_lmms_checkpoint._load_runner() is runner
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "import-time diagnostic\n"


def test_cli_help_explains_qwen_profile_and_native_escape_hatch():
    help_text = evaluate_lmms_checkpoint._build_parser().format_help()
    normalized_help = " ".join(help_text.split())

    assert "--model-profile {auto,none}" in help_text
    assert "Qwen 3.5" in help_text
    assert "reasoning_parser=qwen3" in help_text
    assert "python -m lmms_eval --help" in normalized_help


def test_cli_runs_generic_text_smoke_defaults(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()
    output_root = tmp_path / "results"
    upstream_task = tmp_path / "lmms_eval/tasks/gsm8k/gsm8k.yaml"
    upstream_task.parent.mkdir(parents=True)
    upstream_task.write_text("task: gsm8k\n")
    captured = {}

    def fake_run(checkpoint_path, *, output_root, settings):
        captured["checkpoint"] = checkpoint_path
        captured["output_root"] = output_root
        captured["settings"] = settings
        return {"result_path": "/results/attempt/summary.json", "metrics": {}}

    monkeypatch.setattr(evaluate_lmms_checkpoint, "run_lmms_eval_checkpoint", fake_run)
    monkeypatch.setattr(evaluate_lmms_checkpoint, "_lmms_eval_gsm8k_config", lambda: upstream_task)

    returncode = evaluate_lmms_checkpoint.main(
        ["--checkpoint", str(checkpoint), "--output-dir", str(output_root)]
    )

    assert returncode == 0
    assert captured == {
        "checkpoint": checkpoint.resolve(),
        "output_root": output_root,
        "settings": {
            "tasks": "ifeval,modelopt_gsm8k",
            "limit": 8,
            "batch_size": 1,
            "seed": 42,
            "timeout_seconds": evaluate_lmms_checkpoint.DEFAULT_SMOKE_TIMEOUT_SECONDS,
            "topology": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "prefill_context_parallel_size": 1,
                "decode_context_parallel_size": 1,
                "enable_expert_parallel": False,
                "distributed_executor_backend": "mp",
                "gpu_group_size": 1,
            },
            "model_args": {
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.85,
                "max_model_len": 8192,
                "trust_remote_code": False,
            },
            "extra_args": [
                "--include_path",
                str(output_root / "task_configs"),
            ],
        },
    }
    assert json.loads((output_root / "task_configs/modelopt_gsm8k.yaml").read_text()) == {
        "dataset_path": "openai/gsm8k",
        "fewshot_config": {"sampler": "default"},
        "include": str(upstream_task),
        "task": "modelopt_gsm8k",
    }
    assert json.loads(capsys.readouterr().out) == {
        "metrics": {},
        "result_path": "/results/attempt/summary.json",
    }


def test_cli_auto_configures_qwen_3_5_and_reports_the_choice(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5_text"},
            }
        )
    )
    captured = {}

    def fake_run(_checkpoint_path, *, output_root, settings):
        captured["settings"] = settings
        return {"result_path": str(output_root / "summary.json"), "metrics": {}}

    monkeypatch.setattr(evaluate_lmms_checkpoint, "run_lmms_eval_checkpoint", fake_run)

    returncode = evaluate_lmms_checkpoint.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "results"),
            "--tasks",
            "ifeval",
        ]
    )

    assert returncode == 0
    assert captured["settings"]["model_args"]["reasoning_parser"] == "qwen3"
    captured_streams = capsys.readouterr()
    assert json.loads(captured_streams.out)["metrics"] == {}
    assert "Detected Qwen 3.5 checkpoint" in captured_streams.err
    assert "reasoning_parser=qwen3" in captured_streams.err


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (["--model-profile", "none"], None),
        (["--reasoning-parser", "custom"], "custom"),
    ],
)
def test_cli_can_disable_or_override_qwen_3_5_profile(tmp_path, extra_args, expected):
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "qwen3_5"}\n')

    args = evaluate_lmms_checkpoint._build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "results"),
            *extra_args,
        ]
    )

    assert (
        evaluate_lmms_checkpoint._settings(args)["model_args"].get("reasoning_parser") == expected
    )


def test_cli_ignores_unrecognized_or_malformed_model_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type": ["not", "a", "string"], "text_config": []}\n'
    )

    assert evaluate_lmms_checkpoint._automatic_model_args(checkpoint) == {}


def test_cli_full_run_and_runtime_overrides_are_wired(tmp_path):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()

    args = evaluate_lmms_checkpoint._build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "results"),
            "--full",
            "--tensor-parallel-size",
            "2",
            "--reasoning-parser",
            "qwen3",
            "--trust-remote-code",
        ]
    )

    settings = evaluate_lmms_checkpoint._settings(args)
    assert settings["limit"] is None
    assert settings["timeout_seconds"] == evaluate_lmms_checkpoint.DEFAULT_FULL_TIMEOUT_SECONDS
    assert settings["topology"]["tensor_parallel_size"] == 2
    assert settings["topology"]["gpu_group_size"] == 2
    assert settings["model_args"]["reasoning_parser"] == "qwen3"
    assert settings["model_args"]["trust_remote_code"] is True


def test_cli_maps_gsm8k_to_namespaced_compatibility_task(tmp_path):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()

    args = evaluate_lmms_checkpoint._build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "results"),
            "--tasks",
            " ifeval, gsm8k,custom_task ",
        ]
    )

    assert evaluate_lmms_checkpoint._settings(args)["tasks"] == (
        "ifeval,modelopt_gsm8k,custom_task"
    )


def test_cli_rejects_empty_task_names(tmp_path):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()

    with pytest.raises(SystemExit):
        evaluate_lmms_checkpoint._build_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "results"),
                "--tasks",
                "ifeval,,gsm8k",
            ]
        )


def test_compatibility_task_is_not_written_when_gsm8k_is_not_selected(tmp_path):
    assert (
        evaluate_lmms_checkpoint._prepare_compatibility_tasks(
            tmp_path / "results", "ifeval,custom_task"
        )
        is None
    )
    assert not (tmp_path / "results").exists()


def test_cli_explicit_timeout_overrides_full_default(tmp_path):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()

    args = evaluate_lmms_checkpoint._build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "results"),
            "--full",
            "--timeout-seconds",
            "123",
        ]
    )

    assert evaluate_lmms_checkpoint._settings(args)["timeout_seconds"] == 123


def test_cli_reports_failed_attempt_evidence_payload(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()
    failure = RuntimeError("lmms-eval failed")
    failure.command_path = "/results/attempt/command.json"
    failure.stdout_path = "/results/attempt/stdout.txt"
    failure.stderr_path = "/results/attempt/stderr.txt"

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(evaluate_lmms_checkpoint, "run_lmms_eval_checkpoint", fail)
    monkeypatch.setattr(
        evaluate_lmms_checkpoint,
        "_lmms_eval_gsm8k_config",
        lambda: tmp_path / "lmms_eval/tasks/gsm8k/gsm8k.yaml",
    )

    returncode = evaluate_lmms_checkpoint.main(
        ["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "results")]
    )

    assert returncode == 1
    assert json.loads(capsys.readouterr().err) == {
        "command_path": failure.command_path,
        "error": "RuntimeError",
        "message": "lmms-eval failed",
        "stderr_path": failure.stderr_path,
        "stdout_path": failure.stdout_path,
    }
