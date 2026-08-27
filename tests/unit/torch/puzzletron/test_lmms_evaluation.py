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

"""Tests for the reusable lmms-eval checkpoint backend."""

import json
import os
import sys
from pathlib import Path

import pytest

from modelopt.torch.puzzletron.evaluation import lmms


def _settings(*tasks: str) -> dict:
    return {"tasks": list(tasks), "topology": {"gpu_group_size": 1}}


def _write_result(
    output: Path,
    *,
    results: dict,
    sample_counts: dict,
    group_subtasks: dict | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(
            {
                "results": results,
                "group_subtasks": group_subtasks or {},
                "n-samples": sample_counts,
            }
        )
    )


def test_command_maps_checkpoint_and_vllm_topology(tmp_path):
    argv, env, timeout = lmms._build_command(
        {
            "tasks": ["ifeval", "gsm8k"],
            "batch_size": 2,
            "limit": 8,
            "cache_dir": tmp_path / "cache",
            "timeout_seconds": 123,
            "topology": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 2,
                "data_parallel_size": 1,
                "prefill_context_parallel_size": 1,
                "decode_context_parallel_size": 1,
                "enable_expert_parallel": False,
                "gpu_group_size": 8,
            },
            "model_args": {"dtype": "bfloat16"},
        },
        checkpoint="/ckpts/candidate",
        output_path=tmp_path / "results",
    )

    model_args = argv[argv.index("--model_args") + 1]
    assert argv[:5] == [sys.executable, "-m", "lmms_eval", "--model", "vllm"]
    assert argv[argv.index("--tasks") + 1] == "ifeval,gsm8k"
    assert argv[argv.index("--batch_size") + 1] == "2"
    assert argv[argv.index("--limit") + 1] == "8"
    assert "model=/ckpts/candidate" in model_args
    assert "tensor_parallel_size=4" in model_args
    assert "pipeline_parallel_size=2" in model_args
    assert "gpu_group_size" not in model_args
    assert env["LMMS_EVAL_HOME"] == str(tmp_path / "cache")
    assert timeout == 123


def test_command_uses_bounded_default_timeout(tmp_path):
    _, _, timeout = lmms._build_command(
        _settings("ifeval"),
        checkpoint="/ckpts/candidate",
        output_path=tmp_path / "results",
    )

    assert timeout == lmms.DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_command_rejects_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite positive number"):
        lmms._build_command(
            {**_settings("ifeval"), "timeout_seconds": timeout},
            checkpoint="/ckpts/candidate",
            output_path=tmp_path / "results",
        )


@pytest.mark.parametrize(
    ("model_args", "expected"),
    [
        ({"model": "/ckpts/wrong"}, "model"),
        ("dtype=bfloat16,tensor_parallel_size=1", "tensor_parallel_size"),
    ],
)
def test_command_rejects_reserved_model_args_setting(tmp_path, model_args, expected):
    with pytest.raises(ValueError, match="reserved lmms-eval model arguments") as exc_info:
        lmms._build_command(
            {**_settings("ifeval"), "model_args": model_args},
            checkpoint="/ckpts/candidate",
            output_path=tmp_path / "results",
        )

    assert expected in str(exc_info.value)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"model": "hf"}, "settings.model must be 'vllm'"),
        ({"checkpoint_arg": "pretrained"}, "settings.checkpoint_arg must be 'model'"),
    ],
)
def test_command_rejects_unsupported_backend_contract(tmp_path, settings, expected):
    with pytest.raises(ValueError, match=expected):
        lmms._build_command(
            {**_settings("ifeval"), **settings},
            checkpoint="/ckpts/candidate",
            output_path=tmp_path / "results",
        )


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
def test_timeout_kills_ignored_process_group_members(monkeypatch, tmp_path):
    script = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('partial stdout', flush=True); "
        "time.sleep(60)"
    )
    monkeypatch.setattr(lmms, "_PROCESS_CLEANUP_TIMEOUT_SECONDS", 0.1)

    with pytest.raises(lmms.LmmsEvalTimeoutError) as exc_info:
        lmms._run_process(
            [sys.executable, "-c", script],
            cwd=str(tmp_path),
            env=os.environ.copy(),
            timeout=1.0,
        )

    assert exc_info.value.timeout == 1.0
    assert exc_info.value.output == "partial stdout\n"
    assert exc_info.value.stderr == ""


def test_run_checkpoint_flattens_metrics_and_preserves_artifacts(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    def fake_run(argv, *, cwd, env, timeout):
        del env, timeout
        _write_result(
            Path(cwd) / "nested",
            results={
                "ifeval": {"prompt_level_strict_acc,none": 0.5},
                "gsm8k": {"exact_match,strict-match": 0.75},
            },
            sample_counts={
                "ifeval": {"original": 541, "effective": 4},
                "gsm8k": {"original": 1319, "effective": 4},
            },
        )
        return lmms._ProcessResult(argv, 0, stdout="done\n", stderr="")

    monkeypatch.setattr(lmms, "_run_process", fake_run)

    result = lmms.run_lmms_eval_checkpoint(
        checkpoint,
        output_root=tmp_path / "results",
        settings={**_settings("ifeval", "gsm8k"), "limit": 4},
    )

    assert result["metrics"] == {
        "gsm8k.exact_match_strict-match": 0.75,
        "ifeval.prompt_level_strict_acc_none": 0.5,
    }
    assert Path(result["command_path"]).is_file()
    assert Path(result["stdout_path"]).read_text() == "done\n"
    assert Path(result["stderr_path"]).read_text() == ""
    summary = json.loads(Path(result["result_path"]).read_text())
    assert summary["checkpoint"] == str(checkpoint.resolve())
    assert summary["raw_result_path"] == result["raw_result_path"]
    assert "result_path" not in summary
    assert summary["sample_counts"] == {"gsm8k": 4.0, "ifeval": 4.0}


def test_run_checkpoint_executes_real_process(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    script = """
import json
from pathlib import Path

Path("results.json").write_text(json.dumps({
    "results": {"ifeval": {"accuracy": 0.5}},
    "n-samples": {"ifeval": {"effective": 2}},
}))
print("evaluator stdout")
print("evaluator stderr", file=__import__("sys").stderr)
"""

    result = lmms.run_lmms_eval_checkpoint(
        checkpoint,
        output_root=tmp_path / "results",
        settings={
            **_settings("ifeval"),
            "command_prefix": [sys.executable, "-c", script],
        },
    )

    assert result["metrics"] == {"ifeval.accuracy": 0.5}
    assert Path(result["stdout_path"]).read_text() == "evaluator stdout\n"
    assert Path(result["stderr_path"]).read_text() == "evaluator stderr\n"
    assert Path(result["raw_result_path"]).name == "results.json"
    command = json.loads(Path(result["command_path"]).read_text())
    assert command["argv"][:2] == [sys.executable, "-c"]


def test_run_checkpoint_preserves_failure_artifacts(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        lmms,
        "_run_process",
        lambda argv, **kwargs: lmms._ProcessResult(
            argv, 2, stdout="partial evaluator output\n", stderr="backend failed\n"
        ),
    )

    with pytest.raises(RuntimeError, match="lmms-eval failed with exit code 2") as exc_info:
        lmms.run_lmms_eval_checkpoint(
            checkpoint,
            output_root=tmp_path / "results",
            settings=_settings("ifeval"),
        )

    error = exc_info.value
    assert Path(error.command_path).is_file()
    assert Path(error.stdout_path).read_text() == "partial evaluator output\n"
    assert Path(error.stderr_path).read_text() == "backend failed\n"


def test_run_checkpoint_preserves_timeout_artifacts(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    def time_out(argv, **_kwargs):
        raise lmms.LmmsEvalTimeoutError(
            argv,
            7,
            output="partial evaluator output\n",
            stderr="evaluation timed out\n",
        )

    monkeypatch.setattr(lmms, "_run_process", time_out)

    with pytest.raises(lmms.LmmsEvalTimeoutError) as exc_info:
        lmms.run_lmms_eval_checkpoint(
            checkpoint,
            output_root=tmp_path / "results",
            settings=_settings("ifeval"),
        )

    error = exc_info.value
    assert Path(error.command_path).is_file()
    assert Path(error.stdout_path).read_text() == "partial evaluator output\n"
    assert Path(error.stderr_path).read_text() == "evaluation timed out\n"


def test_completion_validates_resolved_task_expansion():
    sample_counts = lmms._validate_completion(
        {
            "results": {
                "arc_challenge": {"acc,none": 0.25},
                "hellaswag": {"acc_norm,none": 0.5},
            },
            "group_subtasks": {"leaderboard": ["arc_challenge", "hellaswag"]},
            "n-samples": {
                "arc_challenge": {"original": 1172, "effective": 8},
                "hellaswag": {"original": 10042, "effective": 8},
            },
        },
        ("leaderboard",),
    )

    assert sample_counts == {"arc_challenge": 8.0, "hellaswag": 8.0}


@pytest.mark.parametrize(
    ("results", "sample_counts", "expected"),
    [
        (
            {"ifeval": {"accuracy": 0.5}},
            {"ifeval": {"effective": 4}},
            "missing configured task results",
        ),
        (
            {"ifeval": {"accuracy": 0.5}, "gsm8k": {"accuracy": 0.75}},
            {"ifeval": {"effective": 4}, "gsm8k": {"effective": 0}},
            "zero effective samples",
        ),
    ],
)
def test_run_checkpoint_rejects_incomplete_results(
    monkeypatch, tmp_path, results, sample_counts, expected
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    def fake_run(argv, *, cwd, env, timeout):
        del env, timeout
        _write_result(Path(cwd), results=results, sample_counts=sample_counts)
        return lmms._ProcessResult(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(lmms, "_run_process", fake_run)

    with pytest.raises(RuntimeError, match=expected):
        lmms.run_lmms_eval_checkpoint(
            checkpoint,
            output_root=tmp_path / "results",
            settings=_settings("ifeval", "gsm8k"),
        )


def test_run_checkpoint_requires_local_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint is not a local directory"):
        lmms.run_lmms_eval_checkpoint(
            tmp_path / "missing",
            output_root=tmp_path / "results",
            settings=_settings("ifeval"),
        )
