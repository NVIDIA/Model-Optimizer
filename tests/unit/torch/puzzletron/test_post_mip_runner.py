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

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.post_mip import runner
from modelopt.torch.puzzletron.post_mip.records import ArtifactKind
from modelopt.torch.puzzletron.post_mip.runner import (
    _exception_diagnostics,
    _needs_puzzletron_process_group,
    _post_mip_kd_settings,
    _worker_group,
)


def test_worker_group_uses_torchrun_world_size(monkeypatch):
    monkeypatch.setenv("PUZZLETRON_GROUP_RANK", "0")
    monkeypatch.setenv("PUZZLETRON_GROUP_SIZE", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("PUZZLETRON_TASK_LAUNCHER", "torchrun")

    assert _worker_group() == (1, 2)


def test_worker_group_uses_puzzletron_identity_for_direct_tasks(monkeypatch):
    monkeypatch.setenv("PUZZLETRON_GROUP_RANK", "0")
    monkeypatch.setenv("PUZZLETRON_GROUP_SIZE", "1")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", "7")
    monkeypatch.setenv("PUZZLETRON_TASK_LAUNCHER", "direct")

    assert _worker_group() == (0, 1)


def test_exception_diagnostics_preserve_traceback():
    try:
        raise RuntimeError()
    except RuntimeError as error:
        diagnostics = _exception_diagnostics(error)

    assert diagnostics["error"] == "RuntimeError"
    assert "raise RuntimeError()" in diagnostics["traceback"]


def test_global_kd_lets_automodel_initialize_its_nccl_process_group():
    assert _needs_puzzletron_process_group("evaluation")
    assert not _needs_puzzletron_process_group("global_kd")


def test_post_mip_kd_always_requests_a_consolidated_output():
    settings = _post_mip_kd_settings(
        {"global_distillation": {"save_consolidated": False}},
        {"max_steps": 8},
    )

    assert settings["save_consolidated"] is True
    assert settings["max_steps"] == 8


def test_online_eval_settings_deep_merge_automodel_overrides():
    scoring = OmegaConf.create(
        {
            "eval_samples": 32,
            "automodel": {
                "force_hf": False,
                "use_puzzletron_dataloader": True,
                "parallel": {"tp": 1, "pp": 1, "dp_shard": 1},
            },
        }
    )

    merged = runner._merge_scoring_settings(
        scoring,
        {
            "eval_samples": 128,
            "automodel": {
                "teacher_cache_device": "cuda",
                "parallel": {"pp": 2, "dp_shard": 2},
            },
        },
    )

    assert merged.eval_samples == 128
    assert merged.automodel.force_hf is False
    assert merged.automodel.use_puzzletron_dataloader is True
    assert merged.automodel.teacher_cache_device == "cuda"
    assert dict(merged.automodel.parallel) == {"tp": 1, "pp": 2, "dp_shard": 2}


def test_online_eval_injects_resolved_hidden_width_into_solution(monkeypatch):
    source = SimpleNamespace(
        artifact={"hidden_width": 1792},
    )
    monkeypatch.setattr(
        runner,
        "_raw_solution",
        lambda _source: {"chosen_replacements": [{"layer_replacement": {}}]},
    )
    monkeypatch.setattr(
        runner,
        "_scenario_checkpoint_roles",
        lambda scenario, width: (Path("/sorted"), None),
    )

    work = runner._config_evaluation_work(
        {"puzzle_dir": "/puzzle"},
        "revision-1",
        source,
    )

    assert work.hidden_width == 1792
    assert work.raw_solution["hidden_width"] == 1792


def test_aiperf_consumes_request_count_without_forwarding_setup_only_keys(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_run_aiperf_sweep(checkpoint, **settings):
        captured["checkpoint"] = checkpoint
        captured.update(settings)
        return [SimpleNamespace(concurrency=8, metrics={}, raw_artifacts={})]

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.run_aiperf_sweep",
        fake_run_aiperf_sweep,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    node = SimpleNamespace(
        node_id="serving",
        flow_id="params",
        config={
            "config": {
                "concurrency": [8],
                "request_count": 23,
                "minimum_request_count": 4,
                "requests_per_concurrency": 2,
                "best_selection_mode": "individual_best",
                "input_tokens": 1024,
                "output_tokens": 128,
                "topology": {"gpu_group_size": 1},
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    result = runner._aiperf(
        {"puzzle_dir": str(tmp_path)},
        node,
        source,
        "execution",
    )

    assert captured["checkpoint"] == str(tmp_path / "checkpoint")
    assert captured["concurrencies"] == (8,)
    assert captured["request_counts"] == {8: 23}
    assert "request_count" not in captured
    assert "minimum_request_count" not in captured
    assert "requests_per_concurrency" not in captured
    assert "best_selection_mode" not in captured
    assert result["metrics"] == {}


def test_lmms_eval_command_maps_checkpoint_and_vllm_topology(tmp_path):
    argv, env, timeout = runner._lmms_eval_command(
        {
            "command_prefix": ["python", "-m", "lmms_eval"],
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
    assert argv[:5] == ["python", "-m", "lmms_eval", "--model", "vllm"]
    assert argv[argv.index("--tasks") + 1] == "ifeval,gsm8k"
    assert argv[argv.index("--batch_size") + 1] == "2"
    assert argv[argv.index("--limit") + 1] == "8"
    assert "model=/ckpts/candidate" in model_args
    assert "tensor_parallel_size=4" in model_args
    assert "pipeline_parallel_size=2" in model_args
    assert "gpu_group_size" not in model_args
    assert env["LMMS_EVAL_HOME"] == str(tmp_path / "cache")
    assert timeout == 123


def test_lmms_eval_command_uses_bounded_default_timeout(tmp_path):
    _, _, timeout = runner._lmms_eval_command(
        {
            "tasks": ["ifeval"],
            "topology": {"gpu_group_size": 1},
        },
        checkpoint="/ckpts/candidate",
        output_path=tmp_path / "results",
    )

    assert timeout == runner._DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS


def test_lmms_eval_command_rejects_reserved_model_args(tmp_path):
    cases = (
        ({"model": "/ckpts/wrong"}, "model"),
        ("dtype=bfloat16,tensor_parallel_size=1", "tensor_parallel_size"),
    )
    for model_args, expected in cases:
        try:
            runner._lmms_eval_command(
                {
                    "tasks": ["ifeval"],
                    "topology": {"gpu_group_size": 1},
                    "model_args": model_args,
                },
                checkpoint="/ckpts/candidate",
                output_path=tmp_path / "results",
            )
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("expected reserved lmms-eval model_args to fail")

        assert "reserved lmms-eval model arguments" in message
        assert expected in message


def test_lmms_eval_command_rejects_reserved_extra_args(tmp_path):
    cases = (
        (["--tasks", "gsm8k"], "--tasks"),
        ("--output_path /tmp/other", "--output_path"),
        (["--model_args=model=/ckpts/wrong"], "--model_args"),
    )
    for extra_args, expected in cases:
        try:
            runner._lmms_eval_command(
                {
                    "tasks": ["ifeval"],
                    "topology": {"gpu_group_size": 1},
                    "extra_args": extra_args,
                },
                checkpoint="/ckpts/candidate",
                output_path=tmp_path / "results",
            )
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("expected reserved lmms-eval extra_args to fail")

        assert "reserved lmms-eval flags" in message
        assert expected in message


def test_lmms_eval_timeout_terminates_process_group(monkeypatch, tmp_path):
    created = []
    signals = []

    class FakeProcess:
        pid = 1234
        returncode = None

        def __init__(self):
            self.communicate_timeouts = []

        def communicate(self, timeout=None):
            self.communicate_timeouts.append(timeout)
            if len(self.communicate_timeouts) == 1:
                raise subprocess.TimeoutExpired(
                    ["python", "-m", "lmms_eval"],
                    timeout,
                    output="partial stdout",
                    stderr="partial stderr",
                )
            self.returncode = -signal.SIGTERM
            return "partial stdout", "partial stderr"

    def fake_popen(argv, **kwargs):
        process = FakeProcess()
        created.append((argv, kwargs, process))
        return process

    def fake_killpg(pid, signal_number):
        if signal_number == 0:
            raise ProcessLookupError
        signals.append((pid, signal_number))

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner.os, "killpg", fake_killpg)

    try:
        runner._run_lmms_eval_process(
            ["python", "-m", "lmms_eval"],
            cwd=str(tmp_path),
            env={},
            timeout=7.0,
        )
    except subprocess.TimeoutExpired as error:
        assert error.timeout == 7.0
        assert error.output == "partial stdout"
        assert error.stderr == "partial stderr"
    else:
        raise AssertionError("expected lmms-eval timeout to be raised")

    argv, kwargs, process = created[0]
    assert argv == ["python", "-m", "lmms_eval"]
    assert kwargs["start_new_session"] is True
    assert process.communicate_timeouts == [
        7.0,
        runner._LMMS_EVAL_PROCESS_CLEANUP_TIMEOUT_SECONDS,
    ]
    assert signals == [(1234, signal.SIGTERM)]


def test_lmms_eval_timeout_kills_remaining_process_group(monkeypatch, tmp_path):
    signals = []

    class FakeProcess:
        pid = 3456
        returncode = None

        def communicate(self, timeout=None):
            if timeout == 7.0:
                raise subprocess.TimeoutExpired(
                    ["python", "-m", "lmms_eval"],
                    timeout,
                    output="partial stdout",
                    stderr="partial stderr",
                )
            self.returncode = -signal.SIGTERM
            return "partial stdout", "partial stderr"

    def fake_killpg(pid, signal_number):
        if signal_number != 0:
            signals.append((pid, signal_number))

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(runner.os, "killpg", fake_killpg)

    try:
        runner._run_lmms_eval_process(
            ["python", "-m", "lmms_eval"],
            cwd=str(tmp_path),
            env={},
            timeout=7.0,
        )
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("expected lmms-eval timeout to be raised")

    assert signals == [(3456, signal.SIGTERM), (3456, signal.SIGKILL)]


def test_lmms_eval_timeout_kills_stubborn_process_group(monkeypatch, tmp_path):
    signals = []

    class FakeProcess:
        pid = 5678
        returncode = None

        def __init__(self):
            self.communicate_timeouts = []

        def communicate(self, timeout=None):
            self.communicate_timeouts.append(timeout)
            if len(self.communicate_timeouts) < 3:
                raise subprocess.TimeoutExpired(
                    ["python", "-m", "lmms_eval"],
                    timeout,
                    output="partial stdout",
                    stderr="partial stderr",
                )
            self.returncode = -signal.SIGKILL
            return "partial stdout", "partial stderr"

    process = FakeProcess()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda pid, signal_number: signals.append((pid, signal_number)),
    )

    try:
        runner._run_lmms_eval_process(
            ["python", "-m", "lmms_eval"],
            cwd=str(tmp_path),
            env={},
            timeout=7.0,
        )
    except subprocess.TimeoutExpired as error:
        assert error.output == "partial stdout"
        assert error.stderr == "partial stderr"
    else:
        raise AssertionError("expected lmms-eval timeout to be raised")

    assert process.communicate_timeouts == [
        7.0,
        runner._LMMS_EVAL_PROCESS_CLEANUP_TIMEOUT_SECONDS,
        runner._LMMS_EVAL_PROCESS_CLEANUP_TIMEOUT_SECONDS,
    ]
    assert signals == [(5678, signal.SIGTERM), (5678, signal.SIGKILL)]


def test_downstream_evaluation_runs_lmms_eval_and_flattens_metrics(monkeypatch, tmp_path):
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        del env, timeout
        captured["argv"] = argv
        output = Path(cwd) / "nested"
        output.mkdir(parents=True)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "results": {
                        "ifeval": {"prompt_level_strict_acc,none": 0.5},
                        "gsm8k": {"exact_match,strict-match": 0.75},
                    },
                    "group_subtasks": {"ifeval": [], "gsm8k": []},
                    "n-samples": {
                        "ifeval": {"original": 541, "effective": 4},
                        "gsm8k": {"original": 1319, "effective": 4},
                    },
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_lmms_eval_process", fake_run)
    node = SimpleNamespace(
        node_id="lmms_eval",
        flow_id="runtime",
        stage_id="post.runtime.lmms_eval",
        config={
            "config": {
                "command_prefix": ["python", "-m", "lmms_eval"],
                "tasks": ["ifeval", "gsm8k"],
                "limit": 4,
                "topology": {"gpu_group_size": 1},
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    result = runner._downstream_evaluation(
        {"puzzle_dir": str(tmp_path)},
        node,
        source,
        "execution",
    )

    assert captured["argv"][:3] == ["python", "-m", "lmms_eval"]
    assert result["metrics"] == {
        "gsm8k.exact_match_strict-match": 0.75,
        "ifeval.prompt_level_strict_acc_none": 0.5,
    }
    assert Path(result["result_path"]).is_file()
    assert Path(result["raw_result_path"]).name == "results.json"
    summary = json.loads(Path(result["result_path"]).read_text())
    assert summary["sample_counts"] == {"gsm8k": 4.0, "ifeval": 4.0}


def test_lmms_eval_completion_validates_resolved_task_expansion():
    sample_counts = runner._validate_lmms_eval_completion(
        {
            "results": {
                "arc_challenge": {"acc,none": 0.25},
                "hellaswag": {"acc_norm,none": 0.5},
            },
            "group_subtasks": {
                "leaderboard": ["arc_challenge", "hellaswag"],
                "arc_challenge": [],
                "hellaswag": [],
            },
            "n-samples": {
                "arc_challenge": {"original": 1172, "effective": 8},
                "hellaswag": {"original": 10042, "effective": 8},
            },
        },
        ("leaderboard",),
    )

    assert sample_counts == {"arc_challenge": 8.0, "hellaswag": 8.0}


def test_downstream_evaluation_rejects_missing_configured_task(monkeypatch, tmp_path):
    def fake_run(argv, *, cwd, env, timeout):
        del env, timeout
        output = Path(cwd)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "results": {"ifeval": {"prompt_level_strict_acc,none": 0.5}},
                    "group_subtasks": {"ifeval": []},
                    "n-samples": {"ifeval": {"original": 541, "effective": 4}},
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_lmms_eval_process", fake_run)
    node = SimpleNamespace(
        node_id="lmms_eval",
        flow_id="runtime",
        stage_id="post.runtime.lmms_eval",
        config={
            "config": {
                "command_prefix": ["python", "-m", "lmms_eval"],
                "tasks": ["ifeval", "gsm8k"],
                "topology": {"gpu_group_size": 1},
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    try:
        runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected incomplete lmms-eval result to fail")

    assert "missing configured task results" in message
    assert "gsm8k" in message


def test_downstream_evaluation_rejects_zero_sample_task(monkeypatch, tmp_path):
    def fake_run(argv, *, cwd, env, timeout):
        del env, timeout
        output = Path(cwd)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "results": {
                        "ifeval": {"prompt_level_strict_acc,none": 0.5},
                        "gsm8k": {"exact_match,strict-match": 0.75},
                    },
                    "group_subtasks": {"ifeval": [], "gsm8k": []},
                    "n-samples": {
                        "ifeval": {"original": 541, "effective": 4},
                        "gsm8k": {"original": 1319, "effective": 0},
                    },
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_lmms_eval_process", fake_run)
    node = SimpleNamespace(
        node_id="lmms_eval",
        flow_id="runtime",
        stage_id="post.runtime.lmms_eval",
        config={
            "config": {
                "command_prefix": ["python", "-m", "lmms_eval"],
                "tasks": ["ifeval", "gsm8k"],
                "topology": {"gpu_group_size": 1},
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    try:
        runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected zero-sample lmms-eval result to fail")

    assert "zero effective samples" in message
    assert "gsm8k" in message


def test_downstream_evaluation_reports_lmms_eval_output_when_results_are_missing(
    monkeypatch, tmp_path
):
    def fake_run(argv, *, cwd, env, timeout):
        del cwd, env, timeout
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="Saving results aggregated\nCould not save results aggregated\n",
            stderr="",
        )

    monkeypatch.setattr(runner, "_run_lmms_eval_process", fake_run)
    node = SimpleNamespace(
        node_id="lmms_eval",
        flow_id="runtime",
        stage_id="post.runtime.lmms_eval",
        config={
            "config": {
                "command_prefix": ["python", "-m", "lmms_eval"],
                "tasks": ["ifeval"],
                "limit": 1,
                "topology": {"gpu_group_size": 1},
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    try:
        runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise AssertionError("expected missing lmms-eval results to fail")

    assert "lmms-eval wrote no JSON results" in message
    assert "stdout tail:" in message
    assert "Could not save results aggregated" in message
    stream_root = (
        tmp_path
        / "artifacts/post_mip/nodes/lmms_eval/executions/execution/raw/architecture/lmms_eval"
    )
    assert list(stream_root.glob("attempt_*/stdout.txt"))
