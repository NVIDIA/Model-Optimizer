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


def test_downstream_evaluation_runs_lmms_eval_and_flattens_metrics(monkeypatch, tmp_path):
    captured = {}

    def fake_run(argv, *, cwd, env, capture_output, text, timeout, check):
        del env, capture_output, text, timeout, check
        captured["argv"] = argv
        output = Path(cwd) / "nested"
        output.mkdir(parents=True)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "results": {
                        "ifeval": {"prompt_level_strict_acc,none": 0.5},
                        "gsm8k": {"exact_match,strict-match": 0.75},
                    }
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
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
