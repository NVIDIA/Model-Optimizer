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

"""Future-stage configuration and artifact-selection contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.security_policy import require_boolean_policy
from modelopt.torch.puzzletron.stages import future

# Security-policy validation


def test_security_policy_rejects_non_boolean_values():
    with pytest.raises(ValueError, match="^policy must be a boolean$"):
        require_boolean_policy("false", path="policy")


def test_security_policy_resolves_none_only_with_an_explicit_default():
    assert require_boolean_policy(None, path="policy", default=False) is False
    assert require_boolean_policy(None, path="policy", default=True) is True
    with pytest.raises(ValueError, match="^policy must be a boolean$"):
        require_boolean_policy(None, path="policy")


def test_evaluation_stage_rejects_scalar_checkpoints():
    config = {
        "zero_shot_evaluation": {
            "enabled": True,
            "checkpoints": "/checkpoint",
        }
    }
    with pytest.raises(
        ValueError,
        match=r"^zero_shot_evaluation\.checkpoints must be a list or tuple$",
    ):
        future.evaluation_stage(config, object())


def test_evaluation_stage_keeps_configured_checkpoints_authoritative(monkeypatch, tmp_path):
    class DescriptorReachedError(RuntimeError):
        pass

    def reject_implicit_teacher(*args, **kwargs):
        pytest.fail("configured checkpoints must not add the teacher")

    def stop_at_descriptor(*args, **kwargs):
        raise DescriptorReachedError

    monkeypatch.setattr(future, "_with_teacher_checkpoint", reject_implicit_teacher)
    monkeypatch.setattr(future, "_resolve_evaluation_descriptor", stop_at_descriptor)
    config = {
        "experiment": {"dir": str(tmp_path)},
        "convert": {"teacher_dir": str(tmp_path / "teacher")},
        "zero_shot_evaluation": {
            "enabled": True,
            "checkpoints": [str(tmp_path / "student")],
        },
    }

    with pytest.raises(DescriptorReachedError):
        future.evaluation_stage(config, object())


def test_evaluation_descriptor_is_inferred_from_checkpoint(monkeypatch, tmp_path):
    sentinel = object()
    calls = []

    class Resolution:
        descriptor = sentinel

    def resolve(pretrained, *, trust_remote_code=False):
        calls.append((pretrained, trust_remote_code))
        return Resolution()

    monkeypatch.setattr(future, "resolve_descriptor_from_pretrained", resolve)

    checkpoint = tmp_path / "solution_0"
    config = {"model": {"trust_remote_code": True}}

    assert future._resolve_evaluation_descriptor(config, checkpoint) is sentinel
    assert calls == [(str(checkpoint), True)]


# Global-distillation selection and publication


def test_distillation_sanity_accepts_packed_cache_without_raw_dataset(tmp_path):
    cache = tmp_path / "train.tokens"
    assert future._distillation_dataset_source(
        {"packed_token_cache_path": str(cache)},
        {},
    ) == ("", str(cache))


def test_distillation_sanity_requires_raw_dataset_or_packed_cache():
    with pytest.raises(ValueError, match="dataset_path or packed_token_cache_path"):
        future._distillation_dataset_source({}, {})


def test_distributed_barrier_propagates_failure_with_stage_context(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail():
        raise RuntimeError("peer exited")

    monkeypatch.setattr(torch.distributed, "barrier", fail)
    with pytest.raises(RuntimeError, match="global distillation publication.*peer exited"):
        future._distributed_barrier("global distillation publication")


def test_scenario_grid_kd_builds_one_isolated_config_per_realized_checkpoint(monkeypatch, tmp_path):
    puzzle_dir = tmp_path / "model"
    checkpoints = []
    for width, depth in ((512, 0), (1024, 1)):
        checkpoint = (
            puzzle_dir
            / "scenarios"
            / f"width-{width:04d}"
            / f"depth-{depth:02d}"
            / "checkpoints"
            / "solution_0"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        checkpoints.append(checkpoint)

    class Resolution:
        name = "generic"

    monkeypatch.setattr(
        future,
        "resolve_descriptor_from_pretrained",
        lambda *args, **kwargs: Resolution(),
    )
    config = {
        "experiment": {"dir": str(puzzle_dir)},
        "model": {"trust_remote_code": True},
        "convert": {"teacher_dir": str(puzzle_dir / "ckpts" / "teacher")},
        "distillation": {"scenario_grid": True, "max_steps": 8},
    }

    candidates = future._scenario_grid_kd_configs(config)
    assert [Path(item["distillation"]["student_dir"]) for item in candidates] == checkpoints
    assert [item["_runtime"]["descriptor"] for item in candidates] == [
        "generic",
        "generic",
    ]
    assert candidates[0]["distillation"]["output_dir"].endswith(
        "artifacts/global_kd/scenarios/width-0512/depth-00"
    )
    assert candidates[1]["distillation"]["output_dir"].endswith(
        "artifacts/global_kd/scenarios/width-1024/depth-01"
    )


def test_scenario_grid_kd_checkpoints_select_latest_consolidated(tmp_path):
    puzzle_dir = tmp_path / "model"
    for width, depth in ((512, 0), (1024, 1)):
        root = (
            puzzle_dir
            / "artifacts"
            / "global_kd"
            / "scenarios"
            / f"width-{width:04d}"
            / f"depth-{depth:02d}"
            / "checkpoints"
        )
        for step in (0, 7):
            consolidated = root / f"epoch_{step}_step_{step}" / "model" / "consolidated"
            consolidated.mkdir(parents=True)
            (consolidated / "config.json").write_text("{}")

    checkpoints = future._scenario_grid_global_kd_checkpoints(puzzle_dir)

    assert [name for name, _ in checkpoints] == [
        "width-0512__depth-00",
        "width-1024__depth-01",
    ]
    assert all(path.name == "consolidated" for _, path in checkpoints)
    assert all("step_7" in str(path) for _, path in checkpoints)


# AIPerf execution


def test_bounded_map_does_not_queue_work_after_failure():
    observed = []

    def fail_first(value):
        observed.append(value)
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="^stop$"):
        future._bounded_map(fail_first, range(5), max_workers=1)
    assert observed == [0]


def test_aiperf_stage_forwards_repeated_measurement_policy(monkeypatch, tmp_path):
    teacher = tmp_path / "teacher"
    candidate = tmp_path / "candidates" / "solution_0"
    teacher.mkdir()
    candidate.mkdir(parents=True)
    (teacher / "config.json").write_text("{}")
    (candidate / "config.json").write_text("{}")
    evaluation_summary = tmp_path / "evaluation_summary.json"
    evaluation_summary.write_text(
        json.dumps([{"checkpoint": str(candidate), "metrics": {"lm_loss": 1.0}}])
    )
    calls = []

    def fake_run_aiperf_sweep(checkpoint, **settings):
        calls.append((checkpoint, settings))
        return []

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.run_aiperf_sweep",
        fake_run_aiperf_sweep,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.write_aiperf_report",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        future,
        "complete_stage",
        lambda _config, _manifest, *, outputs: outputs,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    config = {
        "experiment": {"dir": str(tmp_path)},
        "convert": {"teacher_dir": str(teacher)},
        "aiperf": {
            "enabled": True,
            "solution_checkpoints_dir": str(candidate.parent),
            "evaluation_summary_path": str(evaluation_summary),
            "concurrency": [1],
            "topology": {"gpu_group_size": 1},
            "warmup_request_count": 8,
            "warmup_seed": 99,
            "repetitions": 3,
            "collect_peak_gpu_memory": True,
        },
    }

    outputs = future.aiperf_stage(config, SimpleNamespace())

    assert outputs["result_count"] == 0
    assert len(calls) == 2
    for _, settings in calls:
        assert settings["warmup_request_count"] == 8
        assert settings["warmup_seed"] == 99
        assert settings["repetitions"] == 3
        assert settings["collect_peak_gpu_memory"] is True
