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

import pytest
import torch

from modelopt.torch.puzzletron.security_policy import require_boolean_policy
from modelopt.torch.puzzletron.stages.future import evaluation_stage


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
        evaluation_stage(config, object())


def test_distillation_sanity_accepts_packed_cache_without_raw_dataset(tmp_path):
    from modelopt.torch.puzzletron.stages.future import _distillation_dataset_source

    cache = tmp_path / "train.tokens"
    assert _distillation_dataset_source(
        {"packed_token_cache_path": str(cache)},
        {},
    ) == ("", str(cache))


def test_distillation_sanity_requires_raw_dataset_or_packed_cache():
    from modelopt.torch.puzzletron.stages.future import _distillation_dataset_source

    with pytest.raises(ValueError, match="dataset_path or packed_token_cache_path"):
        _distillation_dataset_source({}, {})


def test_distributed_barrier_propagates_failure_with_stage_context(monkeypatch):
    from modelopt.torch.puzzletron.stages.future import _distributed_barrier

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail():
        raise RuntimeError("peer exited")

    monkeypatch.setattr(torch.distributed, "barrier", fail)
    with pytest.raises(RuntimeError, match="global distillation publication.*peer exited"):
        _distributed_barrier("global distillation publication")


def test_evaluation_descriptor_is_inferred_from_checkpoint(monkeypatch, tmp_path):
    from modelopt.torch.puzzletron.stages import future

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


def test_evaluation_descriptor_honors_explicit_legacy_override(monkeypatch, tmp_path):
    from modelopt.torch.puzzletron.stages import future

    sentinel = object()
    monkeypatch.setattr(future.ModelDescriptorFactory, "get", lambda name: (name, sentinel))

    assert future._resolve_evaluation_descriptor({"descriptor": "legacy"}, Path(tmp_path)) == (
        "legacy",
        sentinel,
    )


def test_scenario_grid_kd_builds_one_isolated_config_per_realized_checkpoint(monkeypatch, tmp_path):
    from modelopt.torch.puzzletron.stages import future

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
    from modelopt.torch.puzzletron.stages import future

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


def test_profile_solution_checkpoints_use_selected_mip_registry(tmp_path):
    from modelopt.torch.puzzletron.stages import future

    puzzle_dir = tmp_path / "model"
    profile_root = puzzle_dir / "mip/profiles/params-090"
    profile_root.mkdir(parents=True)
    teacher = puzzle_dir / "ckpts/teacher"
    candidate = profile_root / "scenarios/width-3840/depth-01/checkpoints/solution_0"
    (puzzle_dir / "mip/profiles/index.json").write_text(
        json.dumps({"profiles": [{"id": "params-090"}]})
    )
    (profile_root / "selected_solutions.json").write_text(
        json.dumps(
            {
                "solutions": [
                    {"solution_id": "teacher", "checkpoint": str(teacher)},
                    {"solution_id": "h3840-d1", "checkpoint": str(candidate)},
                ]
            }
        )
    )

    assert future._profile_solution_checkpoints(puzzle_dir) == [
        ("teacher", teacher),
        ("h3840-d1", candidate),
    ]
    assert future._profile_solution_checkpoints(puzzle_dir, profile_id="params-090") == [
        ("teacher", teacher),
        ("h3840-d1", candidate),
    ]


def test_global_kd_checkpoints_include_canonical_distillation_exports(tmp_path):
    from modelopt.torch.puzzletron.stages import future

    puzzle_dir = tmp_path / "model"
    run = (
        puzzle_dir
        / "artifacts/global_distillation/profiles/latency-095"
        / "text-n4096-l16384-s256-b16-seed444/h4096-d4"
    )
    checkpoint = run / "post_kd_export/checkpoint/model/consolidated"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (run / "global_distillation_summary.json").write_text(
        json.dumps(
            {
                "profile_id": "latency-095",
                "solution_id": "h4096-d4",
                "post_kd_checkpoint": str(checkpoint),
            }
        )
    )

    checkpoints = future._scenario_grid_global_kd_checkpoints(puzzle_dir)

    assert checkpoints == [("latency-095__h4096-d4", checkpoint)]


def test_aiperf_executable_prefers_config_then_environment(monkeypatch):
    from modelopt.torch.puzzletron.stages import future

    monkeypatch.setenv("AIPERF_EXECUTABLE", "/shared/aiperf")
    assert future._aiperf_executable({}) == "/shared/aiperf"
    assert future._aiperf_executable({"executable": "/configured/aiperf"}) == "/configured/aiperf"


def test_bounded_map_does_not_queue_work_after_failure():
    from modelopt.torch.puzzletron.stages import future

    observed = []

    def fail_first(value):
        observed.append(value)
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="^stop$"):
        future._bounded_map(fail_first, range(5), max_workers=1)
    assert observed == [0]


def test_aiperf_checkpoint_work_keeps_concurrencies_serial_per_checkpoint():
    from modelopt.torch.puzzletron.stages import future

    work = future._aiperf_checkpoint_work(
        [("teacher", Path("/teacher")), ("student", Path("/student"))], [1, 2]
    )
    assert work == [
        ("teacher", Path("/teacher"), (1, 2)),
        ("student", Path("/student"), (1, 2)),
    ]
