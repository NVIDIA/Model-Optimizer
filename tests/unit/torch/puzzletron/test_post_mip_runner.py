# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.post_mip import runner
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
