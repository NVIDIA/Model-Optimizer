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

"""Tests for Puzzletron local-KD configuration, recipes, and launch behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.plugins.automodel import (
    local_kd_config,
    local_kd_launch,
    local_kd_recipe,
)


def test_local_kd_treats_ep_as_an_overlay_not_a_sample_axis():
    assert (
        local_kd_config._logical_dp_size(
            OmegaConf.create({}),
            {"dp_size": 8, "ep_size": 4},
        )
        == 2
    )


def test_lane_axis_counts_reject_model_parallel_replica_disagreement():
    gathered = [
        {"dp_lane": 0, "hidden_width_counts": {4096: 1}},
        {"dp_lane": 0, "hidden_width_counts": {3840: 1}},
    ]

    with pytest.raises(RuntimeError, match="logical data lane 0"):
        local_kd_recipe._merge_lane_axis_counts(
            gathered,
            count_key="hidden_width_counts",
        )


def test_local_kd_rejects_a_disabled_data_parallel_axis_with_ep(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        local_kd_config,
        "build_stage_recipe_config",
        lambda _cfg: {
            "model": {},
            "distributed": {
                "dp_size": "none",
                "ep_size": 2,
                "pp_size": 2,
                "cp_size": 1,
                "pipeline": {},
            },
        },
    )
    monkeypatch.setattr(local_kd_config, "_teacher_dir", lambda _cfg: tmp_path)
    monkeypatch.setattr(
        local_kd_config, "inject_descriptor_model_kwargs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        local_kd_config,
        "_inject_canonical_data",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        local_kd_config,
        "inject_descriptor_pipeline_config",
        lambda recipe, **_kwargs: recipe,
    )
    cfg = OmegaConf.create(
        {
            "descriptor": "gpt_oss",
            "model": {"force_hf": False, "trust_remote_code": True},
            "data": {"modality": "text"},
            "bypass": {
                "automodel": {
                    "parallel": {
                        "tp": 1,
                        "cp": 1,
                        "pp": 2,
                        "ep": 2,
                        "dp_shard": 2,
                        "dp_replicate": 1,
                    }
                },
                "elastic": False,
                "single_batch_overfit": True,
                "dtype": "bf16",
                "experiment_dir": str(tmp_path / "run"),
                "data": {"block_size": 2048},
                "training": {
                    "micro_batch_size": 1,
                    "grad_accumulation_steps": 1,
                    "max_steps": 16,
                    "optimizer": "adamw",
                    "learning_rate": 1.0e-5,
                    "weight_decay": 0.0,
                    "beta1": 0.9,
                    "beta2": 0.95,
                },
            },
        }
    )

    with pytest.raises(ValueError, match="dp_size must be divisible by ep_size"):
        local_kd_config.build_local_kd_recipe_config(cfg)


def test_local_kd_reads_parallel_size_from_canonical_automodel_recipe() -> None:
    recipe_cfg = OmegaConf.create({"distributed": {"tp_size": 2}})

    assert local_kd_recipe._recipe_parallel_size(recipe_cfg, "tp_size") == 2
    assert local_kd_recipe._recipe_parallel_size(recipe_cfg, "cp_size") == 1


def test_disjoint_local_loss_backpropagates_immediately_and_returns_detached_value() -> None:
    class IdentityScaler:
        @staticmethod
        def scale(loss):
            return loss

    first = torch.tensor(3.0, requires_grad=True)
    second = torch.tensor(4.0, requires_grad=True)

    first_value = local_kd_recipe._backward_disjoint_loss(
        (2 * first).square(), grad_scaler=IdentityScaler(), grad_accum=2
    )
    second_value = local_kd_recipe._backward_disjoint_loss(
        (second - 1).square(), grad_scaler=IdentityScaler(), grad_accum=2
    )

    assert not first_value.requires_grad
    assert not second_value.requires_grad
    torch.testing.assert_close(first.grad, torch.tensor(12.0))
    torch.testing.assert_close(second.grad, torch.tensor(3.0))


def test_local_kd_hidden_mask_excludes_padded_tokens_from_replay_loss() -> None:
    student = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0], [200.0, 200.0]]])
    teacher = torch.zeros_like(student)
    hidden_mask = torch.tensor([[True, True, False, False]])

    masked_student, masked_teacher = local_kd_recipe._mask_local_kd_tensors(
        student,
        teacher,
        hidden_mask,
        record_index=0,
        record_count=1,
    )

    assert masked_student.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert masked_teacher.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_local_kd_empty_mask_shard_backpropagates_finite_zero_without_metric() -> None:
    source = torch.ones((1, 2, 3), requires_grad=True)
    student = source[torch.zeros((1, 2), dtype=torch.bool)]
    teacher = torch.zeros_like(student)

    loss, contributes_metric = local_kd_recipe._local_kd_loss_or_zero(
        lambda actual, expected: ((actual - expected) ** 2).mean(),
        student,
        teacher,
    )

    assert not contributes_metric
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(source.grad, torch.zeros_like(source))


def test_elastic_local_kd_trend_is_reported_without_incomparable_hard_gate() -> None:
    records = [
        {"loss": loss, "hidden_width": width}
        for loss, width in (
            (0.0, 2048),
            (0.05, 1024),
            (0.01, 2048),
            (0.05, 1024),
            (0.03, 2048),
            (0.06, 1024),
            (0.02, 2048),
            (0.06, 1024),
        )
    ]

    trend = local_kd_recipe._loss_trend_summary(records, comparable=False)

    assert trend["decreased"] is False
    assert trend["comparable"] is False
    assert trend["hard_gate_passed"] is None
    assert set(trend["per_hidden_width"]) == {"1024", "2048"}


def test_inverse_width_policy_is_reproducible_and_favors_thinner_width() -> None:
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)
    first = [
        local_kd_recipe._select_hidden_width(
            (2688, 1344),
            step=step,
            cycle=False,
            policy="inverse_width",
            generator=first_generator,
        )
        for step in range(1, 2001)
    ]
    second = [
        local_kd_recipe._select_hidden_width(
            (2688, 1344),
            step=step,
            cycle=False,
            policy="inverse_width",
            generator=second_generator,
        )
        for step in range(1, 2001)
    ]

    assert first == second
    assert first.count(1344) > first.count(2688)


def test_elastic_selection_record_contains_width_layer_candidate_and_axes() -> None:
    targets = {
        3: SimpleNamespace(
            identity=SimpleNamespace(value="candidate-3"),
            metadata={"slice_axes": {"moe_top_k": 3}},
        ),
        1: SimpleNamespace(
            identity=SimpleNamespace(value="candidate-1"),
            metadata={"slice_axes": {"kv_groups": 1}},
        ),
    }

    record = local_kd_recipe._elastic_selection_record(
        step=7,
        hidden_width=1344,
        targets=targets,
    )

    assert record == {
        "step": 7,
        "hidden_width": 1344,
        "ple_width": None,
        "layers": [
            {
                "layer_idx": 1,
                "candidate_id": "candidate-1",
                "parameter_count": None,
                "changed_axes": {"kv_groups": 1},
            },
            {
                "layer_idx": 3,
                "candidate_id": "candidate-3",
                "parameter_count": None,
                "changed_axes": {"moe_top_k": 3},
            },
        ],
    }


def test_elastic_probe_can_disable_global_checkpoint_publication() -> None:
    assert local_kd_launch._should_publish_elastic_checkpoint(
        SimpleNamespace(bypass={"elastic": True})
    )
    assert not local_kd_launch._should_publish_elastic_checkpoint(
        SimpleNamespace(bypass={"elastic": True, "publish_elastic_checkpoint": False})
    )
    assert not local_kd_launch._should_publish_elastic_checkpoint(
        SimpleNamespace(bypass={"elastic": False})
    )


def test_overfit_probe_repeats_one_batch_without_mutating_main_run() -> None:
    config = OmegaConf.create(
        {
            "bypass": {
                "experiment_id": "nested-main",
                "experiment_dir": "/tmp/nested-main",
                "publish_elastic_checkpoint": True,
                "find_last_ckpt_for_resume": True,
                "step_num": 1024,
                "iter_num": 8192,
                "token_count": 1073741824,
                "overfit": {"enabled": True, "repetitions": 32},
                "training": {"max_steps": 8},
            }
        }
    )
    config._set_flag("allow_objects", True)
    config.runtime_pruning_mixin = SimpleNamespace(name="instantiated")

    probe = local_kd_launch._overfit_probe_config(config)

    assert probe.bypass.single_batch_overfit is True
    assert probe.bypass.single_batch_overfit_steps == 32
    assert probe.bypass.training.max_steps == 32
    assert probe.bypass.publish_elastic_checkpoint is False
    assert probe.bypass.find_last_ckpt_for_resume is False
    assert probe.bypass.step_num == 1
    assert probe.bypass.iter_num == 0
    assert probe.bypass.token_count == 0
    assert probe.bypass.overfit.enabled is False
    assert config.bypass.training.max_steps == 8
    assert config.bypass.publish_elastic_checkpoint is True
    assert config.bypass.step_num == 1024
    assert config.bypass.iter_num == 8192
    assert config.bypass.token_count == 1073741824
    assert probe.runtime_pruning_mixin.name == "instantiated"
    assert not local_kd_launch._should_publish_final_checkpoint(probe)
    assert local_kd_launch._should_publish_final_checkpoint(config)


def test_distributed_path_consensus_rejects_split_checkpoint_roots(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    run_a = str(Path("/tmp/run-a").resolve())
    run_b = str(Path("/tmp/run-b").resolve())

    def gather(values, local):
        assert local == run_a
        values[:] = [run_a, run_b]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="checkpoint save.*run-a.*run-b"):
        local_kd_launch._require_distributed_path_consensus(
            "/tmp/run-a",
            "checkpoint save",
        )


def test_overfit_only_runs_probe_without_starting_main_nested_run(monkeypatch) -> None:
    config = OmegaConf.create(
        {
            "bypass": {
                "experiment_id": "nested-main",
                "experiment_dir": "/tmp/nested-main",
                "publish_elastic_checkpoint": True,
                "find_last_ckpt_for_resume": False,
                "overfit": {
                    "enabled": True,
                    "repetitions": 64,
                    "only": True,
                    "learning_rate": 3.0e-4,
                    "decay_lr": False,
                    "grad_clip": 10.0,
                    "selection": "smallest",
                },
                "training": {
                    "max_steps": 8,
                    "learning_rate": 1.0e-5,
                    "decay_lr": True,
                    "grad_clip": 1.0,
                },
            }
        }
    )
    config._set_flag("allow_objects", True)
    observed = []
    monkeypatch.setattr(local_kd_launch, "_run_one", observed.append)

    local_kd_launch._run_with_optional_overfit(config)

    assert len(observed) == 1
    assert observed[0].bypass.single_batch_overfit is True
    assert observed[0].bypass.single_batch_overfit_steps == 64
    assert observed[0].bypass.training.max_steps == 64
    assert observed[0].bypass.training.learning_rate == 3.0e-4
    assert observed[0].bypass.training.decay_lr is False
    assert observed[0].bypass.training.grad_clip == 10.0
    assert observed[0].bypass.elastic_fixed_selection == "smallest"
