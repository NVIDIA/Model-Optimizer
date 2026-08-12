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

"""Tests for Puzzletron's canonical global-distillation behavior."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.distillation.global_automodel import (
    GlobalKDConfig,
    GlobalKDResult,
    build_automodel_global_kd_recipe,
    build_global_kd_config,
)


def test_global_kd_pp_shape_reset_uses_pipeline_activation_dtype(monkeypatch):
    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    calls = []
    monkeypatch.setattr(
        global_kd_recipe,
        "reset_pp_stage_shapes",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    pp = SimpleNamespace(
        dtype=torch.bfloat16,
        pp_microbatch_size=2,
        info=SimpleNamespace(schedule=object(), stages=[]),
        parts=[SimpleNamespace(config=object())],
    )

    global_kd_recipe._reset_global_kd_pp_stage_shapes(pp, seq_len=4096)

    args, kwargs = calls.pop()
    assert args[3:] == (2, 4096)
    assert kwargs == {"tensor_dtype": torch.bfloat16}


def test_global_kd_pp_hidden_meta_uses_pipeline_activation_dtype():
    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    submod = torch.nn.Linear(4, 4, dtype=torch.float32)
    submod._puzzletron_distillation_hidden_output = True
    stage = SimpleNamespace(
        submod=submod,
        is_last=True,
        inputs_meta=(torch.empty(2, 8, 4, device="meta", dtype=torch.bfloat16),),
        _outputs_meta=(),
    )
    pp = SimpleNamespace(dtype=torch.bfloat16, info=SimpleNamespace(stages=[stage]))

    assert global_kd_recipe._refresh_pp_hidden_output_meta(pp) == 1
    assert stage._outputs_meta[0].dtype is torch.bfloat16


def test_global_kd_checkpoint_context_uses_active_anymodel_block_configs(
    monkeypatch,
):
    """Checkpoint conversion keeps the student’s per-layer MoE geometry."""
    from modelopt.torch.puzzletron.anymodel.automodel import AutoModelDescriptorFactory
    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    observed = []

    class Descriptor:
        @classmethod
        @contextmanager
        def native_state_dict_adapter_context(cls, block_configs):
            observed.append(block_configs)
            yield

    monkeypatch.setattr(AutoModelDescriptorFactory, "get", lambda name: Descriptor)
    part = SimpleNamespace(
        config=SimpleNamespace(
            anymodel_descriptor="nemotron_h",
            block_configs=[{"subblock_configs": []}],
        )
    )

    with global_kd_recipe._global_kd_checkpoint_adapter_context([part]):
        pass

    assert len(observed) == 1
    assert observed[0][0].to_dict() == {"subblock_configs": []}


def test_distillation_overfit_stage_disables_mtp_objectives_by_default(monkeypatch, tmp_path):
    """Nano-like checkpoints without MTP must not enable MTP loss implicitly."""
    from modelopt.torch.puzzletron.manifest import StageManifest
    from modelopt.torch.puzzletron.stages import future

    registry_path = tmp_path / "selected_solutions.json"
    registry_path.write_text(
        json.dumps(
            {
                "solutions": [
                    {"solution_id": "teacher", "checkpoint": str(tmp_path / "teacher")},
                    {"solution_id": "candidate", "checkpoint": str(tmp_path / "candidate")},
                ]
            }
        )
    )
    captured = {}

    def fake_build_global_kd_config(candidate):
        captured["objective"] = candidate["distillation"]["objective"]
        captured["student_model_kwargs"] = candidate["distillation"]["student_model_kwargs"]
        captured["teacher_model_kwargs"] = candidate["distillation"]["teacher_model_kwargs"]
        return candidate["distillation"]

    def fake_run_global_kd(kd_config):
        training_log = Path(kd_config["output_dir"]) / "checkpoints" / "training.jsonl"
        training_log.parent.mkdir(parents=True)
        training_log.write_text(json.dumps({"step": 1, "loss": 1.0}) + "\n")
        return SimpleNamespace(to_dict=dict)

    monkeypatch.setattr(
        future,
        "resolve_descriptor_from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(name="nemotron3_nano"),
    )
    monkeypatch.setattr(future, "build_global_kd_config", fake_build_global_kd_config)
    monkeypatch.setattr(future, "run_global_kd", fake_run_global_kd)
    monkeypatch.setattr(future, "_distributed_barrier", lambda *args: None)
    monkeypatch.delenv("RANK", raising=False)

    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"trust_remote_code": True},
        "global_distillation_sanity": {
            "enabled": True,
            "profile_id": "runtime-075",
            "registry_path": str(registry_path),
            "dataset_path": str(tmp_path / "data"),
            "sample_count": 1,
            "sequence_length": 8,
            "max_steps": 1,
            "local_batch_size": 1,
            "student_model_kwargs": {"torch_dtype": "float32"},
            "teacher_model_kwargs": {"torch_dtype": "bfloat16"},
        },
    }

    future.distillation_overfit_stage(config, StageManifest(stage="global_distillation_sanity"))

    assert captured["objective"] == {
        "main_ce": {"weight": 1.0},
        "main_kd": {"weight": 1.0},
        "mtp_ce": {"weight": 0.0},
        "mtp_kd": {"weight": 0.0},
    }
    assert captured["student_model_kwargs"] == {"torch_dtype": "float32"}
    assert captured["teacher_model_kwargs"] == {"torch_dtype": "bfloat16"}


def test_global_distillation_summary_publishes_canonical_training_records(tmp_path):
    from modelopt.torch.puzzletron.stages.future import _write_global_distillation_summary

    output_dir = (
        tmp_path
        / "artifacts/global_distillation/profiles/latency-095"
        / "text-n4096-l16384-s256-b16-seed444/h4096-d4"
    )
    training_log = output_dir / "checkpoints/training.jsonl"
    training_log.parent.mkdir(parents=True)
    training_log.write_text(
        json.dumps({"step": 1, "loss": 2.0}) + "\n" + json.dumps({"step": 2, "loss": 1.0}) + "\n"
    )
    for step in (1, 2):
        checkpoint = output_dir / "checkpoints" / f"epoch_0_step_{step}" / "model" / "consolidated"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint.parents[1] / "saving_completed").touch()
    config = GlobalKDConfig(
        teacher_dir=tmp_path / "teacher",
        student_dir=tmp_path / "student",
        output_dir=output_dir,
        descriptor="qwen3_5_text",
        domain="llm",
        global_batch_size=16,
        local_batch_size=2,
        max_steps=256,
        save_consolidated=True,
        metadata={"llm": {"dataset": {"num_samples": 4096, "seq_length": 16384}}},
    )
    result = GlobalKDResult(kd_id="kd-id", output_dir=output_dir, metrics={"loss_trend": {}})

    summary_path = _write_global_distillation_summary(config, result)

    payload = json.loads(summary_path.read_text())
    assert payload["profile_id"] == "latency-095"
    assert payload["solution_id"] == "h4096-d4"
    assert payload["max_steps"] == 256
    assert payload["sequence_length"] == 16384
    assert payload["records"][-1] == {"step": 2, "loss": 1.0}
    assert payload["post_kd_checkpoint"].endswith("checkpoints/epoch_0_step_2/model/consolidated")


def test_global_kd_metric_logger_flushes_every_optimizer_step():
    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    logger = type("Logger", (), {"buffer_size": 100, "flush": False})()
    recipe = object.__new__(_WeightedObjectiveMixin)
    recipe.metric_logger_train = logger

    recipe._configure_incremental_metric_logging()

    assert logger.buffer_size == 1
    assert logger.flush is True


def test_global_kd_uses_memory_bounded_1f1b_by_default_and_allows_override(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_pipeline_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distillation.global_automodel._model_recipe",
        lambda *args, teacher=False, **kwargs: {"teacher": teacher},
    )
    common = {
        "teacher_dir": tmp_path / "teacher",
        "student_dir": tmp_path / "student",
        "output_dir": tmp_path / "output",
        "descriptor": "qwen3_5",
        "domain": "llm",
        "pp": 8,
        "global_batch_size": 8,
        "local_batch_size": 8,
    }

    default_recipe = build_automodel_global_kd_recipe(GlobalKDConfig(**common))
    override_recipe = build_automodel_global_kd_recipe(
        GlobalKDConfig(**common, pp_schedule="interleaved1f1b")
    )

    assert default_recipe["distributed"]["pipeline"]["pp_schedule"] == "1f1b"
    assert default_recipe["distributed"]["pipeline"]["pp_microbatch_size"] == 1
    assert default_recipe["distributed"]["pipeline"]["pp_batch_size"] == 8
    assert default_recipe["dataloader"]["batch_size"] == 8
    assert override_recipe["distributed"]["pipeline"]["pp_schedule"] == "interleaved1f1b"


def test_global_kd_auto_domain_uses_canonical_text_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distillation.global_automodel._model_recipe",
        lambda *args, teacher=False, **kwargs: {"teacher": teacher},
    )
    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"descriptor_override": "qwen3_5"},
        "data": {
            "path": str(tmp_path / "dataset"),
            "modality": "text",
            "layout": "fixed",
            "max_sample_length": 1024,
            "calibration": {"num_samples": 4096},
            "replacement_scoring": {"num_samples": 128},
        },
        "train_token_cache_path": str(tmp_path / "train.tokens"),
        "validation_token_cache_path": str(tmp_path / "validation.tokens"),
        "distillation": {
            "domain": "auto",
            "max_steps": 128,
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 1,
                    "ep": 1,
                    "dp_shard": 2,
                    "dp_replicate": 1,
                }
            },
        },
    }

    recipe = build_automodel_global_kd_recipe(build_global_kd_config(config))

    assert recipe["recipe"] == "KnowledgeDistillationRecipeForNextTokenPrediction"
    assert recipe["dataset"] == {
        "_target_": ("modelopt.torch.puzzletron.distillation.dataset.make_puzzletron_llm_dataset"),
        "dataset_path": str(tmp_path / "dataset"),
        "split": "train",
        "num_samples": 4096,
        "seq_length": 1024,
        "seed": 1111,
        "packed_token_cache_path": str(tmp_path / "train.tokens"),
    }
    assert recipe["validation_dataset"]["packed_token_cache_path"] == str(
        tmp_path / "validation.tokens"
    )


def test_global_kd_packed_text_uses_native_chat_data_and_canonical_pack_size(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distillation.global_automodel._model_recipe",
        lambda *args, **kwargs: {},
    )
    config = GlobalKDConfig(
        teacher_dir=tmp_path / "teacher",
        student_dir=tmp_path / "student",
        output_dir=tmp_path / "output",
        descriptor="qwen3_5",
        domain="llm",
        student_force_hf=False,
        teacher_force_hf=False,
        validation_enabled=False,
        data={
            "path": str(tmp_path / "dataset"),
            "modality": "text",
            "layout": "packed_varlen",
            "max_sample_length": 192,
            "packing": {
                "pack_size": 256,
                "packing_ratio": 0.8,
                "drop_long_samples": True,
            },
        },
    )

    recipe = build_automodel_global_kd_recipe(config)

    assert recipe["dataset"]["_target_"].endswith("make_puzzletron_chat_dataset")
    assert recipe["dataloader"]["collate_fn"].endswith("default_collater")
    assert recipe["packed_sequence"] == {
        "packed_sequence_size": 256,
        "packing_strategy": "neat",
        "drop_long_samples": True,
        "max_packs": 128,
    }


def test_global_kd_recipe_publishes_explicit_resume_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_pipeline_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distillation.global_automodel._model_recipe",
        lambda *args, **kwargs: {},
    )
    common = {
        "teacher_dir": tmp_path / "teacher",
        "student_dir": tmp_path / "student",
        "output_dir": tmp_path / "output",
        "descriptor": "qwen3_5",
        "domain": "llm",
        "pp": 1,
    }

    assert (
        build_automodel_global_kd_recipe(GlobalKDConfig(**common, resume=True))["puzzletron_resume"]
        is True
    )
    assert (
        build_automodel_global_kd_recipe(GlobalKDConfig(**common, resume=False))[
            "puzzletron_resume"
        ]
        is False
    )


def test_global_kd_config_preserves_per_model_dtype_overrides(tmp_path):
    from modelopt.torch.puzzletron.distillation import global_automodel

    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"descriptor_override": "nemotron_h", "torch_dtype": "bfloat16"},
        "distillation": {
            "teacher_dir": str(tmp_path / "teacher"),
            "student_dir": str(tmp_path / "student"),
            "output_dir": str(tmp_path / "output"),
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 2,
                    "ep": 4,
                    "dp_shard": 4,
                    "dp_replicate": 1,
                }
            },
            "student_model_kwargs": {"torch_dtype": "float32"},
            "teacher_model_kwargs": {"torch_dtype": "bfloat16"},
        },
    }

    kd = build_global_kd_config(config)

    assert (
        global_automodel._model_recipe(kd, teacher=False, domain="llm")["torch_dtype"] == "float32"
    )
    assert (
        global_automodel._model_recipe(kd, teacher=True, domain="llm")["torch_dtype"] == "bfloat16"
    )


def test_global_kd_load_checkpoint_honors_resume_policy():
    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    class Parent:
        def load_checkpoint(self, restore_from=None):
            self.loaded_from = restore_from
            return restore_from

    class Recipe(_WeightedObjectiveMixin, Parent):
        pass

    disabled = Recipe()
    disabled.cfg = {"puzzletron_resume": False}
    assert disabled.load_checkpoint() is None
    assert not hasattr(disabled, "loaded_from")

    enabled = Recipe()
    enabled.cfg = {"puzzletron_resume": True}
    assert enabled.load_checkpoint() == "LATEST"
    assert enabled.loaded_from == "LATEST"


def test_global_distillation_stage_promotes_canonical_namespace(tmp_path):
    from modelopt.torch.puzzletron.stages.future import _promote_global_distillation_config

    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"descriptor_override": "qwen3_5_text"},
        "distillation": {"local_batch_size": 1},
        "global_distillation": {
            "local_batch_size": 2,
            "global_batch_size": 16,
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 4,
                    "pp": 2,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 8,
                }
            },
        },
    }

    promoted = _promote_global_distillation_config(config)
    kd_config = build_global_kd_config(promoted)

    assert kd_config.local_batch_size == 2
    assert kd_config.global_batch_size == 16
    assert (kd_config.pp, kd_config.cp, kd_config.dp) == (2, 4, 8)


def test_global_kd_preserves_physical_dp_mesh_when_ep_overlays_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_pipeline_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distillation.global_automodel._model_recipe",
        lambda *args, teacher=False, **kwargs: {"teacher": teacher},
    )
    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"descriptor_override": "nemotron_h"},
        "distillation": {
            "domain": "llm",
            "local_batch_size": 2,
            "global_batch_size": 16,
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 2,
                    "ep": 4,
                    "dp_shard": 4,
                    "dp_replicate": 1,
                }
            },
        },
    }

    kd_config = build_global_kd_config(config)
    recipe = build_automodel_global_kd_recipe(kd_config)

    assert (kd_config.pp, kd_config.ep, kd_config.dp) == (2, 4, 4)
    assert recipe["distributed"]["dp_size"] == 4


def test_teacher_hidden_is_rewrapped_on_equivalent_lm_head_mesh(monkeypatch):
    import torch

    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    class Mesh:
        def __init__(self):
            self.mesh = torch.tensor([0, 1])
            self.mesh_dim_names = ("tp",)

    class FakeDTensor:
        calls = []

        def __init__(self, local, mesh, placements=("replicate",)):
            self.local = local
            self.device_mesh = mesh
            self.placements = placements

        def to_local(self):
            return self.local

        @classmethod
        def from_local(cls, local, *, device_mesh, placements, run_check):
            cls.calls.append((local, device_mesh, placements, run_check))
            return cls(local, device_mesh, placements)

    monkeypatch.setattr(global_kd_recipe, "DTensor", FakeDTensor)
    source_mesh = Mesh()
    head_mesh = Mesh()
    hidden = FakeDTensor(torch.ones(2, 3), source_mesh)
    base_layer = type("BaseLayer", (), {"weight": FakeDTensor(torch.ones(4, 3), head_mesh)})()
    head = type("WrappedHead", (), {"base_layer": base_layer})()

    aligned = global_kd_recipe._align_dtensor_to_module_mesh(hidden, head)

    assert aligned.device_mesh is head_mesh
    assert aligned.placements == hidden.placements
    assert FakeDTensor.calls == [(hidden.local, head_mesh, hidden.placements, False)]


def test_teacher_mtp_projection_uses_local_head_and_student_logit_mesh(monkeypatch):
    import torch

    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    class Mesh:
        pass

    class FakeDTensor:
        def __init__(self, local, mesh, placements):
            self.local = local
            self.device_mesh = mesh
            self.placements = placements

        def to_local(self):
            return self.local

        @classmethod
        def from_local(cls, local, *, device_mesh, placements, run_check):
            assert run_check is False
            return cls(local, device_mesh, placements)

    monkeypatch.setattr(global_kd_recipe, "DTensor", FakeDTensor)
    teacher_mesh = Mesh()
    student_mesh = Mesh()
    hidden = FakeDTensor(torch.tensor([[1.0, 2.0]]), teacher_mesh, ("replicate",))
    weight = FakeDTensor(torch.tensor([[1.0, 0.0], [0.0, 2.0]]), teacher_mesh, ("shard0",))
    head = type(
        "WrappedHead", (), {"base_layer": type("Base", (), {"weight": weight, "bias": None})()}
    )()
    reference = FakeDTensor(torch.empty(1, 2), student_mesh, ("shard_vocab",))

    projected = global_kd_recipe._project_teacher_hidden_on_reference_mesh(hidden, head, reference)

    assert projected.device_mesh is student_mesh
    assert projected.placements == reference.placements
    torch.testing.assert_close(projected.local, torch.tensor([[1.0, 4.0]]))


def test_teacher_mtp_projection_accepts_pipeline_local_hidden(monkeypatch):
    import torch

    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    class FakeDTensor:
        def __init__(self, local):
            self.local = local

        def to_local(self):
            return self.local

    monkeypatch.setattr(global_kd_recipe, "DTensor", FakeDTensor)
    weight = FakeDTensor(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    base = type("Base", (), {"weight": weight, "bias": None})()

    class WrappedHead:
        base_layer = base

        def __call__(self, hidden):
            raise AssertionError("distributed head must not receive pipeline-local hidden")

    projected = global_kd_recipe._project_teacher_hidden_on_reference_mesh(
        torch.tensor([[1.0, 2.0]]),
        WrappedHead(),
        torch.empty(1, 2),
    )

    torch.testing.assert_close(projected, torch.tensor([[1.0, 4.0]]))


def test_teacher_mtp_projection_unshards_fsdp_but_preserves_tp_vocab(monkeypatch):
    import torch

    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    class Mesh:
        mesh_dim_names = ("dp_cp", "tp")

    class Shard:
        def __init__(self, dim):
            self.dim = dim

    class Replicate:
        pass

    class FakeDTensor:
        def __init__(self, local, placements, *, gathered=None):
            self.local = local
            self.device_mesh = Mesh()
            self.placements = placements
            self.gathered = gathered
            self.redistributions = []

        def to_local(self):
            return self.local

        def redistribute(self, *, device_mesh, placements):
            self.redistributions.append((device_mesh, placements))
            return FakeDTensor(self.gathered, placements)

        @classmethod
        def from_local(cls, local, *, device_mesh, placements, run_check):
            assert run_check is False
            return cls(local, placements)

    monkeypatch.setattr(global_kd_recipe, "DTensor", FakeDTensor)
    monkeypatch.setattr(global_kd_recipe, "Replicate", Replicate, raising=False)
    weight = FakeDTensor(
        torch.tensor([[1.0, 0.0]]),
        (Shard(0), Shard(0)),
        gathered=torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
    )
    base = type("Base", (), {"weight": weight, "bias": None})()
    head = type("WrappedHead", (), {"base_layer": base})()
    reference = FakeDTensor(torch.empty(1, 2), (Replicate(), Shard(-1)))

    projected = global_kd_recipe._project_teacher_hidden_on_reference_mesh(
        torch.tensor([[1.0, 2.0]]), head, reference
    )
    projected_again = global_kd_recipe._project_teacher_hidden_on_reference_mesh(
        torch.tensor([[2.0, 1.0]]), head, reference
    )

    assert len(weight.redistributions) == 1
    _, placements = weight.redistributions[0]
    assert isinstance(placements[0], Replicate)
    assert placements[1] is weight.placements[1]
    torch.testing.assert_close(projected.local, torch.tensor([[1.0, 4.0]]))
    torch.testing.assert_close(projected_again.local, torch.tensor([[2.0, 2.0]]))


def test_gradient_groups_are_observed_at_optimizer_step():
    import torch

    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.vision = torch.nn.Linear(2, 2, bias=False)
            self.visual.merger = torch.nn.Linear(2, 2, bias=False)
            self.language = torch.nn.Linear(2, 2, bias=False)
            self.mtp = torch.nn.Linear(2, 2, bias=False)

    student = Student()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    recipe = object.__new__(_WeightedObjectiveMixin)
    recipe.model_parts = [student]
    recipe.optimizer = [optimizer]
    recipe.dist_env = type("DistEnv", (), {"device": torch.device("cpu")})()
    recipe._gradient_hook_handles = []
    recipe._install_gradient_norm_observers()

    for parameter in student.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert all(value.item() > 0 for value in recipe._gradient_squared.values())


def test_global_kd_checkpoint_forwards_best_metric_key(tmp_path, monkeypatch):
    # Lazy import keeps the optional NeMo AutoModel runtime out of test collection.
    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    calls = []
    refreshes = []

    class BaseRecipe:
        def save_checkpoint(
            self,
            epoch,
            step,
            train_loss,
            val_loss,
            best_metric_key="default",
        ):
            calls.append((epoch, step, train_loss, val_loss, best_metric_key))
            checkpoint = tmp_path / f"epoch_{epoch}_step_{step}"
            consolidated = checkpoint / "model/consolidated"
            consolidated.mkdir(parents=True)
            (consolidated / "config.json").write_text(
                json.dumps({"block_configs": [{"subblock_configs": []}]})
            )
            return "saved"

    class Recipe(_WeightedObjectiveMixin, BaseRecipe):
        pass

    recipe = Recipe()
    recipe.checkpointer = type(
        "Checkpointer",
        (),
        {"config": type("Config", (), {"checkpoint_dir": tmp_path})()},
    )()
    recipe.dist_env = type("DistEnv", (), {"is_main": True})()
    recipe.cfg = {"model": {"trust_remote_code": True}}
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.utils.vllm_adapter.refresh_realized_checkpoint_config",
        lambda path, **kwargs: refreshes.append(
            (path, kwargs, (tmp_path / "epoch_2_step_17/saving_completed").exists())
        ),
    )

    result = recipe.save_checkpoint(
        2,
        17,
        0.5,
        {"lm_loss": 0.25},
        best_metric_key="lm_loss",
    )

    assert result == "saved"
    assert calls == [(2, 17, 0.5, {"lm_loss": 0.25}, "lm_loss")]
    assert refreshes == [
        (
            tmp_path / "epoch_2_step_17/model/consolidated",
            {"trust_remote_code": True},
            False,
        )
    ]
    assert (tmp_path / "epoch_2_step_17" / "saving_completed").is_file()

    recipe.cfg = {"model": {"trust_remote_code": "false"}}
    with pytest.raises(ValueError, match=r"^model\.trust_remote_code must be a boolean$"):
        recipe.save_checkpoint(
            2,
            18,
            0.5,
            {"lm_loss": 0.25},
            best_metric_key="lm_loss",
        )
    assert not (tmp_path / "epoch_2_step_18" / "saving_completed").exists()


def test_global_kd_optimizer_save_uses_the_actual_pipeline_model_parts():
    import torch

    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    original = torch.nn.Linear(2, 2, bias=False)
    empty_tracked_model = torch.nn.Identity()
    optimizer = torch.optim.AdamW(original.parameters())
    saved_models = []

    class Checkpointer:
        def save_model(self, model, path):
            del path
            saved_models.append(model)

        def save_optimizer(self, saved_optimizer, model, path, scheduler):
            del path, scheduler
            saved_models.append(model)
            assert saved_optimizer.param_groups[0]["params"] == list(original.parameters())

    recipe = object.__new__(_WeightedObjectiveMixin)
    recipe.model_parts = [original]
    recipe.optimizer = [optimizer]
    recipe.checkpointer = Checkpointer()
    recipe._install_pre_optimizer_save_rebind()

    recipe.checkpointer.save_model(empty_tracked_model, "unused")
    recipe.checkpointer.save_optimizer(
        optimizer,
        empty_tracked_model,
        "unused",
        None,
    )

    assert saved_models == [recipe.model_parts, recipe.model_parts]


def test_global_kd_objective_setup_initializes_text_observability_buffers():
    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    recipe = object.__new__(_WeightedObjectiveMixin)
    recipe.cfg = {"objective": {}}

    recipe._configure_objective()

    assert recipe._vision_monitors == []
    assert recipe._media_input_checksums == []


def test_global_kd_text_optimizer_excludes_inactive_modality_branches():
    import torch

    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(2, 2)
            self.mm_projector = torch.nn.Linear(2, 2)
            self.language = torch.nn.Linear(2, 2)
            self.mtp = torch.nn.Linear(2, 2)

    model = ToyModel()
    recipe = object.__new__(_WeightedObjectiveMixin)
    recipe.model_parts = [model]
    recipe.optimizer = torch.optim.AdamW(model.parameters())

    recipe._remove_text_inactive_optimizer_parameters()

    optimized = {
        id(parameter) for group in recipe.optimizer.param_groups for parameter in group["params"]
    }
    assert all(id(parameter) not in optimized for parameter in model.visual.parameters())
    assert all(id(parameter) not in optimized for parameter in model.mm_projector.parameters())
    assert all(id(parameter) in optimized for parameter in model.language.parameters())
    assert all(id(parameter) in optimized for parameter in model.mtp.parameters())


def test_global_kd_quarantines_unmarked_checkpoint_even_with_dcp_metadata(tmp_path):
    from modelopt.torch.puzzletron.distillation.global_automodel import (
        _quarantine_incomplete_global_kd_checkpoints,
    )

    incomplete = tmp_path / "epoch_0_step_3"
    (incomplete / "model").mkdir(parents=True)
    (incomplete / "optim").mkdir()
    (incomplete / "model" / ".metadata").write_text("partial")
    (incomplete / "optim" / ".metadata").write_text("partial")
    (incomplete / "step_scheduler.pt").write_text("partial")

    complete = tmp_path / "epoch_0_step_2"
    complete.mkdir()
    (complete / "saving_completed").touch()
    poisoned = tmp_path / "epoch_0_step_4.incomplete-legacy"
    poisoned.mkdir()
    (poisoned / "saving_completed").touch()

    _quarantine_incomplete_global_kd_checkpoints(tmp_path)

    assert not incomplete.exists()
    quarantined = list((tmp_path / "_incomplete").glob("epoch_0_step_3*"))
    assert len(quarantined) == 1
    assert complete.is_dir()
    assert not poisoned.exists()
    assert len(list((tmp_path / "_incomplete").glob("epoch_0_step_4*"))) == 1

    _quarantine_incomplete_global_kd_checkpoints(tmp_path)

    assert list((tmp_path / "_incomplete").glob("epoch_0_step_3*")) == quarantined


def test_global_kd_training_log_is_reconciled_to_latest_durable_checkpoint(tmp_path):
    from modelopt.torch.puzzletron.distillation.global_automodel import (
        _reconcile_global_kd_training_log,
    )

    completed = tmp_path / "epoch_1_step_1"
    completed.mkdir()
    (completed / "saving_completed").touch()
    rows = [
        {"step": 0, "loss": 9.0, "attempt": "interrupted"},
        {"step": 1, "loss": 8.0, "attempt": "interrupted"},
        {"step": 0, "loss": 7.0, "attempt": "durable"},
        {"step": 1, "loss": 6.0, "attempt": "durable"},
        {"step": 2, "loss": 5.0, "attempt": "uncheckpointed-tail"},
    ]
    training_log = tmp_path / "training.jsonl"
    training_log.write_text("".join(json.dumps(row) + "\n" for row in rows))

    _reconcile_global_kd_training_log(tmp_path)

    actual = [json.loads(line) for line in training_log.read_text().splitlines()]
    assert actual == [rows[2], rows[3]]


def test_global_kd_training_log_is_cleared_without_a_durable_checkpoint(tmp_path):
    from modelopt.torch.puzzletron.distillation.global_automodel import (
        _reconcile_global_kd_training_log,
    )

    training_log = tmp_path / "training.jsonl"
    training_log.write_text(json.dumps({"step": 0, "loss": 9.0}) + "\n")

    _reconcile_global_kd_training_log(tmp_path)

    assert training_log.read_text() == ""


def test_llm_pp_optimizer_step_publishes_every_weighted_objective(monkeypatch):
    import torch

    from modelopt.torch.puzzletron.distillation import global_kd_recipe

    log_data = type(
        "LogData",
        (),
        {"step": 7, "metrics": {"num_label_tokens": 2}},
    )()
    monkeypatch.setattr(
        global_kd_recipe._AutoModelLLMKD,
        "_run_train_optim_step",
        lambda self, batches, max_grad_norm: log_data,
    )

    recipe = object.__new__(global_kd_recipe.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.needs_teacher = True
    recipe.pp_enabled = True
    recipe.device_mesh = type(
        "Mesh",
        (),
        {"mesh_dim_names": ("pp",), "mesh": torch.tensor([0])},
    )()
    recipe.dist_env = type(
        "DistEnv",
        (),
        {"device": torch.device("cpu"), "rank": 0, "is_main": True},
    )()
    recipe._objective_buffers = {
        "main_ce": [torch.tensor(2.0)],
        "main_kd": [torch.tensor(4.0)],
        "mtp_ce": [torch.tensor(6.0)],
        "mtp_kd": [torch.tensor(8.0)],
    }
    recipe._objective_step_cursor = dict.fromkeys(recipe._objective_buffers, 0)
    recipe._gradient_squared = {
        name: torch.tensor(0.0) for name in ("vision", "projector", "language", "mtp")
    }
    recipe._dp_allreduce = lambda value, include_cp: value

    actual = recipe._run_train_optim_step([], max_grad_norm=None)

    assert actual.metrics["main_ce"] == 1.0
    assert actual.metrics["main_kd"] == 2.0
    assert actual.metrics["mtp_ce"] == 3.0
    assert actual.metrics["mtp_kd"] == 4.0
    assert all(not values for values in recipe._objective_buffers.values())


def test_global_kd_uses_canonical_multimodal_packing_and_train_all(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"force_hf": False},
        "_runtime": {"descriptor": "qwen3_5"},
        "data": {
            "path": str(tmp_path / "intersyn"),
            "modality": "multimodal",
            "layout": "packed_varlen",
            "max_sample_length": 1536,
            "packing": {
                "pack_size": 2048,
                "packing_ratio": 0.9,
                "drop_long_samples": True,
            },
        },
        "distillation": {
            "domain": "vlm",
            "freeze_policy": "train_all",
            "validation_enabled": False,
            "student_dir": str(tmp_path / "student"),
            "teacher_dir": str(tmp_path / "teacher"),
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 1,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 1,
                }
            },
        },
    }

    kd = build_global_kd_config(config)
    recipe = build_automodel_global_kd_recipe(kd)

    assert kd.freeze_policy == "train_all"
    assert kd.teacher_descriptor == "qwen3_5"
    assert kd.student_descriptor == "qwen3_5"
    assert recipe["dataset"]["_target_"].endswith("load_materialized_conversation_dataset")
    assert recipe["dataset"]["path_or_dataset"] == str(tmp_path / "intersyn")
    assert recipe["packed_sequence"]["pack_size"] == 2048
    assert recipe["packed_sequence"]["max_packs"] == 128
    assert recipe["freeze_config"] == {
        "freeze_vision_tower": False,
        "freeze_language_model": False,
    }


def test_global_kd_bounds_canonical_padded_multimodal_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"force_hf": False},
        "_runtime": {"descriptor": "qwen3_5"},
        "data": {
            "path": str(tmp_path / "vlm"),
            "modality": "multimodal",
            "layout": "padded_varlen",
            "max_sample_length": 1024,
            "calibration": {"num_samples": 24},
        },
        "distillation": {
            "domain": "vlm",
            "validation_enabled": False,
            "student_dir": str(tmp_path / "student"),
            "teacher_dir": str(tmp_path / "teacher"),
            "automodel": {
                "parallel": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 1,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 1,
                }
            },
        },
    }

    recipe = build_automodel_global_kd_recipe(build_global_kd_config(config))

    assert recipe["dataset"]["num_samples"] == 24
    assert recipe["packed_sequence"] == {
        "packed_sequence_size": 0,
        "split_across_pack": False,
    }
