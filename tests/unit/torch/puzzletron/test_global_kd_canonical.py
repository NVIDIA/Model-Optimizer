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


def test_global_kd_checkpoint_publication_failure_reaches_all_ranks(tmp_path, monkeypatch):
    """Preserve the failing rank's exception while every rank completes collectives."""

    # Import the dynamic mixin at test runtime to preserve lightweight module collection.
    from modelopt.torch.puzzletron.distillation.global_kd_recipe import _WeightedObjectiveMixin

    publication = {"error": None, "broadcasts": 0}
    parent_save_errors = [None, None]

    class BaseRecipe:
        fail_rank = None

        def save_checkpoint(
            self,
            epoch,
            step,
            train_loss,
            val_loss,
            best_metric_key="default",
        ):
            del train_loss, val_loss, best_metric_key
            rank = 0 if self.dist_env.is_main else 1
            if self.fail_rank == rank:
                raise OSError("parent save failed")
            consolidated = tmp_path / f"epoch_{epoch}_step_{step}/model/consolidated"
            consolidated.mkdir(parents=True, exist_ok=True)
            (consolidated / "config.json").write_text(
                json.dumps({"block_configs": [{"subblock_configs": []}]})
            )
            return "saved"

    class Recipe(_WeightedObjectiveMixin, BaseRecipe):
        pass

    def broadcast_object_list(payload, *, src):
        assert src == 0
        publication["broadcasts"] += 1
        if payload[0] is not None:
            publication["error"] = payload[0]
        else:
            payload[0] = publication["error"]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, _value: output.__setitem__(slice(None), parent_save_errors),
    )
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.utils.vllm_adapter.refresh_realized_checkpoint_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("refresh failed")),
    )

    def recipe(*, is_main):
        instance = Recipe()
        instance.checkpointer = type(
            "Checkpointer",
            (),
            {"config": type("Config", (), {"checkpoint_dir": tmp_path})()},
        )()
        instance.dist_env = type("DistEnv", (), {"is_main": is_main})()
        instance.cfg = {"model": {"trust_remote_code": False}}
        return instance

    # A main-rank publication failure is re-raised locally and reported to its peer.
    with pytest.raises(ValueError, match="refresh failed"):
        recipe(is_main=True).save_checkpoint(2, 19, 0.5, {"lm_loss": 0.25})
    assert publication == {
        "error": "publication failed on rank 0: ValueError: refresh failed",
        "broadcasts": 1,
    }
    assert not (tmp_path / "epoch_2_step_19/saving_completed").exists()

    with pytest.raises(
        RuntimeError,
        match="global KD checkpoint publication failed on rank 0: ValueError: refresh failed",
    ):
        recipe(is_main=False).save_checkpoint(2, 19, 0.5, {"lm_loss": 0.25})
    assert publication["broadcasts"] == 2

    # A main-rank parent-save failure follows the same collective path.
    publication.update(error=None, broadcasts=0)
    parent_save_errors[:] = ["OSError: parent save failed", None]
    BaseRecipe.fail_rank = 0
    with pytest.raises(OSError, match="parent save failed"):
        recipe(is_main=True).save_checkpoint(2, 23, 0.5, {"lm_loss": 0.25})
    assert publication == {
        "error": "parent save failed on rank 0: OSError: parent save failed",
        "broadcasts": 1,
    }
    assert not (tmp_path / "epoch_2_step_23/saving_completed").exists()

    with pytest.raises(
        RuntimeError,
        match="global KD checkpoint parent save failed on rank 0: OSError: parent save failed",
    ):
        recipe(is_main=False).save_checkpoint(2, 23, 0.5, {"lm_loss": 0.25})
    assert publication["broadcasts"] == 2

    # A non-main parent-save failure is preserved there and contextualized on rank zero.
    publication.update(error=None, broadcasts=0)
    parent_save_errors[:] = [None, "OSError: parent save failed"]
    BaseRecipe.fail_rank = 1
    with pytest.raises(
        RuntimeError,
        match="global KD checkpoint parent save failed on rank 1: OSError: parent save failed",
    ):
        recipe(is_main=True).save_checkpoint(2, 29, 0.5, {"lm_loss": 0.25})
    assert publication == {
        "error": "parent save failed on rank 1: OSError: parent save failed",
        "broadcasts": 1,
    }
    assert not (tmp_path / "epoch_2_step_29/saving_completed").exists()

    with pytest.raises(OSError, match="parent save failed"):
        recipe(is_main=False).save_checkpoint(2, 29, 0.5, {"lm_loss": 0.25})
    assert publication["broadcasts"] == 2


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
