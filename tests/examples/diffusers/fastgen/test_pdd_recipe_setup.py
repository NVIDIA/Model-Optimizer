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

"""Released-AutoModel seam tests for the ModelOpt-owned PDD setup."""

from __future__ import annotations

import copy
import os
import pathlib
import subprocess
import sys

import pytest
import torch
import yaml
from _test_utils.torch.diffusers_models import create_tiny_qwen_image_pipeline_dir
from torch import nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from pdd.recipe import (
    _materialize_zero_step_adamw_state,
    _projection_identity,
    _require_fp32_optimizer_storage,
    _require_immutable_model_source,
    _resolve_model_source,
    _stage_and_shard_training_models,
    build_pdd_export_setup,
    build_pdd_setup,
    initialize_pdd_distributed,
    resolve_pdd_recipe_config,
)

from modelopt.torch.fastgen import PDDLayerSpec, convert_to_pdd_output_projection


def test_zero_step_adamw_state_preserves_first_lazy_update() -> None:
    torch.manual_seed(7)
    eager_model = nn.Linear(4, 3)
    lazy_model = copy.deepcopy(eager_model)
    optimizer_options = {
        "lr": 2.0e-5,
        "weight_decay": 0.01,
        "foreach": False,
        "fused": False,
    }
    eager_optimizer = torch.optim.AdamW(eager_model.parameters(), **optimizer_options)
    lazy_optimizer = torch.optim.AdamW(lazy_model.parameters(), **optimizer_options)
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in eager_model.named_parameters()
    }

    _materialize_zero_step_adamw_state(eager_optimizer)

    for name, parameter in eager_model.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name], rtol=0, atol=0)
        assert parameter.grad is None
        state = eager_optimizer.state[parameter]
        assert state["step"].item() == 0
        assert not state["exp_avg"].count_nonzero()
        assert not state["exp_avg_sq"].count_nonzero()

    eager_model.weight.square().mean().backward()
    lazy_model.weight.square().mean().backward()
    eager_optimizer.step()
    lazy_optimizer.step()

    for eager_parameter, lazy_parameter in zip(
        eager_model.parameters(), lazy_model.parameters(), strict=True
    ):
        torch.testing.assert_close(eager_parameter, lazy_parameter, rtol=0, atol=0)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            eager_optimizer.state[eager_model.weight][key],
            lazy_optimizer.state[lazy_model.weight][key],
            rtol=0,
            atol=0,
        )
    assert eager_optimizer.state[eager_model.bias]["step"].item() == 0
    assert lazy_model.bias not in lazy_optimizer.state


def test_fp32_optimizer_storage_rejects_low_precision_masters_and_state() -> None:
    model = nn.Linear(2, 2, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
    _materialize_zero_step_adamw_state(optimizer)
    with pytest.raises(RuntimeError, match="master parameters must be FP32"):
        _require_fp32_optimizer_storage(optimizer)

    fp32_model = nn.Linear(2, 2)
    fp32_optimizer = torch.optim.AdamW(fp32_model.parameters(), foreach=False, fused=False)
    _materialize_zero_step_adamw_state(fp32_optimizer)
    fp32_optimizer.state[fp32_model.weight]["exp_avg"] = torch.zeros_like(
        fp32_model.weight, dtype=torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="exp_avg state must be FP32"):
        _require_fp32_optimizer_storage(fp32_optimizer)


def test_training_setup_shards_student_before_staging_teacher() -> None:
    events: list[str] = []

    class TrackedModel(nn.Module):
        def __init__(self, label: str, *, projection: bool = False) -> None:
            super().__init__()
            self.label = label
            self.proj_out = nn.Linear(2, 2) if projection else nn.Identity()

        def to(self, *args, **kwargs):
            events.append(f"{self.label}.to")
            return super().to(*args, **kwargs)

    class TrackedManager:
        def parallelize(self, model):
            events.append(f"{model.label}.parallelize")
            return model

    student = TrackedModel("student", projection=True)
    teacher = TrackedModel("teacher")
    projection = convert_to_pdd_output_projection(
        student,
        PDDLayerSpec("proj_out", "channel_major"),
        grid_size=4,
    )

    staged_student, staged_teacher = _stage_and_shard_training_models(
        student,
        teacher,
        projection,
        _projection_identity(projection),
        TrackedManager(),
        device=torch.device("cpu"),
        fuse_qkv_projections=False,
    )

    assert staged_student is student
    assert staged_teacher is teacher
    assert {parameter.dtype for parameter in staged_student.parameters()} == {torch.float32}
    assert events == [
        "student.to",
        "student.parallelize",
        "teacher.to",
        "teacher.parallelize",
    ]


def test_training_setup_upcasts_bf16_models_to_fp32_masters() -> None:
    class IdentityManager:
        @staticmethod
        def parallelize(model):
            return model

    student = nn.Module()
    student.proj_out = nn.Linear(2, 2, dtype=torch.bfloat16)
    teacher = nn.Linear(2, 2, dtype=torch.bfloat16)
    projection = convert_to_pdd_output_projection(
        student,
        PDDLayerSpec("proj_out", "channel_major"),
        grid_size=4,
    )

    staged_student, staged_teacher = _stage_and_shard_training_models(
        student,
        teacher,
        projection,
        _projection_identity(projection),
        IdentityManager(),
        device=torch.device("cpu"),
        fuse_qkv_projections=False,
    )

    assert {parameter.dtype for parameter in staged_student.parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in staged_teacher.parameters()} == {torch.float32}


def _raw_config(model_dir: pathlib.Path, *, qkv: bool = False) -> dict:
    return {
        "model": {
            "pretrained_model_name_or_path": str(model_dir),
            "torch_dtype": "bfloat16",
            "device": "cpu",
            "transformer_engine_linear": False,
            "peft": None,
            "guidance_embeds": False,
            "fuse_qkv_projections": qkv,
        },
        "pdd": {
            "pred_type": "flow",
            "num_train_timesteps": None,
            "guidance_scale": 4.0,
            "student_sample_steps": 2,
            "student_sample_type": "ode",
            "grid_size": 4,
            "grid_max_t": 0.999,
            "flow_shift": 5.0,
            "block_size_min": 1,
            "block_size_max": 4,
            "teacher_integrator": "euler",
            "inference_blocks": [2, 2],
            "data_free": False,
        },
        "seed": 42,
        "optim": {
            "learning_rate": 2.0e-5,
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "weight_decay": 0.01,
            },
        },
        "lr_scheduler": {
            "lr_decay_style": "constant",
            "lr_warmup_steps": 0,
            "min_lr": 2.0e-5,
        },
        "step_scheduler": {
            "max_steps": 10,
            "num_epochs": 2,
            "log_every": 1,
            "ckpt_every_steps": 5,
            "local_batch_size": 1,
            "global_batch_size": 1,
            "save_checkpoint_every_epoch": False,
        },
        "training_health": {"max_grad_norm": 1.0, "zero_grad_warmup_steps": 0},
        "validation": {"count": 3, "seed": 11, "split_seed": 7, "every_steps": 5},
        "data": {
            "dataloader": {
                "_target_": "fastgen_data.build_text_to_image_multiresolution_dataloader",
                "batch_size": 1,
                "drop_last": True,
                "shuffle": True,
                "dynamic_batch_size": False,
            }
        },
        "fsdp": {
            "dp_size": 1,
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "activation_checkpointing": False,
        },
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": "checkpoints/test",
            "model_save_format": "torch_save",
            "save_consolidated": False,
        },
    }


def test_example_recipe_explicitly_pins_grid_max_t() -> None:
    raw = yaml.safe_load((_FASTGEN_DIR / "pdd" / "configs" / "qwen_image.yaml").read_text())
    assert type(raw["pdd"]["grid_max_t"]) is float
    assert raw["pdd"]["grid_max_t"] == 0.999
    assert raw["validation"]["count"] == 2000
    assert raw["validation"]["split_seed"] == 2026


def test_split_config_fields_are_strict(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["validation"] = {"count": 3, "seed": 11, "split_seed": 7, "every_steps": 5}
    config = resolve_pdd_recipe_config(raw)
    assert config.validation.count == 3
    assert config.validation.split_seed == 7

    for invalid_count in (0, -1, True, 1.5):
        raw["validation"]["count"] = invalid_count
        with pytest.raises((TypeError, ValueError), match=r"validation\.count"):
            resolve_pdd_recipe_config(raw)

    raw["validation"] = {"count": 3, "seed": 11, "split_seed": -1, "every_steps": 5}
    with pytest.raises(ValueError, match="split_seed"):
        resolve_pdd_recipe_config(raw)


def test_config_node_and_canonical_dotted_values_are_consumed(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["step_scheduler"].update(
        max_steps=50_000,
        ckpt_every_steps=1_000,
        global_batch_size=1,
    )
    raw["optim"]["learning_rate"] = 3.0e-5
    raw["lr_scheduler"]["min_lr"] = 3.0e-5

    class ConfigNodeLike:
        def to_dict(self):
            return copy.deepcopy(raw)

    config = resolve_pdd_recipe_config(ConfigNodeLike())
    assert config.step_scheduler.max_steps == 50_000
    assert config.step_scheduler.ckpt_every_steps == 1_000
    assert config.step_scheduler.global_batch_size == 1
    assert config.learning_rate == 3.0e-5


def test_automodel_parser_dotted_overrides_reach_the_pdd_resolver(tmp_path, monkeypatch) -> None:
    parser_module = pytest.importorskip("nemo_automodel.components.config._arg_parser")
    config_path = tmp_path / "pdd.yaml"
    config_path.write_text(yaml.safe_dump(_raw_config(tmp_path)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune.py",
            "--config",
            str(config_path),
            "--step_scheduler.max_steps=50000",
            "--step_scheduler.ckpt_every_steps=1000",
            "--step_scheduler.global_batch_size=1",
            "--optim.learning_rate=3e-5",
            "--lr_scheduler.min_lr=3e-5",
        ],
    )

    parsed = parser_module.parse_args_and_load_config(str(config_path))
    resolved = resolve_pdd_recipe_config(parsed)
    assert resolved.step_scheduler.max_steps == 50_000
    assert resolved.step_scheduler.ckpt_every_steps == 1_000
    assert resolved.step_scheduler.global_batch_size == 1
    assert resolved.learning_rate == 3.0e-5


@pytest.mark.parametrize(
    ("legacy_key", "replacement"),
    [
        ("max_steps", "step_scheduler.max_steps"),
        ("global_batch_size", "step_scheduler.global_batch_size"),
        ("checkpoint_every_steps", "step_scheduler.ckpt_every_steps"),
        ("log_every_steps", "step_scheduler.log_every"),
    ],
)
def test_legacy_training_lifecycle_keys_are_rejected(tmp_path, legacy_key, replacement) -> None:
    raw = _raw_config(tmp_path)
    raw["training"] = {legacy_key: 2}
    with pytest.raises(ValueError, match=replacement.replace(".", r"\.")):
        resolve_pdd_recipe_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lr_decay_style", "cosine", "lr_decay_style='constant'"),
        ("lr_warmup_steps", 1, "lr_warmup_steps=0"),
        ("min_lr", 1.0e-5, "min_lr must equal optim.learning_rate"),
    ],
)
def test_nonconstant_lr_declarations_are_rejected(tmp_path, field, value, message) -> None:
    raw = _raw_config(tmp_path)
    raw["lr_scheduler"][field] = value
    with pytest.raises(ValueError, match=message):
        resolve_pdd_recipe_config(raw)


@pytest.mark.parametrize("field", ["weight_decay", "betas", "eps"])
def test_legacy_optimizer_fields_are_rejected(tmp_path, field) -> None:
    raw = _raw_config(tmp_path)
    raw["optim"][field] = {
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
    }[field]
    with pytest.raises(ValueError, match=rf"optim\.optimizer\.{field}"):
        resolve_pdd_recipe_config(raw)


def test_pdd_rejects_external_split_manifest(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["data"] = {"dataloader": {"metadata_index": "metadata_train.json"}}
    with pytest.raises(ValueError, match="metadata_index is unsupported"):
        resolve_pdd_recipe_config(raw)


def test_pdd_finetune_namespace_module_help() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "examples.diffusers.fastgen.pdd.finetune", "--help"],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Qwen-Image PDD training" in result.stdout


@pytest.mark.parametrize(
    ("scope", "name", "value", "message"),
    [
        ("model", "transformer_engine_linear", True, "TE-linear"),
        ("model", "peft", {"rank": 8}, "PEFT/LoRA"),
        ("root", "peft_cfg", {"rank": 8}, "PEFT/LoRA"),
        ("model", "guidance_embeds", True, "guidance embeddings"),
        ("model", "device_map", "auto", "device_map"),
        ("model", "quantization_config", {"bits": 8}, "quantization_config"),
    ],
)
def test_incompatible_modes_fail_during_config_resolution(
    tmp_path, scope, name, value, message
) -> None:
    raw = _raw_config(tmp_path)
    target = raw if scope == "root" else raw[scope]
    target[name] = value

    with pytest.raises(ValueError, match=message):
        resolve_pdd_recipe_config(raw)


def test_model_revision_and_compute_dtype_follow_loader_contract(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["model"]["pretrained_model_name_or_path"] = "Qwen/Qwen-Image"
    raw["model"]["revision"] = "a" * 40
    raw["model"]["torch_dtype"] = "float32"
    raw["model"]["fuse_qkv_projections"] = True

    config = resolve_pdd_recipe_config(raw)

    assert config.model_revision == "a" * 40
    assert config.dtype == torch.float32
    assert config.fuse_qkv_projections is True


@pytest.mark.parametrize("revision", [None, "main", "A" * 40, "a" * 39])
def test_remote_model_requires_exact_lowercase_commit(tmp_path, revision) -> None:
    raw = _raw_config(tmp_path)
    raw["model"]["pretrained_model_name_or_path"] = "Qwen/Qwen-Image"
    raw["model"]["revision"] = revision

    with pytest.raises(ValueError, match="exact lowercase 40-character"):
        resolve_pdd_recipe_config(raw)


def test_model_source_resolution_requires_the_requested_snapshot(tmp_path, monkeypatch) -> None:
    commit = "a" * 40
    raw = _raw_config(tmp_path)
    raw["model"]["pretrained_model_name_or_path"] = "Qwen/Qwen-Image"
    raw["model"]["revision"] = commit
    config = resolve_pdd_recipe_config(raw)
    snapshot = tmp_path / "hub" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    calls = []

    def matching_snapshot(model_id, *, revision):
        calls.append((model_id, revision))
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", matching_snapshot)
    assert _resolve_model_source(config) == str(snapshot.resolve())
    assert calls == [("Qwen/Qwen-Image", commit)]
    _require_immutable_model_source(config, context="test")

    wrong = snapshot.with_name("b" * 40)
    wrong.mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *_args, **_kwargs: str(wrong))
    with pytest.raises(RuntimeError, match=r"does not match model\.revision"):
        _resolve_model_source(config)


def test_local_model_source_is_limited_to_low_level_setup(tmp_path, monkeypatch) -> None:
    config = resolve_pdd_recipe_config(_raw_config(tmp_path))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_args, **_kwargs: pytest.fail("local model resolution must not access the Hub"),
    )

    assert _resolve_model_source(config) == str(tmp_path.resolve())
    with pytest.raises(ValueError, match="Checkpointed PDD training requires"):
        _require_immutable_model_source(config, context="Checkpointed PDD training")


def test_non_dp_parallelism_is_rejected(tmp_path) -> None:
    raw = _raw_config(tmp_path)

    raw["fsdp"]["tp_size"] = 2
    with pytest.raises(ValueError, match="tp_size must be 1"):
        resolve_pdd_recipe_config(raw)


@pytest.mark.parametrize(
    ("section", "name", "value", "message"),
    [
        (
            "step_scheduler",
            "save_checkpoint_every_epoch",
            True,
            "save_checkpoint_every_epoch=false",
        ),
        ("training_health", "max_grad_norm", 0.0, "max_grad_norm must be > 0"),
        ("validation", "every_steps", 0, "validation.every_steps"),
        ("guidance", "rescale", 1.1, "does not support guidance overrides"),
        ("optimizer", "betas", [0.9, 1.0], "optim.optimizer.betas values"),
        ("optimizer", "eps", 0.0, "optim.optimizer.eps must be > 0"),
    ],
)
def test_training_config_gates_fail_during_resolution(
    tmp_path, section, name, value, message
) -> None:
    raw = _raw_config(tmp_path)
    target = (
        raw["optim"].setdefault("optimizer", {})
        if section == "optimizer"
        else raw.setdefault(section, {})
    )
    target[name] = value

    with pytest.raises(ValueError, match=message):
        resolve_pdd_recipe_config(raw)


def test_restore_requires_enabled_checkpointing(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["checkpoint"]["enabled"] = False
    raw["checkpoint"]["restore_from"] = "LATEST"

    with pytest.raises(ValueError, match=r"restore_from requires checkpoint\.enabled=true"):
        resolve_pdd_recipe_config(raw)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("drop_last", False, "drop_last=true"),
        ("dynamic_batch_size", True, "dynamic_batch_size=false"),
        ("train_text_encoder", True, "cached text embeddings"),
    ],
)
def test_training_dataloader_modes_are_gated_during_resolution(
    tmp_path, name, value, message
) -> None:
    raw = _raw_config(tmp_path)
    raw["data"] = {"dataloader": {name: value}}

    with pytest.raises(ValueError, match=message):
        resolve_pdd_recipe_config(raw)


def test_payload_hash_verification_mode_must_be_bool(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["data"]["dataloader"]["verify_payload_hashes"] = "false"

    with pytest.raises(TypeError, match=r"data\.dataloader\.verify_payload_hashes must be bool"):
        resolve_pdd_recipe_config(raw)


def test_real_loader_manager_optimizer_and_checkpoint_restore(tmp_path) -> None:
    model_dir = create_tiny_qwen_image_pipeline_dir(tmp_path)
    initialize_pdd_distributed(backend="gloo", timeout_minutes=1)
    config = resolve_pdd_recipe_config(_raw_config(model_dir))

    source = build_pdd_setup(config)

    assert source.lifecycle == (
        "load/select",
        "pdd_conversion",
        "device",
        "qkv",
        "parallelize",
        "optimizer",
        "checkpoint",
    )
    assert type(source.pipe).__name__ == "QwenImagePipeline"
    assert source.pipe.text_encoder is None
    assert source.pipe.tokenizer is None
    assert source.pipe.vae is None
    assert source.pipe.transformer is source.student
    assert source.student.get_submodule("proj_out") is source.projection
    assert source.projection.out_features == source.projection.base_out_features * 4
    assert "proj_out.weight" in source.checkpoint_keys
    assert source.student.state_dict()["proj_out.weight"].shape[0] == source.projection.out_features
    policy = source.distributed_setup.strategy_config.mp_policy
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype == torch.bfloat16
    assert policy.cast_forward_inputs is True
    assert all(
        base.__module__ != "modelopt.torch.fastgen.plugins.qwen_image_pdd"
        for base in type(source.student).__mro__
    )
    assert not any(parameter.requires_grad for parameter in source.teacher.parameters())
    optimizer_parameters = [
        parameter for group in source.optimizer.param_groups for parameter in group["params"]
    ]
    assert any(parameter is source.projection.weight for parameter in optimizer_parameters)
    assert set(source.optimizer.state) == set(optimizer_parameters)
    assert all(
        source.optimizer.state[parameter]["step"].item() == 0 for parameter in optimizer_parameters
    )
    unused_parameter = next(
        parameter for parameter in optimizer_parameters if parameter is not source.projection.weight
    )
    unused_name = next(
        name
        for name, parameter in source.student.named_parameters()
        if parameter is unused_parameter
    )
    # Diffusers 0.38 accepts the Qwen object API but currently performs no effective fusion.
    assert not any(
        getattr(module, "fused_projections", False) for module in source.student.modules()
    )

    source.optimizer.zero_grad(set_to_none=True)
    # Exercise strict stock-DCP restore after a partial-gradient update. Eager step-zero state must
    # retain exact lazy-Adam semantics for untouched parameters while keeping every DCP key present.
    source.projection.weight.float().square().mean().backward()
    source.optimizer.step()
    expected_weight = source.projection.weight.detach().clone()
    expected_exp_avg = source.optimizer.state[source.projection.weight]["exp_avg"].clone()
    assert source.optimizer.state[unused_parameter]["step"].item() == 0
    checkpoint_root = tmp_path / "checkpoint"
    source.checkpointer.save_model(source.student, str(checkpoint_root))
    source.checkpointer.save_optimizer(source.optimizer, source.student, str(checkpoint_root))

    export_setup = build_pdd_export_setup(config)
    assert export_setup.lifecycle == (
        "load/select",
        "pdd_conversion",
        "device",
        "qkv",
        "parallelize",
        "checkpoint",
    )
    assert export_setup.metadata == source.metadata
    assert export_setup.checkpoint_keys == source.checkpoint_keys
    assert not hasattr(export_setup, "optimizer")
    export_setup.checkpointer.load_model(
        export_setup.student,
        str(checkpoint_root / "model"),
    )
    torch.testing.assert_close(
        export_setup.student.state_dict()["proj_out.weight"],
        expected_weight,
    )

    destination = build_pdd_setup(config)
    assert destination.metadata == source.metadata
    destination_projection = destination.projection
    destination_unused = dict(destination.student.named_parameters())[unused_name]
    destination_weight_id = id(destination_projection.weight)
    destination.checkpointer.load_model(
        destination.student,
        str(checkpoint_root / "model"),
    )
    destination.checkpointer.load_optimizer(
        destination.optimizer,
        destination.student,
        str(checkpoint_root),
    )

    assert destination.student.get_submodule("proj_out") is destination_projection
    assert id(destination_projection.weight) == destination_weight_id
    torch.testing.assert_close(destination_projection.weight, expected_weight)
    torch.testing.assert_close(
        destination.optimizer.state[destination_projection.weight]["exp_avg"],
        expected_exp_avg,
    )
    assert destination.optimizer.state[destination_unused]["step"].item() == 0
    assert not destination.optimizer.state[destination_unused]["exp_avg"].count_nonzero()
    assert not destination.optimizer.state[destination_unused]["exp_avg_sq"].count_nonzero()
    source.checkpointer.close()
    destination.checkpointer.close()
    export_setup.checkpointer.close()


def test_qwen_pdd_adapter_has_no_automodel_import() -> None:
    source = (
        _REPO_ROOT / "modelopt" / "torch" / "fastgen" / "plugins" / "qwen_image_pdd.py"
    ).read_text()
    assert "nemo_automodel" not in source
