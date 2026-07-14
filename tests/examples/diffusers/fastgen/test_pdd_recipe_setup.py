# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Released-AutoModel seam tests for the ModelOpt-owned PDD setup."""

from __future__ import annotations

import importlib.metadata
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch
from _test_utils.torch.diffusers_models import create_tiny_qwen_image_pipeline_dir

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from pdd_recipe import build_pdd_setup, initialize_pdd_distributed, resolve_pdd_recipe_config
from verify_readonly_automodel import snapshot_installed_distribution


def _raw_config(model_dir: pathlib.Path, *, qkv: bool = False) -> dict:
    return {
        "model": {
            "pretrained_model_name_or_path": str(model_dir),
            "torch_dtype": "float32",
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
            "flow_shift": 5.0,
            "block_size_min": 1,
            "block_size_max": 4,
            "teacher_integrator": "euler",
            "inference_blocks": [2, 2],
            "data_free": False,
        },
        "optim": {"learning_rate": 2.0e-5, "weight_decay": 0.01},
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


def test_remote_model_requires_full_revision_and_non_dp_parallelism_is_rejected(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["model"]["pretrained_model_name_or_path"] = "Qwen/Qwen-Image"
    with pytest.raises(ValueError, match=r"exact model\.revision"):
        resolve_pdd_recipe_config(raw)

    raw["model"]["revision"] = "a" * 40
    raw["fsdp"]["tp_size"] = 2
    with pytest.raises(ValueError, match="tp_size must be 1"):
        resolve_pdd_recipe_config(raw)


@pytest.mark.parametrize(
    ("section", "name", "value", "message"),
    [
        ("training", "grad_accumulation_steps", 2, "grad_accumulation_steps=1"),
        ("training", "max_grad_norm", 0.0, "max_grad_norm must be > 0"),
        ("training", "validation_every_steps", 0, "validation_every_steps"),
        ("guidance", "rescale", 1.1, "guidance.rescale must be <= 1"),
        ("optim", "betas", [0.9, 1.0], "optim.betas values"),
        ("optim", "eps", 0.0, "optim.eps must be > 0"),
    ],
)
def test_training_config_gates_fail_during_resolution(
    tmp_path, section, name, value, message
) -> None:
    raw = _raw_config(tmp_path)
    raw.setdefault(section, {})[name] = value

    with pytest.raises(ValueError, match=message):
        resolve_pdd_recipe_config(raw)


def test_restore_requires_enabled_checkpointing(tmp_path) -> None:
    raw = _raw_config(tmp_path)
    raw["checkpoint"]["enabled"] = False
    raw["checkpoint"]["restore_from"] = "LATEST"

    with pytest.raises(ValueError, match="restore_from requires checkpoint.enabled=true"):
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


def test_frozen_automodel_distribution_snapshot_is_stable() -> None:
    try:
        version = importlib.metadata.version("nemo_automodel")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("nemo_automodel is not installed")
    assert version == "0.5.0"

    before = snapshot_installed_distribution()
    after = snapshot_installed_distribution()

    assert before == after
    assert before["version"] == "0.5.0"
    assert before["release_commit"] == "d02f49cb314554715aabb97e8dba6599c9f6e9e0"
    assert before["runtime_versions"] == {"diffusers": "0.38.0"}
    assert before["package_file_count"] == 490
    assert before["package_tree_sha256"] == (
        "b43cb34e04992c66d1888abc0529b760b5b69fc121ff4268b42ecb4a89b1e528"
    )


def test_exact_wheel_install_below_git_checkout_is_accepted(tmp_path) -> None:
    try:
        distribution = importlib.metadata.distribution("nemo_automodel")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("nemo_automodel is not installed")
    assert distribution.version == "0.5.0"

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    site_packages = checkout / ".venv" / "lib" / "python" / "site-packages"
    site_packages.mkdir(parents=True)
    installed_root = pathlib.Path(distribution.locate_file("")).resolve()
    shutil.copytree(installed_root / "nemo_automodel", site_packages / "nemo_automodel")
    dist_info_name = "nemo_automodel-0.5.0.dist-info"
    shutil.copytree(installed_root / dist_info_name, site_packages / dist_info_name)

    output = tmp_path / "snapshot.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(site_packages), environment.get("PYTHONPATH")))
    )
    subprocess.run(
        [
            sys.executable,
            str(_FASTGEN_DIR / "verify_readonly_automodel.py"),
            "snapshot",
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )

    snapshot = json.loads(output.read_text())
    assert pathlib.Path(snapshot["root"]) == site_packages.resolve()
    assert pathlib.Path(snapshot["import_origin"]).is_relative_to(site_packages.resolve())
    assert snapshot["package_tree_sha256"] == (
        "b43cb34e04992c66d1888abc0529b760b5b69fc121ff4268b42ecb4a89b1e528"
    )


def test_real_loader_manager_optimizer_and_checkpoint_restore(tmp_path) -> None:
    before = snapshot_installed_distribution()
    model_dir = create_tiny_qwen_image_pipeline_dir(tmp_path)
    initialize_pdd_distributed(backend="gloo", timeout_minutes=1)
    config = resolve_pdd_recipe_config(_raw_config(model_dir, qkv=True))

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
    assert not any(parameter.requires_grad for parameter in source.teacher.parameters())
    optimizer_parameters = [
        parameter for group in source.optimizer.param_groups for parameter in group["params"]
    ]
    assert any(parameter is source.projection.weight for parameter in optimizer_parameters)
    # Diffusers 0.38 accepts the Qwen object API but currently performs no effective fusion.
    assert not any(
        getattr(module, "fused_projections", False) for module in source.student.modules()
    )

    source.optimizer.zero_grad(set_to_none=True)
    # A real PDD forward touches the backbone and projection. Exercise the strict stock
    # optimizer restore with complete Adam state rather than an artificial partial update.
    sum(parameter.float().square().mean() for parameter in optimizer_parameters).backward()
    source.optimizer.step()
    expected_weight = source.projection.weight.detach().clone()
    expected_exp_avg = source.optimizer.state[source.projection.weight]["exp_avg"].clone()
    checkpoint_root = tmp_path / "checkpoint"
    source.checkpointer.save_model(source.student, str(checkpoint_root))
    source.checkpointer.save_optimizer(source.optimizer, source.student, str(checkpoint_root))

    destination = build_pdd_setup(config)
    assert destination.metadata == source.metadata
    destination_projection = destination.projection
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
    assert snapshot_installed_distribution() == before
    source.checkpointer.close()
    destination.checkpointer.close()


def test_qwen_pdd_adapter_has_no_automodel_import() -> None:
    source = (
        _REPO_ROOT / "modelopt" / "torch" / "fastgen" / "plugins" / "qwen_image_pdd.py"
    ).read_text()
    assert "nemo_automodel" not in source
