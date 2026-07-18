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

"""Tests for authenticated PDD export, reconstruction, and schedules."""

from __future__ import annotations

import copy
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pdd.export import (
    PDD_INFERENCE_SCHEDULES,
    inspect_pdd_export,
    load_pdd_export_into_model,
    pdd_config_from_metadata,
    write_pdd_export,
)
from pdd.inference_qwen_image import (
    _model_identity,
    _normalize_prompt_condition,
    _validate_qwen_projection,
)

import modelopt.torch.fastgen.plugins.qwen_image_pdd as qwen_image_pdd_plugin
from modelopt.torch.fastgen import PDDConfig, PDDMetadata, PDDPipeline
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QWEN_IMAGE_PDD_EXECUTION,
    QwenImagePDDAdapter,
    convert_qwen_image_to_pdd,
)


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(guidance_embeds=False, in_channels=4)
        self._modelopt_qwen_image_pdd_execution = QWEN_IMAGE_PDD_EXECUTION
        self.backbone = nn.Linear(4, 5, dtype=torch.bfloat16)
        self.proj_out = nn.Linear(5, 4, dtype=torch.bfloat16)
        self.calls = 0

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        img_shapes,
        max_txt_seq_len,
        return_dict,
    ):
        del img_shapes, max_txt_seq_len
        assert return_dict is False
        condition = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        condition += (encoder_hidden_states_mask.sum(dim=1)[:, None, None] / 100).to(
            condition.dtype
        )
        hidden = torch.tanh(self.backbone(hidden_states))
        self.calls += 1
        hidden = hidden + condition
        hidden = hidden + (timestep[:, None, None] / 10).to(hidden.dtype)
        return (self.proj_out(hidden),)


@pytest.fixture(autouse=True)
def _allow_tiny_qwen_protocol_double(monkeypatch):
    require_production_forward = qwen_image_pdd_plugin.require_qwen_image_mr210_forward

    def require_forward(model: nn.Module) -> str:
        if type(model) is _TinyQwen:
            if model._modelopt_qwen_image_pdd_execution != QWEN_IMAGE_PDD_EXECUTION:
                raise RuntimeError(
                    "Qwen-Image PDD requires the bound FastGen MR210 forward execution."
                )
            return QWEN_IMAGE_PDD_EXECUTION
        return require_production_forward(model)

    monkeypatch.setattr(qwen_image_pdd_plugin, "require_qwen_image_mr210_forward", require_forward)


def _config(blocks=(32, 32, 32, 32)) -> PDDConfig:
    return PDDConfig(
        grid_size=128,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=4,
        block_size_max=64,
        inference_blocks=list(blocks),
        student_sample_steps=len(blocks),
        guidance_scale=4.0,
        num_train_timesteps=None,
    )


def _converted(seed: int = 17):
    torch.manual_seed(seed)
    model = _TinyQwen()
    config = _config()
    projection = convert_qwen_image_to_pdd(model, config)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) / 10)
    return model, config, PDDMetadata.from_config(config, projection)


def _identity(metadata: PDDMetadata) -> dict:
    return {
        "schema_version": 5,
        "qwen_image": {"execution": QWEN_IMAGE_PDD_EXECUTION},
        "model": {"id": "synthetic-qwen", "revision": "f" * 40, "dtype": "bfloat16"},
        "pdd_metadata": metadata.to_dict(),
        "guidance": {"scale": 4.0},
        "topology": {"world_size": 1, "pure_data_parallel": True},
    }


def _write(tmp_path: pathlib.Path):
    model, config, metadata = _converted()
    output = write_pdd_export(
        tmp_path / "export",
        model.state_dict(),
        metadata=metadata,
        transformer_config={"_class_name": "SyntheticQwen", "in_channels": 4},
        identity=_identity(metadata),
        source_checkpoint={
            "name": "step_00000010",
            "manifest_sha256": "3" * 64,
            "completed_steps": 10,
        },
        max_shard_bytes=5_800,
    )
    return output, model, config, metadata


def _condition():
    return torch.tensor([[[0.2, -0.3], [0.1, 0.4]]], dtype=torch.bfloat16), torch.ones(
        1, 2, dtype=torch.long
    )


def _sample(model: nn.Module, config: PDDConfig, noise: torch.Tensor) -> torch.Tensor:
    pipeline = PDDPipeline(
        model,
        nn.Identity(),
        config,
        QwenImagePDDAdapter(config, compute_dtype=torch.bfloat16),
    )
    return pipeline.sample(noise.clone(), condition=_condition())


def test_bounded_safe_export_round_trip_and_seeded_schedules(tmp_path, monkeypatch) -> None:
    output, source, _source_config, metadata = _write(tmp_path)
    descriptor = inspect_pdd_export(output)

    shards = sorted(output.glob("*.safetensors"))
    assert len(shards) >= 2
    assert all(path.stat().st_size <= descriptor.manifest["max_shard_bytes"] for path in shards)
    assert descriptor.metadata == metadata

    restored = _TinyQwen()
    convert_qwen_image_to_pdd(restored, _config())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unsafe torch.load"))
    load_pdd_export_into_model(output, restored)
    for key, tensor in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)

    noise = torch.randn((1, 1, 4, 4), generator=torch.Generator().manual_seed(91))
    for schedule, blocks in PDD_INFERENCE_SCHEDULES.items():
        config = pdd_config_from_metadata(
            metadata,
            schedule=schedule,
            guidance_scale=4.0,
        )
        source.calls = 0
        restored.calls = 0
        expected = _sample(source, config, noise)
        actual = _sample(restored, config, noise)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert source.calls == restored.calls == len(blocks)
        torch.testing.assert_close(_sample(restored, config, noise), actual, rtol=0, atol=0)

    arbitrary = pdd_config_from_metadata(metadata, blocks=[1, 127], guidance_scale=4.0)
    assert arbitrary.inference_blocks == [1, 127]
    assert arbitrary.student_sample_steps == 2


def test_inference_config_preserves_authenticated_nondefault_grid_max_t() -> None:
    _model, _config_value, metadata = _converted()
    payload = metadata.to_dict()
    payload["grid_max_t"] = 1.0
    boundary_metadata = PDDMetadata.from_dict(payload)

    config = pdd_config_from_metadata(
        boundary_metadata,
        schedule="pdd-4",
        guidance_scale=4.0,
    )

    assert config.grid_max_t == 1.0
    assert (
        PDDPipeline(
            _TinyQwen(),
            nn.Identity(),
            config,
            QwenImagePDDAdapter(config),
        ).time_grid()[0]
        == 1.0
    )


def test_pinned_qwen_none_prompt_mask_is_normalized_for_pdd() -> None:
    """Diffusers 0.38 returns None when the single-prompt mask is all ones."""
    embeddings = torch.randn(1, 3, 5, dtype=torch.float32)
    resolved_embeddings, resolved_mask = _normalize_prompt_condition(
        embeddings,
        None,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert resolved_embeddings.dtype == torch.bfloat16
    assert resolved_mask.dtype == torch.long
    assert resolved_mask.shape == embeddings.shape[:2]
    assert torch.equal(resolved_mask, torch.ones_like(resolved_mask))


def test_qwen_projection_rejects_inconsistent_packed_width() -> None:
    _model, _config_value, metadata = _converted()
    base = _TinyQwen()
    assert _validate_qwen_projection(base, metadata) is base.proj_out
    base.config.in_channels = 8
    with pytest.raises(RuntimeError, match="proj_out width"):
        _validate_qwen_projection(base, metadata)


def test_export_is_complete_before_atomic_rename(tmp_path, monkeypatch) -> None:
    original_rename = pathlib.Path.rename
    observed = False

    def checked_rename(path, target):
        nonlocal observed
        if path.name.endswith(".staging"):
            inspect_pdd_export(path)
            assert (path / "COMPLETE").is_file()
            observed = True
        return original_rename(path, target)

    monkeypatch.setattr(pathlib.Path, "rename", checked_rename)
    _write(tmp_path)
    assert observed


def test_export_accepts_an_immutable_model_revision(tmp_path) -> None:
    model, _config_value, metadata = _converted()
    identity = _identity(metadata)
    output = write_pdd_export(
        tmp_path / "immutable-revision",
        model.state_dict(),
        metadata=metadata,
        transformer_config={"in_channels": 4},
        identity=identity,
        source_checkpoint={
            "name": "step_00000010",
            "manifest_sha256": "3" * 64,
            "completed_steps": 10,
        },
        max_shard_bytes=12_000,
    )
    assert inspect_pdd_export(output).manifest["identity"]["model"]["revision"] == "f" * 40


@pytest.mark.parametrize("execution", [None, "canonical_diffusers"])
def test_export_and_inference_reject_incompatible_qwen_execution(tmp_path, execution) -> None:
    model, _config_value, metadata = _converted()
    identity = _identity(metadata)
    if execution is None:
        identity.pop("qwen_image")
    else:
        identity["qwen_image"] = {"execution": execution}

    with pytest.raises(ValueError, match=r"Qwen execution identity|missing keys"):
        write_pdd_export(
            tmp_path / f"bad-execution-{execution}",
            model.state_dict(),
            metadata=metadata,
            transformer_config={"in_channels": 4},
            identity=identity,
            source_checkpoint={
                "name": "step_00000010",
                "manifest_sha256": "3" * 64,
                "completed_steps": 10,
            },
            max_shard_bytes=12_000,
        )

    descriptor = SimpleNamespace(manifest={"identity": identity})
    with pytest.raises(RuntimeError, match="Qwen execution identity"):
        _model_identity(descriptor)


@pytest.mark.parametrize("revision", [None, "main", "F" * 40])
def test_export_and_inference_reject_mutable_model_revisions(tmp_path, revision) -> None:
    model, _config_value, metadata = _converted()
    identity = _identity(metadata)
    identity["model"]["revision"] = revision
    with pytest.raises(ValueError, match="exact lowercase commit"):
        write_pdd_export(
            tmp_path / f"bad-revision-{str(revision)[:8]}",
            model.state_dict(),
            metadata=metadata,
            transformer_config={"in_channels": 4},
            identity=identity,
            source_checkpoint={
                "name": "step_00000010",
                "manifest_sha256": "3" * 64,
                "completed_steps": 10,
            },
            max_shard_bytes=12_000,
        )

    descriptor = SimpleNamespace(manifest={"identity": identity})
    with pytest.raises(RuntimeError, match="exact lowercase commit"):
        _model_identity(descriptor)


def test_export_rejects_nonfinite_and_existing_destination(tmp_path) -> None:
    output, model, _config_value, metadata = _write(tmp_path)
    with pytest.raises(FileExistsError):
        write_pdd_export(
            output,
            model.state_dict(),
            metadata=metadata,
            transformer_config={"in_channels": 4},
            identity=_identity(metadata),
            source_checkpoint={
                "name": "step_00000010",
                "manifest_sha256": "3" * 64,
                "completed_steps": 10,
            },
            max_shard_bytes=12_000,
        )

    bad = copy.deepcopy(model.state_dict())
    bad["backbone.weight"][0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        write_pdd_export(
            tmp_path / "bad",
            bad,
            metadata=metadata,
            transformer_config={"in_channels": 4},
            identity=_identity(metadata),
            source_checkpoint={
                "name": "step_00000010",
                "manifest_sha256": "3" * 64,
                "completed_steps": 10,
            },
            max_shard_bytes=12_000,
        )


@pytest.mark.parametrize("corruption", ["complete", "shard", "extra", "symlink"])
def test_export_authentication_rejects_corruption(tmp_path, corruption) -> None:
    output, _model, _config_value, _metadata = _write(tmp_path)
    if corruption == "complete":
        (output / "COMPLETE").unlink()
    elif corruption == "shard":
        shard = next(output.glob("*.safetensors"))
        with shard.open("ab") as stream:
            stream.write(b"corrupt")
    elif corruption == "extra":
        (output / "undeclared.bin").write_bytes(b"extra")
    else:
        (output / "linked").symlink_to(output / "config.json")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        inspect_pdd_export(output)


def test_safe_load_rejects_wrong_model_inventory(tmp_path) -> None:
    output, _model, _config_value, _metadata = _write(tmp_path)
    unconverted = _TinyQwen()
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_pdd_export_into_model(output, unconverted)

    wrong_dtype = _TinyQwen().to(torch.float64)
    convert_qwen_image_to_pdd(wrong_dtype, _config())
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        load_pdd_export_into_model(output, wrong_dtype)
