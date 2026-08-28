# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for Puzzletron AnyModel checkpoint conversion."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from safetensors import safe_open

pytest.importorskip("transformers")

from _test_utils.torch.transformers_models import create_tiny_qwen3_5vl_dir, create_tiny_qwen3_dir
from transformers import AutoModelForCausalLM

import modelopt.torch.puzzletron as mtpz
import modelopt.torch.puzzletron.stages.convert as convert_stage_module
from modelopt.torch.puzzletron.stages.convert import (
    _conversion_source_matches,
    _is_complete_checkpoint,
    _write_conversion_source_metadata,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config


def _weight_map(checkpoint_dir):
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text())["weight_map"]
    weights_path = checkpoint_dir / "model.safetensors"
    with safe_open(weights_path, framework="pt") as handle:
        return dict.fromkeys(handle.keys(), weights_path.name)


def _patch_single_rank_convert(monkeypatch):
    monkeypatch.setattr(convert_stage_module, "_register_automodel_config_aliases", lambda: None)
    monkeypatch.setattr(convert_stage_module, "_distributed_if_needed", nullcontext)


def test_nonmaster_raises_broadcast_conversion_failure(monkeypatch):
    monkeypatch.setattr(
        convert_stage_module.dist,
        "broadcast",
        lambda value, src: "ConversionError: failed conversion",
    )

    with pytest.raises(RuntimeError, match="rank 0 failed during teacher conversion"):
        convert_stage_module._raise_on_master_failure(None, action="teacher conversion")


def test_master_broadcasts_failure_before_reraising(monkeypatch):
    events = []

    class ConversionError(RuntimeError):
        pass

    error = ConversionError("failed conversion")

    def broadcast(value, src):
        events.append(("broadcast", value))
        return value

    monkeypatch.setattr(convert_stage_module.dist, "broadcast", broadcast)

    with pytest.raises(ConversionError) as raised:
        convert_stage_module._raise_on_master_failure(error, action="teacher conversion")
    events.append(("raised", str(raised.value)))

    assert raised.value is error
    assert events == [
        ("broadcast", "ConversionError: failed conversion"),
        ("raised", "failed conversion"),
    ]


@pytest.mark.parametrize("revision", ["pinned-sha", None], ids=["pinned", "default"])
def test_convert_stage_pins_optional_hugging_face_revision(tmp_path, monkeypatch, revision):
    calls = []

    class SourceResolvedError(RuntimeError):
        pass

    def snapshot_download(*, repo_id, revision):
        calls.append({"repo_id": repo_id, "revision": revision})
        raise SourceResolvedError

    _patch_single_rank_convert(monkeypatch)
    monkeypatch.setattr(
        convert_stage_module, "_is_complete_checkpoint", lambda *args, **kwargs: False
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    model = {"source": "Qwen/Qwen3.5-0.8B"}
    if revision is not None:
        model["revision"] = revision

    with pytest.raises(SourceResolvedError):
        convert_stage_module.convert_stage(
            {"model": model, "convert": {"teacher_dir": str(tmp_path / "teacher")}},
            manifest=object(),
        )

    assert calls == [{"repo_id": "Qwen/Qwen3.5-0.8B", "revision": revision}]


def test_conversion_resume_requires_matching_source_revision(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()

    assert _conversion_source_matches(teacher_dir, source="Qwen/Qwen3.5-0.8B", revision=None)
    assert not _conversion_source_matches(
        teacher_dir, source="Qwen/Qwen3.5-0.8B", revision="new-sha"
    )

    _write_conversion_source_metadata(
        teacher_dir,
        source="Qwen/Qwen3.5-0.8B",
        revision="old-sha",
        source_identity="source_model_abc",
    )
    assert _conversion_source_matches(teacher_dir, source="Qwen/Qwen3.5-0.8B", revision="old-sha")
    assert not _conversion_source_matches(
        teacher_dir, source="Qwen/Qwen3.5-0.8B", revision="new-sha"
    )
    assert not _conversion_source_matches(
        teacher_dir, source="Qwen/Another-Model", revision="old-sha"
    )

    metadata_path = teacher_dir / convert_stage_module._CONVERSION_SOURCE_METADATA
    metadata_path.write_text("not-json")
    assert not _conversion_source_matches(
        teacher_dir, source="Qwen/Qwen3.5-0.8B", revision="old-sha"
    )
    metadata_path.write_text("[]")
    assert not _conversion_source_matches(
        teacher_dir, source="Qwen/Qwen3.5-0.8B", revision="old-sha"
    )


def test_convert_stage_reconverts_teacher_from_different_revision(tmp_path, monkeypatch):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "old-shard.bin").write_text("old")
    _write_conversion_source_metadata(
        teacher_dir,
        source="Qwen/Qwen3.5-0.8B",
        revision="old-sha",
        source_identity="source_model_abc",
    )
    metadata_path = teacher_dir / convert_stage_module._CONVERSION_SOURCE_METADATA
    metadata_before = metadata_path.read_bytes()
    resolved = []
    events = []

    class SourceResolvedError(RuntimeError):
        pass

    def resolve_source(source, *, revision):
        resolved.append({"source": source, "revision": revision})
        raise SourceResolvedError("source failed")

    def broadcast(value, src):
        events.append(("broadcast", value))
        return value

    _patch_single_rank_convert(monkeypatch)
    monkeypatch.setattr(convert_stage_module.dist, "broadcast", broadcast)
    monkeypatch.setattr(
        convert_stage_module.dist, "barrier", lambda: events.append(("barrier", None))
    )
    monkeypatch.setattr(
        convert_stage_module, "_is_complete_checkpoint", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(convert_stage_module, "_resolve_source_path", resolve_source)

    with pytest.raises(SourceResolvedError):
        convert_stage_module.convert_stage(
            {
                "model": {"source": "Qwen/Qwen3.5-0.8B", "revision": "new-sha"},
                "convert": {"teacher_dir": str(teacher_dir)},
            },
            manifest=object(),
        )

    assert resolved == [{"source": "Qwen/Qwen3.5-0.8B", "revision": "new-sha"}]
    assert events == [
        ("broadcast", None),
        ("barrier", None),
        ("broadcast", "SourceResolvedError: source failed"),
    ]
    assert metadata_path.read_bytes() == metadata_before
    assert (teacher_dir / "old-shard.bin").read_text() == "old"
    assert not convert_stage_module._conversion_sibling(
        teacher_dir, convert_stage_module._CONVERSION_TRANSACTION_SUFFIX
    ).exists()
    assert not convert_stage_module._conversion_sibling(
        teacher_dir, convert_stage_module._CONVERSION_BACKUP_SUFFIX
    ).exists()


def test_failed_reconversion_preserves_teacher_and_retries_cleanly(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "old-shard.bin").write_text("old")
    _write_conversion_source_metadata(
        teacher_dir,
        source="Qwen/Qwen3.5-0.8B",
        revision="old-sha",
        source_identity="source_model_old",
    )
    source_config = SimpleNamespace(architectures=["Qwen3ForCausalLM"])

    class ConversionError(RuntimeError):
        pass

    class Resolution:
        name = "qwen3"
        descriptor = object()

        @staticmethod
        def to_dict():
            return {"name": "qwen3"}

    class RetryConverter:
        attempts = 0

        def convert(self, *, output_dir, **kwargs):
            self.attempts += 1
            (output_dir / "new-shard.bin").write_text("new")
            if self.attempts == 1:
                raise ConversionError

    converter = RetryConverter()
    barriers = []

    _patch_single_rank_convert(monkeypatch)
    monkeypatch.setattr(convert_stage_module.dist, "barrier", lambda: barriers.append(None))
    monkeypatch.setattr(
        convert_stage_module,
        "_is_complete_checkpoint",
        lambda path, **kwargs: path == teacher_dir
        or (
            (path / "new-shard.bin").is_file()
            and (path / convert_stage_module._CONVERSION_SOURCE_METADATA).is_file()
        ),
    )
    monkeypatch.setattr(
        convert_stage_module, "_resolve_source_path", lambda *args, **kwargs: source_dir
    )
    monkeypatch.setattr(
        convert_stage_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: source_config,
    )
    monkeypatch.setattr(
        convert_stage_module,
        "model_identity",
        lambda config: SimpleNamespace(value="source_model_new"),
    )
    monkeypatch.setattr(
        convert_stage_module,
        "resolve_descriptor_from_pretrained",
        lambda *args, **kwargs: Resolution(),
    )
    monkeypatch.setattr(
        convert_stage_module, "_descriptor_checkpoint_layout_complete", lambda *args: True
    )
    monkeypatch.setattr(
        convert_stage_module,
        "complete_stage",
        lambda config, manifest, *, outputs, status="success", message=None: {
            "outputs": outputs,
            "status": status,
        },
    )
    monkeypatch.setattr(convert_stage_module.ConverterFactory, "get", lambda name: converter)

    config = {
        "model": {"source": "Qwen/Qwen3.5-0.8B", "revision": "new-sha"},
        "convert": {"teacher_dir": str(teacher_dir)},
    }

    with pytest.raises(ConversionError):
        convert_stage_module.convert_stage(config, manifest=object())

    assert len(barriers) == 1

    old_metadata = json.loads(
        (teacher_dir / convert_stage_module._CONVERSION_SOURCE_METADATA).read_text()
    )
    assert old_metadata["revision"] == "old-sha"
    assert (teacher_dir / "old-shard.bin").read_text() == "old"
    transaction_dir = convert_stage_module._conversion_sibling(
        teacher_dir, convert_stage_module._CONVERSION_TRANSACTION_SUFFIX
    )
    assert (transaction_dir / "new-shard.bin").read_text() == "new"

    result = convert_stage_module.convert_stage(config, manifest=object())

    assert result["status"] == "success"
    assert len(barriers) == 3
    assert converter.attempts == 2
    assert not (teacher_dir / "old-shard.bin").exists()
    assert (teacher_dir / "new-shard.bin").read_text() == "new"
    new_metadata = json.loads(
        (teacher_dir / convert_stage_module._CONVERSION_SOURCE_METADATA).read_text()
    )
    assert new_metadata["revision"] == "new-sha"
    assert not transaction_dir.exists()


def test_conversion_transaction_recovers_interrupted_swap(tmp_path, monkeypatch):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "old-shard.bin").write_text("old")
    transaction_dir = convert_stage_module._conversion_sibling(
        teacher_dir, convert_stage_module._CONVERSION_TRANSACTION_SUFFIX
    )
    transaction_dir.mkdir()
    (transaction_dir / "new-shard.bin").write_text("new")
    backup_dir = convert_stage_module._conversion_sibling(
        teacher_dir, convert_stage_module._CONVERSION_BACKUP_SUFFIX
    )
    teacher_dir.replace(backup_dir)

    convert_stage_module._recover_conversion_transaction(teacher_dir)

    assert (teacher_dir / "old-shard.bin").read_text() == "old"
    assert not transaction_dir.exists()
    assert not backup_dir.exists()

    teacher_dir.replace(backup_dir)
    teacher_dir.mkdir()
    (teacher_dir / "published-shard.bin").write_text("published")

    convert_stage_module._recover_conversion_transaction(teacher_dir)

    assert (teacher_dir / "published-shard.bin").read_text() == "published"
    assert not backup_dir.exists()

    backup_dir.mkdir()
    (backup_dir / "backup-shard.bin").write_text("backup")
    transaction_dir.mkdir()
    (transaction_dir / "transaction-shard.bin").write_text("transaction")
    events = []

    _patch_single_rank_convert(monkeypatch)
    monkeypatch.setattr(
        convert_stage_module.dist,
        "broadcast",
        lambda value, src: events.append(("broadcast", value)) or value,
    )
    monkeypatch.setattr(
        convert_stage_module.dist, "barrier", lambda: events.append(("barrier", None))
    )

    with pytest.raises(RuntimeError, match="ambiguous conversion transaction state"):
        convert_stage_module.convert_stage(
            {
                "model": {"source": "Qwen/Qwen3.5-0.8B"},
                "convert": {"teacher_dir": str(teacher_dir)},
            },
            manifest=object(),
        )

    assert events == [
        (
            "broadcast",
            f"RuntimeError: ambiguous conversion transaction state for checkpoint: {teacher_dir}",
        )
    ]
    assert (teacher_dir / "published-shard.bin").read_text() == "published"
    assert (backup_dir / "backup-shard.bin").read_text() == "backup"
    assert (transaction_dir / "transaction-shard.bin").read_text() == "transaction"


def test_convert_anymodel(tmp_path):
    input_dir = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    output_dir = tmp_path / "qwen3-0.6b-anymodel"
    mtpz.anymodel.convert_model(input_dir, output_dir, converter="qwen3")

    descriptor = mtpz.anymodel.ModelDescriptorFactory.get("qwen3")
    with mtpz.anymodel.deci_x_patcher(descriptor):
        _ = AutoModelForCausalLM.from_pretrained(output_dir)


def test_conversion_resume_rejects_partial_hf_checkpoint_without_block_configs(tmp_path):
    partial_dir = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)

    assert not _is_complete_checkpoint(partial_dir, trust_remote_code=False)


def test_convert_anymodel_qwen3_5_vl_sets_text_layer_config(tmp_path):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    input_dir = create_tiny_qwen3_5vl_dir(tmp_path, with_tokenizer=True)
    output_dir = tmp_path / "qwen3_5vl-anymodel"
    mtpz.anymodel.convert_model(input_dir, output_dir, converter="qwen3_5")

    config = load_model_config(output_dir)
    assert len(config.block_configs) == config.text_config.num_hidden_layers
    assert config.text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]

    weight_map = _weight_map(output_dir)
    assert any("visual" in key or "vision" in key for key in weight_map)
