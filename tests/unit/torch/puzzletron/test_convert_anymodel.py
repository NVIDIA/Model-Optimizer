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

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from safetensors import safe_open

pytest.importorskip("transformers")

from _test_utils.torch.transformers_models import (
    create_tiny_qwen3_5_dir,
    create_tiny_qwen3_5vl_dir,
    create_tiny_qwen3_dir,
)
from transformers import AutoModelForCausalLM

import modelopt.torch.puzzletron as mtpz
import modelopt.torch.puzzletron.stages.convert as convert_stage_module
from modelopt.torch.puzzletron.stages.convert import (
    _descriptor_checkpoint_layout_complete,
    _is_complete_checkpoint,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config


def _weight_map(checkpoint_dir):
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text())["weight_map"]
    weights_path = checkpoint_dir / "model.safetensors"
    with safe_open(weights_path, framework="pt") as handle:
        return dict.fromkeys(handle.keys(), weights_path.name)


@pytest.mark.parametrize("revision", ["pinned-sha", None], ids=["pinned", "default"])
def test_convert_stage_pins_optional_hugging_face_revision(tmp_path, monkeypatch, revision):
    calls = []

    class SourceResolvedError(RuntimeError):
        pass

    def snapshot_download(*, repo_id, revision):
        calls.append({"repo_id": repo_id, "revision": revision})
        raise SourceResolvedError

    monkeypatch.setattr(convert_stage_module, "_register_automodel_config_aliases", lambda: None)
    monkeypatch.setattr(convert_stage_module, "_distributed_if_needed", nullcontext)
    monkeypatch.setattr(
        convert_stage_module, "_is_complete_checkpoint", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(convert_stage_module.dist, "is_master", lambda: True)
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


def test_conversion_resume_rejects_unmigrated_descriptor_checkpoint_layout(tmp_path):
    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model.safetensors",
                    "language_model.lm_head.weight": "model.safetensors",
                }
            }
        )
    )

    class Descriptor:
        @staticmethod
        def generic_decoder_contract(config):
            return SimpleNamespace(
                checkpoint_key_rewrites=(
                    (r"^language_model\.model\.", "model.language_model."),
                    (r"^language_model\.lm_head\.", "lm_head."),
                )
            )

    assert not _descriptor_checkpoint_layout_complete(
        checkpoint, Descriptor, SimpleNamespace()
    )


def test_conversion_resume_skips_generic_layout_check_when_descriptor_has_no_contract(tmp_path):
    """Specialized family converters must not be routed through the generic converter."""

    class Descriptor:
        @staticmethod
        def generic_decoder_contract(config):
            return None

    assert _descriptor_checkpoint_layout_complete(
        tmp_path, Descriptor, SimpleNamespace(block_configs=[])
    )


def test_convert_anymodel_qwen3_5_text_preserves_mtp(tmp_path):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    input_dir = create_tiny_qwen3_5_dir(tmp_path, with_tokenizer=True, with_mtp=True)
    output_dir = tmp_path / "qwen3_5-anymodel"
    mtpz.anymodel.convert_model(input_dir, output_dir, converter="qwen3_5_text")

    weight_map = _weight_map(output_dir)
    assert "mtp.0.norm.weight" in weight_map

    descriptor = mtpz.anymodel.ModelDescriptorFactory.get("qwen3_5_text")
    with mtpz.anymodel.deci_x_patcher(descriptor):
        _ = AutoModelForCausalLM.from_pretrained(output_dir)


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
