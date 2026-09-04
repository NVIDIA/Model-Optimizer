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

"""Tests for VLM checkpoint and prompt-template validation."""

import json

import pytest

from examples.puzzletron.evaluation.vlm import model as vlm_model
from tests.unit.torch.puzzletron.evaluation.vlm._test_utils import _write_checkpoint


def _homogeneous_qwen_block_configs() -> list[dict[str, object]]:
    block = {
        "subblock_configs": [
            {
                "kind": "attention",
                "name": "attention",
                "no_op": False,
                "num_kv_heads": 2,
                "num_query_heads": 8,
            },
            {
                "kind": "ffn",
                "name": "ffn",
                "no_op": False,
                "intermediate_size": 3584,
            },
        ]
    }
    return [json.loads(json.dumps(block)) for _ in range(24)]


def test_no_think_template_is_local_and_requires_checkpoint_switch(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    (checkpoint_path / "chat_template.jinja").write_text(
        "{% if enable_thinking is defined and enable_thinking is true %}"
        "<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}"
    )
    generated = vlm_model.no_think_chat_template(checkpoint_path, tasks_root)
    assert generated.parent == tasks_root
    (checkpoint_path / "chat_template.jinja").write_text("unsupported\n")
    with pytest.raises(ValueError, match="cannot disable thinking"):
        vlm_model.no_think_chat_template(checkpoint_path, tasks_root)


def test_no_think_template_rejects_unsafe_checkpoint_expression(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    (checkpoint_path / "chat_template.jinja").write_text(
        "{{ ''.__class__.__mro__ }}"
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "<think>\n\n</think>\n\n{% endif %}"
    )
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    with pytest.raises(ValueError, match="chat template is invalid"):
        vlm_model.no_think_chat_template(checkpoint_path, tasks_root)


def test_checkpoint_contract_accepts_only_matching_realized_anymodel(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    config_path = checkpoint_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        architectures=["AnyModel"],
        base_architecture="Qwen3_5ForConditionalGeneration",
        block_configs=_homogeneous_qwen_block_configs(),
    )
    config_path.write_text(json.dumps(config) + "\n")

    vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config.pop("block_configs")
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="cannot prove.*homogeneous"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config["block_configs"] = _homogeneous_qwen_block_configs()
    config["base_architecture"] = "OtherForConditionalGeneration"
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="AnyModel base_architecture"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config.update(
        architectures=["AnyModel", "Qwen3_5ForConditionalGeneration"],
        base_architecture="Qwen3_5ForConditionalGeneration",
    )
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="AnyModel base_architecture"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")


def test_checkpoint_contract_routes_heterogeneous_anymodel_to_vllm(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    config_path = checkpoint_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        architectures=["AnyModel"],
        base_architecture="Qwen3_5ForConditionalGeneration",
        block_configs=_homogeneous_qwen_block_configs(),
    )
    config["block_configs"][19]["subblock_configs"][0]["num_query_heads"] = 6
    config["text_config"]["per_layer_config"] = {
        "19": {"num_attention_heads": 6, "num_key_value_heads": 2}
    }
    config_path.write_text(json.dumps(config) + "\n")

    with pytest.raises(ValueError, match="native qwen3_5 backend cannot load"):
        vlm_model.verify_checkpoint(
            checkpoint_path,
            profile="VLM benchmark",
            model_backend="qwen3_5",
        )

    vlm_model.verify_checkpoint(
        checkpoint_path,
        profile="VLM benchmark",
        model_backend="vllm",
    )


def test_checkpoint_contract_accepts_other_positive_qwen35_geometry(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    config_path = checkpoint_path / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"].update(
        hidden_size=2560,
        intermediate_size=9728,
        num_attention_heads=20,
        num_hidden_layers=40,
        num_key_value_heads=4,
    )
    config_path.write_text(json.dumps(config) + "\n")

    vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config["text_config"]["hidden_size"] = 0
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="invalid Qwen 3.5 geometry"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")


@pytest.mark.parametrize("processor_content", [None, "", "[]\n", "{\n", b"\xff"])
def test_checkpoint_contract_requires_valid_local_processor_assets(tmp_path, processor_content):
    checkpoint_path = _write_checkpoint(tmp_path)
    processor_path = checkpoint_path / "preprocessor_config.json"
    if processor_content is None:
        processor_path.unlink()
    elif isinstance(processor_content, bytes):
        processor_path.write_bytes(processor_content)
    else:
        processor_path.write_text(processor_content)

    with pytest.raises(ValueError, match="processor asset"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")


def test_checkpoint_contract_rejects_malformed_companion_processor_asset(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    (checkpoint_path / "video_preprocessor_config.json").write_text("{\n")

    with pytest.raises(ValueError, match="video_preprocessor_config.json"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")
