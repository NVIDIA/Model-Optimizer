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

import json

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import convert_ggml_export_to_int8, export_hf_checkpoint
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

# Layer 0 takes a format with 32-weight sub-blocks and layer 1 one with 16, so the export needs
# one compressed-tensors config group per group size.
LAYER_FORMATS = {0: "iq1_s", 1: "iq2_xs"}


def _export(tmp_path, second_layer_cfg):
    source = create_tiny_llama_dir(tmp_path, hidden_size=256, intermediate_size=512)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    layer_cfgs = [
        {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"},
        second_layer_cfg,
    ]
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            *(
                {"quantizer_name": f"*layers.{i}.mlp*weight_quantizer", "cfg": cfg}
                for i, cfg in enumerate(layer_cfgs)
            ),
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, quant_cfg, lambda m: m(torch.randint(0, 32, (1, 8))))
    export_dir = tmp_path / "export"
    export_hf_checkpoint(model, export_dir=export_dir)
    return export_dir


def test_ggml_export_converts_to_int8_levels_and_steps(tmp_path):
    export_dir = _export(
        tmp_path, {"num_bits": "iq2_xs", "block_sizes": {-1: 256}, "backend": "ggml"}
    )
    before = load_file(export_dir / "model.safetensors")

    convert_ggml_export_to_int8(export_dir)

    config = json.loads((export_dir / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "compressed-tensors"
    layers = {
        i: [f"model.layers.{i}.mlp.{proj}" for proj in ("down_proj", "gate_proj", "up_proj")]
        for i in LAYER_FORMATS
    }
    assert {
        group["weights"]["group_size"]: group["targets"]
        for group in config["config_groups"].values()
    } == {32: layers[0], 16: layers[1]}
    assert not (export_dir / "hf_quant_config.json").exists()

    after = load_file(export_dir / "model.safetensors")
    for i, name in LAYER_FORMATS.items():
        for layer in layers[i]:
            payload = before.pop(f"{layer}.weight")
            shape = after[f"{layer}.weight_shape"]
            levels, steps = GGML_FORMAT_REGISTRY[name].decode_int8(payload, shape)
            # compressed-tensors' layout: offset by 128, four values per int32, lowest byte first.
            packed = after[f"{layer}.weight_packed"]
            unpacked = torch.stack([(packed >> (8 * k)) & 0xFF for k in range(4)], -1).flatten(1)
            assert torch.equal(unpacked - 128, levels.long())
            assert torch.equal(after[f"{layer}.weight_scale"], steps.to(torch.bfloat16))
            assert shape.tolist() == list(levels.shape)
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor)


def test_export_with_a_non_ggml_layer_is_refused_unchanged(tmp_path):
    export_dir = _export(tmp_path, {"num_bits": (4, 3), "axis": None})
    files = {path.name: path.read_bytes() for path in export_dir.iterdir()}

    with pytest.raises(ValueError, match="Only GGML layers"):
        convert_ggml_export_to_int8(export_dir)

    assert {path.name: path.read_bytes() for path in export_dir.iterdir()} == files
