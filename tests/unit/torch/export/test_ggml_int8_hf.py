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
from modelopt.torch.export import export_hf_ggml_int8_checkpoint
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

IQ1_S = {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"}
IQ2_XS = {"num_bits": "iq2_xs", "block_sizes": {-1: 256}, "backend": "ggml"}
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")


def _quantized_tiny_llama(tmp_path, layer_cfgs):
    source = create_tiny_llama_dir(tmp_path, hidden_size=256, intermediate_size=256)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
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
    token_ids = torch.randint(0, 32, (1, 8))
    mtq.quantize(model, quant_cfg, lambda m: m(token_ids))
    # A fake-quant forward caches each payload, so the test and the export encode it once.
    model(token_ids)
    return model


def test_ggml_layers_export_as_their_int8_levels_and_steps(tmp_path):
    # A format with 32-weight sub-blocks and one with 16 need one config group each.
    model = _quantized_tiny_llama(tmp_path, [IQ1_S, IQ2_XS])
    layers = {i: [f"model.layers.{i}.mlp.{proj}" for proj in PROJECTIONS] for i in (0, 1)}
    expected = {}
    for layer in layers[0] + layers[1]:
        module = model.get_submodule(layer)
        ggml_format = GGML_FORMAT_REGISTRY[module.weight_quantizer.num_bits]
        payload = ggml_format.pack(module.weight, module.weight_quantizer)
        expected[layer] = ggml_format.decode_int8(payload, torch.tensor(module.weight.shape))
    unquantized = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if "quantizer" not in k and k.removesuffix(".weight") not in expected
    }

    export_hf_ggml_int8_checkpoint(model, tmp_path / "export")

    config = json.loads((tmp_path / "export" / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "compressed-tensors"
    assert {
        group["weights"]["group_size"]: group["targets"]
        for group in config["config_groups"].values()
    } == {32: layers[0], 16: layers[1]}
    saved = load_file(tmp_path / "export" / "model.safetensors")
    for layer, (levels, steps) in expected.items():
        # compressed-tensors' layout: offset by 128, four values per int32, lowest byte first.
        packed = saved.pop(f"{layer}.weight_packed")
        unpacked = torch.stack([(packed >> (8 * k)) & 0xFF for k in range(4)], -1).flatten(1)
        assert torch.equal(unpacked - 128, levels.long())
        assert torch.equal(saved.pop(f"{layer}.weight_scale"), steps.to(torch.bfloat16))
        assert saved.pop(f"{layer}.weight_shape").tolist() == list(levels.shape)
    assert saved.keys() <= unquantized.keys()
    for name, tensor in saved.items():
        assert torch.equal(tensor, unquantized[name])


def test_a_non_ggml_quantizer_is_refused_before_writing(tmp_path):
    model = _quantized_tiny_llama(tmp_path, [IQ1_S, {"num_bits": (4, 3), "axis": None}])

    with pytest.raises(ValueError, match="only GGML weight quantizers"):
        export_hf_ggml_int8_checkpoint(model, tmp_path / "export")

    assert not (tmp_path / "export").exists()
