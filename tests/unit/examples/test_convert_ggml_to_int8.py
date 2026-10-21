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

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

_SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq" / "convert_ggml_to_int8.py"
_SPEC = importlib.util.spec_from_file_location("convert_ggml_to_int8", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
convert_ggml_to_int8 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(convert_ggml_to_int8)

# Layer 0 takes a format with 32-weight sub-blocks and layer 1 one with 16, so the export needs
# one compressed-tensors config group per group size.
LAYER_FORMATS = {0: "iq1_s", 1: "iq2_xs"}
IQ2_XS = {"num_bits": "iq2_xs", "block_sizes": {-1: 256}, "backend": "ggml"}


def _export(tmp_path, second_layer_cfg):
    source = create_tiny_llama_dir(tmp_path, hidden_size=256, intermediate_size=256)
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


def _files(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def ggml_export(tmp_path_factory):
    export_dir = _export(tmp_path_factory.mktemp("ggml"), IQ2_XS)
    # Files a checkpoint can ship beside its indexed weights: a nested sidecar and an off-index
    # weight file such as an MTP head.
    (export_dir / "chat_templates").mkdir()
    (export_dir / "chat_templates" / "tool_use.jinja").write_text("{{ messages }}")
    save_file({"mtp.fc.weight": torch.ones(2, 2)}, export_dir / "mtp.safetensors")
    return export_dir


def test_ggml_export_converts_to_int8_levels_and_steps(ggml_export, tmp_path):
    ggml_files = _files(ggml_export)
    before = load_file(ggml_export / "model.safetensors")
    int8_dir = tmp_path / "int8"

    # A small shard size splits the output, so the index has to stitch the shards together.
    convert_ggml_to_int8.convert_ggml_export_to_int8(ggml_export, int8_dir, max_shard_size=2**18)

    assert _files(ggml_export) == ggml_files
    assert not (int8_dir / "hf_quant_config.json").exists()
    for sidecar in (Path("chat_templates/tool_use.jinja"), Path("mtp.safetensors")):
        assert (int8_dir / sidecar).read_bytes() == ggml_files[sidecar]
    config = json.loads((int8_dir / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "compressed-tensors"
    layers = {
        i: [f"model.layers.{i}.mlp.{proj}" for proj in ("down_proj", "gate_proj", "up_proj")]
        for i in LAYER_FORMATS
    }
    assert {
        group["weights"]["group_size"]: group["targets"]
        for group in config["config_groups"].values()
    } == {32: layers[0], 16: layers[1]}

    weight_map = json.loads((int8_dir / "model.safetensors.index.json").read_text())["weight_map"]
    assert len(set(weight_map.values())) > 1
    after = {}
    for shard in set(weight_map.values()):
        after.update(load_file(int8_dir / shard))
    assert after.keys() == weight_map.keys()
    for i, name in LAYER_FORMATS.items():
        for layer in layers[i]:
            payload = before.pop(f"{layer}.weight")
            shape = after.pop(f"{layer}.weight_shape")
            levels, steps = GGML_FORMAT_REGISTRY[name].decode_int8(payload, shape)
            # compressed-tensors' layout: offset by 128, four values per int32, lowest byte first.
            packed = after.pop(f"{layer}.weight_packed")
            unpacked = torch.stack([(packed >> (8 * k)) & 0xFF for k in range(4)], -1).flatten(1)
            assert torch.equal(unpacked - 128, levels.long())
            assert torch.equal(after.pop(f"{layer}.weight_scale"), steps.to(torch.bfloat16))
            assert shape.tolist() == list(levels.shape)
    assert after.keys() == before.keys()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor)


def test_a_failed_conversion_leaves_the_output_directory_empty(ggml_export, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    # The last step: by then the shards, index and config are all written.
    monkeypatch.setattr(convert_ggml_to_int8, "copy_non_model_files", fail)

    with pytest.raises(OSError, match="disk full"):
        convert_ggml_to_int8.convert_ggml_export_to_int8(ggml_export, tmp_path / "int8")

    assert not any((tmp_path / "int8").iterdir())


def test_export_with_a_non_ggml_layer_is_refused_before_writing(tmp_path):
    export_dir = _export(tmp_path, {"num_bits": (4, 3), "axis": None})

    with pytest.raises(ValueError, match="Only GGML layers"):
        convert_ggml_to_int8.convert_ggml_export_to_int8(export_dir, tmp_path / "int8")

    assert not (tmp_path / "int8").exists()
