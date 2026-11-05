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
import shutil
from pathlib import Path

import pytest
import torch
from _test_utils.examples.ggml_export import FP8_ATTENTION, IQ1_S, quantize_tiny_llama
from _test_utils.examples.hf_ptq_example_utils import convert_ggml_to_int8
from _test_utils.examples.run_command import run_example_command
from safetensors.torch import load_file, save_file

from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

# quantize_tiny_llama's layer 0 is IQ1_S (group 32) and layer 1 IQ2_XS (group 16).
LAYER_FORMATS = {0: "iq1_s", 1: "iq2_xs"}
FP8_LAYERS = [
    f"model.layers.0.self_attn.{proj}" for proj in ("k_proj", "o_proj", "q_proj", "v_proj")
]


def _convert(ggml_dir, int8_dir, *args):
    run_example_command(
        ["python", "convert_ggml_to_int8.py", str(ggml_dir), str(int8_dir), *args], "hf_ptq"
    )


def _files(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def ggml_export(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ggml")
    model, _ = quantize_tiny_llama(tmp_path, extra_quant_cfg=FP8_ATTENTION)
    export_dir = tmp_path / "export"
    export_hf_checkpoint(model, export_dir=export_dir)
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

    # Shards of about 200 KB split the output, so the index has to stitch them together.
    _convert(ggml_export, int8_dir, "--max_shard_size_gb", "0.0002")

    assert _files(ggml_export) == ggml_files
    for sidecar in (Path("chat_templates/tool_use.jinja"), Path("mtp.safetensors")):
        assert (int8_dir / sidecar).read_bytes() == ggml_files[sidecar]
    layers = {
        i: [f"model.layers.{i}.mlp.{proj}" for proj in ("down_proj", "gate_proj", "up_proj")]
        for i in LAYER_FORMATS
    }
    # Only the GGML layers' entries change; the FP8 attention's are kept as they are.
    original = json.loads(ggml_files[Path("hf_quant_config.json")])["quantization"]
    int8 = {
        layer: {
            "quant_algo": "INT8_PB_WO",
            "group_size": GGML_FORMAT_REGISTRY[name].int8_group_size,
        }
        for i, name in LAYER_FORMATS.items()
        for layer in layers[i]
    }
    expected = {**original, "quantized_layers": {**original["quantized_layers"], **int8}}
    assert {expected["quantized_layers"][layer]["quant_algo"] for layer in FP8_LAYERS} == {"FP8"}
    assert json.loads((int8_dir / "hf_quant_config.json").read_text())["quantization"] == expected
    config = json.loads((int8_dir / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "modelopt"
    assert config["quantized_layers"] == expected["quantized_layers"]

    weight_map = json.loads((int8_dir / "model.safetensors.index.json").read_text())["weight_map"]
    assert len(set(weight_map.values())) > 1
    after = {}
    for shard in set(weight_map.values()):
        after.update(load_file(int8_dir / shard))
    assert after.keys() == weight_map.keys()
    for i, name in LAYER_FORMATS.items():
        for layer in layers[i]:
            payload = before.pop(f"{layer}.weight")
            weight = after.pop(f"{layer}.weight")
            levels, steps = GGML_FORMAT_REGISTRY[name].decode_int8(
                payload, torch.tensor(weight.shape)
            )
            assert torch.equal(weight, levels)
            assert torch.equal(after.pop(f"{layer}.weight_scale"), steps.to(torch.bfloat16))
    assert after.keys() == before.keys()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor)


def test_dtype_is_read_from_text_config(ggml_export, tmp_path):
    # Multimodal configs keep the dtype in text_config only.
    ggml_dir = tmp_path / "ggml"
    shutil.copytree(ggml_export, ggml_dir)
    config = json.loads((ggml_dir / "config.json").read_text())
    dtype = config.pop("dtype", None) or config.pop("torch_dtype")
    config.pop("torch_dtype", None)
    config["text_config"] = {"dtype": dtype}
    (ggml_dir / "config.json").write_text(json.dumps(config))

    int8_dir = tmp_path / "int8"
    convert_ggml_to_int8.convert_ggml_export_to_int8(ggml_dir, int8_dir)

    name = "model.layers.0.mlp.up_proj.weight_scale"
    shard = json.loads((int8_dir / "model.safetensors.index.json").read_text())["weight_map"][name]
    assert load_file(int8_dir / shard)[name].dtype == torch.bfloat16


def test_a_failed_conversion_leaves_the_output_directory_empty(ggml_export, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    # Called in-process to fail the last step: by then the shards, index and config are written.
    monkeypatch.setattr(convert_ggml_to_int8, "copy_non_model_files", fail)

    with pytest.raises(OSError, match="disk full"):
        convert_ggml_to_int8.convert_ggml_export_to_int8(ggml_export, tmp_path / "int8")

    assert not any((tmp_path / "int8").iterdir())


def test_nvfp4_layers_are_carried_over_as_they_are(tmp_path):
    # NVFP4 weights are uint8 like GGML payloads; only the config tells the two apart.
    nvfp4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    model, _ = quantize_tiny_llama(tmp_path, (IQ1_S, nvfp4))
    export_hf_checkpoint(model, export_dir=tmp_path / "ggml")
    before = load_file(tmp_path / "ggml" / "model.safetensors")

    convert_ggml_to_int8.convert_ggml_export_to_int8(tmp_path / "ggml", tmp_path / "int8")

    layers = json.loads((tmp_path / "int8" / "hf_quant_config.json").read_text())["quantization"]
    nvfp4_layers = [f"model.layers.1.mlp.{proj}" for proj in ("down_proj", "gate_proj", "up_proj")]
    assert {layers["quantized_layers"][layer]["quant_algo"] for layer in nvfp4_layers} == {
        "W4A16_NVFP4"
    }
    weight_map = json.loads((tmp_path / "int8" / "model.safetensors.index.json").read_text())
    after = {}
    for shard in set(weight_map["weight_map"].values()):
        after.update(load_file(tmp_path / "int8" / shard))
    for name, tensor in before.items():
        if name.startswith("model.layers.1.mlp."):
            assert torch.equal(after[name], tensor)
