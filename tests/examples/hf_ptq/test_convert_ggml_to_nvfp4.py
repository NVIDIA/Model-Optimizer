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
from _test_utils.examples.ggml_export import quantize_tiny_llama
from _test_utils.examples.hf_ptq_example_utils import convert_ggml_to_nvfp4
from _test_utils.examples.run_command import run_example_command
from safetensors.torch import load_file, save_file

from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.nvfp4 import nvfp4_transcode
from modelopt.torch.quantization.qtensor import NVFP4QTensor

# quantize_tiny_llama's layer 0 is IQ1_S and layer 1 IQ2_XS.
LAYER_FORMATS = {0: "iq1_s", 1: "iq2_xs"}
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
NVFP4_INPUT = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}


def _export(tmp_path, with_input_scales):
    model, _ = quantize_tiny_llama(tmp_path)
    if with_input_scales:
        # Set as calibration would; NVFP4 activation fake quant itself needs CUDA.
        for i in LAYER_FORMATS:
            for proj in PROJECTIONS:
                quantizer = model.get_submodule(f"model.layers.{i}.mlp.{proj}").input_quantizer
                quantizer.set_from_attribute_config({**NVFP4_INPUT, "enable": True})
                quantizer.amax = torch.tensor(2.0 + i)
    export_dir = tmp_path / "ggml"
    export_hf_checkpoint(model, export_dir=export_dir)
    return export_dir


def _load(directory):
    tensors = {}
    for shard in set(
        json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"].values()
    ):
        tensors.update(load_file(directory / shard))
    return tensors


@pytest.mark.parametrize("with_input_scales", [True, False])
def test_ggml_export_converts_to_the_nvfp4_weights_fake_quant_computes(tmp_path, with_input_scales):
    ggml_dir = _export(tmp_path, with_input_scales)
    before = load_file(ggml_dir / "model.safetensors")
    nvfp4_dir = tmp_path / "nvfp4"

    run_example_command(
        [
            "python",
            "convert_ggml_to_nvfp4.py",
            str(ggml_dir),
            str(nvfp4_dir),
            "--max_shard_size_gb",
            "0.0002",
        ],
        "hf_ptq",
    )

    quant_algo = "NVFP4" if with_input_scales else "W4A16_NVFP4"
    hf_quant_config = json.loads((nvfp4_dir / "hf_quant_config.json").read_text())
    assert hf_quant_config["quantization"]["quant_algo"] == quant_algo
    config = json.loads((nvfp4_dir / "config.json").read_text())["quantization_config"]
    assert (config["quant_method"], config["quant_algo"]) == ("modelopt", quant_algo)
    after = _load(nvfp4_dir)
    for i, name in LAYER_FORMATS.items():
        ggml_format = GGML_FORMAT_REGISTRY[name]
        gate_up_scales = set()
        for proj in PROJECTIONS:
            layer = f"model.layers.{i}.mlp.{proj}"
            payload = before.pop(f"{layer}.weight")
            shape = torch.tensor((payload.shape[0], payload.shape[1] * ggml_format.block_size))
            decoded = ggml_format.dequantize(payload, shape, dtype=torch.float32)
            scale, scale_2 = (
                after.pop(f"{layer}.weight_scale"),
                after.pop(f"{layer}.weight_scale_2"),
            )
            stored = NVFP4QTensor(decoded.shape, torch.float32, after.pop(f"{layer}.weight"))
            stored = stored.dequantize(scale=scale, double_scale=scale_2, block_sizes={-1: 16})
            # What the transcode: nvfp4 fake quant computed for this weight during calibration.
            assert torch.equal(stored, nvfp4_transcode(decoded))
            if proj != "down_proj":
                gate_up_scales.add(scale_2.item())
            if with_input_scales:
                assert torch.equal(
                    after.pop(f"{layer}.input_scale"), before.pop(f"{layer}.input_scale")
                )
        # vLLM fuses gate_proj and up_proj into one GEMM with one per-tensor scale.
        assert len(gate_up_scales) == 1
    assert after.keys() == before.keys()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor)


def test_an_export_with_some_activation_scales_is_refused(tmp_path):
    ggml_dir = _export(tmp_path, with_input_scales=True)
    tensors = load_file(ggml_dir / "model.safetensors")
    del tensors["model.layers.1.mlp.up_proj.input_scale"]
    save_file(tensors, ggml_dir / "model.safetensors", {"format": "pt"})

    with pytest.raises(ValueError, match="Only some GGML layers have activation scales"):
        convert_ggml_to_nvfp4.convert_ggml_export_to_nvfp4(ggml_dir, tmp_path / "nvfp4")

    assert not (tmp_path / "nvfp4").exists()
