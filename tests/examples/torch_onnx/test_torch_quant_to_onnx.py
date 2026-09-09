# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from collections import defaultdict

import onnx
import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command

from modelopt.recipe import load_recipe

# TODO: Add int4_awq once the INT4 exporter supports non-MatMul/Gemm consumer patterns
# (e.g., DQ -> Reshape -> Slice in small ViT / SwinTransformer ONNX graphs).
_QFORMATS = ["fp8", "int8", "mxfp8", "nvfp4", "auto"]
_RESNET_RECIPE_QFORMATS = {"fp8", "int8"}

_MODELS = {
    "vit_tiny": ("vit_tiny_patch16_224", '{"depth": 1}'),
    "swin_tiny": ("swin_tiny_patch4_window7_224", '{"depths": [1, 1, 1, 1]}'),
    "swinv2_tiny": ("swinv2_tiny_window8_256", '{"depths": [1, 1, 1, 1]}'),
    "resnet50": ("resnet50", None),
}


def _assert_residual_inputs_are_quantized(onnx_save_path):
    model = onnx.load(onnx_save_path)
    consumers = defaultdict(list)
    producers = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
        for output_name in node.output:
            producers[output_name] = node

    residual_adds = [
        node
        for node in model.graph.node
        if node.op_type == "Add"
        and [consumer.op_type for consumer in consumers[node.output[0]]] == ["Relu"]
    ]
    assert len(residual_adds) == 16
    for add in residual_adds:
        input_producers = [producers[input_name] for input_name in add.input]
        input_producers = [
            producers[node.input[0]] if node.op_type == "Cast" else node for node in input_producers
        ]
        assert any(
            node.op_type.endswith("DequantizeLinear")
            and "/downsample/residual_quantizer/" in node.name
            and producers[node.input[0]].op_type.endswith("QuantizeLinear")
            for node in input_producers
        )


@pytest.mark.parametrize("qformat", _QFORMATS)
@pytest.mark.parametrize("model_key", list(_MODELS))
def test_torch_onnx(tmp_path, model_key, qformat):
    if model_key == "resnet50" and qformat not in _RESNET_RECIPE_QFORMATS:
        pytest.skip("Only FP8 and INT8 quantization are supported for ResNet")

    timm_model_name, model_kwargs = _MODELS[model_key]
    onnx_save_path = tmp_path / f"{model_key}.{qformat}.onnx"

    cmd_parts = extend_cmd_parts(
        ["python", "torch_quant_to_onnx.py"],
        timm_model_name=timm_model_name,
        model_kwargs=model_kwargs,
        qformat=qformat,
        recipe=(
            f"timm/resnet/ptq/{qformat}"
            if model_key == "resnet50" and qformat in _RESNET_RECIPE_QFORMATS
            else None
        ),
        onnx_save_path=str(onnx_save_path),
        calibration_data_size="1",
        num_score_steps="1",
    )
    cmd_parts.extend(["--no_pretrained", "--trt_build"])
    run_example_command(cmd_parts, "torch_onnx")

    if model_key == "resnet50" and qformat in _RESNET_RECIPE_QFORMATS:
        _assert_residual_inputs_are_quantized(onnx_save_path)


def test_torch_onnx_recipe_flag(tmp_path):
    timm_model_name, model_kwargs = _MODELS["vit_tiny"]
    onnx_save_path = tmp_path / "vit_tiny.recipe.onnx"
    recipe_path = tmp_path / "disable_all.yaml"
    recipe_path.write_text(
        "metadata:\n"
        "  recipe_type: ptq\n"
        "  description: Disable all quantizers.\n"
        "quantize:\n"
        "  algorithm: max\n"
        "  quant_cfg:\n"
        "    - quantizer_name: '*'\n"
        "      enable: false\n"
    )

    cmd_parts = extend_cmd_parts(
        ["python", "torch_quant_to_onnx.py"],
        timm_model_name=timm_model_name,
        model_kwargs=model_kwargs,
        qformat="int8",
        recipe=str(recipe_path),
        onnx_save_path=str(onnx_save_path),
        calibration_data_size="1",
        num_score_steps="1",
    )
    cmd_parts.append("--no_pretrained")
    run_example_command(cmd_parts, "torch_onnx")

    quantize_ops = {
        "QuantizeLinear",
        "DequantizeLinear",
        "TRT_FP4DynamicQuantize",
        "TRT_FP8QuantizeLinear",
    }
    assert not quantize_ops & {node.op_type for node in onnx.load(onnx_save_path).graph.node}


def test_torch_onnx_auto_quantize_recipe(tmp_path):
    timm_model_name, model_kwargs = _MODELS["vit_tiny"]
    onnx_save_path = tmp_path / "vit_tiny.auto_recipe.onnx"
    recipe_path = tmp_path / "auto.yaml"
    recipe_path.write_text(
        "imports:\n"
        "  fp8: configs/ptq/presets/model/fp8\n"
        "  int8: configs/ptq/presets/model/int8\n"
        "metadata:\n"
        "  recipe_type: auto_quantize\n"
        "  description: Test AutoQuantize recipe.\n"
        "auto_quantize:\n"
        "  constraints:\n"
        "    effective_bits: 8.0\n"
        "  candidate_formats:\n"
        "    - $import: fp8\n"
        "    - $import: int8\n"
        "  auto_quantize_method: gradient\n"
        "  score_size: 1\n"
    )

    cmd_parts = extend_cmd_parts(
        ["python", "torch_quant_to_onnx.py"],
        timm_model_name=timm_model_name,
        model_kwargs=model_kwargs,
        qformat="int8",
        recipe=str(recipe_path),
        onnx_save_path=str(onnx_save_path),
        calibration_data_size="1",
    )
    cmd_parts.append("--no_pretrained")
    run_example_command(cmd_parts, "torch_onnx")


def test_auto_quantize_recipe_mapping():
    from examples.torch_onnx.torch_quant_to_onnx import (
        _enables_resnet_residual_quantization,
        _mtq_inputs_from_auto_quantize_config,
    )

    recipe = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits")
    inputs = _mtq_inputs_from_auto_quantize_config(recipe.auto_quantize, recipe.quantize)

    assert inputs["num_score_steps"] == recipe.auto_quantize.score_size
    assert len(inputs["quantization_formats"]) == 2
    block_format = next(
        config
        for config in inputs["quantization_formats"]
        if any(
            isinstance(entry.get("cfg"), dict) and entry["cfg"].get("block_sizes")
            for entry in config["quant_cfg"]
        )
    )
    assert any(entry.get("parent_class") == "nn.Conv2d" for entry in block_format["quant_cfg"])
    assert _enables_resnet_residual_quantization(load_recipe("timm/resnet/ptq/fp8"))
