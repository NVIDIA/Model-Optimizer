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


import copy
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import onnx
import pytest
import torch
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.recipe.config import AutoQuantizeConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_example_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dynamo_export", [False, True])
def test_export_to_onnx_defers_default_opset_to_modelopt(monkeypatch, tmp_path, dynamo_export):
    download_example_onnx = _load_example_module(
        "download_example_onnx_for_test", "examples/onnx_ptq/download_example_onnx.py"
    )

    export_call = {}

    def fake_export(**kwargs):
        export_call.update(kwargs)
        return b"onnx", {}

    class FakeOnnxBytes:
        def write_to_disk(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(download_example_onnx, "get_onnx_bytes_and_metadata", fake_export)
    monkeypatch.setattr(
        download_example_onnx.OnnxBytes,
        "from_bytes",
        lambda _onnx_bytes: FakeOnnxBytes(),
    )

    download_example_onnx.export_to_onnx(
        torch.nn.Linear(2, 2),
        (1, 2),
        tmp_path / "model.onnx",
        torch.device("cpu"),
        dynamo_export=dynamo_export,
    )

    assert export_call["dynamo_export"] is dynamo_export
    assert export_call["onnx_opset"] is None


@pytest.mark.parametrize("num_convs", [1, 2])
def test_dynamo_fp8_export_with_unquantized_rgb_input(tmp_path, num_convs):
    torch_quant_to_onnx = _load_example_module(
        "torch_quant_to_onnx_for_test", "examples/torch_onnx/torch_quant_to_onnx.py"
    )
    inputs = torch.randn(1, 3, 4, 4)
    model = torch.nn.Sequential(
        *(torch.nn.Conv2d(3 if index == 0 else 8, 8, 1) for index in range(num_convs))
    ).eval()
    model = mtq.quantize(
        model,
        torch_quant_to_onnx.get_quant_config("fp8"),
        forward_loop=lambda candidate: candidate(inputs),
    )
    torch_quant_to_onnx._disable_low_channel_fp8_conv_input_quantizers(model)
    onnx_path = tmp_path / "rgb_conv.onnx"

    torch_quant_to_onnx.export_to_onnx(
        model,
        inputs.shape,
        onnx_path,
        torch.device("cpu"),
        weights_dtype="fp16",
        dynamo_export=True,
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    initializers = {tensor.name: tensor for tensor in exported.graph.initializer}
    producers = {output: node for node in exported.graph.node for output in node.output}
    convs = [node for node in exported.graph.node if node.op_type == "Conv"]
    assert len(convs) == num_convs
    assert convs[0].input[0] == exported.graph.input[0].name
    for conv in convs:
        weight_dq = producers[conv.input[1]]
        assert (weight_dq.domain, weight_dq.op_type) == ("", "DequantizeLinear")
        assert initializers[weight_dq.input[0]].data_type == onnx.TensorProto.FLOAT8E4M3FN


@pytest.mark.parametrize(
    ("opset", "trt_build", "message"),
    [
        (22, False, "requires --onnx_opset=23 or newer"),
        (25, True, "supports --onnx_opset up to 24"),
    ],
)
def test_cli_rejects_unsupported_dynamo_opset(
    monkeypatch, capsys, tmp_path, opset, trt_build, message
):
    torch_quant_to_onnx = _load_example_module(
        "torch_quant_to_onnx_for_test", "examples/torch_onnx/torch_quant_to_onnx.py"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch_quant_to_onnx.py",
            "--onnx_save_path",
            str(tmp_path / "model.onnx"),
            "--dynamo_export",
            f"--onnx_opset={opset}",
            *(["--trt_build"] if trt_build else []),
        ],
    )
    monkeypatch.setattr(
        torch_quant_to_onnx.timm,
        "create_model",
        lambda *_args, **_kwargs: pytest.fail("invalid opset should fail before model loading"),
    )

    with pytest.raises(SystemExit, match="2"):
        torch_quant_to_onnx.main()

    assert message in capsys.readouterr().err


@pytest.mark.parametrize("enabled_int4_layer", [None, 0, 1])
def test_cli_trt_build_checks_enabled_int4_weights(
    monkeypatch, capsys, tmp_path, enabled_int4_layer
):
    example = _load_example_module(
        "torch_quant_to_onnx_for_test", "examples/torch_onnx/torch_quant_to_onnx.py"
    )
    model = torch.nn.Sequential(*(torch.nn.Linear(128, 128) for _ in range(3))).eval()
    inputs = torch.randn(1, 128)
    config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    config["quant_cfg"].extend(
        [
            {
                "quantizer_name": "2.weight_quantizer",
                "cfg": {"num_bits": (4, 3), "block_sizes": None},
            },
            {"quantizer_name": "2.input_quantizer", "cfg": {"num_bits": (4, 3)}, "enable": True},
        ]
    )
    model = mtq.quantize(model, config, forward_loop=lambda candidate: candidate(inputs))
    for index in range(2):
        if index != enabled_int4_layer:
            model[index].weight_quantizer.disable()

    onnx_path = tmp_path / "mixed.onnx"
    monkeypatch.setattr(
        sys, "argv", ["torch_quant_to_onnx.py", "--onnx_save_path", str(onnx_path), "--trt_build"]
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(example.timm, "create_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(example, "get_model_input_shape", lambda _model: (128,))
    monkeypatch.setattr(example, "load_calibration_data", lambda *_args, **_kwargs: [inputs])
    monkeypatch.setattr(example, "quantize_model", lambda *_args: model)
    builds = []
    monkeypatch.setattr(
        example, "build_trt_engine", lambda path, *_args, **_kwargs: builds.append(path)
    )

    if enabled_int4_layer is not None:
        with pytest.raises(SystemExit, match="2"):
            example.main()
        assert "exported graph contains INT4 AWQ weights" in capsys.readouterr().err
        assert not onnx_path.exists()
        assert builds == []
    else:
        example.main()
        exported = onnx.load(onnx_path)
        onnx.checker.check_model(exported, full_check=True)
        assert all(
            tensor.data_type != onnx.TensorProto.INT4 for tensor in exported.graph.initializer
        )
        assert builds == [str(onnx_path)]


# TODO: Add int4_awq once the INT4 exporter supports non-MatMul/Gemm consumer patterns
# (e.g., DQ -> Reshape -> Slice in small ViT / SwinTransformer ONNX graphs).
_QFORMATS = ["fp8", "int8", "mxfp8", "nvfp4", "auto"]
_RESNET_RECIPE_QFORMATS = {"fp8", "int8"}
_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "torch_onnx"

_MODELS = {
    "vit_tiny": ("vit_tiny_patch16_224", '{"depth": 1}'),
    "swin_tiny": ("swin_tiny_patch4_window7_224", '{"depths": [1, 1, 1, 1]}'),
    "swinv2_tiny": ("swinv2_tiny_window8_256", '{"depths": [1, 1, 1, 1]}'),
    "resnet50": ("resnet50", None),
}

# PRs run swinv2 for fp8 and nvfp4 only (swin_tiny covers every format); the other cases are nightly.
_NIGHTLY_ONLY = {("swinv2_tiny", "int8"), ("swinv2_tiny", "mxfp8"), ("swinv2_tiny", "auto")}


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


@pytest.mark.parametrize(
    ("model_key", "qformat"),
    [
        pytest.param(
            model_key,
            qformat,
            id=f"{model_key}-{qformat}",
            marks=pytest.mark.nightly if (model_key, qformat) in _NIGHTLY_ONLY else (),
        )
        for model_key in _MODELS
        for qformat in _QFORMATS
    ],
)
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
        trt_builder_optimization_level=0,
    )
    cmd_parts.extend(["--no_pretrained", "--trt_build"])
    run_example_command(cmd_parts, "torch_onnx")

    if model_key == "resnet50" and qformat in _RESNET_RECIPE_QFORMATS:
        _assert_residual_inputs_are_quantized(onnx_save_path)


@pytest.mark.timeout(600)
def test_torch_onnx_dynamo_fp8_default_opset(tmp_path):
    timm_model_name, model_kwargs = _MODELS["vit_tiny"]
    onnx_save_path = tmp_path / "vit_tiny.fp8.dynamo.onnx"
    cmd_parts = extend_cmd_parts(
        ["python", "torch_quant_to_onnx.py"],
        timm_model_name=timm_model_name,
        model_kwargs=model_kwargs,
        qformat="fp8",
        onnx_save_path=str(onnx_save_path),
        calibration_data_size="1",
    )
    cmd_parts.extend(["--no_pretrained", "--dynamo_export", "--trt_build"])

    env = os.environ.copy()
    env["TORCH_EXTENSIONS_DIR"] = str(tmp_path / "torch_extensions")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton_cache")
    run_example_command(cmd_parts, "torch_onnx", env=env)

    model = onnx.load(onnx_save_path, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    assert next(opset.version for opset in model.opset_import if not opset.domain) == 23
    assert any(
        initializer.data_type == onnx.TensorProto.FLOAT8E4M3FN
        for initializer in model.graph.initializer
    )


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


def _import_torch_quant_to_onnx(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLE_DIR))
    return importlib.import_module("torch_quant_to_onnx")


def test_auto_quantize_recipe_mapping(monkeypatch):
    torch_quant_to_onnx = _import_torch_quant_to_onnx(monkeypatch)

    recipe = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits")
    inputs = torch_quant_to_onnx._mtq_inputs_from_auto_quantize_config(
        recipe.auto_quantize, recipe.quantize
    )

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
    assert torch_quant_to_onnx._enables_resnet_residual_quantization(
        load_recipe("timm/resnet/ptq/fp8")
    )


def test_auto_quantize_damage_bound_recipe_mapping(monkeypatch):
    torch_quant_to_onnx = _import_torch_quant_to_onnx(monkeypatch)

    base = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    values = {name: getattr(base, name) for name in type(base).model_fields}
    auto_config = AutoQuantizeConfig.model_validate(
        {
            **values,
            "constraints": {},
            "auto_quantize_method": "aumann_shapley",
            "method_options": {
                "num_path_nodes": 3,
                "damage_link": "additive",
                "max_predicted_damage": 0.05,
            },
        }
    )

    inputs = torch_quant_to_onnx._mtq_inputs_from_auto_quantize_config(auto_config)

    assert "effective_bits" not in inputs["constraints"]
    assert inputs["constraints"]["cost_model"] == "weight"
    assert inputs["method"] == {
        "method": "aumann_shapley",
        "num_path_nodes": 3,
        "damage_link": "additive",
        "max_predicted_damage": 0.05,
    }
