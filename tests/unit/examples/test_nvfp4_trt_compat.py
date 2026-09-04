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
import sys
import types
from pathlib import Path
from unittest import mock

import onnx
import pytest

from examples.onnx_ptq import _trt_compat

REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    ("quantize_mode", "recipe", "auto_formats", "expected"),
    [
        ("nvfp4", None, [], True),
        ("nvfp4", "w4a16_nvfp4", [], False),
        ("nvfp4", "custom_recipe", [], True),
        ("auto", None, ["nvfp4_awq_lite", "fp8"], True),
        ("auto", None, ["mxfp8", "fp8"], False),
        ("fp8", None, [], False),
    ],
)
def test_request_uses_dynamic_nvfp4(quantize_mode, recipe, auto_formats, expected):
    assert _trt_compat.request_uses_dynamic_nvfp4(quantize_mode, recipe, auto_formats) is expected


@pytest.mark.parametrize("op_type", ["TRT_FP4DynamicQuantize", "QuantizeLinear"])
def test_onnx_uses_dynamic_nvfp4(tmp_path, op_type):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node(op_type, ["input"], ["output"])],
        "test_graph",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph), path)

    assert _trt_compat.onnx_uses_dynamic_nvfp4(path) is (op_type == "TRT_FP4DynamicQuantize")


@pytest.mark.parametrize("attribute_type", [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS])
def test_onnx_uses_dynamic_nvfp4_in_subgraph(tmp_path, attribute_type):
    subgraph = onnx.helper.make_graph(
        [onnx.helper.make_node("TRT_FP4DynamicQuantize", ["input"], ["output"])],
        "subgraph",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    container = onnx.helper.make_node("Container", [], [])
    attribute = container.attribute.add()
    attribute.name = "subgraphs"
    attribute.type = attribute_type
    if attribute_type == onnx.AttributeProto.GRAPH:
        attribute.g.CopyFrom(subgraph)
    else:
        attribute.graphs.append(subgraph)
    graph = onnx.helper.make_graph([container], "test_graph", [], [])
    path = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph), path)

    assert _trt_compat.onnx_uses_dynamic_nvfp4(path)


def test_check_dynamic_nvfp4_trt_support_uses_minimum_version(monkeypatch):
    check = mock.Mock()
    monkeypatch.setattr(_trt_compat, "_check_for_trtexec", check)

    _trt_compat.check_dynamic_nvfp4_trt_support()

    check.assert_called_once_with(min_version="11.0")


def test_check_dynamic_nvfp4_trt_support_reports_fallback(monkeypatch):
    def reject_trt10(*, min_version):
        raise ImportError(f"trtexec version must be >= {min_version}, found 10.16")

    monkeypatch.setattr(_trt_compat, "_check_for_trtexec", reject_trt10)

    with pytest.raises(ImportError, match="--recipe=w4a16_nvfp4"):
        _trt_compat.check_dynamic_nvfp4_trt_support()


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_example(monkeypatch, name, path):
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_torch_build_rejects_before_model_creation(monkeypatch, capsys, tmp_path):
    create_model = mock.Mock()
    monkeypatch.setitem(sys.modules, "timm", _module("timm", create_model=create_model))
    monkeypatch.setitem(sys.modules, "datasets", _module("datasets", load_dataset=mock.Mock()))
    monkeypatch.setitem(
        sys.modules,
        "download_example_onnx",
        _module("download_example_onnx", export_to_onnx=mock.Mock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "evaluation",
        _module("evaluation", evaluate=mock.Mock()),
    )
    module = _load_example(
        monkeypatch,
        "test_torch_quant_to_onnx_entrypoint",
        REPO_ROOT / "examples/torch_onnx/torch_quant_to_onnx.py",
    )

    def reject_trt10():
        raise ImportError("dynamic NVFP4 requires TensorRT 11.0")

    monkeypatch.setattr(module, "check_dynamic_nvfp4_trt_support", reject_trt10)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch_quant_to_onnx.py",
            "--quantize_mode=nvfp4",
            f"--onnx_save_path={tmp_path / 'model.onnx'}",
            "--trt_build",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        module.main()

    create_model.assert_not_called()
    assert "dynamic NVFP4 requires TensorRT 11.0" in capsys.readouterr().err


def test_evaluate_rejects_before_runtime_creation(monkeypatch, capsys):
    runtime_get = mock.Mock()
    monkeypatch.setitem(sys.modules, "timm", _module("timm", create_model=mock.Mock()))
    monkeypatch.setitem(
        sys.modules,
        "evaluation",
        _module("evaluation", evaluate=mock.Mock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch._deploy._runtime",
        _module(
            "modelopt.torch._deploy._runtime",
            RuntimeRegistry=types.SimpleNamespace(get=runtime_get),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch._deploy.device_model",
        _module("modelopt.torch._deploy.device_model", DeviceModel=mock.Mock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch._deploy.utils",
        _module("modelopt.torch._deploy.utils", OnnxBytes=mock.Mock()),
    )
    module = _load_example(
        monkeypatch,
        "test_onnx_ptq_evaluate_entrypoint",
        REPO_ROOT / "examples/onnx_ptq/evaluate.py",
    )
    monkeypatch.setattr(module, "onnx_uses_dynamic_nvfp4", lambda _: True)

    def reject_trt10():
        raise ImportError("dynamic NVFP4 requires TensorRT 11.0")

    monkeypatch.setattr(module, "check_dynamic_nvfp4_trt_support", reject_trt10)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--onnx_path=model.onnx", "--model_name=vit_small_patch16_224"],
    )

    with pytest.raises(SystemExit, match="2"):
        module.main()

    runtime_get.assert_not_called()
    assert "dynamic NVFP4 requires TensorRT 11.0" in capsys.readouterr().err
