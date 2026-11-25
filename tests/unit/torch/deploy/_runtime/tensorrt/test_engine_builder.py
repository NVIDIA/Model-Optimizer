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
import os
import stat
import sys
import tempfile
from importlib.machinery import ModuleSpec
from itertools import product
from pathlib import Path
from types import ModuleType
from unittest import mock

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.autocast.graphsanitizer import GraphSanitizer

# Patch mock tensorrt module
tensorrt = ModuleType("tensorrt")
tensorrt.__spec__ = ModuleSpec("tensorrt", loader=None)
tensorrt.Logger = mock.Mock()
tensorrt.tensorrt = mock.Mock()
tensorrt.__version__ = "10.10"
sys.modules["tensorrt"] = tensorrt

from modelopt.torch._deploy._runtime.tensorrt import engine_builder
from modelopt.torch._deploy._runtime.tensorrt.engine_builder import (
    _default_artifact_dir,
    build_engine,
    profile_engine,
)
from modelopt.torch._deploy.utils import OnnxBytes


@pytest.fixture
def setup_mocks(tmp_path):
    engine_bytes = b"engine_bytes"
    dummy_hash = "997930f2"
    expected_engine_bytes = b"\x99y0\xf2" + b"A" * 28 + b",engine_bytes"

    os.makedirs(tmp_path / "onnx", exist_ok=True)
    (tmp_path / "model.engine").write_bytes(engine_bytes)

    (tmp_path / f"{dummy_hash}-profile.json").write_text(
        json.dumps([{"count": 1}, {"name": "dummy_layer", "averageMs": 0.001}])
    )
    (tmp_path / f"{dummy_hash}-layerInfo.json").write_text(
        json.dumps({"Layers": [{"Name": "dummy_layer"}]})
    )

    mock_onnx = mock.Mock()
    mock_onnx.model_name = "model"

    with (
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder._run_trtexec_with_logging"
        ) as mock_run,
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder.TemporaryDirectory"
        ) as mock_tmp,
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder.prepend_hash_to_bytes"
        ) as mock_hash,
    ):
        mock_run.return_value = (0, "success")
        mock_tmp.return_value.__enter__.return_value = str(tmp_path)
        mock_hash.return_value = expected_engine_bytes
        yield tmp_path, mock_run, mock_onnx, dummy_hash


def _assert_engine_saved(tmp_path, dummy_hash, engine_bytes, out):
    assert not (tmp_path / "model.engine").exists()
    assert (tmp_path / f"{dummy_hash}-model.engine").exists()
    assert engine_bytes is not None and engine_bytes.startswith(b"\x99y0\xf2")
    assert out == "success"


def test_build_engine(setup_mocks):
    tmp_path, mock_run, mock_onnx, dummy_hash = setup_mocks
    engine_bytes, out = build_engine(onnx_bytes=mock_onnx, verbose=True, output_dir=tmp_path)

    mock_run.assert_called_once()
    _assert_engine_saved(tmp_path, dummy_hash, engine_bytes, out)


def test_build_engine_dynamic_shapes(setup_mocks):
    tmp_path, mock_run, mock_onnx, dummy_hash = setup_mocks
    shapes = {
        "minShapes": {"input": [1, 3, 244, 244]},
        "optShapes": {"input": [16, 3, 244, 244]},
        "maxShapes": {"input": [32, 3, 244, 244]},
    }
    engine_bytes, out = build_engine(
        onnx_bytes=mock_onnx, dynamic_shapes=shapes, output_dir=tmp_path
    )

    mock_run.assert_called_once()
    _assert_engine_saved(tmp_path, dummy_hash, engine_bytes, out)


@pytest.mark.parametrize("enable_layerwise", [False, True])
@pytest.mark.parametrize("profiling_runs", [1, 2])
def test_profile_engine_variants(setup_mocks, enable_layerwise, profiling_runs):
    tmp_path, mock_run, _, dummy_hash = setup_mocks
    engine_bytes = b"\x99y0\xf2" + b"A" * 28 + b",engine_bytes"

    kwargs = {
        "engine_bytes": engine_bytes,
        "onnx_node_names": [],
        "output_dir": tmp_path,
        "profiling_runs": profiling_runs,
        "enable_layerwise_profiling": enable_layerwise,
    }

    profiled_data, out = profile_engine(**kwargs)
    mock_run.assert_called_once()

    if enable_layerwise:
        assert profiled_data == {"dummy_layer": 0.001}
        assert (tmp_path / f"{dummy_hash}-profile.json").exists()
        assert (tmp_path / f"{dummy_hash}-layerInfo.json").exists()
    else:
        assert profiled_data == {}

    assert out == "success"


def _write_engine_like_trtexec(cmd):
    for arg in cmd:
        if arg.startswith("--saveEngine="):
            Path(arg.split("=", 1)[1]).write_bytes(b"engine_bytes")
    return 0, "success"


def test_default_artifact_dir_is_private_and_reused_within_a_process(setup_mocks, monkeypatch):
    tmp_path, mock_run, mock_onnx, _ = setup_mocks
    shared_tmp = tmp_path / "shared_tmp"
    shared_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(shared_tmp))
    _default_artifact_dir.cache_clear()
    mock_run.side_effect = _write_engine_like_trtexec

    build_engine(onnx_bytes=mock_onnx)
    build_engine(onnx_bytes=mock_onnx)
    profile_engine(engine_bytes=b"\x99y0\xf2" + b"A" * 28 + b",engine_bytes")
    _default_artifact_dir.cache_clear()

    # A fixed shared path would let another local user pre-create it or plant symlinks in it. One
    # unpredictable private directory per process keeps the timing cache shared across calls.
    (artifact_dir,) = shared_tmp.iterdir()
    if sys.platform != "win32":  # POSIX file modes
        assert stat.S_IMODE(artifact_dir.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("version", "precision", "external", "layout"),
    [
        (*case, "named")
        for case in product(
            ["10.10", "11.4"], ["fp16", "bf16", "fp32", "stronglyTyped"], [False, True]
        )
    ]
    + [
        ("11.4", precision, True, layout)
        for precision, layout in product(
            ["fp16", "bf16"],
            [
                "unnamed",
                "input_output",
                "initializer_output",
                "weight_output",
                "overridable_weight",
                "overridable_mixed",
            ],
        )
    ],
)
def test_build_engine_precision_graph(tmp_path, monkeypatch, version, precision, external, layout):
    monkeypatch.setattr(tensorrt, "__version__", version)
    source = tmp_path / "source"
    source.mkdir()
    weights = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="matmul")],
        "precision",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 32])],
        initializer=[numpy_helper.from_array(weights, "weight")],
    )
    if layout == "unnamed":
        graph.node[0].name = ""
    elif layout == "input_output":
        graph.output.append(graph.input[0])
    elif layout == "initializer_output":
        graph.initializer.append(numpy_helper.from_array(weights, "constant"))
        graph.output.append(helper.make_tensor_value_info("constant", TensorProto.FLOAT, [32, 32]))
    elif layout == "weight_output":
        graph.output.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [32, 32]))
    elif layout in ("overridable_weight", "overridable_mixed"):
        graph.input.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [32, 32]))
        graph.output.append(graph.input[-1])
        if layout == "overridable_mixed":
            graph.node.append(helper.make_node("Celu", ["weight"], ["high"], name="celu"))
            graph.output.append(helper.make_tensor_value_info("high", TensorProto.FLOAT, [32, 32]))
    original_io = [value.SerializeToString() for value in (*graph.input, *graph.output)]
    path = source / "model.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10),
        path,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    original_files = {p.name: p.read_bytes() for p in source.iterdir()}
    model = OnnxBytes(str(path))
    original_bytes = dict(model.onnx_model)
    observed = []

    def compile_graph(arguments, cwd=None):
        onnx_path = next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--onnx="))
        header = onnx.load(onnx_path, load_external_data=False)
        assert all(
            onnx.external_data_helper.uses_external_data(t) == external
            for t in header.graph.initializer
        )
        compiled_graph = onnx.load(onnx_path)
        onnx.checker.check_model(compiled_graph)
        expected_type = TensorProto.FLOAT
        if version.startswith("11"):
            expected_type = {"fp16": TensorProto.FLOAT16, "bf16": TensorProto.BFLOAT16}.get(
                precision, TensorProto.FLOAT
            )
            assert not set(arguments) & {"--fp16", "--bf16", "--fp8", "--int8", "--int4", "--best"}
        elif precision != "fp32":
            assert "--" + precision in arguments
        matmul = next(node for node in compiled_graph.graph.node if node.op_type == "MatMul")
        if layout in ("overridable_weight", "overridable_mixed"):
            cast = next(
                node for node in compiled_graph.graph.node if matmul.input[1] in node.output
            )
            assert cast.op_type == "Cast" and list(cast.input) == ["weight"]
            assert next(attr.i for attr in cast.attribute if attr.name == "to") == expected_type
        else:
            assert (
                next(
                    t for t in compiled_graph.graph.initializer if t.name == matmul.input[1]
                ).data_type
                == expected_type
            )
        assert [
            value.SerializeToString()
            for value in (*compiled_graph.graph.input, *compiled_graph.graph.output)
        ] == original_io
        if layout in (
            "initializer_output",
            "weight_output",
            "overridable_weight",
            "overridable_mixed",
        ):
            name = "constant" if layout == "initializer_output" else "weight"
            constant = next(t for t in compiled_graph.graph.initializer if t.name == name)
            np.testing.assert_array_equal(numpy_helper.to_array(constant), weights)
        observed.append(expected_type)
        engine = next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--saveEngine="))
        Path(engine).write_bytes(b"engine_bytes")
        return 0, b"compiled"

    # Only replace the unavailable native compiler; graph conversion and file handling are real.
    monkeypatch.setattr(engine_builder, "_run_trtexec_with_logging", compile_graph)
    engine, output = build_engine(model, trt_mode=precision, output_dir=tmp_path / "artifacts")
    assert engine is not None and engine.endswith(b"engine_bytes")
    assert output == b"compiled" and observed
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files
    assert model.onnx_model == original_bytes


@pytest.mark.parametrize(
    ("precision", "calibration_cache"),
    [("fp8", None), ("int8", None), ("int4", None), ("best", None), ("fp32", "legacy-cache")],
)
def test_trt11_rejects_implicit_precision_before_writing(
    tmp_path, monkeypatch, precision, calibration_cache
):
    monkeypatch.setattr(tensorrt, "__version__", "11.4")
    output = tmp_path / "artifacts"
    with pytest.raises(ValueError):
        build_engine(None, trt_mode=precision, calib_cache=calibration_cache, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("precision", "plugin_types", "output_type", "layout"),
    [
        ("fp16", (TensorProto.FLOAT,), TensorProto.FLOAT, "default"),
        ("bf16", (TensorProto.FLOAT,), TensorProto.FLOAT, "default"),
        ("fp16", (TensorProto.FLOAT16,), TensorProto.FLOAT16, "default"),
        ("fp16", (TensorProto.FLOAT,), TensorProto.FLOAT16, "default"),
        ("bf16", (TensorProto.FLOAT,), TensorProto.FLOAT16, "default"),
        ("fp16", (TensorProto.FLOAT,), TensorProto.FLOAT16, "overridable"),
        ("bf16", (TensorProto.FLOAT,), TensorProto.FLOAT16, "overridable"),
        ("fp16", (TensorProto.FLOAT16,), TensorProto.FLOAT, "default"),
    ]
    + [
        (precision, (TensorProto.FLOAT,), TensorProto.FLOAT16, "output_consumer")
        for precision in ["fp16", "bf16"]
    ]
    + [
        (precision, (TensorProto.FLOAT, input_type), TensorProto.FLOAT16, "default")
        for precision, input_type in product(
            ["fp16", "bf16"], [TensorProto.FLOAT16, TensorProto.BFLOAT16]
        )
    ]
    + [
        (precision, (TensorProto.FLOAT, TensorProto.FLOAT16), TensorProto.FLOAT16, layout)
        for precision, layout in product(["fp16", "bf16"], ["aliased", "empty"])
    ],
)
def test_trt11_preserves_plugin_precision(
    tmp_path, monkeypatch, precision, plugin_types, output_type, layout
):
    plugin_type = plugin_types[0]
    overridable = layout in ("overridable", "aliased")
    monkeypatch.setattr(tensorrt, "__version__", "11.4")

    def discover_plugin(self):
        self.custom_ops = {"FloatPlugin"}

    # Discovery requires native TensorRT; conversion and boundary casts remain real.
    monkeypatch.setattr(GraphSanitizer, "find_custom_nodes", discover_plugin)
    graph = helper.make_graph(
        [
            helper.make_node("FloatPlugin", ["x"], ["p"], domain="test.plugins", name="plugin"),
            helper.make_node("Cast", ["p"], ["p_cast"], to=plugin_type, name="plugin_output_cast"),
            helper.make_node("MatMul", ["p_cast", "weight"], ["y"], name="matmul"),
        ],
        "plugin_precision",
        [helper.make_tensor_value_info("x", plugin_type, [1, 4])],
        [helper.make_tensor_value_info("y", plugin_type, [1, 4])],
        initializer=[
            numpy_helper.from_array(
                np.eye(4, dtype=np.float32 if plugin_type == TensorProto.FLOAT else np.float16),
                "weight",
            )
        ],
        value_info=[helper.make_tensor_value_info("p", output_type, [1, 4])],
    )
    if overridable:
        graph.input.append(helper.make_tensor_value_info("weight", plugin_type, [4, 4]))
    if len(plugin_types) > 1:
        shape = [0, 4] if layout == "empty" else [1, 4]
        graph.node[0].input.append("half_input")
        if layout == "aliased":
            graph.node.insert(
                0, helper.make_node("Cast", ["x"], ["half_input"], to=plugin_types[1])
            )
        else:
            graph.input.append(helper.make_tensor_value_info("half_input", plugin_types[1], shape))
        graph.node.extend(
            [
                helper.make_node("Cast", ["half_input"], ["half_as_float"], to=TensorProto.FLOAT),
                helper.make_node("Celu", ["half_as_float"], ["high"], name="celu"),
            ]
        )
        graph.output.append(helper.make_tensor_value_info("high", TensorProto.FLOAT, shape))
    if layout == "output_consumer":
        graph.node.extend(
            [
                helper.make_node("Cast", ["p"], ["p_as_float"], to=TensorProto.FLOAT),
                helper.make_node("Celu", ["p_as_float"], ["high"], name="celu"),
            ]
        )
        graph.output.append(helper.make_tensor_value_info("high", TensorProto.FLOAT, [1, 4]))
    original_io = [v.SerializeToString() for v in (*graph.input, *graph.output)]
    path = tmp_path / "plugin.onnx"
    onnx.save_model(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("test.plugins", 1)],
            ir_version=10,
        ),
        path,
    )
    source = path.read_bytes()
    observed = []

    def compile_graph(arguments, cwd=None):
        path = next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--onnx="))
        model = onnx.load(path)
        types = {
            v.name: v.type.tensor_type.elem_type
            for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)
        }
        plugin = next(node for node in model.graph.node if node.op_type == "FloatPlugin")
        assert [types[name] for name in plugin.input] == list(plugin_types)
        if len(plugin_types) > 1 or layout == "output_consumer":
            celu = next(node for node in model.graph.node if node.op_type == "Celu")
            assert types[celu.input[0]] == TensorProto.FLOAT
        assert types[plugin.output[0]] == output_type
        assert model.graph.initializer[0].data_type == (
            plugin_type
            if overridable
            else {"fp16": TensorProto.FLOAT16, "bf16": TensorProto.BFLOAT16}[precision]
        )
        assert [
            v.SerializeToString() for v in (*model.graph.input, *model.graph.output)
        ] == original_io
        onnx.checker.check_model(model)
        observed.append(True)
        return _write_engine_like_trtexec(arguments)

    monkeypatch.setattr(engine_builder, "_run_trtexec_with_logging", compile_graph)
    engine, _ = build_engine(
        OnnxBytes(str(path)), trt_mode=precision, output_dir=tmp_path / "artifacts"
    )
    assert engine is not None and observed
    assert path.read_bytes() == source
