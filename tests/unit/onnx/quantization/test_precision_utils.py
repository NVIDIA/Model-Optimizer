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

import onnx
import pytest

import modelopt.onnx.quantization.autotune.export_utils as export_utils
import modelopt.onnx.quantization.precision_utils as precision_utils
from modelopt.onnx.quantization.autotune.common import Config


@pytest.mark.parametrize(
    ("quantize_mode", "expected_events"),
    [
        pytest.param("int8", ["convert"], id="int8"),
        pytest.param(
            "fp8",
            ["import", "remove_outputs", "convert_io", "export", "convert", "upgrade", "mha"],
            id="fp8",
        ),
    ],
)
def test_convert_to_runtime_precision_preserves_mode_specific_steps(
    monkeypatch, quantize_mode, expected_events
):
    source_model = onnx.ModelProto()
    graph_copy = object()
    io_converted_model = onnx.ModelProto()
    low_precision_model = onnx.ModelProto()
    low_precision_model.opset_import.add(domain="", version=17)
    upgraded_model = onnx.ModelProto()
    final_model = onnx.ModelProto()
    events = []

    monkeypatch.setattr(
        precision_utils.gs,
        "import_onnx",
        lambda model: events.append("import") or graph_copy,
    )
    monkeypatch.setattr(
        precision_utils,
        "remove_output_initializers",
        lambda graph, initializers: events.append("remove_outputs"),
    )
    monkeypatch.setattr(
        precision_utils,
        "convert_fp16_io",
        lambda graph: events.append("convert_io"),
    )
    monkeypatch.setattr(
        precision_utils.gs,
        "export_onnx",
        lambda graph: events.append("export") or io_converted_model,
    )

    def convert_to_f16(model, **kwargs):
        events.append("convert")
        assert model is (io_converted_model if quantize_mode == "fp8" else source_model)
        assert kwargs == {
            "keep_io_types": True,
            "op_block_list": ["Resize"],
            "tensor_block_dict": {"Custom": {"inputs": [0]}},
            "low_precision_type": "fp16",
            "trt_plugins": ["plugin.so"],
            "opset": 17,
        }
        return low_precision_model

    monkeypatch.setattr(precision_utils, "convert_to_f16", convert_to_f16)
    monkeypatch.setattr(
        precision_utils,
        "_upgrade_opset_21",
        lambda model: events.append("upgrade") or upgraded_model,
    )
    monkeypatch.setattr(
        precision_utils,
        "insert_fp8_mha_casts",
        lambda model: events.append("mha") or final_model,
    )

    result = precision_utils._convert_to_runtime_precision(
        source_model,
        quantize_mode=quantize_mode,
        high_precision_dtype="fp16",
        direct_io_types=False,
        op_types_to_exclude_fp16=["Resize"],
        custom_ops_to_cast_fp32={"Custom": {"inputs": [0]}},
        trt_extra_plugin_lib_paths=["plugin.so"],
        opset=17,
        mha_accumulation_dtype="fp32",
    )

    assert events == expected_events
    assert result is (final_model if quantize_mode == "fp8" else low_precision_model)


def test_export_transform_runs_between_int8_qdq_and_fp8(monkeypatch):
    source_model = onnx.ModelProto()
    source_bytes = source_model.SerializeToString()
    graph_copy = type("GraphCopy", (), {"toposort": lambda self: None})()
    int8_model = onnx.ModelProto()
    transformed_model = onnx.ModelProto()
    fp8_model = onnx.ModelProto()
    events = []

    monkeypatch.setattr(export_utils.gs, "import_onnx", lambda model: graph_copy)
    monkeypatch.setattr(
        export_utils.gs,
        "export_onnx",
        lambda graph: events.append("export") or int8_model,
    )

    def insert_qdq(graph, insertion_points, config):
        events.append("insert_int8_qdq")
        assert config.default_quant_type == "int8"

    monkeypatch.setattr(export_utils, "insert_qdq_at_tensors", insert_qdq)
    monkeypatch.setattr(
        export_utils,
        "fix_zero_point_initializers",
        lambda model: events.append("fix_zero_point"),
    )

    def transform(model):
        events.append("transform")
        assert model is int8_model
        return transformed_model

    def int8_to_fp8(model):
        events.append("convert_fp8")
        assert model is transformed_model
        return fp8_model

    monkeypatch.setattr(export_utils, "int8_to_fp8", int8_to_fp8)

    result = export_utils.export_qdq_onnx(
        source_model,
        {object()},
        Config(default_quant_type="fp8"),
        needs_fp8_conversion=True,
        model_transform=transform,
    )

    assert events == [
        "insert_int8_qdq",
        "export",
        "fix_zero_point",
        "transform",
        "convert_fp8",
    ]
    assert result is fp8_model
    assert source_model.SerializeToString() == source_bytes
