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

from types import SimpleNamespace
from unittest.mock import Mock

import onnx
import pytest

import modelopt.onnx.quantization.autotune.export_utils as export_utils
import modelopt.onnx.quantization.precision_utils as precision_utils
from modelopt.onnx.quantization.autotune import workflows
from modelopt.onnx.quantization.autotune.common import Config


@pytest.mark.parametrize(
    ("quantize_mode", "expected_events"),
    [
        ("int8", ["convert"]),
        ("fp8", ["import", "remove_outputs", "convert_io", "export", "convert", "upgrade", "mha"]),
    ],
)
@pytest.mark.parametrize("direct_io_types", [False, True])
def test_runtime_precision_conversion_preserves_mode_steps(
    monkeypatch, quantize_mode, expected_events, direct_io_types
):
    source = onnx.ModelProto()
    graph = object()
    io_model = onnx.ModelProto()
    converted = onnx.ModelProto()
    converted.opset_import.add(domain="", version=17)
    final = onnx.ModelProto()
    events = []

    monkeypatch.setattr(
        precision_utils.gs, "import_onnx", lambda model: events.append("import") or graph
    )
    monkeypatch.setattr(
        precision_utils,
        "remove_output_initializers",
        lambda graph, initializers: events.append("remove_outputs"),
    )
    monkeypatch.setattr(
        precision_utils, "convert_fp16_io", lambda graph: events.append("convert_io")
    )
    monkeypatch.setattr(
        precision_utils.gs, "export_onnx", lambda graph: events.append("export") or io_model
    )

    def convert(model, **kwargs):
        events.append("convert")
        assert model is (io_model if quantize_mode == "fp8" else source)
        assert kwargs == {
            "keep_io_types": not direct_io_types,
            "op_block_list": ["Resize"],
            "tensor_block_dict": {"Custom": {"inputs": [0]}},
            "low_precision_type": "fp16",
            "trt_plugins": ["plugin.so"],
            "opset": 17,
        }
        return converted

    monkeypatch.setattr(precision_utils, "convert_to_f16", convert)
    monkeypatch.setattr(
        precision_utils,
        "_upgrade_opset_21",
        lambda model: events.append("upgrade") or converted,
    )
    monkeypatch.setattr(
        precision_utils,
        "insert_fp8_mha_casts",
        lambda model: events.append("mha") or final,
    )

    result = precision_utils._convert_to_runtime_precision(
        source,
        quantize_mode=quantize_mode,
        high_precision_dtype="fp16",
        direct_io_types=direct_io_types,
        op_types_to_exclude_fp16=["Resize"],
        custom_ops_to_cast_fp32={"Custom": {"inputs": [0]}},
        trt_extra_plugin_lib_paths=["plugin.so"],
        opset=17,
        mha_accumulation_dtype="fp32",
    )

    assert events == expected_events
    assert result is (final if quantize_mode == "fp8" else converted)


def test_export_transform_runs_between_int8_qdq_and_fp8(monkeypatch):
    source = onnx.ModelProto()
    source_bytes = source.SerializeToString()
    graph = type("Graph", (), {"toposort": lambda self: None})()
    int8_model, transformed, fp8_model = (onnx.ModelProto() for _ in range(3))
    events = []

    monkeypatch.setattr(export_utils.gs, "import_onnx", lambda model: graph)
    monkeypatch.setattr(
        export_utils.gs, "export_onnx", lambda graph: events.append("export") or int8_model
    )
    monkeypatch.setattr(
        export_utils,
        "insert_qdq_at_tensors",
        lambda graph, points, config: events.append(f"insert_{config.default_quant_type}"),
    )
    monkeypatch.setattr(
        export_utils,
        "fix_zero_point_initializers",
        lambda model: events.append("fix_zero_point"),
    )
    monkeypatch.setattr(
        export_utils,
        "int8_to_fp8",
        lambda model: events.append("convert_fp8") or fp8_model,
    )

    def transform(model):
        assert model is int8_model
        events.append("transform")
        return transformed

    result = export_utils.export_qdq_onnx(
        source,
        {object()},
        Config(default_quant_type="fp8"),
        needs_fp8_conversion=True,
        model_transform=transform,
    )

    assert events == ["insert_int8", "export", "fix_zero_point", "transform", "convert_fp8"]
    assert result is fp8_model
    assert source.SerializeToString() == source_bytes


def test_workflow_transforms_every_benchmark_export(monkeypatch, tmp_path):
    autotuner = Mock(
        regions=[SimpleNamespace(id=0, level=0)],
        baseline_latency_ms=None,
        current_profile_pattern_schemes=SimpleNamespace(schemes=[]),
    )
    autotuner.generate.return_value = 0
    autotuner.export_onnx.return_value = onnx.ModelProto().SerializeToString()
    monkeypatch.setattr(workflows, "QDQAutotuner", lambda model: autotuner)
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: 1.0)

    def transform(model):
        return model

    workflows.region_pattern_autotuning_workflow(
        onnx.ModelProto(), output_dir=tmp_path, num_schemes_per_region=1, model_transform=transform
    )

    exports = autotuner.export_onnx.call_args_list
    assert [(call.kwargs["insert_qdq"], call.kwargs.get("best", False)) for call in exports] == [
        (False, False),
        (True, False),
        (True, True),
        (True, False),
    ]
    assert all(call.kwargs["model_transform"] is transform for call in exports)
