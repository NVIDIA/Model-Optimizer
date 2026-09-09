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

from examples.onnx_ptq import _trt_compat


@pytest.mark.parametrize(
    ("trt_build", "recipe_provided", "qformat", "auto_formats", "expected"),
    [
        (True, False, "nvfp4", [], True),
        (True, False, "auto", ["nvfp4_awq_lite", "fp8"], True),
        (True, True, "nvfp4", [], False),
        (True, False, "fp8", [], False),
        (False, False, "nvfp4", [], False),
    ],
)
def test_request_needs_dynamic_nvfp4_check(
    trt_build, recipe_provided, qformat, auto_formats, expected
):
    assert (
        _trt_compat.request_needs_dynamic_nvfp4_check(
            qformat,
            auto_formats,
            recipe_provided=recipe_provided,
            trt_build=trt_build,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("placement", "expected"),
    [
        ("top_level", True),
        ("nested_graph", True),
        ("local_function", True),
        ("absent", False),
    ],
)
def test_onnx_uses_dynamic_nvfp4(tmp_path, placement, expected):
    op_type = "Identity" if placement == "absent" else "TRT_FP4DynamicQuantize"
    node = onnx.helper.make_node(op_type, ["input"], ["output"])
    function = None
    if placement == "nested_graph":
        subgraph = onnx.helper.make_graph([node], "subgraph", [], [])
        node = onnx.helper.make_node("Container", [], [], body=subgraph)
    elif placement == "local_function":
        function = onnx.helper.make_function(
            "local",
            "DynamicQuantize",
            ["input"],
            ["output"],
            [node],
            opset_imports=[onnx.helper.make_opsetid("", 20)],
        )
        node = onnx.helper.make_node("DynamicQuantize", ["input"], ["output"], domain="local")
    graph = onnx.helper.make_graph([node], "graph", [], [])
    model = onnx.helper.make_model(graph)
    if function is not None:
        model.functions.append(function)
        model.opset_import.append(onnx.helper.make_opsetid("local", 1))
    path = tmp_path / "model.onnx"
    onnx.save(model, path)

    assert _trt_compat.onnx_uses_dynamic_nvfp4(path) is expected


def test_check_dynamic_nvfp4_trt_support_reports_action(monkeypatch):
    def reject_trt10(*, min_version):
        assert min_version == "11.0"
        raise ImportError("`trtexec` version must be >= 11.0, found 10.16")

    monkeypatch.setattr(_trt_compat, "_check_for_trtexec", reject_trt10)

    with pytest.raises(ImportError) as error:
        _trt_compat.check_dynamic_nvfp4_trt_support()

    message = str(error.value)
    assert "TensorRT 11.0 or newer" in message
    assert "--qformat=fp8" in message
    assert "without `--trt_build`" in message
