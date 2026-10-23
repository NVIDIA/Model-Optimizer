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

"""Validation for unavailable stages in the private DLA pipeline scaffold."""

import importlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

pipeline = importlib.import_module("modelopt.onnx.graph_surgery.make_dla_compatible")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for pipeline execution")


@pytest.mark.parametrize(
    "options",
    [
        {"remove_all_qdq": True},
        {"qdq_quantized_dtypes": ["int16"]},
        {"remove_all_qdq": True, "qdq_quantized_dtypes": ["int16"]},
    ],
)
def test_pipeline_rejects_unavailable_qdq_options(monkeypatch, options):
    """Unavailable QDQ options and conflicting options fail without changing the model."""
    monkeypatch.setattr(
        pipeline,
        "DLA_TRANSFORMS",
        tuple(stage for stage in pipeline.DLA_TRANSFORMS if stage[0] != "RemoveQDQ"),
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["X"], ["Y"])],
            "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    before = model.SerializeToString()
    error = ValueError if len(options) == 2 else NotImplementedError
    with pytest.raises(error):
        pipeline._transform_make_dla_compatible(model, verbose=False, **options)
    assert model.SerializeToString() == before


def test_pipeline_applies_cast_reorders_nodes_and_returns_model():
    """The real pipeline dispatches stages, orders support nodes and returns executable outputs."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["K"],
                    name="constant",
                    value=helper.make_tensor("k", TensorProto.FLOAT, [2], [1.0, 2.0]),
                ),
                helper.make_node("Cast", ["X"], ["C"], name="cast", to=TensorProto.INT32),
                helper.make_node("Identity", ["C"], ["Y"], name="consumer"),
                helper.make_node("Identity", ["K"], ["Z"], name="constant_consumer"),
            ],
            "pipeline",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
            [
                helper.make_tensor_value_info("Y", TensorProto.INT32, [2]),
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model, full_check=True)
    result = pipeline._transform_make_dla_compatible(model, verbose=False)
    names = [node.name for node in result.graph.node]
    assert names.index("cast") < names.index("consumer") < names.index("constant_consumer")
    if "constant" in names:
        # Later slices turn Constants into initializers before the final reorder.
        assert names.index("consumer") < names.index("constant") < names.index("constant_consumer")
    cast = next(node for node in result.graph.node if node.name == "cast")
    assert (
        next(attribute for attribute in cast.attribute if attribute.name == "to").i
        == TensorProto.FLOAT
    )
    onnx.checker.check_model(result, full_check=True)
    x = np.array([1.5, -2.5], dtype=np.float32)
    session = ort.InferenceSession(result.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = dict(zip([value.name for value in result.graph.output], session.run(None, {"X": x})))
    np.testing.assert_array_equal(outputs["Y"], x)
    np.testing.assert_array_equal(outputs["Z"], np.array([1.0, 2.0], dtype=np.float32))
