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

import pytest
from onnx import TensorProto, helper

pipeline = importlib.import_module("modelopt.onnx.graph_surgery.make_dla_compatible")


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
