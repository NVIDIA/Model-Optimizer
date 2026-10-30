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

import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from examples.onnx_ptq.download_example_onnx import export_to_onnx


@pytest.mark.parametrize(
    ("dynamo_export", "onnx_opset", "expected_opset"),
    [(False, None, 20), (True, None, 23), (True, 24, 24)],
    ids=["legacy-default", "dynamo-default", "dynamo-explicit"],
)
def test_export_to_onnx_preserves_mode_and_opset(
    tmp_path, dynamo_export, onnx_opset, expected_opset
):
    model = torch.nn.Linear(2, 2).eval()
    output = tmp_path / "model.onnx"

    export_to_onnx(
        model,
        (1, 2),
        output,
        torch.device("cpu"),
        dynamo_export=dynamo_export,
        onnx_opset=onnx_opset,
    )

    exported = onnx.load(output)
    onnx.checker.check_model(exported, full_check=True)
    assert (
        next(item.version for item in exported.opset_import if item.domain == "") == expected_opset
    )
    inputs = torch.tensor([[1.0, 2.0]])
    actual = ReferenceEvaluator(exported).run(None, {exported.graph.input[0].name: inputs.numpy()})[
        0
    ]
    torch.testing.assert_close(torch.from_numpy(actual), model(inputs))
