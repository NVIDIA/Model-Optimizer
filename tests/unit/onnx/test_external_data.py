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

import numpy as np
import onnx
import pytest

from modelopt.onnx.utils import save_onnx


@pytest.mark.parametrize("last_byte", [0x00, 0x0A, 0x1A])
def test_save_external_data_preserves_tensor_boundaries(tmp_path, last_byte):
    # Windows text-mode file opening can truncate a trailing Ctrl-Z before append.
    first = np.full(1024, 0x41, dtype=np.uint8)
    first[-1] = last_byte
    second = np.full(1024, 0x42, dtype=np.uint8)
    tensors = [
        onnx.numpy_helper.from_array(value, name)
        for name, value in [("first", first), ("second", second)]
    ]
    graph = onnx.helper.make_graph([], "external_data", [], [], initializer=tensors)
    model = onnx.helper.make_model(graph)
    path = tmp_path / "model.onnx"

    save_onnx(model, str(path), save_as_external_data=True)

    assert (tmp_path / "model.onnx_data").read_bytes() == first.tobytes() + second.tobytes()
    restored = onnx.load(path)
    for actual, expected in zip(restored.graph.initializer, [first, second], strict=True):
        np.testing.assert_array_equal(onnx.numpy_helper.to_array(actual), expected)
