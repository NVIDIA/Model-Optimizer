# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import onnx_graphsurgeon as gs


def _assert_tensors_are_fp16(model: onnx.ModelProto):
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        for tensor in node.inputs + node.outputs:
            assert tensor.dtype == "float16", (
                f"Tensor '{tensor.name}' in node '{node.name}' has type '{tensor.dtype}' instead of 'float16'!"
            )
    return True
