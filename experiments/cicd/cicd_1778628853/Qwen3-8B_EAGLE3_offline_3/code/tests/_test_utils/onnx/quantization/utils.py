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

import onnx_graphsurgeon as gs


def assert_nodes_are_quantized(nodes):
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable):
                producer = node.i(inp_idx)
                # Quantized path may include a Cast right after DQ
                if producer and producer.op == "Cast":
                    producer = producer.i(0)
                assert producer and producer.op == "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                )
    return True
