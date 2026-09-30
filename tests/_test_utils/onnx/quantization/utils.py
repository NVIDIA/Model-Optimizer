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


def assert_nodes_are_quantized(nodes, *, ignore_identity_inputs: bool = False):
    """Assert every variable input of ``nodes`` is produced by a DequantizeLinear.

    ``ignore_identity_inputs`` skips inputs fed by an ``Identity`` node, for graphs where the
    quantizer legitimately leaves such a passthrough in place (e.g. concat elimination).
    """
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable):
                producer = node.i(inp_idx)
                if ignore_identity_inputs and producer and producer.op == "Identity":
                    continue
                # Quantized path may include a Cast right after DQ
                if producer and producer.op == "Cast":
                    producer = producer.i(0)
                assert producer and producer.op == "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                )
    return True
