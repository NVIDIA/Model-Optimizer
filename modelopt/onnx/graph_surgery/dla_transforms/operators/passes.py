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

"""Graph-wide type binding needed by logical consumer conversion."""

import onnx
from onnxscript import ir

__all__ = ["FloatConsumerViews"]


class FloatConsumerViews(ir.passes.InPlacePass):
    """Cast promoted integer Unsqueeze outputs only at proven FP32 consumers.

    ONNX schema type variables determine which consumer ports require FLOAT.
    Shared owners retain their integer data/control views; one private Cast is
    reused by all selected consumers, including arbitrary standard operators.
    """

    def call(self, model):
        """Insert shared FLOAT casts at schema-proven consumer ports."""
        selected = {}
        version = model.graph.opset_imports[""]
        for node in model.graph:
            if node.domain or node.op_type == "Cast":
                continue
            try:
                schema = onnx.defs.get_schema(node.op_type, version, "")
            except onnx.defs.SchemaError:
                continue
            constraints = {c.type_param_str: c.allowed_type_strs for c in schema.type_constraints}
            for port, value in enumerate(node.inputs):
                producer = value.producer() if value is not None else None
                if (
                    producer is None
                    or producer.domain
                    or producer.op_type != "Unsqueeze"
                    or value.dtype not in (ir.DataType.INT32, ir.DataType.INT64)
                ):
                    continue
                formal = schema.inputs[min(port, len(schema.inputs) - 1)].type_str
                binding = [
                    other.dtype
                    for i, other in enumerate(node.inputs)
                    if other is not None
                    and other is not value
                    and schema.inputs[min(i, len(schema.inputs) - 1)].type_str == formal
                ]
                binding.extend(
                    other.dtype
                    for i, other in enumerate(node.outputs)
                    if schema.outputs[min(i, len(schema.outputs) - 1)].type_str == formal
                    and other.dtype is not None
                )
                if formal == "tensor(float)" or (
                    "tensor(float)" in constraints.get(formal, [])
                    and binding
                    and all(d == ir.DataType.FLOAT for d in binding)
                ):
                    selected.setdefault(node, []).append(port)
        views = {}
        for node, ports in selected.items():
            for port in ports:
                source = node.inputs[port]
                if source not in views:
                    cast = ir.Node(
                        "", "Cast", [source], [ir.AttrInt64("to", int(ir.DataType.FLOAT))]
                    )
                    value = cast.outputs[0]
                    value.type, value.shape = ir.TensorType(ir.DataType.FLOAT), source.shape
                    model.graph.insert_before(node, cast)
                    views[source] = value
                node.replace_input_with(port, views[source])
        self.transformed_nodes = len(selected)
        return ir.passes.PassResult(model=model, modified=bool(selected))
