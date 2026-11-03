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

"""Retain required FP32 regions and prepare FP16 storage and graph boundaries."""

import numpy as np
import onnx
from onnxscript import GraphBuilder, ir

from ..common import static_shape

__all__ = ["CastFp16Boundaries", "PrepareFp16"]


class PrepareFp16(ir.passes.InPlacePass):
    """Analyze native FP32 regions, convert eligible storage and protect public outputs.

    The retained values and shared boundary-cast cache are reused after the
    activation Cast rule runs. Schema-mandated FP32 ports, unknown native ops,
    public input adapters and integer-fed terminal Concats keep their policy.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Prepare initializer storage, output boundaries and internal value dtype metadata."""
        self.count = 0
        graph = model.graph
        originals = tuple(graph)
        retained, native, required = fp32_regions(model)
        public = set(graph.outputs)
        adapters, terminal_concats = set(), set()
        for node in originals:
            if (
                node.op_type == "Unsqueeze"
                and node.inputs[0].is_graph_input()
                and node.inputs[0].dtype == ir.DataType.FLOAT
                and node not in native
            ):
                shape, output = static_shape(node.inputs[0]), static_shape(node.outputs[0])
                if shape is not None and len(shape) < 4 and output is not None and len(output) == 4:
                    adapters.add(node)
                    retained.add(node.outputs[0])
            if node.op_type == "Squeeze" and node.outputs[0] in public:
                concat = node.inputs[0].producer()
                if (
                    concat is not None
                    and concat.op_type == "Concat"
                    and len(concat.outputs[0].uses()) == 1
                    and any(
                        value.producer() is not None
                        and value.producer().op_type == "Cast"
                        and value.producer().inputs[0].dtype
                        not in {ir.DataType.FLOAT, ir.DataType.FLOAT16, None}
                        for value in concat.inputs
                    )
                ):
                    terminal_concats.add(concat)
                    retained.add(concat.outputs[0])
        preserved_initializers = {
            value
            for value in graph.initializers.values()
            if value.is_graph_input()
            or value in retained
            or any(
                consumer in native or (consumer, port) in required
                for consumer, port in value.uses()
            )
        }
        self.builder, self.casts = GraphBuilder(graph), {}
        used_names = {
            value.name for value in (*graph.inputs, *graph.outputs, *graph.initializers.values())
        }
        used_names.update(value.name for node in originals for value in node.outputs)

        def internal_name(name):
            """Allocate an internal endpoint name without replacing an existing owner."""
            candidate, suffix = name + "_fp16", 0
            while candidate in used_names:
                suffix += 1
                candidate = f"{name}_fp16_{suffix}"
            used_names.add(candidate)
            return candidate

        for index, value in enumerate(tuple(graph.outputs)):
            producer = value.producer()
            if value.dtype != ir.DataType.FLOAT or value in retained or value.is_graph_input():
                continue
            if producer is not None and producer.op_type in {"Squeeze", "Cast"}:
                continue
            name = value.name
            value.name = internal_name(name)
            result = self.cast(value, ir.DataType.FLOAT)
            result.name = name
            graph.outputs[index] = result
        for value in graph.initializers.values():
            if value.dtype == ir.DataType.FLOAT and value not in preserved_initializers:
                # Avoid infinities from overflow turning zero masks into NaNs.
                value.const_value = ir.tensor(_fp16_array(value.const_value.numpy()))
                value.type = ir.TensorType(ir.DataType.FLOAT16)
                self.count += 1
        metadata_changed = False
        for node in originals:
            for value in node.outputs:
                if (
                    value.dtype == ir.DataType.FLOAT
                    and value not in retained
                    and not value.is_graph_output()
                ):
                    value.type = ir.TensorType(ir.DataType.FLOAT16)
                    metadata_changed = True

        self.originals = originals
        self.retained, self.native, self.required = retained, native, required
        self.adapters, self.terminal_concats = adapters, terminal_concats
        return ir.passes.PassResult(model, modified=bool(self.count) or metadata_changed)

    def cast(self, value: ir.Value, dtype: ir.DataType) -> ir.Value:
        """Share a mechanical cast between readers requiring the same operand dtype."""
        key = value, dtype
        if key not in self.casts:
            result = self.builder.op.Cast(value, to=int(dtype))
            result.type, result.shape = ir.TensorType(dtype), value.shape
            self.casts[key] = result
            self.count += 1
        return self.casts[key]


class CastFp16Boundaries(ir.passes.InPlacePass):
    """Adapt original consumers after storage and explicit activation Casts change dtype."""

    def __init__(self, preparation: PrepareFp16):
        """Reuse the preparation analysis and its cache of shared boundary casts."""
        self.preparation = preparation
        self.count = 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Reconnect consumers to operands with their required FP32 or FP16 dtype."""
        preparation = self.preparation
        before = preparation.count
        modified = False
        for node in preparation.originals:
            if node.graph is None or (
                node.op_type == "Cast" and node.attributes.get_int("to") == int(ir.DataType.FLOAT16)
            ):
                continue
            for port, value in enumerate(tuple(node.inputs)):
                if value is None or value.dtype not in {ir.DataType.FLOAT, ir.DataType.FLOAT16}:
                    continue
                dtype = (
                    ir.DataType.FLOAT
                    if (
                        node in preparation.native
                        or (node, port) in preparation.required
                        or node in preparation.adapters
                        or node in preparation.terminal_concats
                        or (node.op_type == "Squeeze" and node.outputs[0].is_graph_output())
                    )
                    else ir.DataType.FLOAT16
                )
                if value.dtype != dtype:
                    node.replace_input_with(port, preparation.cast(value, dtype))
                    modified = True

        self.count = preparation.count - before
        return ir.passes.PassResult(model, modified=modified)


def _fp16_array(array: np.ndarray) -> np.ndarray:
    """Clamp finite overflow without copying the full FP32 tensor."""
    result = np.empty_like(array, dtype=np.float16, order="K", subok=False)
    limit = np.finfo(np.float16).max
    with np.nditer(
        [array, result],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["writeonly"]],
        order="K",
        buffersize=65536,
    ) as chunks:
        for source, target in chunks:
            bounded = source.copy()
            np.clip(bounded, -limit, limit, out=bounded, where=np.isfinite(bounded))
            target[...] = bounded
    return result


def fp32_regions(model: ir.Model) -> tuple[set[ir.Value], set[ir.Node], set[tuple[ir.Node, int]]]:
    """Retain native regions and schema-mandated FP32 operand dependencies."""
    native, required, retained, pending = set(), set(), set(), []
    version = model.graph.opset_imports[""]
    for node in model.graph:
        domain, schema_version = node.domain, version
        if domain == "com.microsoft" and node.op_type in {
            "QuantizeLinear",
            "DequantizeLinear",
            "Gelu",
        }:
            domain = ""
            schema_version = 20 if node.op_type == "Gelu" else 19
        try:
            schema = onnx.defs.get_schema(node.op_type, schema_version, domain)
        except onnx.defs.SchemaError:
            schema = None
        values = [value for value in (*node.inputs, *node.outputs) if value is not None]
        if (
            schema is None
            or node.overload
            or any(not isinstance(value.type, ir.TensorType) for value in values)
        ):
            native.add(node)
            pending.extend(values)
            continue
        only_float = {
            item.type_param_str
            for item in schema.type_constraints
            if "tensor(float)" in item.allowed_type_strs
            and "tensor(float16)" not in item.allowed_type_strs
        } | {"tensor(float)"}
        for port, value in enumerate(node.inputs):
            if value is None:
                continue
            parameter = schema.inputs[min(port, len(schema.inputs) - 1)]
            if parameter.type_str in only_float or (node.op_type == "Resize" and port in (1, 2)):
                required.add((node, port))
                pending.append(value)
        generated_float = node.attributes.get_int("dtype", 0) == int(ir.DataType.FLOAT) or (
            node.op_type != "Cast"
            and not any(
                value.dtype == ir.DataType.FLOAT for value in node.inputs if value is not None
            )
        )
        for port, value in enumerate(node.outputs):
            parameter = schema.outputs[min(port, len(schema.outputs) - 1)]
            if value.dtype == ir.DataType.FLOAT and (
                generated_float or parameter.type_str in only_float
            ):
                retained.add(value)
    visited = set()
    while pending:
        value = pending.pop()
        if value in visited or value.dtype != ir.DataType.FLOAT:
            continue
        visited.add(value)
        producer = value.producer()
        if producer is not None:
            native.add(producer)
            pending.extend(
                value for value in (*producer.inputs, *producer.outputs) if value is not None
            )
    return retained | visited, native, required
