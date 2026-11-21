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

"""Analyze connected schema type regions and commit their approved metadata."""

from collections import defaultdict

import onnx
from onnxscript import ir

__all__ = ["AnalyzeFp32Casts", "PromoteFp32Values"]


class AnalyzeFp32Casts(ir.passes.InPlacePass):
    """Find Cast-connected regions that can use FLOAT without changing fixed controls."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Compute eligible Cast roots and propagated values without mutating the graph."""
        graph = model.graph
        candidates = {
            node.outputs[0]
            for node in graph
            if not node.domain
            and node.op_type == "Cast"
            and not node.overload
            and node.attributes.get_int("to") != int(ir.DataType.FLOAT)
        }
        if not candidates:
            self.candidates, self.promoted = set(), set()
            return ir.passes.PassResult(model, modified=False)
        adjacent = defaultdict(set)
        blocked = {
            value
            for value in (*graph.inputs, *graph.initializers.values())
            if value.dtype != ir.DataType.FLOAT
        }
        for node in graph:
            values = [value for value in (*node.inputs, *node.outputs) if value is not None]
            try:
                schema = onnx.defs.get_schema(
                    node.op_type, graph.opset_imports[node.domain], node.domain
                )
            except (onnx.defs.SchemaError, KeyError):
                blocked.update(values)
                continue
            if node.overload or any(
                parameter.option == onnx.defs.OpSchema.FormalParameterOption.Variadic
                and not parameter.is_homogeneous
                for parameter in (*schema.inputs, *schema.outputs)
            ):
                blocked.update(values)
                continue
            groups = defaultdict(list)
            allowed = {
                constraint.type_param_str: constraint.allowed_type_strs
                for constraint in schema.type_constraints
            }
            for actual, formal in ((node.inputs, schema.inputs), (node.outputs, schema.outputs)):
                for index, value in enumerate(actual):
                    if value is None:
                        continue
                    parameter = formal[min(index, len(formal) - 1)]
                    groups[parameter.type_str].append(value)
                    if "tensor(float)" not in allowed.get(parameter.type_str, [parameter.type_str]):
                        blocked.add(value)
            for group in groups.values():
                for value in group:
                    adjacent[value].update(group)
            if node.op_type == "Cast" and node.inputs[0].producer() is not None:
                producer = node.inputs[0].producer()
                if producer.op_type == "TopK" and node.inputs[0] is producer.outputs[1]:
                    blocked.add(node.outputs[0])
        unavailable = set(blocked)
        pending = list(blocked)
        while pending:
            value = pending.pop()
            for neighbor in adjacent[value] - unavailable:
                unavailable.add(neighbor)
                pending.append(neighbor)
        candidates -= unavailable
        promoted = set(candidates)
        pending = list(candidates)
        while pending:
            value = pending.pop()
            for neighbor in adjacent[value] - promoted:
                promoted.add(neighbor)
                pending.append(neighbor)

        self.candidates, self.promoted = candidates, promoted
        return ir.passes.PassResult(model, modified=False)


class PromoteFp32Values(ir.passes.InPlacePass):
    """Commit FLOAT metadata after the approved Cast operations have been rewritten."""

    def __init__(self, analysis: AnalyzeFp32Casts):
        """Bind the completed analysis whose approved Cast roots have been rewritten."""
        self.analysis = analysis

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Set the approved connected values to FLOAT and report actual metadata changes."""
        modified = any(value.dtype != ir.DataType.FLOAT for value in self.analysis.promoted)
        for value in self.analysis.promoted:
            value.type = ir.TensorType(ir.DataType.FLOAT)
        return ir.passes.PassResult(model, modified=modified)
