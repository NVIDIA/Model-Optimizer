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

"""Fixed rules for quantization removal, constants and FP32 preparation."""

from collections.abc import Collection
from typing import TYPE_CHECKING

import numpy as np
from onnxscript import ir
from onnxscript.rewriter import pattern

from ..common import constant_array
from ..operators.rules import weight_eligible

if TYPE_CHECKING:
    from ..context import DlaGraph
    from .passes import AnalyzeFp32Casts

__all__ = ["ConstantInitializer", "Fp32Cast", "SelectedQdq", "StaticDequantization"]


class SelectedQdq(pattern.RewriteRuleClassBase):
    """Bypass selected Q/DQ branches while retaining shared quantizers.

    For example, ``x -> Q(INT16) -> DQ -> y`` becomes ``x -> y``; a public
    endpoint keeps its name through an Identity or a dtype-preserving Cast.
    Standard and Microsoft domains, optional zero points and explicit storage
    dtype attributes are supported. Other users of Q remain connected to it.
    """

    def __init__(
        self,
        graph: "DlaGraph",
        domain: str = "",
        *,
        quantized_dtypes: Collection[int] | None = None,
        remove_all_qdq: bool = False,
        source_names: dict[ir.Value, str] | None = None,
    ):
        """Bind the storage selection, domain and shared endpoint-name restoration table."""
        super().__init__(name="DlaSelectedQdq")
        self.domain = domain
        self.selected = (
            {ir.DataType.INT16, ir.DataType.UINT16}
            if quantized_dtypes is None
            else set(quantized_dtypes)
        )
        if remove_all_qdq:
            self.selected = set(ir.DataType)
        self.source_names = source_names if source_names is not None else {}

    def pattern(self, op, quantized, scale):
        """Match a DQ branch with either an explicit or omitted zero point."""
        return op.DequantizeLinear(
            quantized, scale, _domain=self.domain, _allow_other_inputs=True, _outputs=["value"]
        )

    def check(self, context, quantized, scale, value):
        """Require a selected Q storage dtype and known floating endpoint types."""
        quantizer = quantized.producer()
        if (
            quantizer is None
            or quantizer.op_type != "QuantizeLinear"
            or quantizer.domain not in {"", "com.microsoft"}
            or quantizer.overload
            or context.root.overload
        ):
            return False
        storage = quantizer.attributes.get_int("output_dtype", 0) or (
            quantizer.inputs[2].dtype
            if len(quantizer.inputs) > 2 and quantizer.inputs[2] is not None
            else ir.DataType.UINT8
        )
        if storage not in self.selected:
            return False
        if quantizer.inputs[0].dtype is None or value.dtype is None:
            raise ValueError("Infer Q/DQ endpoint types before DLA preprocessing")
        return True

    def rewrite(self, op, quantized, scale, value):
        """Reconnect the original floating source while preserving endpoint names and types."""
        source = quantized.producer().inputs[0]
        if source.dtype != value.dtype:
            return op.Cast(source, to=int(value.dtype))
        if value.is_graph_output():
            if source.producer() is None or source.is_graph_output():
                return op.Identity(source)
            self.source_names[source] = value.name
            return source
        self.source_names.setdefault(source, source.name)
        return source


class ConstantInitializer(pattern.RewriteRuleClassBase):
    """Replace dense Constant nodes with typed graph initializers.

    Tensor, float/int scalar and float/int list attributes retain their storage
    dtype and rank: ``Constant(value_float=1.25)`` becomes an FP32 scalar
    initializer. Sparse and string attribute forms require separate support.
    """

    def __init__(self, graph: "DlaGraph"):
        """Create a fresh constant-materialization rule for this conversion."""
        super().__init__(name="DlaConstantInitializer")

    def pattern(self, op):
        """Match a Constant regardless of its dense attribute representation."""
        return op.Constant(_outputs=["value"])

    def check(self, context, value):
        """Require a supported dense tensor, numeric scalar or numeric vector attribute."""
        if not (
            {"value", "value_float", "value_floats", "value_int", "value_ints"}
            & context.root.attributes.keys()
        ):
            raise NotImplementedError("This Constant form needs an additional DLA transformation")
        return True

    def rewrite(self, op, value):
        """Materialize the same tensor storage type and scalar or vector rank."""
        attributes = value.producer().attributes
        if "value" in attributes:
            tensor = attributes["value"].value
        else:
            name = next(iter(attributes))
            dtype = np.float32 if name.startswith("value_float") else np.int64
            tensor = ir.tensor(np.asarray(attributes[name].value, dtype=dtype))
        return op.initializer(tensor, name=value.name)


class Fp32Cast(pattern.RewriteRuleClassBase):
    """Promote Casts approved by the connected schema type analysis to FP32.

    For example, ``Cast(INT32) -> Abs`` can become ``Cast(FLOAT) -> Abs`` when
    the whole type-connected region admits FLOAT. Integer shape controls,
    TopK indices and regions tied to fixed integer operands stay unchanged.
    The companion metadata pass updates the remaining approved values.
    """

    def __init__(self, graph: "DlaGraph", analysis: "AnalyzeFp32Casts"):
        """Bind the schema-connected analysis computed before this rule batch."""
        super().__init__(name="DlaFp32Cast")
        self.analysis = analysis

    def pattern(self, op, source):
        """Match an explicit Cast without assuming its original target dtype."""
        return op.Cast(source, _outputs=["value"])

    def check(self, context, source, value):
        """Accept only Cast results approved by the connected type analysis."""
        return value in self.analysis.candidates

    def rewrite(self, op, source, value):
        """Emit a FLOAT Cast and set metadata inherited by its replacement output."""
        value.type = ir.TensorType(ir.DataType.FLOAT)
        return op.Cast(source, to=int(ir.DataType.FLOAT))


class StaticDequantization(pattern.RewriteRuleClassBase):
    """Fold immutable scalar or per-axis DequantizeLinear into FP32 weights.

    ``DQ(int8_initializer, scale, zero) -> Add`` becomes an FP32 initializer
    with the same dequantized contents. Standard and Microsoft DQ, omitted
    zero points and INT8/UINT8/INT16/UINT16/INT32 storage are supported.
    Eligible quantized convolution/matrix weights remain explicit for their
    later lowering. Runtime controls, blocked/packed values and non-FP32
    outputs remain operations; equivalent fixed DQ expressions share storage.
    """

    def __init__(self, graph: "DlaGraph", domain: str = "", *, folded=None):
        """Bind one supported DQ domain and the batch-shared folded-initializer cache."""
        super().__init__(name="DlaStaticDequantization")
        self.graph = graph
        self.domain = domain
        self.folded = folded if folded is not None else {}

    def pattern(self, op, data, scale):
        """Match DQ with optional zero-point controls captured from its root."""
        return op.DequantizeLinear(
            data, scale, _domain=self.domain, _allow_other_inputs=True, _outputs=["value"]
        )

    def check(self, context, data, scale, value):
        """Require immutable FP32 dequantization and preserve later weight-lowering ownership."""
        node = context.root
        arrays = _static_data(node)
        if arrays is None:
            return False
        for consumer, port in value.uses():
            if port != 1 or consumer.domain:
                continue
            if consumer.op_type in {"Conv", "ConvTranspose"} and data.dtype in {
                ir.DataType.INT8,
                ir.DataType.UINT8,
            }:
                return False
            if (
                consumer.op_type in {"MatMul", "Gemm"}
                and data.dtype in {ir.DataType.INT8, ir.DataType.UINT8}
                and weight_eligible(consumer, value)
            ):
                return False
        _broadcast_data(node, arrays)
        return True

    def rewrite(self, op, data, scale, value):
        """Evaluate fixed DQ arithmetic with bounded NumPy ufunc temporaries."""
        node = value.producer()
        key = (
            tuple(node.inputs),
            tuple((key, attr.value) for key, attr in node.attributes.items()),
        )
        if key not in self.folded:
            raw, factors, zeros = _broadcast_data(node, _static_data(node))
            if raw.dtype == np.int32:
                difference = np.subtract(raw, zeros, dtype=np.int32)
                result = np.empty(raw.shape, np.float32)
                np.multiply(difference, factors, out=result, dtype=np.float64, casting="unsafe")
            else:
                result = raw.astype(np.float32)
                np.subtract(result, zeros.astype(np.float32), out=result)
                np.multiply(result, factors, out=result)
            self.folded[key] = self.graph.add_initializer(op, result, data.name + "_dequantized")
        replacement = self.folded[key]
        return op.Identity(replacement) if value.is_graph_output() else replacement


def _static_data(node):
    data, scale = node.inputs[:2]
    raw, factors = constant_array(data), constant_array(scale)
    zero = node.inputs[2] if len(node.inputs) > 2 else None
    zeros = constant_array(zero) if zero is not None else np.asarray(0, np.int32)
    if (
        node.overload
        or node.attributes.get_int("block_size", 0)
        or raw is None
        or factors is None
        or zeros is None
        or data.dtype
        not in {
            ir.DataType.INT8,
            ir.DataType.UINT8,
            ir.DataType.INT16,
            ir.DataType.UINT16,
            ir.DataType.INT32,
        }
        or factors.dtype != np.float32
        or node.outputs[0].dtype != ir.DataType.FLOAT
    ):
        return None
    return raw, factors, zeros


def _broadcast_data(node, arrays):
    raw, factors, zeros = arrays
    if factors.size > 1 or zeros.size > 1:
        axis = node.attributes.get_int("axis", 1) % raw.ndim
        shape = [1] * raw.ndim
        shape[axis] = -1
        factors, zeros = factors.reshape(shape), zeros.reshape(shape)
    else:
        factors, zeros = factors.reshape(()), zeros.reshape(())
    return raw, factors, zeros
