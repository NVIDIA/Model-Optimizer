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

"""Finish fixed layout chains and deduplicate immutable initializer storage."""

import hashlib
from collections.abc import Collection, Iterator
from itertools import zip_longest
from typing import TYPE_CHECKING

import numpy as np
from onnxscript import ir
from onnxscript.rewriter import pattern

from .common import constant_array, static_shape

if TYPE_CHECKING:
    from .context import DlaGraph

__all__ = [
    "DeduplicateInitializers",
    "IdentityReshape",
    "InternalIdentity",
    "ReshapeChain",
    "RestoreValueNames",
    "deduplicate_initializers",
]


class _BypassRule(pattern.RewriteRuleClassBase):
    def __init__(self, graph: "DlaGraph", source_names: dict[ir.Value, str]):
        super().__init__(name="Dla" + type(self).__name__)
        self.graph = graph
        self.source_names = source_names

    def rewrite(self, op, source, value, **_):
        self.source_names.setdefault(source, source.name)
        return op.Identity(source) if value.is_graph_output() else source


class ReshapeChain(_BypassRule):
    """Collapse two fixed adjacent Reshapes with a private intermediate tensor.

    For example, ``[2,3] -> [3,2] -> [1,6]`` becomes one Reshape to ``[1,6]``.
    Immutable shape controls and positive concrete endpoint dimensions are
    required. Equal endpoint shapes bypass the views; public names survive.
    """

    def pattern(self, op, source, first_shape, final_shape):
        """Match two adjacent Reshapes with a private intermediate output."""
        return op.Reshape(op.Reshape(source, first_shape), final_shape, _outputs=["value"])

    def check(self, context, source, value, **controls):
        """Require immutable controls and concrete positive endpoint dimensions."""
        return _fixed_reshape(source, value, controls)

    def rewrite(self, op, source, value, **controls):
        """Use the final inferred shape or bypass shape-preserving endpoint views."""
        if static_shape(source) == static_shape(value):
            return super().rewrite(op, source, value)
        return op.Reshape(
            source,
            self.graph.add_initializer(
                op, np.asarray(static_shape(value), np.int64), "reshape_shape"
            ),
        )


class IdentityReshape(_BypassRule):
    """Bypass a fixed Reshape whose input and output dimensions are identical.

    ``Reshape(x[2,3], [2,3])`` becomes ``x`` for internal consumers, or an
    Identity when the public output needs to retain its own name. Dynamic,
    zero-sized and negative endpoint dimensions are outside this rule.
    """

    def pattern(self, op, source, shape):
        """Match a single Reshape and its shape control."""
        return op.Reshape(source, shape, _outputs=["value"])

    def check(self, context, source, value, **controls):
        """Require identical fixed input and output dimensions before bypassing."""
        return _fixed_reshape(source, value, controls) and static_shape(source) == static_shape(
            value
        )


class InternalIdentity(_BypassRule):
    """Remove internal Identity nodes except explicitly preserved incoming nodes.

    ``x -> Identity -> Add`` reconnects Add directly to x. Public Identity
    outputs remain the owners of their interface names; the caller may retain
    preexisting Identity nodes while removing those emitted by other rules.
    """

    def __init__(
        self,
        graph: "DlaGraph",
        source_names: dict[ir.Value, str],
        preserve_identities: Collection[ir.Node] | None = None,
    ):
        """Bind shared name restoration and the caller's preserved incoming identities."""
        super().__init__(graph, source_names)
        self.preserved = set(preserve_identities or ())

    def pattern(self, op, source):
        """Match an Identity and expose its result for public-interface checks."""
        return op.Identity(source, _outputs=["value"])

    def check(self, context, source, value):
        """Allow removal only for internal results outside the preserved node set."""
        return context.root not in self.preserved and not value.is_graph_output()


class RestoreValueNames(ir.passes.InPlacePass):
    """Restore saved names after the SDK aliases replacement outputs to existing values."""

    def __init__(self, names: dict[ir.Value, str]):
        """Bind the names saved by one ordered rewrite batch."""
        self.names = names

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Restore saved endpoint names after SDK replacement aliasing."""
        modified = any(value.name != name for value, name in self.names.items())
        for value, name in self.names.items():
            value.name = name
        return ir.passes.PassResult(model, modified=modified)


class DeduplicateInitializers(ir.passes.InPlacePass):
    """Merge exact fixed payloads with bounded reads, preserving public initializers."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Deduplicate fixed initializer payloads and expose the removed initializer count."""
        self.removed_count = deduplicate_initializers(model)
        return ir.passes.PassResult(model, modified=self.removed_count > 0)


def _fixed_reshape(source, value, controls):
    start, end = static_shape(source), static_shape(value)
    return (
        start is not None
        and end is not None
        and bool(start)
        and bool(end)
        and min((*start, *end)) > 0
        and all(constant_array(control) is not None for control in controls.values())
    )


def deduplicate_initializers(model: ir.Model) -> int:
    """Merge fixed initializers with identical dtype, shape and exact contents."""
    graph = model.graph
    candidates = {}
    count = 0
    for value in tuple(graph.initializers.values()):
        tensor = value.const_value
        if value.is_graph_input() or value.is_graph_output() or tensor is None:
            continue
        digest = hashlib.sha256()
        for chunk in _tensor_chunks(tensor):
            digest.update(chunk)
        key = (tensor.dtype, tuple(tensor.shape), digest.digest())
        bucket = candidates.setdefault(key, [])
        for existing in bucket:
            if all(
                left == right
                for left, right in zip_longest(
                    _tensor_chunks(existing.const_value), _tensor_chunks(tensor)
                )
            ):
                value.replace_all_uses_with(existing)
                del graph.initializers[value.name]
                count += 1
                break
        else:
            bucket.append(value)
    return count


def _tensor_chunks(tensor: ir.TensorProtocol) -> Iterator[bytes]:
    """Read logical C-order bytes without making a full-sized contiguous copy."""
    if isinstance(tensor, ir.StringTensor):
        for data in tensor.string_data():
            yield len(data).to_bytes(8, "little")
            for offset in range(0, len(data), 65536):
                yield data[offset : offset + 65536]
    else:
        array = tensor.numpy()
        for chunk in np.nditer(
            array,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            order="C",
            buffersize=max(1, 65536 // array.dtype.itemsize),
        ):
            yield chunk.tobytes()
