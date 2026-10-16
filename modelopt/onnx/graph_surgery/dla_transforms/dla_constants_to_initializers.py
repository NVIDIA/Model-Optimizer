# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fold ``Constant`` ops into graph ``initializer`` entries."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ...logging_config import logger
from ._common import add_unique_initializers, save_model


def _tensor_from_constant_node(node: onnx.NodeProto) -> onnx.TensorProto:
    """Build a dense ``TensorProto`` initializer from a ``Constant`` node's attributes."""
    if not node.output:
        raise ValueError(f"Constant node {node.name!r} has no output")
    out_name = node.output[0]

    for attr in node.attribute:
        if attr.name == "value":
            arr = numpy_helper.to_array(attr.t)
            return numpy_helper.from_array(arr, name=out_name)
        if attr.name == "value_float":
            arr = np.asarray(attr.f, dtype=np.float32).reshape(())
            return numpy_helper.from_array(arr, name=out_name)
        if attr.name == "value_floats":
            arr = np.asarray(list(attr.floats), dtype=np.float32)
            return numpy_helper.from_array(arr, name=out_name)
        if attr.name == "value_int":
            arr = np.asarray(np.int64(attr.i), dtype=np.int64).reshape(())
            return numpy_helper.from_array(arr, name=out_name)
        if attr.name == "value_ints":
            arr = np.asarray(list(attr.ints), dtype=np.int64)
            return numpy_helper.from_array(arr, name=out_name)
        if attr.name == "sparse_value":
            sparse = attr.sparse_tensor
            values = numpy_helper.to_array(sparse.values)
            indices = numpy_helper.to_array(sparse.indices).astype(np.int64)
            dense = np.zeros(tuple(sparse.dims), dtype=values.dtype)
            if indices.ndim == 1:
                # Flat indices into the dense tensor.
                dense.reshape(-1)[indices] = values
            else:
                # Per-axis indices: shape [NNZ, rank].
                dense[tuple(indices.T)] = values
            return numpy_helper.from_array(dense, name=out_name)

    attr_names = [a.name for a in node.attribute]
    raise ValueError(
        f"Constant node {node.name!r} has no supported value attribute "
        f"(expected value, value_float(s), value_int(s), or sparse_value); got {attr_names}"
    )


def _constants_to_initializers_from_model(
    model: onnx.ModelProto,
    *,
    verbose: bool,
) -> int:
    graph = model.graph
    unique: set[str] = {i.name for i in graph.initializer}
    nodes_to_remove: list[onnx.NodeProto] = []

    for node in list(graph.node):
        if node.op_type != "Constant":
            continue
        if not node.output:
            continue
        try:
            init = _tensor_from_constant_node(node)
        except (ValueError, TypeError, AttributeError) as exc:
            if verbose:
                logger.warning("%s", exc)
            continue
        before = len(unique)
        add_unique_initializers(graph, unique, [init])
        if len(unique) > before:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.node.remove(node)

    if verbose:
        logger.debug("Converted %d Constant nodes to initializers", len(nodes_to_remove))
    return len(nodes_to_remove)


def constants_to_initializers(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
) -> onnx.ModelProto:
    """Replace ``Constant`` nodes with graph initializers carrying the same data.

    Supported ``Constant`` payload attributes: ``value`` (tensor), ``value_float``,
    ``value_floats``, ``value_int``, ``value_ints``, and ``sparse_value`` (densified).

    Args:
        model_path: Input ONNX model.
        output_path: Path for the transformed model.
        use_external_data: Whether to save weights as external data when large.
        external_data_name: External data filename (default: ``<output_basename>_data``).
        verbose: Log progress and skipped nodes.

    Returns:
        Updated ``ModelProto`` (also written to ``output_path``).
    """
    if verbose:
        logger.debug("Loading model from: %s", model_path)

    model = onnx.load(model_path, load_external_data=True)
    _constants_to_initializers_from_model(model, verbose=verbose)

    save_model(
        model,
        output_path,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
    )

    if verbose:
        logger.debug("constants_to_initializers finished: %s", output_path)

    return model


def transform_constants_to_initializers(
    model: onnx.ModelProto,
    verbose: bool = False,
) -> onnx.ModelProto:
    """Olive-style in-place API; same core as :func:`constants_to_initializers` without I/O."""
    _constants_to_initializers_from_model(model, verbose=verbose)
    return model
