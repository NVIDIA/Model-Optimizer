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

"""DLA graph transforms in the order recorded by the C++ reference pipeline.

DLA_TRANSFORMS is the single ordered list of supported passes. Individual reviews
can add passes alongside their focused tests. Shape inference is explicit:
inserting it between rank-conversion passes changes original shape metadata used
to adjust axes.
"""

from __future__ import annotations

import time
from functools import partial
from typing import TYPE_CHECKING

import onnx

from ..logging_config import logger
from .dla_transforms.dla_cast_to_fp32 import _apply_cast_to_fp32
from .dla_transforms.dla_constants_to_initializers import _constants_to_initializers_from_model
from .dla_transforms.dla_decompose_lstm import _apply_decompose_lstm
from .dla_transforms.dla_graph_cleanup import _remove_unused_initializers
from .dla_transforms.dla_matmul_to_transpose_conv_transpose import (
    _apply_matmul_to_transpose_conv_transpose,
)
from .dla_transforms.dla_model_boundaries import (
    _REDUCTION_OPS,
    _apply_fix_instancenorm_channels,
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
    _apply_remove_intermediary_squeeze_unsqueeze,
    _require_output_rank_conversion,
)
from .dla_transforms.dla_remove_deqlin import _apply_remove_deqlin
from .dla_transforms.dla_remove_qdq import _remove_qdq_from_model
from .dla_transforms.dla_reorder_nodes import _apply_reorder_nodes
from .dla_transforms.dla_reshape_patterns import (
    _apply_reshape_qdq_reduce_sum,
    _apply_reshape_reduce_sum,
    _apply_reshape_transpose_reshape,
    _reshape_reduce_sum,
)
from .utils.graph_utils import (
    DEFAULT_QDQ_REMOVE_QUANT_TYPES,
    infer_shapes,
    parse_qdq_quantized_dtype_list,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
__all__ = []


def _remove_deqlin_and_unused_initializers(model):
    """Fold eligible constant dequantizations and discard unused initializers."""
    return _remove_unused_initializers(_apply_remove_deqlin(model))


def _apply_remove_qdq(model, **options):
    """Run QDQ removal while retaining the model result for ordered stage dispatch."""
    _remove_qdq_from_model(model, **options)
    return model


def _apply_constants_to_initializers(model, **options):
    """Fold constants while retaining the model result for ordered stage dispatch."""
    _constants_to_initializers_from_model(model, **options)
    return model


DLA_TRANSFORMS = (
    ("RemoveQDQ", _apply_remove_qdq),
    ("ConstantsToInitializers", _apply_constants_to_initializers),
    ("Cast", _apply_cast_to_fp32),
    ("RemoveDeqLin", _remove_deqlin_and_unused_initializers),
    ("ShapeInference", infer_shapes),
    ("ReshapeTransposeReshape", _apply_reshape_transpose_reshape),
    ("ReshapeQDQReduceSum", _apply_reshape_qdq_reduce_sum),
    ("ReshapeReduceSum", _apply_reshape_reduce_sum),
    ("DecomposeLstm", partial(_apply_decompose_lstm, defer_rank_conversion=True)),
    (
        "MatMulToTransposeConvTranspose",
        partial(_apply_matmul_to_transpose_conv_transpose, defer_rank_conversion=True),
    ),
    ("Non4dModelInputs", _apply_non4d_model_inputs),
    ("Non4dModelOutputs", partial(_apply_non4d_model_outputs, defer_rank_conversion=True)),
    (
        "RemoveIntermediarySqueezeAndUnsqueeze",
        partial(_apply_remove_intermediary_squeeze_unsqueeze, rank_converted=True),
    ),
    ("FixInstanceNormChannelMismatch", _apply_fix_instancenorm_channels),
    ("ShapeInference", infer_shapes),
    ("RemoveUnusedInitializers", _remove_unused_initializers),
)


def _transform_make_dla_compatible(
    model: onnx.ModelProto,
    *,
    verbose: bool = True,
    qdq_quantized_dtypes: Iterable[int | str] | None = None,
    remove_all_qdq: bool = False,
) -> onnx.ModelProto:
    """Apply the logged DLA sequence in memory, then infer shapes and validate."""
    started = time.perf_counter()
    if remove_all_qdq and qdq_quantized_dtypes is not None:
        raise ValueError("Specify remove_all_qdq or qdq_quantized_dtypes, not both")
    if (remove_all_qdq or qdq_quantized_dtypes is not None) and not any(
        name == "RemoveQDQ" for name, _ in DLA_TRANSFORMS
    ):
        raise NotImplementedError("QDQ removal is not available in this pipeline")
    stages = frozenset(stage for stage, _ in DLA_TRANSFORMS)
    planned_reduction_outputs: set[str] = set()
    if "StandaloneReduceOperations" not in stages:
        if any(node.op_type == "ReduceSum" for node in model.graph.node):
            for stage, quantized in (("ReshapeReduceSum", False), ("ReshapeQDQReduceSum", True)):
                if stage in stages:
                    planned_reduction_outputs.update(
                        _reshape_reduce_sum(model, quantized=quantized, validate_only=True)
                    )
        local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
        for node in model.graph.node:
            if (
                node.op_type in _REDUCTION_OPS
                and node.domain in {"", "ai.onnx"}
                and not node.overload
                and (node.domain, node.op_type, node.overload) not in local_functions
                and any(attr.name == "keepdims" and attr.i == 0 for attr in node.attribute)
                and not planned_reduction_outputs.intersection(node.output)
            ):
                raise NotImplementedError(
                    "Standalone reductions are not available in this pipeline"
                )
    prepared_output_ops = stages | (
        _REDUCTION_OPS if "StandaloneReduceOperations" in stages else frozenset()
    )
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    for node in model.graph.node:
        if not planned_reduction_outputs.intersection(node.output):
            _require_output_rank_conversion(node, local_functions, prepared_output_ops)
    quant_types = (
        DEFAULT_QDQ_REMOVE_QUANT_TYPES
        if qdq_quantized_dtypes is None
        else parse_qdq_quantized_dtype_list(qdq_quantized_dtypes)
    )
    if remove_all_qdq:
        quant_types = frozenset(onnx.TensorProto.DataType.values())
    model = infer_shapes(model)
    for name, transform in DLA_TRANSFORMS:
        if verbose:
            logger.info("[DLA pipeline] Applying %s", name)
        options = {}
        if name == "RemoveQDQ":
            options = {
                "keep_clip_after_inputs": False,
                "max_chained_qdq_pairs": 5,
                "quant_types_to_remove": quant_types,
            }
        elif name == "ConstantsToInitializers":
            options = {"verbose": verbose}
        elif name == "Non4dModelOutputs":
            options = {"prepared_output_ops": prepared_output_ops}
        result = transform(model, **options)
        if result is not None:
            model = result
    model = infer_shapes(model)
    model = _apply_reorder_nodes(model)
    onnx.checker.check_model(model)
    if verbose:
        logger.info("[DLA pipeline] Done (%.1fs).", time.perf_counter() - started)
    return model
