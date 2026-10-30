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

import copy
import time
from functools import partial
from typing import TYPE_CHECKING

import onnx

from ..logging_config import logger
from .dla_transforms.dla_cast_to_fp32 import _apply_cast_to_fp32
from .dla_transforms.dla_constants_to_initializers import _constants_to_initializers_from_model
from .dla_transforms.dla_decompose_lstm import _apply_decompose_lstm
from .dla_transforms.dla_graph_cleanup import _remove_unused_initializers
from .dla_transforms.dla_greater import _apply_greater
from .dla_transforms.dla_indexing import (
    _apply_gather,
    _apply_gather_elements,
    _apply_group_query_attention,
    _apply_standalone_reduce_operations,
)
from .dla_transforms.dla_matmul_to_transpose_conv_transpose import (
    _apply_matmul_to_transpose_conv_transpose,
)
from .dla_transforms.dla_model_boundaries import (
    _REDUCTION_OPS,
    _apply_fix_instancenorm_channels,
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
    _apply_remove_intermediary_squeeze_unsqueeze,
    _BoundaryPreflightError,
    _require_output_rank_conversion,
)
from .dla_transforms.dla_not import _apply_not
from .dla_transforms.dla_rank_conversion import (
    _apply_argmax,
    _apply_flatten,
    _apply_non4d_expand,
    _apply_non4d_initializers,
    _apply_non4d_lpnorm,
    _apply_non4d_reshape,
    _apply_non4d_slice_axis,
    _apply_non4d_tile,
    _apply_non4d_transpose,
    _apply_update_axis,
)
from .dla_transforms.dla_remove_deqlin import _apply_remove_deqlin
from .dla_transforms.dla_remove_qdq import _remove_qdq_from_model
from .dla_transforms.dla_reorder_nodes import _apply_reorder_nodes
from .dla_transforms.dla_reshape_patterns import (
    _apply_reshape_qdq_reduce_sum,
    _apply_reshape_reduce_sum,
    _apply_reshape_transpose_reshape,
)
from .dla_transforms.dla_topk import _apply_topk
from .dla_transforms.dla_unsqueeze import _apply_unsqueeze
from .dla_transforms.dla_where import _apply_where
from .utils.graph_utils import (
    DEFAULT_QDQ_REMOVE_QUANT_TYPES,
    GraphCache,
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
    ("Non4dSliceAxis", _apply_non4d_slice_axis),
    ("Gather", _apply_gather),
    ("DecomposeLstm", partial(_apply_decompose_lstm, defer_rank_conversion=True)),
    (
        "MatMulToTransposeConvTranspose",
        partial(_apply_matmul_to_transpose_conv_transpose, defer_rank_conversion=True),
    ),
    ("Non4dModelInputs", _apply_non4d_model_inputs),
    ("Non4dModelOutputs", partial(_apply_non4d_model_outputs, defer_rank_conversion=True)),
    ("Non4DReshape", _apply_non4d_reshape),
    ("ArgMax", _apply_argmax),
    ("StandaloneReduceOperations", _apply_standalone_reduce_operations),
    (
        "RemoveIntermediarySqueezeAndUnsqueeze",
        partial(_apply_remove_intermediary_squeeze_unsqueeze, rank_converted=True),
    ),
    ("GroupQueryAttention", _apply_group_query_attention),
    ("GatherElements", _apply_gather_elements),
    ("Non4dInitializers", _apply_non4d_initializers),
    ("Non4DExpand", _apply_non4d_expand),
    ("Non4DTile", _apply_non4d_tile),
    ("Flatten", _apply_flatten),
    ("Non4DTranspose", _apply_non4d_transpose),
    ("Non4DLpNorm", _apply_non4d_lpnorm),
    ("FixInstanceNormChannelMismatch", _apply_fix_instancenorm_channels),
    ("UpdateAxis", _apply_update_axis),
    ("Where", _apply_where),
    ("Not", _apply_not),
    ("Unsqueeze", _apply_unsqueeze),
    ("Greater", _apply_greater),
    ("ShapeInference", infer_shapes),
    ("TopK", partial(_apply_topk, rank_converted=True)),
    ("RemoveUnusedInitializers", _remove_unused_initializers),
)


def _boundary_stage_options(name, quant_types, verbose, prepared_input_ops, prepared_output_ops):
    """Share stage options between the prerequisite probe and the actual ordered pipeline."""
    if name == "Cast":
        return {"preserve_topk_indices": "TopK" in prepared_output_ops}
    if name in {"Gather", "GatherElements", "StandaloneReduceOperations", "GroupQueryAttention"}:
        return {"prepared_data_ops": prepared_input_ops}
    if name == "RemoveQDQ":
        return {
            "keep_clip_after_inputs": False,
            "max_chained_qdq_pairs": 5,
            "quant_types_to_remove": quant_types,
        }
    if name == "ConstantsToInitializers":
        return {"verbose": verbose}
    if name == "Non4dModelInputs":
        return {"prepared_input_ops": prepared_input_ops}
    if name == "Non4dModelOutputs":
        return {"prepared_output_ops": prepared_output_ops, "prepared_data_ops": prepared_input_ops}
    if name == "RemoveIntermediarySqueezeAndUnsqueeze":
        return {"prepared_consumer_ops": prepared_input_ops}
    if name in {
        "Non4dSliceAxis",
        "Non4DTranspose",
        "Non4DLpNorm",
        "ArgMax",
        "Flatten",
        "UpdateAxis",
        "Non4DExpand",
        "Non4DTile",
    }:
        return {"prepared_data_ops": prepared_input_ops}
    if name == "FixInstanceNormChannelMismatch":
        return {"prepared_consumer_ops": prepared_input_ops}
    return {}


def _validate_boundary_contract(
    model, prepared_input_ops, prepared_output_ops, *, require_inferred_shapes=False
):
    """Refuse unsupported native layouts before adapting any public boundary."""
    if "GatherElements" in prepared_input_ops:
        _apply_gather_elements(model, prepared_data_ops=prepared_input_ops, validate_only=True)
    if "StandaloneReduceOperations" in prepared_input_ops:
        _apply_standalone_reduce_operations(
            model, prepared_data_ops=prepared_input_ops, validate_only=True
        )
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    cache = GraphCache(model.graph)
    for node in model.graph.node:
        if (
            node.op_type in _REDUCTION_OPS
            and node.op_type not in prepared_output_ops
            and node.domain in {"", "ai.onnx"}
            and not node.overload
            and (node.domain, node.op_type, node.overload) not in local_functions
            and any(attr.name == "keepdims" and attr.i == 0 for attr in node.attribute)
        ):
            raise NotImplementedError("Standalone reductions are not available in this pipeline")
        _require_output_rank_conversion(node, local_functions, prepared_output_ops, cache)
    if "FixInstanceNormChannelMismatch" in prepared_input_ops:
        _apply_fix_instancenorm_channels(
            model,
            prepared_consumer_ops=prepared_input_ops,
            validate_only=True,
            require_inferred_shapes=require_inferred_shapes,
        )
    _apply_non4d_model_inputs(model, prepared_input_ops=prepared_input_ops, validate_only=True)
    _apply_non4d_model_outputs(
        model,
        defer_rank_conversion=True,
        prepared_output_ops=prepared_output_ops,
        prepared_data_ops=prepared_input_ops,
        validate_only=True,
    )
    if "RemoveIntermediarySqueezeAndUnsqueeze" in prepared_input_ops:
        _apply_remove_intermediary_squeeze_unsqueeze(
            model, rank_converted=True, prepared_consumer_ops=prepared_input_ops, validate_only=True
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
    prepared_output_ops = stages | (
        _REDUCTION_OPS if "StandaloneReduceOperations" in stages else frozenset()
    )
    prepared_input_ops = set(prepared_output_ops)
    for stage, operations in {
        "Non4DReshape": {"Reshape"},
        "Non4DExpand": {"Expand"},
        "Non4DTile": {"Tile"},
        "Non4DTranspose": {"Transpose"},
        "Non4dSliceAxis": {"Slice"},
        "Non4DLpNorm": {"LpNormalization"},
        "FixInstanceNormChannelMismatch": {"InstanceNormalization"},
        "RemoveIntermediarySqueezeAndUnsqueeze": {"Squeeze"},
        "UpdateAxis": {
            "Softmax",
            "LogSoftmax",
            "TopK",
            "Concat",
            "Split",
            "QuantizeLinear",
            "DequantizeLinear",
        },
    }.items():
        if stage in stages:
            prepared_input_ops.update(operations)
    quant_types = (
        DEFAULT_QDQ_REMOVE_QUANT_TYPES
        if qdq_quantized_dtypes is None
        else parse_qdq_quantized_dtype_list(qdq_quantized_dtypes)
    )
    if remove_all_qdq:
        quant_types = frozenset(onnx.TensorProto.DataType.values())
    try:
        _validate_boundary_contract(
            model, prepared_input_ops, prepared_output_ops, require_inferred_shapes=True
        )
    except _BoundaryPreflightError:
        raise
    except (ValueError, NotImplementedError):
        # Earlier lowering may consume native operators that lack a later rank pass.
        probe = infer_shapes(copy.deepcopy(model))
        for name, transform in DLA_TRANSFORMS:
            if name == "Non4dModelInputs":
                break
            options = _boundary_stage_options(
                name, quant_types, False, prepared_input_ops, prepared_output_ops
            )
            result = transform(probe, **options)
            if result is not None:
                probe = result
        _validate_boundary_contract(probe, prepared_input_ops, prepared_output_ops)
        del probe
    model = infer_shapes(model)
    for name, transform in DLA_TRANSFORMS:
        if verbose:
            logger.info("[DLA pipeline] Applying %s", name)
        options = _boundary_stage_options(
            name, quant_types, verbose, prepared_input_ops, prepared_output_ops
        )
        result = transform(model, **options)
        if result is not None:
            model = result
    model = infer_shapes(model)
    model = _apply_reorder_nodes(model)
    onnx.checker.check_model(model)
    if verbose:
        logger.info("[DLA pipeline] Done (%.1fs).", time.perf_counter() - started)
    return model
