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
from .dla_transforms.dla_force_fp16_activations import (
    _apply_force_fp16_activations,
    _apply_fp16_lpnorm,
)
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
    _apply_fix_instancenorm_channels,
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
    _apply_remove_intermediary_squeeze_unsqueeze,
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
from .dla_transforms.dla_remove_reshapes import _apply_remove_reshapes
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
    infer_shapes,
    parse_qdq_quantized_dtype_list,
    save_onnx_model,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["DLA_TRANSFORMS", "dla_make_dla_compatible"]


def _remove_deqlin_and_unused_initializers(model):
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
    ("Non4dModelOutputs", _apply_non4d_model_outputs),
    ("Non4DReshape", _apply_non4d_reshape),
    ("ArgMax", _apply_argmax),
    ("StandaloneReduceOperations", _apply_standalone_reduce_operations),
    ("RemoveIntermediarySqueezeAndUnsqueeze", _apply_remove_intermediary_squeeze_unsqueeze),
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
    ("RemoveReshapes", _apply_remove_reshapes),
    ("RemoveUnusedInitializers", _remove_unused_initializers),
)


def _transform_make_dla_compatible(
    model: onnx.ModelProto,
    *,
    verbose: bool = True,
    qdq_quantized_dtypes: Iterable[int | str] | None = None,
    remove_all_qdq: bool = False,
    force_fp16_activations: bool = False,
) -> onnx.ModelProto:
    """Apply the logged DLA sequence in memory, then infer shapes and validate."""
    started = time.perf_counter()
    if remove_all_qdq and qdq_quantized_dtypes is not None:
        raise ValueError("Specify remove_all_qdq or qdq_quantized_dtypes, not both")
    if (remove_all_qdq or qdq_quantized_dtypes is not None) and not any(
        name == "RemoveQDQ" for name, _ in DLA_TRANSFORMS
    ):
        raise NotImplementedError("QDQ removal is not available in this pipeline")
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
        elif name == "Non4DLpNorm" and force_fp16_activations:
            transform = _apply_fp16_lpnorm
        result = transform(model, **options)
        if result is not None:
            model = result
    model = infer_shapes(model)
    model = _apply_reorder_nodes(model)
    onnx.checker.check_model(model)
    if force_fp16_activations:
        if verbose:
            logger.info("[DLA pipeline] Applying ForceFp16Activations")
        model = _apply_force_fp16_activations(model)
        # Match C++: symbolic inference stays before precision conversion.
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
    if verbose:
        logger.info("[DLA pipeline] Done (%.1fs).", time.perf_counter() - started)
    return model


def dla_make_dla_compatible(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    qdq_quantized_dtypes: Iterable[int | str] | None = None,
    remove_all_qdq: bool = False,
    force_fp16_activations: bool = False,
) -> onnx.ModelProto:
    """Load an ONNX model, apply the ordered DLA transforms, and save the result.

    Args:
        model_path: Input file; external weights are loaded from its directory.
        output_path: Destination ONNX file.
        use_external_data: Store large tensor payloads in a sidecar file.
        external_data_name: Sidecar filename; defaults to output filename plus "_data".
        verbose: Log each transform stage.
        qdq_quantized_dtypes: Zero-point dtypes whose QDQ pairs are removed.
            Defaults to UINT16 and INT16. Accepts ONNX dtype IDs or names.
        remove_all_qdq: Remove QDQ pairs for every dtype, matching the C++ flag.
        force_fp16_activations: Use FP16 internals while preserving FP32 public boundaries.
            Cannot be combined with qdq_quantized_dtypes.

    Returns:
        The transformed model. Invalid models and failed transformations raise.
    """
    model = _transform_make_dla_compatible(
        onnx.load(model_path, load_external_data=True),
        verbose=verbose,
        qdq_quantized_dtypes=qdq_quantized_dtypes,
        remove_all_qdq=remove_all_qdq,
        force_fp16_activations=force_fp16_activations,
    )
    save_onnx_model(
        model,
        output_path,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
    )
    return model
