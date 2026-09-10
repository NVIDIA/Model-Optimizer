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

"""End-to-end DLA compatibility pipeline for ONNX models.

NVIDIA DLA (Deep Learning Accelerator) is a fixed-function hardware engine
optimised for deep learning inference.  It imposes strict constraints that
require the ONNX graph to be transformed before deployment:

DLA Hardware Constraints
------------------------
- **Tensor rank:** only **4-D tensors** are supported.  All intermediate and
  graph-boundary tensors must be promoted to rank 4.
- **Data types:** only **FP16** and **INT8 / UINT8** are supported on-chip;
  float32 arithmetic is emulated in software.
- **Fixed-function ops:** only a curated subset of ONNX operators is
  natively accelerated; unsupported patterns must be decomposed or rewritten.

This module provides :func:`dla_make_dla_compatible`, which applies all
necessary graph-surgery transforms in the canonical 16-step pipeline order
so that the resulting model satisfies DLA's hardware requirements.

Transform sequence
------------------
Stage 1 - Early preprocessing
    1.  ``dla_remove_qdq``                   - strip QDQ pairs for uint16/int16 weights
    2.  ``dla_constants_to_initializers``    - hoist Constant ops into initializers
    3.  ``dla_cast_to_fp32``                 - redirect Cast targets to FLOAT (with DLA-safe skips)
    4.  ``dla_remove_deqlin``                - fold static DequantizeLinear into float32 initializers

Stage 2 - Specialised pre-4D transforms
    5.  ``dla_5d_reshape_to_4d``             - reduce 5-D reshape patterns to 4-D
    6.  ``dla_matmul_to_transpose_conv_transpose`` - convert MatMul/Gemm to ConvTranspose chains

Stage 3 - Op-specific DLA compatibility transforms
    7.  ``dla_fix_instancenorm_channel_mismatch`` - 3-D InstanceNorm to 4-D Reshape/IN/Reshape
    8.  ``dla_where``                        - Where(cond, 0, y) to Mul; Where(cond, x, 0) to Mul
    9.  ``dla_not``                          - Cast->Not to Cast(FLOAT)->Clip->Sub
   10.  ``dla_greater``                      - ensure Greater inputs are float32
   11.  ``dla_topk``                         - TopK->Cast->Reshape->Tile->GatherElements chain rewrite
   12.  ``dla_handle_qdq``                   - wrap Q/DQ nodes with Unsqueeze/Squeeze for 4-D

Stage 4 - Main 4-D conversion
   13.  ``dla_convert_ops_to_4d``            - wrap non-4D ops (Gather, Expand, Transpose,
                                               ReduceSum, ArgMax, GQA, LpNorm, etc.) with
                                               Unsqueeze/Squeeze to make all tensors 4-D

Stage 5 - Decompose LSTM + Final graph cleanup
   14.  ``dla_decompose_lstm``               - decompose LSTM into primitive ops
   15.  ``dla_graph_cleanup``                - canonicalise non-4D graph inputs, replace
                                               intermediary Squeeze/Unsqueeze with 4-D Reshape,
                                               collapse Reshape chains, fold constants,
                                               remove unused initializers

Stage 6 - Post-cleanup int-tensor cast
   16.  ``dla_unsqueeze``                    - insert Cast(FLOAT) after Unsqueeze of int tensors
"""

from __future__ import annotations

import time

import onnx

from ..logging_config import logger
from .dla_transforms._common import infer_shapes, save_model
from .dla_transforms.dla_5d_reshape_to_4d import _apply_5d_reshape_to_4d
from .dla_transforms.dla_cast_to_fp32 import _apply_cast_to_fp32
from .dla_transforms.dla_constants_to_initializers import transform_constants_to_initializers
from .dla_transforms.dla_convert_ops_to_4d import _apply_convert_ops_to_4d
from .dla_transforms.dla_decompose_lstm import _apply_decompose_lstm
from .dla_transforms.dla_fix_instancenorm_channel_mismatch import (
    _apply_fix_instancenorm_channel_mismatch,
)
from .dla_transforms.dla_graph_cleanup import _apply_graph_cleanup
from .dla_transforms.dla_greater import _apply_greater
from .dla_transforms.dla_handle_qdq import _apply_handle_qdq
from .dla_transforms.dla_matmul_to_transpose_conv_transpose import (
    _apply_matmul_to_transpose_conv_transpose,
)
from .dla_transforms.dla_not import _apply_not
from .dla_transforms.dla_remove_deqlin import _apply_remove_deqlin
from .dla_transforms.dla_remove_qdq import transform_remove_qdq
from .dla_transforms.dla_topk import _apply_topk
from .dla_transforms.dla_unsqueeze import _apply_unsqueeze
from .dla_transforms.dla_where import _apply_where


def _transform_make_dla_compatible(
    model: onnx.ModelProto,
    *,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Pure in-memory DLA compatibility transformation.

    Takes an already-loaded ONNX model, applies the full 16-step DLA
    compatibility pipeline, and returns the modified model without saving.
    This is the function registered in ``_SURGERY_REGISTRY`` and called by
    :func:`run_graph_surgery`.

    Args:
        model: Already-loaded ONNX ModelProto.
        verbose: Emit progress logs when ``True`` (default).
        **kwargs: Accepted for registry compatibility; currently unused.

    Returns:
        The modified ONNX ModelProto.
    """
    t0 = time.perf_counter()

    # -- Pre-pipeline: initial shape inference ----------------------------------------------------
    model = infer_shapes(model)

    # -- Stage 1: Early preprocessing -------------------------------------------------------------
    transform_remove_qdq(model)
    model = transform_constants_to_initializers(model)
    model = _apply_cast_to_fp32(model)
    model = _apply_remove_deqlin(model)

    # -- Stage 2: Specialised pre-4D transforms ---------------------------------------------------
    model = _apply_5d_reshape_to_4d(model)
    model = infer_shapes(model)
    model = _apply_matmul_to_transpose_conv_transpose(model)

    # -- Stage 3: Op-specific DLA compatibility transforms -----------------------------------------
    model = infer_shapes(model)
    model = _apply_fix_instancenorm_channel_mismatch(model)
    model = _apply_where(model)
    model = _apply_not(model)
    model = _apply_greater(model)
    model = _apply_topk(model)
    model = _apply_handle_qdq(model)

    # -- Stage 4: Main 4-D conversion -------------------------------------------------------------
    model = _apply_convert_ops_to_4d(model)

    # -- Stage 5: Decompose LSTM + Final graph cleanup ---------------------------------------------
    model = infer_shapes(model)
    model = _apply_decompose_lstm(model)
    model = _apply_graph_cleanup(model)

    # -- Stage 6: Post-cleanup int-tensor cast -----------------------------------------------------
    model = infer_shapes(model)
    model = _apply_unsqueeze(model)

    # -- Post-pipeline: final shape inference + ONNX validity check ---------------------------------
    model = infer_shapes(model)

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        logger.error("[DLA pipeline] ONNX validation failed: %s", exc)
        raise
    except (MemoryError, OSError) as exc:
        logger.warning(
            "[DLA pipeline] ONNX in-memory check skipped (model may be too large): %s", exc
        )

    elapsed = time.perf_counter() - t0
    logger.info("[DLA pipeline] Done (%.1fs).", elapsed)
    return model


def dla_make_dla_compatible(
    model_path: str,
    output_path: str | None = None,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
) -> onnx.ModelProto:
    """Transform an ONNX model to be compatible with the NVIDIA DLA accelerator.

    DLA is a fixed-function hardware engine that requires all tensors to be
    4-D and uses only FP16 / INT8 data types.  This function applies the full
    16-step graph-surgery pipeline (see module docstring for the complete
    transform sequence) to satisfy those hardware constraints.

    The transformation is lossless for numerics: every graph rewrite preserves
    the mathematical equivalence of the original model.

    Args:
        model_path: Path to the input ONNX model.
        output_path: Optional path to save the transformed model. If omitted,
            the transformed model is returned without writing to disk.
        use_external_data: Whether to save weights as external data when
            ``output_path`` is provided.
        external_data_name: External data filename when ``output_path`` is
            provided (default: ``<output_basename>_data``).
        verbose: When ``True`` (default), progress messages are emitted via
            the module logger at each pipeline stage.

    Returns:
        The transformed :class:`onnx.ModelProto`.

    Raises:
        RuntimeError: If ORT symbolic shape inference fails at any pipeline step.

    Example:
        >>> from modelopt.onnx.graph_surgery import make_dla_compatible
        >>> model = make_dla_compatible(
        ...     model_path="model.onnx",
        ...     output_path="model_dla.onnx",
        ... )
    """
    if verbose:
        logger.info("Loading model from: %s", model_path)
    model = onnx.load(model_path, load_external_data=True)

    model = _transform_make_dla_compatible(model, verbose=verbose)

    if output_path is not None:
        save_model(
            model,
            output_path,
            use_external_data=use_external_data,
            external_data_name=external_data_name,
            verbose=verbose,
        )

    return model
