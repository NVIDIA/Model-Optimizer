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

"""Graph surgery module for post-processing ONNX models.

This module provides utilities for performing graph-level transformations on ONNX models
after export. Common use cases include:

- Replacing standard attention patterns with GroupQueryAttention (GQA) for LLMs
- Adding cross-attention KV cache outputs to encoder models
- Converting model precision (e.g., FP16 to BF16)
- Transposing DequantizeLinear weights for column-major storage optimization
- Graph cleanup and optimization
- Making models compatible with the NVIDIA DLA accelerator

Example usage:
    >>> from modelopt.onnx.graph_surgery import (
    ...     replace_attention_with_gqa,
    ...     convert_fp16_to_bf16,
    ...     transpose_dequantize_linear_weights,
    ...     add_cross_kv_to_encoder,
    ...     make_dla_compatible,
    ... )
    >>> # Replace attention with GQA for LLMs (FP16 model)
    >>> replace_attention_with_gqa(
    ...     model_path="model_fp16.onnx",
    ...     output_path="model_gqa.onnx",
    ...     hf_model_id="meta-llama/Llama-2-7b-hf",
    ...     io_dtype="float16",
    ... )
    >>> # Replace attention with GQA and convert to BF16 in one step
    >>> replace_attention_with_gqa(
    ...     model_path="model_fp16.onnx",
    ...     output_path="model_gqa_bf16.onnx",
    ...     hf_model_id="meta-llama/Llama-2-7b-hf",
    ...     io_dtype="bfloat16",  # Automatically converts FP16 to BF16
    ... )
    >>> # Add cross-attention KV cache outputs to encoder (GenAI compatible)
    >>> add_cross_kv_to_encoder(
    ...     model_path="encoder_model.onnx",
    ...     output_path="encoder_with_kv.onnx",
    ...     hf_model_id="openai/whisper-large-v3-turbo",
    ... )
    >>> # Standalone FP16 to BF16 conversion
    >>> convert_fp16_to_bf16(
    ...     model_path="model_fp16.onnx",
    ...     output_path="model_bf16.onnx",
    ... )
    >>>
    >>> # Transpose DequantizeLinear weights for column-major storage
    >>> transpose_dequantize_linear_weights(
    ...     model_path="model_quantized.onnx",
    ...     output_path="model_quantized_transposed.onnx",
    ... )
    >>> # Apply the full DLA compatibility pipeline (16-step transform sequence)
    >>> make_dla_compatible(
    ...     model_path="model.onnx",
    ...     output_path="model_dla.onnx",
    ... )
"""

import os

import onnx

from .dq_transpose import _transform_dq_transpose, transpose_dequantize_linear_weights
from .encoder_cross_kv import _transform_cross_kv, add_cross_kv_to_encoder
from .gqa_replacement import _transform_gqa, replace_attention_with_gqa
from .make_dla_compatible import _transform_make_dla_compatible
from .make_dla_compatible import dla_make_dla_compatible as make_dla_compatible
from .utils.dtype_conversion import _transform_fp16_to_bf16, convert_fp16_to_bf16

_SURGERY_REGISTRY = {
    "replace-gqa": _transform_gqa,
    "add-cross-kv": _transform_cross_kv,
    "convert-bf16": _transform_fp16_to_bf16,
    "transpose-dq": _transform_dq_transpose,
    "make-dla-compatible": _transform_make_dla_compatible,
}


def get_available_surgeries() -> list[str]:
    """Return a list of all registered graph surgery names."""
    return list(_SURGERY_REGISTRY.keys())


def _save_model(
    model: onnx.ModelProto,
    output_path: str,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    size_threshold: int = 1024,
    verbose: bool = True,
) -> None:
    """Unified model saving logic for all graph surgeries.

    Args:
        model: The ONNX model to save.
        output_path: Path to save the model.
        use_external_data: Whether to save weights as external data.
        external_data_name: Name for external data file.
            Defaults to ``<output_filename>_data``.
        size_threshold: Minimum tensor size (bytes) to externalize.
        verbose: Whether to print progress messages.
    """
    from ..logging_config import logger

    if verbose:
        logger.info(f"\nSaving modified model to: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if use_external_data:
        if external_data_name is None:
            external_data_name = os.path.basename(output_path) + "_data"

        if verbose:
            logger.info(f"  Saving weights to external file: {external_data_name}")

        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=size_threshold,
        )
    else:
        onnx.save(model, output_path)
    model = onnx.load(output_path, load_external_data=True)

    # Run shape inference (beneficial for all surgeries, no-op if nothing changed)
    if verbose:
        logger.info("\nRunning shape inference (file-to-file)...")
    try:
        onnx.shape_inference.infer_shapes_path(
            output_path, output_path, check_type=False, strict_mode=False, data_prop=False
        )
        if verbose:
            logger.info("  Shape inference completed")
    except Exception as e:
        if verbose:
            logger.info(f"  Shape inference failed (non-fatal, model already saved): {e}")

    if verbose:
        logger.info("Done!")


def run_graph_surgery(
    surgery_name: str,
    model_path: str,
    output_path: str,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Run a graph surgery by name with centralized model loading and saving.

    This is the unified entry point for all graph surgeries. It handles:
    1. Loading the input model from ``model_path``
    2. Dispatching to the appropriate transform function
    3. Saving the result to ``output_path`` with unified save logic

    When new surgeries are added to the registry, they are automatically
    available through this function without any changes to calling code.

    Args:
        surgery_name: Name of the surgery to run (e.g. 'replace-gqa', 'transpose-dq').
            Use get_available_surgeries() to see all available options.
        model_path: Path to the input ONNX model.
        output_path: Path to save the output ONNX model.
        use_external_data: Whether to save weights as external data file.
        external_data_name: Name for external data file.
            Defaults to ``<output_filename>_data``.
        verbose: Whether to print progress messages.
        **kwargs: Surgery-specific parameters passed directly to the transform function.

    Returns:
        The modified ONNX ModelProto.

    Raises:
        ValueError: If surgery_name is not registered.

    Example:
        >>> from modelopt.onnx.graph_surgery import run_graph_surgery, get_available_surgeries
        >>> print(get_available_surgeries())
        ['replace-gqa', 'add-cross-kv', 'convert-bf16', 'transpose-dq']
        >>> run_graph_surgery(
        ...     "replace-gqa",
        ...     model_path="model.onnx",
        ...     output_path="model_gqa.onnx",
        ...     hf_model_id="meta-llama/Llama-2-7b-hf",
        ... )
    """
    from ..logging_config import logger

    if surgery_name not in _SURGERY_REGISTRY:
        available = ", ".join(f"'{s}'" for s in _SURGERY_REGISTRY)
        raise ValueError(f"Unknown surgery: '{surgery_name}'. Available surgeries: {available}")

    # Load
    if verbose:
        logger.info(f"Loading model from: {model_path}")
    model = onnx.load(model_path, load_external_data=True)

    # Transform
    transform_fn = _SURGERY_REGISTRY[surgery_name]
    kwargs.setdefault("verbose", verbose)
    model = transform_fn(model=model, **kwargs)

    # Save
    _save_model(
        model,
        output_path,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
    )

    return model


__all__ = [
    "add_cross_kv_to_encoder",
    "convert_fp16_to_bf16",
    "get_available_surgeries",
    "make_dla_compatible",
    "replace_attention_with_gqa",
    "run_graph_surgery",
    "transpose_dequantize_linear_weights",
]
