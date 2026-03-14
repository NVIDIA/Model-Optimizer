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

Example usage:
    >>> from modelopt.onnx.graph_surgery import (
    ...     replace_attention_with_gqa,
    ...     convert_fp16_to_bf16,
    ...     transpose_dequantize_linear_weights,
    ...     add_cross_kv_to_encoder,
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
    ...     encoder_path="encoder_model.onnx",
    ...     output_path="encoder_with_kv.onnx",
    ...     hf_model_id="openai/whisper-large-v3-turbo",
    ... )
    >>> # Standalone FP16 to BF16 conversion
    >>> convert_fp16_to_bf16(
    ...     input_path="model_fp16.onnx",
    ...     output_path="model_bf16.onnx",
    ... )
    >>>
    >>> # Transpose DequantizeLinear weights for column-major storage
    >>> transpose_dequantize_linear_weights(
    ...     model_path="model_quantized.onnx",
    ...     output_path="model_quantized_transposed.onnx",
    ... )
"""

from .dq_transpose import transpose_dequantize_linear_weights
from .encoder_cross_kv import add_cross_kv_to_encoder
from .gqa_replacement import replace_attention_with_gqa
from .utils.dtype_conversion import convert_fp16_to_bf16

__all__ = [
    "add_cross_kv_to_encoder",
    "convert_fp16_to_bf16",
    "replace_attention_with_gqa",
    "transpose_dequantize_linear_weights",
]
