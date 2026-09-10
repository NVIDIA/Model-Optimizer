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

"""Utility functions for graph surgery operations."""

from .dtype_conversion import convert_fp16_to_bf16, fp16_to_bf16_array
from .graph_utils import (
    add_initializer,
    array_to_initializer,
    cleanup_unused_ios,
    convert_model_fp16_to_bf16,
    detect_model_dtype,
    find_initializer,
    find_node_by_name,
    find_node_by_output,
    find_nodes_by_pattern,
    get_all_tensors_used,
    get_consumers,
    get_onnx_dtype,
    initializer_to_array,
    remove_node,
    topological_sort_nodes,
    uses_external_data,
)
from .rope_cache import get_rope_caches
from .whisper_utils import (
    generate_audio_processor_config,
    generate_genai_config,
    save_audio_processor_config,
    save_genai_config,
    update_genai_config_decoder,
    update_genai_config_encoder,
)

__all__ = [
    "add_initializer",
    "array_to_initializer",
    "cleanup_unused_ios",
    "convert_fp16_to_bf16",
    "convert_model_fp16_to_bf16",
    "detect_model_dtype",
    "find_initializer",
    "find_node_by_name",
    "find_node_by_output",
    "find_nodes_by_pattern",
    "fp16_to_bf16_array",
    "generate_audio_processor_config",
    "generate_genai_config",
    "get_all_tensors_used",
    "get_consumers",
    "get_onnx_dtype",
    "get_rope_caches",
    "initializer_to_array",
    "remove_node",
    "save_audio_processor_config",
    "save_genai_config",
    "topological_sort_nodes",
    "update_genai_config_decoder",
    "update_genai_config_encoder",
    "uses_external_data",
]
