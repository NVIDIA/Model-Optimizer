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

"""Whisper-specific graph surgery operations.

This module provides graph transformations for converting Optimum-exported
Whisper models to ONNX Runtime GenAI compatible format.

Operations:
- Cross KV: Add cross-attention KV cache outputs to encoder

Example usage:
    >>> from modelopt.onnx.graph_surgery.whisper import (
    ...     add_cross_kv_to_encoder,
    ... )
    >>>
    >>> # Add cross-attention KV outputs to encoder
    >>> add_cross_kv_to_encoder(
    ...     encoder_path="encoder_model.onnx",
    ...     output_path="encoder_with_kv.onnx",
    ...     hf_model_id="openai/whisper-large-v3-turbo",
    ... )
"""

from ..utils.whisper_utils import (
    generate_audio_processor_config,
    generate_genai_config,
    save_audio_processor_config,
    save_genai_config,
    update_genai_config_decoder,
    update_genai_config_encoder,
)
from .encoder_cross_kv import add_cross_kv_to_encoder

__all__ = [
    "add_cross_kv_to_encoder",
    "generate_audio_processor_config",
    "generate_genai_config",
    "save_audio_processor_config",
    "save_genai_config",
    "update_genai_config_decoder",
    "update_genai_config_encoder",
]
