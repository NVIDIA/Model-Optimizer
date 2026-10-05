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
"""TensorRT-LLM decoder type detection for the deprecated TensorRT-LLM checkpoint export.

The decoder types here are TensorRT-LLM's own model names, not Hugging Face ``model_type`` values.
Only the TensorRT-LLM checkpoint export consumes them, so this module goes away with it in 0.49.0.
"""

import torch.nn as nn

__all__ = []

# Substring-matched against the lowercased model class name, in insertion order.
MODEL_NAME_TO_DECODER_TYPE = {
    "GPT2": "gpt",
    "Mllama": "mllama",
    "Llama4": "llama4",
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "Qwen3Moe": "qwen3moe",
    "Qwen3Next": "qwen3next",
    "QWen": "qwen",
    "RecurrentGemma": "recurrentgemma",
    # DiffusionGemma must come before "Gemma": "gemma" is a substring of "diffusiongemma".
    "DiffusionGemma": "diffusion_gemma",
    "Gemma3": "gemma3",
    "Gemma2": "gemma2",
    "Gemma": "gemma",
    "phi3small": "phi3small",
    "phi3": "phi3",
    "PhiMoEForCausalLM": "phi3",
    "phi": "phi",
    "TLGv4ForCausalLM": "phi",
    "MixtralForCausalLM": "llama",
    "ArcticForCausalLM": "llama",
    "StarCoder": "gpt",
    "Dbrx": "dbrx",
    "T5": "t5",
    "Bart": "bart",
    "GLM": "glm",
    "InternLM2ForCausalLM": "internlm",
    "ExaoneForCausalLM": "exaone",
    "NemotronH": "nemotron_h",
    "Nemotron": "gpt",
    "Deepseek": "deepseek",
    "Whisper": "whisper",
    "gptoss": "gptoss",
    "MiniMax": "minimax",
}


def get_decoder_type(model: nn.Module) -> str | None:
    """Return the TensorRT-LLM decoder type for ``model``'s class name, or None if unknown."""
    name = type(model).__name__.lower()
    for k, v in MODEL_NAME_TO_DECODER_TYPE.items():
        if k.lower() in name:
            return v
    return None
