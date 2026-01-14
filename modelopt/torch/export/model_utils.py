# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utility functions for model type detection and classification."""

import torch.nn as nn

MODEL_NAME_TO_TYPE = {
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
    "Gemma3": "gemma3",
    "Gemma2": "gemma2",
    "Gemma": "gemma",
    "phi3small": "phi3small",
    "phi3": "phi3",
    "PhiMoEForCausalLM": "phi3",
    "Phi4MMForCausalLM": "phi4mm",
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
    "Nemotron": "gpt",
    "Deepseek": "deepseek",
    "Whisper": "whisper",
    "gptoss": "gptoss",
}

__doc__ = f"""Utility functions for model type detection and classification.

    .. code-block:: python

        {MODEL_NAME_TO_TYPE=}
"""

__all__ = ["get_language_model_from_vl", "get_model_type", "is_multimodal_model"]


def get_model_type(model):
    """Try get the model type from the model name. If not found, return None."""
    for k, v in MODEL_NAME_TO_TYPE.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def is_multimodal_model(model):
    """Check if a model is a Vision-Language Model (VLM) or multimodal model.

    This function detects various multimodal model architectures by checking for:
    - Standard vision configurations (vision_config)
    - Language model attributes (language_model)
    - Specific multimodal model types (phi4mm)
    - Vision LoRA configurations
    - Audio processing capabilities
    - Image embedding layers

    Args:
        model: The HuggingFace model instance to check

    Returns:
        bool: True if the model is detected as multimodal, False otherwise

    Examples:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> is_multimodal_model(model)
        True

        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-multimodal-instruct")
        >>> is_multimodal_model(model)
        True
    """
    config = model.config

    return (
        hasattr(config, "vision_config")  # Standard vision config (e.g., Qwen2.5-VL)
        or hasattr(model, "language_model")  # Language model attribute (e.g., LLaVA)
        or getattr(config, "model_type", "") == "phi4mm"  # Phi-4 multimodal
        or hasattr(config, "vision_lora")  # Vision LoRA configurations
        or hasattr(config, "audio_processor")  # Audio processing capabilities
        or (
            hasattr(config, "embd_layer") and hasattr(config.embd_layer, "image_embd_layer")
        )  # Image embedding layers
    )


def get_language_model_from_vl(model) -> list[nn.Module] | None:
    """Extract the language model lineage from a Vision-Language Model (VLM).

    This function handles the common patterns for accessing the language model component
    in various VLM architectures. It checks multiple possible locations where the
    language model might be stored.

    Args:
        model: The VLM model instance to extract the language model from

    Returns:
        list: the lineage path towards the language model

    Examples:
        >>> # For LLaVA-style models
        >>> lineage = get_language_model_from_vl(vlm_model)
        >>> # lineage[0] is vlm_model
        >>> # lineage[1] is vlm_model.language_model
    """
    # always prioritize model.model.langauge_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return [model, model.model, model.model.language_model]

    if hasattr(model, "language_model"):
        return [model, model.language_model]

    # Pattern 3: No language_model found
    return None
