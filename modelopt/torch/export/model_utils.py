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

import re
from collections import defaultdict

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
    # DiffusionGemma must come before "Gemma" — get_model_type substring-matches
    # in order, and "gemma" is a substring of "diffusiongemma".
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
    - Nemotron-Parse conditional generation models

    Args:
        model: The HuggingFace model instance to check

    Returns:
        bool: True if the model is detected as multimodal, False otherwise

    Examples:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> is_multimodal_model(model)
        True
    """
    config = model.config

    # Check for Nemotron-Parse encoder-decoder architecture
    architectures = getattr(config, "architectures", [])
    is_nemotron_parse = any("nemotronparse" in arch.lower() for arch in architectures)

    return (
        hasattr(config, "vision_config")  # Standard vision config (e.g., Qwen2.5-VL)
        or hasattr(model, "language_model")  # Language model attribute (e.g., LLaVA)
        or is_nemotron_parse  # Nemotron-Parse conditional generation model
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

    # Pattern 3: For encoder-decoder VL models (e.g., Nemotron-Parse), the decoder is the language model.
    # Only match if the model is detected as multimodal to avoid matching non-VLM encoder-decoder
    # models like T5, Bart, Whisper which also have .decoder.
    if hasattr(model, "decoder") and is_multimodal_model(model):
        return [model, model.decoder]

    # Pattern 4: No language_model found
    return None


def _build_tied_weight_map(model: nn.Module) -> dict[str, str]:
    """Map tied parameter aliases to a canonical name before export mutates the model.

    Ties come only from groups that share the same live :class:`nn.Parameter` object.
    ``_tied_weights_keys`` and ``tie_word_embeddings`` choose which name to retain; they
    never create a tie by themselves.
    """
    groups: dict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups[id(parameter)].append(name)

    declared_aliases: set[str] = set()
    for module_name, module in model.named_modules():
        tied = getattr(module, "_tied_weights_keys", None)
        if not isinstance(tied, dict):
            continue
        prefix = f"{module_name}." if module_name else ""
        for pattern in tied:
            try:
                alias_pattern = re.compile(pattern)
            except re.error:
                continue
            for names in groups.values():
                for name in names:
                    if name.startswith(prefix) and alias_pattern.search(name[len(prefix) :]):
                        declared_aliases.add(name)

    embedding_canonical: dict[int, str] = {}
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        try:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
        except (AttributeError, NotImplementedError):
            input_embeddings = output_embeddings = None
        if (
            input_embeddings is not None
            and output_embeddings is not None
            and getattr(input_embeddings, "weight", None)
            is getattr(output_embeddings, "weight", object())
        ):
            parameter = input_embeddings.weight
            for name in groups.get(id(parameter), []):
                owner_name = name.rsplit(".", 1)[0] if "." in name else ""
                try:
                    owner = model.get_submodule(owner_name) if owner_name else model
                except AttributeError:
                    continue
                if owner is input_embeddings:
                    embedding_canonical[id(parameter)] = name
                    break

    tied_weight_map: dict[str, str] = {}
    for parameter_id, names in groups.items():
        if len(names) < 2:
            continue
        canonical = embedding_canonical.get(parameter_id)
        if canonical is None:
            canonical = next((name for name in names if name not in declared_aliases), names[0])
        tied_weight_map.update({name: canonical for name in names if name != canonical})
    return tied_weight_map
