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
    - Specific multimodal model types (phi4mm)
    - Vision LoRA configurations
    - Audio processing capabilities
    - Image embedding layers
    - Nemotron-Parse conditional generation models

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

    # Check for Nemotron-Parse encoder-decoder architecture
    architectures = getattr(config, "architectures", [])
    is_nemotron_parse = any("nemotronparse" in arch.lower() for arch in architectures)

    return (
        hasattr(config, "vision_config")  # Standard vision config (e.g., Qwen2.5-VL)
        or hasattr(model, "language_model")  # Language model attribute (e.g., LLaVA)
        or getattr(config, "model_type", "") == "phi4mm"  # Phi-4 multimodal
        or hasattr(config, "vision_lora")  # Vision LoRA configurations
        or hasattr(config, "audio_processor")  # Audio processing capabilities
        or (
            hasattr(config, "embd_layer") and hasattr(config.embd_layer, "image_embd_layer")
        )  # Image embedding layers
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


def _collect_canonical_tied_patterns(
    model: nn.Module,
) -> tuple[list[re.Pattern], list[str]]:
    """Walk the model and collect canonical-side tied-weight matchers.

    Patterns are submodule-prefixed regexes from each module's
    ``_tied_weights_keys`` dict-style declaration (the prefix matters
    for nested models where the dict lives on an inner submodule).
    Side substrings are dot-separated tokens that appear only on the
    canonical side of those declarations — needed because modelopt's
    per-expert unpacking creates post-export keys (e.g.
    ``…experts.Y.gate_proj.input_scale``) that HF's regexes never knew
    about. List-style (legacy) declarations are skipped.
    """
    patterns: list[re.Pattern] = []
    alias_token_set: set[str] = set()
    canonical_token_set: set[str] = set()

    def _tokens(s: str) -> set[str]:
        """Identifiers in a regex string, with regex specials as separators."""
        return {tok for tok in re.split(r"[^A-Za-z0-9_]+", s) if tok}

    for name, submodule in model.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", None)
        if not isinstance(tied, dict) or not tied:
            continue
        prefix = f"{name}." if name else ""
        for alias_pat, canonical_pat in tied.items():
            patterns.append(re.compile(prefix + canonical_pat))
            alias_token_set.update(_tokens(prefix + alias_pat))
            canonical_token_set.update(_tokens(prefix + canonical_pat))

    # Tokens unique to the canonical side become substring matchers.
    side_substrings = sorted(canonical_token_set - alias_token_set)
    return patterns, side_substrings


def _reorder_canonical_first(state_dict: dict, model: nn.Module) -> dict:
    r"""Reorder ``state_dict`` so canonical-side tied keys iterate first.

    Lets the downstream first-wins data_ptr dedup keep canonical names.
    Uses both regex patterns and substring matchers from
    :func:`_collect_canonical_tied_patterns`. Gated on the model class
    name to scope the reorder to DiffusionGemma; other tied
    encoder-decoder models that ship dict-style ``_tied_weights_keys``
    can be added to the allowlist here. Mirrors the ``model_type``
    dispatch used for the Whisper and Nemotron-VL branches elsewhere
    in ``unified_export_hf.py``.
    """
    model_type = type(model).__name__.lower()
    if "diffusiongemma" not in model_type and "diffusion_gemma" not in model_type:
        return state_dict

    canonical_patterns, side_substrings = _collect_canonical_tied_patterns(model)
    if not canonical_patterns and not side_substrings:
        return state_dict

    def _has_side_substring(key: str) -> bool:
        # Require the token to appear as a proper dot-separated path
        # component, not just as a substring of an unrelated identifier.
        for tok in side_substrings:
            if (
                f".{tok}." in key
                or key.startswith(f"{tok}.")
                or key.endswith(f".{tok}")
                or key == tok
            ):
                return True
        return False

    head: dict = {}
    tail: dict = {}
    for k, v in state_dict.items():
        if any(p.search(k) for p in canonical_patterns) or _has_side_substring(k):
            head[k] = v
        else:
            tail[k] = v
    head.update(tail)
    return head
