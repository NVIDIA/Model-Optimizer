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

import warnings

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

__all__ = [
    "TiedWeightMap",
    "get_language_model_from_vl",
    "get_model_type",
    "is_multimodal_model",
]


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


class TiedWeightMap:
    """Name-based lookups over HF's ``{alias: canonical}`` tie map (``model.all_tied_weights_keys``).

    Export sites ask for a *group key*: both sides of a tie share one key, an untied parameter
    returns ``None``. The key is a name, so it survives packing / FSDP / offload, where a
    ``data_ptr`` would not.
    """

    def __init__(self, model: nn.Module) -> None:
        """Source the tie map from HF's ``all_tied_weights_keys`` (transformers >=5.0).

        HF's ``{target: source}`` == our ``{alias: canonical}``, resolved at load, config-gated,
        ``torch.equal``-pruned, and name-based so it survives FSDP shard / offload. Absent on
        transformers <5.0 -> empty map (the ``data_ptr`` backstop in postprocess is the net).
        """
        all_tied = getattr(model, "all_tied_weights_keys", None)
        # Warn whenever a tie is declared (embedding tie or any ``_tied_weights_keys`` entry, e.g.
        # encoder/decoder or fused-MoE) but the name-based map is missing, not just for embeddings.
        declares_tie = bool(
            getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        ) or bool(getattr(model, "_tied_weights_keys", None))
        if all_tied is None and declares_tie:
            warnings.warn(
                "This model may contain tied/shared weights, but deduplicating them on export "
                "requires transformers>=5.0 (it uses model.all_tied_weights_keys, which is only "
                "supported in newer versions). On older versions the exported checkpoint may keep "
                "duplicate copies of the tied weights (larger files), and tied weights may not be "
                "deduplicated correctly during export. Upgrade to transformers>=5.0 for correct "
                "tied-weight export."
            )
        # Drop any self-entry (alias == canonical): HF should not emit one, but a target==source
        # pair would schedule the kept canonical for deletion, so filter it out defensively.
        self.alias_to_canonical: dict[str, str] = {
            alias: canonical for alias, canonical in (all_tied or {}).items() if alias != canonical
        }
        self.canonical_names: set[str] = set(self.alias_to_canonical.values())

    def group_key(self, param_full_name: str) -> str | None:
        """Canonical group key for a parameter name, or ``None`` if untied.

        Both sides of a tie return the same key, so it does not matter which side export
        visits first.
        """
        if param_full_name in self.alias_to_canonical:
            return self.alias_to_canonical[param_full_name]
        if param_full_name in self.canonical_names:
            return param_full_name
        return None

    def container_group_key(self, container_name: str, first_proj_attr: str) -> str | None:
        """Group key for a fused-experts container, or ``None`` if untied.

        The tie lives on the container's 3-D projection (e.g. ``…experts.gate_up_proj``);
        stripping that suffix gives one key shared by all the container's projections.
        """
        gk = self.group_key(f"{container_name}.{first_proj_attr}")
        if gk is None:
            return None
        return gk.removesuffix(f".{first_proj_attr}")
