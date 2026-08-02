# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig

from ..tools.checkpoint_utils_hf import force_cache_dynamic_modules
from .capabilities import PuzzletronCapabilities, default_capabilities

__all__ = [
    "DescriptorResolution",
    "infer_descriptor_name",
    "register_native_config_aliases",
    "resolve_descriptor",
    "resolve_descriptor_by_name",
    "resolve_descriptor_from_pretrained",
]


def register_native_config_aliases() -> None:
    """Install optional AutoModel config aliases before any AutoConfig lookup.

    Native backends may introduce config model types before they are available
    in the installed Transformers release.  Descriptor resolution is the
    common loading boundary for every Puzzletron stage, so registration belongs
    here rather than in an individual stage or family adapter.
    """

    try:
        import nemo_automodel._transformers.registry  # noqa: F401
    except ImportError:
        # HF-only installations remain supported for legacy/fallback models.
        return


_MODEL_TYPE_TO_DESCRIPTOR = {
    "llama": "llama",
    "mistral": "mistral_small",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen3_5_text": "qwen3_5_text",
    "qwen3_5": "qwen3_5",
    "qwen3_6_text": "qwen3_6_text",
    "qwen3_6": "qwen3_6",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe_text",
    "qwen3_vl": "qwen3_vl",
    "nemotron_h": "nemotron_h",
    "nemotron_h_v2": "nemotron_h_v2",
    "gpt_oss": "gpt_oss",
}

_ARCH_TO_DESCRIPTOR = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral_small",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3_5ForConditionalGeneration": "qwen3_5",
    "Qwen3_5ForCausalLM": "qwen3_5_text",
    "Qwen3_6ForConditionalGeneration": "qwen3_6",
    "Qwen3_6ForCausalLM": "qwen3_6_text",
    "Qwen3_5MoeForConditionalGeneration": "qwen3_5_moe",
    "Qwen3_5MoeForCausalLM": "qwen3_5_moe_text",
    "Qwen3VLForConditionalGeneration": "qwen3_vl",
    "NemotronHForCausalLM": "nemotron_h",
    "NemotronHV2ForCausalLM": "nemotron_h_v2",
    "GptOssForCausalLM": "gpt_oss",
}


@dataclass(frozen=True)
class DescriptorResolution:
    name: str
    descriptor: type
    capabilities: PuzzletronCapabilities
    confidence: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "descriptor": f"{self.descriptor.__module__}.{self.descriptor.__name__}",
            "capabilities": self.capabilities.to_dict(),
            "confidence": self.confidence,
            "reason": self.reason,
        }


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _architectures(config: Any) -> list[str]:
    arch = _get(config, "architectures", None)
    if arch is None:
        text_config = _get(config, "text_config", None)
        arch = _get(text_config, "architectures", None)
    if isinstance(arch, str):
        return [arch]
    return list(arch or [])


def infer_descriptor_name(
    config: Any,
    *,
    descriptor_override: str | None = None,
    descriptor_aliases: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    if descriptor_override:
        return descriptor_override, "explicit descriptor_override"

    anymodel_arch_info = _get(config, "anymodel_arch_info", None)
    if isinstance(anymodel_arch_info, dict):
        descriptor = anymodel_arch_info.get("descriptor") or anymodel_arch_info.get(
            "model_descriptor"
        )
        if descriptor:
            return descriptor, "anymodel_arch_info descriptor"

    for arch in _architectures(config):
        if arch in _ARCH_TO_DESCRIPTOR:
            return _ARCH_TO_DESCRIPTOR[arch], f"architectures contains {arch}"
        if arch == "AnyModel":
            base_arch = _get(config, "base_architecture", None)
            if base_arch in _ARCH_TO_DESCRIPTOR:
                return _ARCH_TO_DESCRIPTOR[base_arch], f"base_architecture is {base_arch}"

    model_type = _get(config, "model_type", None)
    if model_type in _MODEL_TYPE_TO_DESCRIPTOR:
        return _MODEL_TYPE_TO_DESCRIPTOR[model_type], f"model_type is {model_type}"

    text_config = _get(config, "text_config", None)
    text_model_type = _get(text_config, "model_type", None)
    if text_model_type in _MODEL_TYPE_TO_DESCRIPTOR:
        return _MODEL_TYPE_TO_DESCRIPTOR[text_model_type], f"text_config.model_type is {text_model_type}"

    # Aliases are metadata discovered by a campaign preflight. They are deliberately
    # consulted only after every built-in architecture/model-type mapping, so a stale
    # campaign entry can never shadow an exact family implementation.
    aliases = descriptor_aliases or {}
    alias_candidates = [*_architectures(config), model_type, text_model_type]
    for candidate in alias_candidates:
        if candidate and candidate in aliases:
            return aliases[candidate], f"preflight alias for {candidate}"

    known = sorted({*_MODEL_TYPE_TO_DESCRIPTOR, *_ARCH_TO_DESCRIPTOR})
    raise ValueError(
        "Cannot infer Puzzletron descriptor from checkpoint config. "
        f"Known model types/architectures: {known}. Use model.descriptor_override "
        "while adding registry support for this family."
    )


def resolve_descriptor(
    config: Any,
    *,
    descriptor_override: str | None = None,
    descriptor_aliases: Mapping[str, str] | None = None,
) -> DescriptorResolution:
    # Import here to avoid import cycles during descriptor registration.
    from . import models as _models  # noqa: F401
    from .model_descriptor.model_descriptor_factory import ModelDescriptorFactory

    name, reason = infer_descriptor_name(
        config,
        descriptor_override=descriptor_override,
        descriptor_aliases=descriptor_aliases,
    )
    descriptor = ModelDescriptorFactory.get(name)
    if isinstance(descriptor, str):
        raise ValueError(f"Descriptor '{name}' is not registered")

    if hasattr(descriptor, "puzzletron_capabilities"):
        capabilities = descriptor.puzzletron_capabilities(config)
    else:
        capabilities = default_capabilities(descriptor_name=name, model_family=name)

    return DescriptorResolution(
        name=name,
        descriptor=descriptor,
        capabilities=capabilities,
        confidence="exact" if descriptor_override else "inferred",
        reason=reason,
    )


def resolve_descriptor_by_name(name: str) -> DescriptorResolution:
    from . import models as _models  # noqa: F401
    from .model_descriptor.model_descriptor_factory import ModelDescriptorFactory

    descriptor = ModelDescriptorFactory.get(name)
    if isinstance(descriptor, str):
        raise ValueError(f"Descriptor '{name}' is not registered")
    if hasattr(descriptor, "puzzletron_capabilities"):
        try:
            capabilities = descriptor.puzzletron_capabilities(None)
        except TypeError:
            capabilities = default_capabilities(descriptor_name=name, model_family=name)
    else:
        capabilities = default_capabilities(descriptor_name=name, model_family=name)
    return DescriptorResolution(
        name=name,
        descriptor=descriptor,
        capabilities=capabilities,
        confidence="explicit",
        reason="explicit descriptor name",
    )


def resolve_descriptor_from_pretrained(
    pretrained: str,
    *,
    trust_remote_code: bool = False,
    descriptor_override: str | None = None,
    descriptor_aliases: Mapping[str, str] | None = None,
) -> DescriptorResolution:
    register_native_config_aliases()
    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
    force_cache_dynamic_modules(config, pretrained, trust_remote_code=trust_remote_code)
    return resolve_descriptor(
        config,
        descriptor_override=descriptor_override,
        descriptor_aliases=descriptor_aliases,
    )
