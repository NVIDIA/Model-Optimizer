# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Alexey Kravtsov

from typing import NamedTuple


class ContentReplacement(NamedTuple):
    content: str
    replacement: str


class CompulsoryContent(NamedTuple):
    prefix: str = ""
    suffix: str = ""


class SDKeyOps(NamedTuple):
    """Immutable class representing state dict key operations."""

    name: str
    mapping: tuple[ContentReplacement | CompulsoryContent, ...]  # Immutable tuple of (key, value) pairs


# Predefined SDKeyOps instances
LTXV_MODEL_COMFY_RENAMING_MAP = SDKeyOps(
    name="LTXV_MODEL_COMFY_PREFIX_MAP",
    mapping=(
        CompulsoryContent(prefix="model.diffusion_model."),
        ContentReplacement("model.diffusion_model.", ""),
    ),
)

LTXV_LORA_COMFY_RENAMING_MAP = SDKeyOps(
    name="LTXV_LORA_COMFY_PREFIX_MAP",
    mapping=(ContentReplacement("diffusion_model.", ""),),
)

LTXV_LORA_COMFY_TARGET_MAP = SDKeyOps(
    name="LTXV_LORA_COMFY_TARGET_MAP",
    mapping=(
        ContentReplacement("diffusion_model.", ""),
        ContentReplacement(".lora_A.weight", ".weight"),
        ContentReplacement(".lora_B.weight", ".weight"),
    ),
)

VOCODER_COMFY_KEYS_FILTER = SDKeyOps(
    name="VOCODER_COMFY_KEYS_FILTER",
    mapping=(CompulsoryContent(prefix="vocoder."), ContentReplacement("vocoder.", "")),
)
