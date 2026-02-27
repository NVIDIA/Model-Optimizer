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
# mypy: ignore-errors

import inspect
from typing import Callable, Type

from transformers import AutoConfig

from modelopt.torch.puzzletron.anymodel.model_descriptor.model_descriptor import ModelDescriptor

__all__ = ["ModelDescriptorFactory"]

# Map from HuggingFace config.model_type (in checkpoint config.json) to ModelDescriptorFactory name.
# Local to this script; add entries when supporting new model types for auto-detection.
_MODEL_TYPE_TO_DESCRIPTOR = {
    "llama": "llama",
    "mistral": "mistral_small",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "nemotron_h": "nemotron_h",
    "nemotron_h_v2": "nemotron_h_v2",
    "gpt_oss_20b": "gpt_oss_20b",
}


def resolve_descriptor_from_pretrained(pretrained: str | None, trust_remote_code: bool = True):
    """Resolve the model descriptor by loading the checkpoint config and mapping model_type."""
    if not pretrained:
        raise ValueError("pretrained must be provided")

    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", None)

    if model_type and model_type in _MODEL_TYPE_TO_DESCRIPTOR:
        detected = _MODEL_TYPE_TO_DESCRIPTOR[model_type]
        print(
            f"[resolve_descriptor_from_pretrained] Auto-detected model_type='{model_type}' → descriptor='{detected}'"
        )
        return ModelDescriptorFactory.get(detected)

    known = sorted(_MODEL_TYPE_TO_DESCRIPTOR.keys())
    raise ValueError(
        f"Cannot auto-detect descriptor for model_type='{model_type}'. "
        f"Known model types: {known}. Add this model_type to _MODEL_TYPE_TO_DESCRIPTOR if supported."
    )


class ModelDescriptorFactory:
    """Factory for registering and retrieving ModelDescriptor classes."""

    CLASS_MAPPING = {}

    @classmethod
    def register(cls, **entries: Type):
        """Register model descriptor classes.

        Raises:
            KeyError: if entry key is already in type_dict and points to a different class.
        """
        for cls_name, cls_type in entries.items():
            if cls_name in cls.CLASS_MAPPING:
                ref = cls.CLASS_MAPPING[cls_name]
                # If ref and cls_name point to the same class ignore and don't raise an exception.
                if cls_type == ref:
                    continue
                raise KeyError(
                    f"Could not register `{cls_name}`: {cls_type}, "
                    f"`{cls_name}` is already registered and points to "
                    f"`{inspect.getmodule(ref).__name__}.{ref.__name__}`"
                )
            cls.CLASS_MAPPING[cls_name] = cls_type

    @classmethod
    def register_decorator(cls, name: str | None) -> Callable:
        """Set up a register decorator.

        Args:
            name: If specified, the decorated object will be registered with this name.

        Returns:
            Decorator that registers the callable.
        """

        def decorator(cls_type: Type) -> Callable:
            """Register the decorated callable."""
            cls_name = name if name is not None else cls_type.__name__
            cls.register(**{cls_name: cls_type})
            return cls_type

        return decorator

    @classmethod
    def get(cls, value: str | ModelDescriptor):
        """Get a registered model descriptor by name or return the descriptor if already resolved."""
        if isinstance(value, str):
            if value in cls.CLASS_MAPPING:
                return cls.CLASS_MAPPING[value]
        return value
