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

"""Configuration classes for PEFT methods."""

import importlib
import inspect
from collections.abc import Callable
from typing import Annotated, Any

import torch.nn.init as init
from pydantic import PlainSerializer, WithJsonSchema, field_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = ["ExportPEFTConfig", "PEFTAttributeConfig", "PEFTConfig"]

InitRuntimeType = Any


def _qualname(fn) -> str:
    m = inspect.getmodule(fn)
    return f"{m.__name__}.{fn.__name__}" if m else getattr(fn, "__name__", str(fn))


InitField = Annotated[
    InitRuntimeType,
    WithJsonSchema(
        {
            "type": "string",
            "title": "torch initializer",
            "description": (
                "Fully-qualified callable from ``torch.nn.init``. "
                "Must be in-place (name ends with ``\\_``)."
            ),
            "examples": ["torch.nn.init.zeros\\_", "torch.nn.init.kaiming_uniform\\_"],
        }
    ),
    PlainSerializer(lambda v: _qualname(v), return_type=str, when_used="always"),
]


class PEFTAttributeConfig(ModeloptBaseConfig):
    """Configuration for PEFT adapter attributes."""

    enable: bool = ModeloptField(
        default=True,
        title="Enable adapter",
        description="If True, enables the adapter. If False, by-passes the adapter.",
    )

    rank: int = ModeloptField(
        default=64,
        title="LoRA rank",
        description=(
            "The rank (dimension) of the LoRA matrices. "
            "Higher rank allows more expressiveness but uses more memory."
        ),
    )

    scale: float = ModeloptField(
        default=1.0,
        title="LoRA scaling factor",
        description="Scaling factor for the LoRA output. Controls the magnitude of the adaptation.",
    )

    lora_a_init: InitField = ModeloptField(
        default=init.kaiming_uniform_,
        title="LoRA A matrix initializer",
        description="Initializer from ``torch.nn.init`` (in-place; name ends with ``\\_``).",
    )

    lora_b_init: InitField = ModeloptField(
        default=init.zeros_,
        title="LoRA B matrix initializer",
        description="Initializer from ``torch.nn.init`` (in-place; name ends with ``\\_``).",
    )

    @field_validator("lora_a_init", "lora_b_init", mode="before")
    @classmethod
    def _parse_init_callable(cls, v):
        if isinstance(v, str):
            try:
                module_path, func_name = v.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                v = getattr(mod, func_name)
            except Exception as e:
                raise ValueError(
                    f"Could not resolve initializer '{v}' into a callable "
                    "(expected a dotted path like 'torch.nn.init.zeros_')."
                ) from e
        return v

    @field_validator("lora_a_init", "lora_b_init")
    @classmethod
    def validate_init_method(cls, v):
        """Validate initialization method is supported."""
        if callable(v):
            module = inspect.getmodule(v)
            if module is not init:
                raise ValueError(
                    "Callable initialization method must be from torch.nn.init, "
                    f"got {module.__name__ if module else 'unknown'}"
                )
            func_name = getattr(v, "__name__", "")
            if not func_name.endswith("_"):
                raise ValueError(
                    "Initialization method must be in-place (name ends with '_'). "
                    "For example: ``torch.nn.init.kaiming_uniform\\_`` not "
                    "``torch.nn.init.kaiming_uniform``."
                )
        else:
            raise ValueError(
                f"Initialization method must be a callable function from torch.nn.init, got {type(v)}"
            )
        return v

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, v):
        """Validate rank is positive."""
        if v < 1:
            raise ValueError("rank must be a positive integer")
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v):
        """Validate scale is positive."""
        if v <= 0:
            raise ValueError("scale must be a positive number")
        return v


# Type alias for adapter configuration
PEFTAdapterCfgType = dict[str | Callable, PEFTAttributeConfig | dict]


class PEFTConfig(ModeloptBaseConfig):
    """Default configuration for ``peft`` mode.

    For adapter_cfg, later patterns override earlier ones, for example::

        "adapter_cfg": {
            "*": {
                "rank": 32,
                "scale": 1,
                "enable": True,
            },
            "*output_layer*": {"enable": False},
        }

    If a layer name matches ``"*output_layer*"``, the attributes will be replaced with ``{"enable": False}``.
    """

    adapter_name: str = ModeloptField(
        default="default",
        title="Adapter name",
        description="Name of the adapter to create or update.",
        validate_default=True,
    )

    adapter_cfg: PEFTAdapterCfgType = ModeloptField(
        default={"*": {"rank": 64}},
        title="Adapter configuration",
        description="Configuration for adapters. Maps module patterns to PEFTAttributeConfig or dict.",
        validate_default=True,
    )

    adapter_type: str = ModeloptField(
        default="lora",
        title="Adapter type",
        description="Type of PEFT adapter to use. Currently only 'lora' is supported.",
        validate_default=True,
    )

    freeze_base_model: bool = ModeloptField(
        default=True,
        title="Freeze base weights during training",
        description="Whether to freeze the base model weights; in most cases, this should be set to True.",
        validate_default=True,
    )

    freeze_lora_weights: bool = ModeloptField(
        default=False,
        title="Freeze lora weights during training",
        description="Whether to freeze the lora model weights; in most cases, this should be set to False.",
        validate_default=True,
    )

    @field_validator("adapter_type")
    @classmethod
    def validate_adapter_type(cls, v):
        """Validate adapter type."""
        if v not in ["lora"]:
            raise ValueError(f"Unsupported adapter type: {v}. Only 'lora' is currently supported.")
        return v

    @field_validator("adapter_cfg")
    @classmethod
    def validate_adapter_cfg(cls, v):
        """Validate and convert adapter configurations."""
        validated_cfg = {}
        for key, value in v.items():
            if isinstance(value, dict) and not isinstance(value, PEFTAttributeConfig):
                # Convert dict to PEFTAttributeConfig to trigger validation
                try:
                    validated_cfg[key] = PEFTAttributeConfig(**value)
                except Exception as e:
                    raise ValueError(f"Invalid adapter configuration for '{key}': {e}")
            else:
                validated_cfg[key] = value
        return validated_cfg


class ExportPEFTConfig(ModeloptBaseConfig):
    """An empty config."""
