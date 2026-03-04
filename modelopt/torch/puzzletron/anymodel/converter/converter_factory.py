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

from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor

__all__ = ["ConverterFactory"]


class ConverterFactory:
    """Factory for registering and retrieving Converter classes."""

    CLASS_MAPPING = {}

    @classmethod
    def register(cls, **entries: Type):
        """Register converter classes.

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
        """Get a registered converter by name or return the converter if already resolved."""
        if isinstance(value, str):
            if value in cls.CLASS_MAPPING:
                return cls.CLASS_MAPPING[value]
        return value
