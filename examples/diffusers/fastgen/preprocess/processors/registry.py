# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Processor registry for model-agnostic preprocessing.

This module provides a registry pattern for discovering and instantiating
model-specific processors at runtime.
"""

from .base import BaseModelProcessor


class ProcessorRegistry:
    """
    Registry for model processors.

    Allows registering processor classes by name and retrieving them at runtime.
    Uses a decorator pattern for easy registration.

    Example:
        @ProcessorRegistry.register("flux")
        class FluxProcessor(BaseModelProcessor):
            ...

        # Later
        processor = ProcessorRegistry.get("flux")
    """

    _processors: dict[str, type[BaseModelProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a processor class.

        Args:
            name: Name to register the processor under (e.g., 'flux', 'sdxl')

        Returns:
            Decorator function

        Example:
            @ProcessorRegistry.register("my_model")
            class MyModelProcessor(BaseModelProcessor):
                ...
        """

        def decorator(processor_class: type[BaseModelProcessor]):
            if not issubclass(processor_class, BaseModelProcessor):
                raise TypeError(
                    f"Processor {processor_class.__name__} must inherit from BaseModelProcessor"
                )
            cls._processors[name] = processor_class
            return processor_class

        return decorator

    @classmethod
    def get(cls, name: str) -> BaseModelProcessor:
        """
        Get a processor instance by name.

        Args:
            name: Registered processor name

        Returns:
            Instantiated processor

        Raises:
            ValueError: If processor name is not registered
        """
        if name not in cls._processors:
            available = ", ".join(sorted(cls._processors.keys()))
            raise ValueError(f"Unknown processor: '{name}'. Available processors: {available}")
        return cls._processors[name]()

    @classmethod
    def get_class(cls, name: str) -> type[BaseModelProcessor]:
        """
        Get a processor class by name (without instantiating).

        Args:
            name: Registered processor name

        Returns:
            Processor class

        Raises:
            ValueError: If processor name is not registered
        """
        if name not in cls._processors:
            available = ", ".join(sorted(cls._processors.keys()))
            raise ValueError(f"Unknown processor: '{name}'. Available processors: {available}")
        return cls._processors[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all registered processor names.

        Returns:
            List of registered processor names
        """
        return sorted(cls._processors.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a processor is registered.

        Args:
            name: Processor name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._processors
