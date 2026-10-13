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

"""This module provides a registry for low-precision GEMM (General Matrix Multiplication) implementations.

The registry allows registering specialized GEMM implementations for different quantization configurations,
providing a central place to manage and match them.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

all = ["gemm_registry"]


class GEMMRegistry:
    """Registry for specialized GEMM (General Matrix Multiplication) implementations.

    This registry allows registering specialized GEMM implementations with custom
    availability checks, giving users control over matching criteria.
    """

    def __init__(self):
        """Initialize the GEMM registry."""
        self._registry = []
        self._enable_real_quant_gemm = False

    def register(
        self,
        gemm_func: Callable,
        availability_check: Callable[
            [torch.nn.Module, torch.Tensor, list, dict[str, torch.Tensor]], bool
        ]
        | None,
    ) -> None:
        """Register a specialized GEMM implementation.

        Args:
            gemm_func: The specialized GEMM function to use.
            availability_check: Function that determines if this implementation
                                should be used. It receives (module, input, args, kwargs)
                                and returns True if the implementation applies. Default check always
                                returns True if None provided
        """

        def default_check(module, input, args, kwargs):
            return True

        self._registry.append(
            {
                "gemm_func": gemm_func,
                "availability_check": availability_check or default_check,
            }
        )

    def unregister(self, gemm_func: Callable):
        """Unregister a GEMM implementation.

        Args:
            gemm_func: The GEMM function to unregister.
        """
        self._registry = [entry for entry in self._registry if entry["gemm_func"] != gemm_func]

    def find_match(self, module: torch.nn.Module, input: Any, *args, **kwargs) -> Callable | None:
        """Find a matching GEMM implementation for the given module and arguments.

        Args:
            module: The module to find a GEMM implementation for.
            input: The input tensor.
            *args, **kwargs: Additional arguments for the GEMM function.

        Returns:
            A matching GEMM function or None if no match is found.
        """
        for entry in self._registry:
            # Let the availability check handle all matching logic
            if entry["availability_check"](module, input, args, kwargs):
                return entry["gemm_func"]

        return None

    def __contains__(self, gemm_func: Callable):
        return any(entry["gemm_func"] == gemm_func for entry in self._registry)


gemm_registry = GEMMRegistry()


def enable_real_quant_gemm(model: nn.Module):
    """Enable real-quant GEMM for the model.

    This function traverses all modules in the model, checks if each module has
    input and weight quantizers that are enabled, and if so, marks the module
    to use the real quantized GEMM implementation.

    Args:
        model: The model to enable real-quant GEMM for.
    """
    for _, module in model.named_modules():
        # Check if the module has input and weight quantizers
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)

        if (
            input_quantizer
            and weight_quantizer
            and input_quantizer.is_enabled
            and weight_quantizer.is_enabled
        ):
            module._use_real_quant_gemm = True


def disable_real_quant_gemm(model: nn.Module):
    """Disable real-quant GEMM for the model.

    This function traverses all modules in the model and disables the use of
    real quantized GEMM implementations.

    Args:
        model: The model to disable real-quant GEMM for.
    """
    for name, module in model.named_modules():
        if hasattr(module, "_use_real_quant_gemm"):
            module._use_real_quant_gemm = False


def is_real_quant_gemm_enabled(model: nn.Module):
    """Check if real-quant GEMM is enabled for the model.

    This function traverses all modules in the model and checks if any module has
    the "_use_real_quant_gemm" attribute set to True.

    Args:
        model: The model to check if real-quant GEMM is enabled for.

    Returns:
        True if real-quant GEMM is enabled for the model, False otherwise.
    """
    return any(
        getattr(module, "_use_real_quant_gemm", False) for _, module in model.named_modules()
    )
