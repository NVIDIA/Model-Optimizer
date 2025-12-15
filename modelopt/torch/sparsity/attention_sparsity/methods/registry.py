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

"""Registry and base class for sparse attention methods."""

import re
import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch


class SparseAttentionMethod(ABC):
    """Base class for sparse attention methods."""

    # Flag to indicate calibration mode (set by calibrator)
    _calibration_mode: bool = False

    @abstractmethod
    def calculate_sparsity(
        self,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Calculate sparsity mask and statistics without applying.

        Args:
            attention_scores: Pre-softmax attention scores [batch, heads, seq_q, seq_k]

        Returns:
            Tuple of (sparse_mask, stats_dict) where:
            - sparse_mask: Boolean tensor indicating which elements to keep
            - stats_dict: Dictionary with sparsity statistics
        """

    @abstractmethod
    def apply_sparsity(
        self,
        attention_scores: torch.Tensor,
        sparse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparsity mask to attention scores.

        Args:
            attention_scores: Pre-softmax attention scores [batch, heads, seq_q, seq_k]
            sparse_mask: Optional pre-computed mask. If None, calculates internally.

        Returns:
            Masked attention scores with sparse elements set to -inf
        """

    def get_threshold_info(self) -> dict[str, Any]:
        """Get threshold information for display/debugging.

        Returns:
            Dictionary with threshold information. Should include:
            - 'type': 'static', 'dynamic', or 'none'
            - 'value': threshold value (for static)
            - 'scale_factor': scale factor (for dynamic)
            - Other method-specific info
        """
        return {"type": "none", "value": None}

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name identifier."""


# Method Registry with versioning support
_SPARSE_ATTENTION_METHODS: dict[str, dict[str, type[SparseAttentionMethod]]] = {}


def _version_key(version_str: str) -> list[int]:
    """Extract numeric parts for proper version sorting.

    Args:
        version_str: Version string (e.g., "v1", "v2", "v10")

    Returns:
        List of integers extracted from version string for sorting

    Examples:
        >>> _version_key("v1")
        [1]
        >>> _version_key("v10")
        [10]
        >>> _version_key("v2.3.1")
        [2, 3, 1]
    """
    parts = re.findall(r"\d+", version_str)
    return [int(p) for p in parts] if parts else [0]


def register_sparse_method(name: str, version: str = "v1"):
    """Decorator to register sparse attention methods with version support.

    Args:
        name: Method name to register
        version: Version string (default: "v1")

    Example::

        @register_sparse_method("my_method", version="v3")
        class MyMethodV3(SparseAttentionMethod): ...
    """

    def decorator(cls: type[SparseAttentionMethod]):
        if name not in _SPARSE_ATTENTION_METHODS:
            _SPARSE_ATTENTION_METHODS[name] = {}

        if version in _SPARSE_ATTENTION_METHODS[name]:
            warnings.warn(
                f"Overriding existing sparse attention method: {name}@{version}",
                RuntimeWarning,
                stacklevel=2,
            )

        _SPARSE_ATTENTION_METHODS[name][version] = cls
        return cls

    return decorator


def get_sparse_method(name: str, version: str | None = None) -> type[SparseAttentionMethod]:
    """Get sparse attention method by name and optional version.

    Args:
        name: Method name to retrieve
        version: Optional version string. If None, uses latest version.

    Returns:
        Method class

    Raises:
        ValueError: If method name or version is not registered

    Example:
        >>> get_sparse_method("flash_skip_softmax")  # Latest version
        >>> get_sparse_method("flash_skip_softmax", "v1")  # Specific version
    """
    if name not in _SPARSE_ATTENTION_METHODS:
        available = list(_SPARSE_ATTENTION_METHODS.keys())
        raise ValueError(f"Unknown sparse attention method: {name}. Available: {available}")

    method_versions = _SPARSE_ATTENTION_METHODS[name]

    if not version:
        version = sorted(method_versions.keys(), key=_version_key)[-1]

    if version not in method_versions:
        available_versions = list(method_versions.keys())
        raise ValueError(
            f"Unknown version {version} for method {name}. Available: {available_versions}"
        )

    return method_versions[version]
