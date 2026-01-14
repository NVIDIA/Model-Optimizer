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

"""Extensible sparse attention module."""

import torch
import torch.nn.functional as F

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls
from modelopt.torch.quantization.utils import replace_function

from .config import SparseAttentionAttributeConfig
from .methods import get_sparse_method


class SparseAttentionModule(DynamicModule):
    """Generic sparse attention module wrapper for applying sparsity to attention layers.

    This module wraps existing attention implementations to add sparse attention
    capabilities by patching torch.nn.functional.softmax.

    Forward Flow:
    -------------
    1. Check if sparse attention is enabled (pass-through if disabled)
    2. Create softmax patch context with sparse_softmax function
    3. Apply sparse attention by patching F.softmax:
       - Patches torch.nn.functional.softmax with sparse_softmax
       - sparse_softmax applies method's sparsity logic before softmax
    4. Forward through original attention with sparsity applied

    Requirements:
    -------------
    - Model must be loaded with attn_implementation="eager" for proper softmax interception
    - Only PyTorch backend is supported (patches F.softmax)

    Attributes:
    -----------
    _enabled: bool
        Whether sparse attention is enabled
    _method: str
        The sparse attention method to use (e.g., "flash_skip_softmax")
    _method_config: dict
        Configuration dictionary for the sparse method (threshold, br, bc, etc.)
    _sparse_method_instance: SparseAttentionMethod
        Instance of the configured sparse attention method
    """

    def set_from_attribute_config(
        self, attribute_cfg: SparseAttentionAttributeConfig | dict | None = None
    ):
        """Set sparse attention attributes from configuration.

        Similar to TensorQuantizer.set_from_attribute_config.

        Args:
            attribute_cfg: Sparse attention attribute configuration.
        """
        # Ensure config is validated through Pydantic
        if not isinstance(attribute_cfg, SparseAttentionAttributeConfig):
            attribute_cfg = SparseAttentionAttributeConfig(**(attribute_cfg or {}))

        # Store raw config for method initialization
        self._method_config = {}

        # Define which attributes are method-specific vs module-specific
        # Module-specific attributes control the SparseAttentionModule behavior
        _module_attributes = {"enable", "method"}

        # Custom setters for special module attributes
        _custom_setters = {
            "enable": ("_enabled", lambda val: bool(val)),
            "method": ("_method", lambda val: str(val)),
        }

        # Process each attribute from validated config
        for attribute, val in attribute_cfg.model_dump().items():
            # Validate attribute if using config class
            if hasattr(SparseAttentionAttributeConfig, "model_fields"):
                assert attribute in SparseAttentionAttributeConfig.model_fields, (
                    f"{attribute} is not a valid SparseAttentionModule attribute"
                )

            if attribute in _module_attributes:
                # Module-level attribute: store with underscore prefix
                attr_name, setter = _custom_setters.get(attribute, (f"_{attribute}", lambda v: v))
                setattr(self, attr_name, setter(val))
            else:
                # Method-specific attribute: store in config dict
                self._method_config[attribute] = val

        # Initialize sparse method instance
        self._init_sparse_method()

    def _init_sparse_method(self):
        """Initialize the sparse method instance."""
        method_class = get_sparse_method(self._method)

        # Initialize the sparse method instance
        # _method_config is always initialized in set_from_attribute_config
        self._sparse_method_instance = method_class(method_config=self._method_config)  # type: ignore[call-arg]

    def enable(self):
        """Enable sparse attention for this module."""
        self._enabled = True

    def disable(self):
        """Disable sparse attention for this module."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if sparse attention is enabled."""
        return getattr(self, "_enabled", True)

    def get_stats(self) -> dict:
        """Get sparsity statistics from the stats manager.

        Returns:
            Dictionary with sparsity statistics including 'average_sparsity' if available.
            Returns empty dict (statistics collection will be added in calibration PR).
        """
        # TODO: Statistics collection will be added in calibration PR
        return {}

    def _setup(self):
        """Setup called by DynamicModule."""
        # Apply default configuration if not yet configured
        if not hasattr(self, "_method"):
            self.set_from_attribute_config(None)

    def forward(self, *args, **kwargs):
        """Forward with selected sparse attention method.

        This method dispatches to the appropriate sparse attention implementation
        based on the configured method and backend.
        """
        # Pass through if sparse attention is disabled
        if not self.is_enabled:
            return super().forward(*args, **kwargs)

        # Get the appropriate context manager for this configuration
        context = self._get_sparse_context()

        # Apply sparse attention through the context
        with context:
            result = super().forward(*args, **kwargs)

        return result

    def _get_sparse_context(self):
        """Get the softmax patch context for applying sparse attention."""
        return self._create_softmax_patch_context()

    def _create_softmax_patch_context(self):
        """Create context manager for patching softmax function."""
        return replace_function(torch.nn.functional, "softmax", self._create_sparse_softmax())

    def _create_sparse_softmax(self):
        """Create sparse softmax function for current method."""
        original_softmax = F.softmax

        def sparse_softmax(input, dim=-1, *args, **kwargs):
            # Let the method handle the sparsification
            _, _, _, sparse_input = self._sparse_method_instance.apply_sparsity(
                None, None, None, input
            )

            # Use sparse input if modified, otherwise use original
            if sparse_input is not None:
                return original_softmax(sparse_input, dim, *args, **kwargs)
            return original_softmax(input, dim, *args, **kwargs)

        return sparse_softmax


# Create registry for sparse attention modules
SparseAttentionRegistry = _DMRegistryCls("SparseAttention", SparseAttentionModule)
