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

from typing import Any

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls

from .config import SparseAttentionAttributeConfig
from .methods import get_sparse_method
from .stats_manager import SparseAttentionStatsManager


class SparseAttentionModule(DynamicModule):
    """Generic sparse attention module wrapper for applying sparsity to attention layers.

    This module wraps existing attention implementations to add sparse attention
    capabilities. The activation mechanism is delegated to the configured method
    via ``method.get_sparse_context(module)``, so each method defines how it
    integrates with the forward pass (e.g. softmax patching, kernel flags).

    Forward Flow:
    -------------
    1. Check if sparse attention is enabled (pass-through if disabled)
    2. Obtain method-specific context via ``_sparse_method_instance.get_sparse_context(self)``
    3. Run the original forward inside the context
    4. Collect statistics if stats manager is enabled

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
        from .config import VSAAttributeConfig

        # Determine which config class to use based on method
        config_dict = attribute_cfg or {}
        if isinstance(attribute_cfg, dict):
            method = config_dict.get("method", "flash_skip_softmax")
        elif attribute_cfg is not None and hasattr(attribute_cfg, "method"):
            method = attribute_cfg.method
        else:
            method = "flash_skip_softmax"

        # Select appropriate config class based on method
        if method == "vsa":
            config_class = VSAAttributeConfig
        else:
            config_class = SparseAttentionAttributeConfig

        # Ensure config is validated through Pydantic
        if not isinstance(attribute_cfg, (SparseAttentionAttributeConfig, VSAAttributeConfig)):
            attribute_cfg = config_class(**(config_dict))

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
            # Validate attribute against the appropriate config class
            if hasattr(config_class, "model_fields"):
                assert attribute in config_class.model_fields, (
                    f"{attribute} is not a valid {config_class.__name__} attribute"
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

        # Create stats manager based on config
        if self._method_config.get("collect_stats", False):
            self._stats_manager = SparseAttentionStatsManager(
                module_name="sparse_attention", enabled=True
            )
        else:
            self._stats_manager = None

        # Initialize stats storage for collecting stats from sparse_softmax
        self._last_stats: dict | None = None

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
            Returns empty dict if stats manager is not enabled.
        """
        if self._stats_manager is not None and self._stats_manager.enabled:
            return self._stats_manager.get_summary()
        return {}

    def get_threshold_info(self) -> dict[str, Any]:
        """Get threshold information from the sparse method instance.

        Returns:
            Dictionary with threshold information from the sparse method.
        """
        if hasattr(self, "_sparse_method_instance") and self._sparse_method_instance is not None:
            return self._sparse_method_instance.get_threshold_info()
        return {"type": "none", "value": None}

    def _setup(self):
        """Setup called by DynamicModule."""
        # Apply default configuration if not yet configured
        if not hasattr(self, "_method"):
            self.set_from_attribute_config(None)

    def forward(self, *args, **kwargs):
        """Forward with selected sparse attention method.

        - VSA: patches ``F.scaled_dot_product_attention`` to intercept the SDPA
          call inside the original forward. Cross-attention is skipped.
        - Softmax-patching methods (e.g. ``flash_skip_softmax``): use the
          context manager path below.
        """
        # Pass through if sparse attention is disabled
        if not self.is_enabled:
            return super().forward(*args, **kwargs)

        # VSA: patch F.scaled_dot_product_attention so the VSA kernel intercepts
        # the SDPA call inside the original forward. This works for diffusers models
        # since SDPA is the common attention primitive.
        # Only self-attention is replaced. Cross-attention (Q/K have different seq_len) is skipped.
        if self._method == "vsa":
            result = self._forward_with_vsa_sdpa_patch(args, kwargs)

            if self._stats_manager is not None and self._last_stats is not None:
                self._stats_manager.collect(self._last_stats)
                self._last_stats = None
            return result

        # Standard path: softmax patching
        context = self._get_sparse_context()

        # Apply sparse attention through the context
        with context:
            result = super().forward(*args, **kwargs)

        # Collect stats if manager is available
        if self._stats_manager is not None and self._last_stats is not None:
            self._stats_manager.collect(self._last_stats)
            self._last_stats = None  # Clear after collection

        return result

    def _forward_with_vsa_sdpa_patch(self, args, kwargs):
        """Run forward with F.scaled_dot_product_attention patched for VSA.

        Replaces SDPA with the VSA kernel for self-attention calls (Q and K/V
        have the same seq_len).  Cross-attention calls fall through to the
        original SDPA.  Warns if SDPA was never called.
        """
        import torch.nn.functional as F

        from modelopt.torch.quantization.utils import replace_function

        vsa = self._sparse_method_instance
        original_sdpa = F.scaled_dot_product_attention
        self._vsa_sdpa_called = False

        def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
            self._vsa_sdpa_called = True

            # Fall back to original SDPA when VSA cannot handle this call:
            # - Cross-attention: Q and K/V have different seq_len
            # - video_shape not set: VSA cannot compute tile metadata
            # - seq_len mismatch: input doesn't match the configured video shape
            can_apply_vsa = (
                vsa.video_shape is not None
                and query.shape[2] == key.shape[2]
                and query.shape[2] == vsa.video_shape[0] * vsa.video_shape[1] * vsa.video_shape[2]
            )
            if not can_apply_vsa:
                return original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    **kw,
                )
            output, stats = vsa.forward_attention(query, key, value)
            self._last_stats = stats
            return output

        with replace_function(F, "scaled_dot_product_attention", _patched_sdpa):
            result = super().forward(*args, **kwargs)

        if not self._vsa_sdpa_called:
            import warnings

            warnings.warn(
                f"VSA: F.scaled_dot_product_attention was not called during "
                f"{type(self).__name__}.forward(). The attention layer may use a "
                f"custom kernel that bypasses SDPA. VSA had no effect on this layer.",
            )

        return result

    def _get_sparse_context(self):
        """Get the context manager for applying sparse attention.

        Delegates to the method instance so each method defines its own
        activation mechanism (softmax patching, kernel flags, etc.).
        """
        return self._sparse_method_instance.get_sparse_context(self)


# Create registry for sparse attention modules
SparseAttentionRegistry = _DMRegistryCls("SparseAttention", SparseAttentionModule)
