# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Patch Megatron Bridge provider so provide() runs under heterogeneous patchers when block_configs is set.

Apply this patch so that when a provider (GPTModelProvider, MambaModelProvider, etc.) has
block_configs set, calling provide() or provide_distributed_model() builds the model with
per-layer config overrides and no_op support. No changes to Megatron-Bridge or Megatron-LM
are required.
"""

from __future__ import annotations

import functools
import logging
import types
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)

_MBRIDGE_PROVIDER_PATCH_APPLIED = "_mbridge_provider_patch_applied"


def load_block_configs(block_configs_path: Union[str, Path]) -> List[dict]:
    """Load block configs from a JSON file (e.g. block_configs.json)."""
    path = Path(block_configs_path)
    if not path.exists():
        raise FileNotFoundError(f"Block configs not found: {path}")
    import json
    with open(path) as f:
        data = json.load(f)
    if "block_configs" in data:
        return data["block_configs"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected 'block_configs' key or list in {path}")


def _patched_provide(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Patched provide method that wraps execution in mbridge patchers."""
    from mbridge_gpt_patcher import mbridge_gpt_patcher
    from mbridge_mamba_patcher import mbridge_mamba_patcher

    # Retrieve the original method
    # If patched on instance, it's on self. If patched on class, it's looked up on self via MRO.
    orig = getattr(self, "_mbridge_orig_provide", None)
    if orig is None:
        # Fallback: try to find it on the class if not found (shouldn't happen if patched correctly)
        # But if we are here, we must have been called.
        raise RuntimeError("Original provide method not found on provider instance")

    block_configs = getattr(self, "block_configs", None)
    
    # Helper to call original
    def call_orig():
        if hasattr(orig, "__self__"): # Bound method (instance patch)
            return orig(*args, **kwargs)
        else: # Unbound function (class patch)
            return orig(self, *args, **kwargs)

    if not block_configs:
        return call_orig()

    print(
        f"[mbridge] _patched_provide called; block_configs has {len(block_configs)} entries",
        flush=True,
    )

    num_attention_heads = getattr(self, "num_attention_heads", None) or getattr(
        self, "num_heads", None
    )
    hidden_size = getattr(self, "hidden_size", None)
    if num_attention_heads is None or hidden_size is None:
        print(
            f"[mbridge] block_configs set but num_attention_heads={num_attention_heads!r} or hidden_size={hidden_size!r} missing; skipping patchers",
            flush=True,
        )
        logger.warning(
            "block_configs set but num_attention_heads or hidden_size missing on provider; "
            "skipping heterogeneous patchers"
        )
        return call_orig()

    # For Mamba/hybrid models (MambaStack), each layer slot is already a single dedicated
    # layer type (mamba, attention, mlp, or moe) whose spec handles the no_op intrinsically.
    # Applying IdentityOp replacements on top would break the forward pass (IdentityOp
    # returns a plain tensor, but self_attention is expected to return a (output, bias) tuple).
    is_mamba_model = "Mamba" in type(self).__name__
    apply_no_ops = not is_mamba_model

    print(
        f"[mbridge] entering mbridge_gpt_patcher with {len(block_configs)} block configs"
        f" (apply_no_ops={apply_no_ops})",
        flush=True,
    )
    logger.info(
        "Using mbridge heterogeneous patchers with %d block configs (apply_no_ops=%s)",
        len(block_configs) if block_configs else 0,
        apply_no_ops,
    )
    with mbridge_gpt_patcher(
        block_configs=block_configs,
        num_attention_heads=int(num_attention_heads),
        hidden_size=int(hidden_size),
        apply_no_ops=apply_no_ops,
    ):
        with mbridge_mamba_patcher(
            block_configs=block_configs,
            num_attention_heads=int(num_attention_heads),
            hidden_size=int(hidden_size),
        ):
            return call_orig()


def apply_patch() -> None:
    """Patch ModelProviderMixin.provide so that when provider.block_configs is set,
    model building runs inside mbridge_gpt_patcher and mbridge_mamba_patcher contexts.
    """
    from megatron.bridge.models.model_provider import ModelProviderMixin

    if getattr(ModelProviderMixin, _MBRIDGE_PROVIDER_PATCH_APPLIED, False):
        logger.debug("Mbridge provider patch already applied")
        return

    _orig_provide = ModelProviderMixin.provide
    
    # Store original on the class
    setattr(ModelProviderMixin, "_mbridge_orig_provide", _orig_provide)
    ModelProviderMixin.provide = _patched_provide
    setattr(ModelProviderMixin, _MBRIDGE_PROVIDER_PATCH_APPLIED, True)
    logger.info("Applied mbridge heterogeneous patch to ModelProviderMixin.provide")


def remove_patch() -> None:
    """Restore ModelProviderMixin.provide to its original implementation."""
    from megatron.bridge.models.model_provider import ModelProviderMixin

    if not getattr(ModelProviderMixin, _MBRIDGE_PROVIDER_PATCH_APPLIED, False):
        logger.debug("Mbridge provider patch was not applied")
        return

    orig = getattr(ModelProviderMixin, "_mbridge_orig_provide", None)
    if orig is not None:
        ModelProviderMixin.provide = orig
    if hasattr(ModelProviderMixin, "_mbridge_orig_provide"):
        delattr(ModelProviderMixin, "_mbridge_orig_provide")
    delattr(ModelProviderMixin, _MBRIDGE_PROVIDER_PATCH_APPLIED)
    logger.info("Removed mbridge heterogeneous patch from ModelProviderMixin.provide")


def set_provider_block_configs(
    provider: Any,
    block_configs: Optional[List[Union[dict, Any]]],
    block_configs_path: Optional[Union[str, Path]] = None,
) -> None:
    """Attach block_configs to a provider (e.g. from bridge.to_megatron_provider()).

    Call this after obtaining the provider if you want heterogeneous layer config.
    Either pass block_configs directly or block_configs_path to load from JSON.
    
    Also ensures the provider instance is patched to use the heterogeneous builder,
    even if the provider class overrides provide() and thus skipped the ModelProviderMixin patch.

    Args:
        provider: GPTModelProvider, MambaModelProvider, or any ModelProviderMixin.
        block_configs: List of block config dicts (or BlockConfig-like), or None.
        block_configs_path: If set and block_configs is None, load from this path.
    """
    if block_configs_path is not None and block_configs is None:
        block_configs = load_block_configs(block_configs_path)
    provider.block_configs = block_configs
    
    # Ensure provider.provide is patched.
    # If provider's class overrode provide(), it won't be using ModelProviderMixin.provide (which we patched).
    # So we check if the current provide method is our _patched_provide.
    
    current_provide = getattr(provider.provide, "__func__", provider.provide)
    if current_provide != _patched_provide:
        logger.info(
            "Provider instance's provide method is not patched (likely overridden by subclass). "
            "Patching instance method."
        )
        # Store original bound method
        provider._mbridge_orig_provide = provider.provide
        # Patch instance with bound method
        provider.provide = types.MethodType(_patched_provide, provider)
