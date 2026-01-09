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

"""Dynamic sparse attention registration for HuggingFace models."""

import torch.nn as nn
import transformers

from modelopt.torch.opt.dynamic import DynamicModule

from ..sparse_attention import SparseAttentionModule, SparseAttentionRegistry


class _GenericSparseAttention(SparseAttentionModule):
    """Generic sparse attention that works with any HF attention module.

    This class provides a universal sparse attention wrapper that can
    work with various transformer attention implementations.
    """

    def _setup(self):
        """Setup sparse attention for any attention type.

        The base SparseAttentionModule handles detection and initialization.
        """
        super()._setup()

    def get_attn_type(self, attn_module) -> type:
        """Get the original attention type.

        Args:
            attn_module: Attention module (possibly wrapped)

        Returns:
            Original class type
        """
        # If this is a DynamicModule, get the original class
        if isinstance(attn_module, DynamicModule):
            return attn_module.get_original_cls_by_level(level=0)
        return type(attn_module)


def register_sparse_attention_on_the_fly(model: nn.Module) -> bool:
    """Dynamically register sparse attention for any model.

    This function automatically detects attention modules in the model
    and registers them for sparse attention optimization.

    Args:
        model: Model to process

    Returns:
        True if any modules were registered
    """
    if not _is_supported_model(model):
        return False

    registered_count = 0
    attention_types = set()

    for name, module in model.named_modules():
        # Skip if already a sparse attention module
        if isinstance(module, SparseAttentionModule):
            continue

        # Check if this is an attention module by name
        module_type = type(module)
        type_name = module_type.__name__

        # Common attention module patterns
        is_attention = "attention" in type_name.lower() or type_name.endswith(
            ("Attention", "SelfAttention")
        )

        if is_attention and module_type not in SparseAttentionRegistry:
            # Register attention type
            if module_type not in attention_types:
                SparseAttentionRegistry.register({module_type: type_name})(_GenericSparseAttention)
                attention_types.add(module_type)
                registered_count += 1
                print(f"Registered {type_name} for sparse attention optimization")

    if registered_count > 0:
        print(f"Dynamically registered {registered_count} attention module types for sparsity")

    return registered_count > 0


def _is_supported_model(model: nn.Module) -> bool:
    """Check if model is supported for sparse attention.

    Supports HuggingFace PreTrainedModel and any PyTorch model with attention modules.

    Args:
        model: Model to check

    Returns:
        True if model is supported
    """
    # Check for HuggingFace PreTrainedModel
    try:
        if isinstance(model, transformers.PreTrainedModel):
            return True
    except ImportError:
        pass

    # Support any PyTorch model with attention modules
    return isinstance(model, nn.Module)
