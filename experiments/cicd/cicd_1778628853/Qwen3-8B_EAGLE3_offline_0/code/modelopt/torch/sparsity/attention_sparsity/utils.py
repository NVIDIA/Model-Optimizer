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

"""Utility functions for sparse attention module discovery."""

import torch.nn as nn

from .sparse_attention import SparseAttentionModule


def get_sparse_attention_modules(model: nn.Module) -> list[SparseAttentionModule]:
    """Get all sparse attention modules in a model.

    Args:
        model: Model to search for sparse attention modules.

    Returns:
        List of SparseAttentionModule instances found in the model.
    """
    return [m for m in model.modules() if isinstance(m, SparseAttentionModule)]


def get_named_sparse_attention_modules(
    model: nn.Module,
) -> list[tuple[str, SparseAttentionModule]]:
    """Get all sparse attention modules in a model with their names.

    Args:
        model: Model to search for sparse attention modules.

    Returns:
        List of (name, module) tuples for all SparseAttentionModule instances.
    """
    return [(name, m) for name, m in model.named_modules() if isinstance(m, SparseAttentionModule)]
