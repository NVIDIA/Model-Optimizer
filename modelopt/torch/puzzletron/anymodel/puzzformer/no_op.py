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

"""No-op modules for replacing layers during pruning."""

from functools import cache

import torch
import torch.nn as nn


@cache
def return_tuple_of_size(cls: type[nn.Module], size: int) -> type[nn.Module]:
    """Create a wrapper class that returns a tuple of the given size.

    Useful for replacing modules that return multiple outputs (e.g., attention layers
    that return (hidden_states, attn_weights)).

    Args:
        cls: The base module class to wrap.
        size: The size of the tuple to return.

    Returns:
        A new class that wraps the base class and returns a tuple of the given size.

    Example:
        >>> decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()
    """

    class Wrapped(cls):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            outputs = [None] * size
            outputs[0] = result[0]
            return tuple(outputs)

        def extra_repr(self) -> str:
            return f"[{cls.__name__}]"

    return Wrapped


class MatchingZeros(nn.Module):
    """Module that returns zeros matching the input shape.

    Used to replace MLP or attention layers with no-ops. Returns zeros because
    the hidden_states are added to the residuals, so a no-op implementation
    should leave the residual unchanged.
    """

    def forward(self, hidden_states, *args, **kwargs):
        return torch.zeros_like(hidden_states)


class Same(nn.Module):
    """Module that returns the input unchanged.

    Used to replace normalization layers with identity operations.
    """

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states

    @property
    def weight(self):
        """Support NemotronH with scoring_activations, when lm_head is called `self.lm_head.weight.dtype`."""
        return torch.empty(0)
