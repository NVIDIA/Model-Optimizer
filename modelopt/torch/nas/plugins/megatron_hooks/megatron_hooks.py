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
"""Megatron-specific hooks with tensor parallelism support."""

import torch
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region

from .base_hooks import L2NormHook

__all__ = ["MegatronL2NormHook"]


class MegatronL2NormHook(L2NormHook):
    """L2NormHook with tensor parallelism support for Megatron models.

    Extends L2NormHook to gather activations across all tensor parallel regions
    before computing importance scores.
    """

    def _get_input_tensor(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Gather input tensor from all TP regions."""
        return gather_from_tensor_model_parallel_region(args[0]).detach()
