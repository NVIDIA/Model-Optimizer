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

"""General utilities for Megatron models."""

from typing import Any

import torch
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.transformer.module import MegatronModule

from ..network import param_num_from_forward

__all__ = ["param_num_megatron"]


def param_num_megatron(
    model: MegatronModule, *, from_forward: bool = False, args: Any = None
) -> float:
    """Get the number of parameters in the model (reduced across TP and PP ranks).

    Args:
        model: The Megatron model.
        from_forward: To get the number of params from a forward pass instead of directly counting the params.
            This can helpful for MoE or dynamic modules, where the state dict might contain extra parameters that
            is not actively used in the model, e.g., because of a DynamicModule that is deactivated for the
            forward pass. We circumvent this issue by just counting parameters of modules that appear in a
            forward pass.
        args: The arguments to pass to the forward pass. Only used if from_forward is True.

    Returns:
        The number of parameters in the model (reduced across TP and PP ranks).
    """
    if from_forward:
        assert args is not None, "args must be provided if from_forward is True"
        params = int(param_num_from_forward(model, args, unit=1.0))
    else:
        params = sum(p.numel() for p in model.parameters())
    reduced_params = torch.Tensor([params]).to(device=next(model.parameters()).device)
    torch.distributed.all_reduce(reduced_params, group=get_pipeline_model_parallel_group())
    torch.distributed.all_reduce(reduced_params, group=get_tensor_model_parallel_group())
    return reduced_params.item()
