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


import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from modelopt.torch.opt.dynamic import DynamicModule


def param_num(network: nn.Module, trainable_only: bool = False, unit=1e6) -> float:
    """Get the number of parameters of a PyTorch model.

    Args:
        network: The PyTorch model.
        trainable_only: Whether to only count trainable parameters. Default is False.
        unit: The unit to return the number of parameters in. Default is 1e6 (million).

    Returns:
        The number of parameters in the model in the given unit.
    """

    if isinstance(network, DynamicModule):
        # NOTE: model.parameters() doesnt consider active_slice so we dont get sorted or trimmed parameters!
        raise NotImplementedError(
            "param_num doesn't support DynamicModule. Please use param_num_from_forward instead."
        )
    return (
        sum(
            p.numel() if not trainable_only or p.requires_grad else 0
            for mod in network.modules()
            for p in mod.parameters(recurse=False)
            if not isinstance(mod, _BatchNorm)
        )
        / unit
    )
