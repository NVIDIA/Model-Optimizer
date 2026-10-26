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

"""Support quantization for megatron linear layers."""

import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import get_data_parallel_group, get_model_parallel_group
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear

from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModuleRegistry
from .custom import _ParallelLinear

__all__ = []


class _FairscaleParallelLinear(_ParallelLinear):
    _functionals_to_replace = [(F, "linear")]

    def _setup(self):
        self.parallel_state = ParallelState(get_data_parallel_group(), get_model_parallel_group())
        super()._setup()


@QuantModuleRegistry.register({ColumnParallelLinear: "fairscale_ColumnParallelLinear"})
class _FairscaleColumnParallelLinear(_FairscaleParallelLinear):
    _is_column_parallel = True


@QuantModuleRegistry.register({RowParallelLinear: "fairscale_RowParallelLinear"})
class _FairscaleRowParallelLinear(_FairscaleParallelLinear):
    _is_row_parallel = True
