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

"""Support quantization for apex linear layers."""

from functools import partial

import apex.transformer.tensor_parallel.layers as apex_parallel
from apex.transformer.parallel_state import get_data_parallel_group, get_tensor_model_parallel_group

from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModuleRegistry
from ..nn.modules.quant_linear import _QuantLinear
from .custom import _ParallelLinear


class _ApexParallelLinear(_ParallelLinear):
    def _setup(self):
        quantized_linear_fn = partial(
            _QuantLinear.quantized_linear_fn,
            apex_parallel,
            "linear_with_grad_accumulation_and_async_allreduce",
            self,
        )
        self._forward_impl = quantized_linear_fn
        self.parallel_state = ParallelState(
            get_data_parallel_group(), get_tensor_model_parallel_group()
        )
        super()._setup()


@QuantModuleRegistry.register({apex_parallel.ColumnParallelLinear: "apex_ColumnParallelLinear"})
class _ApexColumnParallelLinear(_ApexParallelLinear):
    _is_column_parallel = True


@QuantModuleRegistry.register({apex_parallel.RowParallelLinear: "apex_RowParallelLinear"})
class _ApexRowParallelLinear(_ApexParallelLinear):
    _is_row_parallel = True
