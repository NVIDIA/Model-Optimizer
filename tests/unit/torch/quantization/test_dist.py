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

from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.quantization.models import SimpleLinear
from torch.nn.parallel import DistributedDataParallel

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import unwrap_model


def _test_data_parallel_helper(rank, size):
    model = DistributedDataParallel(SimpleLinear())
    unwrapped_model = unwrap_model(model)
    calib_data = [unwrapped_model.get_input() for _ in range(2)]

    def forward_loop(model):
        for data in calib_data:
            model(data)

    unwrapped_model = mtq.quantize(unwrapped_model, mtq.INT8_DEFAULT_CFG, forward_loop)

    for name, module in unwrapped_model.named_modules():
        if isinstance(module, TensorQuantizer) and "output_quantizer" not in name:
            amax = module.amax.clone()
            dist.all_reduce(amax, op=dist.ReduceOp.MAX)
            assert torch.allclose(amax, module.amax)
    dist.destroy_process_group()


def test_data_parallel(skip_on_windows):
    spawn_multiprocess_job(2, _test_data_parallel_helper, backend="gloo")


def test_sync_amax_skips_cpu_tensor_with_nccl():
    quantizer = TensorQuantizer()
    quantizer._amax = torch.tensor(1.0)
    parallel_group = Mock()
    parallel_group.is_initialized.return_value = True

    with (
        patch.object(dist, "get_backend", return_value=dist.Backend.NCCL),
        patch.object(dist, "all_reduce") as all_reduce,
        pytest.warns(UserWarning, match="tensor is on CPU, but NCCL only supports CUDA tensors"),
    ):
        quantizer.sync_amax_across_distributed_group(parallel_group)

    all_reduce.assert_not_called()


def test_sync_amax_propagates_collective_errors():
    quantizer = TensorQuantizer()
    quantizer._amax = torch.tensor(1.0)
    parallel_group = Mock()
    parallel_group.is_initialized.return_value = True

    with (
        patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
        patch.object(dist, "all_reduce", side_effect=RuntimeError("out of memory")),
        pytest.raises(RuntimeError, match="out of memory"),
    ):
        quantizer.sync_amax_across_distributed_group(parallel_group)
