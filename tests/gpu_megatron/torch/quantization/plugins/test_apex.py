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

from functools import partial

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.models import RegularQuantModelForTP
from _test_utils.torch.quantization.quantize_common import (
    auto_quantize_helper,
    data_tensor_context_parallel_test_helper,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry

try:
    import apex.transformer.tensor_parallel.layers as apex_parallel
    from apex.transformer.parallel_state import (
        destroy_model_parallel,
        get_data_parallel_group,
        get_tensor_model_parallel_group,
        initialize_model_parallel,
    )
    from apex.transformer.tensor_parallel.random import model_parallel_cuda_manual_seed
except ImportError:
    pytest.skip("apex not available", allow_module_level=True)

SEED = 1234


class ApexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = apex_parallel.ColumnParallelLinear(32, 64, gather_output=False)
        self.activation = nn.ReLU()
        self.fc2 = apex_parallel.RowParallelLinear(64, 32, input_is_parallel=True)

    def forward(self, x):
        for block in [self.fc1, self.activation, self.fc2]:
            x = block(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def get_dummy_input(self, seed: int | None = None):
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            return torch.randn(1, 4, 32, generator=gen)
        return torch.randn(1, 4, 32)


def test_convert_apex_parallel_linear(distributed_setup_size_1):
    initialize_model_parallel()
    model_parallel_cuda_manual_seed(SEED)
    set_seed(SEED)

    assert apex_parallel.ColumnParallelLinear in QuantModuleRegistry

    model_ref = ApexModel().cuda()
    model_test = ApexModel().cuda()
    model_test.load_state_dict(model_ref.state_dict())
    mtq.replace_quant_module(model_test)
    for name, module in model_test.named_modules():
        if isinstance(
            module, (apex_parallel.ColumnParallelLinear, apex_parallel.RowParallelLinear)
        ):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = model_ref.get_dummy_input().cuda()
    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)

    mtq.set_quantizer_attribute(model_test, "*input_quantizer", {"enable": True})
    mtq.set_quantizer_attribute(model_test, "*weight_quantizer", {"enable": True})
    model_ref = RegularQuantModelForTP().cuda()
    model_ref.load_state_dict(model_test.state_dict())

    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2, atol=1e-5)  # atol higher to fix non-deterministic failures

    # Clean up since this is not a spawned process
    destroy_model_parallel()


def _test_tensor_parallel_helper(config, rank, size):
    initialize_model_parallel(tensor_model_parallel_size_=size)
    model_parallel_cuda_manual_seed(SEED)
    model = ApexModel().cuda()

    data_tensor_context_parallel_test_helper(
        model,
        config,
        tp_group=get_tensor_model_parallel_group(),
        dp_group=get_data_parallel_group(),
    )


@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
    ],
)
def test_tensor_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2, job=partial(_test_tensor_parallel_helper, config), backend="nccl"
    )


def _test_auto_quantize_helper(rank, size):
    initialize_model_parallel(tensor_model_parallel_size_=size)
    model_parallel_cuda_manual_seed(SEED)
    model = ApexModel().cuda()
    auto_quantize_helper(model)


def test_auto_quantize(need_2_gpus):
    spawn_multiprocess_job(size=2, job=_test_auto_quantize_helper, backend="nccl")
