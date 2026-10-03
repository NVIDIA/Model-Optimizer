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


import pytest
from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends import gemm_registry
from modelopt.torch.quantization.backends.fp8_per_tensor_gemm import Fp8PerTensorLinear
from modelopt.torch.quantization.backends.utils import fp8_compatible

set_seed()


@pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU")
@pytest.mark.parametrize("model_cls", [SimpleLinear])
@pytest.mark.parametrize("config", [mtq.FP8_DEFAULT_CFG])
def test_fp8_per_tensor_gemm_available(model_cls, config):
    """Test for fp8_per_tensor_gemm function with hardware-friendly dimensions."""
    model = model_cls().cuda()
    calib_data = [model.get_input().cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)
    mtq.compress(model)

    # Take the first module in the net
    module = model.net[0]
    input_tensor = calib_data[0].clone()

    # Find the matching GEMM implementation
    gemm_forward = gemm_registry.find_match(module, input_tensor, [], {})
    assert gemm_forward == Fp8PerTensorLinear.apply
