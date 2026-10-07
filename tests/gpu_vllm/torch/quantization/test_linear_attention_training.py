# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    matmul_gdn,
    matmul_kda,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.fixture(scope="module", params=[False, True])
def compiled_serving_case(request):
    """Compile one shared BF16 shape per model, outside the test-call timer."""
    pytest.importorskip("vllm.model_executor.layers.fla.ops.kda", exc_type=ModuleNotFoundError)
    kda = request.param
    torch.manual_seed(73)
    args = [torch.randn(1, 73, 1, 32, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    args += [
        -torch.rand((1, 73, 1, 32) if kda else (1, 73, 1), device="cuda") * 0.03,
        torch.rand(1, 73, 1, device="cuda") * 0.4,
    ]
    args = [x.requires_grad_() for x in args]
    policy = LinearAttentionConfig(backend="serving", precision="vllm_0_15")
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=8,
            type="dynamic",
            block_sizes={-1: 32},
            narrow_range=True,
            pass_through_bwd=True,
        )
    ).cuda()
    forward = partial(
        matmul_kda if kda else matmul_gdn,
        *args,
        policy=policy,
        state_quantizer=quantizer,
        prefill_lengths=[65],
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )
    output, state = forward()
    torch.autograd.grad(output.float().sum() + state.sum(), args)
    torch.cuda.synchronize()
    return args, quantizer, forward


def test_serving_state_qdq_and_handoff_gradient(compiled_serving_case):
    args, quantizer, forward = compiled_serving_case
    calls = []
    handle = quantizer.register_forward_hook(lambda *_: calls.append(True))
    output, state = forward()
    handle.remove()
    # One handoff QDQ, then one QDQ per suffix update; none inside the fresh prefill.
    assert len(calls) == 9
    with torch.no_grad():
        expected, expected_state = forward()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        quantizer.disable()
        plain, _ = forward()
        assert not torch.equal(output[:, 65:], plain[:, 65:])
    gradients = torch.autograd.grad(
        output[:, 65:].float().square().sum() + state.square().sum(), args
    )
    assert all(torch.isfinite(x).all() for x in gradients)
    assert gradients[1][:, :65].abs().sum() > 0
