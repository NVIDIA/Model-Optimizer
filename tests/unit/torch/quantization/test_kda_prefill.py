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

import pytest
import torch
from _test_utils.torch.linear_attention import FP8, inputs, reference_matmul, values_and_gradients
from _test_utils.torch.quantization.linear_attention_reference import (
    chunk_kda_reference,
    recurrent_delta_rule_reference,
)

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    LinearAttentionMatmulSites,
    matmul_kda,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize(("packed", "state_v_first"), [(False, False), (True, True)])
@pytest.mark.parametrize("strong_decay", [False, True])
def test_kda_exact_outputs_state_and_gradients(packed, state_v_first, strong_decay):
    args, state = inputs(packed=packed, state_v_first=state_v_first, dtype=torch.float64)
    args[2] = args[2][..., :11].detach().requires_grad_()
    state = (state[..., :11, :] if state_v_first else state[..., :11]).detach().requires_grad_()
    gates = -torch.rand(*args[3].shape, args[0].shape[-1], dtype=torch.float64)
    if strong_decay:
        gates = gates * torch.logspace(-8, 3, gates.shape[-1], dtype=torch.float64)
    else:
        gates = gates * 0.1
    args[3] = gates.requires_grad_()
    kwargs = {
        "initial_state": state,
        "state_v_first": state_v_first,
        "cu_seqlens": torch.tensor([0, 5, 73]) if packed else None,
    }
    actual = matmul_kda(
        *args,
        sites=LinearAttentionMatmulSites(),
        policy=LinearAttentionConfig(backend="matmul"),
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        output_final_state=True,
        **kwargs,
    )
    recurrent = recurrent_delta_rule_reference(*args, **kwargs)
    chunk = chunk_kda_reference(*args, **kwargs)
    for result in (actual, chunk):
        for a, e in zip(
            values_and_gradients(result, args, state), values_and_gradients(recurrent, args, state)
        ):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, e, rtol=2e-9, atol=1e-10)


HANDLES = [
    name for name, _ in LinearAttentionMatmulSites().named_modules() if name.endswith("quantizer")
]


@pytest.mark.parametrize("handle", HANDLES)
def test_kda_fp8_site_matches_chunk_oracle(handle):
    args, state = inputs()
    args[3] = (-torch.rand(*args[3].shape, args[0].shape[-1]) * 0.1).requires_grad_()
    sites = LinearAttentionMatmulSites()
    sites.get_submodule(handle).set_from_attribute_config(FP8)
    sites.get_submodule(handle).enable()
    policy = LinearAttentionConfig(backend="matmul")
    actual = matmul_kda(
        *args,
        sites=sites,
        policy=policy,
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        initial_state=state,
        output_final_state=True,
    )
    expected = chunk_kda_reference(
        *args, initial_state=state, matmul=reference_matmul([handle], policy)
    )
    for a, e in zip(
        values_and_gradients(actual, args, state), values_and_gradients(expected, args, state)
    ):
        torch.testing.assert_close(a, e, rtol=5e-4, atol=3e-6)
