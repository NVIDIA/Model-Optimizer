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

from typing import get_args

import pytest
import torch
from _test_utils.torch.linear_attention import check_prefill_case, inputs, values_and_gradients

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    matmul_gdn,
    recurrent_delta_rule_reference,
)
from modelopt.torch.quantization.linear_attention.config import _ElementwiseSite, _PrefillSite
from modelopt.torch.quantization.linear_attention.matmul import LinearAttentionMatmulSites
from modelopt.torch.quantization.nn import TensorQuantizer

SITES = get_args(_PrefillSite)
HANDLES = [
    name
    for name, module in LinearAttentionMatmulSites().named_modules()
    if name.endswith("quantizer")
]


@pytest.mark.parametrize("handle", HANDLES)
def test_prefill_operand_outputs_state_and_all_input_gradients(handle):
    check_prefill_case(LinearAttentionConfig(backend="matmul"), [handle])


@pytest.mark.parametrize("site", SITES)
def test_blocked_accumulator_surrogate(site):
    policy = LinearAttentionConfig(
        backend="matmul", matmul={site: {"accumulator_dtype": "float16", "reduction_block": 4}}
    )
    check_prefill_case(policy)


@pytest.mark.parametrize("site", get_args(_ElementwiseSite))
def test_elementwise_rounding_surrogate(site):
    check_prefill_case(LinearAttentionConfig(backend="matmul", elementwise={site: "float16"}))


@pytest.mark.parametrize(("packed", "state_v_first"), [(False, False), (True, True)])
def test_composed_prefill_with_state_w_and_layouts(packed, state_v_first):
    check_prefill_case(
        LinearAttentionConfig(
            backend="matmul", state={"block_v": 16}, elementwise={"value_residual": "bfloat16"}
        ),
        HANDLES,
        packed=packed,
        state_v_first=state_v_first,
        state_qdq=True,
        w_qdq=True,
    )


@pytest.mark.parametrize(
    "config",
    [
        {"matmul": {"output_score": {"accumulator_dtype": "float16", "reduction_block": 16}}},
        {"backend": "matmul", "matmul": {"unknown": {}}},
        {"backend": "matmul", "matmul": {"output_score": {"accumulator_dtype": "float16"}}},
        {"backend": "matmul", "matmul": {"output_score": {"reduction_block": 8}}},
        {"backend": "matmul", "elementwise": {"output_add": "fp8"}},
    ],
)
def test_reject_unimplemented_or_incomplete_arithmetic(config):
    with pytest.raises(ValueError):
        LinearAttentionConfig(**config)


def test_host_packing_metadata_with_unequal_key_value_dimensions():
    args, state = inputs(packed=True, dtype=torch.float64)
    args[2] = args[2][..., :11].detach().requires_grad_()
    state = state[..., :11].detach().requires_grad_()
    boundaries = torch.tensor([0, 5, 73])
    actual = matmul_gdn(
        *args,
        initial_state=state,
        cu_seqlens_cpu=boundaries,
        sites=LinearAttentionMatmulSites(),
        policy=LinearAttentionConfig(backend="matmul"),
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        output_final_state=True,
    )
    expected = recurrent_delta_rule_reference(*args, initial_state=state, cu_seqlens=boundaries)
    for a, e in zip(
        values_and_gradients(actual, args, state), values_and_gradients(expected, args, state)
    ):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)


def test_outer_autocast_preserves_working_precision_and_gradients():
    args, state = inputs()
    sites = LinearAttentionMatmulSites()
    for name in HANDLES:
        sites.get_submodule(name).set_from_attribute_config(
            {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}
        )
        sites.get_submodule(name).enable()
    kwargs = {
        "sites": sites,
        "policy": LinearAttentionConfig(backend="matmul"),
        "w_quantizer": TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        "initial_state": state,
        "output_final_state": True,
    }
    expected = matmul_gdn(*args, **kwargs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = matmul_gdn(*args, **kwargs)
    for a, e in zip(
        values_and_gradients(actual, args, state), values_and_gradients(expected, args, state)
    ):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
