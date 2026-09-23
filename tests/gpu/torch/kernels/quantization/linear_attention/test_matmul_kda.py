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
from _test_utils.torch.linear_attention import (
    FP8,
    fp8_oracle,
    inputs,
    nvfp4_oracle,
    reference_matmul,
    values_and_gradients,
)
from _test_utils.torch.quantization.linear_attention_reference import chunk_kda_reference

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    LinearAttentionMatmulSites,
    matmul_kda,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.fixture(autouse=True)
def full_precision_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


def _inputs(*, packed=False, dtype=torch.float32):
    args, state = inputs("cuda", packed=packed, dtype=dtype)
    args[3] = (
        -torch.rand(*args[3].shape, args[0].shape[-1], device="cuda") * 0.05
    ).requires_grad_()
    return args, state.float().detach().requires_grad_()


@pytest.mark.timeout(300)
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("packed", [False, True])
def test_fla_baseline_and_fused_gate_gradients(lower_bound, packed):
    pytest.importorskip("fla")
    # FLA is optional except for comparison with its actual CUDA implementation.
    from fla.ops.kda import chunk_kda

    args, state = _inputs(packed=packed, dtype=torch.bfloat16)
    a_log = torch.full((2,), -3.0, device="cuda", requires_grad=True)
    bias = torch.zeros(32, device="cuda", requires_grad=True)
    kwargs = {
        "initial_state": state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": True,
        "A_log": a_log,
        "dt_bias": bias,
        "lower_bound": lower_bound,
        "safe_gate": lower_bound is not None,
        "cu_seqlens": torch.tensor([0, 5, 73], device="cuda", dtype=torch.int32)
        if packed
        else None,
    }
    actual = matmul_kda(
        *args,
        sites=LinearAttentionMatmulSites(),
        policy=LinearAttentionConfig(backend="matmul"),
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        **kwargs,
    )
    # Compare the general FLA algebra, not its optional safe_gate TensorCore rounding.
    expected = chunk_kda(*args, **{**kwargs, "safe_gate": False})
    probes = [torch.randn_like(x) for x in actual]
    gradients = [
        torch.autograd.grad(
            sum((x * p).sum() for x, p in zip(result, probes)), (*args, state, a_log, bias)
        )
        for result in (actual, expected)
    ]
    for i, (a, e) in enumerate(zip((*actual, *gradients[0]), (*expected, *gradients[1]))):
        error = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-6)
        assert error <= (0.03 if i < 2 else 0.05), (i, error.item())


def test_composed_fp8_with_chunk_state_and_w():
    args, state = _inputs(packed=True)
    sites = LinearAttentionMatmulSites()
    handles = [name for name, _ in sites.named_modules() if name.endswith("quantizer")]
    for name in handles:
        sites.get_submodule(name).set_from_attribute_config(FP8)
        sites.get_submodule(name).enable()
    policy = LinearAttentionConfig(backend="matmul", state={"block_v": 16})
    boundaries = torch.tensor([0, 5, 73], device="cuda")
    actual = matmul_kda(
        *args,
        sites=sites,
        policy=policy,
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(**FP8)),
        state_qdq=True,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=boundaries,
    )
    expected = chunk_kda_reference(
        *args,
        initial_state=state,
        cu_seqlens=boundaries,
        state_qdq=True,
        state_qdq_block_v=16,
        w_quantizer=fp8_oracle,
        matmul=reference_matmul(handles, policy),
    )
    for i, (a, e) in enumerate(
        zip(values_and_gradients(actual, args, state), values_and_gradients(expected, args, state))
    ):
        error = (a - e).norm() / e.norm().clamp_min(1e-6)
        assert error <= (0.01 if i < 2 else 0.02), (i, error.item())


@pytest.mark.parametrize("site", ["key_interaction", "output_score"])
@pytest.mark.parametrize("side", ["lhs_quantizer", "rhs_quantizer"])
def test_nvfp4_channel_decay_interaction(site, side):
    args, state = _inputs()
    sites = LinearAttentionMatmulSites()
    target = sites.get_submodule(f"{site}.{side}")
    target.set_from_attribute_config(
        {
            "num_bits": (2, 1),
            "type": "dynamic",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        }
    )
    target.enable()
    captured = []
    hook = target.register_forward_hook(lambda m, a, o: captured.append((a[0], o)))
    actual = matmul_kda(
        *args,
        sites=sites,
        policy=LinearAttentionConfig(backend="matmul"),
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        initial_state=state,
        output_final_state=True,
    )
    hook.remove()
    assert len(captured) == 8
    amaxes = []
    for x, y in captured:
        torch.testing.assert_close(y, nvfp4_oracle(x), rtol=2e-5, atol=2e-6)
        probe = torch.randn_like(y)
        torch.testing.assert_close(
            torch.autograd.grad((y * probe).sum(), x, retain_graph=True)[0], probe, rtol=0, atol=0
        )
        amaxes.append(x.detach().abs().amax())
    index = 0

    def mm(name, lhs, rhs):
        nonlocal index
        if name == site:
            amax = amaxes[(index % 64) // 8]
            index += 1
            if side == "lhs_quantizer":
                lhs = nvfp4_oracle(lhs, amax)
            else:
                rhs = nvfp4_oracle(rhs.transpose(-1, -2), amax).transpose(-1, -2)
        return torch.einsum("hij,hjk->hik", lhs, rhs)

    expected = chunk_kda_reference(*args, initial_state=state, matmul=mm)
    for a, e in zip(
        values_and_gradients(actual, args, state), values_and_gradients(expected, args, state)
    ):
        torch.testing.assert_close(a, e, rtol=5e-4, atol=3e-6)
