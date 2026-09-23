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
    check_prefill_case,
    inputs,
    nvfp4_oracle,
    values_and_gradients,
)
from torch.utils.checkpoint import checkpoint

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    chunk_gdn_reference,
    matmul_gdn,
)
from modelopt.torch.quantization.linear_attention.matmul import LinearAttentionMatmulSites
from modelopt.torch.quantization.nn import TensorQuantizer

HANDLES = [
    name
    for name, module in LinearAttentionMatmulSites().named_modules()
    if name.endswith("quantizer")
]
NVFP4 = {
    "num_bits": (2, 1),
    "type": "dynamic",
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}


@pytest.fixture(autouse=True)
def full_precision_matmul():
    """Keep operand-format comparisons independent of the container TF32 default."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("handle", HANDLES)
def test_cuda_fp8_each_operand(handle):
    check_prefill_case(LinearAttentionConfig(backend="matmul"), [handle], device="cuda")


def test_cuda_composed_arithmetic_state_and_w():
    check_prefill_case(
        LinearAttentionConfig(
            backend="matmul",
            state={"block_v": 16},
            matmul={"output_value": {"accumulator_dtype": "float16", "reduction_block": 16}},
            elementwise={"value_residual": "bfloat16"},
        ),
        # The established FP32 state-QDQ bounds allow GEMM roundoff crossing a QDQ tie.
        HANDLES,
        device="cuda",
        packed=True,
        state_v_first=True,
        state_qdq=True,
        w_qdq=True,
        relative_tolerances=(0.01, 0.02),
    )


@pytest.mark.parametrize("handle", [*HANDLES, "state_read.lhs_quantizer"])
def test_cuda_nvfp4_operand_and_surrogate(handle):
    """Independent codec, GEMM, and operand VJPs at each actual site shape."""
    args, state = inputs("cuda")
    sites = LinearAttentionMatmulSites()
    w = TensorQuantizer(QuantizerAttributeConfig(enable=False))
    target = w if handle == "state_read.lhs_quantizer" else sites.get_submodule(handle)
    target.set_from_attribute_config(NVFP4)
    target.enable()
    captures = []
    observer = target.register_forward_hook(
        lambda module, args, result: captures.append((args[0], result))
    )
    result = matmul_gdn(
        *args,
        sites=sites,
        policy=LinearAttentionConfig(backend="matmul"),
        w_quantizer=w,
        initial_state=state,
        output_final_state=True,
    )
    observer.remove()
    assert captures
    for value, actual in captures:
        expected = nvfp4_oracle(value)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        probe = torch.randn_like(actual)
        actual_grad = torch.autograd.grad((actual * probe).sum(), value, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, probe, rtol=0, atol=0)
    # Match the declared global scale domain: local stages batch all chunks, recurrent
    # stages invoke the quantizer once per chunk over the sequence batch.
    tensor_amaxes = [value.detach().abs().amax() for value, _ in captures]
    target_site, side = handle.split(".")
    call_index = 0

    def reference_mm(name, lhs, rhs):
        nonlocal call_index
        if name == target_site:
            index = call_index if len(tensor_amaxes) > 1 else 0
            call_index += 1
            if side == "lhs_quantizer":
                lhs = nvfp4_oracle(lhs, tensor_amaxes[index])
            else:
                rhs = nvfp4_oracle(rhs.transpose(-1, -2), tensor_amaxes[index]).transpose(-1, -2)
        return torch.einsum("hij,hjk->hik", lhs, rhs)

    expected = chunk_gdn_reference(*args, initial_state=state, matmul=reference_mm)
    for a, e in zip(
        values_and_gradients(result, args, state), values_and_gradients(expected, args, state)
    ):
        torch.testing.assert_close(a, e, rtol=5e-4, atol=3e-6)


def test_cuda_checkpoint_preserves_prefill_gradients():
    args, state = inputs("cuda")
    sites = LinearAttentionMatmulSites()
    for name in HANDLES:
        sites.get_submodule(name).set_from_attribute_config(FP8)
        sites.get_submodule(name).enable()
    w = TensorQuantizer(QuantizerAttributeConfig(**FP8))
    policy = LinearAttentionConfig(backend="matmul")

    def run(*x):
        return matmul_gdn(
            *x[:-1],
            sites=sites,
            policy=policy,
            w_quantizer=w,
            initial_state=x[-1],
            output_final_state=True,
        )

    plain = values_and_gradients(run(*args, state), args, state)
    recomputed = values_and_gradients(
        checkpoint(run, *args, state, use_reentrant=False), args, state
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mixed_result = run(*args, state)
    mixed = values_and_gradients(mixed_result, args, state)
    for result in (recomputed, mixed):
        for a, b in zip(plain, result):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cuda_matmul_baseline_and_fused_gate_gradients():
    # FLA is optional outside this GPU qualification test.
    pytest.importorskip("fla")
    from modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule import (
        chunk_gated_delta_rule,
    )

    args, state = inputs("cuda", dtype=torch.bfloat16)
    state = state.float().detach().requires_grad_()
    a_log = torch.full((2,), -3.0, device="cuda", requires_grad=True)
    bias = torch.zeros(2, device="cuda", requires_grad=True)
    kwargs = {
        "initial_state": state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "A_log": a_log,
        "dt_bias": bias,
    }
    actual = matmul_gdn(
        *args,
        sites=LinearAttentionMatmulSites(),
        policy=LinearAttentionConfig(backend="matmul"),
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        **kwargs,
    )
    expected = chunk_gated_delta_rule(*args, **kwargs)
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
