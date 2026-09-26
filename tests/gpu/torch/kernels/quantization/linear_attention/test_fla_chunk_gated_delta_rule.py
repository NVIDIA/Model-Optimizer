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

"""Numerical forward/backward coverage for the GDN training emulation contract."""

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.quantization.linear_attention_reference import (
    chunk_gdn_reference,
    state_fp8_qdq_reference,
)
from torch.utils.checkpoint import checkpoint

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

fla = pytest.importorskip("fla.ops.gated_delta_rule")
from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0

from modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule import (
    chunk_gated_delta_rule,
)


def require_state_qdq():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("State QDQ needs native E4M3 conversion (SM89+)")


def make_inputs(dtype=None, packed=False, state_v_first=False, grouped=True):
    hopper_backward = IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0
    if dtype is None:
        dtype = torch.bfloat16 if hopper_backward else torch.float32
    if dtype == torch.float32 and hopper_backward:
        pytest.skip("Hopper/TileLang training supports BF16; FP32 is rejected before launch")
    torch.manual_seed(123)
    # Two full chunks plus a tail; grouped-value heads and a partial state scale tile.
    batch, length, heads, value_heads, keys, values = 1, 145, 1, 2, 32, 48
    heads = heads if grouped else value_heads
    q, k = [
        F.normalize(torch.randn(batch, length, heads, keys, device="cuda"), dim=-1)
        for _ in range(2)
    ]
    v = torch.randn(batch, length, value_heads, values, device="cuda")
    g = -torch.rand(batch, length, value_heads, device="cuda") * 0.1
    beta = torch.rand_like(g)
    state = torch.randn(2 if packed else batch, value_heads, keys, values, device="cuda") * 0.1
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
    args = [x.to(dtype).requires_grad_() for x in (q, k, v, g, beta)]
    return args, state.requires_grad_()


def w_quantizer(axis=(0, 1, 2)):
    return TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=axis, type="dynamic"))


def compare(actual, expected, tolerance):
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all()
        error = (a.float() - e.float()).norm()
        bound = tolerance * e.float().norm().clamp_min(1e-6)
        assert error <= bound, (
            f"relative L2 error {(error / e.float().norm()).item():.5g} > {tolerance}"
        )


def compare_forward(actual, expected, dtype, state_qdq):
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.01
    compare(actual[:1], expected[:1], tolerance)
    # FP8 bin crossings amplify small GEMM differences in the stored final state.
    state_tolerance = max(tolerance, 0.02) if state_qdq else tolerance
    compare(actual[1:], expected[1:], state_tolerance)


def values_and_grads(fn, args, state, **kwargs):
    result = fn(*args, initial_state=state, **kwargs)
    torch.manual_seed(15)
    probes = [torch.randn(x.shape, device=x.device, dtype=torch.float32) for x in result]
    grads = torch.autograd.grad(sum((x * p).sum() for x, p in zip(result, probes)), (*args, state))
    return result, grads


# The first comparison compiles and autotunes both upstream and ModelOpt forward/backward kernels.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disabled_matches_upstream_forward_and_backward(dtype):
    # Upstream Hopper/TileLang backward cannot handle grouped value heads.
    args, state = make_inputs(dtype, grouped=False)
    expected = values_and_grads(fla.chunk_gated_delta_rule, args, state, output_final_state=True)
    actual = values_and_grads(chunk_gated_delta_rule, args, state, output_final_state=True)
    for a, e in zip(actual, expected):
        compare(a, e, 0.005 if dtype == torch.bfloat16 else 0.001)


@pytest.mark.parametrize(
    ("state_qdq", "quantize_w"), [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize(("packed", "state_v_first"), [(False, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_numerical_surrogate_reference(state_qdq, quantize_w, packed, state_v_first, dtype):
    if state_qdq:
        require_state_qdq()
    args, state = make_inputs(dtype, packed=packed, state_v_first=state_v_first)
    quantizer = w_quantizer() if quantize_w else None
    kwargs = {
        "state_qdq": state_qdq,
        "state_qdq_block_v": 32,
        "w_quantizer": quantizer,
        "state_v_first": state_v_first,
        "cu_seqlens": torch.tensor([0, 67, 145], device="cuda", dtype=torch.int32)
        if packed
        else None,
    }
    reference_args = [x.detach().float().requires_grad_() for x in args]
    reference_state = state.detach().float().requires_grad_()
    expected = values_and_grads(chunk_gdn_reference, reference_args, reference_state, **kwargs)
    actual = values_and_grads(
        chunk_gated_delta_rule, args, state, output_final_state=True, **kwargs
    )
    compare_forward(actual[0], expected[0], dtype, state_qdq)
    compare(actual[1], expected[1], 0.05 if dtype == torch.bfloat16 else 0.02)


def test_w_qdq_saved_once_and_activation_checkpoint_parity():
    args, state = make_inputs()
    quantizer = w_quantizer()
    calls = []
    handle = quantizer.register_forward_hook(lambda *args: calls.append(1))

    def fn(*x):
        return chunk_gated_delta_rule(
            *x[:5], initial_state=x[5], output_final_state=True, w_quantizer=quantizer
        )

    output = fn(*args, state)
    grads = torch.autograd.grad(sum(x.square().sum() for x in output), (*args, state))
    assert len(calls) == 1, "Backward must not rerun the quantizer or its observers"
    recomputed = checkpoint(fn, *args, state, use_reentrant=False)
    checkpoint_grads = torch.autograd.grad(
        sum(x.square().sum() for x in recomputed), (*args, state)
    )
    compare(recomputed, output, 1e-6)
    compare(checkpoint_grads, grads, 1e-6)
    handle.remove()


@pytest.mark.parametrize("axis", [None, (0, 1), (0, 1, 2)])
def test_zero_initial_state_and_output_only_training_loss(axis):
    args, _ = make_inputs()
    quantizer = w_quantizer(axis)
    expected, _ = chunk_gdn_reference(*(x.float() for x in args), w_quantizer=quantizer)
    actual, final = chunk_gated_delta_rule(*args, w_quantizer=quantizer)
    assert final is None
    compare((actual,), (expected,), 0.03 if args[0].dtype == torch.bfloat16 else 0.01)
    grads = [torch.autograd.grad(x.square().sum(), args) for x in (actual, expected)]
    compare(*grads, 0.05 if args[0].dtype == torch.bfloat16 else 0.02)


@pytest.mark.parametrize("block_v", [16, 64, 128])
def test_state_scale_tile_forward_and_backward(block_v):
    require_state_qdq()
    args, state = make_inputs()
    kwargs = {"state_qdq": True, "state_qdq_block_v": block_v}
    reference_args = [x.detach().float().requires_grad_() for x in args]
    expected = values_and_grads(chunk_gdn_reference, reference_args, state, **kwargs)
    actual = values_and_grads(
        chunk_gated_delta_rule, args, state, output_final_state=True, **kwargs
    )
    compare_forward(actual[0], expected[0], args[0].dtype, state_qdq=True)
    compare(actual[1], expected[1], 0.05 if args[0].dtype == torch.bfloat16 else 0.02)


@pytest.mark.parametrize("block_v", [16, 32, 64, 128])
def test_state_qdq_without_recurrent_updates(block_v):
    require_state_qdq()
    args, state = make_inputs(packed=True, state_v_first=True)
    q, k, v, g, beta = args
    _, final = chunk_gated_delta_rule(
        q,
        torch.zeros_like(k),
        torch.zeros_like(v),
        torch.zeros_like(g),
        beta,
        initial_state=state,
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=torch.tensor([0, 67, 145], device="cuda", dtype=torch.int32),
        state_qdq=True,
        state_qdq_block_v=block_v,
    )
    expected = state_fp8_qdq_reference(state.transpose(-1, -2), block_v).transpose(-1, -2)
    torch.testing.assert_close(final, expected, rtol=1e-6, atol=1e-7)
    (grad,) = torch.autograd.grad(final.sum(), state)
    torch.testing.assert_close(grad, torch.ones_like(state), rtol=0, atol=0)


def test_fused_gate_normalization_and_beta_gradients():
    args, state = make_inputs()
    q, k, v, raw_g, raw_beta = args
    a_log = torch.randn(2, device="cuda", requires_grad=True)
    bias = torch.randn(2, device="cuda", requires_grad=True)
    quantizer = w_quantizer()
    expected = chunk_gdn_reference(
        F.normalize(q.float(), dim=-1),
        F.normalize(k.float(), dim=-1),
        v.float(),
        -a_log.exp() * F.softplus(raw_g.float() + bias),
        2 * raw_beta.float().sigmoid(),
        initial_state=state,
        w_quantizer=quantizer,
    )
    actual = chunk_gated_delta_rule(
        *args,
        initial_state=state,
        output_final_state=True,
        w_quantizer=quantizer,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=bias,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
    )
    compare(actual, expected, 0.03 if q.dtype == torch.bfloat16 else 0.01)
    grads = [
        torch.autograd.grad(sum(x.square().sum() for x in result), (*args, state, a_log, bias))
        for result in (actual, expected)
    ]
    compare(*grads, 0.05 if q.dtype == torch.bfloat16 else 0.02)


@pytest.mark.parametrize("chunk_size", [16, 32, 128])
def test_unsupported_chunk_rejected_before_launch(chunk_size):
    args, state = make_inputs()
    with pytest.raises(ValueError, match="only chunk_size=64"):
        chunk_gated_delta_rule(*args, chunk_size=chunk_size)


def test_unsupported_w_gradient_rejected_before_launch():
    args, _ = make_inputs()
    quantizer = w_quantizer()
    quantizer.set_from_attribute_config({"pass_through_bwd": False})
    with pytest.raises(ValueError, match="pass_through_bwd=True"):
        chunk_gated_delta_rule(*args, w_quantizer=quantizer)


def test_hopper_fp32_rejected_before_launch():
    if not (IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0):
        pytest.skip("Hopper/TileLang capability guard")
    args, _ = make_inputs()
    with pytest.raises(ValueError, match="requires BF16 q/k/v"):
        chunk_gated_delta_rule(*(x.float() for x in args))
