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

from inspect import signature

import pytest
import torch
from _test_utils.torch.quantization.linear_attention_reference import recurrent_delta_rule_reference

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    gdn_state_qat,
    kda_state_qat,
)
from modelopt.torch.quantization.nn import TensorQuantizer

vllm = pytest.importorskip("vllm")

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

from modelopt.torch.kernels.quantization.linear_attention.serving._compat import (
    _kda_state_v_first,
    fla_module,
    gdn_module,
    kda_module,
    optional_module,
)


@torch.no_grad()
def _native_forward(args, quantizer, kda, kimi_k3):
    """Run the CI vLLM 0.20/0.30 Triton APIs with a persistent quantized cache."""
    q, k, v, raw_g, raw_beta, rate, bias = [x.detach().clone() for x in args]
    _, shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, 1, 1, q.shape[-1], v.shape[-1], 4
    )
    # CI KDA kernels specify [H,V,K]; rectangular GDN advertises its own layout.
    value_first = kda or shape == (1, v.shape[-1], q.shape[-1])

    def layout(state):
        return (state.transpose(-1, -2) if value_first else state).contiguous()

    cache = torch.zeros(2, *shape, device=q.device)
    prefix_args = [x[:, :65].contiguous() for x in (q, k, v)]
    kwargs = {
        "initial_state": cache[1:],
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": torch.tensor([0, 65], device=q.device, dtype=torch.int32),
    }
    if kda:
        native = kda_module(kimi_k3)
        g = fla_module("kda").fused_kda_gate(raw_g.flatten(-2), rate, q.shape[-1], g_bias=bias)
        beta = raw_beta.float().sigmoid()
        raw_chunk = getattr(native, "chunk_kda_with_fused_gate", None)
        if raw_chunk is not None:
            beta_name = "raw_beta" if "raw_beta" in signature(raw_chunk).parameters else "beta"
            prefix, state = raw_chunk(
                *prefix_args,
                raw_g=raw_g[:, :65].contiguous(),
                **{beta_name: (raw_beta if beta_name == "raw_beta" else beta)[:, :65].contiguous()},
                A_log=rate,
                g_bias=bias,
                **kwargs,
            )
        else:  # vLLM 0.20 standalone KDA takes activated gates.
            prefix, state = native.chunk_kda(*prefix_args, g[:, :65], beta[:, :65], **kwargs)
    else:
        prepared = gdn_module("fused_gdn_prefill_post_conv").fused_post_conv_prep(
            torch.cat([x[0].flatten(1) for x in prefix_args], dim=-1),
            raw_g[0, :65],
            raw_beta[0, :65],
            rate,
            bias,
            1,
            q.shape[-1],
            v.shape[-1],
        )
        kwargs["use_qk_l2norm_in_kernel"] = False
        prefix, state = fla_module("chunk").chunk_gated_delta_rule(
            *[x[None] for x in prepared], **kwargs
        )
    cache[1] = layout(quantizer(layout(state)[0]))
    outputs = [prefix]
    indices = torch.ones(1, device=q.device, dtype=torch.int32)
    cu = torch.tensor([0, 1], device=q.device, dtype=torch.int32)
    for token in range(65, 73):
        operands = [x[:, token : token + 1].contiguous() for x in (q, k, v)]
        if kimi_k3:
            output, _ = native.fused_recurrent_kda_packed_decode(
                torch.cat([x.flatten() for x in operands])[None],
                raw_g[:, token : token + 1].contiguous(),
                raw_beta[:, token : token + 1].contiguous(),
                rate,
                bias,
                None,
                cache,
                indices,
            )
            state = cache[1:]
        elif kda:
            output, state = native.fused_recurrent_kda(
                *operands,
                g[:, token : token + 1],
                beta[:, token : token + 1],
                initial_state=cache[1:],
                inplace_final_state=False,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            output, _ = fla_module(
                "fused_recurrent"
            ).fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=torch.cat([x.flatten() for x in operands])[None],
                a=raw_g[0, token : token + 1],
                b=raw_beta[0, token : token + 1],
                A_log=rate,
                dt_bias=bias,
                scale=q.shape[-1] ** -0.5,
                initial_state=cache,
                out=torch.empty_like(operands[-1]),
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
            )
            state = cache[1:]
        outputs.append(output)
        cache[1] = layout(quantizer(layout(state)[0]))
    return torch.cat(outputs, dim=1), layout(cache[1:])


@pytest.fixture(scope="module", params=["gdn", "kda", "kimi_k3"])
def compiled_serving_case(request):
    """Compile one shared BF16 shape per model, outside the test-call timer."""
    kda = request.param != "gdn"
    kimi_k3 = request.param == "kimi_k3"
    if kimi_k3 and optional_module("vllm.models.kimi_k3.nvidia.ops.third_party.kda") is None:
        pytest.skip("This vLLM build does not provide the Kimi-K3 backend")
    torch.manual_seed(73)
    keys = 128 if kda else 32
    args = [
        torch.randn(1, 73, 1, dim, device="cuda", dtype=torch.bfloat16)
        # Rectangular GDN state catches transposes; equal padded tile widths also
        # satisfy the fused post-conv kernel's tile-width constraint.
        for dim in (keys, keys, keys if kda else 24)
    ]
    args += [
        torch.randn((1, 73, 1, keys) if kda else (1, 73, 1), device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 73, 1, device="cuda", dtype=torch.bfloat16),
        torch.full((1,), -3.0, device="cuda"),
        torch.zeros(keys if kda else 1, device="cuda"),
    ]
    args = [x.requires_grad_() for x in args]
    policy = LinearAttentionConfig(
        backend="serving", precision="vllm_kimi_k3" if kimi_k3 else "vllm"
    )
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=8,
            type="dynamic",
            block_sizes={-1: 32},
            narrow_range=True,
            pass_through_bwd=True,
        )
    ).cuda()

    def forward(inputs=args, **overrides):
        q, k, v, raw_g, raw_beta, rate, bias = inputs
        kwargs = {
            "policy": policy,
            "state_quantizer": quantizer,
            "prefill_lengths": [65],
            "use_qk_l2norm_in_kernel": True,
            "output_final_state": True,
            "gate_inputs": (raw_g, raw_beta, rate, bias),
        }
        if kda:
            kwargs.update(
                A_log=rate,
                dt_bias=bias,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
            )
        kwargs.update(overrides)
        if kda:
            return kda_state_qat(q, k, v, raw_g, raw_beta, **kwargs)
        gate = -rate.exp() * torch.nn.functional.softplus(raw_g.float() + bias)
        return gdn_state_qat(q, k, v, gate, raw_beta.float().sigmoid(), **kwargs)

    try:
        output, state = forward()
    except NotImplementedError as error:
        if vllm.__version__ == "0.20.0" and "KDA chunk/decode cache readout disagrees" in str(
            error
        ):
            pytest.xfail(str(error))
        raise
    torch.autograd.grad(output.float().sum() + state.sum(), args)
    expected = _native_forward(args, quantizer, kda, kimi_k3)
    torch.cuda.synchronize()
    return args, quantizer, forward, expected, kimi_k3


def test_serving_state_qdq_and_handoff_gradient(compiled_serving_case):
    args, quantizer, forward, (expected, expected_state), _ = compiled_serving_case
    calls = []
    handle = quantizer.register_forward_hook(lambda *_: calls.append(True))
    try:
        output, state = forward()
    finally:
        handle.remove()
    assert len(calls) == 9  # Handoff plus eight suffix writes.
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    gradients = torch.autograd.grad(
        output[:, 65:].float().square().sum() + state.square().sum(), args
    )
    assert all(torch.isfinite(x).all() for x in gradients)
    assert gradients[1][:, :65].abs().sum() > 0
    with pytest.raises(ValueError, match="explicit prefill lengths"):
        forward(prefill_lengths=None)

    with torch.no_grad():
        quantizer.disable()
        try:
            plain, _ = forward()
            assert not torch.equal(output[:, 65:], plain[:, 65:])
            # A nonzero nonsymmetric state checks layout against an independent oracle.
            initial = torch.randn(1, 1, args[0].shape[-1], args[2].shape[-1], device="cuda")
            for prefix in (73, 65):
                actual, final = forward(prefill_lengths=[prefix], initial_state=initial)
                q, k, v, raw_g, raw_beta, rate, bias = [x.float() for x in args]
                q, k = [x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt() for x in (q, k)]
                q[:, :prefix] = q[:, :prefix].bfloat16().float()
                k[:, :prefix] = k[:, :prefix].bfloat16().float()
                gate = -rate.exp().reshape(
                    (1, 1, -1, 1) if raw_g.ndim == 4 else (1, 1, -1)
                ) * torch.nn.functional.softplus(raw_g + bias.reshape(raw_g.shape[2:]))
                expected, expected_state = recurrent_delta_rule_reference(
                    q, k, v, gate, raw_beta.sigmoid(), initial_state=initial
                )
                torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
                torch.testing.assert_close(final, expected_state, rtol=0.02, atol=0.01)
        finally:
            quantizer.enable()


@pytest.mark.parametrize(
    ("overrides", "message", "kimi_only"),
    [
        (
            {"gate_inputs": None, "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False},
            "raw gate_inputs",
            True,
        ),
        ({"use_qk_l2norm_in_kernel": False}, "normalization", True),
        ({"allow_neg_eigval": True}, "allow_neg_eigval", False),
    ],
)
def test_rejects_silent_serving_mismatches(compiled_serving_case, overrides, message, kimi_only):
    args, _, forward, _, kimi_k3 = compiled_serving_case
    if args[3].ndim != 4 or (kimi_only and not kimi_k3):
        pytest.skip("Guard applies only to the selected KDA profile")
    with pytest.raises(ValueError, match=message):
        forward(**overrides)


def test_kda_non_power_of_two_layout(monkeypatch):
    # Probe values must be BF16-exact for K=192, including fused-gate chunk dispatch.
    try:
        for raw_gates in (False, True):
            assert isinstance(
                _kda_state_v_first(torch.device("cuda:0"), False, 192, raw_gates), bool
            )
    except NotImplementedError as error:
        if vllm.__version__ == "0.20.0" and "KDA chunk/decode cache readout disagrees" in str(
            error
        ):
            pytest.xfail(str(error))
        raise
    native = kda_module()
    recurrent = native.fused_recurrent_kda_fwd

    def shifted_readout(*args, **kwargs):
        output, state = recurrent(*args, **kwargs)
        return output.roll(1, -1), state

    _kda_state_v_first.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(native, "fused_recurrent_kda_fwd", shifted_readout)
            with pytest.raises(NotImplementedError, match="cache layout could not be validated"):
                _kda_state_v_first(torch.device("cuda:0"), False, 192)
    finally:
        _kda_state_v_first.cache_clear()


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
def test_padding_does_not_extend_decode(compiled_serving_case, packed):
    args, quantizer, forward, _, _ = compiled_serving_case
    valid = 69  # The existing compiled shape uses a 65-token prefix.
    trimmed = [
        (x[:, :valid] if i < 5 else x).detach().clone().requires_grad_() for i, x in enumerate(args)
    ]
    initial = args[0].new_zeros(
        2, args[2].shape[2], args[0].shape[-1], args[2].shape[-1], dtype=torch.float32
    )
    initial[1].fill_(1.0)  # Empty rows preserve an incoming nonzero state without QDQ.
    expected, expected_state = forward(inputs=trimmed, initial_state=initial[:1])
    expected_grads = torch.autograd.grad(
        expected.square().sum() + expected_state.square().sum(), trimmed
    )
    padded = []
    for i, x in enumerate(args):
        if i < 5:
            x = torch.cat((x, torch.zeros_like(x)), dim=0)
            if packed:
                x = x.flatten(0, 1).unsqueeze(0)
        padded.append(x.detach().clone().requires_grad_())
    calls = []
    hook = quantizer.register_forward_hook(lambda *_: calls.append(True))
    try:
        output, state = forward(
            inputs=padded,
            initial_state=initial,
            prefill_lengths=[65, 0],
            sequence_lengths=[valid, 0],
            cu_seqlens=torch.tensor([0, 73, 146], device="cuda", dtype=torch.int32)
            if packed
            else None,
        )
    finally:
        hook.remove()
    assert len(calls) == 6  # Incoming prefix state, handoff, four writes; none for the empty row.
    output = output.reshape(2, 73, *output.shape[-2:])
    torch.testing.assert_close(output[:1, :valid], expected, rtol=0, atol=0)
    torch.testing.assert_close(state[:1], expected_state, rtol=0, atol=0)
    assert not output[0, valid:].count_nonzero() and not output[1].count_nonzero()
    torch.testing.assert_close(state[1], initial[1], rtol=0, atol=0)
    grads = torch.autograd.grad(output.square().sum() + state.square().sum(), padded)
    for i, (grad, expected_grad) in enumerate(zip(grads, expected_grads)):
        if i < 5:
            grad = grad.reshape(2, 73, *grad.shape[2:])
            assert not grad[0, valid:].count_nonzero() and not grad[1].count_nonzero()
            grad = grad[:1, :valid]
        torch.testing.assert_close(grad, expected_grad)

    if not packed:
        with torch.no_grad():
            prefill, prefill_state = forward(
                inputs=trimmed, prefill_lengths=[valid], initial_state=initial[:1]
            )
            padded_prefill, padded_state = forward(
                inputs=padded,
                prefill_lengths=[valid, 0],
                sequence_lengths=[valid, 0],
                initial_state=initial,
            )
        torch.testing.assert_close(padded_prefill[:1, :valid], prefill, rtol=0, atol=0)
        torch.testing.assert_close(padded_state[:1], prefill_state, rtol=0, atol=0)
        torch.testing.assert_close(padded_state[1], initial[1], rtol=0, atol=0)
        assert (
            not padded_prefill[0, valid:].count_nonzero() and not padded_prefill[1].count_nonzero()
        )
