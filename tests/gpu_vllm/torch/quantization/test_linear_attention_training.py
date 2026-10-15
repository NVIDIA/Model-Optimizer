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
from _test_utils.torch.quantization.linear_attention_reference import recurrent_delta_rule_reference

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    gdn_state_qat,
    kda_state_qat,
)
from modelopt.torch.quantization.nn import TensorQuantizer

vllm = pytest.importorskip("vllm")

if vllm.__version__ != "0.30.0":
    pytest.skip("State QAT profiles target vLLM 0.30.0", allow_module_level=True)

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.third_party.flash_linear_attention import ops as gdn_ops
from vllm.third_party.flash_linear_attention.ops import kda as kda_ops


@torch.no_grad()
def _native_forward(args, quantizer, kda, kimi_k3):
    """Run the vLLM 0.30 Triton APIs with a persistent quantized cache."""
    q, k, v, raw_g, raw_beta, rate, bias = [x.detach().clone() for x in args]
    _, shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, 1, 1, q.shape[-1], v.shape[-1], 4
    )
    assert shape == (1, v.shape[-1], q.shape[-1])

    def layout(state):
        return state.transpose(-1, -2).contiguous()

    cache = torch.zeros(2, *shape, device=q.device)
    prefix_args = [x[:, :65].contiguous() for x in (q, k, v)]
    kwargs = {
        "initial_state": cache[1:],
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": torch.tensor([0, 65], device=q.device, dtype=torch.int32),
    }
    if kda:
        native = kda_ops
        if kimi_k3:
            from vllm.models.kimi_k3.nvidia.ops.third_party import kda as native
        g = kda_ops.fused_kda_gate(raw_g.flatten(-2), rate, q.shape[-1], g_bias=bias)
        beta = raw_beta.float().sigmoid()
        # vLLM 0.30's two explicit KDA profiles have different beta contracts.
        beta_arg = (
            {"raw_beta": raw_beta[:, :65].contiguous()}
            if kimi_k3
            else {"beta": beta[:, :65].contiguous()}
        )
        prefix, state = native.chunk_kda_with_fused_gate(
            *prefix_args,
            raw_g=raw_g[:, :65].contiguous(),
            A_log=rate,
            g_bias=bias,
            **beta_arg,
            **kwargs,
        )
    else:
        prepared = gdn_ops.fused_post_conv_prep(
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
        prefix, state = gdn_ops.chunk_gated_delta_rule(*[x[None] for x in prepared], **kwargs)
    cache[1] = layout(quantizer(layout(state)[0]))
    outputs = [prefix]
    indices = torch.ones(1, device=q.device, dtype=torch.int32)
    cu = torch.tensor([0, 1], device=q.device, dtype=torch.int32)
    for token in range(65, 67):
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
            output, _ = gdn_ops.fused_recurrent_gated_delta_rule_packed_decode(
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
    torch.manual_seed(73)
    keys = 32  # Share one head shape across GDN/KDA to reuse native and FLA compilation.
    args = [torch.randn(1, 67, 1, keys, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    args += [
        torch.randn((1, 67, 1, keys) if kda else (1, 67, 1), device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 67, 1, device="cuda", dtype=torch.bfloat16),
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

    output, state = forward()
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
    assert len(calls) == 3  # Handoff plus two recurrent writes.
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
            # Reuse the compiled 65-token prefix for full-prefill and suffix cases.
            for tokens in (65, 67):
                inputs = [x[:, :tokens] if i < 5 else x for i, x in enumerate(args)]
                actual, final = forward(inputs=inputs, initial_state=initial)
                q, k, v, raw_g, raw_beta, rate, bias = [x.float() for x in inputs]
                q, k = [x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt() for x in (q, k)]
                q[:, :65] = q[:, :65].bfloat16().float()
                k[:, :65] = k[:, :65].bfloat16().float()
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


@pytest.fixture(scope="module")
@torch.no_grad()
def compiled_gdn_layout_case():
    """Compile two-head decode so strided head/value axes are observable."""
    torch.manual_seed(732)
    q, k, v = [torch.randn(1, 2, 2, 32, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    raw_g, raw_beta = [torch.randn(1, 2, 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    gates = (raw_g, raw_beta, torch.full((2,), -3.0, device="cuda"), torch.randn(2, device="cuda"))
    initial = torch.randn(1, 2, 32, 32, device="cuda")

    def forward(values, gate_inputs):
        raw_g, raw_beta, rate, bias = gate_inputs
        return gdn_state_qat(
            q,
            k,
            values,
            -rate.exp() * torch.nn.functional.softplus(raw_g.float() + bias),
            raw_beta.float().sigmoid(),
            policy=LinearAttentionConfig(backend="serving"),
            initial_state=initial,
            prefill_lengths=[0],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            gate_inputs=gate_inputs,
        )

    expected = forward(v, gates)
    torch.cuda.synchronize()
    return v, gates, initial, forward, expected


@pytest.mark.parametrize("strided", ["values", "gates"])
@torch.no_grad()
def test_gdn_strided_inputs_match_contiguous(compiled_gdn_layout_case, strided):
    v, gates, initial, forward, expected = compiled_gdn_layout_case
    original_state = initial.clone()
    if strided == "values":
        v = v.transpose(-1, -2).contiguous().transpose(-1, -2)
    else:
        gates = tuple(torch.stack((x, torch.zeros_like(x)), dim=-1)[..., 0] for x in gates)
    actual = forward(v, gates)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    torch.testing.assert_close(initial, original_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("compiled_serving_case", "overrides", "message"),
    [
        ("gdn", {"gate_inputs": None}, "raw gate_inputs"),
        (
            "kimi_k3",
            {"gate_inputs": None, "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False},
            "raw gate_inputs",
        ),
        ("kimi_k3", {"use_qk_l2norm_in_kernel": False}, "normalization"),
    ],
    indirect=["compiled_serving_case"],
    scope="module",
)
def test_rejects_silent_serving_mismatches(compiled_serving_case, overrides, message):
    _, _, forward, _, _ = compiled_serving_case
    with pytest.raises(ValueError, match=message):
        forward(**overrides)


@pytest.mark.parametrize("compiled_serving_case", ["gdn", "kimi_k3"], indirect=True, scope="module")
@torch.no_grad()
def test_raw_projections_match_bf16_serving(compiled_serving_case):
    args, quantizer, forward, _, kimi_k3 = compiled_serving_case
    inputs = [x.detach() for x in args]
    inputs[3:5] = [x.float() + 0.0007 for x in inputs[3:5]]
    # Rates and biases keep their FP32 precision, independent of projection rounding.
    inputs[5:] = [x + 0.0123 for x in inputs[5:]]
    rounded = [*inputs[:3], *[x.bfloat16() for x in inputs[3:5]], *inputs[5:]]
    expected = _native_forward(rounded, quantizer, kimi_k3, kimi_k3)
    actual = forward(inputs=inputs)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("compiled_serving_case", "packed"),
    [("gdn", False), ("kda", True)],
    indirect=["compiled_serving_case"],
    scope="module",
    ids=["gdn-dense", "kda-packed"],
)
def test_padding_does_not_extend_decode(compiled_serving_case, packed):
    args, quantizer, forward, _, _ = compiled_serving_case
    valid = 66  # One recurrent write and one padding token after the 65-token prefix.
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
            cu_seqlens=torch.tensor([0, 67, 134], device="cuda", dtype=torch.int32)
            if packed
            else None,
        )
    finally:
        hook.remove()
    assert len(calls) == 3  # Incoming prefix state, handoff, one write; none for the empty row.
    output = output.reshape(2, 67, *output.shape[-2:])
    torch.testing.assert_close(output[:1, :valid], expected, rtol=0, atol=0)
    torch.testing.assert_close(state[:1], expected_state, rtol=0, atol=0)
    assert not output[0, valid:].count_nonzero() and not output[1].count_nonzero()
    torch.testing.assert_close(state[1], initial[1], rtol=0, atol=0)
    grads = torch.autograd.grad(output.square().sum() + state.square().sum(), padded)
    for i, (grad, expected_grad) in enumerate(zip(grads, expected_grads)):
        if i < 5:
            grad = grad.reshape(2, 67, *grad.shape[2:])
            assert not grad[0, valid:].count_nonzero() and not grad[1].count_nonzero()
            grad = grad[:1, :valid]
        torch.testing.assert_close(grad, expected_grad)

    if not packed:
        with torch.no_grad():
            prefix_inputs = [x[:, :65] if i < 5 else x for i, x in enumerate(trimmed)]
            prefill, prefill_state = forward(inputs=prefix_inputs, initial_state=initial[:1])
            padded_prefill, padded_state = forward(
                inputs=padded,
                prefill_lengths=[65, 0],
                sequence_lengths=[65, 0],
                initial_state=initial,
            )
        torch.testing.assert_close(padded_prefill[:1, :65], prefill, rtol=0, atol=0)
        torch.testing.assert_close(padded_state[:1], prefill_state, rtol=0, atol=0)
        torch.testing.assert_close(padded_state[1], initial[1], rtol=0, atol=0)
        assert not padded_prefill[0, 65:].count_nonzero() and not padded_prefill[1].count_nonzero()
