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

"""Independent numerical oracles for materialized linear-attention sites."""

import torch
import torch.nn.functional as F

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import chunk_gdn_reference, matmul_gdn
from modelopt.torch.quantization.linear_attention.matmul import LinearAttentionMatmulSites
from modelopt.torch.quantization.nn import TensorQuantizer

FP8 = {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}


def inputs(device="cpu", *, packed=False, dtype=torch.float32, state_v_first=False):
    torch.manual_seed(113)
    shape = (1, 73, 1, 16)
    q, k = [F.normalize(torch.randn(shape, device=device, dtype=dtype), dim=-1) for _ in range(2)]
    v = torch.randn(1, 73, 2, 16, device=device, dtype=dtype)
    g = -torch.rand(1, 73, 2, device=device, dtype=dtype) * 0.05
    beta = torch.rand_like(g) * 0.4
    state = torch.randn(2 if packed else 1, 2, 16, 16, device=device, dtype=dtype) * 0.1
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return [x.requires_grad_() for x in (q, k, v, g, beta)], state.requires_grad_()


def fp8_oracle(x):
    with torch.no_grad():
        amax = x.float().abs().amax(-1, keepdim=True)
        scale = 448 / torch.where(amax > 2**-24, amax, torch.ones_like(amax))
        value = (x.float() * scale).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).float() * scale.reciprocal()
    return x + (value.to(x.dtype) - x).detach()


def nvfp4_oracle(x, tensor_amax=None):
    """Independent block-16 E2M1 rounding with E4M3 scales and a tensor amax."""
    with torch.no_grad():
        width = x.shape[-1]
        padded = F.pad(x.float(), (0, (-width) % 16))
        blocks = padded.reshape(*padded.shape[:-1], -1, 16)
        local_amax = blocks.abs().amax(-1, keepdim=True)
        global_amax = blocks.abs().amax() if tensor_amax is None else tensor_amax
        safe_global = torch.where(global_amax > 0, global_amax, torch.ones_like(global_amax))
        two_level = (6 / safe_global).double() * 448
        scaled = ((local_amax / 6).double() * two_level).float().clamp(0, 448)
        unscale64 = scaled.to(torch.float8_e4m3fn).double() / two_level
        unscale64 = torch.where(local_amax > 0, unscale64, torch.ones_like(unscale64))
        scale = unscale64.reciprocal().float()
        normalized = blocks * scale
        # Even codes precede odd codes so argmin implements round-to-nearest-even ties.
        grid = x.new_tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0], dtype=torch.float32)
        index = (normalized.abs().unsqueeze(-1) - grid).abs().argmin(-1)
        rounded = grid[index] * normalized.sign() * unscale64.float()
        value = rounded.reshape(padded.shape)[..., :width].to(x.dtype)
    return x + (value - x).detach()


def reference_matmul(enabled, policy):
    def mm(name, lhs, rhs):
        if f"{name}.lhs_quantizer" in enabled:
            lhs = fp8_oracle(lhs)
        if f"{name}.rhs_quantizer" in enabled:
            rhs = fp8_oracle(rhs.transpose(-1, -2)).transpose(-1, -2)
        cfg = policy.matmul.get(name)
        if cfg is None or cfg.accumulator_dtype is None:
            return torch.einsum("hij,hjk->hik", lhs, rhs)
        result = lhs.new_zeros((*lhs.shape[:-1], rhs.shape[-1]))
        for start in range(0, lhs.shape[-1], cfg.reduction_block):
            end = start + cfg.reduction_block
            result = result + torch.einsum(
                "hij,hjk->hik", lhs[..., start:end], rhs[..., start:end, :]
            )
            rounded = result.to(getattr(torch, cfg.accumulator_dtype)).to(lhs.dtype)
            result = result + (rounded - result).detach()
        return result

    return mm


def values_and_gradients(result, args, state):
    probes = [
        torch.linspace(-0.7, 0.9, x.numel(), device=x.device, dtype=x.dtype).reshape(x.shape)
        for x in result
    ]
    grad = torch.autograd.grad(
        sum((x * p).sum() for x, p in zip(result, probes)), (*args, state), retain_graph=True
    )
    return (*result, *grad)


def check_prefill_case(
    policy,
    enabled=(),
    *,
    device="cpu",
    packed=False,
    state_v_first=False,
    state_qdq=False,
    w_qdq=False,
    relative_tolerances=None,
):
    args, state = inputs(device, packed=packed, state_v_first=state_v_first)
    sites = LinearAttentionMatmulSites()
    for name in enabled:
        sites.get_submodule(name).set_from_attribute_config(FP8)
        sites.get_submodule(name).enable()
    w = TensorQuantizer(QuantizerAttributeConfig(**FP8, enable=w_qdq))
    kwargs = {
        "initial_state": state,
        "state_v_first": state_v_first,
        "cu_seqlens": torch.tensor([0, 5, 73], device=device) if packed else None,
    }
    actual = matmul_gdn(
        *args,
        sites=sites,
        policy=policy,
        w_quantizer=w,
        state_qdq=state_qdq,
        output_final_state=True,
        **kwargs,
    )

    def arithmetic(name, x):
        target = policy.elementwise.get(name)
        if target is None:
            return x
        return x + (x.to(getattr(torch, target)).to(x.dtype) - x).detach()

    expected = chunk_gdn_reference(
        *args,
        matmul=reference_matmul(enabled, policy),
        arithmetic=arithmetic,
        w_quantizer=fp8_oracle if w_qdq else None,
        state_qdq=state_qdq,
        state_qdq_block_v=policy.state.block_v,
        **kwargs,
    )
    for i, (a, e) in enumerate(
        zip(values_and_gradients(actual, args, state), values_and_gradients(expected, args, state))
    ):
        if relative_tolerances is None:
            torch.testing.assert_close(a, e, rtol=5e-4, atol=3e-6)
        else:
            tolerance = relative_tolerances[i >= 2]
            error = (a - e).norm() / e.norm().clamp_min(1e-6)
            assert error <= tolerance, (i, error.item(), tolerance)
    if enabled or w_qdq or state_qdq or policy.matmul or policy.elementwise:
        baseline = chunk_gdn_reference(*args, **kwargs)
        assert any(not torch.equal(a, b) for a, b in zip(actual, baseline))
