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

"""Keep native vLLM imports and state layout changes at the kernel boundary."""

from functools import cache
from importlib import import_module
from importlib.util import find_spec
from inspect import signature

import torch

__all__ = []

_FLA_OPS = (
    "vllm.third_party.flash_linear_attention.ops"
    if find_spec("vllm.third_party.flash_linear_attention") is not None
    else "vllm.model_executor.layers.fla.ops"
)


@cache
def state_v_first():
    """Read the serving cache contract using unequal dimensions, independent of version tags."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    _, shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_world_size=1,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=32,
        head_v_dim=64,
        conv_kernel_size=4,
    )
    if shape == (1, 64, 32):
        return True
    if shape == (1, 32, 64):
        return False
    raise NotImplementedError(f"Unsupported vLLM recurrent-state layout: {shape}")


def fla_module(name):
    return import_module(f"{_FLA_OPS}.{name}")


@cache
def optional_module(path):
    """Allow missing APIs, but never hide a broken dependency inside an installed module."""
    try:
        return import_module(path)
    except ModuleNotFoundError as error:
        if error.name is not None and (path == error.name or path.startswith(error.name + ".")):
            return None
        raise


@cache
def kda_module(kimi_k3=False):
    """Load only the explicitly selected KDA implementation, without backend fallback."""
    if not kimi_k3:
        return fla_module("kda")
    try:
        return import_module("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
    except ImportError as error:
        raise ImportError(
            "The selected vLLM KDA Triton backend requires its model package dependencies. "
            "Install the dependencies for this vLLM build; ModelOpt will not switch arithmetic "
            "to a different KDA backend after an import failure."
        ) from error


@cache
def gdn_module(name):
    return optional_module(f"{_FLA_OPS}.{name}")


@cache
def kda_prefill_beta(kimi_k3=False):
    """Resolve the raw-gate chunk callable's beta contract, independent of decode APIs."""
    kernel = getattr(kda_module(kimi_k3), "chunk_kda_with_fused_gate", None)
    if kernel is None:
        return None
    parameters = signature(kernel).parameters
    for name in ("raw_beta", "beta"):
        if name in parameters:
            return name
    raise NotImplementedError("Unsupported vLLM chunk_kda_with_fused_gate beta signature")


def kda_chunk(q, k, v, g, beta, *, gate_inputs=None, kimi_k3=False, **kwargs):
    """Use the same native chunk dispatch for execution and cache-contract checks."""
    native = kda_module(kimi_k3)
    if gate_inputs is not None and (beta_name := kda_prefill_beta(kimi_k3)) is not None:
        raw_g, raw_beta, rate, bias = gate_inputs
        return native.chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            **{beta_name: raw_beta if beta_name == "raw_beta" else beta},
            A_log=rate,
            g_bias=bias,
            **kwargs,
        )
    if kimi_k3:
        raise ValueError("vllm_kimi_k3 requires native fused-gate prefill and raw gate_inputs")
    return native.chunk_kda(q, k, v, g, beta, **kwargs)


@cache
def _kda_state_v_first(device, kimi_k3, keys, raw_gates=False):
    """Check the selected KDA paths at the actual head size, once per device."""
    with torch.no_grad():
        q = torch.zeros(1, 1, 1, keys, device=device, dtype=torch.bfloat16)
        q[..., 0] = 1
        zero = torch.zeros_like(q)
        # Distinct row/column readouts made from exactly representable BF16 values,
        # including head sizes such as 192 where arange(K*K) would round.
        pattern = torch.zeros(keys, keys, device=device, dtype=torch.float32)
        markers = torch.arange(1, keys + 1, device=device, dtype=torch.float32)
        pattern[0, :] = -markers
        pattern[:, 0] = markers
        state = pattern.expand(2, 1, keys, keys).contiguous()
        output, _ = kda_module(kimi_k3).fused_recurrent_kda_fwd(
            q=q,
            k=zero,
            v=zero,
            g=zero.float(),
            beta=torch.zeros(1, 1, 1, device=device),
            scale=1.0,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=torch.arange(2, device=device, dtype=torch.int32),
            ssm_state_indices=torch.ones(1, device=device, dtype=torch.int32),
            use_qk_l2norm_in_kernel=False,
        )
        row = output.flatten().float()
        if torch.equal(row, pattern[:, 0]):
            value_first = True
        elif torch.equal(row, pattern[0]):
            value_first = False
        else:
            raise NotImplementedError("The installed vLLM KDA cache layout could not be validated")
        # Some native revisions changed cache storage without updating chunk readout.
        # Reject that backend rather than training against a silently transposed prefix.
        chunk_args = {
            "q": q,
            "k": zero,
            "v": zero.clone(),
            "scale": 1.0,
            "initial_state": state[1:].clone(),
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": kimi_k3,
            "cu_seqlens": torch.arange(2, device=device, dtype=torch.int32),
        }
        gate_inputs = None
        if raw_gates or kimi_k3:
            raw_g = torch.full_like(q, -80)
            raw_beta = torch.full((1, 1, 1), -80.0, device=device)
            rate, bias = torch.zeros(1, device=device), torch.zeros(keys, device=device)
            gate_inputs = (raw_g, raw_beta, rate, bias)
        if kimi_k3:
            packed, _ = kda_module(True).fused_recurrent_kda_packed_decode(
                mixed_qkv=torch.cat((q.flatten(), zero.flatten(), zero.flatten()))[None],
                raw_g=raw_g,
                raw_beta=raw_beta,
                A_log=rate,
                dt_bias=bias,
                lower_bound=None,
                initial_state=state.clone(),
                state_indices=torch.ones(1, device=device, dtype=torch.int32),
                scale=1.0,
            )
            if not torch.equal(packed.flatten().float(), row):
                raise NotImplementedError("The installed vLLM KDA packed cache layout disagrees")
        chunk_output, _ = kda_chunk(
            g=zero.float(),
            beta=torch.zeros(1, 1, 1, device=device),
            gate_inputs=gate_inputs,
            kimi_k3=kimi_k3,
            **chunk_args,
        )
        if not torch.equal(chunk_output.flatten().float(), row):
            raise NotImplementedError(
                "The installed vLLM KDA chunk/decode cache readout disagrees; "
                "use a vLLM build with the KDA chunk-state layout fix"
            )
        return value_first


def state_layout(state, *, kda=False, kimi_k3=False, raw_gates=False):
    """Convert between ModelOpt and the selected model's native state layout."""
    value_first = (
        _kda_state_v_first(state.device, kimi_k3, state.shape[-1], raw_gates)
        if kda
        else state_v_first()
    )
    return state.transpose(-1, -2).contiguous() if value_first else state.contiguous()
