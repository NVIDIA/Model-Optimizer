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

"""GDN adapter for serving-aligned recurrent-state QAT."""

from .config import LinearAttentionConfig
from .training import _prefill_decode_forward, _prepare_prefill_inputs

__all__ = ["gdn_state_qat", "matmul_gdn"]


def gdn_state_qat(
    q,
    k,
    v,
    g,
    beta,
    *,
    policy: LinearAttentionConfig,
    state_qdq=False,
    state_format="fp8_e4m3",
    state_quantizer=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    A_log=None,  # noqa: N803 - match the FLA kernel signature
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    state_v_first=False,
    chunk_size=64,
    cp_context=None,
    prefill_lengths=None,
    replay_gate_inputs=None,
):
    """Adapt Megatron's FLA-style GDN call to serving-aligned state QAT.

    Validate prepared scalar log gates and promote inputs to FP32 working values.
    The shared training forward runs the chunked prefix and recurrent suffix with
    native forward values, configured state QDQ, and a differentiable Torch adjoint.
    """
    if cp_context is not None:
        raise NotImplementedError("GDN state QAT does not support context parallelism")
    output_dtype = q.dtype
    beta_dtype = beta.dtype
    q, k, v, g, beta = _prepare_prefill_inputs(
        q, k, v, g, beta, policy=policy, chunk_size=chunk_size
    )
    if use_gate_in_kernel or use_beta_sigmoid_in_kernel:
        raise ValueError(
            "Serving GDN expects prepared log gates and beta from the Megatron adapter"
        )
    if g.ndim != 3:
        raise ValueError("gdn_state_qat requires scalar GDN log gates")
    return _prefill_decode_forward(
        q,
        k,
        v,
        g,
        beta,
        policy=policy,
        state_qdq=state_qdq,
        state_format=state_format,
        state_quantizer=state_quantizer,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        state_v_first=state_v_first,
        output_dtype=output_dtype,
        beta_dtype=beta_dtype,
        prefill_lengths=prefill_lengths,
        replay_gate_inputs=replay_gate_inputs,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


# Compatibility name for callers using the original adapter API.
matmul_gdn = gdn_state_qat
