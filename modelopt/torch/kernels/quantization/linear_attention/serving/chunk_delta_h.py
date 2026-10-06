# Adapted from: https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/model_executor/layers/fla/ops/chunk_delta_h.py
# Modifications: import the kernel; retain FP32 intermediates for the training adjoint.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Original FLA code: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
# FLA is licensed under the MIT license reproduced in the root LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""Save FP32 training intermediates using vLLM's unchanged chunk-state kernel."""

import torch
import triton
from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_offsets


def chunk_state(k, w, u, g, gk, initial_state, cu_seqlens):
    """Return chunk-start states, residual values, and final state for one sequence."""
    _, length, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    # The Torch adjoint needs unrounded values. Output kernels receive BF16 casts.
    h = k.new_empty(1, triton.cdiv(length, 64), heads, key_dim, value_dim, dtype=torch.float32)
    updated = torch.empty_like(u, dtype=torch.float32)
    final = torch.empty_like(initial_state, dtype=torch.float32)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[
        lambda meta: (triton.cdiv(value_dim, meta["BV"]), heads)
    ](
        k=k,
        v=u,
        w=w,
        v_new=updated,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final,
        cu_seqlens=cu_seqlens,
        chunk_offsets=prepare_chunk_offsets(cu_seqlens, 64),
        T=length,
        H=heads,
        Hg=heads,
        K=key_dim,
        V=value_dim,
        BT=64,
    )
    return h, updated, final
