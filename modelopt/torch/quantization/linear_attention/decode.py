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

"""Differentiable per-token state quantization for a training suffix."""

import torch

from .utils import _state_qdq

__all__ = []


def _recurrent_decode(
    q,
    k,
    v,
    g,
    beta,
    *,
    config,
    state_quantizer,
    initial_state,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    gate_inputs=None,
):
    """Run one training suffix and optionally return its dense final state.

    The caller validates shapes, policy and quantizer, and supplies FP32 inputs and prefill state.
    Empty suffixes preserve that state without a handoff QDQ or native launch.
    """
    if len(q) == 0:
        # Keep empty input gradients defined without introducing a state write.
        zero = (q.sum() + k.sum() + v.sum() + g.sum() + beta.sum()) * 0
        return v + zero, initial_state if output_final_state else None
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if config.precision == "replayssm":
        from .replay import _replay_decode

        output, carry = _replay_decode(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            config.replay_window,
            state_quantizer is not None,
            scale,
            use_qk_l2norm_in_kernel,
            gate_inputs,
        )
        return output, carry.reconstruct() if output_final_state else None
    # Import the optional native backend only for the ordinary state-QDQ path.
    from ._vllm_autograd import step

    state = _state_qdq(initial_state, state_quantizer, config.state_block_v)
    with torch.autocast(device_type=q.device.type, enabled=False):
        outputs = []
        for t in range(len(q)):
            # The output reads the working state before QDQ writes the next token's cache.
            output, working = step(
                q[t],
                k[t],
                v[t],
                g[t],
                beta[t],
                state,
                scale,
                use_qk_l2norm_in_kernel,
                precision=config.precision,
                gate_inputs=(gate_inputs[0][t], gate_inputs[1][t], *gate_inputs[2:])
                if gate_inputs is not None
                else None,
            )
            outputs.append(output)
            # A new tensor preserves the state saved by earlier tokens' backward passes.
            state = _state_qdq(working, state_quantizer, config.state_block_v)
    return torch.stack(outputs), state if output_final_state else None
