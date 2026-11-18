# Reference: https://github.com/huggingface/transformers/blob/084b1fd1b9983dca2ce12441aec4478371b9fe8f/src/transformers/distributed/expert_parallel.py
# Copyright 2026 The HuggingFace Team. All rights reserved.

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


"""Model-owned native EP dispatch preserving V4.1's pre-down routing precision."""

import torch
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.tensor import DTensor
from transformers.distributed.expert_parallel import AllToAllEP, EpDispatchExpertsParallel
from transformers.distributed.parallel_utils import _WaitTensor

__all__ = []


class _V41AllToAll(AllToAllEP):
    def __call__(self, hidden_states, top_k_index, top_k_weights):
        tokens, expert_ids, _, state = self._dispatch_tokens(
            hidden_states, top_k_index, top_k_weights
        )
        # V4.1 rounds the weighted activation before down_proj. HF's ordinary
        # combine applies weights after down_proj, which changes BF16 outputs/VJPs.
        sent = state.top_k_weights.float().reshape(-1)[state.order, None]
        weights = _WaitTensor.apply(
            all_to_all_single_autograd(
                sent, state.recv_sizes, state.send_sizes, self.ep_mesh.get_group()
            )
        )
        if tokens.shape[0] == 0:
            output = self.experts_forward(
                tokens.sum(dim=0, keepdim=True),
                expert_ids.new_zeros((1, 1)),
                weights.sum().reshape(1, 1) + 1,
            )[:0]
        else:
            output = self.experts_forward(tokens, expert_ids, weights)
        # Reuse stable 5.23's TP slicing/gather and inverse exchange, without a
        # second application of the original routing weights on the sender.
        return self._combine_tokens(
            output.float(), state._replace(top_k_weights=torch.ones_like(state.top_k_weights))
        )


class V41EpDispatchExpertsParallel(EpDispatchExpertsParallel):
    supported_implementations = ("auto", "all_to_all")

    def install_forward(self, module, ep_mesh, *, tp_mesh, ep_implementation="auto"):
        if ep_implementation not in self.supported_implementations:
            raise ValueError(
                "V4.1 pre-down routing supports the native all-to-all EP implementation."
            )
        implementation = _V41AllToAll(module, ep_mesh, tp_mesh)

        def ep_forward(hidden_states, top_k_index, top_k_weights):
            hidden_states = (
                hidden_states.to_local() if isinstance(hidden_states, DTensor) else hidden_states
            )
            top_k_weights = (
                top_k_weights.to_local() if isinstance(top_k_weights, DTensor) else top_k_weights
            )
            with self.context_around_forward(module, ep_mesh):
                return implementation(hidden_states, top_k_index, top_k_weights)

        module.forward = ep_forward
        module._modelopt_custom_ep_dispatch = True
        return module
