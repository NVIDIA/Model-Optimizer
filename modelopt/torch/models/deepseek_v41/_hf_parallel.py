# Reference: https://github.com/huggingface/transformers/blob/084b1fd1b9983dca2ce12441aec4478371b9fe8f/src/transformers/distributed/expert_parallel.py
# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned native EP dispatch preserving V4.1's pre-down routing precision."""

import torch
from torch.distributed._functional_collectives import all_to_all_single_autograd
from transformers.distributed.expert_parallel import EpDispatchExpertsParallel
from transformers.distributed.parallel_utils import _WaitTensor

__all__ = []


class V41EpDispatchExpertsParallel(EpDispatchExpertsParallel):
    def _run_weighted_local_experts(self, experts_forward, tokens, expert_ids, weights):
        if tokens.shape[0] == 0:
            dummy = tokens.sum(dim=0, keepdim=True)
            dummy_ids = torch.zeros((1, 1), dtype=expert_ids.dtype, device=tokens.device)
            dummy_weights = weights.sum().reshape(1, 1) + 1
            return experts_forward(dummy, dummy_ids, dummy_weights)[:0]
        return experts_forward(tokens, expert_ids.unsqueeze(-1), weights.reshape(-1, 1))

    def install_forward(self, module, ep_mesh, *, tp_mesh=None):
        experts_forward = module.forward
        ep_group, ep_size = ep_mesh.get_group(), ep_mesh.size()

        def ep_forward(hidden_states, top_k_index, top_k_weights):
            num_tokens = hidden_states.shape[0]
            (hidden_states, top_k_index, top_k_weights), _ = self.transform_inputs_pre_forward(
                module, (hidden_states, top_k_index, top_k_weights), {}, ep_mesh, tp_mesh=tp_mesh
            )
            with self.context_around_forward(module, ep_mesh):
                tokens, expert_ids, order, send_sizes, recv_sizes = self._dispatch_tokens(
                    hidden_states, top_k_index, module.num_experts, ep_group, ep_size
                )
                sent_weights = top_k_weights.float().reshape(-1)[order, None]
                weights = all_to_all_single_autograd(sent_weights, recv_sizes, send_sizes, ep_group)
                tokens, weights = _WaitTensor.apply(tokens), _WaitTensor.apply(weights)
                expert_output = self._run_weighted_local_experts(
                    experts_forward, tokens, expert_ids, weights
                )
                returned = all_to_all_single_autograd(
                    expert_output.float(), send_sizes, recv_sizes, ep_group
                )
                returned = _WaitTensor.apply(returned)
                restored = torch.empty_like(returned).index_copy(0, order, returned)
                output = restored.reshape(*top_k_weights.shape, returned.shape[-1]).sum(dim=1)
            return self.transform_output_post_forward(
                module, output, ep_mesh, tp_mesh=tp_mesh, num_tokens=num_tokens
            )

        module.forward = ep_forward
        module._modelopt_custom_ep_dispatch = True
        return module
