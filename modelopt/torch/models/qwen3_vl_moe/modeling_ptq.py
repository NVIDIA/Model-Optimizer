# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Qwen3-VL-MoE PTQ modeling (HF model type ``qwen3_vl_moe``)."""

import torch
import torch.nn as nn

from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry

__all__: list[str] = []


class _QuantQwen3VLMoeTextExperts(QuantModule):
    """Quantized wrapper for the pre-transformers-5.12 ``Qwen3VLMoeTextExperts`` layout.

    That layout stores ``gate_up_proj`` as (num_experts, hidden_size, 2*expert_dim) and runs
    the experts through ``torch.bmm``/``@``, so it is unrolled into ``nn.Linear`` modules here.
    transformers>=5.12 moved this module to the standard fused layout handled by
    :class:`_QuantFusedExperts`; see the registration site below.
    """

    def _setup(self):
        """Modify the Qwen3VLMoeTextExperts by using nn.Linear layers."""
        from accelerate import init_empty_weights

        dtype, device = self.gate_up_proj.dtype, self.gate_up_proj.device

        def _copy_weight(module, weight):
            module.to_empty(device=device)
            with torch.no_grad():
                module.weight.data = weight.detach().data.to(dtype=dtype, device=device)

        # The attribute name was changed from `intermediate_size` to `intermediate_dim` in
        # https://github.com/huggingface/transformers/commit/0642963ba13f2dae0596fe489415569e1d91fbda
        if hasattr(self, "intermediate_size"):
            expert_dim = self.intermediate_size
        elif hasattr(self, "intermediate_dim"):
            expert_dim = self.intermediate_dim
        else:
            raise AttributeError("Could not find intermediate dimension size in model")

        with init_empty_weights():
            gate_proj = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, expert_dim, bias=False)
                    for _ in range(self.num_experts)
                ]
            )
            up_proj = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, expert_dim, bias=False)
                    for _ in range(self.num_experts)
                ]
            )
            down_proj = nn.ModuleList(
                [
                    nn.Linear(expert_dim, self.hidden_size, bias=False)
                    for _ in range(self.num_experts)
                ]
            )

        for idx in range(self.num_experts):
            _copy_weight(gate_proj[idx], self.gate_up_proj[idx, :, :expert_dim].T)
            _copy_weight(up_proj[idx], self.gate_up_proj[idx, :, expert_dim:].T)
            _copy_weight(down_proj[idx], self.down_proj[idx, :].T)

        delattr(self, "gate_up_proj")
        delattr(self, "down_proj")
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx[0]])
            current_state = hidden_states[token_idx]
            gate = self.gate_proj[expert_idx](current_state)
            up = self.up_proj[expert_idx](current_state)
            gated_output = up * self.act_fn(gate)
            out = self.down_proj[expert_idx](gated_output)
            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        next_states = next_states.view(batch_size, -1, self.hidden_size)

        return next_states


try:
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

    # transformers>=5.12 rewrote Qwen3VLMoeTextExperts onto the standard
    # ``@use_experts_implementation`` fused layout: ``hidden_size``/``expert_dim`` became
    # ``hidden_dim``/``intermediate_dim``, ``gate_up_proj`` was transposed to
    # (num_experts, 2*intermediate_dim, hidden_dim), and the forward now calls ``F.linear``
    # twice per expert. ``_QuantQwen3VLMoeTextExperts`` only understands the older layout,
    # so registering it against the new one crashes on ``self.hidden_size`` (nvbug 6518551).
    # The decorator sets ``_apply_gate`` on the class; use it to detect the new layout and
    # leave those modules to ``register_fused_experts_on_the_fly``, which claims them with
    # the generic ``_QuantFusedExperts``. The old layout must stay explicitly registered:
    # it is structurally indistinguishable from a generic fused-experts module, yet its
    # forward uses ``torch.bmm``/``@`` rather than ``F.linear``, so the generic wrapper
    # would silently quantize nothing.
    if Qwen3VLMoeTextExperts not in QuantModuleRegistry and not hasattr(
        Qwen3VLMoeTextExperts, "_apply_gate"
    ):
        QuantModuleRegistry.register({Qwen3VLMoeTextExperts: "hf.Qwen3VLMoeTextExperts"})(
            _QuantQwen3VLMoeTextExperts
        )
except ImportError:
    pass
