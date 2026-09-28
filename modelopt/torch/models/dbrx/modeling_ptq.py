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

"""DBRX PTQ modeling (HF model type ``dbrx``)."""

import torch
import torch.nn as nn

from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry
from modelopt.torch.quantization.plugins.custom import CUSTOM_MODEL_PLUGINS
from modelopt.torch.quantization.plugins.huggingface import _QuantSparseSequentialMoe

__all__: list[str] = []


# For more information on DbrxExpert, see https://github.com/huggingface/transformers/blob/dcdda532/src/transformers/models/dbrx/modeling_dbrx.py#L756
class _QuantDbrxExperts(QuantModule):
    def _setup(self):
        """Modify the DbrxExpert."""
        # No setup is needed for DbrxExpert, we only need to update DbrxExpertGLU

    def forward(
        self,
        x: torch.Tensor,
        top_experts: torch.LongTensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = nn.functional.one_hot(top_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        for expert_idx in range(self.num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = (
                self.mlp(expert_tokens, expert_idx) * top_weights[token_list, topk_list, None]
            )

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out


class _QuantDbrxExpertGLU(QuantModule):
    def _setup(self):
        """Modify the DbrxExpertGLU by using nn.Linear layers."""
        dtype, device = self.w1.dtype, self.w1.device

        def _copy_weights(modules, weights):
            modules.to(dtype=dtype, device=device)
            for expert_idx, module in enumerate(modules):
                with torch.no_grad():
                    module.weight.copy_(weights[expert_idx].detach())

        # In transformers 5.0, DbrxExpertGLU.forward uses raw matmul: x @ w1[i] where
        # w1[i] has shape (ffn_hidden_size, hidden_size). To match via F.linear (which
        # computes x @ W.T), we store weights transposed: W = w1[i].T.
        self.w1_linear = nn.ModuleList(
            [
                nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
                for _ in range(self.moe_num_experts)
            ]
        )
        _copy_weights(
            self.w1_linear,
            self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size).transpose(
                1, 2
            ),
        )
        delattr(self, "w1")

        self.v1_linear = nn.ModuleList(
            [
                nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
                for _ in range(self.moe_num_experts)
            ]
        )
        _copy_weights(
            self.v1_linear,
            self.v1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size).transpose(
                1, 2
            ),
        )
        delattr(self, "v1")

        # w2: down_proj uses intermediate.matmul(w2[i].t()) = F.linear(intermediate, w2[i])
        # so W = w2[i] directly (no extra transpose needed).
        self.w2_linear = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
                for _ in range(self.moe_num_experts)
            ]
        )
        _copy_weights(
            self.w2_linear,
            self.w2.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size),
        )
        delattr(self, "w2")

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        x1 = self.w1_linear[expert_idx](x)
        x2 = self.v1_linear[expert_idx](x)
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        return self.w2_linear[expert_idx](x1)


class _QuantDbrxFFN(_QuantSparseSequentialMoe):
    @property
    def num_experts(self):
        return self.router.moe_num_experts

    @property
    def top_k(self):
        # In older transformers, top_k was stored on DbrxRouter as moe_top_k.
        # In transformers 5.0, DbrxFFN stores it as a plain attribute (top_k).
        if hasattr(self.router, "moe_top_k"):
            return self.router.moe_top_k
        return self.__dict__.get("top_k", 1)

    @top_k.setter
    def top_k(self, value):
        if hasattr(self.router, "moe_top_k"):
            self.router.moe_top_k = value
        else:
            self.__dict__["top_k"] = value


try:
    from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU, DbrxExperts, DbrxFFN

    if DbrxExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({DbrxExperts: "hf.DbrxExperts"})(_QuantDbrxExperts)

    if DbrxExpertGLU not in QuantModuleRegistry:
        QuantModuleRegistry.register({DbrxExpertGLU: "hf.DbrxExpertGLU"})(_QuantDbrxExpertGLU)

    if DbrxFFN not in QuantModuleRegistry:
        QuantModuleRegistry.register({DbrxFFN: "hf.DbrxFFN"})(_QuantDbrxFFN)
except ImportError:
    pass


def register_dbrx_moe_on_the_fly(model):
    """Register DBRX MoE modules as QUANT_MODULE.

    The MoE class in DBRX is `transformers_modules.modeling_dbrx.DbrxExpertGLU`, which loads dynamically.
    """
    if type(model).__name__ == "DbrxForCausalLM":
        moe_type = type(model.transformer.blocks[0].ffn.experts.mlp)
        # Create a QuantDbrxExpertGLU class on the fly
        if QuantModuleRegistry.get(moe_type) is None:
            QuantModuleRegistry.register({moe_type: moe_type.__name__})(_QuantDbrxExpertGLU)


CUSTOM_MODEL_PLUGINS.add(register_dbrx_moe_on_the_fly)
