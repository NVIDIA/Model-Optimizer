# Adapted from: https://github.com/ctlllll/axolotl/blob/f86767e/src/axolotl/monkeypatch/medusa_utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""Medusa speculative decoding plugin for HuggingFace models."""

import contextlib
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Cache, PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel
from ..utils import ResBlock

__all__ = ["HFMedusaModel"]

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@MedusaDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFMedusaModel(MedusaModel):
    """Medusa Model Class for huggingface models."""

    def modify(self, medusa_num_heads=0, medusa_num_layers=0):
        """Constructor.

        Args:
            medusa_num_heads: number of medusa heads.
            medusa_num_layers: number of ResBlock layers in each head.
        """
        super().modify(medusa_num_heads=medusa_num_heads, medusa_num_layers=medusa_num_layers)
        self.config.medusa = {
            "num_medusa_heads": medusa_num_heads,
            "num_medusa_layers": medusa_num_layers,
        }

        hidden_size = self.lm_head.weight.shape[-1]
        vocab_size = self.lm_head.weight.shape[0]

        # Create a list of Medusa heads
        self.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(hidden_size) for _ in range(self.medusa_num_layers)]),
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(self.medusa_num_heads)
            ]
        )

        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_heads.to(self.lm_head.weight.dtype).to(self.lm_head.weight.device)
        self.medusa_heads.device = self.lm_head.weight.device
        if hasattr(self, "hf_device_map") and "lm_head" in self.hf_device_map:
            self.hf_device_map["medusa_heads"] = self.hf_device_map["lm_head"]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        freeze_base_model: bool = True,
        medusa_heads_coefficient: float | None = 0.2,
        medusa_decay_coefficient: float | None = 0.8,
        **kwargs,
    ) -> Any:
        """Forward pass of the MedusaModel.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
        """
        # Pass input through the base model
        with torch.no_grad() if freeze_base_model else contextlib.nullcontext():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                rcache_position=cache_position,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        medusa_logits = [
            self.medusa_heads[i](hidden_states[:, slice_indices, :])
            for i in range(self.medusa_num_heads)
        ]

        if labels is not None:
            loss = 0
            loss_fct = CrossEntropyLoss()
            # Base model loss
            if not freeze_base_model:
                loss_logits = logits.view(-1, logits.shape[-1])
                loss_labels = labels.view(-1)
                base_model_loss = loss_fct(loss_logits, loss_labels)
                loss += base_model_loss
            # Medusa loss
            for i in range(self.medusa_num_heads):
                labels = labels[..., 1:].contiguous()
                loss_logits = medusa_logits[i][:, : -(1 + i)].contiguous()
                loss_logits = loss_logits.view(-1, loss_logits.shape[-1])
                loss_labels = labels.view(-1)
                loss += (
                    loss_fct(loss_logits, loss_labels)
                    * medusa_decay_coefficient**i
                    * medusa_heads_coefficient
                )
        else:
            loss = None

        return ModelOutput(
            loss=loss,
            logits=logits,
            medusa_logits=medusa_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
