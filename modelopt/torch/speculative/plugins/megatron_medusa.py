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

"""Plugin to add Medusa support for Megatron-Core GPT model."""

import warnings

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.gpt import GPTModel
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer.module import MegatronModule

from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel


class MedusaLayer(MegatronModule):
    """MedusaLayer impl following TensorRT-LLM's model definition.

    Medusa layer consists of a column parallel linear following a silu.
    """

    def __init__(self, config, bias=True):
        """Constructor.

        Args:
            config: MCore transformer config
        """
        super().__init__(config=config)

        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        )

        self.activation_func = F.silu

        self.linear = torch.nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=bias,
            dtype=config.params_dtype,
            device=device,
        )

    def forward(self, x):
        """Forward function."""
        y = self.linear(x)
        return x + self.activation_func(y), None


class MedusaHead(MegatronModule):
    """MedusaHead impl following TensorRT-LLM's model definition.

    https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/medusa/model.py
    Medusa head consists of several MedusaLayers and an lm_head.
    """

    def __init__(self, config, vocab_size: int, num_layers: int = 1, parallel_output: bool = True):
        """Constructor.

        Args:
            config: MCore transformer config
            vocab_size: vocabulary size
            num_layers: number of Medusa layers
            parallel_output: if False, then all_gather the logits
        """
        super().__init__(config=config)

        self.medusa_layers = torch.nn.ModuleList([MedusaLayer(config) for _ in range(num_layers)])

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not parallel_output,
            skip_weight_param_allocation=False,
        )

        def load_state_dict_post_hook(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        self.register_load_state_dict_post_hook(load_state_dict_post_hook)

    def forward(self, x):
        """Forward function."""
        for layer in self.medusa_layers:
            x, _ = layer(x)
        return self.lm_head(x)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Return MCore sharded_state_dict."""
        assert not sharded_offsets, "Unexpected sharded offsets"
        sharded_state_dict = {}
        layer_prefix = f"{prefix}medusa_layers."
        for i, layer in enumerate(self.medusa_layers):
            state_dict_prefix = f"{layer_prefix}{i}."
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            sharded_state_dict.update(layer_sharded_state_dict)
        sharded_state_dict.update(
            self.lm_head.sharded_state_dict(f"{prefix}lm_head.", sharded_offsets, metadata)
        )
        return sharded_state_dict


@MedusaDMRegistry.register({GPTModel: "megatron.core.models.gpt.GPTModel"})
class _DynamicMedusaGPTModel(MedusaModel):
    """A ``megatron.core.models.gpt.GPTModel`` model with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self._register_temp_attribute("medusa_report_acc", True)
        self._register_temp_attribute("medusa_freeze_base_model", True)
        self._register_temp_attribute("calibration_mode", False)

    def modify(
        self,
        medusa_num_heads=0,
        medusa_num_layers=0,
        medusa_freeze_base_model=True,
        medusa_report_acc=True,
    ):
        """Constructor.

        Args:
            config: MedusaConfig that specifies the medusa head configuration as well as
                    weights of base model and medusa head.
        """
        if self.config.pipeline_model_parallel_size > 1:
            warnings.warn(
                "Pipeline parallelism detected! _DynamicMedusaGPTModel only supports "
                "pipeline parallelism during TensorRT-LLM checkpoint export."
            )
        super().modify(medusa_num_heads=medusa_num_heads, medusa_num_layers=medusa_num_layers)

        self.medusa_report_acc = medusa_report_acc
        self.medusa_freeze_base_model = medusa_freeze_base_model

        # Freeze all parameters
        if self.medusa_freeze_base_model:
            for name, param in self.named_parameters():
                param.requires_grad = False

        if self.post_process:
            self.medusa_heads = torch.nn.ModuleList(
                [
                    MedusaHead(self.config, self.vocab_size, num_layers=self.medusa_num_layers)
                    for _ in range(self.medusa_num_heads)
                ]
            )

    def _base_model_forward(self, *args, labels: torch.Tensor = None, **kwargs):
        if self.post_process:
            # Set the post_process to False such that the forward will return the hidden_state.
            self.post_process = False
            # Calling parent's forward to get hidden_states
            hidden_states = GPTModel.forward(self, *args, labels=labels, **kwargs)
            # Reset the post_process to True
            self.post_process = True
        else:
            hidden_states = GPTModel.forward(self, *args, labels=None, **kwargs)

        return hidden_states

    def _medusa_forward(self, hidden_states):
        draft_logits = []
        # Medusa heads forward. We want to run through all the heads just to make sure all modules
        # are exercised during calibration.
        for i, head in enumerate(self.medusa_heads):
            new_logits, _ = head(hidden_states)

            draft_logits.append(new_logits)

        return draft_logits

    def forward(self, *args, labels: torch.Tensor = None, **kwargs):
        """Forward pass of the Medusa GPTModel.

        Returns:
            torch.Tensor: If labels are provided, then return lm_loss of all heads. Otherwise,
                return the original logits.
        """
        hidden_states = self._base_model_forward(*args, labels=labels, **kwargs)

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        # Original output logits
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        draft_logits = self._medusa_forward(hidden_states)

        if self.medusa_report_acc and labels is not None:
            acc = []
            for i, _ in enumerate(self.medusa_heads):
                gathered_logits = gather_from_tensor_model_parallel_region(draft_logits[i])
                medusa_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)[:, : -(1 + i)]
                medusa_labels = labels[:, 1 + i :]
                top1_p = torch.eq(medusa_labels, medusa_top1).sum() / medusa_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print(f"Medusa Training Accuracy: {acc}")

        # Return the original logits untouched.
        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        # Base model loss
        # If medusa_freeze_base_model is set to True,
        # the base model is frozen .
        loss = self.compute_language_model_loss(labels, logits)
        # Medusa loss
        for i, _ in enumerate(self.medusa_heads):
            medusa_labels = labels[:, 1 + i :]
            medusa_loss = self.compute_language_model_loss(
                medusa_labels, draft_logits[i][: -(1 + i), :]
            )
            loss[:, 1 + i :] += medusa_loss

        return loss

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Override the shared_state_dict to take care medusa_heads."""
        assert not sharded_offsets, "Unexpected sharded offsets"

        sharded_state_dict = GPTModel.sharded_state_dict(self, prefix, sharded_offsets, metadata)

        if not hasattr(self, "medusa_heads") or self.medusa_heads is None:
            return sharded_state_dict

        # This is a remedy for nn.ModuleList. GPTModel.sharded_state_dict() is calling into
        # MegatronModule.sharded_state_dict() which requires all children to implement
        # sharded_state_dict(). medusa_heads is an nn.ModuleList which only has state_dict()
        # implemented. As a result, all the submodules will not be sharded.
        #
        # The remedy is to pop all medusa_heads* out and call the MedusaHead sharded_state_dict()
        # again to populate the correct sharded_staet_dict.
        extra_keys = []
        for key in sharded_state_dict:
            if "medusa_heads" in key:
                extra_keys += [key]
        for key in extra_keys:
            sharded_state_dict.pop(key, None)

        layer_prefix = f"{prefix}medusa_heads."
        for i, layer in enumerate(self.medusa_heads):
            layer_sharded_state_dict = layer.sharded_state_dict(f"{layer_prefix}{i}.", [], metadata)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict

    def pseudo_speculative_generate(self, *args, steps=1, **kwargs):
        """Pseudo generate of the Medusa GPTModel.

        Returns:
            base_token (torch.Tensor): token from base model
            draft_tokens (torch.Tensor): draft tokens from medusa heads
        """
        hidden_states = self._base_model_forward(*args, labels=None, **kwargs)

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        # Original output logits
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        draft_logits = self._medusa_forward(hidden_states)

        logits = gather_from_tensor_model_parallel_region(logits.transpose(0, 1).contiguous())
        draft_logits = [
            gather_from_tensor_model_parallel_region(logit.transpose(0, 1).contiguous())
            for logit in draft_logits
        ]

        # [b, s]
        base_token = logits[:, -1:].argmax(dim=-1)
        draft_tokens = [logit[:, -1:].argmax(dim=-1) for logit in draft_logits]
        draft_tokens = torch.cat(draft_tokens, dim=-1)

        return base_token, draft_tokens
