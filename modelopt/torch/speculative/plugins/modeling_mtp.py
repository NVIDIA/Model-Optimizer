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

"""Native MTP modules trained through the shared Hugging Face EAGLE forward."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Cache, PretrainedConfig, PreTrainedModel

from ..config import EagleConfig, MTPConfig
from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel

__all__ = ["MTPModule", "MTPModuleOutput", "create_mtp_model"]


class MTPModuleOutput(NamedTuple):
    """EAGLE round output: LM-head input, recurrent state, and optional cache."""

    hidden_states: torch.Tensor
    raw_hidden_states: torch.Tensor
    past_key_values: Cache | None = None


class MTPModule(nn.Module, ABC):
    """Native decoder contract; EAGLE owns round alignment, masks, HSM, and losses."""

    def __init__(self, config: PretrainedConfig, *, state_shape: tuple[int, ...] | None = None):
        """Describe the native state's trailing dimensions; default to one hidden vector."""
        super().__init__()
        self.config = config
        self.state_shape = (config.hidden_size,) if state_shape is None else state_shape

    def _maybe_init_rope(self, device):
        """Adapters may initialize rotary buffers here, before the shared round loop."""

    def prepare_initial_hidden_states(self, base_outputs):
        """Flatten native recurrent state to EAGLE's [batch, sequence, hidden] layout."""
        state = base_outputs.draft_hiddens
        if state is None or state.ndim < 3 or state.shape[:2] != base_outputs.out_hiddens.shape[:2]:
            raise ValueError("draft_hidden_states must match the teacher batch and sequence")
        if state.shape[2:] != self.state_shape:
            raise ValueError(f"Expected native MTP state dimensions {self.state_shape}")
        return state.detach().flatten(2)

    def forward(
        self,
        hidden_states,
        inputs_embeds,
        attention_mask,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        input_ids=None,
    ) -> MTPModuleOutput:
        """Run a native round, recomputing long cache-free rounds during backward."""
        args = (
            hidden_states.unflatten(-1, self.state_shape),
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            input_ids,
        )
        # Recomputing a round that mutates a KV cache would append its keys twice.
        if (
            self.training
            and torch.is_grad_enabled()
            and hidden_states.shape[1] > 256
            and past_key_values is None
        ):
            output = checkpoint(self._forward, *args, use_reentrant=False)
        else:
            output = self._forward(*args)
        return MTPModuleOutput(
            output.hidden_states, output.raw_hidden_states.flatten(2), output.past_key_values
        )

    @abstractmethod
    def _forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        use_cache: bool,
        input_ids: torch.Tensor | None,
    ) -> MTPModuleOutput:
        """Consume EAGLE-aligned inputs in the native [batch, sequence, *state_shape] layout.

        Adapters handle native positions/state layouts and honor the supplied attention
        mask and cache. With HSM, the cache is None; otherwise it accumulates across rounds.
        """
        raise NotImplementedError


def create_mtp_model(
    mtp: MTPModule,
    embedding: nn.Embedding,
    lm_head: nn.Linear,
    training_config: MTPConfig | None = None,
) -> PreTrainedModel:
    """Attach loaded native modules to EAGLE's existing offline training implementation.

    Offline batches use ``base_model_outputs`` with ``draft_hidden_states`` (native
    state) and ``base_model_hidden_states`` (teacher LM-head input). Only the native
    module trains; adapters reconstruct frozen endpoints before HF Trainer resume.
    """
    cfg = MTPConfig() if training_config is None else training_config.model_copy(deep=True)
    model = PreTrainedModel(
        PretrainedConfig(
            mtp_config=mtp.config.to_dict(),
            mtp_boost=cfg.model_dump(),
            num_hidden_layers=0,
        )
    )
    model = EagleDMRegistry.convert(model)
    EagleModel.modify(
        model,
        EagleConfig(
            eagle_offline=True,
            eagle_ttt_steps=cfg.mtp_ttt_steps,
            eagle_mix_hidden_states=cfg.mtp_mix_hidden_states,
            eagle_mix_hidden_states_mode=cfg.mtp_mix_hidden_states_mode,
            eagle_loss_type=cfg.mtp_loss_type,
            eagle_lk_eta=cfg.mtp_lk_eta,
            eagle_loss_decay_factor=cfg.mtp_loss_decay_factor,
            eagle_use_torch_compile=False,
            eagle_export_rope_scaling={},
        ),
    )
    model.eagle_module = mtp
    model.eagle_config = PretrainedConfig(
        use_aux_hidden_state=False,
        vocab_size=lm_head.out_features,
        draft_vocab_size=lm_head.out_features,
    )
    model.eagle_config._attn_implementation = "sdpa"
    model.embedding = embedding.requires_grad_(False)
    model.lm_head = lm_head.requires_grad_(False)
    model.base_model_embeddings_path = "embedding"
    model.base_model_lm_head_path = "lm_head"
    model._cached_attn_blk_masks = {}
    model._cached_attn_blk_mask_context = None
    model.mtp_config = cfg
    model.is_native_mtp_boost = True
    # HF owns serialization; the adapter reconstructs the frozen endpoints.
    model._keys_to_ignore_on_save = [
        f"{name}.{key}"
        for name, module in (("embedding", embedding), ("lm_head", lm_head))
        for key in module.state_dict()
    ]
    return model
