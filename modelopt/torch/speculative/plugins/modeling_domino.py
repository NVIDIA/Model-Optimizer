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

"""Domino draft module — DFlash backbone plus a lightweight causal correction head.

Domino extends the parallel DFlash draft backbone (``DFlashModule``) with a small
GRU-based correction head. The backbone produces *base* logits for a full draft
block in one parallel forward; the head then injects intra-block causal dependency
(which the parallel backbone lacks) by running a GRU over the block's previously
decoded tokens and adding a logit correction to the suffix positions.

The head consists of:
    - ``prefix_gru``: single-layer GRU over token embeddings of the block prefix,
      producing a causal state summarizing tokens seen so far in the block.
    - ``embed_proj``: MLP mapping ``[backbone_hidden ; gru_state]`` to a vocab-sized
      logit correction.

These two submodules live on ``DominoModule`` so they export under the
``dflash_module.`` prefix and serialize alongside the backbone (matching the
z-lab/SpecForge ``prefix_gru.*`` / ``embed_proj.*`` checkpoint layout).

The head is *applied* by the training wrapper (``HFDominoModel``), which owns the
base model's embedding table; this module only holds the parameters. See
``hf_domino.py`` for the forward/loss orchestration.
"""

from torch import nn

from .modeling_dflash import DFlashModule

__all__ = ["DominoModule"]


class DominoModule(DFlashModule):
    """DFlash draft module augmented with the Domino causal correction head."""

    def __init__(self, config):
        """Initialize the DFlash backbone, then add the GRU + projection head."""
        super().__init__(config)

        self.projector_type = getattr(config, "projector_type", "domino")
        self.gru_hidden_dim = config.gru_hidden_dim
        self.emb_dim = config.emb_dim
        # pure_draft_prefix_len positions at the block start keep base logits only
        # (no causal correction); the GRU correction applies to the suffix.
        self.pure_draft_prefix_len = getattr(config, "pure_draft_prefix_len", 1)
        self.shift_label = getattr(config, "shift_label", True)

        # Causal state over the block's token embeddings. bias=False matches the
        # reference checkpoint (only weight_ih_l0 / weight_hh_l0 are stored).
        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        # [backbone_hidden ; gru_state] -> emb_dim -> vocab logit correction.
        in_dim = config.hidden_size + self.gru_hidden_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(in_dim, self.emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.emb_dim, config.vocab_size, bias=False),
        )

        # DFlashModule.__init__ already ran _init_weights before these modules
        # existed, so initialize the new Linear layers explicitly. The GRU keeps
        # PyTorch's default (uniform) init.
        self._init_head_weights(config)

    def _init_head_weights(self, config):
        """Initialize the correction-head Linear layers (GRU keeps default init)."""
        std = getattr(config, "initializer_range", 0.02)
        for module in self.embed_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
