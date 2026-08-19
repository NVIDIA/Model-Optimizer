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

"""HF DFlash2 model wrapper — DFlash training plus the candidate-selector objective.

DFlash2 differs from DFlash only in the draft module (grouped dynamic convolutions
around every sublayer, plus a candidate selector) and in one extra loss term, so
this wrapper reuses ``HFDFlashModel``'s forward wholesale and overrides just
:meth:`_compute_loss`.

The convolutions need no supervision of their own: they sit inside the backbone
and are trained by the backbone loss. The selector does, because at serving time
it — not an independent argmax — picks the drafted token at each block position.

Selector supervision (following the SGLang/SpecForge reference):

- Take the backbone's top-k candidates per block position.
- Score each candidate against its *teacher-forced* predecessor token, so the
  positions train in parallel exactly as the backbone does.
- When the gold token is missing from the top-k, substitute it into the last
  candidate slot. Without this the selector sees no positive class on the hard
  positions and never learns those edges.
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..dflash.conversion import DFlash2DMRegistry
from .hf_dflash import HFDFlashModel
from .modeling_dflash2 import DFlash2Module

__all__ = ["HFDFlash2Model"]


@DFlash2DMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDFlash2Model(HFDFlashModel):
    """DFlash model with DFlash2's sublayer convolutions and candidate selector.

    Registered in ``DFlash2DMRegistry`` so that ``convert_to_dflash_model`` routes
    to it when ``dflash_architecture_config.projector_type == "dflash2"``.
    """

    def _build_draft_module(self, dflash_config):
        """Build the DFlash2 draft module (DFlash backbone + convolutions + selector)."""
        return DFlash2Module(dflash_config)

    def modify(self, config):
        """Initialize the DFlash2 draft module and read the selector loss weight."""
        arch_config = config.dflash_architecture_config
        missing = [
            name
            for name in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k")
            if arch_config.get(name) is None
        ]
        if missing:
            raise ValueError(
                f"DFlash2 (projector_type='dflash2') requires {missing} in "
                "dflash_architecture_config (convolution taps/group size and the "
                "candidate selector's rank/top-k)."
            )
        super().modify(config)
        self.dflash_selector_loss_alpha = getattr(config, "dflash_selector_loss_alpha", 1.0)

    def get_exporter(self):
        """Get the exporter for the DFlash2 draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DFlash2Exporter

        return DFlash2Exporter(self)

    def _selector_loss(self, logits, target_ids, hidden, predecessor_ids, weight_mask):
        """Cross-entropy over the selector's candidate set, and its top-1 accuracy.

        Args:
            logits: Backbone logits per block position ``[B, N, block_size, V]``.
            target_ids: Gold token ids ``[B, N, block_size]``.
            hidden: Backbone hidden states ``[B, N, block_size, H]``.
            predecessor_ids: Teacher-forced predecessor ids ``[B, N, block_size]``.
            weight_mask: Per-position loss weights ``[B, N, block_size]``.

        Returns:
            ``(loss, accuracy, coverage)`` — coverage is the fraction of supervised
            positions whose gold token was already in the backbone's top-k, i.e. how
            often the selector is choosing rather than being handed the answer.
        """
        selector = self.dflash_module.candidate_selector
        top_k = selector.top_k

        unary_logits, candidate_ids = logits.topk(top_k, dim=-1)

        # Where the gold token is absent from the top-k, overwrite the last (lowest
        # scoring) slot with it, so every supervised position has a correct class.
        gold_in_topk = (candidate_ids == target_ids.unsqueeze(-1)).any(dim=-1)
        gold_slot = torch.where(
            gold_in_topk,
            (candidate_ids == target_ids.unsqueeze(-1)).float().argmax(dim=-1),
            torch.full_like(target_ids, top_k - 1),
        )
        gold_unary = logits.gather(-1, target_ids.unsqueeze(-1))
        candidate_ids = candidate_ids.scatter(-1, gold_slot.unsqueeze(-1), target_ids.unsqueeze(-1))
        unary_logits = unary_logits.scatter(-1, gold_slot.unsqueeze(-1), gold_unary)

        selector_logits = selector.score_candidates(
            candidate_ids, unary_logits, hidden, predecessor_ids
        )

        flat_weights = weight_mask.reshape(-1)
        denominator = flat_weights.sum() + 1e-6
        per_token = F.cross_entropy(
            selector_logits.float().reshape(-1, top_k),
            gold_slot.reshape(-1),
            reduction="none",
        )
        loss = (per_token * flat_weights).sum() / denominator

        with torch.no_grad():
            chosen = selector_logits.argmax(dim=-1).reshape(-1)
            accuracy = (
                (chosen == gold_slot.reshape(-1)).float() * flat_weights
            ).sum() / denominator
            coverage = (gold_in_topk.reshape(-1).float() * flat_weights).sum() / denominator
        return loss, accuracy.item(), coverage.item()

    def _compute_loss(
        self,
        logits,
        input_ids,
        anchor_positions,
        block_keep_mask,
        loss_mask,
        base_logits=None,
        draft_hidden=None,
    ):
        """Backbone DFlash loss plus the candidate-selector cross-entropy.

        Reuses ``HFDFlashModel._compute_loss`` for the backbone term, then rebuilds
        the same target/weight alignment for the selector term. Reported accuracy
        stays the backbone's top-1, so DFlash and DFlash2 runs remain comparable;
        the selector's own accuracy is logged separately.
        """
        loss, accuracy = super()._compute_loss(
            logits, input_ids, anchor_positions, block_keep_mask, loss_mask, base_logits
        )
        if self.dflash_selector_loss_alpha <= 0 or draft_hidden is None:
            return loss, accuracy

        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device

        offsets = torch.arange(block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + offsets
        valid_label = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        expanded_ids = input_ids.unsqueeze(1).expand(-1, n_blocks, -1)
        target_ids = torch.gather(expanded_ids, 2, safe_label_indices)

        # Same supervision mask as the backbone loss: valid block, in bounds, not the
        # anchor slot, and inside the answer span. Position weighting (decay/D-PACE) is
        # deliberately not applied — it shapes *where* the backbone spends capacity,
        # while the selector should learn every position's transition equally.
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, block_size).float()
        weight_mask = weight_mask * valid_label.float()
        weight_mask = weight_mask * (offsets > 0).float()
        weight_mask = weight_mask * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )

        # Teacher-forced predecessor of block position k is the real token at anchor+k-1;
        # position 0's predecessor is the anchor itself, matching the serving-side walk
        # which starts from the last verified token.
        predecessor_ids = torch.gather(expanded_ids, 2, (safe_label_indices - 1).clamp(min=0))

        selector_loss, selector_accuracy, selector_coverage = self._selector_loss(
            logits.reshape(bsz, n_blocks, block_size, -1),
            target_ids,
            draft_hidden.reshape(bsz, n_blocks, block_size, -1),
            predecessor_ids,
            weight_mask,
        )
        self._selector_metrics = {
            "selector_accuracy": selector_accuracy,
            "selector_coverage": selector_coverage,
        }
        return loss + self.dflash_selector_loss_alpha * selector_loss, accuracy
