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
:meth:`_build_objective` and :meth:`_compute_loss`.

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
from .dflash_losses import LossOutput, _block_tokens, objective_term
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
        self.dflash_selector_loss_alpha = config.dflash_selector_loss_alpha
        if config.dflash_lk_loss_type != "ce" and self.dflash_self_logit_distillation:
            raise ValueError(
                f"dflash_lk_loss_type={config.dflash_lk_loss_type!r} needs the draft's "
                "probability of the gold token, which the KD path never forms -- it "
                "optimizes a soft target instead. Set dflash_self_logit_distillation=false, "
                "or dflash_lk_loss_type='ce' to keep distillation."
            )

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

        # Train on the strict top-k, the candidate set serving actually builds. A gold
        # token the backbone did not propose is a backbone recall failure, not a
        # selector classification example: substituting it in would teach the selector
        # to override the unary ranking on a set it will never be shown. Those
        # positions carry no selector gradient and leave the denominator instead.
        gold_matches = candidate_ids == target_ids.unsqueeze(-1)
        gold_in_topk = gold_matches.any(dim=-1)
        gold_slot = gold_matches.long().argmax(dim=-1)

        selector_logits = selector.score_candidates(
            candidate_ids, unary_logits, hidden, predecessor_ids
        )

        covered = weight_mask * gold_in_topk.to(weight_mask.dtype)
        flat_weights = covered.reshape(-1)
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
            # Coverage keeps the full supervised mask as its denominator: it measures
            # how often the selector was given a solvable problem at all.
            # clamp, not an epsilon: a fully covered batch must read exactly 1.0.
            supervised = weight_mask.reshape(-1).sum()
            coverage = flat_weights.sum() / supervised.clamp(min=1.0)
        # Detached tensors, not Python scalars: .item() would force a CPU-GPU sync on
        # every training step. The trainer converts them at the logging boundary.
        return loss, accuracy.detach(), coverage.detach()

    def _build_objective(self, config):
        """DFlash's objective, or the label TV / CE-TV blend ``dflash_lk_loss_type`` selects."""
        if config.dflash_lk_loss_type == "tv":
            return [objective_term("tv", 1.0)]
        if config.dflash_lk_loss_type == "lambda":
            return [
                objective_term(
                    "lambda", 1.0, scale=config.dflash_lk_ce_scale, decay=config.dflash_lk_ce_decay
                )
            ]
        return super()._build_objective(config)

    def _compute_loss(self, ctx):
        """Backbone DFlash loss plus the candidate-selector cross-entropy.

        Reported accuracy stays the backbone's top-1, so DFlash and DFlash2 runs remain
        comparable; the selector's own accuracy and coverage are added to the metrics.
        """
        out = super()._compute_loss(ctx)
        if self.dflash_selector_loss_alpha <= 0:
            return out

        batch, targets = ctx.batch, ctx.targets
        bsz, n_blocks = batch.anchors.shape
        block_size = self.dflash_block_size

        # Teacher-forced predecessor of block position k is the real token at anchor+k-1.
        # Only offsets 1..block_size-1 are supervised (the mask zeroes slot 0), so the
        # first supervised position, k=1, has the anchor's own token as its predecessor.
        # Slot 0's entry here resolves to anchor-1 and never contributes.
        #
        # This is the train/serve contract: CandidateSelector.greedy_path seeds its walk
        # from the anchor token, so ITS position 0 corresponds to block offset 1. A caller
        # that hands greedy_path the full 0..block_size-1 candidate set is off by one.
        predecessor_ids = _block_tokens(
            batch.input_ids, batch.anchors, block_size, self._label_shift - 1
        )

        # The backbone's supervision mask, without its position weighting (decay/D-PACE):
        # that shapes *where* the backbone spends capacity, while the selector should learn
        # every position's transition equally.
        selector_loss, selector_accuracy, selector_coverage = self._selector_loss(
            ctx.logits.backbone,
            targets.token_ids,
            batch.hidden.reshape(bsz, n_blocks, block_size, -1),
            predecessor_ids,
            targets.mask,
        )
        metrics = {
            **out.metrics,
            "selector_accuracy": selector_accuracy,
            "selector_coverage": selector_coverage,
        }
        return LossOutput(
            out.loss + self.dflash_selector_loss_alpha * selector_loss, out.accuracy, metrics
        )
