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
:meth:`_build_objective`, :meth:`_head_request` and :meth:`_compute_loss`.

The convolutions need no supervision of their own: they sit inside the backbone
and are trained by the backbone loss. The selector does, because at serving time
it — not an independent argmax — picks the drafted token at each block position.

Selector supervision (following the SGLang/SpecForge reference):

- Take the backbone's top-k candidates per block position.
- Score each candidate against its *teacher-forced* predecessor token, so the
  positions train in parallel exactly as the backbone does.
- Supervise only the strict top-k, the candidate set serving builds. A gold token
  outside it is a backbone recall miss, not a selector example, and carries no
  selector gradient.
- Weight each position as the backbone loss does (decay or D-PACE) and divide by the
  backbone's weight sum, misses included.
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..dflash.conversion import DFlash2DMRegistry
from .dflash_losses import LossContext, LossOutput, _block_tokens, objective_term
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

    def _head_request(self):
        """The backbone loss's statistics, and the top-k candidates the selector scores."""
        request = super()._head_request()
        if self.dflash_selector_loss_alpha <= 0:
            return request
        return request.extend(top_k=self.dflash_module.candidate_selector.top_k)

    def _selector_loss(
        self,
        unary_logits,
        candidate_ids,
        target_ids,
        hidden,
        predecessor_ids,
        supervised_mask,
        loss_weights,
        weight_sum,
    ):
        """Cross-entropy over the selector's candidate set, and its top-1 accuracy.

        Args:
            unary_logits: The backbone's logits of its top-k candidates per block position
                ``[B, N, block_size, k]``, differentiable.
            candidate_ids: Those candidates' ids, in descending logit order, same shape.
            target_ids: Gold token ids ``[B, N, block_size]``.
            hidden: Backbone hidden states ``[B, N, block_size, H]``.
            predecessor_ids: Teacher-forced predecessor ids ``[B, N, block_size]``.
            supervised_mask: The positions the backbone loss supervises, unweighted
                ``[B, N, block_size]``.
            loss_weights: The backbone loss's per-position weights ``[B, N, block_size]``.
            weight_sum: The backbone loss's denominator, the sum of ``loss_weights``.

        Returns:
            ``(loss, accuracy, coverage)`` — coverage is the fraction of supervised
            positions whose gold token was already in the backbone's top-k, i.e. how
            often the selector is choosing rather than being handed the answer.
        """
        selector = self.dflash_module.candidate_selector
        top_k = selector.top_k

        # Train on the strict top-k, the candidate set serving actually builds. A gold
        # token the backbone did not propose is a backbone recall failure, not a
        # selector classification example: substituting it in would teach the selector
        # to override the unary ranking on a set it will never be shown. Those
        # positions carry no selector gradient.
        gold_matches = candidate_ids == target_ids.unsqueeze(-1)
        gold_in_topk = gold_matches.any(dim=-1)
        gold_slot = gold_matches.long().argmax(dim=-1)

        selector_logits = selector.score_candidates(
            candidate_ids, unary_logits, hidden, predecessor_ids
        )

        in_topk = gold_in_topk.to(loss_weights.dtype)
        per_token = F.cross_entropy(
            selector_logits.float().reshape(-1, top_k),
            gold_slot.reshape(-1),
            reduction="none",
        )
        # The backbone's denominator, misses included: dividing by the covered positions
        # alone would scale the term up whenever backbone recall drops.
        loss = (per_token * (loss_weights * in_topk).reshape(-1)).sum() / weight_sum

        with torch.no_grad():
            covered = (supervised_mask * in_topk).reshape(-1)
            chosen = selector_logits.argmax(dim=-1).reshape(-1)
            accuracy = ((chosen == gold_slot.reshape(-1)).float() * covered).sum() / (
                covered.sum() + 1e-6
            )
            # Coverage keeps the full supervised mask as its denominator: it measures
            # how often the selector was given a solvable problem at all.
            # clamp, not an epsilon: a fully covered batch must read exactly 1.0.
            supervised = supervised_mask.reshape(-1).sum()
            coverage = covered.sum() / supervised.clamp(min=1.0)
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

    def _compute_loss(self, ctx: LossContext) -> LossOutput:
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
        # Only offsets 1..block_size-1 are supervised (the backbone's mask zeroes slot 0), so the
        # first supervised position, k=1, has the anchor's own token as its predecessor.
        # Slot 0's entry here resolves to anchor-1 and never contributes.
        #
        # This is the train/serve contract: CandidateSelector.greedy_path seeds its walk
        # from the anchor token, so ITS position 0 corresponds to block offset 1. A caller
        # that hands greedy_path the full 0..block_size-1 candidate set is off by one.
        predecessor_ids = _block_tokens(
            batch.input_ids, batch.anchors, block_size, self._label_shift - 1
        )

        # The backbone's own mask and position weights: a block position's choice only
        # matters if every earlier one was accepted, which decay / D-PACE model.
        unary_logits, candidate_ids = ctx.topk(
            "backbone", self.dflash_module.candidate_selector.top_k
        )
        selector_loss, selector_accuracy, selector_coverage = self._selector_loss(
            unary_logits,
            candidate_ids,
            targets.token_ids,
            batch.hidden.reshape(bsz, n_blocks, block_size, -1),
            predecessor_ids,
            targets.mask,
            ctx.weights,
            ctx.weight_sum,
        )
        metrics = {
            **out.metrics,
            "selector_accuracy": selector_accuracy,
            "selector_coverage": selector_coverage,
        }
        return LossOutput(
            out.loss + self.dflash_selector_loss_alpha * selector_loss, out.accuracy, metrics
        )
