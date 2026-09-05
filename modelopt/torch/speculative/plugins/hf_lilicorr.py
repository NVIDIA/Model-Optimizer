# Adapted from https://github.com/sgl-project/SpecForge
# Copyright (c) 2025 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""LiLiCorr speculative decoding plugin for HuggingFace models.

LiLiCorr (NVIDIA, "LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts
for Speculative Decoding", https://arxiv.org/abs/2608.20530) reuses the DFlash draft
backbone and adds a reranker over the candidate lattice the backbone already produces
(see ``modeling_lilicorr.LiLiCorrModule``).

The backbone is trained on per-position *marginals*, so the tokens it emits are
individually plausible yet jointly incoherent. Rather than committing the per-slot
argmax, LiLiCorr keeps the top-``k`` candidates per slot and scores transitions
between them, then commits a path greedily. This module trains that scorer jointly
with the backbone.

The objective adds three weighted terms to the DFlash loss::

    loss = dflash_loss + w_ce * L_ce + w_margin * L_margin + w_pen * L_pen

- ``L_ce``: a softmax over each slot's ``k`` candidate scores, pushing the
  ground-truth candidate up and the competing ones down.
- ``L_margin``: a hinge on the same scores, pushing the ground-truth candidate above
  the best competing one by ``margin``. Zero-gradient once it already wins by that
  much, so capacity goes to the near-ties.
- ``L_pen``: the head's own probability mass on the competing candidates, each
  weighted by the target model's logit gap to the ground truth — so a candidate the
  target finds plausible is penalized lightly and a confident wrong one hard.

The weights are absolute, with no outer multiplier, so
``loss == origin_loss + lilicorr_loss`` holds exactly. Two variants ship and differ
*only* in composition — ``base`` (w_ce 0.25, w_margin 0) and ``margin``
(0.125 / 0.125), both with ``w_pen`` 0.25 — so the second variant is two recipe
lines rather than a second implementation.
"""

import logging

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..dflash.conversion import LiLiCorrDMRegistry
from .hf_dflash import HFDFlashModel
from .modeling_lilicorr import LiLiCorrModule

logger = logging.getLogger(__name__)

__all__ = ["HFLiLiCorrModel"]

# The three absolute term weights, in the order the objective composes them. A
# negative value means "unset"; they are validated all-or-nothing so a
# half-specified config cannot silently inherit a composition.
_TERM_WEIGHT_FIELDS = (
    "dflash_lilicorr_w_ce",
    "dflash_lilicorr_w_margin",
    "dflash_lilicorr_w_pen",
)


@LiLiCorrDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFLiLiCorrModel(HFDFlashModel):
    """DFlash model with the LiLiCorr candidate-lattice reranker.

    Registered in ``LiLiCorrDMRegistry`` so that ``convert_to_dflash_model`` routes
    to it when ``dflash_architecture_config.projector_type == "lilicorr"``.
    """

    def _build_draft_module(self, dflash_config):
        """Build the LiLiCorr draft module (DFlash backbone + lattice reranker)."""
        return LiLiCorrModule(dflash_config)

    def modify(self, config):
        """Initialize the LiLiCorr draft module and resolve the objective weights."""
        if config.dflash_offline:
            raise ValueError(
                "LiLiCorr (projector_type='lilicorr') requires online training. Its "
                "distractor penalty weights every competing candidate by the target "
                "model's own logit gap to the ground truth, and offline mode has no "
                "target model to read that from. Use data.mode=online."
            )
        super().modify(config)

        weights = {name: float(getattr(config, name, -1.0)) for name in _TERM_WEIGHT_FIELDS}
        unset = [name for name, value in weights.items() if value < 0.0]
        if unset:
            raise ValueError(
                f"LiLiCorr objective weights are all-or-nothing: {sorted(unset)} are unset. "
                "Set every dflash_lilicorr_w_* explicitly so the objective's composition is "
                "readable from the config alone."
            )
        self.dflash_lilicorr_w_ce = weights["dflash_lilicorr_w_ce"]
        self.dflash_lilicorr_w_margin = weights["dflash_lilicorr_w_margin"]
        self.dflash_lilicorr_w_pen = weights["dflash_lilicorr_w_pen"]
        if not any(weights.values()):
            raise ValueError(
                "All three LiLiCorr objective weights are 0, so the reranker would never "
                "receive a gradient and would be exported randomly initialized. Set at "
                "least one of dflash_lilicorr_w_ce / _w_margin / _w_pen above 0."
            )

        self.dflash_lilicorr_margin = float(getattr(config, "dflash_lilicorr_margin", -1.0))
        if self.dflash_lilicorr_w_margin > 0.0 and self.dflash_lilicorr_margin <= 0.0:
            raise ValueError(
                "dflash_lilicorr_w_margin > 0 requires a positive dflash_lilicorr_margin "
                "(the hinge width); the shipped 'margin' variant uses 2.0."
            )

        head = self.dflash_module.lilicorr
        # Stated at startup because the composition is the variant: a run's own log has
        # to say which objective produced the checkpoint.
        logger.info(
            "LiLiCorr enabled: candidate_topk=%d, factor_dim=%d, num_layers=%d, "
            "num_heads=%d, logit_scale=%.3g, vector_eps=%.3g | objective: "
            "w_ce=%.4g, w_margin=%.4g, w_pen=%.4g, margin=%.4g. Total loss is "
            "dflash_loss + w_ce*CE + w_margin*hinge + w_pen*penalty, with no outer "
            "multiplier.",
            head.candidate_topk,
            head.factor_dim,
            len(head.layers),
            head.num_heads,
            head.logit_scale,
            head.vector_eps,
            self.dflash_lilicorr_w_ce,
            self.dflash_lilicorr_w_margin,
            self.dflash_lilicorr_w_pen,
            self.dflash_lilicorr_margin,
        )
        self._lilicorr_metrics = None

    def get_exporter(self):
        """Get the exporter for the LiLiCorr draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import LiLiCorrExporter

        return LiLiCorrExporter(self)

    def _block_targets(self, input_ids, anchor_positions, block_keep_mask, loss_mask):
        """Return ``(target_ids [B, N, bs], slot_mask [B, N, bs-1])`` for the lattice.

        ``target_ids[b, n, j]`` is the ground-truth **vocabulary id** at absolute position
        ``anchor[b, n] + j``, covering the whole block including position 0 (the anchor
        token, which the caller slices off). Distinct from ``gt_indices`` in
        :meth:`_compute_lilicorr_loss`, which is a position *within* a slot's k candidates.

        ``slot_mask[b, n, s]`` says whether predicted slot ``s`` — block position
        ``s + 1`` — is supervised: its block is kept, its position is in bounds, and
        ``loss_mask`` allows it.

        Recomputed rather than threaded out of :meth:`HFDFlashModel._compute_loss`: both
        are exact functions of the anchors and the masks, so recomputing them keeps the
        shared method's signature to the tensors a variant cannot derive on its own.
        The slot mask is the parent's *pre-weighting* eval mask, so the two losses
        supervise the same positions regardless of the block-position weighting.
        """
        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device

        label_offsets = torch.arange(0, block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        keep = block_keep_mask.unsqueeze(-1).expand(-1, -1, block_size).float()
        keep = keep * valid_label.float()
        keep = keep * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        # Block position 0 is the anchor — the one row whose input embedding is a real
        # verified token — so the lattice covers positions 1..bs-1.
        return target_ids, keep[:, :, 1:] > 0.5

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
        """Add the LiLiCorr lattice terms to the DFlash block loss.

        The returned accuracy is the reranker's greedy path-prefix fraction — the
        quantity that tracks acceptance length — rather than the backbone's per-token
        accuracy, which is reported as ``origin_accuracy`` in the metrics instead.

        The two tensors this needs beyond the shared signature -- the target-layer
        hidden states it anchors on, and the target's logits its distractor penalty
        weights by -- are read off the instance, where ``HFDFlashModel.forward``
        publishes them for the duration of this call. They are not parameters because
        this signature is shared with every other draft variant.
        """
        target_hidden = getattr(self, "_dflash_loss_target_hidden", None)
        target_logits = getattr(self, "_dflash_loss_target_logits", None)
        if draft_hidden is None or target_hidden is None:
            raise ValueError(
                "LiLiCorr requires draft_hidden and target_hidden in _compute_loss: the "
                "reranker reads the draft's per-slot hidden states and anchors on a "
                "committed target row."
            )

        origin_loss, origin_accuracy = super()._compute_loss(
            logits,
            input_ids,
            anchor_positions,
            block_keep_mask,
            loss_mask,
            base_logits,
        )
        target_ids, slot_mask = self._block_targets(
            input_ids, anchor_positions, block_keep_mask, loss_mask
        )
        lilicorr_loss, accuracy, metrics = self._compute_lilicorr_loss(
            logits=logits,
            target_ids=target_ids,
            slot_mask=slot_mask,
            anchor_positions=anchor_positions,
            draft_hidden=draft_hidden,
            target_hidden=target_hidden,
            target_logits=target_logits,
        )
        # The identity `loss == origin_loss + lilicorr_loss` is the cheap parity check
        # on this objective, so both halves are reported next to the total.
        metrics["origin_loss"] = float(origin_loss.detach())
        metrics["origin_accuracy"] = float(origin_accuracy)
        self._lilicorr_metrics = metrics
        return origin_loss + lilicorr_loss, accuracy

    def _compute_lilicorr_loss(
        self,
        *,
        logits,
        target_ids,
        slot_mask,
        anchor_positions,
        draft_hidden,
        target_hidden,
        target_logits,
    ):
        """Score the candidate lattice and evaluate the three-term objective.

        Returns ``(loss, accuracy, metrics)``. ``loss`` already carries the absolute
        term weights, ``accuracy`` is the greedy path-prefix fraction, and ``metrics``
        holds Python floats read back in a single device sync.
        """
        module = self.dflash_module
        head = module.lilicorr
        bsz, n_blocks, block_size = target_ids.shape
        num_slots = block_size - 1
        topk = head.candidate_topk
        vocab = logits.shape[-1]
        batch_blocks = bsz * n_blocks

        if topk > vocab:
            raise ValueError(f"lilicorr_candidate_topk={topk} exceeds the vocabulary size {vocab}.")

        logits_slots = logits.view(bsz, n_blocks, block_size, vocab)[:, :, 1:, :].reshape(
            batch_blocks, num_slots, vocab
        )
        ground_truth = target_ids[:, :, 1:]
        gt_flat = ground_truth.reshape(batch_blocks, num_slots)
        # A block is trained only when every one of its slots is a supervised position,
        # so the chain is never scored against a partially masked ground truth.
        valid_block = slot_mask.all(dim=-1).reshape(batch_blocks, 1)

        with torch.no_grad():
            # topk already returns candidates in descending log-probability order, which
            # is the rank order `rank_embedding` is indexed by. Taken on the logits
            # rather than on a log_softmax of them: the map is strictly monotone per row,
            # so the ids are identical, and it avoids a vocab-wide float32 buffer.
            _, candidate_ids = torch.topk(logits_slots.detach(), k=topk, dim=-1)
            gt_hits = candidate_ids == gt_flat.unsqueeze(-1)
            gt_in_slot = gt_hits.any(dim=-1)
            gt_indices = gt_hits.long().argmax(dim=-1)

        # Soft-value channel. The candidate *ids* are the hard top-k, so the forward
        # values are unchanged; recomputing their log-probs from the LIVE logits
        # re-attaches the autograd graph, which is how the lattice terms reach the
        # drafter body and not only the head. logsumexp rather than a full log_softmax
        # so backward needs no second vocab-sized buffer.
        candidate_logits = torch.gather(logits_slots, 2, candidate_ids).float()
        log_denominator = torch.logsumexp(logits_slots.float(), dim=-1, keepdim=True)
        candidate_log_probs = candidate_logits - log_denominator

        draft_slots = draft_hidden.view(bsz, n_blocks, block_size, -1)[:, :, 1:, :]
        start_scores, pair_scores = module.score_lattice(
            candidate_token_ids=candidate_ids.view(bsz, n_blocks, num_slots, topk),
            candidate_log_probs=candidate_log_probs.view(bsz, n_blocks, num_slots, topk),
            pass_hidden=draft_slots,
            embed_tokens=self._base_model_embeddings,
            target_hidden=target_hidden,
            anchor_positions=anchor_positions,
        )
        log_start, log_pair = head.compose_log_factors(
            start_scores=start_scores, pair_scores=pair_scores
        )
        log_start = log_start.reshape(batch_blocks, topk)
        log_pair = log_pair.reshape(batch_blocks, max(num_slots - 1, 0), topk, topk)

        # forced[:, i] is True iff the ground truth is in the lattice at slot i AND at
        # every preceding slot — i.e. slot i lies inside the maximal achievable correct
        # prefix. Blocks whose slot 0 already misses self-zero through the mask below.
        forced_prefix = gt_in_slot.long().cumprod(dim=-1).bool()

        # Walk the chain slot by slot. `node` is [batch_blocks, topk]: for every block,
        # the score of each of its k candidates at the current slot. At slot 0 those are
        # the start factors against the anchor; after that they are the transition scores
        # leaving whichever candidate held the ground truth at the previous slot.
        # Conditioning on the ground truth is what makes this the training objective; the
        # greedy decode in `_lattice_metrics` runs the same factors on its own picks.
        neg_inf = torch.finfo(log_start.dtype).min
        node_potentials = []
        ce_slots = []
        gap_slots = []
        node = log_start
        for slot in range(num_slots):
            if slot > 0:
                # log_pair[:, s-1] is [batch_blocks, topk, topk]; entry [b, i, j] scores
                # candidate i at slot s-1 -> candidate j at slot s. `previous` is
                # [batch_blocks]: per block, which candidate held the ground truth at the
                # previous slot, which is the one row worth keeping. With batch_blocks 2,
                # topk 3 and previous [2, 1], block 0 keeps log_pair[0, 2, :] and block 1
                # keeps log_pair[1, 1, :].
                previous = gt_indices[:, slot - 1]
                node = torch.gather(
                    log_pair[:, slot - 1, :, :],
                    1,
                    previous.view(-1, 1, 1).expand(-1, 1, topk),
                ).squeeze(1)
            # gt_indices is [batch_blocks, num_slots]: for every block AND every slot,
            # which of the k candidates is the ground-truth token. This picks out the
            # current slot's column — that one slot, in every block — so gt_column is
            # [batch_blocks, 1]. With batch_blocks 3, if this slot's target is candidate 2
            # in the first block, 0 in the second and 1 in the third, gt_column is
            # [[2], [0], [1]] and selects node[0, 2], node[1, 0], node[2, 1] below.
            gt_column = gt_indices[:, slot : slot + 1]
            node_potentials.append(node)
            ce_slots.append(-torch.gather(F.log_softmax(node, dim=-1), 1, gt_column).squeeze(-1))
            # gap = z_gt - max_{k != gt} z_k, so gap > 0 iff the ground truth is the
            # per-slot argmax. Reused by the hinge (with gradient) and the metrics.
            # The scatter returns a *copy* with the ground truth's own score buried at
            # -inf, so the max over it is the best competing candidate; `node` stays
            # intact for the CE above and the penalty below.
            z_gt = torch.gather(node, 1, gt_column).squeeze(-1)
            z_runner_up = node.scatter(1, gt_column, neg_inf).max(dim=-1).values
            gap_slots.append(z_gt - z_runner_up)

        cross_entropy = torch.stack(ce_slots, dim=-1)
        gap = torch.stack(gap_slots, dim=-1)
        supervised = forced_prefix.to(dtype=cross_entropy.dtype) * valid_block.to(
            dtype=cross_entropy.dtype
        )
        denominator = supervised.sum().clamp_min(1.0)

        ce_loss = (cross_entropy * supervised).sum() / denominator
        loss = self.dflash_lilicorr_w_ce * ce_loss

        margin_loss = ce_loss.new_zeros(())
        if self.dflash_lilicorr_w_margin > 0.0:
            hinge = torch.relu(self.dflash_lilicorr_margin - gap)
            margin_loss = (hinge * supervised).sum() / denominator
            loss = loss + self.dflash_lilicorr_w_margin * margin_loss

        penalty_loss = ce_loss.new_zeros(())
        if self.dflash_lilicorr_w_pen > 0.0:
            penalty_loss = self._distractor_penalty(
                node_potentials=node_potentials,
                candidate_ids=candidate_ids,
                gt_indices=gt_indices,
                anchor_positions=anchor_positions,
                target_logits=target_logits,
                supervised=supervised,
                denominator=denominator,
            )
            loss = loss + self.dflash_lilicorr_w_pen * penalty_loss

        raw_metrics = self._lattice_metrics(
            log_start=log_start,
            log_pair=log_pair,
            candidate_ids=candidate_ids,
            ground_truth=ground_truth,
            gt_indices=gt_indices,
            gt_in_slot=gt_in_slot,
            forced_prefix=forced_prefix,
            valid_block=valid_block,
            slot_mask=slot_mask,
            start_scores=start_scores,
            pair_scores=pair_scores,
            gap=gap,
        )
        raw_metrics["lilicorr_loss"] = loss
        raw_metrics["lilicorr_ce"] = ce_loss
        raw_metrics["lilicorr_margin"] = margin_loss
        raw_metrics["lilicorr_penalty"] = penalty_loss

        # One device sync for every scalar, rather than one per metric.
        names = list(raw_metrics)
        stacked = torch.stack([raw_metrics[name].detach().reshape(()).float() for name in names])
        metrics = dict(zip(names, stacked.tolist(), strict=True))
        # Weights recorded alongside the terms so a run states its own composition
        # without the reader expanding the config.
        metrics["lilicorr_w_ce"] = self.dflash_lilicorr_w_ce
        metrics["lilicorr_w_margin"] = self.dflash_lilicorr_w_margin
        metrics["lilicorr_w_pen"] = self.dflash_lilicorr_w_pen
        accuracy = metrics["lilicorr_selected_prefix"] / float(num_slots)
        return loss, accuracy, metrics

    def _distractor_penalty(
        self,
        *,
        node_potentials,
        candidate_ids,
        gt_indices,
        anchor_positions,
        target_logits,
        supervised,
        denominator,
    ):
        """Expected target-rejection of the reranker's own candidate distribution.

        ``w_j = relu(logit_target(gt) - logit_target(cand_j))`` — the full-vocabulary
        log-normalizer cancels in the difference — so ``w`` is large for candidates the
        target rejects, ~0 for ones it finds plausible, and exactly 0 at the ground
        truth. The penalty is ``E_{p_head}[w]``, whose gradient
        ``p(j) * (w_j - E[w])`` pushes the ground truth up, the target-rejected
        confuser down, and defers on genuine ties.
        """
        if target_logits is None:
            raise ValueError(
                "dflash_lilicorr_w_pen > 0 requires the target model's logits, which are "
                "only available in online training (dflash_offline=False)."
            )
        batch_blocks, num_slots, topk = candidate_ids.shape
        bsz, n_blocks = anchor_positions.shape
        device = candidate_ids.device

        # p_head(j | slot): the same per-slot distribution the CE term scores, reused here
        # as the weights of the expectation.
        potentials = torch.stack(node_potentials, dim=1)
        head_probs = F.log_softmax(potentials.float(), dim=-1).exp()

        target_seq_len = target_logits.shape[1]
        # The target's next-token logits that predict the token at anchor+1+s sit at
        # position anchor+s. `sample_index` maps each flattened block back to its batch
        # row, so (sample, position) addresses one target logit vector per slot.
        sample_index = torch.arange(bsz, device=device).repeat_interleave(n_blocks)
        slot_offsets = torch.arange(num_slots, device=device).view(1, -1)
        positions = (anchor_positions.reshape(batch_blocks, 1) + slot_offsets).clamp(
            min=0, max=target_seq_len - 1
        )
        sample_expanded = sample_index.view(batch_blocks, 1, 1).expand(
            batch_blocks, num_slots, topk
        )
        positions_expanded = positions.view(batch_blocks, num_slots, 1).expand(
            batch_blocks, num_slots, topk
        )
        # Advanced indexing reads only the [blocks, slots, k] scalars in play rather
        # than gathering whole vocabulary rows. Teacher weights, hence detached.
        candidate_target_logits = (
            target_logits[sample_expanded, positions_expanded, candidate_ids].detach().float()
        )
        # w_j is 0 at the ground truth itself, and 0 for any candidate the target scores
        # at least as highly — those are not distractors, they are alternatives.
        gt_target_logit = candidate_target_logits.gather(2, gt_indices.unsqueeze(-1))
        rejection_weight = (gt_target_logit - candidate_target_logits).clamp_min(0.0)

        # E_{p_head}[w] per slot, averaged over the supervised slots only.
        per_slot = (head_probs * rejection_weight).sum(dim=-1)
        return (per_slot * supervised).sum() / denominator

    @torch.no_grad()
    def _lattice_metrics(
        self,
        *,
        log_start,
        log_pair,
        candidate_ids,
        ground_truth,
        gt_indices,
        gt_in_slot,
        forced_prefix,
        valid_block,
        slot_mask,
        start_scores,
        pair_scores,
        gap,
    ):
        """Diagnostics for reading a LiLiCorr run, as 0-dim tensors.

        The headline pair is ``selected_prefix`` (what the greedy walk actually commits)
        against ``top1_prefix`` (what the backbone's own argmax would have committed);
        their difference is the reranker's contribution. ``oracle_prefix`` bounds both —
        it is the longest prefix present in the lattice at all — so ``oracle_gap``
        separates a ranking failure from a coverage failure.
        """
        bsz, n_blocks, num_slots = ground_truth.shape
        topk = candidate_ids.shape[-1]

        # Greedy autoregressive decode, exactly what inference does: argmax the same
        # per-slot node potentials, conditioning each step on the pick before it.
        current = log_start.argmax(dim=-1)
        selected = [current]
        for slot in range(1, num_slots):
            transition = torch.gather(
                log_pair[:, slot - 1, :, :], 1, current.view(-1, 1, 1).expand(-1, 1, topk)
            ).squeeze(1)
            current = transition.argmax(dim=-1)
            selected.append(current)
        selected_indices = torch.stack(selected, dim=-1)
        selected_ids = torch.gather(candidate_ids, 2, selected_indices.unsqueeze(-1)).squeeze(-1)

        def prefix_length(matches):
            """Length of the leading run of correct slots, per block."""
            return matches.long().cumprod(dim=-1).sum(dim=-1)

        def factor_mean(values):
            """Mean factor, or 0 when there are none.

            ``block_size == 2`` leaves a single slot and therefore no transitions, so
            the pair tensors are empty and a plain ``mean()`` would report NaN — which
            reads as a diverging run rather than as a degenerate lattice.
            """
            return values.float().mean() if values.numel() else values.new_zeros(())

        def factor_spread(values, dims):
            """Mean std of the factors across ``dims``, or 0 when it is undefined.

            Undefined both for an empty tensor (no transitions, as above) and at
            ``candidate_topk == 1``, where a per-row std over one candidate is NaN.
            Either way the factors cannot spread, which is what 0 records.
            """
            if values.numel() == 0 or values.shape[-1] < 2:
                return values.new_zeros(())
            return values.float().std(dim=dims).mean()

        selected_prefix = prefix_length(selected_ids.view(bsz, n_blocks, num_slots) == ground_truth)
        top1_prefix = prefix_length(
            candidate_ids[:, :, 0].view(bsz, n_blocks, num_slots) == ground_truth
        )
        oracle_prefix = prefix_length(gt_in_slot.view(bsz, n_blocks, num_slots))

        block_weight = valid_block.reshape(bsz, n_blocks).float()
        block_count = block_weight.sum().clamp_min(1.0)
        slot_weight = slot_mask.float()
        supervised_weight = forced_prefix.reshape(bsz, n_blocks, num_slots).float()
        supervised_count = supervised_weight.sum().clamp_min(1.0)

        def block_mean(values):
            return (values.float() * block_weight).sum() / block_count

        metrics = {
            "lilicorr_selected_prefix": block_mean(selected_prefix),
            "lilicorr_top1_prefix": block_mean(top1_prefix),
            "lilicorr_selected_minus_top1": block_mean(selected_prefix - top1_prefix),
            "lilicorr_oracle_prefix": block_mean(oracle_prefix),
            "lilicorr_oracle_gap": block_mean(oracle_prefix - selected_prefix),
            "lilicorr_exact": block_mean(selected_prefix == num_slots),
            # Blocks with no correct token even at slot 0: they contribute no loss.
            "lilicorr_zero_prefix_rate": block_mean(~forced_prefix[:, 0].reshape(bsz, n_blocks)),
            "lilicorr_slot_gt_in_lattice": (
                gt_in_slot.view(bsz, n_blocks, num_slots).float() * slot_weight
            ).sum()
            / slot_weight.sum().clamp_min(1.0),
            "lilicorr_slot_acc": (
                (selected_indices == gt_indices).reshape(bsz, n_blocks, num_slots).float()
                * supervised_weight
            ).sum()
            / supervised_count,
            "lilicorr_start_factor_mean": factor_mean(start_scores),
            "lilicorr_pair_factor_mean": factor_mean(pair_scores),
            # The direct collapse signal: when the chain degenerates to its uniform
            # ln(k) fixed point every factor is equal and the spread goes to 0. A
            # healthy chain keeps it well clear of 0.
            "lilicorr_logfactor_spread": (
                factor_spread(log_start, -1) + factor_spread(log_pair, (-2, -1))
            )
            / 2.0,
            "lilicorr_gap_mean": (gap * supervised_weight.reshape(gap.shape)).sum()
            / supervised_count,
            "lilicorr_gap_frac_hit": (
                (gap > 0).float() * supervised_weight.reshape(gap.shape)
            ).sum()
            / supervised_count,
        }
        return metrics

    def forward(self, *args, **kwargs):
        """Run the DFlash training forward and attach the LiLiCorr metrics.

        The whole variant is the objective, so the pipeline above it is inherited
        verbatim rather than reimplemented; this only carries the metrics that
        ``_compute_loss`` produced out to the caller, the way DSpark and Domino do
        from their own forwards.
        """
        self._lilicorr_metrics = None
        outputs = super().forward(*args, **kwargs)
        if self._lilicorr_metrics is not None:
            outputs["lilicorr_metrics"] = self._lilicorr_metrics
            self._lilicorr_metrics = None
        return outputs
