# Adapted from https://github.com/sgl-project/SpecForge/blob/2dadea1/specforge/algorithms/dspine/model.py
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

"""DSpine speculative decoding plugin for HuggingFace models.

DSpine reuses the DFlash pipeline (anchor sampling, noise blocks, KV injection) with causal
in-block attention and an adjacent causal injection after every draft layer (see
``modeling_dspine.DSpineModule``). The objective follows SpecForge's corrected DSpine objective
(sgl-project/SpecForge#1038, from the DSpine authors). Over each block's contiguous supervised
prefix ``k = 1..bs-1``, with weights ``w_k = exp(-(k-1)/gamma)``::

    backbone   = ce_alpha * sum(w * CE) + l1_alpha * sum(w * L1(p_draft, p_target))
    refinement = (ce_alpha * mean_w(CE_topk) + l1_alpha * mean_w(L1_topk)) * num_supervised
    alignment  = mean_layers(sum(w * (1 - cos(code_l, E[y]))))
    loss       = (backbone + refinement + beta_t * stopgrad(mean_w CE) * alignment) / batch

The refinement terms rescore each position's top-k backbone candidates after redoing the last
write with the true previous token -- the exact-sender refinement served at decode time. The
alignment weight ``beta_t`` warms up linearly, and a curriculum teacher-forces the predecessor
codes of the first draft layers with a probability that decays to zero
(:class:`DSpineScheduleCallback` feeds both the training step).

Unlike SpecForge, the detached CE scaling the alignment term is each micro-batch's own rather
than all-reduced over data-parallel ranks and the accumulation window; it only sets the scale of
a detached regularizer, and keeps the forward free of collectives.
"""

import logging

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, TrainerCallback

from ..dflash.conversion import DSpineDMRegistry
from .hf_dflash import HFDFlashModel, _block_position_weights, _block_targets, _block_tokens
from .modeling_dspine import DSpineModule

logger = logging.getLogger(__name__)

__all__ = ["DSpineScheduleCallback", "HFDSpineModel"]

# Rows of vocab-wide logits one checkpointed objective chunk materializes.
_OBJECTIVE_CHUNK_ROWS = 1024
# The refinement's L1 term skips positions where the target puts no more than this on the candidates.
_CANDIDATE_MASS_THRESHOLD = 1e-4


def _l1_to(logits, target_probs):
    """L1 distance between ``softmax(logits)`` and ``target_probs``, in float32: [..., V] -> [...]."""
    return (logits.float().softmax(dim=-1) - target_probs).abs().sum(dim=-1)


def _acceptance_length(pred, labels, mask):
    """Mean accepted draft prefix plus the target's bonus token, over blocks with a supervised token."""
    accepted = ((pred == labels) & mask).long().cumprod(dim=-1).sum(dim=-1)
    valid = mask.any(dim=-1)
    return ((accepted + 1) * valid).sum() / valid.sum().clamp_min(1)


@DSpineDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDSpineModel(HFDFlashModel):
    """DFlash model with DSpine's adjacent causal injection and objective.

    Registered in ``DSpineDMRegistry`` so that ``convert_to_dflash_model`` routes to it when
    ``dflash_architecture_config.projector_type == "dspine"``. Training reads its step schedules
    from :meth:`set_training_progress`, which :class:`DSpineScheduleCallback` calls.
    """

    def _build_draft_module(self, dflash_config):
        """Build the DSpine draft and derive its codebook and readouts from the target head.

        Skipped while the base is on meta (a ``from_pretrained`` restore, whose checkpoint
        carries them); a ``dflash_init_checkpoint`` warm start overwrites them afterwards.
        """
        module = DSpineModule(dflash_config)
        head_weight = self._base_model_lm_head.weight
        if head_weight.device.type != "meta":
            module.selected_prefix_bridge.initialize_transfer_space(head_weight)
        return module

    def modify(self, config):
        """Check DSpine's fixed choices, build the draft, and read the objective settings."""
        fixed = {
            "dflash_draft_attention": ("causal", "DSpine drafts with causal in-block attention"),
            "dflash_loss_objective": (
                "decay",
                "DSpine weighs positions by dflash_loss_decay_factor",
            ),
            "dflash_attention_sink": (False, "SGLang's DSPINE loader rejects attention sinks"),
        }
        for name, (required, reason) in fixed.items():
            if getattr(config, name) != required:
                raise ValueError(f"{reason}: set {name}={required!r} for projector_type=dspine.")
        super().modify(config)

        if getattr(self._base_model_lm_head, "bias", None) is not None or getattr(
            self._base_llm_config, "final_logit_softcapping", None
        ):
            raise ValueError(
                "DSpine scores candidates with raw target head rows, as its serving kernels do; "
                "a target lm_head with a bias or final_logit_softcapping is unsupported."
            )
        if self.dflash_module.selected_prefix_bridge.rank % 128:
            logger.warning(
                "DSpine code_rank=%d is not a multiple of 128, which SGLang's DSPINE kernels "
                "require: this draft trains but will not serve there.",
                self.dflash_module.selected_prefix_bridge.rank,
            )
        self.dflash_ce_loss_alpha = config.dflash_ce_loss_alpha
        self.dflash_l1_loss_alpha = config.dflash_l1_loss_alpha
        self.dflash_dspine_alignment_weight = config.dflash_dspine_alignment_weight
        self.dflash_dspine_alignment_warmup_steps = config.dflash_dspine_alignment_warmup_steps
        self.dflash_dspine_teacher_forcing_probability = (
            config.dflash_dspine_teacher_forcing_probability
        )
        self.dflash_dspine_teacher_forcing_hold_fraction = (
            config.dflash_dspine_teacher_forcing_hold_fraction
        )
        self.dflash_dspine_teacher_forcing_zero_fraction = (
            config.dflash_dspine_teacher_forcing_zero_fraction
        )
        self._training_progress = None
        self._dspine_metrics = None

    @property
    def _needs_base_logits(self) -> bool:
        """The L1 and refinement terms always read the target's distribution."""
        return True

    def get_exporter(self):
        """Get the exporter for the DSpine draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DSpineExporter

        return DSpineExporter(self)

    def set_training_progress(self, step: int, total_steps: int, lr_warmup_steps: int = 0):
        """Set the optimizer step that the alignment warmup and teacher-forcing curriculum read."""
        self._training_progress = (step, total_steps, lr_warmup_steps)

    def _schedule(self):
        """``(alignment_beta, teacher_forcing_probability)`` at the current training step."""
        if self._training_progress is None:
            raise RuntimeError(
                "DSpine training reads its step schedules from set_training_progress(); add "
                "DSpineScheduleCallback to the Trainer's callbacks."
            )
        step, total_steps, lr_warmup_steps = self._training_progress
        warmup = self.dflash_dspine_alignment_warmup_steps
        warmup = lr_warmup_steps if warmup is None else warmup
        beta = self.dflash_dspine_alignment_weight * (min(step / warmup, 1.0) if warmup else 1.0)
        hold = self.dflash_dspine_teacher_forcing_hold_fraction
        zero = self.dflash_dspine_teacher_forcing_zero_fraction
        decay = (zero - step / max(total_steps, 1)) / (zero - hold)
        return beta, self.dflash_dspine_teacher_forcing_probability * min(max(decay, 0.0), 1.0)

    def forward(self, *args, **kwargs):
        """Run the DFlash training forward and attach the DSpine metrics."""
        self._dspine_metrics = None
        outputs = super().forward(*args, **kwargs)
        if self._dspine_metrics is not None:
            outputs["dspine_metrics"] = self._dspine_metrics
            self._dspine_metrics = None
        return outputs

    def _draft_loss(
        self,
        input_ids,
        anchor_positions,
        block_keep_mask,
        loss_mask,
        base_outputs,
        *,
        noise_embedding,
        position_ids,
        attention_mask,
    ):
        """Run the DSpine draft and its objective; returns ``(loss, refined accuracy)``."""
        alignment_beta, teacher_forcing = self._schedule()
        bsz, n_blocks = anchor_positions.shape
        block_size = self.dflash_block_size
        block_ids = _block_tokens(input_ids, anchor_positions, block_size, 0)
        replace_mask = None
        if teacher_forcing > 0:
            draws = torch.rand(anchor_positions.shape, device=input_ids.device)
            replace_mask = (draws < teacher_forcing) & block_keep_mask
        out = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=base_outputs.target_hidden,
            position_ids=position_ids,
            attention_mask=attention_mask,
            block_token_ids=block_ids,
            replace_mask=replace_mask,
        )

        targets = _block_targets(
            input_ids, anchor_positions, block_keep_mask, loss_mask, block_size, self._label_shift
        )
        # Supervise each block's contiguous supervised prefix only: a draft token after the
        # first unsupervised one is never reached by an accepted draft.
        supervised = F.pad(targets.mask[..., 1:].cumprod(dim=-1), (1, 0))
        weights = _block_position_weights(
            supervised,
            "decay",
            1,
            decay_factor=self.dflash_loss_decay_factor,
            dpace_alpha=self.dflash_dpace_alpha,
        )[..., 1:]
        supervised = supervised[..., 1:]
        labels = block_ids[..., 1:]

        def predicted(x):  # [B, N * bs, ...] -> [B, N, bs - 1, ...] at the predicted positions
            return x.view(bsz, n_blocks, block_size, *x.shape[2:])[:, :, 1:]

        per_block = (
            predicted(out.hidden_states),
            predicted(out.pre_injection),
            out.layer_features.permute(1, 2, 0, 3, 4)[..., 1:, :],  # [B, N, L, bs - 1, C]
            labels,
            block_ids[..., :-1],
            targets.positions[..., 1:],
            weights,
            supervised,
        )
        chunk = max(1, _OBJECTIVE_CHUNK_ROWS // (bsz * (block_size - 1)))
        head_weight = self._base_model_lm_head.weight
        chunk_sums, base_ids, refined_ids = [], [], []
        for args in zip(*(x.split(chunk, dim=1) for x in per_block)):
            args = (*args, head_weight, base_outputs)
            if torch.is_grad_enabled():
                # Recompute the chunk's vocab-wide logits in backward instead of keeping them all.
                result = checkpoint(self._objective_chunk, *args, use_reentrant=False)
            else:
                result = self._objective_chunk(*args)
            chunk_sums.append(torch.stack(result[:8]))
            base_ids.append(result[8])
            refined_ids.append(result[9])
        ce_sum, l1_sum, rce_sum, rce_weight, rl1_sum, rl1_weight, alignment_sum, hits = torch.stack(
            chunk_sums
        ).sum(dim=0)

        ce_alpha, l1_alpha = self.dflash_ce_loss_alpha, self.dflash_l1_loss_alpha
        weight_sum = weights.sum().clamp_min(1e-30)
        valid_tokens = supervised.sum()
        # A gate that keeps no position also has a zero sum, so the clamp yields 0 there.
        refine_ce = rce_sum / rce_weight.clamp_min(1e-30)
        refine_l1 = rl1_sum / rl1_weight.clamp_min(1e-30)
        refinement = (ce_alpha * refine_ce + l1_alpha * refine_l1) * valid_tokens
        alignment_scale = alignment_beta * ce_sum.detach() / weight_sum
        loss = (
            ce_alpha * ce_sum + l1_alpha * l1_sum + refinement + alignment_scale * alignment_sum
        ) / bsz

        with torch.no_grad():
            mask = supervised.bool()
            base, refined = torch.cat(base_ids, dim=1), torch.cat(refined_ids, dim=1)
            n_valid = valid_tokens.clamp_min(1)
            metrics = {
                "accuracy": ((refined == labels) & mask).sum() / n_valid,
                "base_accuracy": ((base == labels) & mask).sum() / n_valid,
                "accept_len": _acceptance_length(refined, labels, mask),
                "base_accept_len": _acceptance_length(base, labels, mask),
                "backbone_ce": ce_sum / weight_sum,
                "backbone_l1": l1_sum / weight_sum,
                "refine_ce": refine_ce,
                "refine_l1": refine_l1,
                "alignment": alignment_sum / weight_sum,
                "candidate_coverage": hits / n_valid,
            }
            # One device sync for every scalar instead of one per value.
            values = torch.stack(list(metrics.values())).tolist()
        self._dspine_metrics = dict(zip(metrics, values))
        self._dspine_metrics.update(alignment_beta=alignment_beta, teacher_forcing=teacher_forcing)
        return loss, self._dspine_metrics["accuracy"]

    def _objective_chunk(
        self,
        hidden,
        pre_injection,
        layer_features,
        labels,
        predecessors,
        label_positions,
        weights,
        supervised,
        head_weight,
        base_outputs,
    ):
        """The objective's sums over one chunk of blocks, plus its backbone and refined picks.

        Per predicted position ``[B, n, bs - 1]``: ``hidden`` / ``pre_injection`` (..., H) are
        the final draft states and the last layer's pre-write states, ``layer_features``
        ([B, n, L, bs - 1, C]) every layer's codes, ``predecessors`` the true previous tokens.
        """
        draft = self.dflash_module
        # The block slices are strided; F.linear sends a strided input to a batched kernel
        # that is ~20x slower at the vocab-wide head.
        hidden, pre_injection = hidden.contiguous(), pre_injection.contiguous()
        head_logits = F.linear(hidden, head_weight)
        # Same candidates as on the float32 upcast below, at a fraction of the cost.
        candidates = head_logits.detach().topk(draft.head_topk, dim=-1).indices
        logits = head_logits.float()
        target_logits = self._teacher_logits(base_outputs, label_positions=label_positions).float()
        candidate_target_logits = target_logits.gather(-1, candidates)
        target_probs = target_logits.softmax(dim=-1)
        del target_logits
        ce = F.cross_entropy(logits.flatten(0, -2), labels.flatten(), reduction="none")
        ce = ce.view_as(labels)
        l1 = _l1_to(logits, target_probs)

        refined = draft.refine(pre_injection, predecessors)
        scores = torch.einsum("...h,...kh->...k", refined, F.embedding(candidates, head_weight))
        scores = scores.float()
        covered = candidates == labels.unsqueeze(-1)
        hit = covered.any(dim=-1)
        refine_ce = F.cross_entropy(
            scores.flatten(0, -2), covered.long().argmax(dim=-1).flatten(), reduction="none"
        ).view_as(labels)
        # The gate reads the target's full-vocabulary mass on the candidates, before renormalizing.
        eligible = target_probs.gather(-1, candidates).sum(dim=-1) > _CANDIDATE_MASS_THRESHOLD
        refine_l1 = _l1_to(scores, candidate_target_logits.softmax(dim=-1))

        codebook = draft.selected_prefix_bridge.codebook
        reference = F.embedding(labels, codebook).float().unsqueeze(2)
        alignment = 1.0 - F.cosine_similarity(layer_features.float(), reference, dim=-1)

        with torch.no_grad():
            refined_ids = candidates.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
        return (
            (ce * weights).sum(),
            (l1 * weights).sum(),
            (refine_ce * hit * weights).sum(),
            (hit * weights).sum(),
            (refine_l1 * eligible * weights).sum(),
            (eligible * weights).sum(),
            (alignment * weights.unsqueeze(2)).sum() / layer_features.shape[2],
            (hit * supervised).sum(),
            logits.detach().argmax(dim=-1),
            refined_ids,
        )

    @torch.no_grad()
    def pseudo_speculative_generate(self, input_ids, steps=1):
        """Draft one block for AR validation with DSpine's serial exact-sender refinement.

        Position ``t`` takes the backbone's top-k candidates and rescores them after redoing the
        last write with the token chosen at ``t - 1`` (the anchor for ``t = 1``), as SGLang's
        ``serial`` decode does.
        """
        base_token, hidden_states = self._generation_base(input_ids)
        if steps < 1:
            return base_token, None
        inputs = self._draft_block_inputs(base_token, hidden_states)
        device = inputs["noise_embedding"].device
        block_ids = torch.full(
            (base_token.shape[0], 1, self.dflash_block_size), self.mask_token_id, device=device
        )
        block_ids[:, 0, 0] = base_token.squeeze(-1).to(device)
        out = self.dflash_module(**inputs, block_token_ids=block_ids)

        draft = self.dflash_module
        head_weight = self._base_model_lm_head.weight
        backbone_logits = F.linear(out.hidden_states, head_weight)
        prev_token = block_ids[:, 0, 0]
        draft_tokens = []
        for t in range(1, 1 + min(steps, self.dflash_block_size - 1)):
            candidates = backbone_logits[:, t].topk(draft.head_topk, dim=-1).indices
            refined = draft.refine(out.pre_injection[:, t], prev_token)
            scores = torch.einsum("bh,bkh->bk", refined, head_weight[candidates])
            prev_token = candidates.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
            draft_tokens.append(prev_token)
        return base_token, torch.stack(draft_tokens, dim=1).to(input_ids.device)


class DSpineScheduleCallback(TrainerCallback):
    """Feed the HF Trainer's step to DSpine's alignment warmup and teacher-forcing curriculum."""

    def on_step_begin(self, args, state, control, **kwargs):
        """Set the DSpine model's training progress for the upcoming optimizer step."""
        model = kwargs.get("model")
        inner = getattr(model, "module", model)
        if isinstance(inner, HFDSpineModel):
            total_steps = state.max_steps
            inner.set_training_progress(
                state.global_step, total_steps, args.get_warmup_steps(total_steps)
            )
