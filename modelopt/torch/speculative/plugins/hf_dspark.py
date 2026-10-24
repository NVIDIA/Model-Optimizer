# Adapted from https://github.com/deepseek-ai/DeepSpec/blob/add63ba/deepspec/modeling/dspark/loss.py
# Copyright (c) 2026 The DeepSpec Authors
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

"""DSpark speculative decoding plugin for HuggingFace models.

DSpark reuses the DFlash draft backbone and training pipeline (anchor sampling,
noise/mask construction, KV-injection attention) and adds a lightweight
sequential head (see ``modeling_dspark.DSparkModule``):

- The backbone produces *base* logits for a full draft block in parallel.
- A Markov head adds a prefix-dependent transition bias ``B_k`` to each block
  position, inducing a causal block distribution
  ``p_k(x_k | x_0, x_<k) = softmax(U_k + B_k)`` that the parallel backbone lacks.
- An optional confidence head predicts the per-position acceptance probability,
  supervised against the analytical accept rate. (Inference-time
  confidence-scheduled verification — the hardware-aware scheduler — lives in the
  serving engine and is out of scope here; we only train/export the head.)

Training uses next-token (``shift_label``) alignment — position ``k`` predicts the
token at ``anchor+k+1`` and position 0 is *not* excluded — and a three-term loss::

    loss = ce_alpha * CE(final) + l1_alpha * TVD(final, target) + conf_alpha * BCE(conf)

where ``TVD`` is the total-variation distance between the corrected draft
distribution and the target (base-model) distribution at the aligned position,
and the confidence target is ``c* = 1 - 0.5 * TVD`` (the analytical accept rate).
"""

import logging

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..dflash.conversion import DSparkDMRegistry
from .dflash_losses import DraftLogits, LossOutput, _block_tokens, objective_term
from .hf_dflash import HFDFlashModel
from .modeling_dflash import DFlashBaseModelOutput
from .modeling_dspark import DSparkModule

logger = logging.getLogger(__name__)

__all__ = ["HFDSparkModel"]


@DSparkDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDSparkModel(HFDFlashModel):
    """DFlash model with the DSpark sequential (Markov) + confidence head.

    Registered in ``DSparkDMRegistry`` so that ``convert_to_dflash_model`` routes
    to it when ``dflash_architecture_config.projector_type == "dspark"``.
    """

    _label_shift = 1

    def _build_draft_module(self, dflash_config):
        """Build the DSpark draft module (DFlash backbone + Markov/confidence head)."""
        return DSparkModule(dflash_config)

    def modify(self, config):
        """Initialize the DSpark draft module and read the confidence term's weight."""
        arch_config = config.dflash_architecture_config
        if arch_config.get("markov_rank") is None:
            raise ValueError(
                "DSpark (projector_type='dspark') requires 'markov_rank' (> 0) in "
                "dflash_architecture_config (the Markov head's low-rank dimension)."
            )
        super().modify(config)
        self.dflash_confidence_head_alpha = config.dflash_confidence_head_alpha
        if self.dflash_confidence_head_alpha > 0 and not self.dflash_module.use_confidence_head:
            raise ValueError(
                "dflash_confidence_head_alpha > 0 but the confidence head was not built; "
                "set dflash_architecture_config.use_confidence_head=true."
            )

    def get_exporter(self):
        """Get the exporter for the DSpark draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DSparkExporter

        return DSparkExporter(self)

    def _build_objective(self, config):
        """CE plus TVD to the base model; the DeepSpec defaults are TVD-dominant."""
        return [
            objective_term("ce", config.dflash_ce_loss_alpha),
            objective_term("tvd", config.dflash_l1_loss_alpha),
        ]

    @property
    def _position_weighting(self):
        # dflash_loss_objective is not wired for DSpark yet: it always decays.
        return "decay"

    def _online_base_outputs(self, input_ids, attention_mask, **kwargs):
        """The target hidden states, and the base's final hidden for the loss to project."""
        # The inner model, not the CausalLM: the loss projects only the rows it reads.
        with torch.no_grad():
            base_out = self._base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        return DFlashBaseModelOutput(
            target_hidden=self._target_hidden(base_out.hidden_states),
            base_hidden=base_out.last_hidden_state,
        )

    def _head_inputs(self, batch):
        """Each block position's teacher-forced previous token, and the draft's hidden states.

        The previous token of position k is the real token at anchor+k (the anchor itself
        for position 0). Returns ``prev_ids [B, N, bs]`` and ``hidden [B, N, bs, H]``.
        """
        bsz, n_blocks = batch.anchors.shape
        bs = self.dflash_block_size
        prev_ids = _block_tokens(batch.input_ids, batch.anchors, bs, self._label_shift - 1)
        return prev_ids, batch.hidden.reshape(bsz, n_blocks, bs, -1)

    def _draft_logits(self, batch):
        """The backbone's logits plus the Markov transition bias."""
        backbone = super()._draft_logits(batch).backbone
        prev_ids, hidden = self._head_inputs(batch)
        bias = self.dflash_module.compute_markov_bias(prev_ids, hidden)
        return DraftLogits(final=backbone + bias, backbone=backbone)

    def _compute_loss(self, ctx):
        """The CE + TVD objective plus the confidence head's BCE against the accept rate."""
        if not ctx.trainable:
            # As forward()'s early return: DDP still reduces the confidence head's gradient.
            out = self._zero_output(ctx, self._draft_zero_loss(ctx.logits.final.device))
            zero = torch.zeros((), device=out.loss.device)
            return out._replace(metrics={**out.metrics, "confidence_loss": zero})

        out = super()._compute_loss(ctx)
        confidence_loss = out.loss.new_zeros(())
        if self.dflash_confidence_head_alpha > 0:
            prev_ids, hidden = self._head_inputs(ctx.batch)
            confidence_logits = self.dflash_module.compute_confidence_logits(prev_ids, hidden)
            # The analytical accept rate c* = 1 - 0.5 * TVD.
            accept_rate = (1.0 - 0.5 * ctx.tvd("final")).clamp(0.0, 1.0).detach()
            conf_bce = F.binary_cross_entropy_with_logits(
                confidence_logits.reshape(-1).float(), accept_rate, reduction="none"
            )
            confidence_loss = (conf_bce * ctx.weights.view(-1)).sum() / ctx.weight_sum
        return LossOutput(
            out.loss + self.dflash_confidence_head_alpha * confidence_loss,
            out.accuracy,
            {**out.metrics, "confidence_loss": confidence_loss.detach()},
        )

    @torch.no_grad()
    def pseudo_speculative_generate(self, input_ids, steps=1):
        """Generate draft tokens for AR validation, with the Markov correction applied.

        Mirrors ``HFDFlashModel.pseudo_speculative_generate`` for the backbone pass,
        then samples the block left-to-right applying the Markov transition bias
        autoregressively (DSpark's semi-autoregressive generation). Uses next-token
        (shift) alignment: the anchor is block position 0 and predicts the first draft
        token; position k's bias conditions on the token decoded at position k-1
        (the anchor for k=0). Without this override DSpark would fall back to the
        DFlash backbone-only generate, under-reporting acceptance length.
        """
        base_token, hidden_states = self._generation_base(input_ids)
        if steps < 1:
            return base_token, None
        draft_hidden = self._draft_block(base_token, hidden_states)
        # Under device_map="auto" a tied lm_head sits with the embedding table, not the draft.
        backbone_logits = self._base_model_lm_head(draft_hidden).to(draft_hidden.device)

        # Autoregressive Markov sampling over the block.
        m = self.dflash_module
        num_tokens = min(steps, self.dflash_block_size)
        # anchor precedes block position 0
        prev_token = base_token.squeeze(-1).to(draft_hidden.device)
        state = None
        draft_tokens = []
        for k in range(num_tokens):
            bias, state = m.markov_step(prev_token, draft_hidden[:, k, :], state)
            tok = (backbone_logits[:, k, :] + bias).argmax(dim=-1)
            draft_tokens.append(tok)
            prev_token = tok
        return base_token, torch.stack(draft_tokens, dim=1).to(input_ids.device)
