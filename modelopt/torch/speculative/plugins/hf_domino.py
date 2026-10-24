# Adapted from https://github.com/sgl-project/SpecForge/blob/8ea5ca6/specforge/core/domino.py
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

"""Domino speculative decoding plugin for HuggingFace models.

Domino reuses the DFlash draft backbone and training pipeline (anchor sampling,
noise/mask construction, KV-injection attention) and adds a lightweight causal
correction head (see ``modeling_domino.DominoModule``):

- Backbone produces *base* logits for a full draft block in parallel.
- A GRU runs over the block's previously decoded (teacher-forced) token
  embeddings to produce a causal state, which is fused with the backbone hidden
  state and projected to a vocab-sized logit correction added to the suffix
  positions. This injects the intra-block causal dependency the parallel
  backbone lacks.

Training uses next-token (shift_label) alignment and a two-term loss::

    loss = (1 - lambda_base) * final_loss + lambda_base * base_loss

where ``final_loss`` is the block objective (CE unless ``dflash_loss_terms`` sets
another) on the corrected logits and ``base_loss`` the same objective on the
backbone-only logits. ``lambda_base`` decays linearly from ``lambda_base_start``
to 0 over ``lambda_base_decay_ratio`` of training (curriculum: learn a good
parallel backbone first, then the causal correction). The schedule is driven by
``DominoLambdaCallback`` from the HF Trainer's global step.
"""

import logging

import torch
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback

from ..dflash.conversion import DominoDMRegistry
from .dflash_losses import DraftLogits, LossOutput, _block_tokens, objective_loss
from .hf_dflash import HFDFlashModel
from .modeling_domino import DominoModule

logger = logging.getLogger(__name__)

__all__ = ["DominoLambdaCallback", "HFDominoModel", "compute_lambda_base"]


def compute_lambda_base(
    global_step: int,
    total_steps: int,
    lambda_start: float = 1.0,
    decay_ratio: float = 1.0,
) -> float:
    """Linearly decay lambda_base from ``lambda_start`` to 0.

    Decay completes after ``decay_ratio * total_steps`` steps; clamped to [0, 1].
    """
    decay_steps = max(1, int(total_steps * decay_ratio))
    progress = min(global_step / decay_steps, 1.0)
    lambda_base = lambda_start * (1.0 - progress)
    return max(0.0, min(1.0, lambda_base))


@DominoDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDominoModel(HFDFlashModel):
    """DFlash model with the Domino causal correction head (HF transformers).

    Registered in ``DominoDMRegistry`` so that ``convert_to_dflash_model`` can
    route to it when ``dflash_architecture_config.projector_type == "domino"``.
    """

    _label_shift = 1

    def _build_draft_module(self, dflash_config):
        """Build the Domino draft module (DFlash backbone + GRU correction head)."""
        return DominoModule(dflash_config)

    def modify(self, config):
        """Initialize the Domino draft module and read the lambda_base schedule."""
        # Validate head fields up front: a clear error here beats a cryptic
        # AttributeError later in DominoModule.__init__ or the exporter.
        arch_config = config.dflash_architecture_config
        missing = [k for k in ("emb_dim", "gru_hidden_dim") if arch_config.get(k) is None]
        if missing:
            raise ValueError(
                f"Domino (projector_type='domino') requires {missing} in "
                "dflash_architecture_config (the GRU correction head dimensions)."
            )
        super().modify(config)
        # Curriculum schedule for the base/final loss mixing weight. Read here
        # (DFlashConfig carries the two fields); updated each step by
        # DominoLambdaCallback. Defaults keep a single forward (e.g. unit tests)
        # well-defined without a scheduler.
        self.dflash_lambda_base_start = config.dflash_lambda_base_start
        self.dflash_lambda_base_decay_ratio = config.dflash_lambda_base_decay_ratio
        self._lambda_base = self.dflash_lambda_base_start

    def get_exporter(self):
        """Get the exporter for the Domino draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DominoExporter

        return DominoExporter(self)

    def _current_lambda_base(self) -> float:
        return float(getattr(self, "_lambda_base", self.dflash_lambda_base_start))

    def _default_loss_terms(self, config):
        """CE on the labels, for the corrected logits and for the backbone's."""
        return {"ce": 1.0}

    @property
    def _position_weighting(self):
        # dflash_loss_objective is not wired for Domino yet: it always decays.
        return "decay"

    def _draft_logits(self, batch):
        """The backbone's logits with the GRU causal correction added to each block's suffix."""
        backbone = super()._draft_logits(batch).backbone
        bsz, n_blocks, bs, _ = backbone.shape
        suffix_start = self.dflash_module.pure_draft_prefix_len

        hidden4d = batch.hidden.reshape(bsz, n_blocks, bs, -1)

        # Teacher-forced previous tokens: the real token at anchor+j for j in [0, bs).
        prev_ids = _block_tokens(batch.input_ids, batch.anchors, bs, self._label_shift - 1)

        block_emb = self._base_model_embeddings(prev_ids)  # [B, N, bs, H]
        gru_in = block_emb.reshape(bsz * n_blocks, bs, block_emb.size(-1))
        gru_out, _ = self.dflash_module.prefix_gru(gru_in)
        gru_out = gru_out.reshape(bsz, n_blocks, bs, -1)

        # Causal state for suffix positions: gru_out[p] summarizes anchor+0..anchor+p.
        prefix_states = gru_out[:, :, suffix_start:, :]
        z_n = hidden4d[:, :, suffix_start:, :]
        logits_e = self.dflash_module.embed_proj(torch.cat([z_n, prefix_states], dim=-1))

        prefix_logits = backbone[:, :, :suffix_start, :]
        suffix_logits = backbone[:, :, suffix_start:, :] + logits_e
        return DraftLogits(
            final=torch.cat([prefix_logits, suffix_logits], dim=2), backbone=backbone
        )

    def _compute_loss(self, ctx):
        """``(1 - lambda_base) * objective(final) + lambda_base * objective(backbone)``."""
        out = super()._compute_loss(ctx)
        lambda_base = self._current_lambda_base()
        loss = backbone_loss = out.loss
        if ctx.trainable:
            backbone_loss, _ = objective_loss(self._objective, ctx, "backbone")
            loss = (1.0 - lambda_base) * out.loss + lambda_base * backbone_loss
        metrics = {
            **out.metrics,
            "loss_final": out.loss.detach(),
            "loss_backbone": backbone_loss.detach(),
            # float64 holds the scheduled value exactly.
            "lambda_base": torch.full((), lambda_base, dtype=torch.float64, device=out.loss.device),
        }
        return LossOutput(loss, out.accuracy, metrics)

    @torch.no_grad()
    def pseudo_speculative_generate(self, input_ids, steps=1):
        """Draft the block left to right, adding the correction head as training does.

        Block position ``k`` predicts the token after the anchor and the ``k`` tokens drafted
        before it. From ``pure_draft_prefix_len`` on, its backbone logits get the correction
        from the GRU state over exactly those tokens.
        """
        base_token, hidden_states = self._generation_base(input_ids)
        if steps < 1:
            return base_token, None
        draft_hidden = self._draft_block(base_token, hidden_states)
        device = draft_hidden.device
        # Under device_map="auto" the embedding table and a tied lm_head sit on the first GPU,
        # not the draft's, and their outputs stay there.
        backbone_logits = self._base_model_lm_head(draft_hidden).to(device)  # [B, block_size, V]

        m = self.dflash_module
        prev_token = base_token.squeeze(-1).to(device)
        gru_state = None
        draft_tokens = []
        for k in range(min(steps, self.dflash_block_size)):
            prev_emb = self._base_model_embeddings(prev_token).to(device)
            gru_out, gru_state = m.prefix_gru(prev_emb.unsqueeze(1), gru_state)
            logits = backbone_logits[:, k, :]
            if k >= m.pure_draft_prefix_len:
                correction = m.embed_proj(torch.cat([draft_hidden[:, k, :], gru_out[:, 0, :]], -1))
                logits = logits + correction
            prev_token = logits.argmax(dim=-1)
            draft_tokens.append(prev_token)
        return base_token, torch.stack(draft_tokens, dim=1).to(input_ids.device)


class DominoLambdaCallback(TrainerCallback):
    """Update the model's ``lambda_base`` from the HF Trainer global step.

    Linearly decays the base-loss weight from ``lambda_base_start`` to 0 over
    ``lambda_base_decay_ratio`` of total training steps.
    """

    def on_step_begin(self, args, state, control, **kwargs):
        """Set ``model._lambda_base`` for the upcoming step."""
        model = kwargs.get("model")
        if model is None:
            return
        # Unwrap DDP/FSDP if needed.
        inner = getattr(model, "module", model)
        if not hasattr(inner, "dflash_lambda_base_start"):
            return
        if state.max_steps and state.max_steps > 0:
            total_steps = state.max_steps
        else:
            # No max_steps -> decay window is one step -> lambda_base is 0 from the
            # start, disabling the curriculum. Warn once instead of doing it silently.
            total_steps = 1
            if not getattr(self, "_warned_no_max_steps", False):
                logger.warning(
                    "DominoLambdaCallback: state.max_steps unset (<=0); lambda_base "
                    "curriculum disabled (decays to 0 from the first step)."
                )
                self._warned_no_max_steps = True
        inner._lambda_base = compute_lambda_base(
            state.global_step,
            total_steps,
            inner.dflash_lambda_base_start,
            inner.dflash_lambda_base_decay_ratio,
        )
