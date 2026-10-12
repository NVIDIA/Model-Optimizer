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

where ``final_loss`` is CE on the corrected logits and ``base_loss`` is CE on the
backbone-only logits. ``lambda_base`` decays linearly from ``lambda_base_start``
to 0 over ``lambda_base_decay_ratio`` of training (curriculum: learn a good
parallel backbone first, then the causal correction). The schedule is driven by
``DominoLambdaCallback`` from the HF Trainer's global step.
"""

import logging

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..dflash.conversion import DominoDMRegistry
from .hf_dflash import (
    HFDFlashModel,
    _block_position_weights,
    _block_targets,
    _block_tokens,
    _masked_top1,
)
from .modeling_dflash import DFlashBaseModelOutput
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
        if not getattr(self.dflash_module, "shift_label", True):
            raise NotImplementedError(
                "Domino currently supports shift_label=True (next-token alignment) only."
            )

    def get_exporter(self):
        """Get the exporter for the Domino draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DominoExporter

        return DominoExporter(self)

    def _current_lambda_base(self) -> float:
        return float(getattr(self, "_lambda_base", self.dflash_lambda_base_start))

    def _apply_domino_head(self, hidden, base_logits, input_ids, anchor_positions, n_blocks):
        """Add the GRU causal correction to the suffix positions of each block.

        Args:
            hidden: Draft backbone output [B, N*block_size, H].
            base_logits: Backbone logits [B, N*block_size, vocab].
            input_ids: Original token IDs [B, seq_len].
            anchor_positions: Anchor positions per block [B, N].
            n_blocks: Number of blocks N.

        Returns:
            Corrected logits [B, N*block_size, vocab].
        """
        bsz = input_ids.shape[0]
        bs = self.dflash_block_size
        suffix_start = self.dflash_module.pure_draft_prefix_len

        hidden4d = hidden.reshape(bsz, n_blocks, bs, hidden.size(-1))
        base4d = base_logits.reshape(bsz, n_blocks, bs, -1)

        # Teacher-forced previous tokens: the real token at anchor+j for j in [0, bs).
        prev_ids = _block_tokens(input_ids, anchor_positions, bs, self._label_shift - 1)

        block_emb = self._base_model_embeddings(prev_ids)  # [B, N, bs, H]
        gru_in = block_emb.reshape(bsz * n_blocks, bs, block_emb.size(-1))
        gru_out, _ = self.dflash_module.prefix_gru(gru_in)
        gru_out = gru_out.reshape(bsz, n_blocks, bs, -1)

        # Causal state for suffix positions: gru_out[p] summarizes anchor+0..anchor+p.
        prefix_states = gru_out[:, :, suffix_start:, :]
        z_n = hidden4d[:, :, suffix_start:, :]
        logits_e = self.dflash_module.embed_proj(torch.cat([z_n, prefix_states], dim=-1))

        prefix_logits = base4d[:, :, :suffix_start, :]
        suffix_logits = base4d[:, :, suffix_start:, :] + logits_e
        final4d = torch.cat([prefix_logits, suffix_logits], dim=2)
        return final4d.reshape(bsz, n_blocks * bs, -1)

    def _compute_domino_loss(
        self, base_logits, final_logits, input_ids, anchor_positions, block_keep_mask, loss_mask
    ):
        """Compute the (1-lambda)*final + lambda*base weighted CE loss and accuracies.

        Uses next-token (shift_label) alignment: position k predicts the token at
        anchor+k+1, and position 0 is *not* excluded (unlike base DFlash).
        """
        targets = _block_targets(
            input_ids,
            anchor_positions,
            block_keep_mask,
            loss_mask,
            self.dflash_block_size,
            self._label_shift,
        )
        binary_eval_mask = targets.mask.view(-1)
        # dflash_loss_objective is not wired for Domino yet: it always decays.
        weight_mask = _block_position_weights(
            targets.mask,
            "decay",
            1 - self._label_shift,
            self.dflash_loss_decay_factor,
            self.dflash_dpace_alpha,
        )

        flat_final = final_logits.reshape(-1, final_logits.size(-1))
        flat_base = base_logits.reshape(-1, base_logits.size(-1))
        flat_targets = targets.token_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        valid_count = flat_weights.sum() + 1e-6

        lambda_base = self._current_lambda_base()

        if valid_count > 1.0:
            final_loss = (
                F.cross_entropy(flat_final, flat_targets, reduction="none") * flat_weights
            ).sum() / valid_count
            base_loss = (
                F.cross_entropy(flat_base, flat_targets, reduction="none") * flat_weights
            ).sum() / valid_count
            loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss

            with torch.no_grad():
                accuracy = _masked_top1(flat_final, flat_targets, binary_eval_mask).item()
                base_accuracy = _masked_top1(flat_base, flat_targets, binary_eval_mask).item()
        else:
            loss = flat_final.sum() * 0.0
            final_loss = loss
            base_loss = loss
            accuracy = 0.0
            base_accuracy = 0.0

        metrics = {
            "final_loss": final_loss.detach().item(),
            "base_loss": base_loss.detach().item(),
            "base_accuracy": base_accuracy,
            "lambda_base": lambda_base,
        }
        return loss, accuracy, metrics

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        """Domino training forward: DFlash backbone + causal correction head + dual loss.

        Mirrors ``HFDFlashModel.forward`` for data preparation (reusing the inherited
        anchor/noise/mask/position helpers), then applies the Domino head and the
        two-term loss. Eval/offline-eval is delegated to the DFlash parent.
        """
        if not self.training:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        device = input_ids.device

        if seq_len % block_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by block_size ({block_size}). "
                f"Adjust training_seq_len or use padding."
            )

        # 1. Target hidden states (Domino does not use target-logit KD).
        if self.dflash_offline:
            assert "base_model_outputs" in kwargs
            base_outputs = DFlashBaseModelOutput.from_offline_dict(kwargs["base_model_outputs"])
            target_hidden = base_outputs.target_hidden
        else:
            with torch.no_grad():
                base_out = self._base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            target_hidden = self._target_hidden(base_out.hidden_states)

        # 2. Build loss mask (same convention as DFlash).
        if labels is not None:
            loss_mask = (labels != LabelSmoother.ignore_index).float()
        elif attention_mask is not None:
            loss_mask = attention_mask.float()
        else:
            loss_mask = torch.ones(bsz, seq_len, device=device)
        if kwargs.get("loss_mask") is not None:
            loss_mask = loss_mask * kwargs["loss_mask"]

        # 3. Random anchor sampling (inherited).
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        if n_blocks == 0 or not block_keep_mask.any():
            return ModelOutput(loss=self._draft_zero_loss(device), logits=None, train_acc=[[0.0]])

        # 4. Build draft inputs (inherited helpers).
        noise_embedding = self._build_noise_embedding(
            input_ids, anchor_positions, block_keep_mask, n_blocks
        )
        full_pos = self._build_position_ids(seq_len, anchor_positions, device)
        attn_mask = self._build_draft_attention_mask(
            seq_len,
            anchor_positions,
            block_keep_mask,
            n_blocks,
            target_hidden.dtype,
            device,
            window=self.dflash_swa_window_size,
        )

        # 5. Draft backbone forward.
        hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=full_pos,
            attention_mask=attn_mask,
        )

        # 6. Base + corrected logits, then dual loss.
        base_logits = self._base_model_lm_head(hidden)
        final_logits = self._apply_domino_head(
            hidden, base_logits, input_ids, anchor_positions, n_blocks
        )
        loss, accuracy, metrics = self._compute_domino_loss(
            base_logits, final_logits, input_ids, anchor_positions, block_keep_mask, loss_mask
        )

        return ModelOutput(loss=loss, logits=None, train_acc=[[accuracy]], domino_metrics=metrics)

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
