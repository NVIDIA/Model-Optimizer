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

# Adapted from https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/loss.py
# Copyright (c) 2025 DeepSeek-AI
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
import torch.utils.checkpoint
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..dflash.conversion import DSparkDMRegistry
from .hf_dflash import HFDFlashModel
from .modeling_dflash import DFlashBaseModelOutput
from .modeling_dspark import DSparkModule

logger = logging.getLogger(__name__)

__all__ = ["HFDSparkModel"]


def _tvd_per_token(final_logits, teacher_logits, chunk_size=1024):
    """Total-variation distance ||softmax(a)-softmax(b)||_1 / ... per token, memory-lean.

    Materializing both [N, vocab] float32 softmax tensors at once OOMs at large
    N (num_anchors*block_size) and a 150k vocab. We compute it in row chunks and
    gradient-checkpoint each chunk so the wide softmaxes are recomputed in backward
    rather than held — peak memory ~ chunk_size*vocab instead of N*vocab. The math
    is identical to ``(softmax(final)-softmax(teacher)).abs().sum(-1)``.
    """

    def _chunk(a, b):
        return (
            (torch.softmax(a.float(), dim=-1) - torch.softmax(b.float(), dim=-1)).abs().sum(dim=-1)
        )

    outs = []
    for i in range(0, final_logits.size(0), chunk_size):
        a, b = final_logits[i : i + chunk_size], teacher_logits[i : i + chunk_size]
        if torch.is_grad_enabled() and a.requires_grad:
            outs.append(torch.utils.checkpoint.checkpoint(_chunk, a, b, use_reentrant=False))
        else:
            outs.append(_chunk(a, b))
    return torch.cat(outs, dim=0)


@DSparkDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDSparkModel(HFDFlashModel):
    """DFlash model with the DSpark sequential (Markov) + confidence head.

    Registered in ``DSparkDMRegistry`` so that ``convert_to_dflash_model`` routes
    to it when ``dflash_architecture_config.projector_type == "dspark"``.
    """

    def _build_draft_module(self, dflash_config):
        """Build the DSpark draft module (DFlash backbone + Markov/confidence head)."""
        return DSparkModule(dflash_config)

    def modify(self, config):
        """Initialize the DSpark draft module and read the loss-mixing weights."""
        arch_config = config.dflash_architecture_config
        if arch_config.get("markov_rank") is None:
            raise ValueError(
                "DSpark (projector_type='dspark') requires 'markov_rank' (> 0) in "
                "dflash_architecture_config (the Markov head's low-rank dimension)."
            )
        super().modify(config)
        # Three-term loss weights (DSpark only). Defaults follow the DeepSpec recipe
        # (L1/TVD-dominant) so a config that only sets the head still trains sensibly.
        self.dflash_ce_loss_alpha = getattr(config, "dflash_ce_loss_alpha", 0.1)
        self.dflash_l1_loss_alpha = getattr(config, "dflash_l1_loss_alpha", 0.9)
        self.dflash_confidence_head_alpha = getattr(config, "dflash_confidence_head_alpha", 0.0)
        if self.dflash_confidence_head_alpha > 0 and not self.dflash_module.use_confidence_head:
            raise ValueError(
                "dflash_confidence_head_alpha > 0 but the confidence head was not built; "
                "set dflash_architecture_config.use_confidence_head=true."
            )

    def get_exporter(self):
        """Get the exporter for the DSpark draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DSparkExporter

        return DSparkExporter(self)

    def _apply_markov_head(self, hidden, backbone_logits, input_ids, anchor_positions, n_blocks):
        """Add the Markov transition bias to the backbone base logits.

        Returns ``(final_logits [B, N, bs, V], confidence_logits [B, N, bs] | None)``.
        """
        bsz, seq_len = input_ids.shape
        bs = self.dflash_block_size
        device = input_ids.device

        hidden4d = hidden.reshape(bsz, n_blocks, bs, hidden.size(-1))
        base4d = backbone_logits.reshape(bsz, n_blocks, bs, -1)

        # Teacher-forced previous token for block position k: the real token at
        # anchor+k (so position 0's predecessor is the anchor itself).
        prev_offsets = torch.arange(bs, device=device).view(1, 1, -1)
        prev_idx = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(max=seq_len - 1)
        prev_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, prev_idx)

        bias = self.dflash_module.compute_markov_bias(prev_ids, hidden4d)
        final4d = base4d + bias

        confidence_logits = None
        if self.dflash_module.use_confidence_head:
            confidence_logits = self.dflash_module.compute_confidence_logits(prev_ids, hidden4d)
        return final4d, confidence_logits

    def _compute_dspark_loss(
        self,
        backbone_logits,
        final_logits,
        confidence_logits,
        input_ids,
        anchor_positions,
        block_keep_mask,
        loss_mask,
        target_model_logits,
    ):
        """Compute the three-term DSpark loss (CE + TVD + confidence BCE) and metrics.

        Uses next-token (shift_label) alignment: block position k predicts the token
        at anchor+k+1; the aligned target distribution is the base model's own
        next-token distribution at position anchor+k (= label index - 1).
        """
        bsz, seq_len = input_ids.shape
        bs = self.dflash_block_size
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device
        vocab = final_logits.size(-1)

        # shift_label=True: label for block position k is the token at anchor+k+1.
        label_offsets = torch.arange(1, 1 + bs, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )

        # Weight mask: valid block * in bounds * loss_mask (no pos-0 exclusion).
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, bs).float()
        weight_mask = weight_mask * valid_label.float()
        orig_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        weight_mask = weight_mask * orig_loss_mask

        binary_eval_mask = weight_mask.view(-1)

        # Exponential position decay (exp(-k/gamma); position 0 gets weight 1).
        if self.dflash_loss_decay_factor > 0:
            k = torch.arange(bs, device=device).view(1, 1, -1)
            decay = torch.exp(-k.float() / self.dflash_loss_decay_factor)
            weight_mask = weight_mask * decay

        flat_final = final_logits.reshape(-1, vocab)
        flat_base = backbone_logits.reshape(-1, vocab)
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        valid_count = flat_weights.sum() + 1e-6

        # Aligned target distribution: base-model logits that predict token anchor+k+1
        # sit at position anchor+k (= label index - 1).
        teacher_indices = (safe_label_indices - 1).clamp(min=0)
        teacher_logits = torch.gather(
            target_model_logits.unsqueeze(1).expand(-1, n_blocks, -1, -1),
            2,
            teacher_indices.unsqueeze(-1).expand(-1, -1, -1, vocab),
        )
        flat_teacher = teacher_logits.reshape(-1, vocab).detach()

        if valid_count <= 1.0:
            loss = flat_final.sum() * 0.0
            metrics = {"ce_loss": 0.0, "l1_loss": 0.0, "confidence_loss": 0.0, "base_accuracy": 0.0}
            return loss, 0.0, metrics

        # Term 1: cross-entropy on the corrected (final) logits.
        ce_per_token = F.cross_entropy(flat_final, flat_targets, reduction="none")
        ce_loss = (ce_per_token * flat_weights).sum() / valid_count

        # Term 2: total-variation distance between the corrected draft and target.
        # Chunked + checkpointed to avoid materializing two [N, vocab] softmaxes at once.
        l1_per_token = _tvd_per_token(flat_final, flat_teacher)
        l1_loss = (l1_per_token * flat_weights).sum() / valid_count

        # Term 3: confidence head BCE against the analytical accept rate c* = 1 - 0.5*TVD.
        confidence_loss = ce_loss.new_zeros(())
        if confidence_logits is not None and self.dflash_confidence_head_alpha > 0:
            accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0).detach()
            conf_bce = F.binary_cross_entropy_with_logits(
                confidence_logits.reshape(-1).float(), accept_rate, reduction="none"
            )
            confidence_loss = (conf_bce * flat_weights).sum() / valid_count

        loss = (
            self.dflash_ce_loss_alpha * ce_loss
            + self.dflash_l1_loss_alpha * l1_loss
            + self.dflash_confidence_head_alpha * confidence_loss
        )

        with torch.no_grad():
            eval_count = binary_eval_mask.sum() + 1e-6
            keep = binary_eval_mask > 0.5
            accuracy = (
                ((flat_final.argmax(dim=-1) == flat_targets) & keep).sum().float() / eval_count
            ).item()
            base_accuracy = (
                ((flat_base.argmax(dim=-1) == flat_targets) & keep).sum().float() / eval_count
            ).item()
            metrics = {
                "ce_loss": ce_loss.detach().item(),
                "l1_loss": l1_loss.detach().item(),
                "confidence_loss": float(confidence_loss.detach().item()),
                "base_accuracy": base_accuracy,
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
        """DSpark training forward: DFlash backbone + Markov head + three-term loss.

        Mirrors ``HFDFlashModel.forward`` for data preparation (reusing the inherited
        anchor/noise/mask/position helpers), then applies the Markov head and the
        DSpark loss. Eval/offline-eval is delegated to the DFlash parent.
        """
        if not self.training:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
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

        # 1. Target hidden states AND target-model logits (DSpark's L1/confidence
        #    terms both need the base model's next-token distribution).
        if self.dflash_offline:
            assert "base_model_outputs" in kwargs
            base_outputs = DFlashBaseModelOutput.from_offline_dict(kwargs["base_model_outputs"])
            if base_outputs.logits is None:
                out_hiddens = kwargs["base_model_outputs"]["base_model_hidden_states"]
                base_outputs.logits = self._base_model_lm_head(out_hiddens)
            target_hidden = base_outputs.target_hidden
            target_model_logits = base_outputs.logits
        else:
            # Call the inner base model directly (NOT super().forward(), which during
            # training runs the full DFlash pipeline). Compute target-model logits via
            # the lm_head — DSpark's TVD/confidence terms need the base distribution.
            with torch.no_grad():
                base_out = self._base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                target_model_logits = self._base_model_lm_head(base_out.last_hidden_state)
            offset = 1
            selected = [base_out.hidden_states[lid + offset] for lid in self.target_layer_ids]
            target_hidden = torch.cat(selected, dim=-1)  # [B, seq, num_layers * H]

        # 2. Build loss mask (same convention as DFlash/Domino).
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
            # Zero loss that still flows through all draft params for DDP sync.
            dummy = sum(p.sum() for p in self.dflash_module.parameters()) * 0.0
            return ModelOutput(loss=dummy, logits=None, train_acc=[[0.0]])

        # 4. Build draft inputs (inherited helpers).
        noise_embedding = self._build_noise_embedding(
            input_ids, anchor_positions, block_keep_mask, n_blocks
        )
        full_pos = self._build_position_ids(seq_len, anchor_positions, device)
        attn_mask = self._build_draft_attention_mask(
            seq_len, anchor_positions, block_keep_mask, n_blocks, target_hidden.dtype, device
        )

        # 5. Draft backbone forward.
        hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=full_pos,
            attention_mask=attn_mask,
        )

        # 6. Backbone logits → Markov correction → three-term loss.
        backbone_logits = self._base_model_lm_head(hidden).reshape(bsz, n_blocks, block_size, -1)
        final_logits, confidence_logits = self._apply_markov_head(
            hidden, backbone_logits, input_ids, anchor_positions, n_blocks
        )
        loss, accuracy, metrics = self._compute_dspark_loss(
            backbone_logits,
            final_logits,
            confidence_logits,
            input_ids,
            anchor_positions,
            block_keep_mask,
            loss_mask,
            target_model_logits,
        )

        return ModelOutput(loss=loss, logits=None, train_acc=[[accuracy]], dspark_metrics=metrics)

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
        if self.dflash_offline:
            raise RuntimeError(
                "DSpark offline model cannot run AR validation / pseudo_speculative_generate — "
                "base model layers were deleted during offline conversion. Reload the full "
                "base model before running AR validation."
            )
        model_output = self._base_model(input_ids=input_ids, output_hidden_states=True)
        base_logits = self._base_model_lm_head(model_output.last_hidden_state)
        base_token = base_logits[:, -1:, :].argmax(dim=-1).to(input_ids.device)

        if steps < 1:
            return base_token, None

        hid_offset = 1
        selected = [model_output.hidden_states[lid + hid_offset] for lid in self.target_layer_ids]
        target_hidden = torch.cat(selected, dim=-1)

        block_size = self.dflash_block_size
        bsz = input_ids.shape[0]
        device = input_ids.device

        # Block input: anchor at position 0, mask tokens elsewhere (parallel backbone).
        block_ids = torch.full(
            (bsz, block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_ids[:, 0] = base_token.squeeze(-1)
        noise_embedding = self._base_model_embeddings(block_ids)

        ctx_len = target_hidden.shape[1]
        ctx_positions = torch.arange(ctx_len, device=device)
        block_positions = torch.arange(ctx_len, ctx_len + block_size, device=device)
        pos_ids = torch.cat([ctx_positions, block_positions]).unsqueeze(0).expand(bsz, -1)

        draft_hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=None,
        )
        backbone_logits = self._base_model_lm_head(draft_hidden)  # [B, block_size, V]

        # Autoregressive Markov sampling over the block.
        m = self.dflash_module
        num_tokens = min(steps, block_size)
        prev_token = base_token.squeeze(-1)  # anchor precedes block position 0
        state = None
        draft_tokens = []
        for k in range(num_tokens):
            bias, state = m.markov_step(prev_token, draft_hidden[:, k, :], state)
            tok = (backbone_logits[:, k, :] + bias).argmax(dim=-1)
            draft_tokens.append(tok)
            prev_token = tok
        return base_token, torch.stack(draft_tokens, dim=1)
