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

"""Support for training a standalone (external) draft model with HuggingFace.

Unlike Eagle/Medusa/DFlash, the draft here is an ordinary pretrained causal LM
that owns its embeddings and lm_head. The base model is not modified and nothing
is grafted onto it: it contributes only per-token hidden states, which are
projected by the base lm_head to obtain teacher logits.

``external_loss`` selects the objective: ``soft_ce`` matches the Eagle offline
objective, ``tvd`` is total-variation distance between the two distributions,
``tvd_deploy`` applies the serving filter to both sides first so the loss is the
acceptance the deployed pair sees, and ``tvd_ce`` adds a cross-entropy ranking
term to ``tvd``.
"""

import json
from pathlib import Path

import torch
from transformers import PreTrainedModel

from ..external.conversion import ExternalDraftDMRegistry
from ..external.external_model import ExternalDraftModel
from .modeling_final_norm import _maybe_apply_base_final_norm

# Objectives with a sparse implementation: they read only the teacher's stored
# top-k policy. soft_ce needs full teacher logits and so cannot run on that data.
# main.py gates data.sparse_data_path on this, so it must be defined beside the
# implementations rather than duplicated there.
# On the sparse path the base's truncation is baked into the dump, so both TVD
# variants are computable there; they differ only in whether the *draft* is put
# through the serving filter too. soft_ce needs full base logits and is dense-only.
SPARSE_CAPABLE_LOSSES = ("tvd", "tvd_deploy", "tvd_ce")

__all__ = ["SPARSE_CAPABLE_LOSSES", "HFExternalDraftModel"]


@ExternalDraftDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFExternalDraftModel(ExternalDraftModel):
    """An external draft model for HuggingFace models."""

    def modify(self, config):
        """Configure the draft and validate that it can score against this base."""
        super().modify(config)

        if self.external_loss not in ("soft_ce", "tvd", "tvd_deploy", "tvd_ce"):
            raise ValueError(
                f"external_loss must be one of 'soft_ce', 'tvd', 'tvd_deploy', "
                f"'tvd_ce', got {self.external_loss!r}."
            )

    def attach_base_lm_head(self, base_lm_head, base_final_norm=None) -> None:
        """Attach the frozen base lm_head used to project cached hidden states to teacher logits."""
        for p in base_lm_head.parameters():
            p.requires_grad = False
        if base_final_norm is not None:
            for p in base_final_norm.parameters():
                p.requires_grad = False
        object.__setattr__(self, "_base_lm_head", base_lm_head)
        object.__setattr__(self, "_base_final_norm", base_final_norm)

    def _teacher_logits(self, hidden_states, base_model_outputs):
        """Project dumped base hidden states through the base lm_head."""
        if getattr(self, "_base_lm_head", None) is None:
            raise RuntimeError(
                "Base lm_head is not attached; call attach_base_lm_head() before training. "
                "Teacher logits cannot be derived from the dumped hidden states without it."
            )
        h = hidden_states
        # The base modules are held off the draft's module tree so they stay out of
        # its state_dict, which also means .to(device) never reaches them.
        if self._base_lm_head.weight.device != h.device:
            self._base_lm_head.to(h.device)
            if getattr(self, "_base_final_norm", None) is not None:
                self._base_final_norm.to(h.device)
        # Shared with the EAGLE/DFlash offline forwards: raises rather than feeding an
        # un-normed hidden into lm_head when the producer captured a pre-norm hidden.
        h = _maybe_apply_base_final_norm(
            h, base_model_outputs, getattr(self, "_base_final_norm", None)
        )
        h = h.to(self._base_lm_head.weight.dtype)
        with torch.no_grad():
            return self._base_lm_head(h)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        loss_mask=None,
        base_model_outputs=None,
        labels=None,
        teacher_topk_tok=None,
        teacher_topk_prob=None,
        **kwargs,
    ):
        """Score the draft against cached base hidden states.

        Only ``base_model_hidden_states`` is consumed; ``aux_hidden_states`` exists in the
        dump for the head-based modes and is not used here.
        """
        if teacher_topk_tok is not None:
            draft_out = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            if loss_mask is None:
                loss_mask = torch.ones_like(input_ids, dtype=draft_out.logits.dtype)
            loss, acc = self.compute_sparse_loss(
                draft_out.logits, teacher_topk_tok, teacher_topk_prob, loss_mask
            )
            draft_out.loss = loss
            if acc is not None:
                object.__setattr__(self, "_last_accuracy", acc.detach())
            return draft_out

        if base_model_outputs is None:
            raise ValueError(
                "External draft training expects offline base_model_outputs in the batch; "
                "online training against a live base model is not supported."
            )
        draft_out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        base_logits = self._teacher_logits(
            base_model_outputs["base_model_hidden_states"], base_model_outputs
        )
        if loss_mask is None:
            loss_mask = torch.ones(
                input_ids.shape, dtype=draft_out.logits.dtype, device=draft_out.logits.device
            )
        loss, acc = self.compute_loss(draft_out.logits, base_logits, loss_mask)
        draft_out.loss = loss
        if acc is not None:
            # EagleTrainerWithAccLog logs outputs.train_acc; anything else is dropped.
            draft_out.train_acc = [acc.detach()]
        return draft_out

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save as an ordinary HF checkpoint.

        ``transformers`` writes ``type(self).__name__`` into ``config.architectures``,
        which after conversion is the registry's dynamic subclass name (for example
        ``ExternalDraftLlamaForCausalLM``). Nothing downstream resolves that: vLLM and
        TRT-LLM look the string up in their own model registries and fail to load the
        checkpoint. Conversion adds no modules to the draft -- it is still the pretrained
        causal LM it started as -- so the original class name is the accurate one, and
        restoring it is what keeps the promise that this mode exports a plain HF model.
        """
        result = super().save_pretrained(save_directory, *args, **kwargs)
        original = self.original_cls.__name__
        self.config.architectures = [original]
        # transformers rewrites architectures during the save, so the correction has to
        # come after it. Only the rank that wrote the file may rewrite it: under FSDP
        # every rank enters save_pretrained, and a read-modify-write from all of them
        # races on a shared filesystem.
        if kwargs.get("is_main_process", True):
            config_file = Path(save_directory) / "config.json"
            if config_file.exists():
                config = json.loads(config_file.read_text())
                if config.get("architectures") != [original]:
                    config["architectures"] = [original]
                    config_file.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        return result

    def validate_against_base(self, base_vocab_size: int) -> None:
        """Check the draft indexes the same vocabulary as the base model.

        A mismatch cannot be caught downstream: training would optimise against
        misaligned columns and yield a plausible loss curve with no acceptance.
        """
        draft_vocab_size = int(self.config.vocab_size)
        if draft_vocab_size != int(base_vocab_size):
            raise ValueError(
                "External draft and base model must share a vocabulary: draft "
                f"vocab_size={draft_vocab_size}, base vocab_size={int(base_vocab_size)}."
            )

    def validate_tokenizer_against_base(self, draft_tokenizer, base_tokenizer, sample: int = 256):
        """Check the two tokenizers agree on what the ids mean, not just how many there are.

        Equal ``vocab_size`` is not enough: different tokenizers are routinely padded to
        the same width, and training against misaligned columns yields a falling loss
        curve and no acceptance.
        """
        base_vocab, draft_vocab = base_tokenizer.get_vocab(), draft_tokenizer.get_vocab()
        checked = mismatched = 0
        for token, base_id in base_vocab.items():
            if base_id >= sample:
                continue
            checked += 1
            if draft_vocab.get(token) != base_id:
                mismatched += 1
        if checked and mismatched:
            raise ValueError(
                f"External draft and base tokenizers disagree on {mismatched}/{checked} of "
                "the first ids despite matching vocab_size. They are different tokenizers, "
                "and training would optimise against misaligned columns."
            )

    @staticmethod
    def _deployment_probs(logits, top_k, top_p):
        """Probabilities after temperature-1 top-k then top-p, renormalised."""
        p = torch.softmax(logits.float(), dim=-1)
        if top_k and top_k < p.shape[-1]:
            kth = p.topk(top_k, dim=-1).values[..., -1:]
            p = p * (p >= kth)
            # Renormalise before the nucleus: serving applies top-p to the
            # redistributed top-k mass, so skipping this makes top-p a no-op
            # whenever the top-k mass is already below it.
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        if top_p and top_p < 1.0:
            srt, idx = p.sort(dim=-1, descending=True)
            keep = (srt.cumsum(-1) - srt) < top_p
            p = p * torch.zeros_like(keep).scatter(-1, idx, keep)
        return p / p.sum(-1, keepdim=True).clamp_min(1e-9)

    def _tvd_sparse(self, draft_logits, topk_tok, topk_prob):
        """Plain TVD against the stored teacher policy, over the whole vocabulary.

        The draft is not renormalised over the teacher's support, so probability
        it places outside that support counts as error. The stored policy is the
        teacher's deployment distribution and is exactly zero off its own support,
        which makes the out-of-support term simply ``1 - sum(q_topk)``.
        """
        p = topk_prob / topk_prob.sum(-1, keepdim=True).clamp_min(1e-9)
        if self.external_loss == "tvd_deploy":
            q_full = self._deployment_probs(draft_logits, self.external_top_k, self.external_top_p)
        else:
            q_full = torch.softmax(draft_logits.float(), dim=-1)
        q = q_full.gather(-1, topk_tok)
        # q sums to 1 over its own support, so the mass it placed off the base's
        # support is whatever is left over.
        outside = (1.0 - q.sum(-1)).clamp_min(0.0)
        return 0.5 * ((p - q).abs().sum(-1) + outside)

    def compute_sparse_loss(self, draft_logits, topk_tok, topk_prob, loss_mask):
        """``tvd`` against a sparse teacher policy; returns ``(loss, accuracy)``.

        The sparse policy is indexed by the token it produced: position ``t`` holds the
        distribution the base sampled ``input_ids[t]`` from, conditioned on ``< t``. The
        draft's logits at ``t`` are conditioned on ``<= t`` and predict ``t + 1``, so the
        draft is shifted one position left before the two are compared. Dropping the shift
        still yields a falling loss curve -- it just trains against the wrong target.
        """
        seq = min(draft_logits.shape[1], topk_tok.shape[1])
        draft = draft_logits[:, : seq - 1]
        tok, prob = topk_tok[:, 1:seq], topk_prob[:, 1:seq]
        mask = loss_mask[:, 1:seq].to(draft_logits.dtype)
        denom = mask.sum() + 1e-5
        tvd = self._tvd_sparse(draft, tok, prob)
        loss = (tvd * mask).sum() / denom
        if self.external_loss == "tvd_ce":
            # TVD is bounded, so it is weak on which token inside the base's
            # support should win; cross-entropy's log penalty is not.
            target = tok.gather(-1, prob.argmax(-1, keepdim=True)).squeeze(-1)
            ce = torch.nn.functional.cross_entropy(
                draft.reshape(-1, draft.shape[-1]).float(),
                target.reshape(-1),
                reduction="none",
            ).reshape(mask.shape)
            loss = self.external_tvd_alpha * loss + self.external_ce_alpha * (
                (ce * mask).sum() / denom
            )

        accuracy = None
        if self.external_report_acc:
            with torch.no_grad():
                teacher_top1 = tok.gather(-1, prob.argmax(-1, keepdim=True)).squeeze(-1)
                valid = mask.bool()
                correct = (teacher_top1 == draft.detach().argmax(-1)) & valid
                accuracy = correct.sum().float() / valid.sum().clamp_min(1).float()
        return loss, accuracy

    def compute_loss(self, draft_logits, base_logits, loss_mask, base_predict_tok=None):
        """Compute the configured objective against the base model's distribution.

        Args:
            loss_mask: per-token mask; response-only when the dump used ``--answer-only-loss``.

        Returns:
            ``(loss, accuracy)``; accuracy is ``None`` unless ``external_report_acc`` is set.
        """
        seq_len = draft_logits.shape[1]
        base_logits = base_logits[:, :seq_len]
        mask = loss_mask[:, :seq_len].to(draft_logits.dtype)
        denom = mask.sum() + 1e-5

        if self.external_loss in ("tvd", "tvd_deploy", "tvd_ce"):
            if self.external_loss == "tvd_deploy":
                # Both sides through the serving filter, so the loss is the
                # acceptance the deployed pair will actually see.
                p = self._deployment_probs(base_logits, self.external_top_k, self.external_top_p)
                q = self._deployment_probs(draft_logits, self.external_top_k, self.external_top_p)
            else:
                p = torch.softmax(base_logits.float(), dim=-1)
                q = torch.softmax(draft_logits.float(), dim=-1)
            loss = (0.5 * (p - q).abs().sum(-1) * mask).sum() / denom
            if self.external_loss == "tvd_ce":
                ce = torch.nn.functional.cross_entropy(
                    draft_logits.reshape(-1, draft_logits.shape[-1]).float(),
                    base_logits.reshape(-1, base_logits.shape[-1]).argmax(-1),
                    reduction="none",
                ).reshape(mask.shape)
                loss = self.external_tvd_alpha * loss + self.external_ce_alpha * (
                    (ce * mask).sum() / denom
                )
        else:
            # Identical in form to hf_eagle._eagle_loss so switching modes does not
            # silently change the objective.
            base_softmax = torch.softmax(base_logits.float(), dim=-1)
            draft_logsoft = torch.log_softmax(draft_logits.float(), dim=-1)
            loss = -torch.sum(mask.unsqueeze(-1) * base_softmax * draft_logsoft) / denom

        accuracy = None
        if self.external_report_acc:
            with torch.no_grad():
                if base_predict_tok is None:
                    base_predict_tok = base_logits.argmax(dim=-1)
                valid = mask.bool()
                correct = (base_predict_tok == draft_logits.detach().argmax(dim=-1)) & valid
                accuracy = correct.sum().float() / valid.sum().clamp_min(1).float()

        return loss, accuracy
