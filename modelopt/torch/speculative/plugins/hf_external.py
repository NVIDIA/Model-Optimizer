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

import copy
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


def _train_acc(acc):
    """Shape a scalar accuracy the way ``EagleTrainerWithAccLog`` consumes it.

    The callback averages the collected values and then iterates them as
    ``[parallel_draft][ttt_step]``, so a scalar has to arrive nested twice over; a
    flatter value makes that iteration raise. This mode has one draft and one step.
    """
    return [[acc.item()]]


@ExternalDraftDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFExternalDraftModel(ExternalDraftModel):
    """An external draft model for HuggingFace models."""

    def modify(self, config):
        """Configure the draft and validate that it can score against this base."""
        super().modify(config)

    def attach_base_lm_head(self, base_lm_head, base_final_norm=None, device=None) -> None:
        """Attach the base lm_head used to project cached hidden states to teacher logits.

        Copies before freezing. The modules passed in belong to the base model and its
        head may be tied to its embeddings, so setting ``requires_grad=False`` on them
        would silently freeze the base's output head for anything that trains it later.
        Copying here rather than on first use also keeps it out of the training step,
        where it would otherwise run on every rank mid-forward and briefly hold two
        copies of a vocab x hidden matrix.
        """
        head = copy.deepcopy(base_lm_head)
        norm = copy.deepcopy(base_final_norm) if base_final_norm is not None else None
        if device is not None:
            head = head.to(device)
            norm = norm.to(device) if norm is not None else None
        for module in (head, norm):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
        object.__setattr__(self, "_base_lm_head", head)
        object.__setattr__(self, "_base_final_norm", norm)

    def _teacher_logits(self, hidden_states, base_model_outputs):
        """Project dumped base hidden states through the base lm_head."""
        if getattr(self, "_base_lm_head", None) is None:
            raise RuntimeError(
                "Base lm_head is not attached; call attach_base_lm_head() before training. "
                "Teacher logits cannot be derived from the dumped hidden states without it."
            )
        h = hidden_states
        # These are our own copies (see attach_base_lm_head), so moving them is safe and
        # reaches nothing the caller holds. The draft's module tree does not contain them,
        # so .to(device) on the draft never moves them either. Each is checked on its own
        # device: they can be attached from different places, and a norm left behind would
        # fail inside _maybe_apply_base_final_norm rather than here.
        if self._base_lm_head.weight.device != h.device:
            object.__setattr__(self, "_base_lm_head", self._base_lm_head.to(h.device))
        norm = getattr(self, "_base_final_norm", None)
        if norm is not None:
            # A norm built with elementwise_affine=False has no parameters at all, and one
            # whose scale is a buffer has none either; probing with a bare next() would
            # raise StopIteration inside the forward and surface as an unrelated
            # "generator raised StopIteration".
            probe = next(norm.parameters(), None)
            if probe is None:
                probe = next(norm.buffers(), None)
            if probe is not None and probe.device != h.device:
                object.__setattr__(self, "_base_final_norm", norm.to(h.device))
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
        teacher_topk_tok=None,
        teacher_topk_prob=None,
        **kwargs,
    ):
        """Score the draft against cached base hidden states.

        Only ``base_model_hidden_states`` is consumed; ``aux_hidden_states`` exists in the
        dump for the head-based modes and is not used here.
        """
        sparse = teacher_topk_tok is not None
        if not sparse and base_model_outputs is None:
            raise ValueError(
                "External draft training expects offline base_model_outputs in the batch; "
                "online training against a live base model is not supported."
            )
        draft_out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if loss_mask is None:
            # Derive from attention_mask so padding is not trained on, matching the
            # other speculative plugins; all-ones only when there is no mask at all.
            # Both paths default identically: the sparse collator always supplies a mask,
            # but when it does not, training on padding is as wrong here as it is there.
            loss_mask = (
                attention_mask.clone().detach()
                if attention_mask is not None
                else torch.ones_like(input_ids)
            ).to(draft_out.logits.dtype)
        if sparse:
            loss, acc = self.compute_sparse_loss(
                draft_out.logits, teacher_topk_tok, teacher_topk_prob, loss_mask
            )
        else:
            base_logits = self._teacher_logits(
                base_model_outputs["base_model_hidden_states"], base_model_outputs
            )
            loss, acc = self.compute_loss(draft_out.logits, base_logits, loss_mask)
        draft_out.loss = loss
        if acc is not None:
            draft_out.train_acc = _train_acc(acc)
        return draft_out

    def save_pretrained(
        self,
        save_directory,
        is_main_process=True,
        state_dict=None,
        push_to_hub=False,
        *args,
        **kwargs,
    ):
        """Save as an ordinary HF checkpoint.

        ``transformers`` writes ``type(self).__name__`` into ``config.architectures``,
        which after conversion is the registry's dynamic subclass name (for example
        ``ExternalDraftLlamaForCausalLM``). Nothing downstream resolves that: vLLM and
        TRT-LLM look the string up in their own model registries and fail to load the
        checkpoint. Conversion adds no modules to the draft -- it is still the pretrained
        causal LM it started as -- so the original class name is the accurate one, and
        restoring it is what keeps the promise that this mode exports a plain HF model.
        """
        # transformers uploads config.json from inside super().save_pretrained, before the
        # correction below can run, so the published copy would carry the dynamic subclass
        # name this override exists to remove. Refuse rather than publish a checkpoint
        # nothing downstream can load.
        if push_to_hub:
            raise ValueError(
                "push_to_hub=True is not supported for an external draft: config.json is "
                "uploaded before this mode can restore the original architecture name, so "
                "the pushed checkpoint would be unloadable. Save locally first, then upload "
                "the corrected directory."
            )
        # Every parameter up to push_to_hub is named rather than swallowed by *args:
        # transformers' own callers pass these positionally, and a kwargs lookup would
        # then see the default -- is_main_process would be True on every rank, and the
        # push_to_hub guard above would never fire. The order mirrors
        # PreTrainedModel.save_pretrained for the pinned range, which takes no
        # save_function argument.
        result = super().save_pretrained(
            save_directory, is_main_process, state_dict, push_to_hub, *args, **kwargs
        )
        original = self.original_cls.__name__
        self.config.architectures = [original]
        # transformers rewrites architectures during the save, so the correction has to
        # come after it. Only the rank that wrote the file may rewrite it: under FSDP
        # every rank enters save_pretrained, and a read-modify-write from all of them
        # races on a shared filesystem.
        if is_main_process:
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
                f"vocab_size={draft_vocab_size}, base vocab_size={int(base_vocab_size)}. "
                "Set external_vocab_swap=true to re-index the draft onto the base "
                "vocabulary, which rebuilds its embeddings and lm_head."
            )

    def validate_tokenizer_against_base(self, draft_tokenizer, base_tokenizer):
        """Check the two tokenizers agree on what the ids mean, not just how many there are.

        Equal ``vocab_size`` is not enough: different tokenizers are routinely padded to
        the same width, and training against misaligned columns yields a falling loss
        curve and no acceptance.

        Compares the whole vocabulary rather than a low-id sample. Byte-level BPE
        tokenizers place the 256 byte symbols at ids 0-255 in the same order, so two
        genuinely different tokenizers agree on exactly the region a sample would check.
        """
        base_vocab, draft_vocab = base_tokenizer.get_vocab(), draft_tokenizer.get_vocab()
        # Both directions: walking only the base misses a draft that carries every base
        # token at the right id and adds more of its own, whose ids still land on base
        # columns that mean something else.
        mismatched = {
            token for token, bid in base_vocab.items() if draft_vocab.get(token) != bid
        } | {token for token, did in draft_vocab.items() if base_vocab.get(token) != did}
        if mismatched:
            raise ValueError(
                f"External draft and base tokenizers disagree on {len(mismatched)} tokens "
                f"(base has {len(base_vocab)}, draft has {len(draft_vocab)}). They are "
                "different tokenizers, and training would optimise against misaligned columns."
            )

    @staticmethod
    def _deployment_probs(logits, top_k, top_p):
        """Probabilities after temperature-1 top-k then top-p, renormalised.

        When top-k is active the nucleus runs on the k kept values rather than on a
        full sort of the vocabulary: at most k entries are non-zero after the first
        stage, so sorting all of V allocates several [B, S, V] temporaries -- values,
        int64 indices, cumsum and mask -- to order a tail that is entirely zero.
        """
        p = torch.softmax(logits.float(), dim=-1)
        vocab = p.shape[-1]
        if top_k and top_k < vocab:
            vals, idx = p.topk(top_k, dim=-1)
            # Renormalise before the nucleus: serving applies top-p to the
            # redistributed top-k mass, so skipping this makes top-p a no-op
            # whenever the top-k mass is already below it.
            vals = vals / vals.sum(-1, keepdim=True).clamp_min(1e-9)
            if top_p and top_p < 1.0:
                vals = vals * ((vals.cumsum(-1) - vals) < top_p)
                vals = vals / vals.sum(-1, keepdim=True).clamp_min(1e-9)
            return torch.zeros_like(p).scatter_(-1, idx, vals)
        if top_p and top_p < 1.0:
            srt, idx = p.sort(dim=-1, descending=True)
            keep = (srt.cumsum(-1) - srt) < top_p
            p = p * torch.zeros_like(keep).scatter(-1, idx, keep)
        return p / p.sum(-1, keepdim=True).clamp_min(1e-9)

    def _draft_probs(self, draft_logits):
        """The draft's distribution under the configured objective.

        ``tvd_deploy`` puts the draft through the serving filter so the loss is the
        acceptance the deployed pair will see; the other objectives use it unfiltered.
        Shared so the two loss paths cannot drift apart on which filter they applied.
        """
        if self.external_loss == "tvd_deploy":
            return self._deployment_probs(draft_logits, self.external_top_k, self.external_top_p)
        return torch.softmax(draft_logits.float(), dim=-1)

    @staticmethod
    def _top1_accuracy(teacher_top1, draft_logits, mask):
        """Fraction of scored positions where the draft's argmax matches the teacher's."""
        with torch.no_grad():
            valid = mask.bool()
            correct = (teacher_top1 == draft_logits.detach().argmax(dim=-1)) & valid
            return correct.sum().float() / valid.sum().clamp_min(1).float()

    def _mix_ce(self, tvd, draft_logits, target, mask):
        """``tvd_ce``: add CE on ``target``, since bounded TVD ranks in-support tokens weakly."""
        ce = torch.nn.functional.cross_entropy(
            draft_logits.float().flatten(0, 1), target.flatten(), reduction="none"
        ).view_as(mask)
        ce = (ce * mask).sum() / (mask.sum() + 1e-5)
        return self.external_tvd_alpha * tvd + self.external_ce_alpha * ce

    def _tvd_sparse(self, draft_logits, topk_tok, topk_prob):
        """Plain TVD against the stored teacher policy, over the whole vocabulary.

        The draft is not renormalised over the teacher's support, so probability
        it places outside that support counts as error. The stored policy is the
        teacher's deployment distribution and is exactly zero off its own support,
        which makes the out-of-support term simply ``1 - sum(q_topk)``.
        """
        p = topk_prob / topk_prob.sum(-1, keepdim=True).clamp_min(1e-9)
        q = self._draft_probs(draft_logits).gather(-1, topk_tok)
        # A ragged dump is padded to a rectangular width, and the pad slots must not
        # charge the draft again for a token the row already holds. Pads identify
        # themselves: both builder paths zero-initialise, so a padded slot is exactly
        # one with zero stored probability. Keying on that rather than on id equality
        # needs no assumption about builder ordering, and leaves a genuinely repeated
        # id alone -- an id-equality collapse would drop its second occurrence from q
        # while p kept both entries.
        q = q * (topk_prob > 0)
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
        loss = (self._tvd_sparse(draft, tok, prob) * mask).sum() / denom
        # A row the dump left empty has no teacher top-1: its argmax is slot 0, i.e.
        # whatever the zero-fill holds. CE and accuracy both skip those rows.
        teacher_top1 = tok.gather(-1, prob.argmax(-1, keepdim=True)).squeeze(-1)
        has_mass = mask * (prob.sum(-1) > 0)
        if self.external_loss == "tvd_ce":
            loss = self._mix_ce(loss, draft, teacher_top1, has_mass)
        accuracy = None
        if self.external_report_acc:
            accuracy = self._top1_accuracy(teacher_top1, draft, has_mass)
        return loss, accuracy

    def compute_loss(self, draft_logits, base_logits, loss_mask, base_predict_tok=None):
        """Compute the configured objective against the base model's distribution.

        Args:
            loss_mask: per-token mask; response-only when the dump used ``--answer-only-loss``.

        Returns:
            ``(loss, accuracy)``; accuracy is ``None`` unless ``external_report_acc`` is set.
        """
        seq_len = draft_logits.shape[1]
        if base_logits.shape[1] < seq_len:
            raise ValueError(
                f"dumped base logits cover {base_logits.shape[1]} positions but the draft was "
                f"given {seq_len}; the dump and the training input disagree in length."
            )
        base_logits = base_logits[:, :seq_len]
        mask = loss_mask[:, :seq_len].to(draft_logits.dtype)
        denom = mask.sum() + 1e-5

        if self.external_loss in ("tvd", "tvd_deploy", "tvd_ce"):
            if self.external_loss == "tvd_deploy":
                # Both sides through the serving filter, so the loss is the
                # acceptance the deployed pair will actually see.
                p = self._deployment_probs(base_logits, self.external_top_k, self.external_top_p)
            else:
                p = torch.softmax(base_logits.float(), dim=-1)
            q = self._draft_probs(draft_logits)
            loss = (0.5 * (p - q).abs().sum(-1) * mask).sum() / denom
            if self.external_loss == "tvd_ce":
                loss = self._mix_ce(loss, draft_logits, base_logits.argmax(-1), mask)
        else:
            # Identical in form to hf_eagle._eagle_loss so switching modes does not
            # silently change the objective.
            base_softmax = torch.softmax(base_logits.float(), dim=-1)
            draft_logsoft = torch.log_softmax(draft_logits.float(), dim=-1)
            loss = -torch.sum(mask.unsqueeze(-1) * base_softmax * draft_logsoft) / denom

        accuracy = None
        if self.external_report_acc:
            if base_predict_tok is None:
                base_predict_tok = base_logits.argmax(dim=-1)
            accuracy = self._top1_accuracy(base_predict_tok, draft_logits, mask)

        return loss, accuracy
