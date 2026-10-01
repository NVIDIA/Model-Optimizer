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

"""End-to-end offline training path for the ``external`` speculative decoding mode."""

import pytest
import torch

import modelopt.torch.speculative as mtsp

transformers = pytest.importorskip("transformers")

VOCAB = 64
DRAFT_HIDDEN = 32
BASE_HIDDEN = 48
BATCH = 2
SEQ = 6


def _causal_lm(hidden, vocab_size=VOCAB):
    cfg = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(cfg)


def _external_draft(**overrides):
    """A converted draft with the base lm_head attached, as main.py wires it."""
    base = _causal_lm(BASE_HIDDEN)
    draft = _causal_lm(DRAFT_HIDDEN)
    mtsp.convert(draft, [("external", {"external_offline": True, **overrides})])
    draft.validate_against_base(base.config.vocab_size)
    draft.attach_base_lm_head(base.get_output_embeddings(), base.model.norm)
    return draft, base


def _offline_batch():
    """A batch shaped like SpeculativeDecodingOfflineCollator output."""
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(0, VOCAB, (BATCH, SEQ)),
        "attention_mask": torch.ones(BATCH, SEQ, dtype=torch.long),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]], dtype=torch.float),
        "base_model_outputs": {
            "base_model_hidden_states": torch.randn(BATCH, SEQ, BASE_HIDDEN),
            "aux_hidden_states": torch.randn(BATCH, SEQ, BASE_HIDDEN * 3),
        },
    }


@pytest.mark.parametrize("loss_fn", ["soft_ce", "tvd_deploy"])
def test_training_steps_reduce_loss(loss_fn):
    """Highest-level check: the real offline path optimises its objective.

    Subsumes the forward-shape and finite-loss checks -- a broken forward cannot
    produce eight decreasing steps.
    """
    draft, _ = _external_draft(external_loss=loss_fn)
    batch = _offline_batch()
    opt = torch.optim.AdamW(draft.parameters(), lr=1e-3)

    first = draft(**batch).loss.item()
    assert draft(**batch).logits.shape == (BATCH, SEQ, VOCAB)
    for _ in range(8):
        opt.zero_grad()
        draft(**batch).loss.backward()
        opt.step()
    last = draft(**batch).loss.item()
    assert last < first, f"{loss_fn} did not decrease: {first:.4f} -> {last:.4f}"


def test_gradient_reaches_draft_but_not_base():
    """The draft trains; the frozen base head must never accumulate gradient."""
    draft, base = _external_draft()
    draft(**_offline_batch()).loss.backward()

    grads = [p.grad for p in draft.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradient reached the draft"
    assert base.get_output_embeddings().weight.grad is None, "gradient leaked into the base lm_head"


def test_forward_ignores_aux_hidden_states():
    """External drafts never read aux_hidden_states; its absence must not break them."""
    draft, _ = _external_draft()
    batch = _offline_batch()
    del batch["base_model_outputs"]["aux_hidden_states"]
    assert torch.isfinite(draft(**batch).loss)


def test_missing_base_lm_head_raises():
    """Training without the base head must fail loudly, not silently skip the target."""
    draft = _causal_lm(DRAFT_HIDDEN)
    mtsp.convert(draft, [("external", {"external_offline": True})])
    with pytest.raises(RuntimeError, match="attach_base_lm_head"):
        draft(**_offline_batch())


def test_online_batch_rejected():
    draft, _ = _external_draft()
    batch = _offline_batch()
    del batch["base_model_outputs"]
    with pytest.raises(ValueError, match="base_model_outputs"):
        draft(**batch)


def test_base_lm_head_absent_from_draft_state_dict():
    """The exported draft must not carry base weights."""
    draft, _ = _external_draft()
    keys = draft.state_dict().keys()
    assert not any("_base_lm_head" in k or "_base_final_norm" in k for k in keys)


def test_dense_tvd_uses_full_vocabulary_and_the_filter_changes_it():
    """The dense path reconstructs the base's whole distribution from hidden states,
    so ``tvd`` is a TVD over the full vocabulary; ``tvd_deploy`` filters both sides."""
    batch = _offline_batch()
    plain, _ = _external_draft(external_loss="tvd")
    deploy, _ = _external_draft(external_loss="tvd_deploy")

    hidden = batch["base_model_outputs"]["base_model_hidden_states"]
    with torch.no_grad():
        base_logits = plain._teacher_logits(hidden, batch["base_model_outputs"])
    assert base_logits.shape[-1] == VOCAB, "teacher logits must span the full vocabulary"

    a = plain(**batch).loss
    b = deploy(**batch).loss
    assert 0.0 <= float(a) <= 1.0, f"TVD out of range: {a}"
    assert not torch.allclose(a, b), "the serving filter must change the loss"


def test_prenorm_without_a_norm_raises_rather_than_corrupting():
    """A producer declaring a pre-norm hidden with no norm attached must fail loudly.

    Silently skipping the norm feeds an un-normed residual to lm_head, which yields a
    plausible loss curve against a garbage teacher.
    """
    base = _causal_lm(BASE_HIDDEN)
    draft = _causal_lm(DRAFT_HIDDEN)
    mtsp.convert(draft, [("external", {"external_offline": True, "external_loss": "tvd"})])
    # No norm located for this base, which is what the buggy lookup produced.
    draft.attach_base_lm_head(base.get_output_embeddings(), None)
    hidden = torch.randn(1, 4, BASE_HIDDEN)
    with pytest.raises(RuntimeError, match="base_hidden_prenorm"):
        draft._teacher_logits(hidden, {"base_hidden_prenorm": True})
