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

"""Tests for re-indexing an external draft onto the base model's vocabulary."""

import pytest
import torch

from modelopt.torch.speculative.external.vocab_swap import swap_draft_vocabulary

transformers = pytest.importorskip("transformers")

HIDDEN = 32


class _FakeTok:
    def __init__(self, vocab, specials=None, added=None):
        self._v = vocab
        # added tokens are deliberately absent from get_vocab(), as they are for several
        # fast tokenizers -- that is the case that used to erase a real id
        self._added = added or {}
        for attr, tok in (specials or {}).items():
            setattr(self, attr, tok)

    def get_vocab(self):
        return self._v

    def convert_tokens_to_ids(self, token):
        if token in self._added:
            return self._added[token]
        return self._v.get(token)


def _causal_lm(hidden=HIDDEN, vocab_size=64):
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


def test_vocab_swap_carries_over_matching_tokens_and_reports_coverage():
    """Rows for tokens both tokenizers spell the same must survive the swap, and the
    frequent-id coverage is the number worth looking at."""
    draft = _causal_lm(hidden=HIDDEN, vocab_size=8)
    before = draft.get_input_embeddings().weight.data.clone()
    stats = swap_draft_vocabulary(
        draft,
        _FakeTok({"a": 0, "b": 1, "c": 2, "zz": 7}),
        _FakeTok({"c": 0, "a": 1, "q": 2, "b": 3}),
        4,
    )

    assert stats["matched"] == 3  # a, b, c; 'q' has no draft row
    assert stats["frequent_coverage"] == 0.75
    assert draft.config.vocab_size == 4
    after = draft.get_input_embeddings().weight.data
    assert torch.allclose(after[0], before[2])  # base 'c' <- draft id 2
    assert torch.allclose(after[1], before[0])  # base 'a' <- draft id 0
    assert torch.allclose(after[3], before[1])  # base 'b' <- draft id 1
    assert not torch.allclose(after[2], before[0])  # 'q' freshly initialised


def test_untied_lm_head_is_rebuilt_from_its_own_rows():
    """An untied output head must carry over *head* rows, not input-embedding rows."""
    draft = _causal_lm(vocab_size=8)
    draft.config.tie_word_embeddings = False
    # Make the head plainly distinct from the embeddings so a mix-up is visible.
    draft.get_output_embeddings().weight.data.normal_(mean=5.0, std=0.01)
    head_before = draft.get_output_embeddings().weight.data.clone()
    embed_before = draft.get_input_embeddings().weight.data.clone()

    swap_draft_vocabulary(draft, _FakeTok({"a": 0, "b": 1}), _FakeTok({"b": 0, "a": 1}), 4)

    head_after = draft.get_output_embeddings().weight.data
    assert torch.allclose(head_after[0], head_before[1])  # base 'b' <- draft head row 1
    assert torch.allclose(head_after[1], head_before[0])  # base 'a' <- draft head row 0
    assert not torch.allclose(head_after[0], embed_before[1]), (
        "the output head must not be overwritten with input-embedding rows"
    )


def test_swap_runs_when_sizes_match_but_tokenizers_differ():
    """The case the swap exists for: padded to the same width, different token ids."""
    draft = _causal_lm(vocab_size=4)
    before = draft.get_input_embeddings().weight.data.clone()
    # same size, different mapping
    stats = swap_draft_vocabulary(draft, _FakeTok({"a": 0, "b": 1}), _FakeTok({"b": 0, "a": 1}), 4)
    assert stats["matched"] == 2
    after = draft.get_input_embeddings().weight.data
    assert torch.allclose(after[0], before[1]), "base 'b' must take the draft's 'b' row"
    assert torch.allclose(after[1], before[0]), "base 'a' must take the draft's 'a' row"


def test_special_token_ids_follow_the_base_vocabulary():
    """The checkpoint is saved with the base tokenizer, so its ids must match it."""
    draft = _causal_lm(vocab_size=8)
    draft.config.eos_token_id = 2  # a draft-vocabulary id

    class _Tok(_FakeTok):
        eos_token = "</s>"
        bos_token = None
        pad_token = None
        unk_token = None

    base = _Tok({"a": 0, "</s>": 3})
    swap_draft_vocabulary(draft, _FakeTok({"a": 0, "</s>": 2}), base, 4)
    assert draft.config.eos_token_id == 3, "eos must point at the base tokenizer's id"


def test_unmatched_embedding_and_head_rows_are_independent():
    """The two fills must not share a normal stream.

    Both draws ask for the same shape, so one seed for both gives every unmatched row
    ``head_i . emb_i`` positive in expectation -- a raised self-logit on exactly the ids
    the draft knows nothing about, biasing it toward proposing them. Determinism needs
    the seed fixed, not shared.
    """
    torch.manual_seed(0)
    draft = _causal_lm(vocab_size=4)  # LlamaConfig defaults to an untied head
    tok_d, tok_b = _FakeTok({"x": 0}), _FakeTok({"x": 0, "q": 1, "r": 2, "s": 3})
    swap_draft_vocabulary(draft, tok_d, tok_b, 4)

    emb = draft.get_input_embeddings().weight.data
    head = draft.get_output_embeddings().weight.data
    # rows 1..3 are unmatched, so they came from the statistical fill
    dots = (emb[1:] * head[1:]).sum(-1)
    shared = (emb[1:] * emb[1:]).sum(-1)  # what a shared stream would look like
    assert not torch.allclose(dots, shared), "head and embedding fills share a stream"


def test_unmatched_rows_are_reproducible_across_calls():
    """The draw is seeded, so every rank fills unmatched rows identically."""
    a = _causal_lm(vocab_size=4)
    b = _causal_lm(vocab_size=4)
    b.get_input_embeddings().weight.data.copy_(a.get_input_embeddings().weight.data)
    tok_d, tok_b = _FakeTok({"x": 0}), _FakeTok({"x": 0, "q": 1})
    swap_draft_vocabulary(a, tok_d, tok_b, 4)
    swap_draft_vocabulary(b, tok_d, tok_b, 4)
    assert torch.allclose(
        a.get_input_embeddings().weight.data, b.get_input_embeddings().weight.data
    )


def test_special_tokens_are_never_overwritten_with_none():
    """A base without a pad_token, or whose eos is an added token, must not erase the draft's.

    ``eos_token_id = None`` on the exported draft means it never signals end-of-sequence,
    so every consumer generates to ``max_new_tokens``.
    """
    draft = _causal_lm(vocab_size=4)
    draft.config.eos_token_id = 3
    draft.config.pad_token_id = 2
    base = _FakeTok(
        {"a": 0, "b": 1},
        specials={"eos_token": "<|eot_id|>"},  # present, but an added token
        added={"<|eot_id|>": 1},
    )
    swap_draft_vocabulary(draft, _FakeTok({"a": 0}), base, 4)
    assert draft.config.eos_token_id == 1, "an added-token eos must resolve, not erase"
    assert draft.config.pad_token_id == 2, "a base without pad_token must leave the draft's"
