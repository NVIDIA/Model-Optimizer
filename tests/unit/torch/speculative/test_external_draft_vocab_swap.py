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
    draft.lm_head.bias = torch.nn.Parameter(torch.arange(8.0))  # Llama has none; others do
    head_before = draft.get_output_embeddings().weight.data.clone()
    bias_before = draft.get_output_embeddings().bias.data.clone()
    embed_before = draft.get_input_embeddings().weight.data.clone()

    swap_draft_vocabulary(draft, _FakeTok({"a": 0, "b": 1}), _FakeTok({"b": 0, "a": 1}), 4)

    head_after = draft.get_output_embeddings().weight.data
    assert torch.allclose(head_after[0], head_before[1])  # base 'b' <- draft head row 1
    assert torch.allclose(head_after[1], head_before[0])  # base 'a' <- draft head row 0
    assert not torch.allclose(head_after[0], embed_before[1]), (
        "the output head must not be overwritten with input-embedding rows"
    )
    # the bias is one scalar per token, so it has to move with the row it belongs to
    bias_after = draft.get_output_embeddings().bias.data
    assert bias_after[0] == bias_before[1] and bias_after[1] == bias_before[0], (
        "a matched token kept another token's bias"
    )


def test_unmatched_embedding_and_head_rows_are_independent():
    """Embedding and head fills must not share a normal stream."""
    draft = _causal_lm(vocab_size=4)  # LlamaConfig defaults to an untied head
    # identical statistics, so a shared seed would draw identical rows
    draft.get_output_embeddings().weight.data.copy_(draft.get_input_embeddings().weight.data)
    swap_draft_vocabulary(draft, _FakeTok({"x": 0}), _FakeTok({"x": 0, "q": 1, "r": 2}), 4)

    emb = draft.get_input_embeddings().weight.data
    head = draft.get_output_embeddings().weight.data
    assert not torch.allclose(emb[1:], head[1:]), "head and embedding fills share a stream"


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
        # eos is an added token; pad is set but unknown to the base
        specials={"eos_token": "<|eot_id|>", "pad_token": "<pad>"},
        added={"<|eot_id|>": 1},
    )
    swap_draft_vocabulary(draft, _FakeTok({"a": 0}), base, 4)
    assert draft.config.eos_token_id == 1, "an added-token eos must resolve, not erase"
    assert draft.config.pad_token_id == 2, "an unresolved pad_token must leave the draft's"
