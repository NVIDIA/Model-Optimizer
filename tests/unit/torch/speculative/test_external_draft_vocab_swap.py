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
    def __init__(self, vocab):
        self._v = vocab

    def get_vocab(self):
        return self._v


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
