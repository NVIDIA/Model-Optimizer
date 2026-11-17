# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch

import modelopt.torch.utils.loss_mask as lm
from modelopt.torch.utils.loss_mask import (
    LossMaskRecovery,
    get_loss_mask_recovery,
    register_loss_mask_recovery,
)

# Fixed ids for Kimi's chat markers; content/role-name tokens use a disjoint range.
_MARKER_IDS = {
    "<|im_user|>": 1,
    "<|im_assistant|>": 2,
    "<|im_system|>": 3,
    "<|im_middle|>": 4,
    "<|im_end|>": 5,
}
_ROLE_NAME = 50  # stand-in token for the role name between marker and <|im_middle|>


class FakeKimiTokenizer:
    """Minimal tokenizer exposing only what the Kimi recovery touches."""

    is_fast = False
    unk_token_id = 999

    def convert_tokens_to_ids(self, token):
        return _MARKER_IDS.get(token, self.unk_token_id)


class FakeOtherTokenizer:
    """A tokenizer without Kimi's markers (everything maps to unk)."""

    is_fast = False
    unk_token_id = 999

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id


def _turn(role_id, content_ids):
    # <|im_{role}|> {role_name} <|im_middle|> {content} <|im_end|>
    return [
        role_id,
        _ROLE_NAME,
        _MARKER_IDS["<|im_middle|>"],
        *content_ids,
        _MARKER_IDS["<|im_end|>"],
    ]


@pytest.fixture
def restore_registry():
    """Snapshot/restore the global registry so register() tests don't leak."""
    saved = list(lm._RECOVERIES)
    yield
    lm._RECOVERIES[:] = saved


def test_kimi_recovery_is_registered():
    recovery = get_loss_mask_recovery(FakeKimiTokenizer())
    assert recovery is not None
    assert recovery.name == "kimi"


def test_no_recovery_for_unknown_tokenizer():
    assert get_loss_mask_recovery(FakeOtherTokenizer()) is None


def test_kimi_mask_marks_only_assistant_content():
    tok = FakeKimiTokenizer()
    ids = (
        _turn(_MARKER_IDS["<|im_system|>"], [200])  # system
        + _turn(_MARKER_IDS["<|im_user|>"], [201])  # user
        + _turn(_MARKER_IDS["<|im_assistant|>"], [300, 301])  # assistant content 300,301
        + _turn(_MARKER_IDS["<|im_user|>"], [202])  # user
        + _turn(_MARKER_IDS["<|im_assistant|>"], [302])  # assistant content 302
    )
    input_ids = torch.tensor(ids, dtype=torch.long)

    recovery = get_loss_mask_recovery(tok)
    mask = recovery.compute(tok, input_ids)

    assert mask.shape == input_ids.shape
    assert mask.dtype == torch.long
    # Exactly the assistant-content tokens are marked.
    marked = {i for i, m in enumerate(mask.tolist()) if m == 1}
    expected = {i for i, v in enumerate(ids) if v in (300, 301, 302)}
    assert marked == expected
    # Role markers, role name, <|im_middle|> and <|im_end|> are never marked.
    for i, v in enumerate(ids):
        if v in (*_MARKER_IDS.values(), _ROLE_NAME):
            assert mask[i] == 0


def test_kimi_mask_ignores_trailing_generation_prompt():
    # A trailing "<|im_assistant|> name <|im_middle|>" with no content/end (e.g. a
    # generation prompt) must not be masked.
    tok = FakeKimiTokenizer()
    ids = [
        *_turn(_MARKER_IDS["<|im_user|>"], [201]),
        _MARKER_IDS["<|im_assistant|>"],
        _ROLE_NAME,
        _MARKER_IDS["<|im_middle|>"],
    ]
    mask = get_loss_mask_recovery(tok).compute(tok, torch.tensor(ids, dtype=torch.long))
    assert int(mask.sum()) == 0


def test_kimi_mask_ignores_assistant_turn_without_middle():
    # A malformed assistant turn (role marker but no <|im_middle|> separator) carries no
    # identifiable content span, so nothing is masked and the next turn still parses.
    tok = FakeKimiTokenizer()
    # assistant marker closed by <|im_end|> with no content separator, then a valid turn.
    malformed = [_MARKER_IDS["<|im_assistant|>"], _ROLE_NAME, _MARKER_IDS["<|im_end|>"]]
    ids = [*malformed, *_turn(_MARKER_IDS["<|im_assistant|>"], [300])]
    mask = get_loss_mask_recovery(tok).compute(tok, torch.tensor(ids, dtype=torch.long))
    marked = {i for i, m in enumerate(mask.tolist()) if m == 1}
    assert marked == {i for i, v in enumerate(ids) if v == 300}


def test_kimi_mask_accepts_list_input():
    tok = FakeKimiTokenizer()
    ids = _turn(_MARKER_IDS["<|im_assistant|>"], [300, 301])
    mask = get_loss_mask_recovery(tok).compute(tok, ids)  # plain list, not a tensor
    assert mask.tolist() == [0, 0, 0, 1, 1, 0]


def test_register_and_lookup_custom_recovery(restore_registry):
    sentinel = object()

    def detect(tok):
        return getattr(tok, "marker", None) is sentinel

    recovery = LossMaskRecovery(name="dummy", detect=detect, compute=lambda t, x: x)
    register_loss_mask_recovery(recovery)

    class Tok:
        marker = sentinel

    assert get_loss_mask_recovery(Tok()) is recovery
    # A tokenizer that matches nothing still returns None.
    assert get_loss_mask_recovery(object()) is None


# Kimi-K3: XTML structural markers are special tokens; tag names/attributes are text.
_K3_IDS = {"<|open|>": 10, "<|close|>": 11, "<|sep|>": 12, "<|end_of_msg|>": 13}
_K3_TEXT = {
    20: "message",
    21: ' role="user"',
    22: ' role="assistant"',
    23: "think",
    24: "response",
    25: ' name="a"',
}
_OPEN, _CLOSE, _SEP, _EOM = _K3_IDS.values()


class FakeK3Tokenizer:
    is_fast = False
    unk_token_id = 999

    def convert_tokens_to_ids(self, token):
        return _K3_IDS.get(token, self.unk_token_id)

    def decode(self, ids):
        return "".join(_K3_TEXT.get(i, f"<{i}>") for i in ids)


def _k3_tag(open_id, *name_ids):
    return [open_id, *name_ids, _SEP]


def _k3_user(content):
    return [*_k3_tag(_OPEN, 20, 21), *content, *_k3_tag(_CLOSE, 20), _EOM]


def _k3_assistant(reasoning, content, attrs=(22,)):
    return [
        *_k3_tag(_OPEN, 20, *attrs),
        *_k3_tag(_OPEN, 23),
        *reasoning,
        *_k3_tag(_CLOSE, 23),
        *_k3_tag(_OPEN, 24),
        *content,
        *_k3_tag(_CLOSE, 24),
        *_k3_tag(_CLOSE, 20),
        _EOM,
    ]


def test_k3_recovery_is_registered():
    assert get_loss_mask_recovery(FakeK3Tokenizer()).name == "kimi_k3"
    assert get_loss_mask_recovery(FakeKimiTokenizer()).name == "kimi"


def test_k3_mask_marks_generated_tokens():
    # Marked: everything after the first channel opener through <|end_of_msg|>, in every
    # assistant turn (named ones too). The header and <|open|>think<|sep|> are prompt.
    tok = FakeK3Tokenizer()
    first = _k3_assistant([300], [301])
    second = _k3_assistant([302], [303], attrs=(22, 25))
    ids = [*_k3_user([200]), *first, *_k3_user([201]), *second]
    mask = get_loss_mask_recovery(tok).compute(tok, torch.tensor(ids))

    prompt_len = len(_k3_tag(_OPEN, 20, 22)) + len(_k3_tag(_OPEN, 23))
    off1 = len(_k3_user([200]))
    off2 = off1 + len(first) + len(_k3_user([201]))
    expected = set(range(off1 + prompt_len, off1 + len(first)))
    expected |= set(range(off2 + prompt_len + 1, off2 + len(second)))  # +1: name attr
    assert {i for i, m in enumerate(mask.tolist()) if m} == expected


def test_k3_mask_truncated_turn_and_generation_prompt():
    tok = FakeK3Tokenizer()
    compute = get_loss_mask_recovery(tok).compute
    # Truncated mid-reasoning: marked to the end of the sequence.
    truncated = [*_k3_tag(_OPEN, 20, 22), *_k3_tag(_OPEN, 23), 300, 301]
    assert compute(tok, truncated).tolist()[-2:] == [1, 1]
    # A trailing generation prompt carries no generated tokens.
    gen_prompt = [*_k3_user([200]), *_k3_tag(_OPEN, 20, 22), *_k3_tag(_OPEN, 23)]
    assert int(compute(tok, gen_prompt).sum()) == 0
