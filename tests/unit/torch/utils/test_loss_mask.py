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
