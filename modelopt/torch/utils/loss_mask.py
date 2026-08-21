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

"""Model-specific recovery of the assistant loss mask.

The standard way to build an answer-only loss mask is
``apply_chat_template(..., return_assistant_tokens_mask=True)``, which maps the
``{% generation %}`` template span to tokens via ``char_to_token`` -- and that is
only available on "fast" tokenizers. Some models ship only a slow/Python tokenizer
and cannot use this path.

This module is a small registry of per-model fallbacks that recover the mask
directly from token ids, keyed by a ``detect`` predicate. Data paths consult
:func:`get_loss_mask_recovery` and stay free of any single model's chat-format
details.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["LossMaskRecovery", "get_loss_mask_recovery", "register_loss_mask_recovery"]


@dataclass(frozen=True)
class LossMaskRecovery:
    """A model-specific fallback for building the assistant loss mask.

    Args:
        name: Identifier for the target model family (for logging/debugging).
        detect: Returns ``True`` if this recovery applies to the given tokenizer.
        compute: Maps ``(tokenizer, input_ids)`` to a ``(seq_len,)`` ``LongTensor``
            mask aligned to ``input_ids`` (1 on tokens that should contribute to
            the loss, 0 otherwise).
    """

    name: str
    detect: Callable[[object], bool]
    compute: Callable[[object, torch.Tensor], torch.Tensor]


_RECOVERIES: list[LossMaskRecovery] = []


def register_loss_mask_recovery(recovery: LossMaskRecovery) -> None:
    """Register a model-specific loss-mask recovery."""
    _RECOVERIES.append(recovery)


def get_loss_mask_recovery(tokenizer) -> LossMaskRecovery | None:
    """Return the first registered recovery whose ``detect`` matches ``tokenizer``."""
    for recovery in _RECOVERIES:
        if recovery.detect(tokenizer):
            return recovery
    return None


# ---------------------------------------------------------------------------
# Kimi
#
# Kimi ships only a Python (tiktoken) tokenizer, so it cannot emit assistant masks
# via apply_chat_template. Its chat turns are rendered as
#   <|im_{role}|> {role_name} <|im_middle|> {content} <|im_end|>
# so the assistant content sits between <|im_middle|> and <|im_end|>.
# ---------------------------------------------------------------------------

_KIMI_ROLE_MARKERS = ("<|im_user|>", "<|im_assistant|>", "<|im_system|>")

# Kimi-K3's XTML structural tags (see the Kimi-K3 section below). Declared here
# because ``_kimi_detect`` uses them to hand K3 tokenizers off to the K3 recovery.
_K3_MARKERS = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")


def _has_all_tokens(tokenizer, tokens) -> bool:
    """Whether ``tokenizer`` maps every one of ``tokens`` to a real (non-unk) id."""
    unk = getattr(tokenizer, "unk_token_id", None)
    try:
        ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    except Exception:
        return False
    return all(i is not None and i != unk for i in ids)


def _kimi_detect(tokenizer) -> bool:
    """Whether ``tokenizer`` defines Kimi's chat role markers as real tokens.

    K3 keeps the K2 ``<|im_*|>`` markers for back-compat but *also* defines the XTML
    structural tags that classic Kimi lacks. When those are present the tokenizer is
    K3, so defer to the ``kimi_k3`` recovery whose ``compute`` understands the XTML
    turn layout; matching here would silently produce an empty mask.
    """
    if not _has_all_tokens(tokenizer, (*_KIMI_ROLE_MARKERS, "<|im_middle|>", "<|im_end|>")):
        return False
    return not _has_all_tokens(tokenizer, _K3_MARKERS)


def _kimi_compute(tokenizer, input_ids) -> torch.Tensor:
    """Recover the assistant-content mask from already-tokenized Kimi chat ids.

    Marks only the ``{content}`` span (between ``<|im_middle|>`` and ``<|im_end|>``,
    both exclusive). This matches the ``{% generation %}`` span used for fast
    tokenizers: the role header and the trailing ``<|im_end|>`` are not masked.
    """
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    assistant_id = tokenizer.convert_tokens_to_ids("<|im_assistant|>")
    middle_id = tokenizer.convert_tokens_to_ids("<|im_middle|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    role_ids = {tokenizer.convert_tokens_to_ids(t) for t in _KIMI_ROLE_MARKERS}

    n = len(ids)
    mask = [0] * n
    i = 0
    while i < n:
        if ids[i] != assistant_id:
            i += 1
            continue
        # Skip the role header (role_name) up to its <|im_middle|> separator.
        j = i + 1
        while j < n and ids[j] != middle_id and ids[j] not in role_ids and ids[j] != end_id:
            j += 1
        if j >= n or ids[j] != middle_id:
            # Malformed turn (no content separator) or a trailing generation prompt.
            i = j
            continue
        # Mark the content span [middle + 1, end): excludes <|im_middle|> and <|im_end|>.
        start = j + 1
        k = start
        while k < n and ids[k] != end_id and ids[k] not in role_ids:
            k += 1
        for t in range(start, k):
            mask[t] = 1
        i = k

    return torch.tensor(mask, dtype=torch.long)


register_loss_mask_recovery(
    LossMaskRecovery(name="kimi", detect=_kimi_detect, compute=_kimi_compute)
)


# ---------------------------------------------------------------------------
# Kimi-K3
#
# K3 replaces the K2/K2.5 <|im_*|> turn markers with an XTML tag format:
#   <|open|> {tag} {attrs..} <|sep|> {content} <|close|> {tag} <|sep|>  [<|end_of_msg|>]
# Tag names and attribute values (including the role) are ORDINARY text tokens;
# only open/close/sep/end_of_msg are special tokens. An assistant turn reads
#   <|open|> message role assistant <|sep|> {content} <|close|> message <|sep|>
# and its content may nest further tags (``think``, ``response``), so the content
# span is found by tracking open/close depth rather than by scanning for the next
# marker.
# ---------------------------------------------------------------------------


def _k3_detect(tokenizer) -> bool:
    """Whether ``tokenizer`` defines K3's XTML structural markers as real tokens."""
    return _has_all_tokens(tokenizer, _K3_MARKERS)


def _k3_compute(tokenizer, input_ids) -> torch.Tensor:
    """Recover the assistant-content mask from already-tokenized K3 chat ids.

    Marks the content of each ``message role=assistant`` turn -- from the token after
    the message-open ``<|sep|>`` up to (excluding) the matching ``<|close|>`` -- which
    includes the nested ``think``/``response`` sub-tags the model generates. Role
    headers, other roles, and the turn-closing tokens stay unmasked, matching the
    ``{% generation %}`` span a fast tokenizer would report.
    """
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    open_id = tokenizer.convert_tokens_to_ids("<|open|>")
    close_id = tokenizer.convert_tokens_to_ids("<|close|>")
    sep_id = tokenizer.convert_tokens_to_ids("<|sep|>")

    n = len(ids)
    mask = [0] * n
    i = 0
    while i < n:
        if ids[i] != open_id:
            i += 1
            continue
        # The header tokens sit between this <|open|> and its <|sep|>.
        j = i + 1
        while j < n and ids[j] != sep_id:
            j += 1
        if j >= n:
            break
        header = tokenizer.decode(ids[i + 1 : j]).lower()
        if "message" not in header or "assistant" not in header:
            i = j + 1
            continue
        # Content = [after this <|sep|>, matching <|close|>), skipping nested tags.
        start = j + 1
        depth = 1
        k = start
        while k < n:
            if ids[k] == open_id:
                depth += 1
            elif ids[k] == close_id:
                depth -= 1
                if depth == 0:
                    break
            k += 1
        for t in range(start, k):
            mask[t] = 1
        i = k + 1

    return torch.tensor(mask, dtype=torch.long)


register_loss_mask_recovery(
    LossMaskRecovery(name="kimi_k3", detect=_k3_detect, compute=_k3_compute)
)
