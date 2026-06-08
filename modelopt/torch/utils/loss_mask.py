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
details. It is intentionally minimal and is meant to seed a broader model-specific
patch registry.
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


def _kimi_detect(tokenizer) -> bool:
    """Whether ``tokenizer`` defines Kimi's chat role markers as real tokens."""
    unk = getattr(tokenizer, "unk_token_id", None)
    try:
        ids = [
            tokenizer.convert_tokens_to_ids(t)
            for t in (*_KIMI_ROLE_MARKERS, "<|im_middle|>", "<|im_end|>")
        ]
    except Exception:
        return False
    return all(i is not None and i != unk for i in ids)


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
