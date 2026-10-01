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

"""Re-index a pretrained draft onto the base model's vocabulary.

Speculative decoding requires the draft to propose tokens the target can verify,
so the two must share a vocabulary. A draft trained with a different tokenizer
can still be used once its input embeddings and lm_head are rebuilt against the
base's vocabulary: rows for tokens both tokenizers spell identically are carried
over, and the rest are initialised from the draft's own embedding statistics.

The replacement matrix is ``base_vocab_size * hidden_size`` parameters, a large
fraction of a small draft, and the swap costs acceptance. Prefer a
vocabulary-matched draft where one exists, and judge the result by an acceptance
measurement rather than the loss curve.
"""

import torch

__all__ = ["swap_draft_vocabulary"]


def swap_draft_vocabulary(draft, draft_tokenizer, base_tokenizer, base_vocab_size: int) -> dict:
    """Rebuild ``draft``'s embeddings against the base tokenizer's vocabulary.

    Must run *before* ``mtsp.convert`` so the converted module and the saved
    ModelOpt state describe the post-swap architecture; doing it afterwards leaves
    the checkpoint disagreeing with its config on ``vocab_size``.

    Args:
        base_vocab_size: taken from the base model's config rather than
            ``len(base_tokenizer)``, which disagrees whenever the config pads the
            embedding table.

    Returns:
        ``matched``, ``total``, and the fraction of the base's most frequent ids that
        carried over. These describe the swap, not how well the swapped draft will
        accept.
    """
    old_embed = draft.get_input_embeddings()
    old_weight = old_embed.weight.data
    hidden = old_weight.shape[1]

    base_vocab = base_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()

    # Fill unmatched rows from the draft's own distribution rather than zeros: a
    # zero row is a uniform logit over the hidden dimension and trains slowly.
    mean = old_weight.mean(dim=0)
    std = old_weight.std(dim=0)
    new_weight = torch.normal(
        mean.expand(base_vocab_size, hidden),
        std.expand(base_vocab_size, hidden).clamp_min(1e-6),
    ).to(old_weight.dtype)

    # The output head is a different learned matrix from the input embeddings when the
    # two are untied, so it has to be rebuilt from its own rows -- copying embeddings
    # over it would discard the pretrained head entirely.
    old_head = draft.get_output_embeddings()
    untied = old_head is not None and not draft.config.tie_word_embeddings
    if untied:
        old_head_weight = old_head.weight.data
        new_head_weight = torch.normal(
            old_head_weight.mean(dim=0).expand(base_vocab_size, hidden),
            old_head_weight.std(dim=0).expand(base_vocab_size, hidden).clamp_min(1e-6),
        ).to(old_head_weight.dtype)

    matched = 0
    frequent_matched = 0
    for token, base_id in base_vocab.items():
        if base_id >= base_vocab_size:
            continue
        draft_id = draft_vocab.get(token)
        if draft_id is None or draft_id >= old_weight.shape[0]:
            continue
        new_weight[base_id] = old_weight[draft_id]
        if untied and draft_id < old_head_weight.shape[0]:
            new_head_weight[base_id] = old_head_weight[draft_id]
        matched += 1
        if base_id < 50000:
            frequent_matched += 1

    # resize_token_embeddings rewires config.vocab_size and re-ties the lm_head;
    # doing that by hand silently leaves a tied head pointing at the old matrix.
    # mean_resizing=False because every row it would compute is overwritten below,
    # and its multivariate-normal fit dominates the cost of the swap.
    draft.resize_token_embeddings(base_vocab_size, mean_resizing=False)
    draft.get_input_embeddings().weight.data.copy_(new_weight)
    if untied:
        draft.get_output_embeddings().weight.data.copy_(new_head_weight)

    return {
        "matched": matched,
        "total": base_vocab_size,
        "frequent_coverage": frequent_matched / min(50000, base_vocab_size),
    }
