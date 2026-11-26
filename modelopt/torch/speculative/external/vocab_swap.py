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


def _retarget_special_tokens(draft, base_tokenizer):
    """Point the draft's special-token ids at the base tokenizer's ids for the same tokens.

    Only ever overwrites with an id that resolved. Writing ``None`` through would be the
    damaging case: a base with no ``pad_token`` is ordinary, and ``eos_token_id = None``
    on the exported draft means it never signals end-of-sequence, so every consumer
    generates to ``max_new_tokens``.
    """
    unk_id = getattr(base_tokenizer, "unk_token_id", None)
    for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token = getattr(base_tokenizer, attr, None)
        if not isinstance(token, str):
            continue
        # convert_tokens_to_ids, not get_vocab: for several fast tokenizers get_vocab()
        # omits added tokens, and chat-template markers such as <|eot_id|> are added
        # tokens -- looking those up in the raw vocab yields None and erases a real id.
        new_id = base_tokenizer.convert_tokens_to_ids(token)
        if new_id is None or (attr != "unk_token" and unk_id is not None and new_id == unk_id):
            continue
        for target in (draft.config, getattr(draft, "generation_config", None)):
            if target is not None and hasattr(target, f"{attr}_id"):
                setattr(target, f"{attr}_id", new_id)


def _fill_from_statistics(old_weight, rows: int, seed: int = 0):
    """Rows drawn from ``old_weight``'s per-column statistics, in fp32 and seeded.

    ``seed`` must differ between the embeddings and the head. Both draws request the
    same shape, so a shared seed hands them an identical normal stream and every
    unmatched row gets ``head_i . emb_i`` positive in expectation -- a systematically
    raised self-logit on exactly the ids the draft has learned nothing about. Fixing
    the seed is what buys cross-rank determinism; sharing it is not.
    """
    stats = old_weight.float()
    mean, std = stats.mean(dim=0), stats.std(dim=0).clamp_min(1e-6)
    generator = torch.Generator(device=old_weight.device).manual_seed(seed)
    draw = torch.normal(
        mean.expand(rows, stats.shape[1]),
        std.expand(rows, stats.shape[1]),
        generator=generator,
    )
    return draw.to(old_weight.dtype)


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

    base_vocab = base_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()

    # Fill unmatched rows from the draft's own distribution rather than zeros: a
    # zero row is a uniform logit over the hidden dimension and trains slowly.
    # The statistics are accumulated in fp32: a bf16 sum over tens of thousands of
    # rows saturates, and the std (a sum of squares) worst of all. The draw is seeded
    # so every rank fills the unmatched rows identically.
    new_weight = _fill_from_statistics(old_weight, base_vocab_size)

    # The output head is a different learned matrix from the input embeddings when the
    # two are untied, so it has to be rebuilt from its own rows -- copying embeddings
    # over it would discard the pretrained head entirely.
    old_head = draft.get_output_embeddings()
    untied = old_head is not None and not draft.config.tie_word_embeddings
    if untied:
        # Built from the head's own row width, not the embeddings': they coincide on
        # standard drafts, but assuming it turns a resized head into a confusing
        # shape error at the copy below rather than a named failure here.
        old_head_weight = old_head.weight.data
        new_head_weight = _fill_from_statistics(old_head_weight, base_vocab_size, seed=1)
        # A head with a bias has one scalar per token. resize keeps those at their old
        # ids while the rows move, so a matched token would inherit another token's
        # bias. Llama and Qwen heads carry none, which is why this is easy to miss.
        old_head_bias = None if old_head.bias is None else old_head.bias.data
        if old_head_bias is not None:
            new_head_bias = _fill_from_statistics(
                old_head_bias.unsqueeze(-1), base_vocab_size, seed=2
            ).squeeze(-1)

    # Collect the pairs first, then move the rows in two gathers. A per-token tensor
    # assignment costs a kernel launch each, which at a 128k base vocabulary is 128k of
    # them before training starts.
    pairs = [
        (base_id, draft_id)
        for token, base_id in base_vocab.items()
        if base_id < base_vocab_size
        and (draft_id := draft_vocab.get(token)) is not None
        and draft_id < old_weight.shape[0]
    ]
    matched = len(pairs)
    frequent_matched = sum(1 for base_id, _ in pairs if base_id < 50000)
    if pairs:
        device = old_weight.device
        base_ids = torch.tensor([b for b, _ in pairs], dtype=torch.long, device=device)
        draft_ids = torch.tensor([d for _, d in pairs], dtype=torch.long, device=device)
        new_weight.index_copy_(0, base_ids, old_weight.index_select(0, draft_ids))
        if untied:
            # The head can be narrower than the embeddings on a draft whose head was
            # resized on its own, so it keeps its own bound rather than reusing theirs.
            keep = draft_ids < old_head_weight.shape[0]
            new_head_weight.index_copy_(
                0, base_ids[keep], old_head_weight.index_select(0, draft_ids[keep])
            )
            if old_head_bias is not None:
                new_head_bias.index_copy_(
                    0, base_ids[keep], old_head_bias.index_select(0, draft_ids[keep])
                )

    # resize_token_embeddings rewires config.vocab_size and re-ties the lm_head;
    # doing that by hand silently leaves a tied head pointing at the old matrix.
    # mean_resizing=False because every row it would compute is overwritten below,
    # and its multivariate-normal fit dominates the cost of the swap.
    draft.resize_token_embeddings(base_vocab_size, mean_resizing=False)
    draft.get_input_embeddings().weight.data.copy_(new_weight)
    if untied:
        draft.get_output_embeddings().weight.data.copy_(new_head_weight)
        if old_head_bias is not None:
            draft.get_output_embeddings().bias.data.copy_(new_head_bias)

    # The special-token ids still index the draft's old vocabulary. The checkpoint is
    # saved with the base tokenizer, so leaving them would pair base tokenizer files
    # with draft ids and every consumer would stop on the wrong token.
    _retarget_special_tokens(draft, base_tokenizer)
    # resize keeps the old padding_idx when it is still in range, so it would point at
    # whatever the base spells at the draft's former pad id.
    embedding = draft.get_input_embeddings()
    if getattr(embedding, "padding_idx", None) is not None:
        embedding.padding_idx = draft.config.pad_token_id

    return {
        "matched": matched,
        "total": base_vocab_size,
        "frequent_coverage": frequent_matched / min(50000, base_vocab_size),
    }
