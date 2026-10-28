# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Read state-QAT boundaries from tokenized prompt metadata, never loss-mask values."""

from itertools import pairwise

import torch

__all__ = ["mask_prefix_labels", "phase_lengths"]


def mask_prefix_labels(batch, prefix, boundaries=None):
    """Mask labels predicted wholly inside each prefix.

    Megatron shifts labels by one position: input position ``p - 1`` predicts the
    first token after a prefix of length ``p`` and therefore remains trainable.
    """
    loss_mask = batch["loss_mask"].clone()
    if boundaries is None:
        for row, count in enumerate(prefix):
            loss_mask[row, : max(count - 1, 0)] = 0
    else:
        flat = loss_mask.view(-1)
        for start, count in zip(boundaries, prefix):
            flat[start : start + max(count - 1, 0)] = 0
    batch["loss_mask"] = loss_mask


def phase_lengths(batch, prefill_tokens=None):
    """Return prefix/valid lengths and optional packed storage boundaries.

    Custom collators provide ``prefill_lengths`` and ``sequence_lengths`` per
    segment. Bridge prompt-completion batches provide ``context_lengths`` and
    ``token_count``; the latter includes the extra next-token label.
    """
    packed = any(batch.get(key) is not None for key in ("cu_seqlens", "cu_seqlens_q"))
    if packed:
        if "prefill_lengths" not in batch or "sequence_lengths" not in batch:
            raise ValueError(
                "Packed state QAT requires prefill_lengths and sequence_lengths per segment"
            )
        prefix, valid = batch["prefill_lengths"], batch["sequence_lengths"]
        boundaries = batch.get("cu_seqlens_q_padded")
        if boundaries is None:
            boundaries = batch.get("cu_seqlens_q", batch.get("cu_seqlens"))
        boundaries = torch.as_tensor(boundaries).flatten().tolist()
        boundaries = [x for x in boundaries if x >= 0]
        widths = [end - start for start, end in pairwise(boundaries)]
    else:
        boundaries = None
        prefix = batch.get("prefill_lengths", batch.get("context_lengths"))
        valid = batch.get("sequence_lengths")
        if valid is None and "token_count" in batch:
            valid = torch.as_tensor(batch["token_count"]) - 1
        widths = [batch["tokens"].shape[-1]] * batch["tokens"].shape[0]
    if prefix is None or valid is None:
        raise ValueError(
            "State QAT needs explicit prompt and valid-token lengths from the collator"
        )
    prefix, valid = [torch.as_tensor(x).flatten().tolist() for x in (prefix, valid)]
    if prefill_tokens is not None:
        # Short SFT records still need one recurrent input position to expose the
        # state-QDQ error. Token-only fixed-length data is unchanged by this clamp.
        prefix = [min(prefill_tokens, max(length - 1, 0)) for length in valid]
    if len(prefix) != len(valid) or len(valid) != len(widths):
        raise ValueError("State QAT needs one prompt and valid length per sequence")
    if any(type(x) is not int for x in prefix + valid + widths) or any(
        not 0 <= p <= v <= width for p, v, width in zip(prefix, valid, widths)
    ):
        raise ValueError("State QAT requires integer lengths with 0 <= prefix <= valid <= storage")
    return prefix, valid, boundaries
