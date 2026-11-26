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

"""Turn rollout records into the keyword arguments of ``HFDFlashModel.forward``.

Each training block is one drafting round of the rollout, with the rows the previous verify rejected
as its memory. The generation API's tokens are authoritative: the engine can stop a request after the
worker committed more tokens, so a record is cut to the API output. This module must not import vLLM.
"""

from collections.abc import Sequence

import torch

from .records import RolloutRecord

__all__ = ["build_batch"]


def build_batch(
    records: Sequence[RolloutRecord],
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    *,
    seq_len: int,
    num_anchors: int,
    block_size: int,
    aux_layer_ids: Sequence[int],
    memory: bool,
    generator: torch.Generator,
) -> dict:
    """Build one micro-batch: B = ``len(records)`` sequences padded to ``seq_len``.

    Args:
        records: validated records, one per request.
        sequences: each request's prompt + output token ids as returned by the generation API.
        prompt_lens: each request's prompt length in ``sequences``.
        num_anchors: K, the rounds kept per sequence, drawn uniformly at random when there are more
            rounds with a label (anchor < L - 1).
        aux_layer_ids: the drafter's target layers, in the order its ``fc`` reads them.
        memory: whether the rollout drafter read the rejected rows. When False the memory kwargs are
            left out, so the rounds train exactly as without memory (CONTRACT section 1).
        generator: CPU generator for the round draw.
    """
    if not len(records) == len(sequences) == len(prompt_lens) > 0:
        raise ValueError("records, sequences and prompt_lens must be non-empty and equally long")
    bsz, num_mem = len(records), block_size - 1
    aux_width = records[0].tensors["ctx_aux_hidden"].shape[1]
    hidden_size = records[0].tensors["ctx_last_hidden"].shape[1]
    input_ids = torch.zeros(bsz, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(bsz, seq_len, dtype=torch.long)
    loss_mask = torch.zeros(bsz, seq_len, dtype=torch.long)
    aux = torch.zeros(bsz, seq_len, aux_width, dtype=torch.bfloat16)
    last = torch.zeros(bsz, seq_len, hidden_size, dtype=torch.bfloat16)
    anchor_pos = torch.zeros(bsz, num_anchors, dtype=torch.long)
    anchor_valid = torch.zeros(bsz, num_anchors, dtype=torch.bool)
    if memory:
        mem_count = torch.zeros(bsz, num_anchors, dtype=torch.int32)
        mem_aux = torch.zeros(bsz, num_anchors, num_mem, aux_width, dtype=torch.bfloat16)

    for b, (rec, seq, prompt_len) in enumerate(zip(records, sequences, prompt_lens)):
        tokens = torch.as_tensor(seq, dtype=torch.long)
        length = tokens.numel()
        rounds = _check_and_select(
            rec, tokens, prompt_len, seq_len, block_size, aux_layer_ids, memory
        )
        if rounds.numel() > num_anchors:
            pick = torch.randperm(rounds.numel(), generator=generator)[:num_anchors]
            rounds = rounds[pick.sort().values]
        rows = min(rec.tensors["ctx_aux_hidden"].shape[0], length)
        input_ids[b, :length] = tokens
        attention_mask[b, :length] = 1
        loss_mask[b, prompt_len:length] = 1
        aux[b, :rows] = rec.tensors["ctx_aux_hidden"][:rows]
        last[b, :rows] = rec.tensors["ctx_last_hidden"][:rows]
        n = rounds.numel()
        anchor_pos[b, :n] = rec.tensors["round_anchor_pos"][rounds]
        anchor_valid[b, :n] = True
        if memory:
            mem_count[b, :n] = rec.mem_count[rounds].to(torch.int32)
            for k, t in enumerate(rounds.tolist()):
                _, rejected = rec.mem_rows(t)
                mem_aux[b, k, : rejected.shape[0]] = rejected

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "base_model_outputs": {
            "aux_hidden_states": aux,
            "base_model_hidden_states": last,
            "base_hidden_prenorm": False,
        },
        "rejmem_anchor_pos": anchor_pos,
        "rejmem_anchor_valid": anchor_valid,
    }
    if memory:
        # All-zero counts would train the same rounds only up to summation order, and cost more.
        batch["rejmem_mem_count"], batch["rejmem_mem_aux_hidden"] = mem_count, mem_aux
    return batch


def _check_and_select(
    rec, tokens, prompt_len, seq_len, block_size, aux_layer_ids, memory
) -> torch.Tensor:
    """Check ``rec`` against the API output; return the indices of its rounds that have a label."""
    t, m = rec.tensors, rec.meta
    tag, length = m["tag"], tokens.numel()
    if m["prompt_len"] != prompt_len or not prompt_len < length <= seq_len:
        raise ValueError(
            f"record {tag}: prompt_len {m['prompt_len']} / API prompt {prompt_len} / API length "
            f"{length} do not fit a training sequence of {seq_len}"
        )
    committed = t["token_ids"]
    if committed.numel() < length or not torch.equal(committed[:length].long(), tokens):
        raise ValueError(f"record {tag}: committed tokens disagree with the API output")
    # The bonus token after the last verify has no target row yet, every earlier one must.
    if t["ctx_aux_hidden"].shape[0] < length - 1:
        raise ValueError(
            f"record {tag}: features for {t['ctx_aux_hidden'].shape[0]} rows, the API output "
            f"needs {length - 1}"
        )
    if m["num_spec_tokens"] != block_size - 1:
        raise ValueError(
            f"record {tag}: {m['num_spec_tokens']} drafts per round, the drafter trains blocks "
            f"of {block_size}"
        )
    # Same width for other layers of equal count: only the ids tell a layer-numbering mismatch.
    if list(m["aux_layer_ids"]) != list(aux_layer_ids):
        raise ValueError(
            f"record {tag}: features of target layers {m['aux_layer_ids']}, the drafter reads "
            f"layers {list(aux_layer_ids)}"
        )
    if bool(m["memory_consumer"]) != memory:
        raise ValueError(
            f"record {tag}: rollout memory_consumer={m['memory_consumer']}, trainer memory={memory}"
        )
    return torch.nonzero(t["round_anchor_pos"] < length - 1).flatten()
