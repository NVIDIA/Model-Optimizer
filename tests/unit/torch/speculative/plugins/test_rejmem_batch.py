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

import pytest
import torch

from modelopt.torch.speculative.plugins.rejmem.batch import build_batch
from modelopt.torch.speculative.plugins.rejmem.records import SCHEMA, RolloutRecord, validate_record

NS = 3  # drafts per round
BLOCK = NS + 1
HIDDEN = 4
AUX_LAYERS = [1, 2]
AUX = len(AUX_LAYERS) * HIDDEN
SEQ_LEN = 64
PROMPT_LEN = 5


def _record(accepts, *, memory=True, seed=0):
    """Spec decoding with the given accepted counts per verify, plus the final unverified round.

    Feature rows carry their position + 1 and memory rows their index + 1, so every copy is checkable.
    Returns the record and the worker's committed tokens.
    """
    g = torch.Generator().manual_seed(seed)

    def rand(n):
        return torch.randint(10, 1000, (n,), generator=g).tolist()

    tokens = rand(PROMPT_LEN + 1)  # the prompt and the token sampled by the prefill
    anchors, accepted, drafts, offsets, mem_tokens, rejected = [], [], [], [0], [], []
    anchor = PROMPT_LEN
    for k in [*accepts, -1]:
        proposal = rand(NS)
        anchors.append(anchor)
        accepted.append(k)
        drafts.append(proposal)
        mem_tokens += rejected  # this round's memory: what the previous verify rejected
        offsets.append(len(mem_tokens))
        if k >= 0:
            tokens += proposal[:k] + rand(1)
            rejected = proposal[k:]
            anchor += k + 1
    rows = len(tokens) - 1  # the last bonus token has not been through the target yet
    position = torch.arange(1, rows + 1, dtype=torch.bfloat16).unsqueeze(1)
    num_mem = len(mem_tokens)
    rec = RolloutRecord(
        tensors={
            "token_ids": torch.tensor(tokens, dtype=torch.int32),
            "ctx_aux_hidden": position.expand(rows, AUX).contiguous(),
            "ctx_last_hidden": -position.expand(rows, HIDDEN).contiguous(),
            "round_anchor_pos": torch.tensor(anchors),
            "round_index": torch.arange(len(anchors)),
            "round_num_accepted": torch.tensor(accepted, dtype=torch.int32),
            "round_draft_token_ids": torch.tensor(drafts, dtype=torch.int32),
            "round_mem_offset": torch.tensor(offsets),
            "mem_token_ids": torch.tensor(mem_tokens, dtype=torch.int32),
            "mem_aux_hidden": torch.arange(1, num_mem + 1, dtype=torch.bfloat16)
            .unsqueeze(1)
            .expand(num_mem, AUX)
            .contiguous(),
        },
        meta={
            "schema": SCHEMA,
            "tag": "t",
            "req_id": "r",
            "prompt_len": PROMPT_LEN,
            "num_spec_tokens": NS,
            "aux_layer_ids": AUX_LAYERS,
            "hidden_size": HIDDEN,
            "weight_version_start": 0,
            "weight_version_end": 0,
            "memory_consumer": memory,
            "num_rounds_total": len(anchors),
            "round_keep": "all",
        },
    )
    validate_record(rec)
    return rec, tokens


def _build(records, sequences, *, num_anchors=64, memory=True, seed=0):
    return build_batch(
        records,
        sequences,
        [PROMPT_LEN] * len(records),
        seq_len=SEQ_LEN,
        num_anchors=num_anchors,
        block_size=BLOCK,
        aux_layer_ids=AUX_LAYERS,
        memory=memory,
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("cut", [0, 4], ids=["finished-at-last-bonus", "engine-truncated"])
def test_build_batch_aligns_rounds_to_api_output(cut):
    """The API may end before the worker's last commit (EOS / max_tokens): only its tokens train."""
    rec, tokens = _record([1, 3, 0, 2, 3, 1])
    api = tokens[: len(tokens) - cut]
    length = len(api)
    kw = _build([rec], [api])

    assert kw["input_ids"][0, :length].tolist() == api
    assert not kw["input_ids"][0, length:].any()
    assert kw["attention_mask"][0].nonzero().flatten().tolist() == list(range(length))
    assert kw["loss_mask"][0].nonzero().flatten().tolist() == list(range(PROMPT_LEN, length))
    # Feature row p holds p + 1; rows past the API output or without a target forward are zero.
    rows = min(rec.tensors["ctx_aux_hidden"].shape[0], length)
    aux = kw["base_model_outputs"]["aux_hidden_states"][0, :, 0].float()
    assert aux.tolist() == list(range(1, rows + 1)) + [0.0] * (SEQ_LEN - rows)
    last = kw["base_model_outputs"]["base_model_hidden_states"][0, :, 0].float()
    assert torch.equal(last, -aux)
    assert kw["base_model_outputs"]["base_hidden_prenorm"] is False

    # Every round with a label (anchor < L - 1) is a block, in order, with its own memory rows.
    anchors = rec.tensors["round_anchor_pos"]
    expected = [t for t in range(anchors.numel()) if anchors[t] < length - 1]
    n = len(expected)
    assert kw["rejmem_anchor_valid"][0].tolist() == [True] * n + [False] * (64 - n)
    assert kw["rejmem_anchor_pos"][0, :n].tolist() == anchors[expected].tolist()
    assert kw["rejmem_mem_count"][0, :n].tolist() == rec.mem_count[expected].tolist()
    assert not kw["rejmem_mem_count"][0, n:].any()
    off = rec.tensors["round_mem_offset"]
    for k, t in enumerate(expected):
        count = int(rec.mem_count[t])
        mem = kw["rejmem_mem_aux_hidden"][0, k, :, 0].float()
        assert mem.tolist() == [float(off[t] + i + 1) for i in range(count)] + [0.0] * (NS - count)
    assert kw["rejmem_mem_aux_hidden"].shape == (1, 64, NS, AUX)


@pytest.mark.parametrize(
    "tamper",
    ["token", "longer_than_record", "prompt_len", "memory_flag", "num_spec_tokens", "aux_layers"],
)
def test_build_batch_rejects_inconsistent_record(tamper):
    rec, tokens = _record([2, 1, 3])
    _build([rec], [tokens])  # consistent as built

    memory = True
    if tamper == "token":
        tokens = [*tokens[:-2], tokens[-2] + 1, tokens[-1]]
    elif tamper == "longer_than_record":
        tokens = [*tokens, 7]
    elif tamper == "prompt_len":
        rec.meta["prompt_len"] = PROMPT_LEN + 1
    elif tamper == "memory_flag":
        memory = False
    elif tamper == "num_spec_tokens":
        rec.meta["num_spec_tokens"] = NS - 1
    else:  # as many layers, numbered differently: same feature width
        rec.meta["aux_layer_ids"] = [i + 1 for i in AUX_LAYERS]
    with pytest.raises(ValueError):
        _build([rec], [tokens], memory=memory)


def test_build_batch_draws_rounds_and_memory_switch():
    """More rounds than K: a seeded uniform draw, ascending; memory off keeps the anchors, drops rows."""
    accepts = [0, 1, 2, 3] * 5
    recs = [_record(accepts, seed=s) for s in range(2)]
    recs_off = [_record(accepts, memory=False, seed=s) for s in range(2)]
    seqs = [tokens for _, tokens in recs]

    def draw(seed, memory=True):
        chosen = recs if memory else recs_off
        return _build([r for r, _ in chosen], seqs, num_anchors=5, memory=memory, seed=seed)

    kw = draw(0)
    assert kw["rejmem_anchor_valid"].all()
    for b, (rec, tokens) in enumerate(recs):
        anchors = kw["rejmem_anchor_pos"][b]
        assert (anchors[1:] > anchors[:-1]).all()
        assert set(anchors.tolist()) <= set(rec.tensors["round_anchor_pos"].tolist())
        assert (anchors < len(tokens) - 1).all()
    assert torch.equal(draw(0)["rejmem_anchor_pos"], kw["rejmem_anchor_pos"])
    assert not torch.equal(draw(1)["rejmem_anchor_pos"], kw["rejmem_anchor_pos"])

    off = draw(0, memory=False)
    assert torch.equal(off["rejmem_anchor_pos"], kw["rejmem_anchor_pos"])
    assert kw["rejmem_mem_count"].any()
    assert "rejmem_mem_count" not in off and "rejmem_mem_aux_hidden" not in off
