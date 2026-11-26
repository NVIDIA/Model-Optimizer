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

"""CPU tests for rejection-memory training in the shared DFlash-family forward (``rejmem_*`` kwargs)."""

from unittest.mock import patch

import pytest
import torch
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import get_tiny_llama
from torch.nn.attention.flex_attention import create_mask

import modelopt.torch.speculative as mtsp
import modelopt.torch.speculative.plugins.hf_dflash as hf_dflash

SEQ_LEN = 32
BLOCK_SIZE = 4
MEM_ROWS = BLOCK_SIZE - 1
WINDOW = 8  # shorter than the sequence, and at least 2 * BLOCK_SIZE - 1
PROMPT_LEN = 2  # positions before it are never labels

# Two rows of five rounds: anchors at the start and near the end of the sequence, padding
# entries (one holding out-of-range values, as padding may), and every memory size 0..MEM_ROWS.
ANCHORS = torch.tensor([[0, 6, 13, 29, 31], [3, 8, 16, -1, 30]])
VALID = torch.tensor([[True, True, False, True, True], [True, True, True, False, True]])
COUNTS = torch.tensor(
    [[0, 1, 3, MEM_ROWS, 2], [MEM_ROWS, 0, 1, 2 * BLOCK_SIZE, 3]], dtype=torch.int32
)
# Rounds the forward trains: valid with a label in the sequence, so not anchor 31.
TRAINED = torch.tensor([[True, True, False, True, False], [True, True, True, False, True]])

_HEADS = {
    "dspark": {"projector_type": "dspark", "markov_rank": 4},
    "dflash2": {
        "projector_type": "dflash2",
        "conv_kernel_size": 2,
        "conv_group_size": 4,
        "selector_rank": 4,
        "selector_top_k": 5,
    },
}


def _reference_layout(anchors, keep, mem_count, window, causal):
    """The draft's keys under rejection memory, written out key by key: visibility and positions.

    Keys are ``[context | memory | draft blocks]``; block ``k`` owns memory keys
    ``SEQ_LEN + k * MEM_ROWS + i`` at positions ``anchor + i``, its draft keys sit at
    ``anchor + j``. The query at offset ``j`` of a kept block anchored at ``a`` with ``r``
    memory rows sees context ``p`` with ``a + j + r - window + 1 <= p <= a - 1`` (``p <= a - 1``
    without a window), the first ``r`` memory keys of its own block whatever ``j``, and its own
    draft keys (offsets ``<= j`` when causal). ``mem_count=None`` lays out no memory keys.
    """
    bsz, n_blocks = anchors.shape
    mem_rows = 0 if mem_count is None else MEM_ROWS
    draft_start = SEQ_LEN + n_blocks * mem_rows
    kv_len = draft_start + n_blocks * BLOCK_SIZE
    visible = torch.zeros(bsz, n_blocks * BLOCK_SIZE, kv_len, dtype=torch.bool)
    positions = torch.zeros(bsz, kv_len, dtype=torch.long)
    for b in range(bsz):
        positions[b, :SEQ_LEN] = torch.arange(SEQ_LEN)
        for k in range(n_blocks):
            a = int(anchors[b, k])
            for i in range(mem_rows):
                positions[b, SEQ_LEN + k * mem_rows + i] = a + i
            for j in range(BLOCK_SIZE):
                positions[b, draft_start + k * BLOCK_SIZE + j] = a + j
            if not keep[b, k]:
                continue
            r = 0 if mem_count is None else int(mem_count[b, k])
            for j in range(BLOCK_SIZE):
                row = visible[b, k * BLOCK_SIZE + j]
                lowest = 0 if window is None else a + j + r - window + 1
                for p in range(SEQ_LEN):
                    row[p] = lowest <= p <= a - 1
                for i in range(r):
                    row[SEQ_LEN + k * mem_rows + i] = True
                for j2 in range(BLOCK_SIZE):
                    row[draft_start + k * BLOCK_SIZE + j2] = j2 <= j or not causal
    return visible, positions


def _model(window=WINDOW, head=None, causal=False, offline=True):
    model = get_tiny_llama(num_hidden_layers=4)
    model.config.num_orig_hidden_layers = 4
    config = get_dflash_config(block_size=BLOCK_SIZE, offline=offline)
    config["dflash_swa_window_size"] = window
    config["dflash_draft_attention"] = "causal" if causal else "bidirectional"
    config["dflash_num_anchors"] = ANCHORS.shape[1]
    config["dflash_architecture_config"].update(head or {})
    mtsp.convert(model, [("dflash", config)])
    return model.float().train()


def _batch(model, counts=COUNTS):
    """An offline batch of the rounds above, with random features."""
    torch.manual_seed(0)
    bsz, n_rounds = ANCHORS.shape
    hidden = model.config.hidden_size
    aux = len(model.target_layer_ids) * hidden
    input_ids = torch.randint(1, model.config.vocab_size, (bsz, SEQ_LEN))
    loss_mask = torch.ones(bsz, SEQ_LEN)
    loss_mask[:, :PROMPT_LEN] = 0.0
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "loss_mask": loss_mask,
        "base_model_outputs": {
            "aux_hidden_states": torch.randn(bsz, SEQ_LEN, aux),
            "base_model_hidden_states": torch.randn(bsz, SEQ_LEN, hidden),
        },
        "rejmem_anchor_pos": ANCHORS.clone(),
        "rejmem_anchor_valid": VALID.clone(),
        "rejmem_mem_count": counts.clone(),
        "rejmem_mem_aux_hidden": torch.randn(bsz, n_rounds, MEM_ROWS, aux),
    }


def _step(model, batch):
    """One forward/backward: the loss and the draft's gradients."""
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    grads = {
        n: p.grad.clone() for n, p in model.dflash_module.named_parameters() if p.grad is not None
    }
    return loss.detach(), grads


def _draft_hidden(model, batch):
    """The draft's output per block: [B, K, BLOCK_SIZE, H]."""
    outputs = []
    hook = model.dflash_module.register_forward_hook(lambda _m, _a, out: outputs.append(out))
    with torch.no_grad():
        model(**batch)
    hook.remove()
    return outputs[0].view(*ANCHORS.shape, BLOCK_SIZE, -1)


def _without_rejmem(batch):
    return {k: v for k, v in batch.items() if not k.startswith("rejmem_")}


@pytest.mark.parametrize("window", [None, WINDOW], ids=["full", "window"])
@pytest.mark.parametrize(
    ("mem_count", "causal"),
    [(None, False), (None, True), (COUNTS, False)],
    ids=["bidirectional", "causal", "memory"],
)
def test_layout_matches_the_reference(mem_count, causal, window):
    """The dense mask, the vmapped FlexAttention rule and the position ids follow the reference."""
    model = _model(window=window, causal=causal)
    visible, positions = _reference_layout(ANCHORS, VALID, mem_count, window, causal)
    dense = model._build_draft_attention_mask(
        SEQ_LEN, ANCHORS, VALID, ANCHORS.shape[1], torch.float32, "cpu", window, mem_count
    )
    assert torch.equal(dense[:, 0] > torch.finfo(torch.float32).min / 2, visible)
    mask_mod = hf_dflash._draft_mask_mod(
        SEQ_LEN, ANCHORS, VALID, BLOCK_SIZE, window, causal, mem_count
    )
    bsz, q_len, kv_len = visible.shape
    assert torch.equal(create_mask(mask_mod, bsz, 1, q_len, kv_len, "cpu")[:, 0], visible)
    mem_rows = 0 if mem_count is None else MEM_ROWS
    assert torch.equal(model._build_position_ids(SEQ_LEN, ANCHORS, "cpu", mem_rows), positions)


@pytest.mark.parametrize("row", [0, 1])
def test_a_memory_row_reaches_its_own_block_only(row):
    """Block 1 of row 0 has one memory row: row 0 changes that block's output, row 1 nothing."""
    model = _model()
    batch = _batch(model)
    before = _draft_hidden(model, batch)
    batch["rejmem_mem_aux_hidden"][0, 1, row] += 1.0
    changed = (_draft_hidden(model, batch) != before).flatten(2).any(-1)
    expected = torch.zeros_like(changed)
    expected[0, 1] = row < COUNTS[0, 1]
    # Dropped blocks see no key, so their (never trained) output is not compared.
    assert torch.equal(changed[TRAINED], expected[TRAINED])


@pytest.mark.parametrize("memory", ["omitted", "zero_counts"])
@pytest.mark.parametrize("window", [None, WINDOW], ids=["full", "window"])
def test_rounds_without_memory_train_as_the_step_without_it(window, memory):
    """Rounds without memory train what the same blocks train without ``rejmem_*`` kwargs.

    Bitwise when the memory pair is left out. Zero counts still append the memory rows, which
    lengthens the projections and SDPA's key axis and so reorders sums: equal to rounding.
    """
    model = _model(window=window)
    batch = _batch(model, counts=torch.zeros_like(COUNTS))
    if memory == "omitted":
        del batch["rejmem_mem_count"], batch["rejmem_mem_aux_hidden"]
    with_rounds = _step(model, batch)
    with patch.object(model, "_sample_anchor_positions", return_value=(ANCHORS, TRAINED)):
        without_rounds = _step(model, _without_rejmem(batch))
    exact = {"rtol": 0, "atol": 0} if memory == "omitted" else {}
    torch.testing.assert_close(with_rounds, without_rounds, **exact)


@pytest.mark.parametrize("rows", ["seen", "unseen"])
def test_memory_trains_the_context_projection(rows):
    """Rows a kept block sees reach the loss and fc, hidden_norm, k/v_proj, k_norm; others nothing.

    The others are rows past the count and rows of dropped blocks, whatever they hold, even NaN.
    """
    model = _model()
    batch = _batch(model)
    loss, grads = _step(model, batch)
    memory = batch["rejmem_mem_aux_hidden"]
    seen = (torch.arange(MEM_ROWS) < COUNTS.unsqueeze(-1)) & TRAINED.unsqueeze(-1)
    if rows == "seen":
        memory[seen] = torch.randn_like(memory[seen])
    else:
        memory[~seen] = float("nan")
    other_loss, other_grads = _step(model, batch)
    if rows == "unseen":
        torch.testing.assert_close((other_loss, other_grads), (loss, grads), rtol=0, atol=0)
        return
    assert not torch.equal(loss, other_loss)
    projections = ["fc.weight", "hidden_norm.weight"] + [
        f"layers.{i}.self_attn.{proj}.weight"
        for i in range(len(model.dflash_module.layers))
        for proj in ("k_proj", "v_proj", "k_norm")
    ]
    assert [name for name in projections if torch.equal(grads[name], other_grads[name])] == []


@pytest.mark.parametrize("head", list(_HEADS))
def test_variants_train_on_rollout_rounds(head):
    model = _model(head=_HEADS[head])
    loss, grads = _step(model, _batch(model))
    assert torch.isfinite(loss)
    ungraded = [
        n for n, p in model.dflash_module.named_parameters() if p.requires_grad and n not in grads
    ]
    assert not ungraded, f"no gradient for {ungraded}"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("online", "offline"),
        ("partial", "rejmem_mem_count"),
        ("memory_rows", "rejmem_mem_aux_hidden"),
        ("memory_dtype", "rejmem_mem_aux_hidden"),
        ("round_count", "dflash_num_anchors"),
        ("negative_anchor", "rejmem_anchor_pos"),
        ("count_past_rows", "rejmem_mem_count"),
        ("short_window", "dflash_swa_window_size"),
        ("causal", "dflash_draft_attention"),
    ],
)
def test_refuses_rounds_it_cannot_train(change, match):
    model = _model(
        window=2 * BLOCK_SIZE - 2 if change == "short_window" else WINDOW,
        causal=change == "causal",
        offline=change != "online",
    )
    batch = _batch(model)
    if change == "partial":
        del batch["rejmem_mem_count"]
    elif change == "memory_rows":
        batch["rejmem_mem_aux_hidden"] = batch["rejmem_mem_aux_hidden"][:, :, :-1]
    elif change == "memory_dtype":
        batch["rejmem_mem_aux_hidden"] = batch["rejmem_mem_aux_hidden"].bfloat16()
    elif change == "round_count":
        batch.update({k: v[:, :-1] for k, v in batch.items() if k.startswith("rejmem_")})
    elif change == "negative_anchor":
        batch["rejmem_anchor_pos"][0, 1] = -1
    elif change == "count_past_rows":
        batch["rejmem_mem_count"][0, 1] = BLOCK_SIZE
    with pytest.raises(ValueError, match=match):
        model(**batch)
