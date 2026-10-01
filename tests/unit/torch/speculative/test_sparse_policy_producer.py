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

"""The sparse-policy producer must write what SparsePolicyDataset reads."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from modelopt.torch.speculative.external.sparse_data import SparsePolicyDataset

_PRODUCER = (
    Path(__file__).parents[4]
    / "examples"
    / "speculative_decoding"
    / "collect_hidden_states"
    / "compute_sparse_policy_hf.py"
)

VOCAB = 64
TOPK = 8


def _load_producer():
    """Import the example script by path; it is not an installed module."""
    sys.path.insert(0, str(_PRODUCER.parent))
    spec = importlib.util.spec_from_file_location("compute_sparse_policy_hf", _PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


producer = pytest.importorskip("datasets") and _load_producer()


def test_deployment_policy_rows_are_normalised_and_nucleus_truncated():
    """Each stored row must sum to 1, or the objective under-penalises out-of-support mass."""
    torch.manual_seed(0)
    logits = torch.randn(5, VOCAB) * 2
    _, lp = producer.deployment_policy(logits, TOPK, 0.95, 1.0)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(5), atol=1e-5)
    # Excluded candidates keep their slot at -inf, which decodes to zero probability.
    _, tight = producer.deployment_policy(logits, TOPK, 0.5, 1.0)
    assert int((tight.exp() > 0).sum()) < int((lp.exp() > 0).sum()), (
        "a tighter nucleus must drop candidates; top-p on un-renormalised mass would not"
    )
    assert torch.allclose(tight.exp().sum(-1), torch.ones(5), atol=1e-5)


def test_record_round_trips_with_the_shift_the_loss_expects(tmp_path):
    """Position t of the stored policy must be the distribution that produced ids[t]."""
    torch.manual_seed(0)
    n_prompt, n_gen = 4, 6
    ids = torch.randint(0, VOCAB, (n_prompt + n_gen,))
    logits = torch.randn(n_prompt + n_gen, VOCAB) * 2

    record = producer.policy_record("c0", ids, n_prompt, ids.shape[0], logits, TOPK, 0.95, 1.0)
    shard = tmp_path / "p.jsonl.gz"
    with gzip.open(shard, "wt") as f:
        f.write(json.dumps(record) + "\n")

    ds = SparsePolicyDataset([str(shard)], max_length=64, top_k=TOPK)
    assert len(ds) == 1
    item = ds[0]
    assert torch.equal(item["input_ids"], ids.long())
    assert item["loss_mask"][:n_prompt].sum() == 0, "context must not be scored"
    assert item["loss_mask"][n_prompt:].sum() == n_gen

    # The policy stored at sequence index t must be the one derived from logits[t - 1].
    for t in range(n_prompt, n_prompt + n_gen):
        expected = logits[t - 1].softmax(-1).topk(TOPK).indices
        assert torch.equal(item["teacher_topk_tok"][t], expected.long()), f"misaligned at {t}"


def test_span_starting_at_zero_is_rejected():
    """Position 0 has no preceding context; emitting an empty policy would lose the record."""
    ids = torch.randint(0, VOCAB, (6,))
    logits = torch.randn(6, VOCAB)
    with pytest.raises(ValueError, match="index >= 1"):
        producer.policy_record("c0", ids, 0, ids.shape[0], logits, TOPK, 0.95, 1.0)


def test_multi_turn_spans_do_not_swallow_the_user_turns():
    """Each assistant turn is its own record; the user turns between them are context."""
    # positions 3-5 and 9-11 are assistant turns; 6-8 is a user turn between them
    mask = torch.zeros(1, 13)
    mask[0, 3:6] = 1.0
    mask[0, 9:12] = 1.0
    assert producer.scored_spans(mask) == [(3, 6), (9, 12)]

    ids = torch.randint(0, VOCAB, (13,))
    logits = torch.randn(13, VOCAB)
    first = producer.policy_record("c", ids, 3, 6, logits, TOPK, 0.95, 1.0)
    second = producer.policy_record("c", ids, 9, 12, logits, TOPK, 0.95, 1.0)

    # the second turn's prompt carries the intervening user turn as context, unscored
    assert len(first["gen_ids"]) == 3 and len(second["gen_ids"]) == 3
    assert len(second["prompt_ids"]) == 9
    # the old behaviour scored everything after the first assistant token
    assert len(first["gen_ids"]) != 10, "span must stop at the end of its own turn"
