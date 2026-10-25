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

"""Qwen3.8's QSA indexer fast path against Transformers' reference loop."""

import types

import pytest
import torch
import torch.nn as nn

modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from modelopt.torch.models.hf import prepare_model_for_calibration

SEQ = 64


class _IndexerCache:
    """Records the indexer keys a forward appends, as the model's cache would."""

    def __init__(self):
        self.keys = []

    def update_indexer(self, keys, layer_idx):
        self.keys.append(keys)
        return keys


def _mask(pads, floating):
    """Causal mask over SEQ tokens, with padded key positions hidden per row."""
    mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool)).expand(len(pads), 1, SEQ, SEQ).clone()
    for row, (side, n) in enumerate(pads):
        mask[row, :, :, slice(0, n) if side == "left" else slice(SEQ - n, SEQ)] = False
    if floating:
        return torch.where(mask, 0.0, torch.finfo(torch.float32).min)
    return mask


def _reference_must_not_run(*args, **kwargs):
    raise AssertionError("the reference indexer loop ran although every visible key fits")


@pytest.mark.parametrize(
    ("budget", "pads", "floating", "cache"),
    [
        (64, [("right", 0), ("right", 9)], False, False),
        (64, [("left", 13), ("right", 0)], True, False),
        (64, [("left", 5), ("right", 3)], False, True),
        (32, [("right", 0)], False, False),  # 16 key blocks, 8 kept: the loop must run
    ],
    ids=["bool_mask", "float_mask", "cache", "over_budget"],
)
def test_qsa_indexer_fast_path_matches_the_reference(monkeypatch, budget, pads, floating, cache):
    torch.manual_seed(0)
    config = Qwen4ExpTextConfig(
        hidden_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=budget,
        indexer_compress_ratio=4,
    )
    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="qwen4_exp")
    model.indexer = modeling.Qwen4ExpTextQSAIndexer(config, layer_idx=0)
    hidden = torch.randn(len(pads), SEQ, config.hidden_size)
    rope = modeling.Qwen4ExpTextRotaryEmbedding(config)(
        hidden, torch.arange(SEQ).expand(3, len(pads), SEQ)
    )
    mask = _mask(pads, floating)
    expected_cache, cache_seen = (_IndexerCache(), _IndexerCache()) if cache else (None, None)

    with torch.no_grad():
        expected = type(model.indexer).forward(model.indexer, hidden, rope, mask, expected_cache)
        prepare_model_for_calibration(model)
        if SEQ // config.indexer_compress_ratio <= model.indexer.block_topk:
            monkeypatch.setattr(type(model.indexer), "forward", _reference_must_not_run)
        actual = model.indexer(hidden, rope, mask, cache_seen)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    if cache:
        assert torch.equal(cache_seen.keys[0], expected_cache.keys[0])
