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

import copy
import sys
import types

import pytest
import torch
import torch.nn as nn

modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from transformers.cache_utils import Cache, DynamicIndexedLayer
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from modelopt.torch.models import hf
from modelopt.torch.models.hf import (
    checkpoint_has_mtp,
    prefers_bundled_code,
    prepare_model_for_calibration,
)


def _model(budget):
    """A model holding one QSA indexer with 4-token key blocks, ``budget // 4`` of them kept."""
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
    return model, config


def _inputs(config, batch, seq):
    hidden = torch.randn(batch, seq, config.hidden_size)
    rope = modeling.Qwen4ExpTextRotaryEmbedding(config)(
        hidden, torch.arange(seq).expand(3, batch, seq)
    )
    return hidden, rope


def _cache():
    return Cache(layers=[DynamicIndexedLayer()])


def _mask(pads, floating, seq, queries=None):
    """Causal mask of the last ``queries`` tokens over ``seq`` keys, padded key positions hidden."""
    queries = seq if queries is None else queries
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))[seq - queries :]
    mask = causal.expand(len(pads), 1, queries, seq).clone()
    for row, (side, n) in enumerate(pads):
        mask[row, :, :, slice(0, n) if side == "left" else slice(seq - n, seq)] = False
    if floating:
        return torch.where(mask, 0.0, torch.finfo(torch.float32).min)
    return mask


def _reference_must_not_run(*args, **kwargs):
    raise AssertionError("the reference indexer loop ran although every visible key fits")


def _forbid_the_loop(monkeypatch, indexer):
    """Fail if the loop runs: through the class, or through the forward the plugin saved."""
    monkeypatch.setattr(type(indexer), "forward", _reference_must_not_run)
    monkeypatch.setattr(indexer, "_modelopt_reference_forward", _reference_must_not_run)


@pytest.mark.parametrize(
    ("budget", "seq", "pads", "floating", "cache"),
    [
        (64, 64, [("right", 0), ("right", 9)], False, False),
        (64, 64, [("left", 13), ("right", 0)], True, False),
        (64, 64, [("left", 5), ("right", 3)], False, True),
        (64, 62, [("left", 7), ("right", 0)], False, True),  # 15 key blocks and a 2-token tail
        (32, 64, [("right", 0)], False, False),  # 16 key blocks, 8 kept: the loop must run
    ],
    ids=["bool_mask", "float_mask", "cache", "incomplete_tail", "over_budget"],
)
def test_qsa_indexer_fast_path_matches_the_reference(
    monkeypatch, budget, seq, pads, floating, cache
):
    torch.manual_seed(0)
    model, config = _model(budget)
    hidden, rope = _inputs(config, len(pads), seq)
    mask = _mask(pads, floating, seq)
    expected_cache, cache_seen = (_cache(), _cache()) if cache else (None, None)

    with torch.no_grad():
        expected = type(model.indexer).forward(model.indexer, hidden, rope, mask, expected_cache)
        prepare_model_for_calibration(model)
        if seq // config.indexer_compress_ratio <= model.indexer.block_topk:
            _forbid_the_loop(monkeypatch, model.indexer)
        projections = []
        model.indexer.index_qk_proj.register_forward_hook(lambda *_: projections.append(None))
        actual = model.indexer(hidden, rope, mask, cache_seen)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    # The projection still runs once, as in the reference, so a quantizer on it gets calibrated.
    assert len(projections) == 1
    if cache:
        assert torch.equal(cache_seen.layers[0].indexer_keys, expected_cache.layers[0].indexer_keys)


def test_qsa_indexer_decoding_past_the_budget_matches_the_reference():
    """Prefill 30 tokens, then decode one token at a time past the 32-token budget: every step
    matches the reference, as the indexer moves from the fast path to the loop, and so do the
    indexer keys the cache accumulates."""
    torch.manual_seed(0)
    model, config = _model(32)
    reference = copy.deepcopy(model.indexer)
    prepare_model_for_calibration(model)
    hidden, rope = _inputs(config, 1, 42)
    caches = (_cache(), _cache())

    with torch.no_grad():
        for start, end in [(0, 30), *((step, step + 1) for step in range(30, 42))]:
            args = (hidden[:, start:end], (rope[0][:, :end], rope[1][:, :end]))
            mask = _mask([("right", 0)], False, end, queries=end - start)
            expected = reference(*args, mask, caches[0])
            assert torch.equal(model.indexer(*args, mask, caches[1]), expected), (start, end)

    assert torch.equal(caches[1].layers[0].indexer_keys, caches[0].layers[0].indexer_keys)


def test_qsa_indexer_fast_path_runs_inside_an_accelerate_hook(monkeypatch):
    """``device_map`` loading puts an Accelerate hook on the indexer before calibration support is
    added; the fast path must run inside it, so the hook still moves the inputs."""
    hooks = pytest.importorskip("accelerate.hooks")

    class _CountingHook(hooks.ModelHook):
        calls = 0

        def pre_forward(self, module, *args, **kwargs):
            self.calls += 1
            return args, kwargs

    torch.manual_seed(0)
    model, config = _model(64)
    hidden, rope = _inputs(config, 1, 64)
    mask = _mask([("right", 0)], False, 64)
    hook = _CountingHook()
    hooks.add_hook_to_module(model.indexer, hook)

    with torch.no_grad():
        expected = model.indexer(hidden, rope, mask, None)
        prepare_model_for_calibration(model)
        _forbid_the_loop(monkeypatch, model.indexer)
        actual = model.indexer(hidden, rope, mask, None)

    assert hook.calls == 2
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_loading_hooks_the_plugin_lacks_keep_their_defaults(model_type):
    """The Qwen3.8 plugin only has a calibration hook; the loading hooks fall back to the defaults."""
    assert not prefers_bundled_code(model_type, None, "ckpt", {})
    assert not checkpoint_has_mtp(model_type, "ckpt")


def test_qsa_indexer_hands_unknown_arguments_to_the_reference():
    """A newer Transformers may pass the indexer an argument the fast path does not know; the call
    goes to the reference forward with it instead of raising ``TypeError``."""
    model, config = _model(64)
    hidden, rope = _inputs(config, 1, 64)
    prepare_model_for_calibration(model)
    calls = []
    model.indexer._modelopt_reference_forward = lambda *a, **kw: calls.append(kw) or "reference"

    mask = _mask([("right", 0)], False, 64)
    assert model.indexer(hidden, rope, mask, None, new_argument=1) == "reference"
    assert calls == [{"new_argument": 1}]


def test_a_qwen3_8_model_without_a_known_indexer_warns():
    """If Transformers renames the indexer class, calibration warns rather than silently running
    the loop."""
    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="qwen4_exp")
    model.layer = nn.Linear(4, 4)
    with pytest.warns(UserWarning, match="Qwen4ExpTextQSAIndexer"):
        prepare_model_for_calibration(model)


def test_a_plugin_exporting_an_unknown_hook_is_refused(monkeypatch):
    """Hooks are optional, so a misspelled one in a plugin's ``__all__`` raises instead of being
    skipped."""
    plugin = types.ModuleType("modelopt.torch.models.misspelled_plugin")
    plugin.__all__ = ["prepare_for_calibraton"]
    plugin.prepare_for_calibraton = lambda model: None
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    monkeypatch.setitem(hf._PLUGINS, "misspelled", "misspelled_plugin")
    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="misspelled")
    with pytest.raises(ValueError, match="prepare_for_calibraton"):
        prepare_model_for_calibration(model)
