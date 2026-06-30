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

import os
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import (
    get_tiny_llama,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForCausalLM

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_eagle import HFEagleModel


@pytest.mark.parametrize("eagle_config", [EAGLE3_DEFAULT_CFG])
def test_eagle_model_convert_save_and_restore(tmp_path, eagle_config):
    model_ref = get_tiny_llama(num_hidden_layers=8)

    config = deepcopy(eagle_config["config"])
    config["eagle_architecture_config"].update(
        {
            "draft_vocab_size": model_ref.config.vocab_size,
            "hidden_size": model_ref.config.hidden_size,
        }
    )

    mtsp.convert(model_ref, mode=[("eagle", config)])
    assert isinstance(model_ref, mtsp.plugins.HFEagleModel)

    model_ref.save_pretrained(tmp_path / "modelopt_model")
    assert os.path.exists(tmp_path / "modelopt_model/modelopt_state.pth")

    model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
    assert isinstance(model_test, mtsp.plugins.HFEagleModel)
    tf_modelopt_state_and_output_tester(model_ref, model_test)


@pytest.mark.parametrize("eagle_config", [EAGLE3_DEFAULT_CFG])
@pytest.mark.parametrize("eagle_ttt_steps", [1, 2])
@pytest.mark.parametrize("eagle_hsm_mode", ["sparse_replace", "uniform_layer_sample"])
def test_eagle_mix_hidden_states_backward(eagle_config, eagle_ttt_steps, eagle_hsm_mode):
    """Regression test for GitHub issue #1088.

    Verifies that the EAGLE training forward+backward pass does not crash with
    ``eagle_mix_hidden_states=True`` due to an in-place tensor modification
    breaking autograd.
    """
    model = get_tiny_llama(num_hidden_layers=8)

    config = deepcopy(eagle_config["config"])
    config["eagle_architecture_config"].update(
        {
            "draft_vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
        }
    )
    config["eagle_mix_hidden_states"] = True
    config["eagle_hsm_mode"] = eagle_hsm_mode
    config["eagle_ttt_steps"] = eagle_ttt_steps
    config["eagle_use_torch_compile"] = False

    mtsp.convert(model, mode=[("eagle", config)])
    model.train()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None
    outputs.loss.backward()

    eagle_grads = [p.grad for p in model.eagle_module.parameters() if p.grad is not None]
    assert len(eagle_grads) > 0, "Expected gradients to flow to eagle_module"


def test_eagle_uniform_layer_sample_picks_whole_candidate_per_token(monkeypatch):
    """Uniform layer sampling must select one full hidden vector per token."""
    hidden_states_history = [torch.full((2, 3, 4), float(candidate)) for candidate in range(3)]
    selected_candidates = torch.tensor([[[[0], [1], [2]], [[2], [1], [0]]]])

    def fake_randint(high, size, *, device=None):
        assert high == len(hidden_states_history)
        assert tuple(size) == tuple(selected_candidates.shape)
        return selected_candidates.to(device)

    monkeypatch.setattr(torch, "randint", fake_randint)

    mixed = HFEagleModel._sample_uniform_hidden_state_history(hidden_states_history)
    expected = selected_candidates.squeeze(0).float().expand(-1, -1, 4)

    assert torch.equal(mixed, expected)


def _build_share_kv_model(eagle_ttt_steps, *, attn_impl="sdpa", roll_query=False, mix_hsm=False):
    """Construct a tiny-Llama EAGLE model wired with ShareKV (and optionally HSM)."""
    model = get_tiny_llama(num_hidden_layers=8)

    config = deepcopy(EAGLE3_DEFAULT_CFG["config"])
    config["eagle_architecture_config"].update(
        {
            "draft_vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "_attn_implementation": attn_impl,
        }
    )
    config["eagle_share_kv"] = True
    config["eagle_share_kv_roll_query"] = roll_query
    config["eagle_mix_hidden_states"] = mix_hsm
    config["eagle_ttt_steps"] = eagle_ttt_steps
    config["eagle_use_torch_compile"] = False

    mtsp.convert(model, mode=[("eagle", config)])
    model.train()
    return model


@pytest.mark.parametrize("eagle_ttt_steps", [2, 3])
@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_eagle_share_kv_backward(eagle_ttt_steps, attn_impl):
    """Forward+backward with eagle_share_kv=True (no roll, no HSM) on tiny-Llama."""
    model = _build_share_kv_model(eagle_ttt_steps, attn_impl=attn_impl)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None
    outputs.loss.backward()

    eagle_grads = [p.grad for p in model.eagle_module.parameters() if p.grad is not None]
    assert len(eagle_grads) > 0, "Expected gradients to flow to eagle_module"


@pytest.mark.parametrize("eagle_ttt_steps", [2, 3])
def test_eagle_share_kv_roll_query_backward(eagle_ttt_steps):
    """Forward+backward with eagle_share_kv_roll_query=True; sdpa only (eager unsupported)."""
    model = _build_share_kv_model(eagle_ttt_steps, attn_impl="sdpa", roll_query=True)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None
    outputs.loss.backward()

    eagle_grads = [p.grad for p in model.eagle_module.parameters() if p.grad is not None]
    assert len(eagle_grads) > 0, "Expected gradients to flow to eagle_module"


@pytest.mark.parametrize("eagle_ttt_steps", [2, 3])
def test_eagle_share_kv_and_hsm_backward(eagle_ttt_steps):
    """ShareKV and HSM enabled simultaneously: orthogonal mechanisms must coexist."""
    model = _build_share_kv_model(eagle_ttt_steps, attn_impl="sdpa", mix_hsm=True)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None
    outputs.loss.backward()

    eagle_grads = [p.grad for p in model.eagle_module.parameters() if p.grad is not None]
    assert len(eagle_grads) > 0, "Expected gradients to flow to eagle_module"


def test_eagle_share_kv_injection_numerics():
    """Each TTT step N>0 must reuse step N-1's captured (K,V) — a sliding window of one.

    With 3 TTT steps this specifically guards against the stale-KV bug where every
    step would reuse step 0's KV instead of the immediately preceding step's.
    """
    from collections import defaultdict

    import modelopt.torch.speculative.plugins.modeling_eagle as me

    model = _build_share_kv_model(eagle_ttt_steps=3, attn_impl="sdpa")
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    labels = input_ids.clone()

    records = []
    original_update = me.EagleShareKVCache.update

    def spy_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        pre_mode = self._mode
        returned_k, returned_v = original_update(
            self, key_states, value_states, layer_idx, cache_kwargs
        )
        records.append(
            {
                "mode": pre_mode,
                "layer_idx": layer_idx,
                "input_k": key_states,
                "returned_k": returned_k,
            }
        )
        return returned_k, returned_v

    me.EagleShareKVCache.update = spy_update
    try:
        model(input_ids=input_ids, labels=labels)
    finally:
        me.EagleShareKVCache.update = original_update

    # Group calls per layer, preserving TTT-step order (one call per layer per step).
    by_layer = defaultdict(list)
    for r in records:
        by_layer[r["layer_idx"]].append(r)
    assert by_layer, "EagleShareKVCache.update was never called"

    for layer_idx, calls in by_layer.items():
        assert len(calls) == 3, f"Layer {layer_idx}: expected 3 TTT calls, got {len(calls)}"
        s0, s1, s2 = calls
        # Step 0 captures and uses its own local KV.
        assert s0["mode"] == "capture"
        assert s0["returned_k"] is s0["input_k"]
        # Step 1 reuses step 0's local KV.
        assert s1["mode"] == "inject"
        assert s1["returned_k"] is s0["input_k"], (
            f"Layer {layer_idx}: step 1 must reuse step 0's KV"
        )
        # Step 2 reuses step 1's local KV — NOT step 0's (the regression guard).
        assert s2["mode"] == "inject"
        assert s2["returned_k"] is s1["input_k"], (
            f"Layer {layer_idx}: step 2 must reuse step 1's KV, not step 0's"
        )
        assert s2["returned_k"] is not s0["input_k"], (
            f"Layer {layer_idx}: step 2 incorrectly reused step 0's KV (stale ShareKV)"
        )
        # Each step's locally computed KV differs from the previous step's.
        assert not torch.equal(s1["input_k"], s0["input_k"])
        assert not torch.equal(s2["input_k"], s1["input_k"])


def test_eagle_share_kv_roll_query_requires_share_kv():
    """Pydantic validator should reject eagle_share_kv_roll_query=True without share_kv."""
    model = get_tiny_llama(num_hidden_layers=8)
    config = deepcopy(EAGLE3_DEFAULT_CFG["config"])
    config["eagle_architecture_config"].update(
        {
            "draft_vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
        }
    )
    config["eagle_share_kv"] = False
    config["eagle_share_kv_roll_query"] = True

    with pytest.raises(Exception, match="eagle_share_kv_roll_query"):
        mtsp.convert(model, mode=[("eagle", config)])


@pytest.mark.parametrize("roll_query", [False, True])
def test_eagle_share_kv_offline_backward(roll_query):
    """ShareKV must work in offline (pre-computed hidden states) training, like HSM.

    Offline is the primary EAGLE training path; ShareKV is internal to the draft
    module's attention, so it is orthogonal to whether the base model runs live.
    """
    model = get_tiny_llama(num_hidden_layers=8)
    hidden = model.config.hidden_size
    vocab = model.config.vocab_size

    config = deepcopy(EAGLE3_DEFAULT_CFG["config"])
    config["eagle_architecture_config"].update(
        {
            "draft_vocab_size": vocab,
            "hidden_size": hidden,
            "_attn_implementation": "sdpa",
        }
    )
    config["eagle_offline"] = True
    config["eagle_share_kv"] = True
    config["eagle_share_kv_roll_query"] = roll_query
    config["eagle_ttt_steps"] = 3
    config["eagle_use_torch_compile"] = False

    mtsp.convert(model, mode=[("eagle", config)])
    model.train()

    num_aux = len(model.eagle_config.eagle_aux_hidden_state_layer_ids)
    dtype = next(model.parameters()).dtype
    b, s = 2, 16
    input_ids = torch.randint(0, vocab, (b, s))
    labels = input_ids.clone()
    # Offline training feeds pre-computed base-model tensors instead of running the
    # (now layer-less) base model; dtype must match the draft module's params.
    base_model_outputs = {
        "base_model_hidden_states": torch.randn(b, s, hidden, dtype=dtype),
        "base_model_input_embeds": torch.randn(b, s, hidden, dtype=dtype),
        "aux_hidden_states": torch.randn(b, s, hidden * num_aux, dtype=dtype),
    }

    outputs = model(input_ids=input_ids, labels=labels, base_model_outputs=base_model_outputs)
    assert outputs.loss is not None
    outputs.loss.backward()

    eagle_grads = [p.grad for p in model.eagle_module.parameters() if p.grad is not None]
    assert len(eagle_grads) > 0, "Expected gradients to flow to eagle_module"
