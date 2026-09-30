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

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.opt.utils import apply_mode_with_sampling
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    tf_modelopt_state_and_output_tester,
)
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM

from modelopt.torch.opt.plugins.transformers import (
    _TRANSFORMERS_GE_5_0,
    _legacy_tied_weights_keys_as_dict,
)


@pytest.mark.parametrize("model_cls", [LlamaForCausalLM, AutoModelForCausalLM])
def test_causal_lm_save_restore(tmp_path, model_cls):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, hidden_size=128, dtype=torch.float32)
    model_ref = model_cls.from_pretrained(tiny_llama_dir)
    # TODO: Add calibrate, compress mode to the test
    model_ref = apply_mode_with_sampling(
        model_ref, ["sparse_magnitude", "export_sparse", "quantize"]
    )
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")

    model_test = model_cls.from_pretrained(tiny_llama_dir / "modelopt_model")
    tf_modelopt_state_and_output_tester(model_ref, model_test)


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_save_pretrained_with_legacy_tied_weights_keys(tmp_path, tie_word_embeddings):
    """A model declaring 4.x list-style `_tied_weights_keys` must still save (nvbug 6518665).

    transformers>=5 expects a `{target: source}` dict there and calls `.keys()` on it while
    saving, which crashes for `trust_remote_code` modeling code that has not migrated yet.

    Both tying configurations are covered because normalizing the list to `{key: key}` feeds
    those keys to the save-time dedup as patterns: an untied weight must survive it, and a
    genuinely tied one must still be deduped down to its canonical name.
    """
    tiny_llama_dir = create_tiny_llama_dir(
        tmp_path, hidden_size=128, dtype=torch.float32, tie_word_embeddings=tie_word_embeddings
    )
    model = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)
    model = apply_mode_with_sampling(model, ["quantize"])

    model._tied_weights_keys = ["lm_head.weight"]
    model.model._tied_weights_keys = ["embed_tokens.weight"]

    save_dir = tiny_llama_dir / "legacy_tied_keys_model"
    model.save_pretrained(save_dir)

    # The declarations the model owned before the save are restored verbatim.
    assert model._tied_weights_keys == ["lm_head.weight"]
    assert model.model._tied_weights_keys == ["embed_tokens.weight"]

    # No weight is silently dropped: `lm_head.weight` is written out unless it really does
    # share storage with the embedding, in which case transformers re-ties it on load.
    saved_keys = set(load_file(save_dir / "model.safetensors"))
    assert "model.embed_tokens.weight" in saved_keys
    assert ("lm_head.weight" in saved_keys) is not tie_word_embeddings

    model_test = AutoModelForCausalLM.from_pretrained(save_dir)
    tf_modelopt_state_and_output_tester(model, model_test)


@pytest.mark.skipif(not _TRANSFORMERS_GE_5_0, reason="list-style keys are native to transformers 4")
def test_legacy_tied_weights_keys_as_dict_restores_class_attribute():
    """The shim must not leave an instance attribute shadowing the class declaration."""

    class _LegacyChild(nn.Module):
        # How remote-code models declare it: on the class, not the instance.
        _tied_weights_keys = ["lm_head.weight"]

    parent = nn.Module()
    parent.child = _LegacyChild()

    with _legacy_tied_weights_keys_as_dict(parent):
        assert parent.child._tied_weights_keys == {"lm_head.weight": "lm_head.weight"}

    assert _LegacyChild._tied_weights_keys == ["lm_head.weight"]
    assert "_tied_weights_keys" not in parent.child.__dict__


def test_causal_lm_from_config(tmp_path):
    """Test loading a model using from_config after applying optimizations"""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, hidden_size=128, dtype=torch.float32)

    model_ref = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)
    model_ref = apply_mode_with_sampling(
        model_ref, ["sparse_magnitude", "export_sparse", "quantize"]
    )
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")

    config = AutoConfig.from_pretrained(tiny_llama_dir / "modelopt_model")

    model_test = AutoModelForCausalLM.from_config(config)

    # from_config doesn't load weights, need to load state_dict separately
    state_dict = model_ref.state_dict()
    model_test.load_state_dict(state_dict)

    tf_modelopt_state_and_output_tester(model_ref, model_test)


# This test is flaky and causes other tests to fail; This seems to run fine in isolation
@pytest.mark.manual(
    reason="Flaky test causing other tests to fail, run this test manually with --run-manual"
)
def test_transformers_load_with_multi_thread(tmp_path):
    """Multi-threaded test for save/restore functionality"""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    workers = 2
    exceptions = []

    def worker_func(worker_id):
        try:
            _ = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)
        except Exception as e:
            traceback.print_exc()
            return e

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_func, i) for i in range(workers)]

        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                exceptions.append(result)

    assert len(exceptions) == 0, "Parallel model loading tests failed, check error log"
