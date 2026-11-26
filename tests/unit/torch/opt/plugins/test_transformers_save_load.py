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

import threading
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.opt.utils import apply_mode_with_sampling
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    create_tiny_qwen3_moe_dir,
    get_tiny_qwen3vl,
    tf_modelopt_state_and_output_tester,
)
from packaging.version import Version
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    LlamaForCausalLM,
    Qwen3VLMoeConfig,
)
from transformers import __version__ as transformers_version
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.opt.plugins.transformers import _legacy_tied_weights_keys_as_dict
from modelopt.torch.quantization.nn import QuantModuleRegistry


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


@pytest.mark.parametrize(
    "quant_config", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG], ids=["fp8", "nvfp4"]
)
@torch.no_grad()
def test_vlm_export_without_modelopt_state(tmp_path, quant_config):
    """Export separately converted VLM components without requiring a single ModelOpt state."""
    model = get_tiny_qwen3vl().eval()
    disabled_config = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": "max"}
    mtq.quantize(model.model.visual, disabled_config)
    mtq.quantize(model.lm_head, disabled_config)
    mtq.quantize(
        model.model.language_model,
        quant_config,
        lambda _: model(input_ids=torch.ones(1, 4, dtype=torch.long)),
    )
    state_path = tmp_path / "modelopt_state.pth"
    torch.save(mto.modelopt_state(model.model.language_model), state_path)

    export_hf_checkpoint(model, export_dir=tmp_path, save_modelopt_state=False)

    assert not state_path.exists()
    saved = load_file(tmp_path / "model.safetensors")
    assert any(key.endswith(".weight_scale") for key in saved)
    assert any("visual" in key for key in saved)


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


def _quantize_experts(model, weight_cfg=None):
    weight_cfg = weight_cfg or {"num_bits": 8, "axis": 0}
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*experts*weight_quantizer*", "cfg": weight_cfg},
    ]
    input_ids = torch.randint(0, model.config.get_text_config().vocab_size, (1, 8))
    mtq.quantize(
        model, {"quant_cfg": quant_cfg, "algorithm": "max"}, lambda m: m(input_ids=input_ids)
    )
    return model


def _tiny_quantized_qwen3_moe(path, weight_cfg=None):
    model_dir = create_tiny_qwen3_moe_dir(path, dtype=torch.float32)
    return _quantize_experts(AutoModelForCausalLM.from_pretrained(model_dir), weight_cfg)


@pytest.mark.parametrize(
    "weight_cfg",
    [
        {"num_bits": 8, "axis": 0},
        {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}},
    ],
    ids=["int8", "nvfp4"],
)
def test_save_pretrained_quantized_fused_experts(tmp_path, weight_cfg):
    """Quantized fused MoE experts save per expert by default and reload to the same state."""
    model_ref = _tiny_quantized_qwen3_moe(tmp_path, weight_cfg)
    model_ref.save_pretrained(tmp_path / "quantized")
    # A caller's state_dict, as the HF Trainer passes, saves the same keys.
    model_ref.save_pretrained(tmp_path / "from_state_dict", state_dict=model_ref.state_dict())

    saved_keys = load_file(tmp_path / "quantized" / "model.safetensors").keys()
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in saved_keys
    assert "model.layers.0.mlp.experts.gate_up_proj_weight_quantizers.0._amax" in saved_keys
    assert load_file(tmp_path / "from_state_dict" / "model.safetensors").keys() == saved_keys

    model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "quantized")
    torch.testing.assert_close(model_test.state_dict(), model_ref.state_dict(), rtol=0, atol=0)
    if weight_cfg["num_bits"] == 8:  # NVFP4 fake quant needs a GPU
        tf_modelopt_state_and_output_tester(model_ref, model_test)


def test_save_pretrained_quantized_fused_experts_in_overlapping_saves(tmp_path):
    """A save that ends while a later one is still running does not undo the later one's fix."""
    models, pauses = {}, {}
    for name in ("first", "second"):
        models[name] = _tiny_quantized_qwen3_moe(tmp_path / name)
        reached, resume = threading.Event(), threading.Event()

        # save_pretrained takes the state dict before it converts the weights' names.
        def pause_once(module, state_dict, prefix, local_metadata, reached=reached, resume=resume):
            if not reached.is_set():
                reached.set()
                assert resume.wait(60)

        models[name].register_state_dict_post_hook(pause_once)
        pauses[name] = (reached, resume)

    with ThreadPoolExecutor(max_workers=2) as pool:
        saves = {}
        try:
            for name in ("first", "second"):  # both saves start before either converts
                saves[name] = pool.submit(
                    models[name].save_pretrained, tmp_path / f"{name}_quantized"
                )
                assert pauses[name][0].wait(60)
            for name in ("first", "second"):  # the first save ends before the second converts
                pauses[name][1].set()
                saves[name].result()
        finally:  # a failed save must not leave the other one waiting
            for _, resume in pauses.values():
                resume.set()

    for name, model in models.items():
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / f"{name}_quantized")
        torch.testing.assert_close(reloaded.state_dict(), model.state_dict(), rtol=0, atol=0)


def test_save_pretrained_warns_when_from_pretrained_cannot_load_fused_experts(tmp_path):
    """Before transformers 5.21, Qwen3-VL-MoE's load-time conversion also matches its quantizers."""
    text_config = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 32,
        # 2 * moe_intermediate_size != hidden_size, so a transposed expert layout cannot pass
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_heads": 2,
        "out_hidden_size": 32,
        "deepstack_visual_indexes": [0],
    }
    config = Qwen3VLMoeConfig(text_config=text_config, vision_config=vision_config)
    AutoModelForImageTextToText.from_config(config).save_pretrained(tmp_path / "base")
    model = AutoModelForImageTextToText.from_pretrained(tmp_path / "base")
    experts_type = type(model.model.language_model.layers[0].mlp.experts)
    registered_before = QuantModuleRegistry.get(experts_type)
    try:
        _quantize_experts(model)
        if Version(transformers_version) < Version("5.21"):
            with pytest.warns(UserWarning, match="from_pretrained cannot load this checkpoint"):
                model.save_pretrained(tmp_path / "quantized")
            # A load records only the conversions it used; a fresh load still runs the others.
            model._weight_conversions = [
                c for c in get_model_conversion_mapping(model) if not isinstance(c, WeightConverter)
            ][:1]
            with pytest.warns(UserWarning, match="from_pretrained cannot load this checkpoint"):
                model.save_pretrained(tmp_path / "quantized_without_recorded_converters")
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="from_pretrained cannot load")
                model.save_pretrained(tmp_path / "quantized")
            reloaded = AutoModelForImageTextToText.from_pretrained(tmp_path / "quantized")
            torch.testing.assert_close(reloaded.state_dict(), model.state_dict(), rtol=0, atol=0)
    finally:
        # Quantizing registers the experts' class on the fly; other tests expect it unregistered.
        if registered_before is None and QuantModuleRegistry.get(experts_type) is not None:
            QuantModuleRegistry.unregister(experts_type)


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
