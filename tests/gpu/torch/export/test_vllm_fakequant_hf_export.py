# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import copy
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_vllm_fq_checkpoint
from modelopt.torch.quantization.model_quant import fold_weight
from modelopt.torch.quantization.utils import enable_weight_access_and_writeback
from modelopt.torch.utils import safe_load


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG])
def test_hf_vllm_export(tmp_path, quant_cfg):
    """Test HuggingFace model export for vLLM with fake quantization.

    This test verifies:
    1. Input model is NOT mutated by export (weights and quantizer state unchanged)
    2. Exported weights match folded (fake-quantized) weights
    3. vllm_fq_modelopt_state.pth is created; hf_quant_config.json is not
    4. Weight quantizer states are empty in saved state dict; input quantizer amaxes preserved
    """

    # Create a tiny LLaMA model for testing
    tiny_model_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=2)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    model = model.cuda()
    model.eval()

    # Quantize the model
    def forward_loop(model):
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128)).cuda()
        with torch.no_grad():
            model(input_ids)

    model = mtq.quantize(model, quant_cfg, forward_loop)
    quantizer_state_dict_before = mtq.utils.get_quantizer_state_dict(model)

    # Compute expected exported weights: deepcopy → fold (export writes folded weights)
    folded_model = deepcopy(model)
    fold_weight(folded_model)
    expected_weights = {k: v for k, v in folded_model.state_dict().items() if "quantizer" not in k}
    del folded_model

    # Snapshot model state before export to verify it is not mutated
    state_dict_before_export = {k: v.clone() for k, v in model.state_dict().items()}

    # Export directory
    export_dir = tmp_path / "vllm_export"
    export_dir.mkdir(exist_ok=True)

    export_hf_vllm_fq_checkpoint(model, export_dir=export_dir)

    # Verify the input model is not mutated: all state dict values unchanged
    state_dict_after_export = model.state_dict()
    for key, param_before in state_dict_before_export.items():
        assert torch.allclose(param_before, state_dict_after_export[key], atol=0), (
            f"Model was mutated by export: {key} changed"
        )

    # check if vllm_fq_modelopt_state.pth file exists
    modelopt_state_file = export_dir / "vllm_fq_modelopt_state.pth"
    assert modelopt_state_file.exists(), (
        f"vllm_fq_modelopt_state.pth file should be created in {export_dir}"
    )

    # make sure hf_quant_config.json file does not exist
    hf_quant_config_file = export_dir / "hf_quant_config.json"
    assert not hf_quant_config_file.exists(), (
        f"hf_quant_config.json file should not be created in {export_dir}"
    )

    # check folded weights match exported model weights
    model_after = AutoModelForCausalLM.from_pretrained(export_dir)
    model_after = model_after.cuda()
    model_after.eval()
    model_after_state_dict = model_after.state_dict()
    for key, param in expected_weights.items():
        assert torch.allclose(param, model_after_state_dict[key], atol=1e-6), (
            f"Weight mismatch for {key}: "
            f"before shape={param.shape}, after shape={model_after_state_dict[key].shape}, "
            f"max diff={torch.abs(param - model_after_state_dict[key]).max()}"
        )

    # Verify quantizer state dict: same keys, weight quantizer amaxes cleared, input amaxes kept
    quantizer_state_dict = safe_load(modelopt_state_file)["modelopt_state_weights"]
    assert len(quantizer_state_dict) > 0, (
        f"modelopt_state_weights should not be empty in {modelopt_state_file}"
    )
    for name, state in quantizer_state_dict.items():
        if "weight_quantizer" in name:
            assert state == {}, f"weight quantizer {name} should have empty state after fold"
        elif "input_quantizer" in name and any(
            "_amax" in k for k in quantizer_state_dict_before[name]
        ):
            assert any("_amax" in k for k in state), f"input quantizer {name} should preserve _amax"


def _make_cpu_offloaded_model(tmp_path, num_hidden_layers=3):
    """Create a tiny LLaMA model with layer 0 offloaded to CPU via accelerate."""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_hidden_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = {
        n: 0
        for n, m in model.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"

    model = load_checkpoint_and_dispatch(model, tiny_llama_dir, device_map=device_map)
    return model, config, tiny_llama_dir


def _make_layerwise_cfg(base_cfg):
    """Add layerwise=True to a quant config's algorithm field."""
    cfg = copy.deepcopy(base_cfg)
    algo = cfg.get("algorithm", "max")
    if isinstance(algo, str):
        cfg["algorithm"] = {"method": algo, "layerwise": True}
    else:
        algo["layerwise"] = True
    return cfg


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG])
def test_hf_vllm_export_offload(tmp_path, quant_cfg):
    """Verifies the inplace_mem_efficient=True path mutates offloaded weights in place
    and produces folded values matching deepcopy+fold_weight reference. Does NOT
    exercise save_pretrained -- transformers' load_offloaded_parameter doesn't unwrap
    SequentialHook, a pre-existing limitation unrelated to this PR's new code.
    """
    num_hidden_layers = 3

    model, _config, _tiny_llama_dir = _make_cpu_offloaded_model(
        tmp_path / "offloaded", num_hidden_layers=num_hidden_layers
    )
    model.eval()

    seq_cfg = _make_layerwise_cfg(quant_cfg)

    def forward_loop(model):
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128)).cuda()
        with torch.no_grad():
            model(input_ids)

    model = mtq.quantize(model, seq_cfg, forward_loop)
    quantizer_state_dict_before = mtq.utils.get_quantizer_state_dict(model)

    folded_model = deepcopy(model)
    with enable_weight_access_and_writeback(folded_model.model.layers[0], folded_model):
        fold_weight(folded_model)
        expected_weights = {
            k: v.detach().clone()
            for k, v in folded_model.state_dict().items()
            if "quantizer" not in k
        }
    del folded_model

    export_dir = tmp_path / "vllm_export_offload"
    export_dir.mkdir(exist_ok=True)

    # Snapshot the offloaded layer's weight before/after export to verify the
    # inplace_mem_efficient path actually mutates offloaded weights (would otherwise
    # be unfalsifiable if the function silently took the copy path).
    with enable_weight_access_and_writeback(model.model.layers[0], model):
        weight_before = model.model.layers[0].self_attn.q_proj.weight.data.clone()

    # Skip save_pretrained: transformers' load_offloaded_parameter doesn't unwrap
    # SequentialHook, a pre-existing upstream limitation unrelated to this PR. The
    # delta under test is inplace fake-quant + weight writeback, which runs before
    # save_pretrained.
    original_save_pretrained = model.save_pretrained
    model.save_pretrained = lambda *args, **kwargs: None
    try:
        export_hf_vllm_fq_checkpoint(model, export_dir=export_dir, inplace_mem_efficient=True)
    finally:
        model.save_pretrained = original_save_pretrained

    with enable_weight_access_and_writeback(model.model.layers[0], model):
        weight_after = model.model.layers[0].self_attn.q_proj.weight.data.clone()
    assert not torch.equal(weight_before, weight_after), (
        "inplace path must mutate offloaded weights"
    )

    with enable_weight_access_and_writeback(model.model.layers[0], model):
        actual_weights = {
            k: v.detach().clone() for k, v in model.state_dict().items() if "quantizer" not in k
        }
    for key, expected in expected_weights.items():
        actual = actual_weights.get(key)
        assert actual is not None, f"missing {key} after export"
        assert torch.allclose(actual, expected, atol=1e-6), f"mismatch at {key}"

    modelopt_state_file = export_dir / "vllm_fq_modelopt_state.pth"
    assert modelopt_state_file.exists(), (
        f"vllm_fq_modelopt_state.pth file should be created in {export_dir}"
    )

    quantizer_state_dict = safe_load(modelopt_state_file)["modelopt_state_weights"]
    assert len(quantizer_state_dict) > 0, (
        f"modelopt_state_weights should not be empty in {modelopt_state_file}"
    )
    for name, state in quantizer_state_dict.items():
        if "weight_quantizer" in name:
            assert state == {}, f"weight quantizer {name} should have empty state after fold"
        elif "input_quantizer" in name and any(
            "_amax" in k for k in quantizer_state_dict_before[name]
        ):
            assert any("_amax" in k for k in state), f"input quantizer {name} should preserve _amax"
