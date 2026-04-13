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
import json
import os

import pytest
import torch
from _test_utils.torch.quantization.quantize_common import INT4_AWQ_CLIP_CFG
from _test_utils.torch.transformers_models import create_tiny_llama_dir

import modelopt.torch.quantization as mtq
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from modelopt.torch.quantization.utils import (
    enable_weight_access_and_writeback,
    is_quantized_linear,
)
from transformers import AutoConfig, AutoModelForCausalLM


@pytest.mark.parametrize(
    "quant_cfg",
    [
        mtq.INT4_AWQ_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        INT4_AWQ_CLIP_CFG,
        mtq.NVFP4_SVDQUANT_DEFAULT_CFG,
        mtq.INT8_DEFAULT_CFG,
    ],
)
def test_cpu_offloaded_tinyllama(tmp_path, quant_cfg):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=2)

    config = AutoConfig.from_pretrained(tiny_llama_dir)

    model_ref = AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir, torch_dtype=config.torch_dtype
    ).cuda()
    inputs = torch.randint(0, model_ref.config.vocab_size, (1, 4)).cuda()

    mtq.quantize(model_ref, quant_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = {
        n: 0
        for n, m in model.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"

    model = load_checkpoint_and_dispatch(model, tiny_llama_dir, device_map=device_map)

    assert all(p.device == torch.device("meta") for p in model.model.layers[0].parameters())

    mtq.quantize(model, quant_cfg, lambda model: model(inputs))
    output_test = model(inputs)

    for name, module in model.named_modules():
        if is_quantized_linear(module):
            with enable_weight_access_and_writeback(module, model):
                assert torch.allclose(module.weight, model_ref.get_submodule(name).weight)

    assert torch.allclose(output_ref.logits, output_test.logits)


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
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()
    return model, config, tiny_llama_dir, inputs


def _make_sequential_cfg(base_cfg):
    """Add use_sequential=True to a quant config's algorithm field."""
    cfg = copy.deepcopy(base_cfg)
    algo = cfg.get("algorithm", "max")
    if isinstance(algo, str):
        cfg["algorithm"] = {"method": algo, "use_sequential": True}
    else:
        algo["use_sequential"] = True
    return cfg


def _make_sequential_checkpoint_cfg(base_cfg, checkpoint_dir):
    """Add use_sequential=True and checkpoint_dir to a quant config's algorithm field."""
    cfg = _make_sequential_cfg(base_cfg)
    cfg["algorithm"]["checkpoint_dir"] = checkpoint_dir
    return cfg


@pytest.mark.parametrize(
    "quant_cfg",
    [mtq.INT4_AWQ_CFG, mtq.NVFP4_DEFAULT_CFG],
    ids=["int4_awq", "nvfp4"],
)
@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["no_ckpt", "ckpt"])
def test_sequential_calibrate_cpu_offloaded(tmp_path, quant_cfg, use_checkpoint):
    """Sequential calibration on CPU-offloaded model matches GPU-only reference."""
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    if use_checkpoint:
        ckpt_dir = str(tmp_path / "seq_ckpt")
        seq_cfg = _make_sequential_checkpoint_cfg(quant_cfg, ckpt_dir)
    else:
        seq_cfg = _make_sequential_cfg(quant_cfg)

    # Reference: GPU-only model with sequential calibration
    ref_cfg = _make_sequential_cfg(quant_cfg)
    model_ref = AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir, torch_dtype=config.torch_dtype
    ).cuda()
    mtq.quantize(model_ref, ref_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    # Test: CPU-offloaded model
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    device_map = {
        n: 0
        for n, m in model.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"
    model = load_checkpoint_and_dispatch(model, tiny_llama_dir, device_map=device_map)

    mtq.quantize(model, seq_cfg, lambda model: model(inputs))
    output_test = model(inputs)

    for name, module in model.named_modules():
        if is_quantized_linear(module):
            with enable_weight_access_and_writeback(module, model):
                assert torch.allclose(module.weight, model_ref.get_submodule(name).weight), (
                    f"Weight mismatch at {name}"
                )

    assert torch.allclose(output_ref.logits, output_test.logits)

    if use_checkpoint:
        manifest_path = os.path.join(ckpt_dir, "manifest.json")
        assert os.path.isfile(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["last_completed_layer"] == num_layers - 1
        assert manifest["num_layers"] == num_layers


@pytest.mark.parametrize(
    "quant_cfg",
    [mtq.INT4_AWQ_CFG, mtq.NVFP4_DEFAULT_CFG],
    ids=["int4_awq", "nvfp4"],
)
def test_sequential_checkpoint_resume_cpu_offloaded(tmp_path, quant_cfg):
    """Resume from a partial checkpoint on a CPU-offloaded model matches a full run."""
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    ckpt_dir = str(tmp_path / "seq_ckpt")
    seq_ckpt_cfg = _make_sequential_checkpoint_cfg(quant_cfg, ckpt_dir)

    # Full reference run with checkpointing
    with init_empty_weights():
        model_ref = AutoModelForCausalLM.from_config(config)
    device_map = {
        n: 0
        for n, m in model_ref.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"
    model_ref = load_checkpoint_and_dispatch(model_ref, tiny_llama_dir, device_map=device_map)
    mtq.quantize(model_ref, seq_ckpt_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    # Simulate crash after layer 0 by truncating the manifest
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": 0, "num_layers": num_layers}, f)

    # Resume from a fresh CPU-offloaded model
    with init_empty_weights():
        model_resumed = AutoModelForCausalLM.from_config(config)
    device_map = {
        n: 0
        for n, m in model_resumed.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"
    model_resumed = load_checkpoint_and_dispatch(
        model_resumed, tiny_llama_dir, device_map=device_map
    )
    mtq.quantize(model_resumed, seq_ckpt_cfg, lambda model: model(inputs))
    output_resumed = model_resumed(inputs)

    assert torch.allclose(output_ref.logits, output_resumed.logits), (
        "Resumed checkpoint should produce identical output to full run"
    )


def test_sequential_checkpoint_resume_multi_offload(tmp_path):
    """Resume with multiple layers offloaded exercises per-layer device resolution."""
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    ckpt_dir = str(tmp_path / "seq_ckpt")
    seq_ckpt_cfg = _make_sequential_checkpoint_cfg(mtq.INT4_AWQ_CFG, ckpt_dir)

    def _make_multi_offload_model():
        with init_empty_weights():
            m = AutoModelForCausalLM.from_config(config)
        dmap = {
            n: 0
            for n, mod in m.named_modules()
            if "layers" not in n or n.split("layers.")[-1].isdigit()
        }
        dmap["model.layers.0"] = "cpu"
        dmap["model.layers.1"] = "cpu"
        return load_checkpoint_and_dispatch(m, tiny_llama_dir, device_map=dmap)

    # Full reference run
    model_ref = _make_multi_offload_model()
    mtq.quantize(model_ref, seq_ckpt_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    # Simulate crash after layer 0
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": 0, "num_layers": num_layers}, f)

    # Resume from fresh model with same offload layout
    model_resumed = _make_multi_offload_model()
    mtq.quantize(model_resumed, seq_ckpt_cfg, lambda model: model(inputs))
    output_resumed = model_resumed(inputs)

    assert torch.allclose(output_ref.logits, output_resumed.logits), (
        "Resumed checkpoint with multi-offload should match full run"
    )


def _make_gptq_sequential_cfg(base_cfg):
    """Create a sequential GPTQ config from a base quantization config."""
    cfg = copy.deepcopy(base_cfg)
    cfg["algorithm"] = {"method": "gptq", "use_sequential": True}
    return cfg


def _make_gptq_sequential_checkpoint_cfg(base_cfg, checkpoint_dir):
    """Create a sequential GPTQ config with checkpoint dir."""
    cfg = _make_gptq_sequential_cfg(base_cfg)
    cfg["algorithm"]["checkpoint_dir"] = checkpoint_dir
    return cfg


@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["no_ckpt", "ckpt"])
def test_sequential_gptq_cpu_offloaded(tmp_path, use_checkpoint):
    """Sequential GPTQ (weight-modifying) on CPU-offloaded model matches GPU-only reference."""
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    if use_checkpoint:
        ckpt_dir = str(tmp_path / "gptq_ckpt")
        seq_cfg = _make_gptq_sequential_checkpoint_cfg(mtq.NVFP4_DEFAULT_CFG, ckpt_dir)
    else:
        seq_cfg = _make_gptq_sequential_cfg(mtq.NVFP4_DEFAULT_CFG)

    # Reference: GPU-only model
    ref_cfg = _make_gptq_sequential_cfg(mtq.NVFP4_DEFAULT_CFG)
    model_ref = AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir, torch_dtype=config.torch_dtype
    ).cuda()
    mtq.quantize(model_ref, ref_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    # Test: CPU-offloaded model
    model, _, _, _ = _make_cpu_offloaded_model(tmp_path / "offloaded", num_hidden_layers=num_layers)
    mtq.quantize(model, seq_cfg, lambda model: model(inputs))
    output_test = model(inputs)

    for name, module in model.named_modules():
        if is_quantized_linear(module):
            with enable_weight_access_and_writeback(module, model):
                assert torch.allclose(module.weight, model_ref.get_submodule(name).weight), (
                    f"Weight mismatch at {name}"
                )

    assert torch.allclose(output_ref.logits, output_test.logits)


def test_sequential_gptq_checkpoint_resume_cpu_offloaded(tmp_path):
    """GPTQ checkpoint resume with CPU offloading restores modified weights correctly."""
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    ckpt_dir = str(tmp_path / "gptq_ckpt")
    seq_ckpt_cfg = _make_gptq_sequential_checkpoint_cfg(mtq.NVFP4_DEFAULT_CFG, ckpt_dir)

    # Full reference run with checkpointing
    with init_empty_weights():
        model_ref = AutoModelForCausalLM.from_config(config)
    device_map = {
        n: 0
        for n, m in model_ref.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"
    model_ref = load_checkpoint_and_dispatch(model_ref, tiny_llama_dir, device_map=device_map)
    mtq.quantize(model_ref, seq_ckpt_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    # Simulate crash after layer 0
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": 0, "num_layers": num_layers}, f)

    # Resume from fresh CPU-offloaded model
    with init_empty_weights():
        model_resumed = AutoModelForCausalLM.from_config(config)
    device_map = {
        n: 0
        for n, m in model_resumed.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"
    model_resumed = load_checkpoint_and_dispatch(
        model_resumed, tiny_llama_dir, device_map=device_map
    )
    mtq.quantize(model_resumed, seq_ckpt_cfg, lambda model: model(inputs))
    output_resumed = model_resumed(inputs)

    assert torch.allclose(output_ref.logits, output_resumed.logits), (
        "GPTQ resumed checkpoint should produce identical output to full run"
    )
