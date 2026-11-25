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
import shutil

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import (
    enable_weight_access_and_writeback,
    is_quantized_linear,
)
from modelopt.torch.quantization.utils.layerwise_calib import _layer_dir


def test_cpu_offloaded_tinyllama(tmp_path):
    quant_cfg = mtq.NVFP4_AWQ_LITE_CFG
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


def _make_layerwise_cfg(base_cfg):
    """Add layerwise=True to a quant config's algorithm field."""
    cfg = copy.deepcopy(base_cfg)
    algo = cfg.get("algorithm", "max")
    if isinstance(algo, str):
        cfg["algorithm"] = {"method": algo, "layerwise": True}
    else:
        algo["layerwise"] = True
    return cfg


def _make_layerwise_checkpoint_cfg(base_cfg, checkpoint_dir):
    """Add layerwise=True and layerwise_checkpoint_dir to a quant config's algorithm field."""
    cfg = _make_layerwise_cfg(base_cfg)
    cfg["algorithm"]["layerwise_checkpoint_dir"] = checkpoint_dir
    return cfg


@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["no_ckpt", "ckpt"])
def test_layerwise_calibrate_cpu_offloaded(tmp_path, use_checkpoint):
    """Layerwise calibration on CPU-offloaded model matches GPU-only reference."""
    quant_cfg = mtq.NVFP4_AWQ_LITE_CFG
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    if use_checkpoint:
        ckpt_dir = str(tmp_path / "seq_ckpt")
        seq_cfg = _make_layerwise_checkpoint_cfg(quant_cfg, ckpt_dir)
    else:
        seq_cfg = _make_layerwise_cfg(quant_cfg)

    # Reference: GPU-only model with layerwise calibration
    ref_cfg = _make_layerwise_cfg(quant_cfg)
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


def test_sequential_checkpoint_resume_cpu_offloaded(tmp_path):
    """Resume from a partial checkpoint on a CPU-offloaded model matches a full run."""
    quant_cfg = mtq.NVFP4_AWQ_LITE_CFG
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    ckpt_dir = str(tmp_path / "seq_ckpt")
    seq_ckpt_cfg = _make_layerwise_checkpoint_cfg(quant_cfg, ckpt_dir)

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

    # Simulate crash after layer 0 by truncating the manifest and removing later layers
    last_completed_layer = 0
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": last_completed_layer, "num_layers": num_layers}, f)
    for i in range(last_completed_layer + 1, num_layers):
        d = _layer_dir(ckpt_dir, i)
        if os.path.isdir(d):
            shutil.rmtree(d)

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
    seq_ckpt_cfg = _make_layerwise_checkpoint_cfg(mtq.INT4_AWQ_CFG, ckpt_dir)

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
    last_completed_layer = 0
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": last_completed_layer, "num_layers": num_layers}, f)
    for i in range(last_completed_layer + 1, num_layers):
        d = _layer_dir(ckpt_dir, i)
        if os.path.isdir(d):
            shutil.rmtree(d)

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
    cfg["algorithm"] = {"method": "gptq", "layerwise": True}
    return cfg


def _make_gptq_sequential_checkpoint_cfg(base_cfg, checkpoint_dir):
    """Create a sequential GPTQ config with checkpoint dir."""
    cfg = _make_gptq_sequential_cfg(base_cfg)
    cfg["algorithm"]["layerwise_checkpoint_dir"] = checkpoint_dir
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
        seq_cfg = _make_gptq_sequential_checkpoint_cfg(mtq.NVFP4_AWQ_LITE_CFG, ckpt_dir)
    else:
        seq_cfg = _make_gptq_sequential_cfg(mtq.NVFP4_AWQ_LITE_CFG)

    # Reference: GPU-only model
    ref_cfg = _make_gptq_sequential_cfg(mtq.NVFP4_AWQ_LITE_CFG)
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
    seq_ckpt_cfg = _make_gptq_sequential_checkpoint_cfg(mtq.NVFP4_AWQ_LITE_CFG, ckpt_dir)

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
    last_completed_layer = 0
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": last_completed_layer, "num_layers": num_layers}, f)
    for i in range(last_completed_layer + 1, num_layers):
        d = _layer_dir(ckpt_dir, i)
        if os.path.isdir(d):
            shutil.rmtree(d)

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


class _TupleReturningBlock(torch.nn.Module):
    """Decoder layer that returns a tuple, mimicking HuggingFace decoder layers."""

    def __init__(self, dim=16):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        return (self.linear(x), None)


class _TupleUnpackingModel(torch.nn.Module):
    """Parent model that unpacks layer outputs as tuples."""

    def __init__(self, n_layers=4, dim=16):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TupleReturningBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


def test_skip_dummy_has_no_hf_hook(monkeypatch):
    """Dummies must not carry _hf_hook from the original layer."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    from modelopt.torch.quantization.utils.layerwise_calib import (
        LayerActivationCollector,
        _SkipLayer,
    )

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )

    model = _TupleUnpackingModel(n_layers=4, dim=16)
    data = [torch.randn(2, 16)]

    for layer in model.layers:
        hook = AlignDevicesHook(execution_device=torch.device("cpu"))
        add_hook_to_module(layer, hook)

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        for layer in list(model.layers):
            collector.get_input_activations(layer, forward_loop)

        for i in range(2):
            dummy = model.layers[i]
            assert isinstance(dummy, _SkipLayer)
            assert not hasattr(dummy, "_hf_hook"), f"Dummy at {i} should not have _hf_hook"
    finally:
        collector._unpatch_all_layers()


def test_persistent_materialization_cpu_offloaded(tmp_path):
    """persistent_materialization keeps CPU-offloaded weights on GPU and writes back modifications."""
    import torch.nn as nn
    from accelerate.hooks import AlignDevicesHook

    from modelopt.torch.quantization.utils import persistent_materialization

    model, config, _, inputs = _make_cpu_offloaded_model(tmp_path)
    offloaded_layer = model.model.layers[0]

    # Verify offloaded (meta device)
    assert all(p.device.type == "meta" for p in offloaded_layer.parameters())

    # Save reference weight
    linear = None
    with enable_weight_access_and_writeback(offloaded_layer, model):
        linear = next(m for m in offloaded_layer.modules() if isinstance(m, nn.Linear))
        ref_weight = linear.weight.clone()

    with persistent_materialization(offloaded_layer):
        # Params materialized on GPU
        assert all(
            p.device.type == "cuda" for p in offloaded_layer.parameters() if p.device.type != "meta"
        )

        # Run multiple forward passes (hooks don't re-offload)
        for _ in range(3):
            model(inputs)

        # Modify a weight
        linear.weight.data.add_(1.0)

        # Verify hooks have offload=False during context
        for mod in offloaded_layer.modules():
            if hasattr(mod, "_hf_hook"):
                hook = mod._hf_hook
                if isinstance(hook, AlignDevicesHook):
                    assert not hook.offload

    # After context: back to meta device (offloaded)
    assert all(p.device.type == "meta" for p in offloaded_layer.parameters())

    # Verify weight modification persisted through writeback
    with enable_weight_access_and_writeback(offloaded_layer, model):
        assert torch.allclose(linear.weight, ref_weight + 1.0)


def _make_disk_offload_device_map(model):
    """Build a device_map with layer 0 on disk, everything else on GPU 0.

    Ancestor modules (``""`` and ``"model"``) are excluded so that
    ``dispatch_model`` does not attach a ``place_submodules=True`` hook that
    would try to move disk-offloaded meta tensors to GPU (which fails because
    no ``value`` is available — unlike CPU offload where weights are on CPU and
    can be moved directly).
    """
    device_map = {
        n: 0
        for n, m in model.named_modules()
        if n not in ("", "model") and ("layers" not in n or n.split("layers.")[-1].isdigit())
    }
    device_map["model.layers.0"] = "disk"
    return device_map


def _make_disk_offloaded_model(tmp_path, num_hidden_layers=3):
    """Create a tiny LLaMA model with layer 0 offloaded to disk via accelerate."""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_hidden_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = _make_disk_offload_device_map(model)
    offload_dir = str(tmp_path / "offload")
    model = load_checkpoint_and_dispatch(
        model, tiny_llama_dir, device_map=device_map, offload_folder=offload_dir
    )
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()
    return model, config, tiny_llama_dir, inputs


def test_disk_offloaded_tinyllama(tmp_path):
    quant_cfg = mtq.NVFP4_AWQ_LITE_CFG
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

    device_map = _make_disk_offload_device_map(model)
    offload_dir = str(tmp_path / "offload")
    model = load_checkpoint_and_dispatch(
        model, tiny_llama_dir, device_map=device_map, offload_folder=offload_dir
    )

    assert all(p.device == torch.device("meta") for p in model.model.layers[0].parameters())

    mtq.quantize(model, quant_cfg, lambda model: model(inputs))
    output_test = model(inputs)

    for name, module in model.named_modules():
        if is_quantized_linear(module):
            with enable_weight_access_and_writeback(module, model):
                assert torch.allclose(module.weight, model_ref.get_submodule(name).weight)

    assert torch.allclose(output_ref.logits, output_test.logits)


def test_persistent_materialization_disk_offloaded(tmp_path):
    """persistent_materialization keeps disk-offloaded weights on GPU and writes back modifications."""
    import torch.nn as nn
    from accelerate.hooks import AlignDevicesHook

    from modelopt.torch.quantization.utils import persistent_materialization

    model, config, _, inputs = _make_disk_offloaded_model(tmp_path)
    offloaded_layer = model.model.layers[0]

    # Verify offloaded (meta device)
    assert all(p.device.type == "meta" for p in offloaded_layer.parameters())

    # Save reference weight
    linear = None
    with enable_weight_access_and_writeback(offloaded_layer, model):
        linear = next(m for m in offloaded_layer.modules() if isinstance(m, nn.Linear))
        ref_weight = linear.weight.clone()

    with persistent_materialization(offloaded_layer):
        # Params materialized on GPU
        assert all(
            p.device.type == "cuda" for p in offloaded_layer.parameters() if p.device.type != "meta"
        )

        # Run multiple forward passes (hooks don't re-offload)
        for _ in range(3):
            model(inputs)

        # Modify a weight
        linear.weight.data.add_(1.0)

        # Verify hooks have offload=False during context
        for mod in offloaded_layer.modules():
            if hasattr(mod, "_hf_hook"):
                hook = mod._hf_hook
                if isinstance(hook, AlignDevicesHook):
                    assert not hook.offload

    # After context: back to meta device (offloaded)
    assert all(p.device.type == "meta" for p in offloaded_layer.parameters())

    # Verify weight modification persisted through writeback
    with enable_weight_access_and_writeback(offloaded_layer, model):
        assert torch.allclose(linear.weight, ref_weight + 1.0)


@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["no_ckpt", "ckpt"])
def test_layerwise_calibrate_disk_offloaded(tmp_path, use_checkpoint):
    """Layerwise calibration on disk-offloaded model matches GPU-only reference."""
    quant_cfg = mtq.NVFP4_AWQ_LITE_CFG
    num_layers = 3
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()

    if use_checkpoint:
        ckpt_dir = str(tmp_path / "seq_ckpt")
        seq_cfg = _make_layerwise_checkpoint_cfg(quant_cfg, ckpt_dir)
    else:
        seq_cfg = _make_layerwise_cfg(quant_cfg)

    # Reference: GPU-only model with layerwise calibration
    ref_cfg = _make_layerwise_cfg(quant_cfg)
    model_ref = AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir, torch_dtype=config.torch_dtype
    ).cuda()
    mtq.quantize(model_ref, ref_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    # Test: disk-offloaded model
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    device_map = _make_disk_offload_device_map(model)
    offload_dir = str(tmp_path / "offload")
    model = load_checkpoint_and_dispatch(
        model, tiny_llama_dir, device_map=device_map, offload_folder=offload_dir
    )

    mtq.quantize(model, seq_cfg, lambda model: model(inputs))
    output_test = model(inputs)

    for name, module in model.named_modules():
        if is_quantized_linear(module):
            with enable_weight_access_and_writeback(module, model):
                assert torch.allclose(module.weight, model_ref.get_submodule(name).weight), (
                    f"Weight mismatch at {name}"
                )

    assert torch.allclose(output_ref.logits, output_test.logits)
