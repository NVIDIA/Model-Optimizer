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

import json
from collections import Counter
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import partial
from importlib.util import find_spec

import pytest
import torch
import yaml
from _test_utils.torch.megatron.models import get_mcore_gpt_model, get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import run_mcore_inference
from _test_utils.torch.transformers_models import create_tiny_llama_dir, create_tiny_nemotron_h_dir
from megatron.core.parallel_state import is_pipeline_last_stage
from safetensors import safe_open

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_mcore_gpt_to_hf_vllm_fq
from modelopt.torch.export.plugins.vllm_fakequant_megatron import (
    VllmFqGPTModelExporter,
    gather_mcore_vllm_fq_quantized_state_dict,
    gather_mcore_vllm_fq_quantizer_recipe,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@contextmanager
def _assert_weight_qdq_once(model, prefix=""):
    quantizers = [
        module
        for name, module in model.named_modules()
        if name.startswith(prefix)
        and name.endswith("weight_quantizer")
        and isinstance(module, TensorQuantizer)
    ]
    assert quantizers or (prefix and not is_pipeline_last_stage())
    calls = Counter()

    def count_qdq(module, args, output):
        calls[module] += 1

    handles = [quantizer.register_forward_hook(count_qdq) for quantizer in quantizers]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
    assert all(calls[quantizer] == 1 for quantizer in quantizers), calls


def _assert_exported_quantizers(export_dir, expected_names, amax=1.001, disabled_names=()):
    state = torch.load(export_dir / "quantizer_state.pth", weights_only=True, map_location="cpu")
    recipe = yaml.safe_load((export_dir / "quant_recipe.yaml").read_text())
    assert expected_names <= recipe.keys()
    assert {name + "._amax" for name in expected_names} <= state.keys()
    for name in expected_names:
        assert recipe[name]["_disabled"] == (name in disabled_names)
        tensor = state[name + "._amax"]
        assert tensor.dtype == torch.float32
        expected_amax = amax[name] if isinstance(amax, dict) else amax
        torch.testing.assert_close(tensor, torch.full_like(tensor, expected_amax), rtol=0, atol=0)
    assert {key.rsplit(".", 1)[0] for key in state} <= recipe.keys()
    assert not any(key.endswith("._quant_recipe_marker") for key in state)
    assert not any(key.endswith("._quant_recipe_marker") for key in recipe)
    assert not (export_dir / "hf_quant_config.json").exists()
    weight_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(weight_map.values()):
        with safe_open(export_dir / shard, framework="pt") as f:
            shard_keys = f.keys()
            assert not any(
                "quantizer" in key or "._quant_recipe_marker" in key for key in shard_keys
            )
    return state, recipe, weight_map


def _test_mcore_vllm_export(tmp_path, rank, size):
    """Test megatron-core model export for vLLM with fake quantization."""
    # Create a tiny mcore GPT model
    num_layers = 2
    hidden_size = 64
    num_attention_heads = 8
    num_query_groups = 1
    ffn_hidden_size = 128
    max_sequence_length = 32
    vocab_size = 64

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="swiglu",
        normalization="RMSNorm",
        transformer_impl="modelopt",
    ).cuda()
    model.eval()

    # Quantize the model
    def forward_loop(model):
        batch_size = 1
        seq_len = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            run_mcore_inference(model, input_ids)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    attribute_cfg = {
        "use_constant_amax": True,
        "unsigned": True,
        "narrow_range": True,
        "type": "dynamic",
    }
    inactive_cfg = {
        "num_bits": 8,
        "unsigned": True,
        "narrow_range": True,
        "fake_quant": False,
        "type": "dynamic",
        "bias": {-1: None},
        "backend": "unused",
    }
    # Preserve calibration precision even when the exported weights are BF16.
    for name, quantizer in model.named_modules():
        if isinstance(quantizer, TensorQuantizer) and "input_quantizer" in name:
            if getattr(quantizer, "_amax", None) is not None:
                quantizer.float()
                quantizer.amax = torch.full_like(quantizer.amax, 1.001)
            if name.startswith("decoder.layers.0."):
                quantizer.set_from_attribute_config(attribute_cfg)
                if name.endswith("linear_qkv.input_quantizer"):
                    quantizer.reset_amax()
                elif name.endswith("linear_fc1.input_quantizer"):
                    quantizer.pre_quant_scale = torch.full((hidden_size,), 2.0, device="cuda")
                    quantizer.disable_quant()
                    quantizer.set_from_attribute_config(inactive_cfg)
                elif name.endswith("linear_fc2.input_quantizer"):
                    quantizer.pre_quant_scale = torch.full((ffn_hidden_size,), 2.0, device="cuda")
                    quantizer._enable_pre_quant_scale = False
                    quantizer.disable()
                    quantizer.set_from_attribute_config({**inactive_cfg, "num_bits": (7, 3)})
    layer = model.decoder.layers[0]
    linears = (
        (
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            layer.self_attention.linear_qkv,
        ),
        (("self_attn.o_proj",), layer.self_attention.linear_proj),
        (("mlp.gate_proj", "mlp.up_proj"), layer.mlp.linear_fc1),
        (("mlp.down_proj",), layer.mlp.linear_fc2),
    )
    expected_weights = []
    for projections, module in linears:
        quantizer = module.weight_quantizer
        if projections[0] == "self_attn.q_proj":
            quantizer.set_from_attribute_config(
                {"num_bits": 8, "unsigned": True, "type": "dynamic"}
            )
            quantizer.reset_amax()
            with torch.no_grad():
                module.weight.abs_()
        elif projections[0] == "self_attn.o_proj":
            quantizer.set_from_attribute_config(
                {"num_bits": 8, "narrow_range": True, "bias": {-1: None}}
            )
            quantizer.bias_value = torch.tensor(0.025, device="cuda")
        elif projections[0] == "mlp.gate_proj":
            quantizer.set_from_attribute_config(
                {
                    "enable": False,
                    "rotate": find_spec("fast_hadamard_transform") is not None,
                    "fake_quant": False,
                    "backend": "unused",
                }
            )
            quantizer.pre_quant_scale = torch.full((hidden_size,), 2.0, device="cuda")
        else:
            quantizer.set_from_attribute_config({"num_bits": "q8_0", "backend": "ggml"})
        with torch.no_grad():
            expected_weights.append(
                quantizer(module.weight.to(torch.bfloat16)).to(torch.bfloat16).cpu()
            )
    # Create HF config for export
    pretrained_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden_size,
        "max_position_embeddings": max_sequence_length,
        "model_type": "llama",
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_query_groups,
        "torch_dtype": "bfloat16",
        "vocab_size": vocab_size,
    }

    stale_quantizer = "stale_source.input_quantizer"
    if rank == 0:
        with open(tmp_path / "config.json", "w") as f:
            json.dump(pretrained_config, f)
        torch.save(
            {stale_quantizer + "._amax": torch.tensor(42.0)}, tmp_path / "quantizer_state.pth"
        )
        with open(tmp_path / "quant_recipe.yaml", "w") as f:
            yaml.safe_dump({stale_quantizer: {"_disabled": True}}, f)
    torch.distributed.barrier()

    export_dir = tmp_path / "vllm_export"
    with _assert_weight_qdq_once(model):
        exporter = VllmFqGPTModelExporter(model, tmp_path, dtype=torch.bfloat16)
        _ = exporter.state_dict
        assert exporter.layer_state_dicts
        exporter.save_pretrained(str(export_dir), tmp_path)

    expected_names = {
        *(
            f"model.layers.{i}.self_attn.{proj}_proj.input_quantizer"
            for i in range(num_layers)
            for proj in ("q", "k", "v", "o")
        ),
        *(
            f"model.layers.{i}.mlp.{proj}_proj.input_quantizer"
            for i in range(num_layers)
            for proj in ("gate", "up", "down")
        ),
    }
    disabled_names = {name for name in expected_names if name.startswith("model.layers.0.mlp.")}
    state, recipe, weight_map = _assert_exported_quantizers(
        export_dir,
        expected_names,
        amax={
            name: 448.0 if name.startswith("model.layers.0.") else 1.001 for name in expected_names
        },
        disabled_names=disabled_names,
    )
    inputs = torch.linspace(-1000, 1000, 64, device="cuda").reshape(1, -1)
    for (projections, module), expected_weight in zip(linears, expected_weights):
        folded_weights = []
        for projection in projections:
            prefix = f"model.layers.{layer.layer_number - 1}.{projection}"
            weight_key = prefix + ".weight"
            with safe_open(export_dir / weight_map[weight_key], framework="pt") as f:
                folded_weights.append(f.get_tensor(weight_key))
            assert recipe[prefix + ".weight_quantizer"]["_disabled"]
        torch.testing.assert_close(torch.cat(folded_weights), expected_weight, rtol=0, atol=0)
        projection = projections[0]
        name = f"model.layers.{layer.layer_number - 1}.{projection}.input_quantizer"
        config = recipe[name]
        restored = TensorQuantizer(
            mtq.QuantizerAttributeConfig(
                num_bits=config.get("_num_bits", 8),
                axis=config.get("_axis"),
                block_sizes=config.get("_block_sizes"),
                enable=not config["_disabled"],
            )
        ).cuda()
        restored.amax = state[name + "._amax"].cuda()
        scale = state.get(name + "._pre_quant_scale")
        if scale is not None:
            restored.pre_quant_scale = scale.cuda()
        torch.testing.assert_close(restored(inputs), module.input_quantizer(inputs), rtol=0, atol=0)
        if projection == "self_attn.q_proj":
            assert not hasattr(module.input_quantizer, "_amax")
        else:
            torch.testing.assert_close(
                module.input_quantizer._amax, torch.tensor(1.001, device="cuda"), rtol=0, atol=0
            )
        if projection == "mlp.gate_proj":
            assert scale is not None
        elif projection == "mlp.down_proj":
            assert scale is None
            assert hasattr(module.input_quantizer, "_pre_quant_scale")
    assert stale_quantizer + "._amax" not in state
    assert stale_quantizer not in recipe
    assert {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        *(f"model.layers.{i}.self_attn.q_proj.weight" for i in range(num_layers)),
    } <= weight_map.keys()


def test_mcore_vllm_export(dist_workers_size_1, tmp_path):
    """Cached export preserves default and supported quantizers from separate layers."""
    dist_workers_size_1.run(partial(_test_mcore_vllm_export, tmp_path))


def _test_mcore_vllm_export_mtp(tmp_path, rank, size):
    model = get_mcore_hybrid_model(
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=4,
        hybrid_layer_pattern="M*EE/*E",
        num_query_groups=4,
        max_sequence_length=32,
        vocab_size=32,
        mamba_num_heads=8,
        num_moe_experts=4,
        normalization="RMSNorm",
        mtp_num_layers=1,
    ).cuda()
    model.eval()

    def forward_loop(model):
        with torch.no_grad():
            run_mcore_inference(model, torch.randint(0, 32, (1, 32), device="cuda"))

    quant_cfg = deepcopy(mtq.FP8_DEFAULT_CFG)
    # The default preset excludes MTP; this regression exercises a quantized live head.
    quant_cfg["quant_cfg"] = [
        entry for entry in quant_cfg["quant_cfg"] if entry.get("quantizer_name") != "mtp.*"
    ]
    model = mtq.quantize(model, quant_cfg, forward_loop)
    mtp_quantizers = {
        name: quantizer
        for name, quantizer in model.named_modules()
        if name.startswith("mtp.") and isinstance(quantizer, TensorQuantizer)
    }
    if is_pipeline_last_stage():
        assert mtp_quantizers
        for name, quantizer in mtp_quantizers.items():
            if quantizer.is_enabled:
                assert getattr(quantizer, "_amax", None) is not None
                if name.endswith("input_quantizer"):
                    quantizer.float()
                    quantizer.amax = torch.full_like(quantizer.amax, 1.001)

    source = tmp_path / "tiny_nemotron_h"
    if rank == 0:
        create_tiny_nemotron_h_dir(
            tmp_path,
            num_hidden_layers=4,
            hybrid_override_pattern="M*EE",
            n_routed_experts=4,
            num_nextn_predict_layers=1,
        )
    torch.distributed.barrier()

    export_dir = tmp_path / "mtp_export"
    with _assert_weight_qdq_once(model, prefix="mtp."):
        export_mcore_gpt_to_hf_vllm_fq(
            model,
            pretrained_model_name_or_path=str(source),
            dtype=torch.bfloat16,
            export_dir=str(export_dir),
        )
    expected_names = {
        "mtp.layers.0.eh_proj.input_quantizer",
        *(f"mtp.layers.0.mixer.{proj}_proj.input_quantizer" for proj in ("q", "k", "v", "o")),
        *(
            f"mtp.layers.1.mixer.experts.{expert}.{proj}_proj.input_quantizer"
            for expert in range(4)
            for proj in ("up", "down")
        ),
        *(
            f"mtp.layers.1.mixer.shared_experts.{proj}_proj.input_quantizer"
            for proj in ("up", "down")
        ),
    }
    _, _, weight_map = _assert_exported_quantizers(export_dir, expected_names)
    assert "mtp.layers.0.eh_proj.weight" in weight_map


def test_mcore_vllm_export_mtp(request, tmp_path):
    """Live Nemotron MTP quantizers reach state and recipe files without leaking into weights."""
    workers = request.getfixturevalue(f"dist_workers_size_{min(torch.cuda.device_count(), 2)}")
    workers.run(partial(_test_mcore_vllm_export_mtp, tmp_path))


def _test_mcore_vllm_export_unsupported_setting(tmp_path, attribute_cfgs, rank, size):
    model = get_mcore_gpt_model(
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        normalization="RMSNorm",
        transformer_impl="modelopt",
    ).cuda()
    quant_cfg = deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg["algorithm"] = None
    model = mtq.quantize(model, quant_cfg)
    for name, quantizer in model.named_modules():
        if isinstance(quantizer, TensorQuantizer) and name.endswith("input_quantizer"):
            quantizer.amax = 1.0
    if rank == size - 1:
        linear = next(
            module
            for module in model.modules()
            if isinstance(getattr(module, "input_quantizer", None), TensorQuantizer)
            and module.input_quantizer.is_enabled
        )
        original_quantizer = linear.input_quantizer

    source = tmp_path / "tiny_llama"
    if rank == 0:
        create_tiny_llama_dir(tmp_path)
    torch.distributed.barrier()
    export_dir = tmp_path / "unsupported_export"
    for attribute_cfg in attribute_cfgs:
        if rank == size - 1:
            linear.input_quantizer = deepcopy(original_quantizer)
            quantizer = linear.input_quantizer
            quantizer.set_from_attribute_config(attribute_cfg)
            if "type" in attribute_cfg:
                quantizer.reset_amax()
        setting = (
            "dynamic_amax"
            if "type" in attribute_cfg
            else next(key for key in attribute_cfg if key != "enable")
        )
        with pytest.raises(ValueError, match=f"Unsupported.*input_quantizer: {setting}"):
            export_mcore_gpt_to_hf_vllm_fq(model, source, export_dir=str(export_dir))
        assert not export_dir.exists()


def test_mcore_vllm_export_unsupported_setting(request, tmp_path):
    """Unsupported settings reject export on every rank, including a final-stage PP2 error."""
    attribute_cfgs = [
        {"unsigned": True, "num_bits": 8},
        {"narrow_range": True, "num_bits": 8},
        {"rotate": True},
        {"rotate": {"enable": True, "rotate_fp32": True}},
        {"enable": False, "rotate": True},
        {"fake_quant": False},
        {"type": "dynamic"},
        {"type": "static"},
        {"bias": {-1: None}},
        {"backend": "custom"},
    ]
    workers = request.getfixturevalue(f"dist_workers_size_{min(torch.cuda.device_count(), 2)}")
    workers.run(partial(_test_mcore_vllm_export_unsupported_setting, tmp_path, attribute_cfgs))


def _test_cross_rank_recipe_merge(tmp_path, error, rank, size):
    assert size == 2
    name = "model.layers.0.self_attn.q_proj.input_quantizer"
    recipe = {"_num_bits": 4 if rank == 1 and error is ValueError else 8}
    if rank == 0 and error is RuntimeError:
        (tmp_path / "quant_recipe.yaml").mkdir()
    match = (
        "Conflicting quantizer recipes"
        if error is ValueError
        else "Failed to save quant_recipe.yaml"
    )
    with pytest.raises(error, match=match) if error else nullcontext():
        gather_mcore_vllm_fq_quantizer_recipe({name: recipe}, tmp_path)
    if error is None:
        with open(tmp_path / "quant_recipe.yaml") as f:
            assert yaml.safe_load(f) == {name: recipe}


def _test_cross_rank_tensor_merge(tmp_path, error, rank, size):
    assert size == 2
    name = "model.layers.0.self_attn.q_proj.input_quantizer._amax"
    tensor = torch.tensor([1.0 + rank if error is ValueError else 1.0])
    match = "Conflicting quantizer tensors"
    with pytest.raises(error, match=match) if error else nullcontext():
        gather_mcore_vllm_fq_quantized_state_dict(None, {1: {name: tensor}}, tmp_path)
    if error is None:
        state = torch.load(tmp_path / "quantizer_state.pth", weights_only=True)
        torch.testing.assert_close(state[name], tensor, rtol=0, atol=0)


def test_cross_rank_quantizer_merge(dist_workers_size_2, tmp_path):
    """Check duplicate recipes and tensors, conflicts, and the shared write failure path."""
    for error in (None, ValueError):
        case_dir = tmp_path / (error.__name__ if error else "matching")
        case_dir.mkdir()
        dist_workers_size_2.run(partial(_test_cross_rank_recipe_merge, case_dir, error))
        dist_workers_size_2.run(partial(_test_cross_rank_tensor_merge, case_dir, error))
    failure_dir = tmp_path / "write_failure"
    failure_dir.mkdir()
    dist_workers_size_2.run(partial(_test_cross_rank_recipe_merge, failure_dir, RuntimeError))
