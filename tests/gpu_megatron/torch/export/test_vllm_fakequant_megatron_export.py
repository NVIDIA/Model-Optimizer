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
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from importlib.util import find_spec

import pytest
import torch
import yaml
from _test_utils.torch.megatron.models import get_mcore_gpt_model, get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import initialize_for_megatron, run_mcore_inference
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
from modelopt.torch.quantization.nn import GroupedQuantizer, TensorQuantizer


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
    assert all(key.endswith("._amax") for key in state)
    assert not any(key.endswith("._vllm_fakequant_recipe_marker") for key in recipe)
    assert not (export_dir / "hf_quant_config.json").exists()
    weight_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(weight_map.values()):
        with safe_open(export_dir / shard, framework="pt") as f:
            shard_keys = f.keys()
            assert not any(
                "quantizer" in key or "._vllm_fakequant_recipe_marker" in key for key in shard_keys
            )
    return state, recipe, weight_map


def _test_mcore_vllm_export(tmp_path, rank, size):
    model = get_mcore_gpt_model(
        initialize_megatron=True,
        num_query_groups=1,
        max_sequence_length=32,
        normalization="RMSNorm",
        transformer_impl="modelopt",
    ).cuda()
    model.eval()

    def forward_loop(model):
        with torch.no_grad():
            run_mcore_inference(model, torch.randint(0, model.vocab_size, (1, 32), device="cuda"))

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    # Calibration precision must survive exporting BF16 weights.
    for name, quantizer in model.named_modules():
        if (
            isinstance(quantizer, TensorQuantizer)
            and name.endswith("input_quantizer")
            and quantizer.amax is not None
        ):
            quantizer.float()
            quantizer.amax = torch.full_like(quantizer.amax, 1.001)

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
    qkv = layer.self_attention.linear_qkv.input_quantizer
    qkv.set_from_attribute_config(
        {"use_constant_amax": True, "unsigned": True, "narrow_range": True, "type": "dynamic"}
    )
    qkv.reset_amax()
    backend_cfg = {
        "num_bits": "test_format",
        "backend": "test_vllm_recipe",
        "backend_extra_args": {"offset": 0.25},
    }
    layer.self_attention.linear_proj.input_quantizer.set_from_attribute_config(
        {**backend_cfg, "use_constant_amax": True}
    )
    for name in ("k_bmm_quantizer", "v_bmm_quantizer"):
        quantizer = getattr(layer.self_attention.core_attention, name)
        quantizer.set_from_attribute_config({**backend_cfg, "enable": True})
        quantizer.amax = torch.tensor(1.001, device="cuda")
    layer.self_attention.core_attention.k_bmm_quantizer.set_from_attribute_config(
        {"use_constant_amax": True}
    )
    layer.self_attention.core_attention.k_bmm_quantizer.reset_amax()
    inactive_cfg = {
        "num_bits": 8,
        "unsigned": True,
        "narrow_range": True,
        "fake_quant": False,
        "type": "dynamic",
        "bias": {-1: None},
        "backend": "unused",
    }
    gate = layer.mlp.linear_fc1.input_quantizer
    gate.set_from_attribute_config(inactive_cfg)
    gate.disable_quant()
    down = layer.mlp.linear_fc2.input_quantizer
    down.set_from_attribute_config({**inactive_cfg, "enable": False})
    down.pre_quant_scale = torch.full((model.config.ffn_hidden_size,), 2.0, device="cuda")
    down._enable_pre_quant_scale = False

    weight_quantizer = layer.self_attention.linear_proj.weight_quantizer
    weight_quantizer.set_from_attribute_config(
        {
            "num_bits": 8,
            "narrow_range": True,
            "bias": {-1: None},
            "rotate": find_spec("fast_hadamard_transform") is not None,
        }
    )
    weight_quantizer.bias_value = torch.tensor(0.025, device="cuda")
    layer.mlp.linear_fc1.weight_quantizer.disable()
    layer.mlp.linear_fc1.weight_quantizer.pre_quant_scale = torch.full(
        (model.config.hidden_size,), 2.0, device="cuda"
    )
    with torch.no_grad():
        expected_weights = [
            module.weight_quantizer(module.weight.to(torch.bfloat16)).to(torch.bfloat16).cpu()
            for _, module in linears
        ]

    source = create_tiny_llama_dir(
        tmp_path,
        hidden_size=model.config.hidden_size,
        intermediate_size=model.config.ffn_hidden_size,
        num_hidden_layers=model.config.num_layers,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_query_groups,
        vocab_size=model.vocab_size,
    )
    stale_quantizer = "stale_source.input_quantizer"
    torch.save({stale_quantizer + "._amax": torch.tensor(42.0)}, source / "quantizer_state.pth")
    (source / "quant_recipe.yaml").write_text(
        yaml.safe_dump({stale_quantizer: {"_disabled": True}})
    )

    export_dir = tmp_path / "vllm_export"
    exporter = VllmFqGPTModelExporter(model, source, dtype=torch.bfloat16)
    _ = exporter.state_dict
    exporter.save_pretrained(str(export_dir), source)

    expected_names = {
        f"model.layers.{i}.{projection}.input_quantizer"
        for i in range(model.config.num_layers)
        for projections, _ in linears
        for projection in projections
    }
    constant_names = {
        f"model.layers.0.{projection}.input_quantizer"
        for projections, _ in linears[:2]
        for projection in projections
    }
    expected_names.update(
        f"model.layers.0.self_attn.{name}" for name in ("k_bmm_quantizer", "v_bmm_quantizer")
    )
    constant_names.add("model.layers.0.self_attn.k_bmm_quantizer")
    state, recipe, weight_map = _assert_exported_quantizers(
        export_dir,
        expected_names,
        amax={name: 448.0 if name in constant_names else 1.001 for name in expected_names},
        disabled_names={name for name in expected_names if name.startswith("model.layers.0.mlp.")},
    )
    for (projections, _), expected_weight in zip(linears, expected_weights):
        folded_weights = []
        for projection in projections:
            prefix = f"model.layers.0.{projection}"
            weight_key = prefix + ".weight"
            with safe_open(export_dir / weight_map[weight_key], framework="pt") as f:
                folded_weights.append(f.get_tensor(weight_key))
            assert recipe[prefix + ".weight_quantizer"]["_disabled"]
        torch.testing.assert_close(torch.cat(folded_weights), expected_weight, rtol=0, atol=0)
    assert not hasattr(qkv, "_amax")
    torch.testing.assert_close(
        layer.self_attention.linear_proj.input_quantizer.amax,
        torch.tensor(1.001, device="cuda"),
        rtol=0,
        atol=0,
    )
    for name in ("o_proj.input_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        custom_recipe = recipe[f"model.layers.0.self_attn.{name}"]
        assert custom_recipe["_num_bits"] == "test_format"
        assert custom_recipe["_backend"] == "test_vllm_recipe"
        assert custom_recipe["_backend_extra_args"] == {"offset": 0.25}
    assert stale_quantizer + "._amax" not in state
    assert stale_quantizer not in recipe
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= weight_map.keys()


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

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

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

    _assert_unsupported_settings(model, source, tmp_path / "unsupported_export", rank, size)

    unsupported_dir = tmp_path / "unsupported_mtp_export"
    if is_pipeline_last_stage():
        mtp_quantizer = model.mtp.layers[0].eh_proj.input_quantizer
        mtp_quantizer.enable()
    with pytest.raises(ValueError, match="MTP quantization is not supported"):
        export_mcore_gpt_to_hf_vllm_fq(model, str(source), export_dir=str(unsupported_dir))
    assert not list(unsupported_dir.glob("*.safetensors"))
    assert not (unsupported_dir / "model.safetensors.index.json").exists()
    if is_pipeline_last_stage():
        mtp_quantizer.disable()

    export_dir = tmp_path / "mtp_export"
    export_mcore_gpt_to_hf_vllm_fq(
        model,
        pretrained_model_name_or_path=str(source),
        dtype=torch.bfloat16,
        export_dir=str(export_dir),
    )
    state, recipe, weight_map = _assert_exported_quantizers(export_dir, set())
    assert not any(key.startswith("mtp.") for key in state.keys() | recipe.keys())
    weight_key = "mtp.layers.0.eh_proj.weight"
    assert weight_key in weight_map
    if is_pipeline_last_stage():
        with safe_open(export_dir / weight_map[weight_key], framework="pt") as f:
            torch.testing.assert_close(
                f.get_tensor(weight_key),
                model.mtp.layers[0].eh_proj.weight.to(torch.bfloat16).cpu(),
                rtol=0,
                atol=0,
            )


def test_mcore_vllm_export_mtp(request, tmp_path):
    """Reject unsupported settings and quantized MTP while preserving BF16 MTP weights."""
    workers = request.getfixturevalue(f"dist_workers_size_{min(torch.cuda.device_count(), 2)}")
    workers.run(partial(_test_mcore_vllm_export_mtp, tmp_path))


def _assert_unsupported_settings(model, source, export_dir, rank, size):
    attribute_cfgs = [
        (
            "input_quantizer",
            {
                "num_bits": 8,
                "unsigned": True,
                "narrow_range": True,
                "rotate": True,
                "pre_quant_scale": True,
                "fake_quant": False,
                "type": "dynamic",
                "bias": {-1: None},
            },
        ),
        ("input_quantizer", {"enable": False, "rotate": True, "pre_quant_scale": True}),
        ("weight_quantizer", {"fake_quant": False}),
    ]
    if rank == size - 1:
        linear = next(
            module
            for module in model.modules()
            if isinstance(getattr(module, "input_quantizer", None), TensorQuantizer)
            and module.input_quantizer.is_enabled
        )

    for quantizer_name, attribute_cfg in attribute_cfgs:
        if rank == size - 1:
            original_quantizer = getattr(linear, quantizer_name)
            quantizer = deepcopy(original_quantizer)
            setattr(linear, quantizer_name, quantizer)
            quantizer.set_from_attribute_config(
                {key: value for key, value in attribute_cfg.items() if key != "pre_quant_scale"}
            )
            if "pre_quant_scale" in attribute_cfg:
                quantizer.pre_quant_scale = torch.full(
                    (linear.weight.shape[1],), 2.0, device="cuda"
                )
            if "type" in attribute_cfg:
                quantizer.reset_amax()
        with pytest.raises(ValueError, match=f"Unsupported.*{quantizer_name}") as exc:
            export_mcore_gpt_to_hf_vllm_fq(model, source, export_dir=str(export_dir))
        for setting in attribute_cfg.keys() - {"enable", "num_bits"}:
            assert ("dynamic_amax" if setting == "type" else setting) in str(exc.value)
        assert not list(export_dir.glob("*.safetensors"))
        assert not (export_dir / "model.safetensors.index.json").exists()
        assert not (export_dir / "quantizer_state.pth").exists()
        assert not (export_dir / "quant_recipe.yaml").exists()
        if rank == size - 1:
            setattr(linear, quantizer_name, original_quantizer)


def _test_cross_rank_quantizer_merge(tmp_path, rank, size):
    name = "model.layers.0.self_attn.q_proj.input_quantizer"
    for error in (None, ValueError):
        recipe = {"_num_bits": 4 if rank == 1 and error else 8}
        tensor = torch.tensor([1.0 + rank if error else 1.0])
        with (
            pytest.raises(error, match="Conflicting quantizer recipes") if error else nullcontext()
        ):
            gather_mcore_vllm_fq_quantizer_recipe({name: recipe}, tmp_path)
        with (
            pytest.raises(error, match="Conflicting quantizer tensors") if error else nullcontext()
        ):
            gather_mcore_vllm_fq_quantized_state_dict(
                None, {1: {name + "._amax": tensor}}, tmp_path
            )
        if error is None:
            assert yaml.safe_load((tmp_path / "quant_recipe.yaml").read_text()) == {name: recipe}
            state = torch.load(tmp_path / "quantizer_state.pth", weights_only=True)
            torch.testing.assert_close(state[name + "._amax"], tensor, rtol=0, atol=0)

    failure_dir = tmp_path / "write_failure"
    if rank == 0:
        (failure_dir / "quant_recipe.yaml").mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"Failed to save quant_recipe\.yaml"):
        gather_mcore_vllm_fq_quantizer_recipe({name: {"_num_bits": 8}}, failure_dir)


def test_cross_rank_quantizer_merge(dist_workers_size_2, tmp_path):
    """Check matching states, conflicts, and write failure in one distributed session."""
    dist_workers_size_2.run(partial(_test_cross_rank_quantizer_merge, tmp_path))


def _grouped_model(tmp_path, quant_cfg, rank, size, expert_parallel=False):
    if expert_parallel:
        initialize_for_megatron(expert_model_parallel_size=size)
    model = (
        get_mcore_hybrid_model(
            initialize_megatron=not expert_parallel,
            expert_model_parallel_size=size if expert_parallel else 1,
            num_layers=1,
            hybrid_layer_pattern="E",
            hidden_size=64,
            num_attention_heads=8,
            num_query_groups=8,
            ffn_hidden_size=128,
            max_sequence_length=16,
            vocab_size=64,
            normalization="RMSNorm",
            transformer_impl="transformer_engine",
            moe_grouped_gemm=True,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_token_dispatcher_type="alltoall",
        )
        .cuda()
        .eval()
    )

    def forward_loop(model):
        with torch.no_grad():
            run_mcore_inference(model, torch.arange(16, device="cuda").unsqueeze(0))

    mtq.quantize(model, quant_cfg, forward_loop)
    if rank == 0:
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["NemotronHForCausalLM"],
                    "model_type": "nemotron_h",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "moe_intermediate_size": 64,
                    "moe_shared_expert_intermediate_size": 32,
                    "hybrid_override_pattern": "E",
                    "num_hidden_layers": 1,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "head_dim": 8,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "vocab_size": 64,
                    "torch_dtype": "bfloat16",
                }
            )
        )
    if expert_parallel:
        torch.distributed.barrier()
    experts = model.decoder.layers[0].mlp.experts
    return model, (experts.linear_fc1, experts.linear_fc2)


def _expected_grouped_weights(grouped_modules, rank, disable_last=False):
    expected = {}
    for module, projection in zip(grouped_modules, ("up_proj", "down_proj")):
        assert isinstance(module.weight_quantizer, GroupedQuantizer)
        if disable_last:
            module.weight_quantizer[-1].disable()
        for local_id, quantizer in enumerate(module.weight_quantizer):
            weight = getattr(module, f"weight{local_id}")
            with torch.no_grad():
                folded = quantizer(weight.to(torch.bfloat16)).cpu()
            if disable_last and quantizer.is_enabled:
                assert not torch.equal(folded, weight.to(torch.bfloat16).cpu())
            global_id = rank * module.num_gemms + local_id
            expected[f"backbone.layers.0.mixer.experts.{global_id}.{projection}.weight"] = folded
    return expected


def _assert_grouped_weights(export_dir, expected):
    weight_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for key, weight in expected.items():
        with safe_open(export_dir / weight_map[key], framework="pt") as f:
            torch.testing.assert_close(f.get_tensor(key), weight, rtol=0, atol=0)


def _test_mcore_vllm_grouped_export(tmp_path, quant_cfg, device, rank, size):
    model, grouped_modules = _grouped_model(tmp_path, quant_cfg, rank, size)
    expected_weights = _expected_grouped_weights(grouped_modules, rank, disable_last=True)
    model.to(device)
    original_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    original_hooks = {module: dict(module._state_dict_hooks) for module in grouped_modules}

    def assert_model_unchanged():
        for module in grouped_modules:
            assert dict(module._state_dict_hooks) == original_hooks[module]
            assert isinstance(module.weight_quantizer, GroupedQuantizer)
            assert not hasattr(module, "weight")
            assert not module.weight_quantizer[-1].is_enabled
        current_state = {
            key: value
            for key, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        assert current_state.keys() == original_state.keys()
        for key, value in original_state.items():
            torch.testing.assert_close(current_state[key], value, rtol=0, atol=0)

    # Fail after the first grouped linear has been processed, then retry.
    def fail_quantization(module, args):
        raise RuntimeError("injected grouped QDQ failure")

    failure_hook = (
        grouped_modules[1].weight_quantizer[0].register_forward_pre_hook(fail_quantization)
    )
    try:
        exporter = VllmFqGPTModelExporter(model, tmp_path, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="injected grouped QDQ failure"):
            exporter.save_pretrained(str(tmp_path / "failed_export"), tmp_path)
    finally:
        failure_hook.remove()
    assert_model_unchanged()

    calls = Counter()

    def count_qdq(module, args, output):
        calls[module] += 1

    handles = [
        quantizer.register_forward_hook(count_qdq)
        for module in grouped_modules
        for quantizer in module.weight_quantizer
    ]
    export_dir = tmp_path / "grouped_export"
    try:
        exporter = VllmFqGPTModelExporter(model, tmp_path, dtype=torch.bfloat16)
        assert exporter.layer_state_dicts  # Cache shards before writing the checkpoint.
        exporter.save_pretrained(str(export_dir), tmp_path)
    finally:
        for handle in handles:
            handle.remove()

    assert_model_unchanged()
    for module in grouped_modules:
        for quantizer in module.weight_quantizer:
            assert calls[quantizer] == 1

    _assert_grouped_weights(export_dir, expected_weights)
    quantizer_state = torch.load(export_dir / "quantizer_state.pth", weights_only=True)
    recipe = yaml.safe_load((export_dir / "quant_recipe.yaml").read_text())
    assert not any("weight_quantizer" in key for key in quantizer_state)
    assert {key.rsplit(".", 1)[0] for key in quantizer_state} <= recipe.keys()
    assert not any("{}" in key for key in recipe)


@pytest.mark.parametrize(
    ("quant_cfg", "device"),
    [(mtq.FP8_DEFAULT_CFG, "cpu"), (mtq.NVFP4_DEFAULT_CFG, "cuda")],
    ids=["fp8-cpu", "nvfp4-cuda"],
)
def test_mcore_vllm_grouped_export(dist_workers_size_1, tmp_path, quant_cfg, device):
    """Fold FP8 and NVFP4 grouped weights once without mutating the model."""
    dist_workers_size_1.run(partial(_test_mcore_vllm_grouped_export, tmp_path, quant_cfg, device))


def _test_mcore_vllm_grouped_ep_export(tmp_path, rank, size):
    """Every EP rank contributes its local folded experts to one checkpoint."""
    model, grouped_modules = _grouped_model(
        tmp_path, mtq.FP8_DEFAULT_CFG, rank, size, expert_parallel=True
    )
    expected_local = _expected_grouped_weights(grouped_modules, rank)
    all_expected = [None] * size
    torch.distributed.all_gather_object(all_expected, expected_local)

    export_dir = tmp_path / "grouped_ep_export"
    export_mcore_gpt_to_hf_vllm_fq(
        model, tmp_path, dtype=torch.bfloat16, export_dir=str(export_dir)
    )
    torch.distributed.barrier()
    if rank == 0:
        for per_rank in all_expected:
            _assert_grouped_weights(export_dir, per_rank)


def test_mcore_vllm_grouped_ep_export(dist_workers_size_2, tmp_path):
    dist_workers_size_2.run(partial(_test_mcore_vllm_grouped_ep_export, tmp_path))
